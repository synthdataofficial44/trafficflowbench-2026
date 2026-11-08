"""v15 Task 2: the ongoing counterpart of nl75 on the April rows, a second onset step behind the same rule, and March rows
that use no March traffic data stamped after their origins - on top of the v12 checkpoints and the v13.1 probabilities,
without editing any earlier module (every v10-v13.1 checkpoint reloads).

What decided it.
  * Ongoing (v14 step 11, task2o): the v12 ongoing models reloaded exactly (80 official windows, max diff 0.0) and the
    2 453 March pseudo windows (a demand draw the models never saw) decoded with the location-free variant weighted 0 ...
    1. Weight 0.75 against today's 0.5: +0.0127 official-style windows (4.1 SEs), +0.0041 all windows (5.8 SEs), gain
    +0.0084 (4.4 SEs), 3 of 4 corridor families, the first week +0.0029, weekday +0.0026 / weekend +0.0039; weight 1.0
    gains less (+0.0058). So `og_w` = 0.75, fixed before this run; the gate below re-measures it on this run's tables.
  * Onset: v13.1 took nl75 (+0.022 on March, 2.3 SEs). `on_w` = 1.0 is the next step on the same axis, through the same
    cross-draw rule (gain > `z15` day-block bootstrap SEs, both March reads positive, no loss beyond one SE on the
    official-style, all or first-week windows, 3 of 4 families, enough windows).
  * The organizers' rulings of 15 and 18 October (discussions 748789 and 749601): for a window with origin T, inputs,
    model fitting and model or threshold *selection* must use only data stamped at or before T, and at award review the
    Task 2 code is rerun with every row after each origin removed. The March pseudo windows lie after the scored March
    origins (1-5 March), so a choice made on them may shape the April rows only - as v4-v12 did. v13.1 broke this for the
    March onset rows (nl75, decided on March pseudo windows, written into the March rows for a public read). Here the
    March onset rows go back to v12's (`march_compliant`), and the ongoing change touches April rows only; the public read
    of the ongoing change comes from a probe file (main + the March ongoing rows reweighted), which is never a final pick.
Where each change lands (both files keep their bets: main = April resembles March, hedge = April is a new draw):
  hedge  April ongoing rows: the standard (train-only) models reweighted - exactly what the March gate measured;
         April onset rows: + the onset step when adopted;
  main   April ongoing rows of the April refit (train + March): reweighted only if the change does not lose within March
         on the gate models (train + March before the cut, scored after it - the v13 check); the ten late-block windows
         (models that also learn April's own first days, a same-draw signal) keep v12's weights; April onset rows keep
         v13.1's (nl75 on the April refit);
  both   March rows: v12's (onset) and v12's = v13.1's (ongoing).
Only data timestamped <= T feeds a window at origin T; no April label is evaluated or used to choose anything.
"""
from __future__ import annotations

import copy
import hashlib
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import common as C
from . import task2 as T
from . import task2m as TM
from . import task2x as X
from . import task2y as Y
from . import task2z as Z

VERSION = "v15"
MAIN_FILE, HEDGE_FILE = "task2_v15.csv", "task2_v15_hedge.csv"
PROBE_FILE = "task2_v15_probe_og.csv"    # the main file with the March ongoing rows reweighted (public read only)
MAIN0_FILE, HEDGE0_FILE = "task2_v15_0.csv", "task2_v15_0_hedge.csv"   # the fallback pair: v13.1's April rows, March = v12
REPORT_FILE, PROBS_FILE = "task2_v15_report.json", "task2_v15_probs.pkl"
V131_PROBS, V131_REPORT = "task2_v131_probs.pkl", "task2_v131_report.json"
V131_MAIN, V131_HEDGE = "task2_v131.csv", "task2_v131_hedge.csv"
V12_MAIN = "task2_v12.csv"
ON, OG = "queue_onset", "queue_ongoing"
KEY, QDT = TM.KEY, TM.QDT
DEFAULTS = dict(og_w=0.75, on_w=1.0, z15=2.0, min_windows15=40, fam_share15=0.75, reload_tol=0.02, main_check=True,
                onset_step=True, probe=True, march_compliant=True, force15=None)
FORCES = (None, "og", "on", "both")


def install():
    Z.install()


def code_hash() -> str:
    h = hashlib.md5((Z.code_hash() + "|").encode())
    h.update(Path(__file__).read_bytes())
    return h.hexdigest()[:16]


def settings(st: dict | None) -> dict:
    s = dict(DEFAULTS)
    s.update({k: v for k, v in (st or {}).items() if v is not None or k == "force15"})
    unknown = set(s) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"v15 settings: unknown keys {sorted(unknown)}")
    if s["force15"] not in FORCES:
        raise ValueError(f"v15 force15 must be one of {FORCES}")
    return s


# ----------------------------------------------------------------------------- fields
def _part(probs_set: dict, cond: str, split: str | None = None) -> dict:
    """{panel: {split: {wid: field}}} of the windows of `cond` in a v12-style probability set."""
    out = {}
    for p, d in probs_set.items():
        for s, ws in d.items():
            if split is not None and s != split:
                continue
            for wid, (c, pr) in ws.items():
                if c == cond:
                    out.setdefault(p, {}).setdefault(s, {})[wid] = np.asarray(pr, np.float64)
    return out


def _maxdiff(a: dict, b: dict) -> float:
    """Largest absolute difference over the windows of `a` (inf when `b` misses one)."""
    m = 0.0
    for p, d in a.items():
        for s, ws in d.items():
            for wid, v in ws.items():
                o = b.get(p, {}).get(s, {}).get(wid)
                if o is None:
                    return float("inf")
                m = max(m, float(np.max(np.abs(np.asarray(v, np.float64) - np.asarray(o, np.float64)))))
    return m


def _blend(fa: dict, fn: dict, w: float) -> dict:
    return {p: {s: {wid: (1.0 - w) * fa[p][s][wid] + w * fn[p][s][wid] for wid in ws} for s, ws in d.items()}
            for p, d in fa.items()}


def _official(models, variants, seeds, vname, offis, pans, Ps, cond, splits) -> dict:
    """{panel: {split: {wid: field}}} of one variant's models on the official windows of `cond`."""
    raw = {}
    T.predict_official(T._models_for(models, variants, seeds, [vname]), offis, pans, Ps, cond, raw, splits)
    return {p: {s: {wid: np.asarray(f, np.float64) for wid, (_, f) in ws.items()} for s, ws in d.items()}
            for p, d in raw.items()}


def _pseudo(models, pseudo, pans, Ps, cond, split="validation") -> dict:
    """{panel: [n windows, ...field]} of `models` on the pseudo windows of a month."""
    out = {}
    for p in pans:
        ent = pseudo.get(p, {}).get(split, {}).get(cond)
        if ent is None:
            continue
        Wc, _, _, Xp = ent
        out[p] = T._field(cond, Xp, T._predict(models, Xp), len(Wc), Ps[p]["L"])
    return out


def _rows(F: dict, pseudo, pans, Ps, cond, split="validation", day_lo=None) -> pd.DataFrame:
    """IoU on the known cells of every pseudo window (days >= day_lo), decoded with the run's decoder."""
    rows = []
    for p in pans:
        ent = pseudo.get(p, {}).get(split, {}).get(cond)
        if ent is None or p not in F:
            continue
        Wc, y, kn, _ = ent
        L = Ps[p]["L"]
        d = Wc.day.to_numpy()
        for w in range(len(Wc)):
            if day_lo is not None and d[w] < day_lo:
                continue
            pred = T.decode_window(cond, F[p][w], L)
            u = int(((pred | y[w]) & kn[w]).sum()); it = int(((pred & y[w]) & kn[w]).sum())
            rows.append(dict(panel=p, condition=cond, split=split, w=int(w), src=Wc.src.iat[w], day=int(d[w]),
                             draws=int(Wc.draws.iat[w]), iou=1.0 if u == 0 else it / u,
                             n_pred=int((pred & kn[w]).sum()), n_true=int((y[w] & kn[w]).sum()), inter=it))
    return pd.DataFrame(rows, columns=["panel", "condition", "split", "w", "src", "day", "draws", "iou", "n_pred",
                                       "n_true", "inter"])


def gate(Ra: pd.DataFrame, Rb: pd.DataFrame, early_day: int, st: dict) -> dict:
    """The cross-draw rule (v13.1) for b against a on the same March pseudo windows."""
    out, M = T.paired(Ra, Rb, early_day)
    gain = 0.5 * (np.nan_to_num(out["d_sim"]) + out["d_all"])
    se = T._boot_gain(M)
    early_bad = out.get("n_early", 0) >= 10 and out.get("d_early", 0) < -np.nan_to_num(out.get("se_early", 0))
    ok = (np.nan_to_num(out["d_sim"]) >= -np.nan_to_num(out["se_sim"]) and out["d_all"] >= -np.nan_to_num(out["se_all"])
          and not early_bad)
    enough = bool(len(M) >= int(st["min_windows15"]) and M.day.nunique() >= 3)
    fam = C.families()
    pmd = M.groupby("panel").d.mean(); pn = M.groupby("panel").size()
    fd = pmd.groupby(pmd.index.map(fam)).mean(); fnn = pn.groupby(pn.index.map(fam)).sum()
    fd = fd[fnn.reindex(fd.index).fillna(0) >= 5]
    fam_ok = bool(len(fd) < 2 or float((fd >= -1e-12).mean()) >= float(st["fam_share15"]))
    strong = bool(np.isfinite(se) and se > 0 and gain > float(st["z15"]) * se and np.nan_to_num(out["d_sim"]) > 0
                  and out["d_all"] > 0)
    wd = T._weekday(M.day.to_numpy())
    out.update(gain=float(gain), se_gain=float(se), eligible=bool(ok), early_bad=bool(early_bad), enough=enough,
               fam_ok=fam_ok, fam_d={str(k): round(float(v), 4) for k, v in fd.items()}, strong=strong,
               take=bool(ok and enough and fam_ok and strong),
               d_weekday=float(M[wd].d.mean()) if wd.any() else None,
               d_weekend=float(M[~wd].d.mean()) if (~wd).any() else None,
               n_pred=float(Rb.n_pred.mean()) if len(Rb) else None, n_true=float(Rb.n_true.mean()) if len(Rb) else None)
    return {k: (float(v) if isinstance(v, (np.floating,)) else v) for k, v in out.items()}


def _reweight(b, fa, fn, dw: float, scale: float = 1.0):
    """Field b (whose classifier part averages 'all' and 'noloc' with weight `scale`) with the location-free weight moved
    by dw: written as a change of b, so a stored field that differs from a recomputation in the last digits keeps them."""
    return np.clip(np.asarray(b, np.float64) + float(scale) * float(dw) * (np.asarray(fn, np.float64) - np.asarray(fa, np.float64)),
                   0.0, 1.0)


# ----------------------------------------------------------------------------- inputs
def _load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _save_probs(path: Path, obj):
    tmp = path.with_suffix(".part")
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=4)
    tmp.replace(path)


def load_v131(work=None) -> tuple:
    """(probabilities, report) of the v13.1 run, checked against each other."""
    d = (work or C.WORK) / "pred"
    pf, rf = d / V131_PROBS, d / V131_REPORT
    if not (pf.exists() and rf.exists()):
        raise FileNotFoundError(f"v15 needs pred/{V131_PROBS} and pred/{V131_REPORT} from the v13.1 run")
    rep = json.load(open(rf))
    if rep.get("probs_md5") and rep["probs_md5"] != T._md5(pf):
        raise RuntimeError(f"pred/{V131_PROBS} is not the file the v13.1 report describes (an interrupted copy?)")
    return _load_pickle(pf), rep


def up_to_date(cfg, st, work=None) -> tuple:
    """(True, report) when the v15 files of this code, these settings, this release and these inputs exist."""
    d = (work or C.WORK) / "pred"
    rp = d / REPORT_FILE
    if not rp.exists():
        return False, None
    try:
        from .data import fingerprint
        install()
        rep = json.load(open(rp))
        fp = fingerprint()
    except Exception:  # noqa: BLE001
        return False, None
    fm = rep.get("files_md5") or {}
    ok = (rep.get("version") == VERSION and rep.get("code") == code_hash() and Z._norm(rep.get("cfg")) == Z._norm(cfg)
          and Z._norm(rep.get("st")) == Z._norm(settings(st)) and rep.get("fingerprint") == fp
          and rep.get("v12_probs_md5") == T._md5(d / Y.PROBS_FILE) and rep.get("v131_probs_md5") == T._md5(d / V131_PROBS)
          and set(fm) >= {MAIN_FILE, HEDGE_FILE, MAIN0_FILE, HEDGE0_FILE} | ({PROBE_FILE} if rep.get("probe") else set())
          and all((d / fn).exists() and T._md5(d / fn) == h for fn, h in fm.items()))
    return ok, rep


# ----------------------------------------------------------------------------- the run
def run(cfg: dict, st: dict | None = None, device: dict | None = None) -> dict:
    """cfg: the v12 run's Task 2 settings (its tables and checkpoints are rebuilt / reloaded exactly); st: DEFAULTS.
    Writes pred/task2_v15.csv, pred/task2_v15_hedge.csv, the ongoing probe, pred/task2_v15_probs.pkl and
    pred/task2_v15_report.json (last)."""
    try:
        return _run(cfg, settings(st), device)
    finally:
        Y.DEC["mode"] = "topm"


def _nfit():
    return TM.DEVICE["fits_cpu"] + TM.DEVICE["fits_gpu"]


def _run(cfg: dict, st: dict, device: dict | None) -> dict:
    from .data import fingerprint
    install()
    t0 = time.time()
    pk, rep12 = Z.load_v12()
    probs12 = pk["probs"]
    if Z._norm(rep12.get("cfg")) != Z._norm(cfg):
        raise RuntimeError("v15 must run with the v12 run's Task 2 settings (CFG['T2']), so its tables and models reload")
    pk131, rep131 = load_v131()
    probs131 = pk131["probs"]
    Y.DEC.update(mode=pk.get("mode", "topm"), rho=pk.get("rho", Y.DEC["rho"]), n=pk.get("n", Y.DEC["n"]),
                 M=pk.get("M", Y.DEC["M"]), seed=pk.get("seed", Y.DEC["seed"]))
    tcfg = Y.task_cfg(cfg)
    TM.use_device(device)
    T.set_checkpoints(tcfg)
    bcfg = dict(tcfg, late_adapt=False, late_early={}, pseudo_april=False, online_adapt=False)
    pans = C.task2_panels()
    rng = np.random.default_rng(tcfg.get("seed", 0))
    imputer = T.fit_imputer(pans) if tcfg.get("imputer", True) else None
    skip = () if st["onset_step"] else (ON,)
    sets, offis, Ps, pseudo = {}, {}, {}, {}
    for i, p in enumerate(pans):
        P, out, offi, ps = T.build_panel_sets(p, imputer, rng, bcfg, skip=skip, skip_april=(ON, OG))
        for split in ps:
            for cond in ps[split]:
                ps[split][cond][3]["panel_code"] = np.float32(i)
        sets[p] = out; offis[p] = offi; pseudo[p] = ps
        Ps[p] = dict(L=P.L, links=P.net.links)
        P.dprof = None
        del P
        C.free()
    report = dict(version=VERSION, code=code_hash(), fingerprint=fingerprint(), cfg=cfg, st=st,
                  v12_probs_md5=T._md5(C.WORK / "pred" / Y.PROBS_FILE), v131_probs_md5=T._md5(C.WORK / "pred" / V131_PROBS),
                  v131_adopted=(rep131.get("gate") or {}).get("adopted"), decoder=Y.DEC["mode"], checks={}, minutes={})
    report["minutes"]["tables"] = round((time.time() - t0) / 60, 1)
    C.log(f"T2 {VERSION} tables ready ({report['minutes']['tables']} min)")
    vd = C.split_days("validation")
    d0 = int(vd.min())
    early = d0 + int(tcfg.get("early_days", 7))
    probs = copy.deepcopy(probs131)
    alpha = float(tcfg.get("adapt_alpha", 2.0))
    cut = int(vd.min() + int(round(len(vd) * tcfg.get("adapt_frac", 0.5))))

    # ======================================================================= ongoing: og_w on the April rows
    Dl = T._frames(sets, pans, OG)
    fset, vnames = rep12["chosen"][OG][0], list(rep12["chosen"][OG][1])
    if sorted(vnames) != ["all", "noloc"]:
        raise RuntimeError(f"T2 v15 expects the ongoing variants all + noloc (v12 chose {vnames})")
    variants = T._variants(vnames)
    seeds = list(tcfg.get("ongoing_seeds", [tcfg.get("seed", 0)]))
    rounds, prm = int(tcfg.get("ongoing_rounds", 700)), T.ONG_P
    n0 = _nfit()
    std = T.fit_set(Dl, OG, prm, rounds, variants, seeds, fset=fset)
    if _nfit() > n0:
        raise RuntimeError("T2 v15: v12's standard ongoing models were not all found in t2_models (some were refitted); "
                           "is CFG['DRIVE_WORK'] the v12 run's folder?")
    ofa = _official(std, variants, seeds, "all", offis, pans, Ps, OG, C.TEST_SPLITS)
    ofn = _official(std, variants, seeds, "noloc", offis, pans, Ps, OG, C.TEST_SPLITS)
    std12 = _part(probs12["std"], OG)
    md = max(_maxdiff(std12, _blend(ofa, ofn, 0.5)), _maxdiff(_blend(ofa, ofn, 0.5), std12))
    report["checks"]["standard ongoing fields vs v12 (max abs diff)"] = md
    if not md <= float(st["reload_tol"]):
        raise RuntimeError(f"T2 v15: the reloaded ongoing models do not reproduce v12's probabilities (max diff {md})")
    pfa = _pseudo(T._models_for(std, variants, seeds, ["all"]), pseudo, pans, Ps, OG)
    pfn = _pseudo(T._models_for(std, variants, seeds, ["noloc"]), pseudo, pans, Ps, OG)
    w_og = float(st["og_w"])
    R0 = _rows({p: 0.5 * pfa[p] + 0.5 * pfn[p] for p in pfa}, pseudo, pans, Ps, OG)
    R1 = _rows({p: (1.0 - w_og) * pfa[p] + w_og * pfn[p] for p in pfa}, pseudo, pans, Ps, OG)
    g_og = gate(R0, R1, early, st)
    g_og.update(march_base=T.pseudo_report(R0), march_cand=T.pseudo_report(R1))
    take_og = bool(g_og["take"]) or st["force15"] in ("og", "both")
    g_og["adopted"] = take_og
    g_og["forced"] = bool(st["force15"] in ("og", "both"))
    report["ongoing"] = dict(gate=g_og, weight=w_og)
    del R0, R1, pfa, pfn
    C.free()
    report["minutes"]["ongoing_gate"] = round((time.time() - t0) / 60, 1)
    C.log(f"T2 {VERSION} ongoing w{w_og}: gain {g_og['gain']:+.4f} se {g_og['se_gain']:.4f} | d_sim {g_og['d_sim']:+.4f} "
          f"d_all {g_og['d_all']:+.4f} | take {take_og}")
    probe_set = None
    if take_og:
        dw = w_og - 0.5
        n_h = 0
        for p, dd in std12.items():
            for wid, b in dd.get("private", {}).items():
                probs["std"][p]["private"][wid] = (OG, _reweight(b, ofa[p]["private"][wid], ofn[p]["private"][wid], dw))
                n_h += 1
        report["ongoing"]["hedge_april_windows"] = n_h
        if st["probe"]:
            probe_set = {p: {"validation": {wid: (OG, _reweight(b, ofa[p]["validation"][wid], ofn[p]["validation"][wid], dw))
                                            for wid, b in dd.get("validation", {}).items()}} for p, dd in std12.items()}
        report["ongoing"]["main"] = _main_og(Dl, pseudo, offis, pans, Ps, prm, rounds, variants, seeds, fset, alpha,
                                             cut, probs12, probs, dw, st)
        report["ongoing"]["late"] = dict(kept=bool(_part(probs12["late"], OG)),
                                         note="the late-block windows (models that also learn April's own first days) "
                                              "keep v12's weights")
    del std, ofa, ofn, Dl
    for p in pans:
        if OG in sets[p]:
            sets[p][OG] = (sets[p][OG][0], sets[p][OG][1], None)
    C.free()
    report["minutes"]["ongoing"] = round((time.time() - t0) / 60, 1)

    # ======================================================================= onset: on_w on the hedge's April rows
    w0 = float((rep131.get("cfg13") or {}).get("nl_w", 0.75)) if report["v131_adopted"] == "nl75" else 0.5
    report["onset"] = dict(current_weight=w0, weight=float(st["on_w"]), adopted=False)
    if st["onset_step"]:
        Dlo = T._frames(sets, pans, ON)
        fso, vno = rep12["chosen"][ON][0], list(rep12["chosen"][ON][1])
        if sorted(vno) != ["all", "noloc"]:
            raise RuntimeError(f"T2 v15 expects the onset variants all + noloc (v12 chose {vno})")
        vo = T._variants(vno)
        so = list(tcfg.get("onset_seeds", [tcfg.get("seed", 0)]))
        ro, pro = int(tcfg.get("onset_rounds", 500)), T.ONSET_P
        n0 = _nfit()
        stdo = T.fit_set(Dlo, ON, pro, ro, vo, so, fset=fso)
        if _nfit() > n0:
            raise RuntimeError("T2 v15: v12's standard onset models were not all found in t2_models (some were refitted)")
        ona = _official(stdo, vo, so, "all", offis, pans, Ps, ON, ("private",))
        onn = _official(stdo, vo, so, "noloc", offis, pans, Ps, ON, ("private",))
        on12 = _part(probs12["std"], ON, "private")
        mdo = max(_maxdiff(on12, _blend(ona, onn, 0.5)), _maxdiff(_blend(ona, onn, 0.5), on12))
        report["checks"]["standard onset fields (April) vs v12 (max abs diff)"] = mdo
        if not mdo <= float(st["reload_tol"]):
            raise RuntimeError(f"T2 v15: the reloaded onset models do not reproduce v12's probabilities (max diff {mdo})")
        qa = _pseudo(T._models_for(stdo, vo, so, ["all"]), pseudo, pans, Ps, ON)
        qn = _pseudo(T._models_for(stdo, vo, so, ["noloc"]), pseudo, pans, Ps, ON)
        w1 = float(st["on_w"])
        Ra = _rows({p: (1.0 - w0) * qa[p] + w0 * qn[p] for p in qa}, pseudo, pans, Ps, ON)
        Rb = _rows({p: (1.0 - w1) * qa[p] + w1 * qn[p] for p in qa}, pseudo, pans, Ps, ON)
        g_on = gate(Ra, Rb, early, st)
        g_on.update(march_base=T.pseudo_report(Ra), march_cand=T.pseudo_report(Rb))
        take_on = bool(g_on["take"]) or st["force15"] in ("on", "both")
        g_on["forced"] = bool(st["force15"] in ("on", "both"))
        report["onset"].update(gate=g_on, adopted=take_on)
        C.log(f"T2 {VERSION} onset w{w1} vs w{w0}: gain {g_on['gain']:+.4f} se {g_on['se_gain']:.4f} | take {take_on}")
        if take_on:
            n_h = 0
            on131 = _part(probs131["std"], ON, "private")
            for p, dd in on131.items():
                for wid, b in dd.get("private", {}).items():
                    probs["std"][p]["private"][wid] = (ON, _reweight(b, ona[p]["private"][wid], onn[p]["private"][wid], w1 - w0))
                    n_h += 1
            report["onset"]["hedge_april_windows"] = n_h
            report["onset"]["main"] = ("the main file's April onset rows keep v13.1's where they come from the April refit "
                                       "(nl75 on the refit); any it takes from the standard models move with the hedge")
        del stdo, Dlo, qa, qn, Ra, Rb
        C.free()
    report["minutes"]["onset"] = round((time.time() - t0) / 60, 1)

    # ======================================================================= March rows: no March data after their origins
    probs0 = copy.deepcopy(probs131)    # the fallback pair: v13.1's April rows with the same March rows
    if st["march_compliant"]:
        on12m = _part(probs12["std"], ON, "validation")
        n_r = 0
        for p, dd in on12m.items():
            for wid, f in dd.get("validation", {}).items():
                old = probs["std"][p]["validation"][wid][1]
                if not np.array_equal(np.asarray(old, np.float64), f):
                    n_r += 1
                probs["std"][p]["validation"][wid] = (ON, f)
                probs0["std"][p]["validation"][wid] = (ON, f)
        report["march"] = dict(onset_windows_reverted_to_v12=n_r,
                               note="March onset rows = v12's: v13.1's nl75 was chosen on March pseudo windows that lie "
                                    "after the scored March origins (rulings 748789 / 749601)")
    else:
        report["march"] = dict(note="march_compliant=False: the March onset rows stay v13.1's")

    # ======================================================================= files
    d = C.WORK / "pred"
    _save_probs(d / PROBS_FILE, dict(pk, probs=probs, v15=dict(settings=st, ongoing=take_og,
                                                               onset=report["onset"]["adopted"])))
    with Y.decoder(Y.DEC["mode"]):
        main = T.write_rows([probs["early"], probs["late"], probs["april"], probs["std"]], offis, pans, Ps)
        hedge = T.write_rows([probs["std"]], offis, pans, Ps)
        probe = (T.write_rows([probe_set, probs["early"], probs["late"], probs["april"], probs["std"]], offis, pans, Ps)
                 if probe_set is not None else None)
        main0 = T.write_rows([probs0["early"], probs0["late"], probs0["april"], probs0["std"]], offis, pans, Ps)
        hedge0 = T.write_rows([probs0["std"]], offis, pans, Ps)
    a = main.set_index(KEY).queue_pred
    h = hedge.set_index(KEY).queue_pred
    TM.check_rows(a, "the v15 main file"); TM.check_rows(h, "the v15 hedge file")
    a0 = main0.set_index(KEY).queue_pred
    h0 = hedge0.set_index(KEY).queue_pred
    TM.check_rows(a0, "the v15 fallback main file"); TM.check_rows(h0, "the v15 fallback hedge file")
    ids = X._ids(offis, pans)
    mar = a.index.get_level_values(0).str.contains("_validation_")
    if not (a[mar] == h.reindex(a.index)[mar]).all():
        raise RuntimeError("T2 v15: the hedge's March rows must be the main file's")
    checks = {}
    olds = {}
    for fn, nm in ((V131_MAIN, "v13.1 main"), (V131_HEDGE, "v13.1 hedge"), (V12_MAIN, "v12 main")):
        f = d / fn
        if not f.exists():
            raise FileNotFoundError(f"T2 v15: pred/{fn} is needed for the checks")
        olds[nm] = pd.read_csv(f, dtype=QDT).set_index(KEY).queue_pred
    for new, nm in ((a, "v13.1 main"), (h, "v13.1 hedge")):
        for c in T.CONDS:
            for s, m in (("validation", "March"), ("private", "April")):
                checks[f"{m} {c} rows ({nm.split()[1]}) vs the {nm} file"] = TM._compare(new, olds[nm], ids[c][s])
    # the invariants: March ongoing rows untouched; March onset rows = v12's when compliant; the main file's April onset
    # rows = v13.1's; nothing ongoing moves without the ongoing change, nothing onset in the hedge without the onset step
    for new, nm in ((a, "v13.1 main"), (h, "v13.1 hedge")):
        if checks[f"March {OG} rows ({nm.split()[1]}) vs the {nm} file"]["cells_differ"]:
            raise RuntimeError(f"T2 v15: March ongoing cells differ from the {nm} file")
        if not take_og and checks[f"April {OG} rows ({nm.split()[1]}) vs the {nm} file"]["cells_differ"]:
            raise RuntimeError(f"T2 v15: the ongoing change was not adopted, yet April ongoing cells differ ({nm})")
    # the main file takes an April window from early / late / april before std: only std-sourced onset windows may move
    refit_on = {wid for k in ("early", "late", "april") for dd in _part(probs[k], ON, "private").values()
                for wid in dd.get("private", {})}
    std_on = set(ids[ON]["private"]) - refit_on
    report["onset"]["main_std_windows"] = len(std_on)
    cmr = TM._compare(a, olds["v13.1 main"], refit_on) if refit_on else dict(cells_differ=0)
    checks["April queue_onset rows of the April refit (main) vs the v13.1 main file"] = cmr
    if cmr["cells_differ"] or (std_on and not report["onset"]["adopted"]
                               and TM._compare(a, olds["v13.1 main"], std_on)["cells_differ"]):
        raise RuntimeError("T2 v15: the main file's April onset rows must be v13.1's (outside the onset step)")
    if not report["onset"]["adopted"] and checks[f"April {ON} rows (hedge) vs the v13.1 hedge file"]["cells_differ"]:
        raise RuntimeError("T2 v15: the onset step was not adopted, yet the hedge's April onset cells differ")
    if st["march_compliant"]:
        cm = TM._compare(a, olds["v12 main"], ids[ON]["validation"])
        checks["March queue_onset rows (main) vs the v12 main file"] = cm
        if cm["cells_differ"]:
            raise RuntimeError("T2 v15: the March onset rows must be v12's (march_compliant)")
    # the fallback pair: March rows = the v15 pair's, April rows = v13.1's
    for new0, new, old, nm in ((a0, a, olds["v13.1 main"], "main"), (h0, h, olds["v13.1 hedge"], "hedge")):
        m0 = new0.index.get_level_values(0).str.contains("_validation_")
        if not (new0[m0] == new.reindex(new0.index)[m0]).all():
            raise RuntimeError(f"T2 v15: the fallback {nm} file's March rows must be the v15 {nm} file's")
        if not (new0[~m0] == old.reindex(new0.index)[~m0]).all():
            raise RuntimeError(f"T2 v15: the fallback {nm} file's April rows must be v13.1's")
    if probe is not None:
        pb = probe.set_index(KEY).queue_pred
        TM.check_rows(pb, "the v15 ongoing probe")
        on_mar = pb.index.get_level_values(0).isin(list(ids[OG]["validation"]))
        if not (pb[~on_mar] == a.reindex(pb.index)[~on_mar]).all():
            raise RuntimeError("T2 v15: the ongoing probe may differ from the main file on the March ongoing windows only")
        checks["March queue_ongoing rows (probe) vs the v15 main file"] = TM._compare(pb, a, ids[OG]["validation"])
    report["checks"].update(checks)
    report["agreement_main_hedge"] = {m: dict(cells_main=int(a[sel].sum()), cells_hedge=int(h.reindex(a.index)[sel].sum()),
                                              differ=int((a[sel] != h.reindex(a.index)[sel]).sum()))
                                      for m, sel in (("March", mar), ("April", ~mar))}
    for fr, fn in ((main, MAIN_FILE), (hedge, HEDGE_FILE), (probe, PROBE_FILE), (main0, MAIN0_FILE), (hedge0, HEDGE0_FILE)):
        if fr is not None:
            X._write(fr, d, fn)
    if probe is None:
        (d / PROBE_FILE).unlink(missing_ok=True)
    report["probe"] = PROBE_FILE if probe is not None else None
    report["fallback"] = dict(main=MAIN0_FILE, hedge=HEDGE0_FILE, note="v13.1's April rows with the v15 March rows")
    report["changed"] = dict(ongoing=take_og, onset=bool(report["onset"]["adopted"]),
                             march_onset=int((report.get("march") or {}).get("onset_windows_reverted_to_v12", 0)))
    report["device"] = dict(params=dict(device or {}), failed=TM.DEVICE["failed"], fits_gpu=TM.DEVICE["fits_gpu"],
                            fits_cpu=TM.DEVICE["fits_cpu"])
    report["minutes"]["total"] = round((time.time() - t0) / 60, 1)
    report["files_md5"] = {fn: T._md5(d / fn) for fn in (MAIN_FILE, HEDGE_FILE, MAIN0_FILE, HEDGE0_FILE)
                           + ((PROBE_FILE,) if probe is not None else ())}
    report["probs_md5"] = T._md5(d / PROBS_FILE)
    tmp = d / (REPORT_FILE + ".part")
    json.dump(report, open(tmp, "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    tmp.replace(d / REPORT_FILE)   # written last: it vouches for the files
    C.log(f"T2 {VERSION} done in {report['minutes']['total']} min: {report['changed']} | main April ongoing "
          f"{json.dumps((report['ongoing'].get('main') or {}), default=str)[:300]}")
    return report


def _main_og(Dl, pseudo, offis, pans, Ps, prm, rounds, variants, seeds, fset, alpha, cut, probs12, probs, dw, st) -> dict:
    """The ongoing change on top of v12's April refit of the main file (train + all March pseudo windows), as a change of
    v12's stored April fields - applied only when, within March, the same change on the gate models (train + March before
    the cut, v12's checkpoints) does not lose on the March windows after the cut."""
    out = dict(applied=False, cut_day=int(cut))
    apr12 = _part(probs12["april"], OG, "private")
    if not apr12:
        out["reason"] = "v12 adopted no April ongoing refit: the main file's April ongoing rows are the standard ones"
        return out
    if not st["main_check"]:
        out["reason"] = "main_check=False: the main file keeps v12's April ongoing rows"
        return out
    Dg = T.pseudo_train_rows(pseudo, pans, "validation", OG, day_hi=cut)
    if Dg is None or len(Dg) == 0:
        out["reason"] = "no March pseudo windows before the cut"
        return out
    n0 = _nfit()
    g = T.fit_set(Dl, OG, prm, rounds, variants, seeds[:1], Dg, alpha, role="gate", fset=fset)
    out["gate_models_refitted"] = _nfit() - n0
    gfa = _pseudo(T._models_for(g, variants, seeds[:1], ["all"]), pseudo, pans, Ps, OG)
    gfn = _pseudo(T._models_for(g, variants, seeds[:1], ["noloc"]), pseudo, pans, Ps, OG)
    del g, Dg
    C.free()
    Rb = _rows({p: 0.5 * gfa[p] + 0.5 * gfn[p] for p in gfa}, pseudo, pans, Ps, OG, day_lo=cut)
    Rc = _rows({p: 0.5 * gfa[p] + 0.5 * gfn[p] + dw * (gfn[p] - gfa[p]) for p in gfa}, pseudo, pans, Ps, OG, day_lo=cut)
    if len(Rb) == 0:
        out["reason"] = "no March pseudo windows after the cut"
        return out
    pst, _ = T.paired(Rb, Rc)
    out["march_second_half"] = {k: pst[k] for k in ("n_sim", "n_all", "S_a_sim", "S_b_sim", "d_sim", "se_sim", "d_all",
                                                     "se_all", "better", "worse")}
    ok = bool(np.nan_to_num(pst["d_sim"]) >= -np.nan_to_num(pst["se_sim"]) and pst["d_all"] >= -np.nan_to_num(pst["se_all"]))
    if st["force15"] in ("og", "both"):
        ok, out["forced"] = True, True
    out["passes"] = ok
    if not ok:
        out["reason"] = "the change loses within March on the April-refit models: the main file keeps v12's April ongoing rows"
        return out
    Df = T.pseudo_train_rows(pseudo, pans, "validation", OG)
    n1 = _nfit()
    ad = T.fit_set(Dl, OG, prm, rounds, variants, seeds, Df, alpha, role="adapt", fset=fset)
    out["adapt_models_refitted"] = _nfit() - n1   # 0 = v12's April refit reloaded from its checkpoints
    afa = _official(ad, variants, seeds, "all", offis, pans, Ps, OG, ("private",))
    afn = _official(ad, variants, seeds, "noloc", offis, pans, Ps, OG, ("private",))
    del ad, Df
    C.free()
    mda = max(_maxdiff(apr12, _blend(afa, afn, 0.5)), _maxdiff(_blend(afa, afn, 0.5), apr12))
    out["reload_max_diff"] = mda
    if not mda <= float(st["reload_tol"]):
        out["reason"] = (f"the reloaded April refit does not reproduce v12's April fields (max diff {mda:.3g}): the main "
                         "file keeps v12's April ongoing rows")
        return out
    n = 0
    for p, dd in apr12.items():
        for wid, b in dd.get("private", {}).items():
            probs["april"][p]["private"][wid] = (OG, _reweight(b, afa[p]["private"][wid], afn[p]["private"][wid], dw))
            n += 1
    out.update(applied=True, windows=n)
    return out
