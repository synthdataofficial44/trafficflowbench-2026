"""v12 Task 2: eligibility-aware onset decoding behind a two-draw gate, three ongoing seeds and the physics forecast as
an ongoing candidate - on top of v11 (task2x) and v10 (task2 / task2m) without editing them, so every v10/v11
checkpoint in Drive stays valid and reloads.

Eligibility-aware decoding (onset). The official IoU counts eligible cells only (pct_observed >= 75 and non-null, about
76 % of cells, independent from cell to cell and over time - diag11), and a window whose union is empty on the eligible
cells scores 1. The official selector asks for a queued cell anywhere in the horizon, not for an eligible one, so a few
percent of the onset windows have no eligible true cell at all (5 % of the train replica's windows; one of the 40
shipped train windows): there an answer scores 1 only if none of its own cells is eligible. The v2-v11 decoder (top-m
cells by the expected-IoU surrogate sum_S p / (|S| + sum_notS p)) ignores both effects. Here, per window and for
m = 1..M, the expected official IoU of the top-m set is computed by Monte Carlo under the model's own marginals (the
true set drawn cell by cell, conditioned on holding a queue somewhere) and independent eligibility at the rate measured
on the T+30 slots of the train onset windows; the best m is taken ("ead"). "exact" is the same with every cell eligible
(the surrogate replaced by the exact expectation). Nothing reads the eligibility of a horizon cell: the rate is a
train statistic, the same for every cell.
Gate: week-blocked 4-fold train CV of the onset recipe (out of fold, scored with the official rule on every
selector-drawn train window, draw-weighted, day-block bootstrap SE) must gain more than `ead_z` SEs, and on the March
pseudo windows (which always hold a known queued cell, so they cannot show the empty-union gain) the change must not be
worse than what the same decoder does on the train windows that hold an eligible queued cell, by more than `ead_zm`
SEs (the model's probabilities must be as trustworthy on another demand draw as on train). The decoder is a rule fixed
from train evidence, so when adopted it decodes the onset windows of both months in both files.

Ongoing. Three seeds instead of one (variance), and v2+pm (the cell-transmission forecast of v7, so far onset-only) as a
candidate through the v5/v6 two-draw gate against v2+m.

Files: main (as v11: refits override the standard rows), hedge (standard train-only models with the month memory for
both months; its March rows are the main file's), and when the decoder changed, main/hedge decoded with the v11 rule
(`*_topm`): main - main_topm on the public board reads the decoder alone, main_topm - v11 main the ongoing change.
The probability sets are saved, so the files can be decoded again without refitting.
Only data timestamped <= T feeds a window at origin T; no April label is evaluated or used to choose anything.
"""
from __future__ import annotations

import contextlib
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

VERSION = "v12"
MAIN_FILE, HEDGE_FILE = "task2_v12.csv", "task2_v12_hedge.csv"
MAIN_TOPM, HEDGE_TOPM = "task2_v12_topm.csv", "task2_v12_hedge_topm.csv"
REPORT_FILE = "task2_v12_report.json"
PROBS_FILE = "task2_v12_probs.pkl"
V11_MAIN, V11_HEDGE = X.MAIN_FILE, X.HEDGE_FILE
DEFAULT_CANDIDATES = {
    "queue_onset": [["v2+pm", ["all", "noloc"]]],
    "queue_ongoing": [["v2+m", ["all", "noloc"]], ["v2+pm", ["all", "noloc"]]],
}
DECODERS = ("topm", "exact", "ead")
DEC = dict(mode="topm", rho=0.764, n=3000, M=0, seed=2026)   # M = 0: no cap on the decoded set size
KEY = TM.KEY
QDT = TM.QDT
_ORIG: dict = {}


# ----------------------------------------------------------------------------- installation into task2
def _unwrap(f):
    return getattr(f, "_v12_orig", f)


def install():
    """v11's pieces (task2x.install) + the onset decoder switch. Idempotent and reload-safe like task2x.install: the
    original decode_window is found by unwrapping whatever task2 holds."""
    X.install()
    if T.decode_window is decode_window:
        return
    _ORIG["dw"] = _unwrap(T.decode_window)
    decode_window._v12_orig = _ORIG["dw"]
    T.decode_window = decode_window


def decode_window(cond, p, L):
    """task2.decode_window; the onset windows are decoded by DEC['mode'] ('topm' = the v2-v11 rule)."""
    if cond == "queue_onset" and DEC["mode"] != "topm":
        out = np.zeros((T.K, L), bool)
        out[T.K - 1] = onset_set(p, DEC["mode"])
        return out
    return _ORIG["dw"](cond, p, L)


@contextlib.contextmanager
def decoder(mode):
    """Temporarily decode the onset windows with `mode`."""
    if mode not in DECODERS:
        raise ValueError(f"unknown onset decoder {mode!r}: one of {DECODERS}")
    old = DEC["mode"]
    DEC["mode"] = mode
    try:
        yield
    finally:
        DEC["mode"] = old


def code_hash() -> str:
    h = hashlib.md5((X.code_hash() + "|").encode())
    h.update(Path(__file__).read_bytes())
    return h.hexdigest()[:16]


# ----------------------------------------------------------------------------- the decoder
def eiou_curve(p, rho=1.0, n=3000, M=0, seed=2026):
    """(top-m cell order [M], expected official IoU of the top-m set for m = 1..M [M]) under independent truth cells
    with marginals p (conditioned on at least one true cell) and independent eligibility at rate rho; M <= 0: every
    cell (no cap on the set size); None when the marginals almost never produce a queue (then the caller keeps the
    surrogate rule)."""
    p = np.clip(np.nan_to_num(np.asarray(p, np.float64), nan=0.0), 0.0, 1.0)
    L = len(p)
    if L == 0 or p.max() <= 0:
        return None
    order = np.argsort(-p, kind="stable")
    M = int(min(M, L)) if M and M > 0 else L
    cand = order[:M]
    act = np.union1d(np.flatnonzero(p > 1e-4), cand)
    pos = np.searchsorted(act, cand)
    rng = np.random.default_rng(seed)
    Tm = rng.random((n, len(act))) < p[act][None, :]
    keep = Tm.any(1)
    if keep.sum() < max(50, n // 20):
        return None
    Tm = Tm[keep]
    E = (rng.random(Tm.shape) < rho) if rho < 1.0 else np.ones(Tm.shape, bool)
    te = Tm & E
    fe = E & ~Tm
    base = te.sum(1)
    inter = np.cumsum(te[:, pos], 1)
    union = base[:, None] + np.cumsum(fe[:, pos], 1)
    iou = np.where(union > 0, inter / np.maximum(union, 1), 1.0)
    return cand, iou.mean(0)


def onset_set(p, mode):
    """Bool [L]: the onset cells at T+30 of one window under decoder `mode`."""
    if mode == "topm":
        return T.eiou_topm(np.asarray(p, np.float64), 1)
    rho = 1.0 if mode == "exact" else float(DEC["rho"])
    r = eiou_curve(p, rho, int(DEC["n"]), int(DEC["M"]), int(DEC["seed"]))
    if r is None:
        return T.eiou_topm(np.asarray(p, np.float64), 1)
    cand, e = r
    m = int(np.argmax(e)) + 1          # the first maximum: ties go to the smaller set
    sel = np.zeros(len(p), bool)
    sel[cand[:m]] = True
    return sel


# ----------------------------------------------------------------------------- the decoder gate
def eligibility_rate(pans) -> float:
    """Share of eligible cells at T+30 over every onset train window (the slots the onset answer is scored on)."""
    s = n = 0
    for p in pans:
        if p in X._EL30:
            el30 = np.asarray(X._EL30[p][0], bool)
            s += int(el30.sum()); n += int(el30.size)
    return float(s / n) if n else float("nan")


def _fit_cached(Dl, cond, prm, rounds, fs, fset, nl, seed, role, masks):
    """task2.fit_lgb on the masked rows, checkpointed like the other models (role 'exp:<tag>')."""
    path = T._ckpt_path(cond, role, nl, seed, rounds, fset, fs)
    m = T._load_ckpt(path, fs, rounds)
    if m is None:
        m = T.fit_lgb(Dl, prm, rounds, fs, None, seed=seed, masks=masks)
        T._save_ckpt(m, path)
    return m


def onset_oof(Dl, sets, pans, Ps, prm, rounds, fset, variants, seed=0) -> dict:
    """{panel: [n_windows, L]} out-of-fold onset probabilities of the train windows: week-block folds 0-3 (the location
    and onset priors in the tables are out of fold the same way), the recipe's variants averaged, seed `seed`."""
    cond = "queue_onset"
    names = [v for v, _ in variants]
    oof = [np.zeros(len(f)) for f in Dl]
    seen = [np.zeros(len(f), bool) for f in Dl]
    with_e = "e" in T._check_fset(fset)[1]   # as task2x.fit_set: an "e" recipe trains without the windows it drops
    for k in range(4):
        tr = [((f.fold != k) & ((f.elok > 0.5) if (with_e and "elok" in f) else True)).to_numpy() for f in Dl]
        te = [(f.fold == k).to_numpy() for f in Dl]
        if not any(m.any() for m in tr) or not any(m.any() for m in te):
            continue
        for v in names:
            fs = T._feats(Dl[0], v, fset)
            mdl = _fit_cached(Dl, cond, prm, rounds, fs, fset, v, seed, f"exp:oof{k}", tr)
            for j, f in enumerate(Dl):
                if te[j].any():
                    oof[j][te[j]] += mdl.predict(T._matrix([f], fs, [te[j]]), num_threads=C.THREADS) / len(names)
                    seen[j] |= te[j]
            del mdl
            C.free()
    out = {}
    for j, f in enumerate(Dl):
        p = pans[int(f.panel_code.iat[0])]
        Wc = sets[p][cond][0]
        pr = np.where(seen[j], oof[j], 0.0)
        out[p] = T._field(cond, f, pr, len(Wc), Ps[p]["L"])
    return out


def decoder_cv(fields, sets, pans, Ps, modes=DECODERS) -> pd.DataFrame:
    """Per selector-drawn train window: the official IoU (eligible cells at T+30 only; an empty union scores 1) of
    every decoder, from the out-of-fold fields."""
    cond = "queue_onset"
    rows = []
    for p in pans:
        if p not in fields or p not in X._EL30 or cond not in sets.get(p, {}):
            continue
        Wc, yc = sets[p][cond][0], sets[p][cond][1]
        el30, ok = X._EL30[p]
        F = fields[p]
        for w in np.flatnonzero((Wc.src == "sim").to_numpy()):
            tru = np.asarray(yc[w][T.K - 1], bool) & el30[w]
            rec = dict(panel=p, w=int(w), day=int(Wc.day.iat[w]), src="sim", draws=int(Wc.draws.iat[w]),
                       elok=bool(ok[w]), n_true=int(tru.sum()), n_true_all=int(np.asarray(yc[w][T.K - 1], bool).sum()))
            for mode in modes:
                sel = onset_set(F[w], mode)
                pe = sel & el30[w]
                u = int((pe | tru).sum())
                rec[f"iou_{mode}"] = 1.0 if u == 0 else float((pe & tru).sum() / u)
                rec[f"m_{mode}"] = int(sel.sum())
            rows.append(rec)
    return pd.DataFrame(rows)


def decoder_gate(Dl, sets, pans, Ps, prm, rounds, fset, variants, std, pseudo, cfg) -> dict:
    """Train CV (official rule) and March pseudo windows for the decoders; sets DEC['mode'] to the adopted one."""
    t0 = time.time()
    rho = cfg.get("ead_rho")
    rho = float(rho) if rho else eligibility_rate(pans)
    if not np.isfinite(rho) or not 0.3 <= rho <= 1.0:
        rho = 0.764
    DEC.update(rho=rho, n=int(cfg.get("ead_n", 3000)), M=int(cfg.get("ead_M") or 0), seed=int(cfg.get("ead_seed", 2026)))
    z, zm = float(cfg.get("ead_z", 2.0)), float(cfg.get("ead_zm", 2.0))
    rep = dict(rho=round(rho, 4), n=DEC["n"], M=DEC["M"], z=z, zm=zm, candidates={}, adopted="topm")
    fields = onset_oof(Dl, sets, pans, Ps, prm, rounds, fset, variants, seed=int(cfg.get("seed", 0)))
    R = decoder_cv(fields, sets, pans, Ps)
    if len(R) == 0:
        rep["reason"] = "no train onset windows"
        DEC["mode"] = "topm"
        return rep
    rep["cv_level"] = {m: round(T._agg1(R, f"iou_{m}", "draws"), 4) for m in DECODERS}
    rep["cv_level_elok"] = {m: round(T._agg1(R[R.elok], f"iou_{m}", "draws"), 4) for m in DECODERS}
    rep["cv_windows"] = int(len(R)); rep["cv_windows_no_eligible_truth"] = int((~R.elok).sum())
    rep["cv_mean_cells"] = {m: round(float(R[f"m_{m}"].mean()), 3) for m in DECODERS}
    Rm = {}
    has_march = any("queue_onset" in pseudo[p].get("validation", {}) for p in pans)
    if has_march:
        for m in DECODERS:
            with decoder(m):
                Rm[m] = T.pseudo_scores(std, pseudo, pans, Ps, "validation", "queue_onset")
    # share of the train replica's onset windows without an eligible true cell (the official draw has the same rule)
    s_noel = float((~R.elok).mean())
    rep["share_no_eligible_truth"] = round(s_noel, 4)
    best, best_d = "topm", 0.0
    for m in ("exact", "ead"):
        Rd = R.assign(d=R[f"iou_{m}"] - R["iou_topm"])
        st = dict(d_cv=T._agg1(Rd, "d", "draws"), se_cv=T._boot_se(Rd, "d", "draws"))
        Re = Rd[Rd.elok]
        st.update(d_cv_elok=T._agg1(Re, "d", "draws"), se_cv_elok=T._boot_se(Re, "d", "draws"),
                  d_cv_no_eligible_truth=float(Rd.loc[~Rd.elok, "d"].mean()) if (~Rd.elok).any() else 0.0,
                  better=int((Rd.d > 1e-9).sum()), worse=int((Rd.d < -1e-9).sum()))
        # (a) the official rule gains on train; (b) not by trading the windows that hold an eligible queued cell for the
        # empty-union ones: those must not lose by more than their SE
        cv_ok = bool(np.isfinite(st["se_cv"]) and st["se_cv"] > 0 and st["d_cv"] > z * st["se_cv"])
        elok_ok = bool(np.isfinite(st["d_cv_elok"]) and st["d_cv_elok"] >= -np.nan_to_num(st["se_cv_elok"]))
        if has_march and len(Rm.get(m, [])):
            ps, M_ = T.paired(Rm["topm"], Rm[m])
            st.update(march=dict(n=ps["n_all"], d_all=ps["d_all"], se_all=ps["se_all"], d_sim=ps["d_sim"],
                                 se_sim=ps["se_sim"], better=ps["better"], worse=ps["worse"],
                                 cells_topm=int(Rm["topm"].n_pred.sum()), cells=int(Rm[m].n_pred.sum())))
            # (c) March (every pseudo window holds a known queued cell) behaves like train on such windows, within zm
            # SEs, and is not below -zm SEs either; (d) the March official effect it implies - its own change on those
            # windows plus train's empty-union gain at the train share - is not negative
            floor = min(st["d_cv_elok"], 0.0) - zm * np.nan_to_num(ps["se_all"])
            est = (1.0 - s_noel) * ps["d_all"] + s_noel * st["d_cv_no_eligible_truth"]
            march_ok = bool(np.isfinite(ps["d_all"]) and ps["d_all"] >= floor and est >= 0.0)
            st.update(march_floor=floor, march_official_est=est)
        else:
            march_ok = False
            st["march"] = None
        take = bool(cv_ok and elok_ok and march_ok)
        if cfg.get("ead_force") == m:   # tests only
            take, st["forced"] = True, True
        st.update(cv_ok=cv_ok, elok_ok=elok_ok, march_ok=march_ok, take=take)
        rep["candidates"][m] = {k: (round(v, 5) if isinstance(v, float) else v) for k, v in st.items()}
        if take and (st["d_cv"] > best_d or best == "topm"):
            best, best_d = m, st["d_cv"]
    if not cfg.get("ead", True):
        best = "topm"
        rep["reason"] = "switched off (ead=False): measured only"
    rep["adopted"] = best
    rep["minutes"] = round((time.time() - t0) / 60, 1)
    DEC["mode"] = best
    return rep


# ----------------------------------------------------------------------------- settings
def candidates(cfg) -> dict:
    c = {k: [list(x) for x in v] for k, v in DEFAULT_CANDIDATES.items()}
    c.update({k: [list(x) for x in v] for k, v in (cfg.get("candidates") or {}).items()})
    for cond, cands in c.items():
        if cond not in T.CONDS:
            raise ValueError(f"candidates: unknown condition {cond!r}")
        if not cands:
            raise ValueError(f"candidates[{cond}]: empty")
        for f, vn in cands:
            T._check_fset(f)
            [T._variant_name(v) for v in vn]
    return c


def task_cfg(cfg) -> dict:
    return dict(cfg, configs=candidates(cfg), probe_onset=None, probe_ongoing=None, experiments=True,
                online_adapt=False, pseudo_april=False)


def _norm(d) -> str:
    return json.dumps({k: v for k, v in (d or {}).items() if k not in ("model_dir",)}, sort_keys=True, default=str)


def _file_set(rep) -> set:
    fs = {MAIN_FILE, HEDGE_FILE}
    if (rep or {}).get("decoder", {}).get("adopted", "topm") != "topm":
        fs |= {MAIN_TOPM, HEDGE_TOPM}
    return fs


def up_to_date(cfg, work=None) -> tuple:
    """(True, report) when the v12 files of this code, these settings and this release are on disk."""
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
    ok = (rep.get("version") == VERSION and rep.get("code") == code_hash() and _norm(rep.get("cfg")) == _norm(cfg)
          and rep.get("fingerprint") == fp and set(fm) >= _file_set(rep)
          and all((d / fn).exists() and T._md5(d / fn) == h for fn, h in fm.items()))
    return ok, rep


# ----------------------------------------------------------------------------- the run
def _save_probs(path: Path, obj):
    tmp = path.with_suffix(".part")
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=4)
    tmp.replace(path)


def write_files(probs: dict, offis, pans, Ps, mode: str) -> dict:
    """{name: rows} of the main and hedge files decoded with onset decoder `mode` (+ the v11-rule versions when
    `mode` is not 'topm')."""
    main_sets = [probs["early"], probs["late"], probs["april"], probs["std"]]
    out = {}
    with decoder(mode):
        out[MAIN_FILE] = T.write_rows(main_sets, offis, pans, Ps)
        out[HEDGE_FILE] = T.write_rows([probs["std"]], offis, pans, Ps)
    if mode != "topm":
        with decoder("topm"):
            out[MAIN_TOPM] = T.write_rows(main_sets, offis, pans, Ps)
            out[HEDGE_TOPM] = T.write_rows([probs["std"]], offis, pans, Ps)
    return out


def onset_cells(rows: pd.Series, ids) -> dict:
    """Cells predicted per onset window (mean, min, max) over the windows `ids`."""
    s = rows[rows.index.get_level_values(0).isin(list(ids))]
    if len(s) == 0:
        return {}
    per = s.groupby(level=0).sum()
    return dict(windows=int(len(per)), mean=round(float(per.mean()), 3), min=int(per.min()), max=int(per.max()))


def run_main(cfg: dict, device: dict | None = None) -> dict:
    """Writes pred/task2_v12.csv (main), pred/task2_v12_hedge.csv, the v11-rule versions *_topm.csv when the onset
    decoder changed (their April rows come from the same April gates, which ran under the adopted decoder),
    pred/task2_v12_probs.pkl (every window's probabilities) and pred/task2_v12_report.json (last).
    `device`: extra LightGBM training parameters ({} / None = the CPU). The onset decoder switch is back on the v11
    rule afterwards, also when the run fails."""
    try:
        return _run_main(cfg, device)
    finally:
        DEC["mode"] = "topm"


def _run_main(cfg: dict, device: dict | None = None) -> dict:
    from .data import fingerprint
    install()
    t0 = time.time()
    DEC["mode"] = "topm"
    tcfg = task_cfg(cfg)
    cands = tcfg["configs"]
    TM.use_device(device)
    T.set_checkpoints(tcfg)
    pans = C.task2_panels()
    rng = np.random.default_rng(tcfg.get("seed", 0))
    imputer = T.fit_imputer(pans) if tcfg.get("imputer", True) else None
    C.log(f"T2 v12 label imputer ready ({time.time() - t0:.0f}s); table groups onset "
          f"{T.extra_groups(tcfg, 'queue_onset')!r}, ongoing {T.extra_groups(tcfg, 'queue_ongoing')!r}")
    sets, offis, Ps, pseudo = {}, {}, {}, {}
    for i, p in enumerate(pans):
        P, out, offi, ps = T.build_panel_sets(p, imputer, rng, tcfg)
        for split in ps:
            for cond in ps[split]:
                ps[split][cond][3]["panel_code"] = np.float32(i)
        sets[p] = out; offis[p] = offi; pseudo[p] = ps
        Ps[p] = dict(L=P.L, links=P.net.links, mp=np.asarray(P.mp, np.float64))
        npw = ", ".join(f"{s[:3]} {c.split('_')[1]} {len(v[0])}" for s in ps for c, v in ps[s].items())
        C.log(f"T2 v12 {p}: train windows onset {len(out.get('queue_onset', ([],))[0])}, "
              f"ongoing {len(out.get('queue_ongoing', ([],))[0])}; pseudo windows: {npw or 'none'}")
        P.dprof = None
        del P
        C.free()
    report = dict(version=VERSION, code=code_hash(), task2_code=T.code_hash(), x_code=X.x_hash(), fingerprint=fingerprint(),
                  cfg=cfg, candidates=cands, chosen={}, choose={}, decoder={}, pseudo={}, diag={}, panel_tod={}, adapt={},
                  locfree={}, late={}, early={}, minutes_by_step={}, windows=T.official_window_calendar(offis, pans))
    probs = dict(std={}, april={}, late={}, early={})
    april_src = {}
    for cond in T.CONDS:
        tc = time.time()
        Dl = T._frames(sets, pans, cond)
        if not Dl:
            raise RuntimeError(f"T2 v12: no training windows for {cond}")
        on = cond == "queue_onset"
        prm, rounds = (T.ONSET_P, tcfg.get("onset_rounds", 500)) if on else (T.ONG_P, tcfg.get("ongoing_rounds", 700))
        seeds = list(tcfg.get("onset_seeds" if on else "ongoing_seeds", [tcfg.get("seed", 0)]))
        lseeds = list(tcfg.get("late_seeds", seeds[:1]))
        has_march = any(cond in pseudo[p].get("validation", {}) for p in pans)
        C.log(f"T2 v12 {cond}: {sum(len(f) for f in Dl):,} training rows; candidates {cands[cond]}; seeds {seeds}")
        # ---- the recipe: the v10/v11 recipe (first candidate) unless a candidate wins on both draws (v5/v6 rules)
        ccfg = dict(tcfg, exp_cv=bool(tcfg.get("exp_cv", True)) and (on or bool(tcfg.get("exp_cv_ongoing", False))))
        if len(cands[cond]) > 1 and has_march and tcfg.get("choose", True):
            (fset, variants), crep = T.choose_config(Dl, cond, pseudo, pans, Ps, prm, rounds, seeds, ccfg, sets=sets)
            report["choose"][cond] = crep
            C.log(f"T2 v12 recipe {cond}: chosen {crep.get('chosen')} | "
                  + " | ".join(f"{k}: gain {v.get('gain', float('nan')):+.4f} se {v.get('se_gain', float('nan')):.4f} "
                               f"cv {v.get('d_cv', float('nan')):+.4f} take {v.get('take')}" for k, v in crep.get("candidates", {}).items()))
        else:
            fset, variants = cands[cond][0][0], T._variants(cands[cond][0][1])
        report["chosen"][cond] = [fset, [v for v, _ in variants]]
        # ---- standard models (train only): the March rows of both files and the April rows of the hedge
        std = T.fit_set(Dl, cond, prm, rounds, variants, seeds, fset=fset)
        # ---- the onset decoder (train CV + March), decided before anything is decoded or gated on March
        if on and tcfg.get("ead_gate", True):
            drep = decoder_gate(Dl, sets, pans, Ps, prm, rounds, fset, variants, std, pseudo, tcfg)
            report["decoder"] = drep
            C.log(f"T2 v12 onset decoder: {drep.get('adopted')} | rho {drep.get('rho')} | cv {drep.get('cv_level')} | "
                  + " | ".join(f"{k}: cv {v.get('d_cv'):+.4f} se {v.get('se_cv'):.4f} march "
                               f"{(v.get('march') or {}).get('d_all', float('nan')):+.4f} take {v.get('take')}"
                               for k, v in drep.get("candidates", {}).items()))
        T.predict_official(std, offis, pans, Ps, cond, probs["std"], ("validation", "private"))
        if has_march:
            R_ = T.pseudo_scores(std, pseudo, pans, Ps, "validation", cond)
            report["pseudo"][cond] = dict(main=T.pseudo_report(R_))
            report["diag"][cond] = T.window_diag(R_, cond)
            report["panel_tod"][cond] = X.panel_tod(R_, cond)
            del R_
        # ---- April refit on train + all March pseudo windows (gate: first half of March -> second half)
        adapted, ad = False, None
        if tcfg.get("april_adapt", True) and has_march:
            rep_ad, ad = T.april_adapt(Dl, std, pseudo, pans, Ps, cond, prm, rounds, variants, seeds, tcfg, fset)
            report["adapt"][cond] = rep_ad
            C.log(f"T2 v12 April adaptation {cond}: {rep_ad}")
        april_src[cond] = "standard"
        if on and tcfg.get("locfree", True) and has_march and tcfg.get("april_adapt", True):
            lrep = X.locfree_gate(Dl, std, pseudo, pans, Ps, cond, prm, rounds, variants, seeds, tcfg, fset)
            report["locfree"][cond] = lrep
            C.log(f"T2 v12 location-free April refit {cond}: {lrep}")
            if lrep["adopted"]:
                Df = T.pseudo_train_rows(pseudo, pans, "validation", cond)
                alpha = float(tcfg.get("adapt_alpha", 2.0))
                if ad is None:
                    ad = T.fit_set(Dl, cond, prm, rounds, variants, seeds, Df, alpha, role="adapt", fset=fset)
                vlf = T._variants([lrep["variant"]])
                ad_lf = T.fit_set(Dl, cond, prm, rounds, vlf, seeds, Df, alpha, role="adapt", fset=fset)
                del Df
                ad = list(ad) + list(ad_lf) * len(variants)
                april_src[cond] = "blend of the full and the location-free refit"
        if ad is not None:
            T.predict_official(ad, offis, pans, Ps, cond, probs["april"], ("private",))
            adapted = True
            if april_src[cond] == "standard":
                april_src[cond] = "full refit (v10 rule)"
        del ad
        C.free()
        # ---- late April windows: + April's own earlier pseudo windows (+ March when adapted)
        if tcfg.get("late_adapt", False) and cond in tcfg.get("late_conds", list(T.CONDS)) and has_march:
            n_late = sum(T.n_official(offis, pans, cond, "private", lo, hi) for lo, hi in T.late_blocks(tcfg, "private"))
            if n_late == 0:
                report["late"][cond] = dict(adopted=False, reason="no official April window of this condition in the late blocks")
            else:
                lrep = T.late_gate(Dl, std[:: len(seeds)], pseudo, pans, Ps, cond, prm, rounds, variants, tcfg, fset,
                                   seed=seeds[0])
                lrep["official_late_windows"] = n_late
                if lrep["adopted"]:
                    lrep["main"] = T.late_fit_predict(Dl, pseudo, offis, pans, Ps, cond, prm, rounds, variants, lseeds,
                                                      tcfg, fset, adapted, probs["late"], "main")
                report["late"][cond] = lrep
                C.log(f"T2 v12 late windows {cond}: {lrep}")
        # ---- early April windows (v8 blocks)
        if T.early_blocks(tcfg, "private", cond) and has_march:
            erep = TM.early_main(Dl, std[:: len(seeds)], pseudo, offis, pans, Ps, cond, prm, rounds, variants, seeds,
                                 lseeds, tcfg, fset, adapted, probs["early"])
            report["early"][cond] = erep
            C.log(f"T2 v12 early April windows {cond}: {erep}")
        del Dl, std
        for p in pans:
            if cond in sets[p]:
                sets[p][cond] = (sets[p][cond][0], sets[p][cond][1], None)
        C.free()
        report["minutes_by_step"][cond] = round((time.time() - tc) / 60, 1)
    report["april_source"] = april_src
    mode = DEC["mode"]
    report.setdefault("decoder", {})["mode_used"] = mode
    # ---- files
    d = C.WORK / "pred"
    _save_probs(d / PROBS_FILE, dict(probs=probs, mode=mode, rho=DEC["rho"], n=DEC["n"], M=DEC["M"], seed=DEC["seed"],
                                     Ps={p: dict(L=v["L"], links=list(v["links"])) for p, v in Ps.items()}))
    files = write_files(probs, offis, pans, Ps, mode)
    rows = {fn: f.set_index(KEY).queue_pred for fn, f in files.items()}
    for fn, r in rows.items():
        TM.check_rows(r, f"the v12 file {fn}")
    ids = X._ids(offis, pans)
    a = rows[MAIN_FILE]
    mar = a.index.get_level_values(0).str.contains("_validation_")
    for m_, h_ in ((MAIN_FILE, HEDGE_FILE), (MAIN_TOPM, HEDGE_TOPM)):
        if m_ in rows and not (rows[m_][mar] == rows[h_].reindex(rows[m_].index)[mar]).all():
            raise RuntimeError(f"T2 v12: the March rows of {h_} must be those of {m_}")
    if MAIN_TOPM in rows:
        tp = rows[MAIN_TOPM].reindex(a.index)
        on_all = a.index.get_level_values(0).isin(list(ids["queue_onset"]["validation"] | ids["queue_onset"]["private"]))
        if not (a[~on_all] == tp[~on_all]).all():
            raise RuntimeError("T2 v12: the decoder may change the onset rows only")
    for fn, f in files.items():
        X._write(f, d, fn)
    # ---- what changed: against the v11 main and hedge files
    checks = {}
    for fn, nm, new in ((V11_MAIN, "v11 main", a), (V11_HEDGE, "v11 hedge", rows[HEDGE_FILE])):
        f = d / fn
        if f.exists():
            old = pd.read_csv(f, dtype=QDT).set_index(KEY).queue_pred
            for c in T.CONDS:
                for s, m in (("validation", "March"), ("private", "April")):
                    checks[f"{m} {c} rows ({nm.split()[1]}) vs the {nm} file"] = TM._compare(new, old, ids[c][s])
            if MAIN_TOPM in rows and nm == "v11 main":
                for s, m in (("validation", "March"), ("private", "April")):
                    checks[f"{m} queue_onset rows (main_topm) vs the v11 main file"] = TM._compare(rows[MAIN_TOPM], old, ids["queue_onset"][s])
    report["checks"] = checks
    report["onset_cells"] = {fn: {m: onset_cells(r, ids["queue_onset"][s]) for s, m in (("validation", "March"), ("private", "April"))}
                             for fn, r in rows.items()}
    v11f = d / V11_MAIN
    if v11f.exists():
        old = pd.read_csv(v11f, dtype=QDT).set_index(KEY).queue_pred
        report["onset_cells"]["v11 main"] = {m: onset_cells(old, ids["queue_onset"][s]) for s, m in (("validation", "March"), ("private", "April"))}
    report["agreement_main_hedge"] = {m: dict(cells_main=int(a[sel].sum()), cells_hedge=int(rows[HEDGE_FILE].reindex(a.index)[sel].sum()),
                                              differ=int((a[sel] != rows[HEDGE_FILE].reindex(a.index)[sel]).sum()))
                                      for m, sel in (("March", mar), ("April", ~mar))}
    report["device"] = dict(params=dict(device or {}), failed=TM.DEVICE["failed"], fits_gpu=TM.DEVICE["fits_gpu"],
                            fits_cpu=TM.DEVICE["fits_cpu"])
    report["rows"] = int(len(a)); report["share_pred"] = float(a.mean())
    report["minutes"] = round((time.time() - t0) / 60, 1)
    report["files_md5"] = {fn: T._md5(d / fn) for fn in files}
    report["probs_md5"] = T._md5(d / PROBS_FILE)
    tmp = d / (REPORT_FILE + ".part")
    json.dump(report, open(tmp, "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    tmp.replace(d / REPORT_FILE)   # written last: it vouches for the files
    DEC["mode"] = "topm"
    C.log(f"T2 v12 done in {report['minutes']} min: chosen {report['chosen']} | onset decoder {mode} | April {april_src} | "
          f"checks {json.dumps(checks)[:600]}")
    return report
