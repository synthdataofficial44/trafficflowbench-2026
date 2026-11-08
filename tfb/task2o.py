"""v14 Task 2 diagnostic (report only, nothing is written but the report): the location-free weighting of the *ongoing*
models on the March draw - the ongoing counterpart of v13's nl75.

The ongoing recipe (v2+m, variants all + noloc, three seeds) averages the variant with the link-identifying columns and
the variant without them 50/50 since v5. For the onset, moving that weight to 0.75 gained +0.022 IoU on the March
pseudo windows (a demand draw the models never saw) while losing on the training-draw CV; the ongoing condition was never
checked. Here the v12 standard ongoing models are reloaded from their checkpoints (they must reproduce v12's stored
ongoing probabilities of the official windows, else the step stops), the March pseudo windows are scored with the weight
of the location-free variant at 0, 0.25, 0.5 (= v12), 0.75 and 1.0, and each is compared with 0.5 on the same windows
(day-block bootstrap SEs, official-style and all windows, first-week windows, corridor families, weekday / weekend).
v15 decides from this report whether an ongoing weighting goes into the files; nothing here changes any file.
Only data timestamped <= T feeds a window at origin T; no April window is evaluated.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from . import common as C
from . import task2 as T
from . import task2m as TM
from . import task2y as Y
from . import task2z as Z

VERSION = "v14o"
REPORT_FILE = "task2o_report.json"
COND = "queue_ongoing"
WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)


def _fields(models, pseudo, pans, Ps, split="validation") -> dict:
    """{panel: [n windows, K, L]} of `models` on the ongoing pseudo windows of a month (candidate cells; others 0)."""
    out = {}
    for p in pans:
        ent = pseudo.get(p, {}).get(split, {}).get(COND)
        if ent is None:
            continue
        Wc, _, _, Xp = ent
        out[p] = T._field(COND, Xp, T._predict(models, Xp), len(Wc), Ps[p]["L"])
    return out


def _rows(F: dict, pseudo, pans, Ps, split="validation") -> pd.DataFrame:
    rows = []
    for p in pans:
        ent = pseudo.get(p, {}).get(split, {}).get(COND)
        if ent is None or p not in F:
            continue
        Wc, y, kn, _ = ent
        L = Ps[p]["L"]
        for w in range(len(Wc)):
            pred = T.decode_window(COND, F[p][w], L)
            u = int(((pred | y[w]) & kn[w]).sum()); it = int(((pred & y[w]) & kn[w]).sum())
            rows.append(dict(panel=p, condition=COND, split=split, w=int(w), src=Wc.src.iat[w], day=int(Wc.day.iat[w]),
                             draws=int(Wc.draws.iat[w]), iou=1.0 if u == 0 else it / u, n_pred=int((pred & kn[w]).sum()),
                             n_true=int((y[w] & kn[w]).sum()), inter=it))
    return pd.DataFrame(rows, columns=["panel", "condition", "split", "w", "src", "day", "draws", "iou", "n_pred",
                                       "n_true", "inter"])


def _compare(Rb, Rc, early_day) -> dict:
    out, M = T.paired(Rb, Rc, early_day)
    gain = 0.5 * (np.nan_to_num(out["d_sim"]) + out["d_all"])
    fam = C.families()
    pmd = M.groupby("panel").d.mean()
    fd = pmd.groupby(pmd.index.map(fam)).mean()
    wd = T._weekday(M.day.to_numpy())
    out.update(gain=float(gain), se_gain=float(T._boot_gain(M)), fam_d={str(k): round(float(v), 4) for k, v in fd.items()},
               d_weekday=float(M[wd].d.mean()) if wd.any() else None,
               d_weekend=float(M[~wd].d.mean()) if (~wd).any() else None,
               n_pred=float(Rc.n_pred.mean()), n_true=float(Rc.n_true.mean()))
    return {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in out.items()}


def run(cfg: dict, device: dict | None = None, work=None) -> dict:
    """cfg: the v12 run's Task 2 settings. Writes WORK/pred/task2o_report.json and returns it."""
    from .data import fingerprint
    Z.install()
    t0 = time.time()
    pk, rep12 = Z.load_v12(work)
    probs12 = pk["probs"]
    if Z._norm(rep12.get("cfg")) != Z._norm(cfg):
        raise RuntimeError("task2o must run with the v12 run's Task 2 settings (CFG['T2'])")
    Y.DEC.update(mode=pk.get("mode", "topm"), rho=pk.get("rho", Y.DEC["rho"]), n=pk.get("n", Y.DEC["n"]),
                 M=pk.get("M", Y.DEC["M"]), seed=pk.get("seed", Y.DEC["seed"]))
    try:
        tcfg = Y.task_cfg(cfg)
        TM.use_device(device)
        T.set_checkpoints(tcfg)
        bcfg = dict(tcfg, late_adapt=False, late_early={}, pseudo_april=False, online_adapt=False)
        pans = C.task2_panels()
        rng = np.random.default_rng(tcfg.get("seed", 0))
        imputer = T.fit_imputer(pans) if tcfg.get("imputer", True) else None
        sets, offis, Ps, pseudo = {}, {}, {}, {}
        for i, p in enumerate(pans):
            P, out, offi, ps = T.build_panel_sets(p, imputer, rng, bcfg, skip=("queue_onset",),
                                                  skip_april=("queue_onset", COND))
            for split in ps:
                for cond in ps[split]:
                    ps[split][cond][3]["panel_code"] = np.float32(i)
            sets[p] = out; offis[p] = offi; pseudo[p] = ps
            Ps[p] = dict(L=P.L)
            P.dprof = None
            del P
            C.free()
        rep = dict(version=VERSION, fingerprint=fingerprint(), cfg=cfg, weights=list(WEIGHTS), minutes={}, checks={})
        rep["minutes"]["tables"] = round((time.time() - t0) / 60, 1)
        Dl = T._frames(sets, pans, COND)
        fset, vnames = rep12["chosen"][COND][0], list(rep12["chosen"][COND][1])
        if sorted(vnames) != ["all", "noloc"]:
            raise RuntimeError(f"task2o expects the ongoing variants all + noloc (v12 chose {vnames})")
        variants = T._variants(vnames)
        seeds = list(tcfg.get("ongoing_seeds", [tcfg.get("seed", 0)]))
        rounds = int(tcfg.get("ongoing_rounds", 700))
        rep["recipe"] = [fset, vnames, seeds, rounds]
        n0 = TM.DEVICE["fits_cpu"] + TM.DEVICE["fits_gpu"]
        std = T.fit_set(Dl, COND, T.ONG_P, rounds, variants, seeds, fset=fset)
        if TM.DEVICE["fits_cpu"] + TM.DEVICE["fits_gpu"] > n0:
            raise RuntimeError("task2o: v12's standard ongoing models were not all found in t2_models (some were refitted)")
        by_var = {v: T._models_for(std, variants, seeds, [v]) for v in vnames}
        # reload check: the official ongoing windows of both months, all seeds and variants, against v12's stored fields
        mine = {}
        T.predict_official(std, offis, pans, Ps, COND, mine, C.TEST_SPLITS)
        md = 0.0
        n = 0
        for p, d in probs12["std"].items():
            for s, ws in d.items():
                for wid, (c, pr) in ws.items():
                    if c != COND:
                        continue
                    other = mine.get(p, {}).get(s, {}).get(wid)
                    if other is None:
                        md = float("inf")
                        continue
                    md = max(md, float(np.max(np.abs(np.asarray(pr, np.float64) - np.asarray(other[1], np.float64)))))
                    n += 1
        rep["checks"]["standard ongoing fields vs v12 (max abs diff)"] = md
        rep["checks"]["windows compared"] = n
        if not md <= float(cfg.get("reload_tol_o", 0.02)):
            raise RuntimeError(f"task2o: the reloaded ongoing models do not reproduce v12's probabilities (max diff {md})")
        rep["minutes"]["reload"] = round((time.time() - t0) / 60, 1)
        fa = _fields(by_var["all"], pseudo, pans, Ps)
        fn = _fields(by_var["noloc"], pseudo, pans, Ps)
        R = {w: _rows({p: (1.0 - w) * fa[p] + w * fn[p] for p in fa}, pseudo, pans, Ps) for w in WEIGHTS}
        d0 = int(C.split_days("validation").min())
        early = d0 + int(tcfg.get("early_days", 7))
        rep["march"] = {str(w): T.pseudo_report(R[w]) for w in WEIGHTS}
        rep["vs_half"] = {str(w): _compare(R[0.5], R[w], early) for w in WEIGHTS if w != 0.5}
        rep["minutes"]["total"] = round((time.time() - t0) / 60, 1)
    finally:
        Y.DEC["mode"] = "topm"
    d = (work or C.WORK) / "pred"
    tmp = d / (REPORT_FILE + ".part")
    json.dump(rep, open(tmp, "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    tmp.replace(d / REPORT_FILE)
    C.log(f"T2 ongoing weighting (report only): " + " | ".join(
        f"w{w}: S_sim {rep['march'][str(w)].get('S_sim')}" + (
            f" d {rep['vs_half'][str(w)]['gain']:+.4f} se {rep['vs_half'][str(w)]['se_gain']:.4f}" if str(w) in rep['vs_half'] else "")
        for w in WEIGHTS))
    return rep
