"""v10 Task 2: the main file from the month-memory recipes (both conditions); the hedge file is the previous run's.

What decided it (v9, before any April data was looked at). The month memory (`mm_*`: where queues sat at this time of
day on the earlier days of the same month, read from the released masked view up to T) against the recipe of the v8
files, on two draws that are not the training draw:
  onset    March pseudo-holdout +0.035 IoU (4.3 SEs, all four corridor families gain), and on the scored March windows
           themselves (public probe9m, the v8 main file with only those rows changed) +0.0095 onset IoU;
  ongoing  March pseudo-holdout +0.0045 (official-style windows) / +0.0074 (all windows), public probe9mo +0.0067.
Only the training draw disagreed (week-blocked train CV -0.004 for the onset; train months share one demand draw, so a
month's own memory adds little there). The main file is the bet that April resembles March: it takes the memory. The
hedge file is the bet that April resembles train: it stays exactly the previous run's (train only, no memory), so the
pair covers the train-CV reading as well.

Main file, per condition (recipe [feature set, variants]; default onset v2+pm all+noloc, ongoing v2+m all+noloc):
  March rows   train-only models of the recipe - the models of the v9 probe files (reloaded from their checkpoints
               when the Task 2 code and settings are v9's), so the March rows repeat probe9m / probe9mo;
  April rows   the same models refit on train + all March pseudo windows when the April gate passes; the late windows
               (from day 7) and the early ongoing windows (2-6 April) refit with April's own earlier pseudo windows when
               their gates pass. Every gate follows the v5-v9 rules on March, but tests the main recipe itself
               (v5-v9 tested the late / early refits with the hedge's recipe).
Only data timestamped <= T feeds a window at origin T; no April label is evaluated or used to choose anything.

The LightGBM device (CPU / GPU) is set by `use_device` without touching task2.py, so the v9 checkpoints stay valid.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import common as C
from . import task2 as T

VERSION = "v10"
DEFAULT_RECIPES = {"queue_onset": ["v2+pm", ["all", "noloc"]], "queue_ongoing": ["v2+m", ["all", "noloc"]]}
MAIN_FILE, HEDGE_FILE, REPORT_FILE = "task2_v10.csv", "task2_v10_hedge.csv", "task2_v10_report.json"
PROBE_OF = {"queue_onset": "task2_probe9m.csv", "queue_ongoing": "task2_probe9mo.csv"}   # v9 files to compare with
KEY = ["window_id", "timestamp", "link_id"]
QDT = {"window_id": str, "timestamp": str, "link_id": str}
DEVICE = {"params": {}, "failed": None, "fits_gpu": 0, "fits_cpu": 0}


# ----------------------------------------------------------------------------- LightGBM device
def use_device(params: dict | None):
    """Task 2's LightGBM fits take `params` on top of their own (e.g. {'device_type': 'gpu'}); {} = the CPU. A fit that
    fails on the GPU is redone on the CPU, and the rest of the run stays there."""
    orig = getattr(T.fit_lgb, "_orig", T.fit_lgb)
    DEVICE.update(params=dict(params or {}), failed=None, fits_gpu=0, fits_cpu=0)

    def fit_lgb(frames, prm, rounds, feats, weight=None, seed=0, masks=None):
        if DEVICE["params"]:
            try:
                m = orig(frames, dict(prm, **DEVICE["params"]), rounds, feats, weight, seed, masks)
                DEVICE["fits_gpu"] += 1
                return m
            except Exception as e:  # noqa: BLE001  any device problem: the CPU takes over
                DEVICE["failed"] = f"{type(e).__name__}: {str(e)[:300]}"
                DEVICE["params"] = {}
                C.log(f"T2 GPU fit failed ({DEVICE['failed']}); this and every later fit on the CPU")
                C.free()
        DEVICE["fits_cpu"] += 1
        return orig(frames, prm, rounds, feats, weight, seed, masks)

    fit_lgb._orig = orig
    T.fit_lgb = fit_lgb


# ----------------------------------------------------------------------------- settings and reuse
def recipes(cfg) -> dict:
    r = dict(DEFAULT_RECIPES)
    r.update({c: list(v) for c, v in (cfg.get("recipes") or {}).items()})
    for c, (f, vn) in r.items():
        if c not in T.CONDS:
            raise ValueError(f"recipes: unknown condition {c!r}")
        T._check_fset(f)
        [T._variant_name(v) for v in vn]
        if not vn:
            raise ValueError(f"recipes[{c}]: no variants")
    return r


def task_cfg(cfg) -> dict:
    """The settings the Task 2 building blocks read: the recipes as the only candidates (their feature groups decide
    which columns the tables carry), no probes."""
    r = recipes(cfg)
    return dict(cfg, configs={c: [r[c]] for c in T.CONDS}, probe_onset=None, probe_ongoing=None,
                experiments=False, online_adapt=False, pseudo_april=False)


def code_hash() -> str:
    """Task 2's code hash + this file's: a stored v10 result is reused only for the same code."""
    h = hashlib.md5((T.code_hash() + "|").encode())
    h.update(Path(__file__).read_bytes())
    return h.hexdigest()[:16]


def _norm(d) -> str:
    return json.dumps({k: v for k, v in (d or {}).items() if k not in ("model_dir",)}, sort_keys=True, default=str)


def previous_hedge(work=None) -> dict:
    """The previous run's hedge file (WORK/pred/task2_hedge.csv), checked against the report that wrote it."""
    d = (work or C.WORK) / "pred"
    rp, fh = d / "task2_report.json", d / "task2_hedge.csv"
    if not (rp.exists() and fh.exists()):
        raise FileNotFoundError("the previous Task 2 files (pred/task2_hedge.csv + pred/task2_report.json) are needed")
    rep = json.load(open(rp))
    h = T._md5(fh)
    want = (rep.get("files_md5") or {}).get("task2_hedge.csv")
    if want is not None and want != h:
        raise RuntimeError("pred/task2_hedge.csv is not the file its report describes (an interrupted copy?)")
    return dict(path=fh, md5=h, version=rep.get("version"), code=rep.get("code"), report=rep)


def _template_keys() -> pd.MultiIndex:
    """(window_id, timestamp, link_id) of every queue row of the test-split templates."""
    parts = []
    for p in C.task2_panels():
        for split in C.TEST_SPLITS:
            f = C.rel() / "task2" / p / split / "sample_submission_queue.csv"
            if f.exists():
                parts.append(pd.read_csv(f, dtype=str, usecols=KEY))
    return pd.MultiIndex.from_frame(pd.concat(parts, ignore_index=True)[KEY])


def check_rows(rows: pd.Series, what: str):
    """Every template queue row once, with a 0/1 value."""
    tk = _template_keys()
    if rows.index.has_duplicates or len(rows) != len(tk) or not rows.index.isin(tk).all():
        raise RuntimeError(f"T2 v10: {what} does not hold exactly the template's queue rows")
    if not set(np.unique(rows.to_numpy())) <= {0, 1}:
        raise RuntimeError(f"T2 v10: {what} has queue_pred values outside {{0, 1}}")


def up_to_date(cfg, work=None) -> tuple:
    """(True, report) when the v10 files of this code, these settings, this release and this hedge source are on disk."""
    d = (work or C.WORK) / "pred"
    rp = d / REPORT_FILE
    if not rp.exists():
        return False, None
    try:
        from .data import fingerprint
        rep = json.load(open(rp))
        prev = previous_hedge(work)
        fp = fingerprint()
    except Exception:  # noqa: BLE001
        return False, None
    ok = (rep.get("version") == VERSION and rep.get("code") == code_hash() and _norm(rep.get("cfg")) == _norm(cfg)
          and rep.get("fingerprint") == fp
          and rep.get("hedge_source", {}).get("md5") == prev["md5"]
          and all((d / fn).exists() and T._md5(d / fn) == h for fn, h in (rep.get("files_md5") or {}).items())
          and set(rep.get("files_md5") or {}) >= {MAIN_FILE, HEDGE_FILE})
    return ok, rep


# ----------------------------------------------------------------------------- the main file
def early_main(Dl, std1, pseudo, offis, pans, Ps, cond, prm, rounds, variants, seeds, lseeds, cfg, fset, adapted,
               early_m):
    """v8's early blocks for the main file alone: gate on March with the main recipe (train vs train + the March pseudo
    windows before each block, scored on the block), then models for the official April windows of each block
    (train + April's pseudo windows before the block, + all of March when the April adaptation was adopted)."""
    vb, pb = T.early_blocks(cfg, "validation", cond), T.early_blocks(cfg, "private", cond)
    a0 = int(C.split_days("private").min())
    n_off = sum(T.n_official(offis, pans, cond, "private", lo, hi) for lo, hi in pb)
    rep = dict(blocks=[lo - a0 for lo, _ in pb], block_ends=[hi - a0 for _, hi in pb], official=n_off, adopted=False)
    if n_off == 0:
        rep["reason"] = "no official April window of this condition in the early blocks"
        return rep
    g = T.late_gate(Dl, std1, pseudo, pans, Ps, cond, prm, rounds, variants,
                    dict(cfg, late_force=cfg.get("early_force", False)), fset, seed=seeds[0], blocks=vb, role="egate")
    rep.update({k: v for k, v in g.items() if k != "blocks"}, gate_blocks=g.get("blocks"))
    if rep["adopted"]:
        rep["main"] = T.late_fit_predict(Dl, pseudo, offis, pans, Ps, cond, prm, rounds, variants, lseeds, cfg, fset,
                                         bool(adapted), early_m, "main", blocks=pb, role="early")
    return rep


def _compare(new: pd.Series, old: pd.Series, windows) -> dict:
    """Cells / windows of `windows` where two row sets differ."""
    sel = new.index.get_level_values(0).isin(list(windows))
    a = new[sel]
    b = old.reindex(a.index)
    dif = (a != b) | b.isna()
    return dict(cells=int(len(a)), cells_differ=int(dif.sum()), windows=int(a.index.get_level_values(0).nunique()),
                windows_differ=int(dif.groupby(level=0).any().sum()))


def run_main(cfg: dict, device: dict | None = None) -> dict:
    """Writes pred/task2_v10.csv (main), pred/task2_v10_hedge.csv (the previous hedge file, unchanged) and
    pred/task2_v10_report.json (written last). `device`: extra LightGBM training parameters (gpu.choose()['params'];
    None / {} = the CPU). Returns the report."""
    from .data import fingerprint
    t0 = time.time()
    tcfg = task_cfg(cfg)
    rec = recipes(cfg)
    prev = previous_hedge()
    hb = pd.read_csv(prev["path"], dtype=QDT).set_index(KEY).queue_pred
    check_rows(hb, "the previous hedge file")   # before hours of fitting
    use_device(device)
    T.set_checkpoints(tcfg)
    pans = C.task2_panels()
    rng = np.random.default_rng(tcfg.get("seed", 0))
    imputer = T.fit_imputer(pans) if tcfg.get("imputer", True) else None
    C.log(f"T2 v10 label imputer ready ({time.time() - t0:.0f}s)")
    sets, offis, Ps, pseudo = {}, {}, {}, {}
    for i, p in enumerate(pans):
        P, out, offi, ps = T.build_panel_sets(p, imputer, rng, tcfg)
        for split in ps:
            for cond in ps[split]:
                ps[split][cond][3]["panel_code"] = np.float32(i)
        sets[p] = out; offis[p] = offi; pseudo[p] = ps
        Ps[p] = dict(L=P.L, links=P.net.links, mp=np.asarray(P.mp, np.float64))
        npw = ", ".join(f"{s[:3]} {c.split('_')[1]} {len(v[0])}" for s in ps for c, v in ps[s].items())
        C.log(f"T2 v10 {p}: train windows onset {len(out.get('queue_onset', ([],))[0])}, "
              f"ongoing {len(out.get('queue_ongoing', ([],))[0])}; pseudo windows: {npw or 'none'}")
        P.dprof = None
        del P
        C.free()
    report = dict(version=VERSION, code=code_hash(), task2_code=T.code_hash(), fingerprint=fingerprint(), cfg=cfg, recipes=rec,
                  hedge_source=dict(md5=prev["md5"], version=prev["version"], code=prev["code"]),
                  pseudo={}, diag={}, adapt={}, late={}, early={}, minutes_by_step={},
                  windows=T.official_window_calendar(offis, pans))
    probs_m, apr_m, late_m, early_m = {}, {}, {}, {}
    for cond in T.CONDS:
        tc = time.time()
        Dl = T._frames(sets, pans, cond)
        if not Dl:
            raise RuntimeError(f"T2 v10: no training windows for {cond}")
        fset, vn = rec[cond]
        variants = T._variants(vn)
        on = cond == "queue_onset"
        prm, rounds = (T.ONSET_P, tcfg.get("onset_rounds", 500)) if on else (T.ONG_P, tcfg.get("ongoing_rounds", 700))
        seeds = list(tcfg.get("onset_seeds" if on else "ongoing_seeds", [tcfg.get("seed", 0)]))
        lseeds = list(tcfg.get("late_seeds", seeds[:1]))
        has_march = any(cond in pseudo[p].get("validation", {}) for p in pans)
        C.log(f"T2 v10 {cond}: {sum(len(f) for f in Dl):,} training rows, recipe {fset} {vn}")
        # ---- standard models (train only): the March rows, and the April rows unless a refit replaces them
        std = T.fit_set(Dl, cond, prm, rounds, variants, seeds, fset=fset)
        T.predict_official(std, offis, pans, Ps, cond, probs_m, ("validation", "private"))
        if has_march:
            R_ = T.pseudo_scores(std, pseudo, pans, Ps, "validation", cond)
            report["pseudo"][cond] = dict(main=T.pseudo_report(R_))
            report["diag"][cond] = T.window_diag(R_, cond)
            C.log(f"T2 v10 pseudo-holdout {cond}: {json.dumps(report['pseudo'][cond], default=str)[:500]}")
        # ---- April adaptation: train + all March pseudo windows (gate: first half of March -> second half)
        adapted = False
        if tcfg.get("april_adapt", True) and has_march:
            rep_ad, ad = T.april_adapt(Dl, std, pseudo, pans, Ps, cond, prm, rounds, variants, seeds, tcfg, fset)
            report["adapt"][cond] = rep_ad
            C.log(f"T2 v10 April adaptation {cond}: {rep_ad}")
            if ad is not None:
                T.predict_official(ad, offis, pans, Ps, cond, apr_m, ("private",))
                adapted = True
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
                                                      tcfg, fset, adapted, late_m, "main")
                report["late"][cond] = lrep
                C.log(f"T2 v10 late windows {cond}: {lrep}")
        # ---- early April windows (v8 blocks): same-month evidence from the first days of April
        if T.early_blocks(tcfg, "private", cond) and has_march:
            erep = early_main(Dl, std[:: len(seeds)], pseudo, offis, pans, Ps, cond, prm, rounds, variants, seeds,
                              lseeds, tcfg, fset, adapted, early_m)
            report["early"][cond] = erep
            C.log(f"T2 v10 early April windows {cond}: {erep}")
        del Dl, std
        for p in pans:   # this condition's training tables are not needed any more
            if cond in sets[p]:
                sets[p][cond] = (sets[p][cond][0], sets[p][cond][1], None)
        C.free()
        report["minutes_by_step"][cond] = round((time.time() - tc) / 60, 1)
    # ---- files: main (refits override the standard rows: early > late > adapted > standard) and hedge (previous)
    d = C.WORK / "pred"
    main = T.write_rows([early_m, late_m, apr_m, probs_m], offis, pans, Ps)
    a = main.set_index(KEY).queue_pred
    check_rows(a, "the v10 main file")
    tmp = d / (MAIN_FILE + ".part")
    main.to_csv(tmp, index=False); tmp.replace(d / MAIN_FILE)
    shutil.copyfile(prev["path"], d / (HEDGE_FILE + ".part")); (d / (HEDGE_FILE + ".part")).replace(d / HEDGE_FILE)
    # ---- what changed: against the previous main file, and the March rows against the v9 probe files
    mar = a.index.get_level_values(0).str.contains("_validation_")
    ids = {c: {s: set() for s in C.TEST_SPLITS} for c in T.CONDS}
    for p in pans:
        for s in C.TEST_SPLITS:
            if s in offis[p]:
                wi = offis[p][s][0]
                for c in T.CONDS:
                    ids[c][s] |= set(wi.window_id[wi.condition == c])
    checks = {}
    for c, fn in PROBE_OF.items():
        f = d / fn
        if f.exists():
            old = pd.read_csv(f, dtype=QDT).set_index(KEY).queue_pred
            checks[f"March {c} rows vs {fn}"] = _compare(a, old, ids[c]["validation"])
    fm = d / "task2.csv"
    if fm.exists():
        old = pd.read_csv(fm, dtype=QDT).set_index(KEY).queue_pred
        for c in T.CONDS:
            for s, nm in (("validation", "March"), ("private", "April")):
                checks[f"{nm} {c} rows vs the previous main file"] = _compare(a, old, ids[c][s])
    report["checks"] = checks
    report["agreement_with_hedge"] = {m: dict(cells_main=int(a[sel].sum()), cells_hedge=int(hb.reindex(a.index)[sel].sum()),
                                              differ=int((a[sel] != hb.reindex(a.index)[sel]).sum()))
                                      for m, sel in (("March", mar), ("April", ~mar))}
    report["device"] = dict(params=dict(device or {}), failed=DEVICE["failed"], fits_gpu=DEVICE["fits_gpu"],
                            fits_cpu=DEVICE["fits_cpu"])
    report["rows"] = int(len(main)); report["share_pred"] = float(main.queue_pred.mean())
    report["minutes"] = round((time.time() - t0) / 60, 1)
    report["files_md5"] = {fn: T._md5(d / fn) for fn in (MAIN_FILE, HEDGE_FILE)}
    tmp = d / (REPORT_FILE + ".part")
    json.dump(report, open(tmp, "w"), indent=1, default=str)
    tmp.replace(d / REPORT_FILE)   # written last: it vouches for the files
    C.log(f"T2 v10 done in {report['minutes']} min: {len(main):,} rows, share predicted {report['share_pred']:.4f}; "
          f"checks {json.dumps(checks)[:600]}")
    return report
