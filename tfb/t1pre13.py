"""v13 Task 1: train a fourth LightGBM model set - a 3x larger transductive sample of the test months and more boosting
rounds - and score it, alone and in blends, on the test-month holdout (observed March and April cells that no member
trained on, hidden and re-predicted like a target) and on the train holdout. Nothing is applied here: pred/task1*.parquet
and t1/choice.json stay exactly as they are; v14 chooses the blend (March cells decide, as in v8) and reruns the stage-2/3
lab on top of it.

Why this set. Of the three v8 sets, the one with a 2x transductive sample (s2) had the best held-out flow and density
(reg_q RMSE 30.41 against 30.78 / 30.80, reg_k 0.524 against 0.548 / 0.552); adding it to the blend gained +0.00053
S_state on held-out March cells and +0.00063 on held-out April cells; and every regular model stopped at the 2500-round
cap still improving. Task 1 is offline, so the test months' own observed cells are legitimate training rows, and the
April holdout cells measure the private month itself (reported here, never used to choose).
"""
from __future__ import annotations

import json
import shutil
import time

import numpy as np

from . import common as C
from . import task1 as T1
from . import t1lab as L

VERSION = "v13"
REPORT = "t1pre13_report.json"
NEW_DEFAULT = dict(seed=3, n_pseudo=450_000, n_pseudo_dark=150_000, pseudo_dark_n=120, max_rounds=3500,
                   lgb_extra=dict(deterministic=True, force_row_wise=True))


def new_set(new: dict | None) -> dict:
    s = dict(NEW_DEFAULT)
    s.update(new or {})
    return s


def set_dir(cfg1: dict, new: dict):
    return C.WORK / "t1" / f"models_{cfg1.get('rows_tag', 'r1')}_s{int(new['seed'])}"


def _set_cfg(cfg1: dict, new: dict, rows_dir) -> dict:
    over = {k: v for k, v in new.items() if k != "seed"}
    return dict(cfg1, **over, tm_exclude_dir=str(rows_dir))


def train(cfg1: dict, new: dict | None = None, on_done=None) -> dict:
    """Train the new set (resumable: a complete set in WORK/t1 is kept). cfg1: the v8-v11 Task 1 settings (they fix the
    test-month holdout cells, which the new set must not train on)."""
    new = new_set(new)
    s = int(new["seed"])
    if s in T1.plan_seeds(cfg1):
        raise ValueError(f"v13 Task 1: set {s} is already one of the plan's sets {T1.plan_seeds(cfg1)}")
    tag = cfg1.get("rows_tag", "r1")
    rows_dir = L.prepare_rows(cfg1)   # row seed 0 (holdout, test) + the test-month holdout cells, as v8 built them
    scfg = _set_cfg(cfg1, new, rows_dir)
    md = set_dir(cfg1, new)
    sig = T1.rows_sig(dict(scfg, row_seed=s))
    t0 = time.time()
    if T1.set_complete(md, s, sig):
        return dict(status="exists", dir=md.name, minutes=0.0)
    rd = C.WORK / "t1" / f"rows_{tag}_rs{s}"
    infos = T1._build_rows_all(dict(scfg, row_seed=s), rd)
    C.log(f"T1 v13 set {s} rows: {sum(i['n_train'] for i in infos):,} train, {sum(i['n_pseudo'] for i in infos):,} "
          f"pseudo, {sum(i['n_dark'] for i in infos):,} dark ({(time.time() - t0) / 60:.0f} min)")
    (md / "report.json").unlink(missing_ok=True)
    T1.train_models(rd, md, dict(scfg, seed=s, row_seed=s), holdout=True)
    shutil.rmtree(rd, ignore_errors=True)
    if on_done is not None:
        on_done()
    return dict(status="trained", dir=md.name, minutes=round((time.time() - t0) / 60, 1),
                rows=dict(train=sum(i["n_train"] for i in infos), pseudo=sum(i["n_pseudo"] for i in infos),
                          dark=sum(i["n_dark"] for i in infos)))


def score(cfg1: dict, new: dict | None = None) -> dict:
    """Report only: the v11 blend's members (choice.json) against blends with the new set, on the test-month holdout
    (March decides in v14; April reported) and the train holdout (raw predictions, no post-processing)."""
    new = new_set(new)
    s = int(new["seed"])
    ch = json.load(open(C.WORK / "t1" / "choice.json"))
    members = ch.get("members") or {}
    cur = {n: float(m["weight"]) for n, m in members.items()}
    if not cur:
        raise RuntimeError("v13 Task 1 scoring: t1/choice.json lists no members")
    mem = {n: (m["kind"], C.WORK / "t1" / m["dir"]) for n, m in members.items()}
    new_name = f"lgb{s}"
    if new_name in mem:
        raise RuntimeError(f"v13 Task 1: {new_name} is already a member of the current blend")
    mem[new_name] = ("lgb", set_dir(cfg1, new))
    for n, (kind, d) in mem.items():
        if not (T1.cb_complete(d) if kind == "cb" else T1.set_complete(d)):
            raise FileNotFoundError(f"v13 Task 1: model set {d.name} is not complete")
    w1 = float(np.mean(list(cur.values())))   # the new set weighs as much as an average current member
    cands = {"base": dict(cur),
             "add_new": dict(cur, **{new_name: w1}),
             "new_x2": dict(cur, **{new_name: 2.0 * w1}),
             "new_only": {new_name: 1.0}}
    if "lgb2" in cur:   # the other set with an enlarged transductive sample
        cands["new_s2"] = {"lgb2": 1.0, new_name: 1.0}
    rows_dir = L.prepare_rows(cfg1)
    t0 = time.time()
    rep = T1.choose_blend(rows_dir, mem, cands, dict(cfg1, tm_z=float(cfg1.get("tm_z", 2.0))))
    rep.pop("chosen", None)   # nothing is chosen here: v14 decides (and reruns the lab)
    rep["members"] = {n: [k, str(d.name)] for n, (k, d) in mem.items()}
    rep["new_set"] = new
    rep["models"] = {n: json.load(open(d / "report.json")) for n, (k, d) in mem.items()
                     if k == "lgb" and (d / "report.json").exists()}
    rep["minutes"] = round((time.time() - t0) / 60, 1)
    rep["version"] = VERSION
    d = C.WORK / "t1"
    tmp = d / (REPORT + ".part")
    json.dump(rep, open(tmp, "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    tmp.replace(d / REPORT)
    T1._HOLD_PRED.clear()
    return rep


def table(rep: dict):
    """(per-candidate test-month and holdout scores, per-set early-stopping RMSE) as plain dicts for display."""
    rows = {}
    for c, d in (rep.get("candidates") or {}).items():
        r = {f"tm_{k}": v for k, v in (d.get("tm") or {}).items()}
        r.update({f"hold_{k}": v for k, v in (d.get("hold") or {}).items()})
        for k in ("tm_gain", "tm_se", "tm_gain_private", "hold_dJ", "ok"):
            if k in d:
                r[k] = d[k]
        rows[c] = r
    es = {n: {k: (v.get("rmse"), v.get("best_iter")) for k, v in m.items()} for n, m in (rep.get("models") or {}).items()}
    return rows, es
