"""v11 Task 1 / Task 3 lab: a stage-3 model on top of the v10 answer (B0 + the stage-2 density model).

The v10 lab showed where the conservation term S_LWR is lost (isolated targets in free flow, runs on the 50% days) and
that smoothing along time only hurts: the information is in the released cells around a target at the same time.
s3 adds what the stage-2 model did not see, and fits the loss each score takes:
  * features: each neighbour's (+-1, +-2 links) deviation from its own previous / next slot (a fluctuation that the
    vehicles carry from cell to cell shows up there at the same time), neighbours' flow and speed against our answer,
    the median flow of the released links of the same ramp-free segment at the same time (a conserved quantity in free
    flow), the on/off-ramp flows of the link, and two conservation estimates (upstream link +- this link's ramps,
    downstream link -+ its ramps);
  * heads: log speed and log flow (L2: S_state is an RMSE) and log density (L1 weighted by the link's vehicle count:
    S_LWR is an L1 norm of vehicle-count changes); the density head acts only on links the conservation check scores;
  * candidates: s3k (density head, split as B0 splits a density change), s3vq (speed and flow heads), s3all (speed and
    flow heads, then the density moved to the density head's value).
Judged on the train holdout against B1 = B0 + s2 (out of fold, cross-fitted by blocks of holdout days) with the v10 rule
(J gain > z day-bootstrap SEs, positive on both halves, S_state not down by more than tol_state); then S_state on the
held-out observed March cells must not fall (tol_tm). The adopted chain is applied to pred/task1.parquet's values ->
pred/task1_v11.parquet; the models are saved in WORK/t1lab/models11. April is never looked at.
"""
from __future__ import annotations

import hashlib
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from . import common as C
from . import t1lab as L
from . import task1 as T1

VERSION = "v11"
S3_L2 = dict(L.S2_PARAMS)
S3_L1 = dict(L.S2_PARAMS, objective="l1")
HEADS = ("k", "v", "q")
CANDS = ("s3k", "s3vq", "s3all")
LAB11_DEFAULTS = dict(z=2.0, tol_state=0.0005, tol_tm=0.0003, boot=400, s2_rounds=400, s2_folds=3, s2_max_rows=3_000_000,
                      s3_rounds=400, s3_max_rows=3_000_000, tm_check=True, force=None, candidates=list(CANDS))
CLIP = 0.7
REPRO_TOL = 2e-5     # |B1 - B0 holdout J gain - v10's| allowed (thread-count noise of the stage-2 refit)
MODEL_DIR = "models11"


# ----------------------------------------------------------------------------- features, targets, weights
def segments(P) -> np.ndarray:
    """Segment id per link: a new segment starts at every link with an attached ramp (link order = downstream)."""
    att = np.asarray(P.net.on_att, bool) | np.asarray(P.net.off_att, bool)
    return np.cumsum(att).astype(np.int64)


def _seg_median(LQ, t, j, seg):
    med = np.full(len(t), np.nan, np.float32); cnt = np.zeros(len(t), np.float32)
    for s in np.unique(seg):
        ls = np.flatnonzero(seg == s)
        if len(ls) < 2:
            continue
        sel = np.flatnonzero(np.isin(j, ls))
        if not len(sel):
            continue
        M = LQ[t[sel][:, None], ls[None, :]].astype(np.float64)
        M[j[sel][:, None] == ls[None, :]] = np.nan
        cnt[sel] = np.isfinite(M).sum(1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            med[sel] = np.nanmedian(M, 1)
    return med, cnt


def s3_features(P, fr, v, q) -> pd.DataFrame:
    """s2's features of the answer (v, q per lane) + the v11 columns (x*)."""
    X = L.s2_features(P, fr, v, q)
    t = fr.t.to_numpy(np.int64); j = fr.j.to_numpy(np.int64)
    T_, Ln = P.NT, P.L
    TG = np.zeros((T_, Ln), bool); TG[t, j] = True
    obs = P.ok & ~TG
    lanes = P.lanes.astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        V = np.where(obs, P.X["v"], np.nan).astype(np.float32)
        Qt = np.where(obs, P.X["q"] * lanes[None, :], np.nan).astype(np.float32)
        LV = np.log(np.maximum(V, 1.0)).astype(np.float32)
        LQ = np.log(np.maximum(Qt, 1.0)).astype(np.float32)
        LK = (LQ - LV).astype(np.float32)
        lq_a = np.log(np.maximum(q * lanes[j], 1.0)); lv_a = np.log(np.maximum(v, 1.0))

    def get(A, dt, dl):
        t2, l2 = t + dt, j + dl
        ok = (t2 >= 0) & (t2 < T_) & (l2 >= 0) & (l2 < Ln)
        out = np.full(len(t), np.nan, np.float32)
        out[ok] = A[t2[ok], l2[ok]]
        return out

    F = {}
    for dl in (-2, -1, 1, 2):
        for nm, A in (("k", LK), ("v", LV), ("q", LQ)):
            F[f"x{nm}{dl}_dev"] = get(A, 0, dl) - 0.5 * (get(A, -1, dl) + get(A, 1, dl))
        F[f"xq{dl}_rel"] = get(LQ, 0, dl) - lq_a
        F[f"xv{dl}_rel"] = get(LV, 0, dl) - lv_a
    F["xq_t_rel"] = 0.5 * (get(LQ, -1, 0) + get(LQ, 1, 0)) - lq_a
    F["xv_t_rel"] = 0.5 * (get(LV, -1, 0) + get(LV, 1, 0)) - lv_a
    med, cnt = _seg_median(LQ, t, j, segments(P))
    F["xseg_q"] = med - lq_a; F["xseg_n"] = cnt
    on_att = np.asarray(P.net.on_att, bool); off_att = np.asarray(P.net.off_att, bool)
    ron = np.where(on_att[None, :], P.ron * lanes[None, :], 0.0).astype(np.float32)
    roff = np.where(off_att[None, :], P.roff * lanes[None, :], 0.0).astype(np.float32)
    qa = np.exp(lq_a)
    F["xron"] = get(ron, 0, 0) / qa; F["xroff"] = get(roff, 0, 0) / qa
    with np.errstate(invalid="ignore", divide="ignore"):
        up = get(Qt, 0, -1) + get(ron, 0, 0) - get(roff, 0, 0)
        dn = get(Qt, 0, 1) - get(ron, 0, 1) + get(roff, 0, 1)
        F["xcons_up"] = np.log(np.maximum(up, 1.0)) - lq_a
        F["xcons_dn"] = np.log(np.maximum(dn, 1.0)) - lq_a
    F["xobs"] = np.asarray(P.observable, np.float32)[j]
    for k_, v_ in F.items():
        X[k_] = np.asarray(v_, np.float32)
    return X


def s3_targets(P, fr, v, q) -> dict:
    t = fr.t.to_numpy(np.int64); j = fr.j.to_numpy(np.int64)
    with np.errstate(invalid="ignore", divide="ignore"):
        yv = np.log(np.maximum(P.tv[t, j], 1.0)) - np.log(np.maximum(v, 1.0))
        yq = np.log(np.maximum(P.tq[t, j], 1.0)) - np.log(np.maximum(q, 1.0))
    yk = yq - yv
    return {h: np.clip(y, -CLIP, CLIP).astype(np.float32) for h, y in (("k", yk), ("v", yv), ("q", yq))}


def s3_weight(P, fr, v, q) -> np.ndarray:
    """Vehicle count of the cell in the answer (density per lane x lanes x length), mean 1 (the density head's L1)."""
    j = fr.j.to_numpy(np.int64)
    n = q / np.maximum(v, 1.0) * P.lanes[j] * P.length[j]
    n = np.clip(n, 0.05 * np.nanmedian(n), None)
    return (n / np.nanmean(n)).astype(np.float32)


def _fit(X, y, rounds, params, w=None, device=None, seed=0):
    import lightgbm as lgb
    ok = np.isfinite(y)
    p = dict(params, num_threads=C.THREADS, seed=seed, **(device or {}))
    ds = lgb.Dataset(X[ok].to_numpy(np.float32), y[ok], weight=None if w is None else w[ok], feature_name=list(X.columns))
    return lgb.train(p, ds, rounds)


def apply_s3(cand, P, fr, v1, q1, yh, pp, shrink=1.0):
    """A candidate on the answer (v1, q1) from head predictions yh {'k', 'v', 'q'} (log-ratio corrections)."""
    j = fr.j.to_numpy(np.int64); kind = fr.kind.to_numpy()
    obs = np.asarray(P.observable, bool)[j]
    vf = P.vf[j].astype(np.float64); lanes = P.lanes[j].astype(np.float64)
    if cand == "s3k":
        return L._apply_dl(P, j, kind, v1, q1, np.where(obs, shrink * yh["k"], 0.0), pp)
    v2 = np.clip(v1 * np.exp(shrink * np.nan_to_num(yh["v"])), 3.0, 1.1 * vf)
    q2 = np.clip(q1 * np.exp(shrink * np.nan_to_num(yh["q"])), 100.0 / lanes, None)
    if cand == "s3vq":
        return v2, q2
    if cand == "s3all":
        lf = L.LOG_FLOOR
        target = np.log(np.maximum(q1 / np.maximum(v1, 1.0), lf)) + shrink * np.nan_to_num(yh["k"])
        cur = np.log(np.maximum(q2 / np.maximum(v2, 1.0), lf))
        return L._apply_dl(P, j, kind, v2, q2, np.where(obs, target - cur, 0.0), pp)
    raise ValueError(cand)


def _base(chain, P, fr, v0, q0, pp, s2_pred):
    """The base answer: the v10 chain (s2 steps only) on B0."""
    v, q = v0, q0
    for fam, prm in chain:
        if fam != "s2":
            raise ValueError(f"t1lab11: base chain step {fam!r} not supported (only s2)")
        v, q = L.s2_apply(P, fr, v, q, s2_pred, float(prm["shrink"]), pp)
    return v, q


# ----------------------------------------------------------------------------- the lab
def lab_key11(cfg1: dict, lab: dict | None, base_chain, work=None) -> dict:
    work = work or C.WORK
    h = hashlib.md5()
    for f in ("task1.py", "physics.py", "t1lab.py", "t1lab11.py", "common.py", "data.py"):
        h.update((Path(__file__).parent / f).read_bytes())
    ch, mems = L.members(work)
    tp = work / "pred" / "task1.parquet"
    cfg1 = {k: v for k, v in cfg1.items() if k != "row_jobs"}
    return json.loads(json.dumps(dict(code=h.hexdigest()[:16], cfg1=cfg1, lab=dict(LAB11_DEFAULTS, **(lab or {})),
                                      base_chain=base_chain, members=L._members_key(mems), pp=ch.get("pp"),
                                      task1_md5=ch.get("task1_md5") or (T1.file_md5(tp) if tp.exists() else None)),
                                 sort_keys=True, default=str))


def _oof(X, y, fold, nf, rounds, params, max_rows, device=None, w=None, rng_seed=0, s2=False):
    """Out-of-fold predictions (fold k predicted by a model fitted on the other folds, seed k); the s2 path repeats
    v10's lab exactly (same subsample stream, same fit function)."""
    oof = np.full(len(X), np.nan, np.float32)
    rng = np.random.default_rng(rng_seed)
    imp = None
    for k in range(nf):
        tr = np.flatnonzero(fold != k)
        if len(tr) > int(max_rows):
            tr = np.sort(rng.choice(tr, int(max_rows), replace=False))
        if s2:
            m = L.s2_fit(X.iloc[tr], y[tr], int(rounds), device, seed=k)
        else:
            m = _fit(X.iloc[tr], y[tr], int(rounds), params, None if w is None else w[tr], device, seed=k)
        te = fold == k
        if te.any():
            oof[te] = m.predict(X[te].to_numpy(np.float32), num_threads=C.THREADS)
        g = pd.Series(m.feature_importance("gain"), index=m.feature_name())
        imp = g if imp is None else imp + g
        del m
        C.free()
    top = {k: round(float(v), 1) for k, v in (imp / max(imp.sum(), 1e-12) * 100).sort_values(ascending=False).head(12).items()} if imp is not None else {}
    return oof, top


def _rmse_table(book, name) -> dict:
    """Pooled RMSE of speed and per-lane flow per regime (all holdout panels)."""
    tot = np.zeros((3, 3))
    for sp, _ in book.parts[name].values():
        tot += sp.sum(0)
    return {f"R{r + 1}": dict(rmse_speed=float(np.sqrt(tot[r, 0] / max(tot[r, 2], 1))),
                              rmse_flow_lane=float(np.sqrt(tot[r, 1] / max(tot[r, 2], 1))), cells=int(tot[r, 2]))
            for r in range(3)}


def run_lab11(cfg1: dict, lab: dict | None = None, base_chain=(), device: dict | None = None,
              v10_dj: float | None = None) -> dict:
    """The v11 lab on the train holdout. Writes WORK/t1lab/lab11_report.json (and, when a candidate passes, the models
    in WORK/t1lab/models11) and returns the report. `v10_dj`: the holdout J gain v10 measured for the base chain; B1
    must reproduce it (within REPRO_TOL) for a candidate to be adopted (the v11 change then sits on exactly the v10
    answer)."""
    lab = dict(LAB11_DEFAULTS, **(lab or {}))
    base_chain = [(f, dict(p)) for f, p in base_chain]
    t0 = time.time()
    ch, mems = L.members()
    pp = dict(ch["pp"])
    rows_dir = L.prepare_rows(cfg1)
    panels = [p for p in C.panels() if (rows_dir / f"{p}_hold.parquet").exists()]
    infos = {p: json.load(open(rows_dir / f"{p}_info.json")) for p in panels}
    book = L.Book({p: infos[p]["hold_days"] for p in panels}, boot=int(lab["boot"]))
    has_s2 = any(f == "s2" for f, _ in base_chain)
    rep = dict(version=VERSION, key=lab_key11(cfg1, lab, [[f, p] for f, p in base_chain]), base_chain=[[f, p] for f, p in base_chain],
               pp=pp, lab=lab, variants={}, minutes={})
    # ---- pass 1: B0 and the stage-2 rows (the v10 lab's, on B0)
    X2, y2, M = [], [], []
    for p in panels:
        P, info, fr = L.hold_ctx(p, rows_dir, mems)
        t, j = fr.t.to_numpy(np.int64), fr.j.to_numpy(np.int64)
        v0, q0 = T1.postprocess_panel(P, fr, pp)
        book.add("B0", p, P, t, j, v0, q0)
        if has_s2:
            X2.append(L.s2_features(P, fr, v0, q0)); y2.append(L.s2_target(P, fr, v0, q0))
        M.append(pd.DataFrame(dict(panel=p, day=(t // C.SLOTS).astype(np.int64))))
        del P, fr
        C.free()
    M = pd.concat(M, ignore_index=True)
    days = book.alldays
    nf = int(lab["s2_folds"])
    fold_of_day = {int(d): min(nf - 1, i * nf // len(days)) for i, d in enumerate(days)}
    fold = M.day.map(fold_of_day).to_numpy()
    pidx = {p: np.flatnonzero((M.panel == p).to_numpy()) for p in panels}
    s2_oof = None
    if has_s2:
        X2 = pd.concat(X2, ignore_index=True); y2 = np.concatenate(y2)
        s2_oof, top2 = _oof(X2, y2, fold, nf, lab["s2_rounds"], None, lab["s2_max_rows"], device, s2=True)
        ok = np.isfinite(y2)
        rep["s2"] = dict(rows=int(len(X2)), rmse_before=float(np.sqrt(np.mean(y2[ok] ** 2))),
                         rmse_oof=float(np.sqrt(np.mean((y2[ok] - s2_oof[ok]) ** 2))), top_features=top2)
    rep["minutes"]["pass1"] = round((time.time() - t0) / 60, 1)
    # ---- pass 2: the base answer B1 and the stage-3 rows
    X3, Y3, W3 = [], {h: [] for h in HEADS}, []
    for p in panels:
        P, info, fr = L.hold_ctx(p, rows_dir, mems)
        t, j = fr.t.to_numpy(np.int64), fr.j.to_numpy(np.int64)
        v0, q0 = T1.postprocess_panel(P, fr, pp)
        v1, q1 = _base(base_chain, P, fr, v0, q0, pp, None if s2_oof is None else s2_oof[pidx[p]])
        book.add("B1", p, P, t, j, v1, q1)
        X3.append(s3_features(P, fr, v1, q1))
        for h, y in s3_targets(P, fr, v1, q1).items():
            Y3[h].append(y)
        W3.append(s3_weight(P, fr, v1, q1))
        del P, fr
        C.free()
    X3 = pd.concat(X3, ignore_index=True); Y3 = {h: np.concatenate(v) for h, v in Y3.items()}; W3 = np.concatenate(W3)
    rep["minutes"]["pass2"] = round((time.time() - t0) / 60, 1)
    oof3, rep["s3"] = {}, dict(rows=int(len(X3)), features=int(X3.shape[1]), heads={})
    for h in HEADS:
        prm = S3_L1 if h == "k" else S3_L2
        oof3[h], top = _oof(X3, Y3[h], fold, nf, lab["s3_rounds"], prm, lab["s3_max_rows"], device,
                            w=W3 if h == "k" else None, rng_seed=1)
        y = Y3[h]; ok = np.isfinite(y)
        rep["s3"]["heads"][h] = dict(objective=prm["objective"], rmse_before=float(np.sqrt(np.mean(y[ok] ** 2))),
                                     rmse_oof=float(np.sqrt(np.mean((y[ok] - oof3[h][ok]) ** 2))),
                                     mae_before=float(np.mean(np.abs(y[ok]))), mae_oof=float(np.mean(np.abs(y[ok] - oof3[h][ok]))),
                                     top_features=top)
    rep["minutes"]["s3_fit"] = round((time.time() - t0) / 60, 1)
    # ---- pass 3: the candidates (out-of-fold heads)
    for p in panels:
        P, info, fr = L.hold_ctx(p, rows_dir, mems)
        t, j = fr.t.to_numpy(np.int64), fr.j.to_numpy(np.int64)
        v0, q0 = T1.postprocess_panel(P, fr, pp)
        v1, q1 = _base(base_chain, P, fr, v0, q0, pp, None if s2_oof is None else s2_oof[pidx[p]])
        yh = {h: oof3[h][pidx[p]] for h in HEADS}
        for cand in lab["candidates"]:
            v, q = apply_s3(cand, P, fr, v1, q1, yh, pp)
            book.add(cand, p, P, t, j, v, q)
        del P, fr
        C.free()
    rep["minutes"]["pass3"] = round((time.time() - t0) / 60, 1)
    rep["B0"] = book.mean("B0"); rep["B1"] = book.mean("B1")
    rep["base_vs_B0"] = book.diff("B0", "B1")
    rep["v10_dj"] = v10_dj
    rep["reproduces_v10"] = (None if v10_dj is None or not base_chain
                             else bool(abs(rep["base_vs_B0"]["d_J"] - float(v10_dj)) <= REPRO_TOL))
    rep["rmse_B1"] = _rmse_table(book, "B1")
    for cand in lab["candidates"]:
        rep["variants"][cand] = book.diff("B1", cand)
    forced = lab.get("force")
    best = forced if forced in rep["variants"] else max(rep["variants"], key=lambda n: rep["variants"][n]["d_J"])
    d = rep["variants"][best]
    rep["candidate"] = best
    rep["passes_holdout"] = bool(L._passes(d, lab))
    rep["adopted"] = bool(forced == best or (rep["passes_holdout"] and rep["reproduces_v10"] is not False))
    rep["table"] = pd.DataFrame({n: {k: v for k, v in dd.items() if k in ("d_J", "se_J", "d_S_state", "d_S_LWR",
                                                                             "dJ_first_half", "dJ_second_half")}
                                 for n, dd in rep["variants"].items()}).T.sort_values("d_J", ascending=False).round(6).to_dict("index")
    # ---- the models for the test months (only when adopted): s2 as v10 fitted it, the s3 heads on every holdout row
    if rep["adopted"]:
        models = {}
        if has_s2:
            X, y = X2, y2
            if len(X) > int(lab["s2_max_rows"]):
                idx = np.sort(np.random.default_rng(0).choice(len(X), int(lab["s2_max_rows"]), replace=False))
                X, y = X.iloc[idx], y[idx]
            models["s2"] = L.s2_fit(X, y, int(lab["s2_rounds"]), device, seed=0)
        idx = np.arange(len(X3))
        if len(idx) > int(lab["s3_max_rows"]):
            idx = np.sort(np.random.default_rng(2).choice(len(X3), int(lab["s3_max_rows"]), replace=False))
        for h in HEADS:
            prm = S3_L1 if h == "k" else S3_L2
            models[f"s3_{h}"] = _fit(X3.iloc[idx], Y3[h][idx], int(lab["s3_rounds"]), prm, W3[idx] if h == "k" else None,
                                     device, seed=0)
        save_models(models, rep)
    rep["minutes"]["total_lab"] = round((time.time() - t0) / 60, 1)
    C.log(f"T1 lab11: B1 {rep['B1']} (vs B0 dJ {rep['base_vs_B0']['d_J']:+.6f}) | candidate {best} "
          f"{ {k: round(d[k], 6) for k in ('d_J', 'se_J', 'd_S_state', 'd_S_LWR')} } | adopted (holdout) {rep['adopted']}")
    _save(rep)
    return rep


def _save(rep):
    d = C.WORK / "t1lab"
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / "lab11_report.json.part"
    json.dump(rep, open(tmp, "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    tmp.replace(d / "lab11_report.json")


def save_models(models: dict, rep: dict):
    d = C.WORK / "t1lab" / MODEL_DIR
    d.mkdir(parents=True, exist_ok=True)
    md5 = {}
    for name, m in models.items():
        f = d / f"{name}.txt"
        m.save_model(str(f.with_suffix(".part")))
        f.with_suffix(".part").replace(f)
        md5[name] = hashlib.md5(f.read_bytes()).hexdigest()
    tmp = d / "models.json.part"
    json.dump(dict(key=rep["key"], candidate=rep["candidate"], md5=md5), open(tmp, "w"), indent=1, default=str)
    tmp.replace(d / "models.json")   # written last: it vouches for the model files


def load_models(rep: dict) -> dict | None:
    import lightgbm as lgb
    d = C.WORK / "t1lab" / MODEL_DIR
    try:
        meta = json.load(open(d / "models.json"))
    except Exception:  # noqa: BLE001
        return None
    if meta.get("key") != rep.get("key"):
        return None
    out = {}
    for name, h in meta.get("md5", {}).items():
        raw = (d / f"{name}.txt").read_bytes() if (d / f"{name}.txt").exists() else b""
        if hashlib.md5(raw).hexdigest() != h:
            return None
        out[name] = lgb.Booster(model_str=raw.decode())
    return out


# ----------------------------------------------------------------------------- the test months
def _chain_predict(models, rep, P, fr, v0, q0, pp):
    """B1 (the base chain with the full s2 model) and the adopted candidate's answer on rows fr (B0 answers v0, q0)."""
    base_chain = [(f, p) for f, p in rep["base_chain"]]
    s2p = None
    if any(f == "s2" for f, _ in base_chain):
        s2p = models["s2"].predict(L.s2_features(P, fr, v0, q0).to_numpy(np.float32), num_threads=C.THREADS)
    v1, q1 = _base(base_chain, P, fr, v0, q0, pp, s2p)
    X = s3_features(P, fr, v1, q1).to_numpy(np.float32)
    yh = {h: models[f"s3_{h}"].predict(X, num_threads=C.THREADS) for h in HEADS}
    v2, q2 = apply_s3(rep["candidate"], P, fr, v1, q1, yh, pp)
    return (v1, q1), (v2, q2)


def tm_check11(rep: dict, cfg1: dict, lab: dict | None = None) -> dict:
    """S_state of the candidate against B1 on the held-out observed March cells (as v10's tm_check: each cell hidden
    and answered as an isolated target among the real targets). Needs prepare_rows' test-month files."""
    lab = dict(LAB11_DEFAULTS, **(lab or {}))
    if not lab.get("tm_check", True):
        return dict(n=0, ok=True, reason="tm_check switched off")
    models = load_models(rep)
    if models is None:
        raise RuntimeError("t1lab11: the lab11 models are missing or belong to another lab result - rerun the lab")
    ch, mems = L.members()
    pp = dict(ch["pp"])
    rows_dir = C.WORK / "t1" / f"rows_{cfg1.get('rows_tag', 'r1')}"
    tm_files = {p: rows_dir / f"{p}_tm.parquet" for p in C.panels()}
    if not any(f.exists() for f in tm_files.values()):
        raise RuntimeError(f"t1lab11: no test-month holdout files in {rows_dir} - run prepare_rows first")
    t1 = pd.read_parquet(C.WORK / "pred" / "task1.parquet")
    recs = []
    for p in C.panels():
        f = tm_files[p]
        if not f.exists():
            continue
        tm = L._cached_pred(f"tm_{p}", f, mems, cols=("t", "j", "kind", "y_v", "y_q"),
                            filt=lambda H_: H_.tm_split.to_numpy() == 0)
        if not len(tm):
            continue
        P = T1.P1(p)
        tt, jj = tm.t.to_numpy(np.int64), tm.j.to_numpy(np.int64)
        for c in T1.CH:
            P.X[c][tt, jj] = np.nan
        P.refresh()
        vb, qb = T1.postprocess_panel(P, tm, pp)
        real = L._test_frame(P, t1[t1.panel.astype(str) == p]).assign(k=np.nan)
        fr = pd.concat([real, tm[["t", "j", "kind", "k"]].assign(v=vb, q=qb)], ignore_index=True)
        (v1, q1), (v2, q2) = _chain_predict(models, rep, P, fr, fr.v.to_numpy(), fr.q.to_numpy(), pp)
        n = len(real)
        day = tt // C.SLOTS
        reg = P.dayreg[day]
        yv, yq = tm.y_v.to_numpy(), tm.y_q.to_numpy()
        for nm, v_, q_ in (("B1", v1[n:], q1[n:]), ("cand", v2[n:], q2[n:])):
            g = pd.DataFrame(dict(panel=p, day=day, reg=reg, sv=(v_ - yv) ** 2, sq=(q_ - yq) ** 2, n=1, what=nm))
            recs.append(g.groupby(["what", "panel", "day", "reg"], as_index=False)[["sv", "sq", "n"]].sum())
        del P, real, fr
        C.free()
    if not recs:
        return dict(n=0, ok=False, reason="no March test-month holdout cells")
    R = pd.concat(recs, ignore_index=True)

    def score(R_):
        a = R_.groupby(["what", "panel", "reg"])[["sv", "sq", "n"]].sum().reset_index()
        a["S"] = T1._state_score(a.sv.to_numpy(), a.sq.to_numpy(), a.n.to_numpy())
        a = a.groupby(["what", "panel"]).S.mean().reset_index()
        return a.groupby("what").S.mean()

    S = score(R)
    days = np.unique(R.day.to_numpy())
    rng = np.random.default_rng(1)
    diffs = []
    for _ in range(int(lab["boot"])):
        pick = pd.Series(days[rng.integers(0, len(days), len(days))]).value_counts()
        Rb = R.merge(pick.rename("rep").rename_axis("day").reset_index(), on="day")
        for c in ("sv", "sq", "n"):
            Rb[c] = Rb[c] * Rb["rep"]
        Sb = score(Rb)
        diffs.append(Sb["cand"] - Sb["B1"])
    out = dict(S_B1=float(S["B1"]), S_cand=float(S["cand"]), d=float(S["cand"] - S["B1"]),
               se=float(np.std(diffs, ddof=1)), n=int(R[R.what == "B1"].n.sum()))
    out["ok"] = bool(out["d"] >= -float(lab["tol_tm"]))
    return out


def apply_test11(rep: dict, cfg1: dict) -> dict:
    """The adopted chain on pred/task1.parquet's values (both test months) -> pred/task1_v11.parquet (same rows and
    columns). Also checks that B1 on the test rows equals pred/task1_v10.parquet (the v10 answer) when that exists."""
    models = load_models(rep)
    if models is None:
        raise RuntimeError("t1lab11: the lab11 models are missing or belong to another lab result - rerun the lab")
    ch, mems = L.members()
    pp = dict(ch["pp"])
    t1 = pd.read_parquet(C.WORK / "pred" / "task1.parquet")
    f10 = C.WORK / "pred" / "task1_v10.parquet"
    t10 = pd.read_parquet(f10) if f10.exists() else None
    outs, chk = [], dict(max_abs_dv_B1_vs_v10=0.0, max_abs_dq_B1_vs_v10=0.0, compared=t10 is not None)
    for p in C.panels():
        rows = t1[t1.panel.astype(str) == p]
        if not len(rows):
            continue
        P = T1.P1(p)
        fr = L._test_frame(P, rows)
        (v1, q1), (v, q) = _chain_predict(models, rep, P, fr, fr.v.to_numpy(), fr.q.to_numpy(), pp)
        lanes = P.lanes[fr.j.to_numpy()]
        if t10 is not None:
            r10 = t10[t10.panel.astype(str) == p]
            if len(r10) == len(rows) and (r10.timestamp.to_numpy() == rows.timestamp.to_numpy()).all():
                chk["max_abs_dv_B1_vs_v10"] = max(chk["max_abs_dv_B1_vs_v10"], float(np.max(np.abs(v1 - r10.speed_kmh.to_numpy()))))
                chk["max_abs_dq_B1_vs_v10"] = max(chk["max_abs_dq_B1_vs_v10"], float(np.max(np.abs(q1 * lanes - r10.flow_vph.to_numpy()))))
            else:
                chk["compared"] = False
        o = rows.copy()
        o["speed_kmh"] = v
        o["flow_vph"] = q * lanes
        outs.append(o)
        C.log(f"T1 v11 test {p}: {len(o):,} rows; mean |dlog v| {float(np.mean(np.abs(np.log(np.maximum(v, 1) / np.maximum(v1, 1))))):.4f}, "
              f"mean |dlog q| {float(np.mean(np.abs(np.log(np.maximum(q, 1e-9) / np.maximum(q1, 1e-9))))):.4f} against B1")
        del P, fr
        C.free()
    out = pd.concat(outs, ignore_index=True)
    if len(out) != len(t1):
        raise RuntimeError("t1lab11: the v11 rows do not cover task1.parquet")
    tmp = C.WORK / "pred" / "task1_v11.parquet.part"
    out.to_parquet(tmp, index=False)
    tmp.replace(C.WORK / "pred" / "task1_v11.parquet")
    chk["rows"] = int(len(out))
    return chk
