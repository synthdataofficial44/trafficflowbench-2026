"""v10 Task 1 / Task 3 lab: where the conservation term S_LWR loses, and post-processings aimed at it.

The public decomposition of the v8 file (v9 probes) put the physics term at 0.735 (S_FD 0.988, S_LWR 0.608 against
about 0.95 for a perfect answer): 0.034 of the total score, the largest loss after Task 2. S_LWR is an L1 norm of the
error of the one-step change of every link's vehicle count N = k L lanes. Inside a run of target cells (t = a..b, the
released cells a-1 and b+1 known exactly) it adds up |e_a| + sum |e_t+1 - e_t| + |e_b|: the total variation of the
density error padded with zeros. Every isolated target pays 2 |e|; a constant error over a long run pays only twice;
independent cell-by-cell errors pay at every step. So S_LWR rewards density errors that are small at run ends and
smooth inside runs, which S_state (cell-wise RMSE of speed and flow) does not see.

Candidates, each applied on top of the v8 answer (`B0`: the chosen blend's raw predictions post-processed with the v8
choice, exactly the values of pred/task1.parquet), the density change split between speed and flow the way B0 splits
it (speed share `a` in dense traffic and in blackouts, `a_out` in free flow, so the accurate free-flow speeds stay):
  kfree  B0 reconciles the density with the dedicated density model only in dense traffic (v < gate v_f); in free flow
         the density is q/v of two separate models. kfree moves the free-flow density a share w of the way to the
         density model (in log).
  rts    a Kalman filter + RTS smoother of each link's log density along time: released cells observed almost exactly,
         target cells observed through our values with a larger variance (larger still in blackouts), a random walk in
         between whose variance can follow the local volatility of the released record (physics.smooth_targets).
  s2     a stage-2 model of the log-density error of a target cell from the answer and the released record around it in
         time and space (its neighbours' answers included, which the stage-1 models never see), cross-fitted by blocks of
         holdout days.
Judged on the train holdout (last 30 train days, every target, realistic blackouts; the v8 models never trained on
them) by J = 0.35 S_state + 0.10 S_LWR with the exact S_state and the conservation proxy that tracked the public
score. Adopted only when the gain over B0 exceeds `z` day-bootstrap SEs, is positive on both halves of the holdout days
and S_state does not fall by more than `tol_state`; then S_state on the held-out observed March cells (test-month
holdout, cells no member trained on) must not fall by more than its tolerance either. The adopted change is applied to
pred/task1.parquet's values -> pred/task1_v10.parquet. April is never looked at.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import common as C
from . import physics as PH
from . import task1 as T1

VERSION = "v10"
LOG_FLOOR = 1e-3
KFREE_GRID = tuple(dict(name=f"kfree_w{w}_a{a}", w=w, a=a) for a in (0.0, 0.25) for w in (0.25, 0.5, 0.75, 1.0))
RTS_GRID = (
    dict(name="rts_a", s_o=0.01, s_m=0.04, s_m_dark=0.12, q0=0.002, q_adapt=0.0),
    dict(name="rts_b", s_o=0.01, s_m=0.03, s_m_dark=0.10, q0=0.002, q_adapt=0.5),
    dict(name="rts_c", s_o=0.01, s_m=0.05, s_m_dark=0.15, q0=0.001, q_adapt=0.5),
    dict(name="rts_d", s_o=0.01, s_m=0.04, s_m_dark=0.12, q0=0.004, q_adapt=1.0),
    dict(name="rts_e", s_o=0.03, s_m=0.04, s_m_dark=0.12, q0=0.002, q_adapt=0.5),
    dict(name="rts_f", s_o=0.01, s_m=0.08, s_m_dark=0.20, q0=0.001, q_adapt=0.25),
    dict(name="rts_g", s_o=0.01, s_m=0.02, s_m_dark=0.08, q0=0.002, q_adapt=0.5),
)
S2_SHRINK = (0.5, 1.0)
S2_PARAMS = dict(objective="regression", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.8,
                 bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, max_bin=255, verbose=-1)
LAB_DEFAULTS = dict(z=2.0, tol_state=0.0005, tol_tm=0.0003, boot=400, s2=True, s2_rounds=400, s2_folds=3,
                    s2_max_rows=3_000_000, rts=True, kfree=True, tm_check=True, force=None)


# ----------------------------------------------------------------------------- members, rows, raw predictions
def members(work=None):
    """choice.json of the v8 Task 1 result and its members [(kind, path, weight)]."""
    work = work or C.WORK
    ch = json.load(open(work / "t1" / "choice.json"))
    mem = [(m["kind"], work / "t1" / m["dir"], float(m["weight"])) for _, m in sorted((ch.get("members") or {}).items())]
    if not mem:   # results of versions before v8: averaged model sets
        mem = [("lgb", work / "t1" / f"models_{ch.get('rows_tag', 'r1')}_s{s}", 1.0) for s in ch.get("seeds", [0])]
    return ch, T1.as_members(mem)


def _members_key(mems) -> str:
    out = []
    for kind, path, w in mems:
        files = T1.CB_FILES if kind == "cb" else T1.MODEL_FILES
        out.append([kind, str(Path(path).name), round(float(w), 6),
                    [[f, (Path(path) / f).stat().st_size] for f in files if (Path(path) / f).exists()]])
    return json.dumps(out)


def prepare_rows(cfg1: dict) -> Path:
    """The holdout and test rows of row seed 0 (and the test-month holdout cells) exactly as the v8 run built them
    (same code, same settings; the training rows are written too, the row builder makes them in the same pass)."""
    rows_dir = C.WORK / "t1" / f"rows_{cfg1.get('rows_tag', 'r1')}"
    infos = T1._build_rows_all(dict(cfg1, row_seed=0), rows_dir)
    C.log(f"T1 lab rows: {sum(i.get('n_hold', 0) for i in infos):,} holdout, {sum(i.get('n_test', 0) for i in infos):,} test")
    seeds = T1.plan_seeds(cfg1)
    scfg = {int(k) for k in (cfg1.get("seed_cfg") or {})}
    legacy = [s for s in seeds if s not in scfg]
    if cfg1.get("tm_holdout", True) and (scfg & set(seeds) or cfg1.get("cb", False)):
        T1.build_tm_all(cfg1, rows_dir, avoid=[dict(cfg1, row_seed=s) for s in legacy])
    return rows_dir


def _cache_file(name: str, key: dict) -> Path:
    """WORK/t1lab/<name>_<hash of key>.parquet: a cache file can only be found by the key it was written for (older
    files of the same name are removed when a new one is written)."""
    import hashlib
    h = hashlib.md5(json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()[:12]
    d = C.WORK / "t1lab"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{name}_{h}.parquet"


def _write_cache(f: Path, df: pd.DataFrame):
    for old in f.parent.glob(f.name.rsplit("_", 1)[0] + "_" + "?" * 12 + ".parquet"):
        if old != f:
            old.unlink(missing_ok=True)
    tmp = f.with_suffix(".part")
    df.to_parquet(tmp, index=False)
    tmp.replace(f)


def _cached_pred(name: str, src: Path, mems, cols=("t", "j", "kind"), filt=None) -> pd.DataFrame:
    """Blended raw predictions (v, q, k per lane) of the rows of the parquet file `src` (optionally only the rows
    where filt(frame) is True), cached in WORK/t1lab (keyed by the member files and the source file); the feature
    table is read only when the cache is missing or stale."""
    import pyarrow.parquet as pq
    key = dict(members=_members_key(mems), src=src.name, src_rows=int(pq.ParquetFile(src).metadata.num_rows),
               src_size=int(src.stat().st_size), cols=list(cols))
    f = _cache_file(name, key)
    if f.exists():
        return pd.read_parquet(f)
    H = pd.read_parquet(src)
    if filt is not None:
        H = H[filt(H)]
    preds = [T1.predict_member(H, m) for m in mems]
    pred = T1.blend_preds(preds, [w for _, _, w in mems])
    out = pd.DataFrame({c: H[c].to_numpy() for c in cols})
    out["t"] = out.t.to_numpy(np.int64); out["j"] = out.j.to_numpy(np.int64)
    for c in ("v", "q", "k"):
        out[c] = pred[c].to_numpy(np.float64)
    _write_cache(f, out)
    del H, preds, pred
    C.free()
    return out


def hold_ctx(panel: str, rows_dir: Path, mems):
    """Holdout panel (the release view with the simulated blackouts of the row build), the holdout days and the blended
    raw predictions of every holdout target (t, j, kind, v, q, k) - the frame T1.postprocess_panel takes."""
    info = json.load(open(rows_dir / f"{panel}_info.json"))
    P = T1.P1(panel)
    P.apply_blackouts(np.array(info["fit_origins"] + info["hold_origins"], np.int64))
    pred = _cached_pred(f"hold_{panel}", rows_dir / f"{panel}_hold.parquet", mems)
    return P, info, pred


# ----------------------------------------------------------------------------- scoring parts (per day and regime)
def state_parts(P, t, j, v, q) -> np.ndarray:
    """[n_days_total, 3 regimes, 3] summed squared speed error, summed squared per-lane flow error, cell count."""
    day = t // C.SLOTS
    reg = P.dayreg[day].astype(np.int64) - 1
    nd = P.NT // C.SLOTS
    out = np.zeros((nd, 3, 3))
    for c, x in enumerate(((v - P.tv[t, j]) ** 2, (q - P.tq[t, j]) ** 2, np.ones(len(t)))):
        np.add.at(out[:, :, c], (day, reg), x)
    return out


def lwr_parts(P, t, j, v, q, days) -> np.ndarray:
    """[n_days_total, 3 regimes, 2] summed |dN_sub - dN_true| and |dN_true| over the transitions of each day (by the
    day of the transition's first slot), exactly the sums T1.score_lwr takes."""
    nd = P.NT // C.SLOTS
    out = np.zeros((nd, 3, 2))
    rows = np.zeros(P.NT, bool)
    for d in days:
        rows[d * C.SLOTS:(d + 1) * C.SLOTS] = True
    sl = np.flatnonzero(rows)
    a, b = int(sl.min()), int(sl.max()) + 1
    TV = P.tv[a:b].astype(np.float64); TQ = P.tq[a:b].astype(np.float64)
    Lk = (P.length * P.lanes)[None, :].astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        Nt = TQ / np.maximum(TV, 1.0) * Lk
        SV = TV.copy(); SQ = TQ.copy()
        SV[t - a, j] = v; SQ[t - a, j] = q
        Ns = SQ / np.maximum(SV, 1.0) * Lk
    el = P.el[a:b] & np.isfinite(TV) & np.isfinite(TQ)
    reg = np.repeat(P.dayreg, C.SLOTS)[a:b]
    rr = rows[a:b]
    same = rr[:-1] & rr[1:] & (reg[:-1] == reg[1:])
    ok = el[:-1] & el[1:] & same[:, None] & P.rv[a:b][:-1] & P.observable[None, :]
    dNt = Nt[1:] - Nt[:-1]
    res = np.where(ok, np.abs((Ns[1:] - Ns[:-1]) - dNt), 0.0).sum(1)
    den = np.where(ok, np.abs(dNt), 0.0).sum(1)
    tday = (a + np.arange(b - a - 1)) // C.SLOTS
    treg = reg[:-1].astype(np.int64) - 1
    np.add.at(out[:, :, 0], (tday, treg), res)
    np.add.at(out[:, :, 1], (tday, treg), den)
    return out


def _scores(sp: np.ndarray, lp: np.ndarray):
    """(S_state, S_LWR) of one panel from summed parts [regime, ...] (regimes with data averaged, as the metric)."""
    ss, sl = [], []
    for r in range(3):
        if sp[r, 2] > 0:
            rv, rq = np.sqrt(sp[r, 0] / sp[r, 2]), np.sqrt(sp[r, 1] / sp[r, 2])
            ss.append(0.54 * max(0.0, 1 - rv / 25) + 0.46 * max(0.0, 1 - rq / 600))
        if lp[r, 1] > 0:
            sl.append(1 - min(1.0, lp[r, 0] / lp[r, 1]))
    return (float(np.mean(ss)) if ss else np.nan), (float(np.mean(sl)) if sl else np.nan)


class Book:
    """Scoring parts of every variant, per panel; J tables, day-bootstrap SEs and half splits of differences."""

    def __init__(self, days_by_panel: dict, boot: int = 400, seed: int = 0):
        self.days = days_by_panel
        self.parts = {}            # variant -> panel -> (state [D, 3, 3], lwr [D, 3, 2]) on that panel's holdout days
        alld = sorted({int(d) for v in days_by_panel.values() for d in v})
        self.alldays = np.array(alld, np.int64)
        rng = np.random.default_rng(seed)
        self.W = np.stack([np.bincount(rng.integers(0, len(alld), len(alld)), minlength=len(alld)) for _ in range(boot)])

    def add(self, name, panel, P, t, j, v, q):
        d = np.asarray(self.days[panel], np.int64)
        self.parts.setdefault(name, {})[panel] = (state_parts(P, t, j, v, q)[d], lwr_parts(P, t, j, v, q, d)[d])

    def _panel_w(self, panel, w):
        """Day weights of a panel's holdout days from weights over all days."""
        idx = np.searchsorted(self.alldays, np.asarray(self.days[panel], np.int64))
        return w[..., idx]

    def _rows(self, name, w=None):
        out = []
        for p, (sp, lp) in self.parts[name].items():
            ww = np.ones(len(sp)) if w is None else self._panel_w(p, w)
            s, l_ = _scores(np.tensordot(ww, sp, 1), np.tensordot(ww, lp, 1))
            out.append((p, s, l_, 0.35 * s + 0.10 * l_))
        return out

    def table(self, name, w=None) -> pd.DataFrame:
        return pd.DataFrame(self._rows(name, w), columns=["panel", "S_state", "S_LWR", "J"])

    def mean(self, name, w=None) -> dict:
        r = np.array([x[1:] for x in self._rows(name, w)], np.float64)
        return dict(zip(("S_state", "S_LWR", "J"), (float(x) for x in np.nanmean(r, 0))))

    def diff(self, a, b) -> dict:
        """b - a: mean differences, day-bootstrap SE of dJ, and dJ on the first / second half of the days."""
        ma, mb = self.mean(a), self.mean(b)
        out = {f"d_{k}": mb[k] - ma[k] for k in ma}
        dj = []
        for w in self.W:
            dj.append(self.mean(b, w)["J"] - self.mean(a, w)["J"])
        out["se_J"] = float(np.std(dj, ddof=1))
        h = len(self.alldays) // 2
        for nm, m in (("first_half", np.arange(len(self.alldays)) < h), ("second_half", np.arange(len(self.alldays)) >= h)):
            w = m.astype(float)
            out[f"dJ_{nm}"] = self.mean(b, w)["J"] - self.mean(a, w)["J"]
        out.update({f"{k}_{a}": v for k, v in ma.items()}, **{f"{k}_new": v for k, v in mb.items()})
        return out


# ----------------------------------------------------------------------------- candidates
def _share(P, j, kind, v, q, pp, a_free=None):
    """Speed share of a density change the way B0 splits it: pp['a'] in dense traffic (v < gate v_f) and in blackouts,
    pp['a_out'] (or a_free) elsewhere."""
    vf = P.vf[j].astype(np.float64)
    dense = (np.asarray(kind) > 0.5) | (v < pp.get("gate", 0.6) * vf)
    return np.where(dense, pp.get("a", 0.75), pp.get("a_out", 0.0) if a_free is None else a_free)


def _apply_dl(P, j, kind, v, q, dl, pp, a_free=None):
    a = _share(P, j, kind, v, q, pp, a_free)
    dl = np.nan_to_num(dl)
    v2 = v * np.exp(-a * dl); q2 = q * np.exp((1 - a) * dl)
    vf = P.vf[j].astype(np.float64)
    return np.clip(v2, 3.0, 1.1 * vf), np.clip(q2, 100.0 / P.lanes[j], None)


def kfree(P, fr, v, q, prm, pp, cap=0.5):
    """Free-flow cells (outside B0's dense reconciliation): log density a share w of the way to the density model."""
    j = fr.j.to_numpy(np.int64); kind = fr.kind.to_numpy()
    km = fr.k.to_numpy(np.float64) if "k" in fr else np.full(len(fr), np.nan)
    vf = P.vf[j].astype(np.float64)
    m = (v >= pp.get("gate", 0.6) * vf) & (np.asarray(kind) < 0.5) & np.isfinite(km) & (km > 0)
    k0 = q / np.maximum(v, 1.0)
    dl = np.where(m, np.clip(np.log(np.maximum(km, LOG_FLOOR)) - np.log(np.maximum(k0, LOG_FLOOR)), -cap, cap) * prm["w"], 0.0)
    return _apply_dl(P, j, kind, v, q, dl, pp, a_free=prm["a"])


def rts(P, fr, v, q, prm, pp):
    """Kalman / RTS smoothing of the log density (physics.smooth_targets), density change split as B0 splits it."""
    t = fr.t.to_numpy(np.int64); j = fr.j.to_numpy(np.int64); kind = fr.kind.to_numpy()
    _, q2 = PH.smooth_targets(P, t, j, v, q, kind, dict(prm, a=0.0))
    dl = np.log(np.maximum(q2, 1e-9)) - np.log(np.maximum(q, 1e-9))
    return _apply_dl(P, j, kind, v, q, dl, pp)


S2_DT = (-3, -2, -1, 1, 2, 3)


def s2_features(P, fr, v, q) -> pd.DataFrame:
    """Features of every target cell of `fr` (t, j, kind) for the stage-2 model, from the answer (v, q) at the target
    cells of `fr` and the released record elsewhere (P.X; NaN where neither)."""
    t = fr.t.to_numpy(np.int64); j = fr.j.to_numpy(np.int64); kind = fr.kind.to_numpy().astype(np.float32)
    T_, L = P.NT, P.L
    with np.errstate(invalid="ignore", divide="ignore"):
        K = np.where(P.ok, P.X["q"] / np.maximum(P.X["v"], 1.0), np.nan).astype(np.float32)
        V = np.where(P.ok, P.X["v"], np.nan).astype(np.float32)
        K[t, j] = q / np.maximum(v, 1.0); V[t, j] = v
        LK = np.log(np.maximum(K, LOG_FLOOR)).astype(np.float32)
    TG = np.zeros((T_, L), bool); TG[t, j] = True
    TGf = TG.astype(np.float32)
    f0 = LK[t, j]
    vf = P.vf[j].astype(np.float32)
    F = {"lk0": f0, "vr0": (v / vf).astype(np.float32), "kind": kind}

    def get(A, dt, dl, fill=np.nan):
        t2, l2 = t + dt, j + dl
        ok = (t2 >= 0) & (t2 < T_) & (l2 >= 0) & (l2 < L)
        out = np.full(len(t), fill, np.float32)
        out[ok] = A[t2[ok], l2[ok]]
        return out

    for dt in S2_DT:
        F[f"d{dt}"] = get(LK, dt, 0) - f0
        F[f"tg{dt}"] = get(TGf, dt, 0)
    F["curv"] = get(LK, -1, 0) + get(LK, 1, 0) - 2 * f0
    for dl in (-1, 1):
        for dt in (-1, 0, 1):
            F[f"n{dl}_{dt}"] = get(LK, dt, dl) - f0
        F[f"n{dl}_tg"] = get(TGf, 0, dl)
    F["vr_m1"] = get(V, -1, 0) / vf; F["vr_p1"] = get(V, 1, 0) / vf
    # the run of target cells the cell sits in (along time, same link) and its released anchors
    pos = np.zeros(len(t), np.float32); rlen = np.zeros(len(t), np.float32)
    al = np.full(len(t), np.nan, np.float32); ar = np.full(len(t), np.nan, np.float32)
    order = np.lexsort((t, j))
    tj, jj = t[order], j[order]
    ls, starts = np.unique(jj, return_index=True)
    ends = np.append(starts[1:], len(jj))
    for l, a_, b_ in zip(ls, starts, ends):
        sel = np.arange(a_, b_)
        s, e = C.runs_1d(TG[:, l])
        if not len(s):
            continue
        ri = np.searchsorted(s, tj[sel], side="right") - 1
        st, en = s[ri], e[ri]
        idx = order[sel]
        pos[idx] = tj[sel] - st; rlen[idx] = en - st
        al[idx] = np.where(st - 1 >= 0, LK[np.clip(st - 1, 0, T_ - 1), l], np.nan)
        ar[idx] = np.where(en < T_, LK[np.clip(en, 0, T_ - 1), l], np.nan)
    F["pos"] = pos; F["rlen"] = rlen; F["pos_rel"] = pos / np.maximum(rlen - 1, 1)
    F["anc_l"] = al - f0; F["anc_r"] = ar - f0
    F["regime"] = P.dayreg[t // C.SLOTS].astype(np.float32)
    F["tod"] = (t % C.SLOTS).astype(np.float32)
    F["lanes"] = P.lanes[j]; F["length"] = P.length[j]; F["vf"] = vf
    F["dark_pos"] = P.dark_pos[t].astype(np.float32); F["dark_rem"] = P.dark_rem[t].astype(np.float32)
    return pd.DataFrame(F)


def s2_target(P, fr, v, q) -> np.ndarray:
    t = fr.t.to_numpy(np.int64); j = fr.j.to_numpy(np.int64)
    with np.errstate(invalid="ignore", divide="ignore"):
        kt = P.tq[t, j] / np.maximum(P.tv[t, j], 1.0)
        y = np.log(np.maximum(kt, LOG_FLOOR)) - np.log(np.maximum(q / np.maximum(v, 1.0), LOG_FLOOR))
    return np.clip(y, -0.7, 0.7).astype(np.float32)


def s2_fit(X: pd.DataFrame, y: np.ndarray, rounds: int, device: dict | None = None, seed: int = 0):
    import lightgbm as lgb
    p = dict(S2_PARAMS, num_threads=C.THREADS, seed=seed, **(device or {}))
    ok = np.isfinite(y)
    try:
        return lgb.train(p, lgb.Dataset(X[ok].to_numpy(np.float32), y[ok], feature_name=list(X.columns)), rounds)
    except Exception:  # noqa: BLE001  a device problem: the CPU
        if not device:
            raise
        p = dict(S2_PARAMS, num_threads=C.THREADS, seed=seed)
        return lgb.train(p, lgb.Dataset(X[ok].to_numpy(np.float32), y[ok], feature_name=list(X.columns)), rounds)


def s2_apply(P, fr, v, q, yhat, shrink, pp):
    j = fr.j.to_numpy(np.int64); kind = fr.kind.to_numpy()
    return _apply_dl(P, j, kind, v, q, shrink * np.asarray(yhat, np.float64), pp)


# ----------------------------------------------------------------------------- the lab
def lab_key(cfg1: dict, lab: dict | None = None, work=None) -> dict:
    """What a lab result depends on: the code (task1, physics, this file), the settings, the members' model files and
    the v8 answer (task1.parquet + choice.json). A stored lab report is reused only when all of it is the same."""
    import hashlib
    work = work or C.WORK
    h = hashlib.md5()
    for f in ("task1.py", "physics.py", "t1lab.py", "common.py", "data.py"):
        h.update((Path(__file__).parent / f).read_bytes())
    ch, mems = members(work)
    tp = work / "pred" / "task1.parquet"
    cfg1 = {k: v for k, v in cfg1.items() if k != "row_jobs"}
    return json.loads(json.dumps(dict(code=h.hexdigest()[:16], cfg1=cfg1, lab=dict(LAB_DEFAULTS, **(lab or {})),
                                      members=_members_key(mems), pp=ch.get("pp"),
                                      task1_md5=ch.get("task1_md5") or (T1.file_md5(tp) if tp.exists() else None)),
                                 sort_keys=True, default=str))


def _apply_chain(chain: list, P, fr, v, q, pp, s2_pred=None):
    """Apply the steps of a candidate chain [(family, params)] in order; s2 takes its predictions from s2_pred(P, fr, v, q)."""
    for fam, prm in chain:
        if fam == "kfree":
            v, q = kfree(P, fr, v, q, prm, pp)
        elif fam == "rts":
            v, q = rts(P, fr, v, q, prm, pp)
        elif fam == "s2":
            v, q = s2_apply(P, fr, v, q, s2_pred(P, fr, v, q), prm["shrink"], pp)
        else:
            raise ValueError(fam)
    return v, q


def _passes(d: dict, lab: dict) -> bool:
    return bool(d["d_J"] > lab["z"] * max(d["se_J"], 1e-12) and d["dJ_first_half"] > 0 and d["dJ_second_half"] > 0
                and d["d_S_state"] >= -lab["tol_state"])


def run_lab(cfg1: dict, lab: dict | None = None, device: dict | None = None) -> dict:
    """The lab on the train holdout (see the module docstring). Writes WORK/t1lab/lab_report.json and returns it."""
    lab = dict(LAB_DEFAULTS, **(lab or {}))
    t0 = time.time()
    ch, mems = members()
    pp = dict(ch["pp"])
    rows_dir = prepare_rows(cfg1)
    panels = [p for p in C.panels() if (rows_dir / f"{p}_hold.parquet").exists()]
    infos = {p: json.load(open(rows_dir / f"{p}_info.json")) for p in panels}
    book = Book({p: infos[p]["hold_days"] for p in panels}, boot=int(lab["boot"]))
    rep = dict(version=VERSION, key=lab_key(cfg1, lab), pp=pp, members=[[k, Path(p).name, w] for k, p, w in mems], lab=lab,
               plan_matches=(ch.get("plan") == T1.plan_signature(cfg1)) if ch.get("plan") else None,
               breakdown={}, variants={}, minutes={})
    # ---- pass 1: B0 (the v8 answer) and the free-flow density variants
    for p in panels:
        P, info, fr = hold_ctx(p, rows_dir, mems)
        t, j = fr.t.to_numpy(np.int64), fr.j.to_numpy(np.int64)
        v0, q0 = T1.postprocess_panel(P, fr, pp)
        book.add("B0", p, P, t, j, v0, q0)
        try:
            rep["breakdown"][p] = PH.lwr_breakdown(P, t, j, v0, q0, info["hold_days"])
        except Exception as e:  # noqa: BLE001  diagnostics never stop the lab
            rep["breakdown"][p] = {"error": repr(e)[:200]}
        if lab["kfree"]:
            for prm in KFREE_GRID:
                v, q = kfree(P, fr, v0, q0, prm, pp)
                book.add(prm["name"], p, P, t, j, v, q)
        C.log(f"T1 lab {p}: B0 {book.table('B0').iloc[-1].to_dict()}")
        del P, fr
        C.free()
    rep["minutes"]["pass1"] = round((time.time() - t0) / 60, 1)
    b0 = book.mean("B0")
    rep["B0"] = b0
    hold = {r["variant"]: r for r in (ch.get("holdout") or [])}
    best_row = (ch.get("holdout") or [{}])[0]
    rep["v8_holdout_best"] = best_row
    rep["reproduces_v8"] = bool(best_row and all(abs(b0[k] - best_row.get(k, np.nan)) < 1e-3 for k in ("S_state", "S_LWR")))
    rep["reproduce_diff"] = {k: (b0[k] - best_row[k]) if k in best_row else None for k in ("S_state", "S_LWR", "J")}
    for name in book.parts:
        if name != "B0":
            rep["variants"][name] = book.diff("B0", name)
    chain = []
    kf = [n for n in rep["variants"] if n.startswith("kfree")]
    if kf:
        best = max(kf, key=lambda n: rep["variants"][n]["d_J"])
        d = rep["variants"][best]
        if d["d_J"] > 0 and d["dJ_first_half"] > 0 and d["dJ_second_half"] > 0:
            chain = [("kfree", next(g for g in KFREE_GRID if g["name"] == best))]
    rep["base_chain"] = [[f, prm["name"]] for f, prm in chain]
    base_name = "B0" if not chain else chain[0][1]["name"]
    # ---- pass 2: RTS variants on the base chain, and the stage-2 features
    s2_X, s2_y, s2_meta = [], [], []
    for p in panels:
        P, info, fr = hold_ctx(p, rows_dir, mems)
        t, j = fr.t.to_numpy(np.int64), fr.j.to_numpy(np.int64)
        v0, q0 = T1.postprocess_panel(P, fr, pp)
        v1, q1 = _apply_chain(chain, P, fr, v0, q0, pp)
        if lab["rts"]:
            for prm in RTS_GRID:
                v, q = rts(P, fr, v1, q1, prm, pp)
                book.add(f"{base_name}+{prm['name']}", p, P, t, j, v, q)
        if lab["s2"]:
            X = s2_features(P, fr, v1, q1)
            y = s2_target(P, fr, v1, q1)
            s2_X.append(X); s2_y.append(y)
            s2_meta.append(pd.DataFrame(dict(panel=p, day=(t // C.SLOTS).astype(np.int64), i=np.arange(len(t)))))
        del P, fr
        C.free()
    rep["minutes"]["pass2"] = round((time.time() - t0) / 60, 1)
    s2_oof = {}
    if lab["s2"] and s2_X:
        X = pd.concat(s2_X, ignore_index=True); y = np.concatenate(s2_y); M = pd.concat(s2_meta, ignore_index=True)
        del s2_X, s2_y
        days = book.alldays
        fold_of_day = {int(d): min(int(lab["s2_folds"]) - 1, i * int(lab["s2_folds"]) // len(days)) for i, d in enumerate(days)}
        fold = M.day.map(fold_of_day).to_numpy()
        oof = np.full(len(X), np.nan, np.float32)
        rng = np.random.default_rng(0)
        imp = None
        for k in range(int(lab["s2_folds"])):
            tr = np.flatnonzero(fold != k)
            if len(tr) > int(lab["s2_max_rows"]):
                tr = np.sort(rng.choice(tr, int(lab["s2_max_rows"]), replace=False))
            m = s2_fit(X.iloc[tr], y[tr], int(lab["s2_rounds"]), device, seed=k)
            te = fold == k
            oof[te] = m.predict(X[te].to_numpy(np.float32), num_threads=C.THREADS)
            g = pd.Series(m.feature_importance("gain"), index=m.feature_name())
            imp = g if imp is None else imp + g
        ok = np.isfinite(y)
        rep["s2"] = dict(rows=int(len(X)), rmse_before=float(np.sqrt(np.mean(y[ok] ** 2))),
                         rmse_oof=float(np.sqrt(np.mean((y[ok] - oof[ok]) ** 2))),
                         top_features={k: round(float(v), 1) for k, v in (imp / imp.sum() * 100).sort_values(ascending=False).head(12).items()})
        for p in panels:
            s2_oof[p] = oof[(M.panel == p).to_numpy()]
        del X, y, M
        C.free()
        # ---- pass 3: score the stage-2 variants (out-of-fold predictions)
        for p in panels:
            P, info, fr = hold_ctx(p, rows_dir, mems)
            t, j = fr.t.to_numpy(np.int64), fr.j.to_numpy(np.int64)
            v0, q0 = T1.postprocess_panel(P, fr, pp)
            v1, q1 = _apply_chain(chain, P, fr, v0, q0, pp)
            for s in S2_SHRINK:
                v, q = s2_apply(P, fr, v1, q1, s2_oof[p], s, pp)
                book.add(f"{base_name}+s2_x{s}", p, P, t, j, v, q)
            del P, fr
            C.free()
    rep["minutes"]["pass3"] = round((time.time() - t0) / 60, 1)
    for name in book.parts:
        if name != "B0" and name not in rep["variants"]:
            rep["variants"][name] = book.diff("B0", name)
    # ---- the candidate: the best J among the free-flow variants and every variant on top of the base chain
    cands = {g["name"]: [("kfree", g)] for g in KFREE_GRID if g["name"] in rep["variants"]}
    for name in rep["variants"]:
        if name.startswith(base_name + "+rts"):
            prm = next(g for g in RTS_GRID if g["name"] == name.split("+")[-1])
            cands[name] = chain + [("rts", prm)]
        elif name.startswith(base_name + "+s2"):
            cands[name] = chain + [("s2", dict(shrink=float(name.split("_x")[-1])))]
    forced = lab.get("force")
    best = None
    if cands:
        best = forced if forced in cands else max(cands, key=lambda n: rep["variants"][n]["d_J"])
    rep["candidate"] = best
    rep["candidate_chain"] = [[f, prm] for f, prm in cands[best]] if best else []
    d = rep["variants"].get(best) if best else None
    rep["passes_holdout"] = bool(d is not None and _passes(d, lab))
    rep["adopted"] = bool(best is not None and (forced == best or (rep["passes_holdout"] and rep["reproduces_v8"] is not False)))
    rep["minutes"]["total_lab"] = round((time.time() - t0) / 60, 1)
    rep["table"] = pd.DataFrame({n: {k: v for k, v in dd.items() if k in ("d_J", "se_J", "d_S_state", "d_S_LWR",
                                                                             "dJ_first_half", "dJ_second_half")}
                                 for n, dd in rep["variants"].items()}).T.sort_values("d_J", ascending=False).round(6).to_dict("index")
    C.log(f"T1 lab: B0 {b0} | candidate {best} {d and {k: round(d[k], 6) for k in ('d_J', 'se_J', 'd_S_state', 'd_S_LWR')}} "
          f"| adopted (holdout) {rep['adopted']}")
    _save(rep)
    return rep


def _save(rep):
    d = C.WORK / "t1lab"
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / "lab_report.json.part"
    json.dump(rep, open(tmp, "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    tmp.replace(d / "lab_report.json")


# ----------------------------------------------------------------------------- the test months
_S2_CACHE: dict = {}


def _s2_model(chain, cfg1, mems, pp, lab, device):
    """The stage-2 model fitted on every holdout day (for the test months), on the answer of the chain before it
    (fitted once per session and chain)."""
    pre = [c for c in chain if c[0] != "s2"]
    ck = json.dumps([pre, _members_key(mems), pp, lab.get("s2_rounds"), lab.get("s2_max_rows")], sort_keys=True, default=str)
    if ck in _S2_CACHE:
        return _S2_CACHE[ck]
    rows_dir = C.WORK / "t1" / f"rows_{cfg1.get('rows_tag', 'r1')}"
    Xs, ys = [], []
    for p in [p for p in C.panels() if (rows_dir / f"{p}_hold.parquet").exists()]:
        P, info, fr = hold_ctx(p, rows_dir, mems)
        v0, q0 = T1.postprocess_panel(P, fr, pp)
        v1, q1 = _apply_chain(pre, P, fr, v0, q0, pp)
        Xs.append(s2_features(P, fr, v1, q1)); ys.append(s2_target(P, fr, v1, q1))
        del P, fr
        C.free()
    X = pd.concat(Xs, ignore_index=True); y = np.concatenate(ys)
    if len(X) > int(lab["s2_max_rows"]):
        idx = np.sort(np.random.default_rng(0).choice(len(X), int(lab["s2_max_rows"]), replace=False))
        X, y = X.iloc[idx], y[idx]
    _S2_CACHE.clear()
    _S2_CACHE[ck] = s2_fit(X, y, int(lab["s2_rounds"]), device, seed=0)
    return _S2_CACHE[ck]


def _predict_k(D: pd.DataFrame, mems) -> np.ndarray:
    """Blended density-model predictions (per lane) of the rows D (only the k models are evaluated)."""
    out = np.zeros(len(D))
    for kind_name, path, w in mems:
        if kind_name == "cb":
            out += w * T1.predict_member(D, (kind_name, path, w)).k.to_numpy()
            continue
        k = np.full(len(D), np.nan)
        for kind, kv in (("reg", 0.0), ("dark", 1.0)):
            m = (D.kind == kv).to_numpy()
            if not m.any():
                continue
            Dm = D.loc[m]
            bst = T1._booster(Path(path) / f"{kind}_k.txt")
            k[m] = bst.predict(np.ascontiguousarray(Dm[list(bst.feature_name())].to_numpy(np.float32)),
                               num_threads=C.THREADS) + T1.base(Dm, "k")
        out += w * k
    return out


def _test_k(panel: str, rows_dir: Path, mems) -> pd.Series:
    """Blended density-model predictions of a panel's test rows, indexed by (t, j); cached (about 2 minutes per million
    rows and member on 8 cores)."""
    import pyarrow.parquet as pq
    src = rows_dir / f"{panel}_test.parquet"
    key = dict(members=_members_key(mems), src=src.name, src_rows=int(pq.ParquetFile(src).metadata.num_rows),
               src_size=int(src.stat().st_size))
    f = _cache_file(f"testk_{panel}", key)
    if f.exists():
        d = pd.read_parquet(f)
    else:
        D = pd.read_parquet(src)
        d = pd.DataFrame(dict(t=D.t.to_numpy(np.int64), j=D.j.to_numpy(np.int64), k=_predict_k(D, mems)))
        del D
        C.free()
        _write_cache(f, d)
    return pd.Series(d.k.to_numpy(), index=pd.MultiIndex.from_arrays([d.t.to_numpy(np.int64), d.j.to_numpy(np.int64)]))


def _with_k(fr: pd.DataFrame, kser: pd.Series, panel: str) -> pd.DataFrame:
    k = kser.reindex(pd.MultiIndex.from_arrays([fr.t.to_numpy(np.int64), fr.j.to_numpy(np.int64)])).to_numpy()
    if np.isnan(k).mean() > 0.001:
        raise RuntimeError(f"T1 lab: density predictions missing for {np.isnan(k).mean():.1%} of {panel}'s test cells")
    return fr.assign(k=k)


def _test_frame(P, rows: pd.DataFrame) -> pd.DataFrame:
    """task1.parquet rows of one panel -> (t, j, kind, v, q per lane)."""
    lidx = {str(s): i for i, s in enumerate(P.net.links)}
    t = C.tindex(rows.timestamp.astype(str)).astype(np.int64)
    j = rows.link_id.astype(str).map(lidx)
    if j.isna().any():
        raise RuntimeError(f"T1 lab: {int(j.isna().sum())} task1 rows of {P.panel} with unknown links")
    j = j.to_numpy(np.int64)
    return pd.DataFrame(dict(t=t, j=j, kind=P.dark[t].astype(np.float32), v=rows.speed_kmh.to_numpy(np.float64),
                             q=rows.flow_vph.to_numpy(np.float64) / P.lanes[j]))


def tm_check(rep: dict, cfg1: dict, lab: dict | None = None, device: dict | None = None, s2m=None) -> dict:
    """S_state of the candidate against B0 on the held-out observed test-month cells (cells no member trained on),
    each hidden and answered as an isolated target among the real targets (their v8 answers) - March decides, April is
    never looked at. Returns {S_B0, S_cand, d, se, n, ok} for March: ok when S_state falls by at most `tol_tm`.
    Needs the rows of prepare_rows (the test-month holdout cells, and the test rows when the chain has kfree: the real
    targets then get the density-model values too, as in apply_test)."""
    lab = dict(LAB_DEFAULTS, **(lab or {}))
    if not lab.get("tm_check", True):
        return dict(n=0, ok=True, reason="tm_check switched off")
    ch, mems = members()
    pp = dict(ch["pp"])
    chain = [(f, prm) for f, prm in rep["candidate_chain"]]
    rows_dir = C.WORK / "t1" / f"rows_{cfg1.get('rows_tag', 'r1')}"
    tm_files = {p: rows_dir / f"{p}_tm.parquet" for p in C.panels()}
    if not any(f.exists() for f in tm_files.values()):
        raise RuntimeError(f"T1 lab: no test-month holdout files in {rows_dir} - run prepare_rows first")
    t1 = pd.read_parquet(C.WORK / "pred" / "task1.parquet")
    dec = 0   # March
    need_k = any(f == "kfree" for f, _ in chain)
    if any(f == "s2" for f, _ in chain) and s2m is None:
        s2m = _s2_model(chain, cfg1, mems, pp, lab, device)
    recs = []
    for p in C.panels():
        f = tm_files[p]
        if not f.exists():
            continue
        tm = _cached_pred(f"tm_{p}", f, mems, cols=("t", "j", "kind", "y_v", "y_q"),
                          filt=lambda H_: H_.tm_split.to_numpy() == dec)
        if not len(tm):
            continue
        P = T1.P1(p)
        tt, jj = tm.t.to_numpy(np.int64), tm.j.to_numpy(np.int64)
        for c in T1.CH:   # the held-out cells become isolated targets
            P.X[c][tt, jj] = np.nan
        P.refresh()
        vb, qb = T1.postprocess_panel(P, tm, pp)
        real = _test_frame(P, t1[t1.panel.astype(str) == p])
        real = _with_k(real, _test_k(p, rows_dir, mems), p) if need_k else real.assign(k=np.nan)
        fr = pd.concat([real, tm[["t", "j", "kind", "k"]].assign(v=vb, q=qb)], ignore_index=True)
        v_all, q_all = fr.v.to_numpy(), fr.q.to_numpy()
        s2_pred = (lambda P_, fr_, v_, q_: s2m.predict(s2_features(P_, fr_, v_, q_).to_numpy(np.float32),
                                                        num_threads=C.THREADS)) if s2m is not None else None
        vc, qc = _apply_chain(chain, P, fr, v_all, q_all, pp, s2_pred)
        n = len(real)
        vc, qc = vc[n:], qc[n:]
        day = tt // C.SLOTS
        reg = P.dayreg[day]
        yv, yq = tm.y_v.to_numpy(), tm.y_q.to_numpy()
        for nm, v_, q_ in (("B0", vb, qb), ("cand", vc, qc)):
            g = pd.DataFrame(dict(panel=p, day=day, reg=reg, sv=(v_ - yv) ** 2, sq=(q_ - yq) ** 2, n=1, what=nm))
            recs.append(g.groupby(["what", "panel", "day", "reg"], as_index=False)[["sv", "sq", "n"]].sum())
        del P, real, fr
        C.free()
    if not recs:   # nothing to check is not a pass
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
        diffs.append(Sb["cand"] - Sb["B0"])
    out = dict(S_B0=float(S["B0"]), S_cand=float(S["cand"]), d=float(S["cand"] - S["B0"]),
               se=float(np.std(diffs, ddof=1)), n=int(R[R.what == "B0"].n.sum()))
    out["ok"] = bool(out["d"] >= -float(lab["tol_tm"]))
    return out


def apply_test(rep: dict, cfg1: dict, lab: dict | None = None, device: dict | None = None, s2m=None) -> pd.DataFrame:
    """The adopted chain on pred/task1.parquet's values (both test months) -> pred/task1_v10.parquet (same rows and
    columns). The density model is evaluated on the test rows only when the chain needs it (kfree)."""
    lab = dict(LAB_DEFAULTS, **(lab or {}))
    ch, mems = members()
    pp = dict(ch["pp"])
    chain = [(f, prm) for f, prm in rep["candidate_chain"]]
    rows_dir = C.WORK / "t1" / f"rows_{cfg1.get('rows_tag', 'r1')}"
    t1 = pd.read_parquet(C.WORK / "pred" / "task1.parquet")
    if any(f == "s2" for f, _ in chain) and s2m is None:
        s2m = _s2_model(chain, cfg1, mems, pp, lab, device)
    need_k = any(f == "kfree" for f, _ in chain)
    outs = []
    for p in C.panels():
        rows = t1[t1.panel.astype(str) == p]
        if not len(rows):
            continue
        P = T1.P1(p)
        fr = _test_frame(P, rows)
        if need_k:
            fr = _with_k(fr, _test_k(p, rows_dir, mems), p)
        s2_pred = (lambda P_, fr_, v_, q_: s2m.predict(s2_features(P_, fr_, v_, q_).to_numpy(np.float32),
                                                        num_threads=C.THREADS)) if s2m is not None else None
        v, q = _apply_chain(chain, P, fr, fr.v.to_numpy(), fr.q.to_numpy(), pp, s2_pred)
        o = rows.copy()
        o["speed_kmh"] = v
        o["flow_vph"] = q * P.lanes[fr.j.to_numpy()]
        outs.append(o)
        C.log(f"T1 v10 test {p}: {len(o):,} rows, mean |d log k| "
              f"{float(np.mean(np.abs(np.log(np.maximum(q, 1e-9) / np.maximum(fr.q.to_numpy(), 1e-9)) + np.log(np.maximum(fr.v.to_numpy(), 1) / np.maximum(v, 1))))):.4f}")
        del P, fr
        C.free()
    out = pd.concat(outs, ignore_index=True)
    if len(out) != len(t1):
        raise RuntimeError("T1 lab: the v10 rows do not cover task1.parquet")
    tmp = C.WORK / "pred" / "task1_v10.parquet.part"
    out.to_parquet(tmp, index=False)
    tmp.replace(C.WORK / "pred" / "task1_v10.parquet")
    return out
