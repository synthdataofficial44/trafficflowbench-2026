"""v10: the Task 3 conservation term (S_LWR) - where its loss sits, and a post-processing that targets it.

The public decomposition of the v8 file (v9 probes) put S_LWR at 0.608 against about 0.95 for a perfect answer: 0.034 of
the total score, the second largest loss after Task 2. S_LWR is an L1 norm of the error of the one-step change of the
vehicle count N = k L on every link,

    E_LWR = sum_t |dN_sub(t) - rhs(t)| / sum_t |rhs(t)|,      rhs = the organizer's boundary flux (consistent with the
                                                                  released observations),

so what matters at a target cell is its density error measured against how much N changes in five minutes - a cell
error of a few % of the density is small for S_state and large for S_LWR. Our Task 1 values come from separate speed
and flow models (density = q / v), cell by cell: their errors are independent from one five-minute slot to the next,
which S_LWR counts twice per isolated target (once going in, once coming out) and once per step inside runs.

`smooth_targets` treats each link's log density as a local-level state (random walk with variance Q per slot) observed
exactly-ish on the released cells (variance s_o^2) and through our model on the target cells (s_m^2, larger on the
blackout rows), runs a Kalman filter + RTS smoother along time, and moves each target's density to the smoothed value,
the speed taking a share `a` of the log change and the flow the rest. For an isolated target this is the variance-
weighted blend of our value and the interpolation of its released neighbours; inside runs it makes the errors coherent
in time. Q can follow the local volatility of the released record (`q_adapt`). Every variant is judged on the train
holdout by J = 0.35 S_state + 0.10 S_LWR (the same exact S_state and conservation proxy as the post-processing choice),
so it is adopted only where it does not cost S_state more than it gains in S_LWR.

`lwr_breakdown` splits the conservation residual by regime, by blackout vs regular rows, by isolated vs run targets
and by free vs congested traffic, to show where the loss is.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import common as C

LOG_FLOOR = 1e-3


def rts_local_level(Y: np.ndarray, R: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Kalman filter + Rauch-Tung-Striebel smoother of a local-level model, vectorised over columns.
    Y [T, L] observations (NaN = none), R [T, L] observation variances, Q [T, L] process variance of the step into t.
    Returns the smoothed mean [T, L] (columns without any observation stay NaN)."""
    T, L = Y.shape
    xf = np.zeros((T, L)); pf = np.zeros((T, L)); xp = np.zeros((T, L)); pp = np.zeros((T, L))
    obs = np.isfinite(Y)
    first = np.where(obs.any(0), np.argmax(obs, 0), -1)
    x = np.where(first >= 0, Y[np.clip(first, 0, T - 1), np.arange(L)], 0.0)
    p = np.full(L, 1e6)
    for t in range(T):
        xpt = x; ppt = p + Q[t]
        o = obs[t]
        k = np.where(o, ppt / (ppt + R[t]), 0.0)
        x = np.where(o, xpt + k * (np.nan_to_num(Y[t]) - xpt), xpt)
        p = np.where(o, (1 - k) * ppt, ppt)
        xp[t] = xpt; pp[t] = ppt; xf[t] = x; pf[t] = p
    xs = np.empty((T, L)); xs[T - 1] = xf[T - 1]
    for t in range(T - 2, -1, -1):
        g = pf[t] / np.maximum(pp[t + 1], 1e-12)
        xs[t] = xf[t] + g * (xs[t + 1] - xp[t + 1])
    xs[:, first < 0] = np.nan
    return xs


def _span(t: np.ndarray, T: int, pad: int = 36):
    return max(0, int(t.min()) - pad), min(T, int(t.max()) + pad + 1)


def smooth_targets(P, t, j, v, q, kind, prm: dict):
    """New (v, q) (per lane) at the target cells (t, j) of panel arrays P (task1.P1: released view P.X, lanes...):
    the local-level RTS smoother of log density per link (see the module docstring). prm: s_o, s_m, s_m_dark, q0,
    q_adapt (0 = constant Q; > 0 adds q_adapt x the local mean squared one-step change of the released log density
    over +-6 slots), a (speed share of the change), cap (largest |log change| applied)."""
    t = np.asarray(t, np.int64); j = np.asarray(j, np.int64)
    v = np.asarray(v, np.float64); q = np.asarray(q, np.float64)
    if len(t) == 0:
        return v, q
    a0, b0 = _span(t, P.NT)
    tt = t - a0
    T, L = b0 - a0, P.L
    xv, xq = P.X["v"][a0:b0].astype(np.float64), P.X["q"][a0:b0].astype(np.float64)
    rel = np.isfinite(xv) & np.isfinite(xq) & (xv > 0) & (xq > 0)
    Y = np.full((T, L), np.nan)
    Y[rel] = np.log(np.maximum(xq[rel] / xv[rel], LOG_FLOOR))
    R = np.full((T, L), np.inf)
    R[rel] = prm.get("s_o", 0.01) ** 2
    ky = np.log(np.maximum(q / np.maximum(v, 1.0), LOG_FLOOR))
    Y[tt, j] = ky
    sm = np.where(np.asarray(kind) > 0.5, prm.get("s_m_dark", 0.12), prm.get("s_m", 0.04))
    R[tt, j] = sm ** 2
    Q = np.full((T, L), prm.get("q0", 0.002))
    if prm.get("q_adapt", 0.0) > 0:
        d = np.diff(np.where(rel, Y, np.nan), axis=0) ** 2
        d = np.vstack([np.full((1, L), np.nan), d])
        f = np.isfinite(d)
        cs = np.vstack([np.zeros((1, L)), np.cumsum(np.where(f, d, 0), 0)])
        cn = np.vstack([np.zeros((1, L)), np.cumsum(f, 0)])
        w = 6
        lo = np.clip(np.arange(T) - w, 0, T); hi = np.clip(np.arange(T) + w + 1, 0, T)
        loc = (cs[hi] - cs[lo]) / np.maximum(cn[hi] - cn[lo], 1)
        loc = np.where(cn[hi] - cn[lo] > 0, loc, np.nanmedian(np.where(f, d, np.nan)) if f.any() else 0.0)
        Q = Q + prm["q_adapt"] * loc
    R = np.where(np.isfinite(R), R, 1e12)
    X = rts_local_level(Y, R, Q)
    new = X[tt, j]
    dl = np.clip(np.nan_to_num(new - ky), -prm.get("cap", 0.5), prm.get("cap", 0.5))
    a = prm.get("a", 0.25)
    return v * np.exp(-a * dl), q * np.exp((1 - a) * dl)


def lwr_breakdown(P, t, j, v, q, days) -> dict:
    """Where the conservation proxy loses (train holdout: truth everywhere except the predicted cells): S_LWR per
    regime, and per regime the share of |rhs| (the denominator) and of the residual (the numerator) carried by
    transitions touching a blackout target, an isolated regular target, a regular target inside a run, and by
    congested transitions (either end below 0.6 v_f)."""
    rows = np.zeros(P.NT, bool)
    for d in days:
        rows[d * C.SLOTS:(d + 1) * C.SLOTS] = True
    sl = np.flatnonzero(rows)
    a, b = sl.min(), sl.max() + 1
    TV = P.tv[a:b].astype(np.float64); TQ = P.tq[a:b].astype(np.float64)
    Lk = (P.length * P.lanes)[None, :].astype(np.float64)
    Nt = TQ / np.maximum(TV, 1.0) * Lk
    SV = TV.copy(); SQ = TQ.copy()
    SV[t - a, j] = v; SQ[t - a, j] = q
    Ns = SQ / np.maximum(SV, 1.0) * Lk
    tg = np.zeros_like(TV, bool); tg[t - a, j] = True
    dk = np.repeat(P.dark[a:b][:, None], P.L, 1) & tg
    run = np.zeros_like(tg)
    run[1:] |= tg[1:] & tg[:-1]; run[:-1] |= tg[:-1] & tg[1:]
    el = P.el[a:b] & np.isfinite(TV) & np.isfinite(TQ)
    ok = el[:-1] & el[1:] & rows[a:b][:-1, None] & rows[a:b][1:, None] & P.rv[a:b][:-1] & P.observable[None, :]
    reg = np.repeat(P.dayreg, C.SLOTS)[a:b]
    cong = (TV < 0.6 * P.vf[None, :])
    dNt = Nt[1:] - Nt[:-1]; res = np.abs((Ns[1:] - Ns[:-1]) - dNt)
    touch = lambda m: m[:-1] | m[1:]
    cats = {"dark": touch(dk), "isolated": touch(tg & ~run & ~dk), "run": touch(tg & run & ~dk)}
    out = {}
    for r in (1, 2, 3):
        m = ok & (reg[:-1] == r)[:, None]
        den = np.abs(dNt[m]).sum()
        if den <= 0:
            continue
        o = dict(S_LWR=1 - min(1.0, res[m].sum() / den))
        for k_, c in cats.items():
            mc = m & c
            o[f"den_{k_}"] = float(np.abs(dNt[mc]).sum() / den); o[f"res_{k_}"] = float(res[mc].sum() / den)
        mc = m & touch(cong)
        o["den_congested"] = float(np.abs(dNt[mc]).sum() / den); o["res_congested"] = float(res[mc].sum() / den)
        out[f"R{r}"] = o
    return out


DEFAULT_GRID = (
    dict(name="ks_a", s_o=0.01, s_m=0.04, s_m_dark=0.12, q0=0.002, q_adapt=0.0, a=0.25),
    dict(name="ks_b", s_o=0.01, s_m=0.03, s_m_dark=0.10, q0=0.002, q_adapt=0.5, a=0.25),
    dict(name="ks_c", s_o=0.01, s_m=0.05, s_m_dark=0.15, q0=0.001, q_adapt=0.5, a=0.25),
    dict(name="ks_d", s_o=0.01, s_m=0.04, s_m_dark=0.12, q0=0.004, q_adapt=1.0, a=0.25),
    dict(name="ks_e", s_o=0.01, s_m=0.04, s_m_dark=0.12, q0=0.002, q_adapt=0.5, a=0.0),
    dict(name="ks_f", s_o=0.01, s_m=0.04, s_m_dark=0.12, q0=0.002, q_adapt=0.5, a=0.5),
    dict(name="ks_g", s_o=0.03, s_m=0.04, s_m_dark=0.12, q0=0.002, q_adapt=0.5, a=0.25),
    dict(name="ks_h", s_o=0.01, s_m=0.08, s_m_dark=0.2, q0=0.001, q_adapt=0.25, a=0.25),
)
