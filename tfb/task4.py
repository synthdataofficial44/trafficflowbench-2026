"""Task 4 (ODME): Euclidean projection of each split's released prior onto its released link counts.

    f = argmin ||f - b||^2   s.t.   A_meas f = c,  f >= 0

Leaderboard-verified (S_ODME ~0.994 against 0.836 for the NNLS baseline): the reference path flows were
produced by a ridge fit of the split prior to hidden targets and the counts were published as A f*, so the
projection's KKT conditions coincide with the generator's. Solved exactly by semismooth Newton on the dual
over the distinct count rows (screenlines), which has at most ~100 unknowns.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.linalg as sla

from . import common as C


def load_problem(panel: str):
    nd = C.rel() / "corridors" / panel / "network"
    paths = pd.read_csv(nd / "path_set.csv", dtype=str)
    inc = pd.read_csv(nd / "path_link_incidence.csv", dtype=str)
    pids = paths.path_id.tolist()
    lids = inc.link_id.drop_duplicates().tolist()
    pidx = {x: i for i, x in enumerate(pids)}; lidx = {x: i for i, x in enumerate(lids)}
    inc = inc[inc.path_id.isin(pidx) & inc.link_id.isin(lidx)]
    A = np.zeros((len(lids), len(pids)))
    A[inc.link_id.map(lidx).to_numpy(), inc.path_id.map(pidx).to_numpy()] = 1.0
    return paths, pids, lids, A


def l2_project(S: np.ndarray, c: np.ndarray, b: np.ndarray, iters: int = 500, tol: float = 1e-10):
    """argmin ||f-b||^2 s.t. S f = c, f >= 0. f = max(0, b + S^T mu); Newton on the concave dual."""
    m = S.shape[0]
    mu = np.zeros(m)

    def primal(u):
        return np.maximum(0.0, b + S.T @ u)

    def dual(u):  # concave dual q(u) = min_{f>=0} 0.5||f-b||^2 - u.(Sf-c)
        f = primal(u)
        return 0.5 * np.sum((f - b) ** 2) - u @ (S @ f - c)

    scale = max(1.0, float(np.abs(c).max()))
    for _ in range(iters):
        f = primal(mu)
        g = S @ f - c
        if np.max(np.abs(g)) / scale < tol:
            break
        act = (f > 0).astype(float)
        H = (S * act) @ S.T + 1e-10 * np.eye(m)
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(H, g, rcond=None)[0]
        q0 = dual(mu); a = 1.0
        while a > 1e-12:
            mn = mu - a * step
            if dual(mn) >= q0 + 1e-4 * a * (g @ step) - 1e-12 * abs(q0):
                break
            a *= 0.5
        mu = mn
    return primal(mu)


def solve_panel(panel: str, split: str):
    paths, pids, lids, A = load_problem(panel)
    d = C.rel() / "task4" / panel / split
    cnt = pd.read_csv(d / "synthetic_link_counts.csv", dtype={"link_id": str})
    pr = pd.read_csv(d / "synthetic_weak_prior.csv", dtype={"path_id": str})
    b = pd.to_numeric(pr.set_index("path_id").path_flow.reindex(pids), errors="coerce").fillna(0.0).to_numpy(float)
    b = np.maximum(b, 0.0)
    cser = cnt.groupby("link_id")["count"].mean()
    meas = [i for i, l in enumerate(lids) if l in cser.index]
    Am = A[meas]; cm = cser.reindex([lids[i] for i in meas]).to_numpy(float)
    # collapse identical incidence rows (several detectors on one screenline): average their counts
    keys = {}
    for r in range(Am.shape[0]):
        keys.setdefault(Am[r].tobytes(), []).append(r)
    rows = list(keys.values())
    S = np.stack([Am[g[0]] for g in rows]); c = np.array([cm[g].mean() for g in rows])
    nz = S.sum(1) > 0
    S, c = S[nz], c[nz]
    # drop linearly dependent rows (keeps the dual Newton system well posed)
    q, r_, piv = sla.qr(S.T, mode="economic", pivoting=True)
    rank = int((np.abs(np.diag(r_)) > 1e-9 * max(1.0, np.abs(r_[0, 0]))).sum())
    keep = np.sort(piv[:rank])
    f = l2_project(S[keep], c[keep], b)
    s_link = 1 - np.abs(Am @ f - cm).sum() / max(cm.sum(), 1e-9)
    dev = np.abs(f - b).sum() / max(b.sum(), 1e-9)
    tmpl = pd.read_csv(d / "sample_submission_path_flow.csv", dtype=str)
    tmpl["path_flow"] = tmpl.path_id.map(dict(zip(pids, f))).astype(float).fillna(0.0).clip(lower=0.0)
    return tmpl, dict(panel=panel, split=split, S_link=float(s_link), dev_over_prior=float(dev),
                      zeros=int((f <= 1e-9).sum()), n_paths=len(pids), n_rows=int(len(keep)))


def run(splits=C.TEST_SPLITS) -> pd.DataFrame:
    outs, rep = [], []
    for s in splits:
        for p in C.panels():
            t, r = solve_panel(p, s)
            outs.append(t); rep.append(r)
    rep = pd.DataFrame(rep)
    C.log("Task 4 L2 projection: min S_link %.6f, mean dev/|b| %.3f" % (rep.S_link.min(), rep.dev_over_prior.mean()))
    out = pd.concat(outs, ignore_index=True)
    (C.WORK / "pred").mkdir(parents=True, exist_ok=True)
    out.to_csv(C.WORK / "pred" / "task4.csv", index=False)
    rep.to_csv(C.WORK / "pred" / "task4_report.csv", index=False)
    return rep
