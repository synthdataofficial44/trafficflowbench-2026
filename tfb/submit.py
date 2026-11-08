"""Merge the three task outputs into the upload file (template order), validate it, zip it, submit it."""
from __future__ import annotations

import subprocess
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from . import common as C

COMP = "2026-ieee-big-data-traffic-flow-bench"


def key_table() -> pd.DataFrame:
    cache = C.WORK / "submission_key.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    k = pd.read_csv(C.rel() / "submission_key.csv", dtype=str, keep_default_na=False)
    k["submission_id"] = k.submission_id.astype(np.int64)
    k.to_parquet(cache, index=False)
    return k


def _join(keys_left: pd.Series, keys_right: pd.Series) -> np.ndarray:
    idx = pd.Index(keys_right.to_numpy())
    if idx.has_duplicates:
        raise ValueError(f"{int(idx.duplicated().sum())} duplicate keys in a task file")
    return idx.get_indexer(keys_left.to_numpy())


def build(state: pd.DataFrame, queue: pd.DataFrame, odme: pd.DataFrame, out_csv: Path,
          drop_tasks: tuple = ()) -> pd.DataFrame:
    """drop_tasks: tasks to leave at 0 (probe submissions that isolate one task's leaderboard score)."""
    k = key_table()
    tmpl = pd.read_csv(C.rel() / "sample_submission.csv", usecols=["submission_id", "task"])
    k = k.set_index("submission_id").loc[tmpl.submission_id.to_numpy()].reset_index()
    out = pd.DataFrame({"submission_id": k.submission_id.to_numpy(), "task": k.task.to_numpy(),
                        "speed_kmh": 0.0, "flow_vph": 0.0, "queue_pred": 0, "path_flow": 0.0})
    rep = {}
    m = (k.task == "state").to_numpy()
    if "state" not in drop_tasks:
        st = state.copy()
        kl = k.panel[m] + "|" + k.timestamp[m] + "|" + k.link_id[m]
        kr = st.panel.astype(str) + "|" + st.timestamp.astype(str) + "|" + st.link_id.astype(str)
        pos = _join(kl, kr)
        rep["state_unmatched"] = int((pos < 0).sum())
        sp = st.speed_kmh.to_numpy(float); fl = st.flow_vph.to_numpy(float)
        out.loc[m, "speed_kmh"] = np.where(pos >= 0, sp[np.maximum(pos, 0)], 100.0)
        out.loc[m, "flow_vph"] = np.where(pos >= 0, fl[np.maximum(pos, 0)], 3000.0)
    m = (k.task == "queue").to_numpy()
    if "queue" not in drop_tasks:
        kl = k.window_id[m] + "|" + k.timestamp[m] + "|" + k.link_id[m]
        kr = queue.window_id.astype(str) + "|" + queue.timestamp.astype(str) + "|" + queue.link_id.astype(str)
        pos = _join(kl, kr)
        rep["queue_unmatched"] = int((pos < 0).sum())
        qp = queue.queue_pred.to_numpy().astype(int)
        out.loc[m, "queue_pred"] = np.where(pos >= 0, qp[np.maximum(pos, 0)], 0)
    m = (k.task == "odme").to_numpy()
    if "odme" not in drop_tasks:
        kl = k.panel[m] + "|" + k.departure_time[m] + "|" + k.path_id[m]
        kr = odme.panel.astype(str) + "|" + odme.departure_time.astype(str) + "|" + odme.path_id.astype(str)
        pos = _join(kl, kr)
        rep["odme_unmatched"] = int((pos < 0).sum())
        pf = odme.path_flow.to_numpy(float)
        out.loc[m, "path_flow"] = np.where(pos >= 0, pf[np.maximum(pos, 0)], 0.0)
    bad = {kk: v for kk, v in rep.items() if v}
    if bad:
        raise RuntimeError(f"unmatched submission rows {bad}: a task file does not cover the template (check keys)")
    check(out, k, drop_tasks)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False, float_format="%.4f")
    zp = out_csv.with_suffix(".zip")
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(out_csv, arcname=out_csv.name)
    C.log(f"submission {out_csv.name}: {len(out):,} rows, {rep}, zip {zp.stat().st_size / 1e6:.0f} MB")
    return out


def check(out: pd.DataFrame, k: pd.DataFrame, drop_tasks=()):
    assert out.submission_id.is_unique, "duplicate submission_id"
    assert not out.isna().any().any(), "NaN in submission"
    num = out[["speed_kmh", "flow_vph", "path_flow"]].to_numpy()
    assert np.isfinite(num).all(), "non-finite values"
    assert set(np.unique(out.queue_pred)) <= {0, 1}, "queue_pred must be 0/1"
    assert (out.path_flow >= 0).all(), "negative path_flow"
    st = (out.task == "state").to_numpy()
    if "state" not in drop_tasks:
        assert (out.speed_kmh[st] > 0).all() and (out.flow_vph[st] >= 0).all()
        low = pd.Series(out.flow_vph[st].to_numpy() < 50).groupby([k.panel[st].to_numpy(), k.mask_regime[st].to_numpy()]).mean()
        assert (low <= 0.2).all(), f"empty-road guard would fire: {low[low > 0.2]}"
    q = out[out.task == "queue"]
    if "queue" not in drop_tasks:
        per_w = q.groupby(k.window_id[out.task == "queue"].to_numpy()).queue_pred.sum()
        if (per_w == 0).any():
            C.log(f"WARNING {int((per_w == 0).sum())} queue windows have no predicted cell")


def kaggle_submit(path: Path, message: str) -> bool:
    """True when the Kaggle CLI confirms the upload (a refused upload, e.g. the daily cap, returns False, never raises)."""
    cmd = ["kaggle", "competitions", "submit", "-c", COMP, "-f", str(path), "-m", message[:500]]
    C.log("submitting:", " ".join(cmd[:6]), path.name)
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    except Exception as e:  # network trouble: report, do not stop the notebook
        print("submission did not complete:", type(e).__name__, e)
        return False
    print(r.stdout[-2000:], r.stderr[-2000:])
    return r.returncode == 0 and "Successfully submitted" in (r.stdout + r.stderr)


def kaggle_scores(n: int = 8):
    r = subprocess.run(["kaggle", "competitions", "submissions", "-c", COMP, "-v"], capture_output=True, text=True)
    lines = r.stdout.strip().splitlines()
    print("\n".join(lines[: n + 1]))
    return r.stdout
