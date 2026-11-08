"""v12 diagnostics and the leaderboard snapshot (nothing here feeds a decision of the pipeline).

stations: Task 1 is scored per detector station, while the pipeline works per link (several stations on one link are
  averaged into one value and every station of the link gets that value). On train truth: how many links carry more
  than one station, the share of the Task 1 target rows on them, and whether the stations of a link differ by a stable
  offset (median per station over the first half of the sampled days) that would carry over to the second half - the
  RMSE an offset correction would remove, in S_state units.
leaderboard: the public leaderboard CSV (Kaggle CLI download) summarised - post-rebuild ranks (teams whose last entry is
  older than the rebuilt dataset of 2026-09-11 cannot hold a valid score), our rank, the score distribution and the
  entry counts of the teams above us.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import common as C
from .data import pfiles, read_pq

VERSION = "v12"
REBUILD_UTC = pd.Timestamp("2026-09-11 16:30:00")
# pre-rebuild scores the organizers could not remove (named in the public notes of other teams, 2026-10-09)
LEAK_ERA = ("Pardheev Krishna", "Musa Peker", "Jason Karpeles", "Kushal Khemani", "Kanak Khemani", "minghao mei")


# ----------------------------------------------------------------------------- stations per link
def _lanes(panel: str) -> tuple:
    """({station: lanes}, {link: lanes}) from fd_parameters.csv (the scorer's per-station lane count; the link's mean
    for a station the file does not list)."""
    f = C.rel() / "corridors" / panel / "network" / "fd_parameters.csv"
    if not f.exists():
        return {}, {}
    fd = pd.read_csv(f, dtype={"station_id": str, "link_id": str})
    if "lanes" not in fd:
        return {}, {}
    fd["lanes"] = pd.to_numeric(fd.lanes, errors="coerce").clip(lower=1.0)
    st = dict(zip(fd.station_id.astype(str), fd.lanes)) if "station_id" in fd else {}
    lk = fd.groupby(fd.link_id.astype(str)).lanes.mean().to_dict() if "link_id" in fd else {}
    return {k: v for k, v in st.items() if np.isfinite(v)}, {k: v for k, v in lk.items() if np.isfinite(v)}


def panel_stations(panel: str, days: int = 28) -> dict:
    """Train truth of the last `days` train days of one panel, per station (see the module docstring)."""
    base = C.rel() / "corridors" / panel / "train" / "mainline_states"
    files = pfiles(base)
    out = dict(panel=panel)
    if not files:
        out["error"] = "no train mainline_states"
        return out
    files = files[-days:]
    cols = ["timestamp", "station_id", "link_id", "speed_kmh", "flow_vph", "is_score_eligible"]
    d = pd.concat([read_pq(f, cols) for f in files], ignore_index=True)
    d["station_id"] = d.station_id.astype(str); d["link_id"] = d.link_id.astype(str)
    per_link = d.groupby("link_id").station_id.nunique()
    multi = per_link[per_link > 1].index
    out.update(links=int(len(per_link)), multi_station_links=int(len(multi)),
               stations=int(d.station_id.nunique()), stations_on_multi=int(d[d.link_id.isin(multi)].station_id.nunique()))
    tp = C.rel() / "task1" / panel / "validation" / "sample_submission_state.csv"
    if tp.exists():
        tm = pd.read_csv(tp, usecols=["link_id"], dtype=str)
        out["template_rows"] = int(len(tm))
        out["template_rows_on_multi"] = int(tm.link_id.isin(multi).sum())
    if len(multi) == 0:
        return out
    s_lanes, l_lanes = _lanes(panel)
    x = d[d.link_id.isin(multi) & d.is_score_eligible.fillna(False).astype(bool)].copy()
    x["v"] = pd.to_numeric(x.speed_kmh, errors="coerce")
    ln = x.station_id.map(s_lanes).fillna(x.link_id.map(l_lanes)).fillna(1.0)
    x["ql"] = pd.to_numeric(x.flow_vph, errors="coerce") / ln
    x = x.dropna(subset=["v", "ql"])
    g = x.groupby(["timestamp", "link_id"])
    x["n"] = g.station_id.transform("nunique")
    x = x[x.n > 1]
    if x.empty:
        return out
    x["dv"] = x.v - g.v.transform("mean").reindex(x.index)
    x["dq"] = x.ql - g.ql.transform("mean").reindex(x.index)
    ts = pd.to_datetime(x.timestamp, utc=True)
    half = ts < ts.min() + (ts.max() - ts.min()) / 2
    a, b = x[half.to_numpy()], x[~half.to_numpy()]
    for ch in ("dv", "dq"):
        off = a.groupby("station_id")[ch].median()
        rb = b[ch].to_numpy(); corr = rb - b.station_id.map(off).fillna(0.0).to_numpy()
        out[f"{ch}_sd"] = float(np.std(rb)) if len(rb) else float("nan")
        out[f"{ch}_offset_abs_median"] = float(off.abs().median()) if len(off) else float("nan")
        out[f"{ch}_rmse_before"] = float(np.sqrt(np.mean(rb ** 2))) if len(rb) else float("nan")
        out[f"{ch}_rmse_after_offset"] = float(np.sqrt(np.mean(corr ** 2))) if len(rb) else float("nan")
    return out


def stations(panels=None, days: int = 28) -> dict:
    rows = [panel_stations(p, days) for p in (panels or C.panels())]
    tot = {k: int(sum(r.get(k, 0) for r in rows)) for k in ("links", "multi_station_links", "template_rows",
                                                             "template_rows_on_multi")}
    share = tot["template_rows_on_multi"] / max(tot["template_rows"], 1)
    res = dict(panels={r["panel"]: r for r in rows}, total=tot, share_template_rows_on_multi=share)
    # what an offset correction could remove from S_state (the rows on multi-station links only; their RMSE part that
    # the offset explains, through the S_state slopes 0.54/25 per km/h and 0.46/600 per vph/lane)
    gains = []
    for r in rows:   # every panel counts (S_state averages the corridors): 0 where no link has several stations
        g = 0.0
        if "dv_rmse_before" in r and r.get("template_rows"):
            w = r.get("template_rows_on_multi", 0) / max(r["template_rows"], 1)
            gv = max(0.0, r["dv_rmse_before"] - r["dv_rmse_after_offset"]) * 0.54 / 25
            gq = max(0.0, r["dq_rmse_before"] - r["dq_rmse_after_offset"]) * 0.46 / 600
            g = w * (np.nan_to_num(gv) + np.nan_to_num(gq))
        gains.append(g)
    res["s_state_gain_upper_bound"] = float(np.mean(gains)) if gains else 0.0
    return res


def run(force: bool = False) -> dict:
    """WORK/diag12/diag12.json (reused for the same code and release)."""
    from .data import fingerprint
    d = C.WORK / "diag12"
    d.mkdir(parents=True, exist_ok=True)
    f = d / "diag12.json"
    key = dict(version=VERSION, code=hashlib.md5(Path(__file__).read_bytes()).hexdigest()[:12], fingerprint=fingerprint())
    if f.exists() and not force:
        try:
            rep = json.load(open(f))
            if rep.get("key") == key:
                return rep
        except Exception:  # noqa: BLE001
            pass
    t0 = time.time()
    rep = dict(key=key, stations=stations())
    rep["minutes"] = round((time.time() - t0) / 60, 1)
    tmp = d / "diag12.json.part"
    json.dump(rep, open(tmp, "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    tmp.replace(f)
    C.log(f"diag12 in {rep['minutes']} min: multi-station links {rep['stations']['total']} | share of template rows "
          f"{rep['stations']['share_template_rows_on_multi']:.4f} | S_state gain bound {rep['stations']['s_state_gain_upper_bound']:.5f}")
    return rep


# ----------------------------------------------------------------------------- leaderboard
def read_leaderboard(path) -> pd.DataFrame:
    """The public leaderboard CSV (or the zip the Kaggle CLI downloads) as a frame with Rank, TeamName, Score,
    LastSubmissionDate (datetime), SubmissionCount, TeamMemberUserNames."""
    path = Path(path)
    if path.suffix == ".zip":
        import zipfile
        with zipfile.ZipFile(path) as z:
            name = [n for n in z.namelist() if n.endswith(".csv")][0]
            with z.open(name) as fh:
                df = pd.read_csv(fh)
    else:
        df = pd.read_csv(path)
    df.columns = [c.strip().lstrip("﻿") for c in df.columns]
    if "Score" in df:
        df["Score"] = pd.to_numeric(df.Score, errors="coerce")
    if "LastSubmissionDate" in df:
        df["LastSubmissionDate"] = pd.to_datetime(df.LastSubmissionDate, errors="coerce", utc=True).dt.tz_convert(None)
    if "SubmissionCount" in df:
        df["SubmissionCount"] = pd.to_numeric(df.SubmissionCount, errors="coerce")
    return df.sort_values("Score", ascending=False).reset_index(drop=True)


def leaderboard_summary(df: pd.DataFrame, me=("biohack44", "Emre Cirak"), top: int = 30) -> dict:
    """Post-rebuild ranking (rows whose last entry predates the rebuild, or named leak-era rows, are set aside), our
    place in it, the score levels of ranks 1/3/10/20/50 and the entry counts above us."""
    df = df.copy()
    if "TeamMemberUserNames" not in df:
        df["TeamMemberUserNames"] = ""
    if "TeamName" not in df:
        df["TeamName"] = ""
    df["TeamMemberUserNames"] = df.TeamMemberUserNames.fillna("").astype(str)
    df["TeamName"] = df.TeamName.fillna("").astype(str)
    old = (df.LastSubmissionDate < REBUILD_UTC).to_numpy() if "LastSubmissionDate" in df else np.zeros(len(df), bool)
    df["leak_era"] = old | df.TeamName.isin(LEAK_ERA).to_numpy()
    post = df[~df.leak_era].reset_index(drop=True)
    post["post_rank"] = np.arange(1, len(post) + 1)
    pm = post[post.TeamName.eq(me[1]) | post.TeamMemberUserNames.str.contains(me[0], regex=False)]
    out = dict(teams=int(len(df)), leak_era_rows=int(df.leak_era.sum()), post_rebuild_teams=int(len(post)))
    if len(pm):
        r = int(pm.post_rank.iat[0]); s = float(pm.Score.iat[0])
        out.update(my_post_rank=r, my_score=s, my_entries=int(pm.SubmissionCount.iat[0]) if "SubmissionCount" in pm else None)
        above = post[post.post_rank < r]
        if "SubmissionCount" in above and len(above):
            out["entries_above_me"] = dict(median=float(above.SubmissionCount.median()),
                                           share_ge_50=float((above.SubmissionCount >= 50).mean()))
    for k in (1, 3, 10, 20, 50, 100):
        if len(post) >= k:
            out[f"post_rank_{k}"] = float(post.Score.iat[k - 1])
    for th in (0.90, 0.895, 0.89, 0.885, 0.88, 0.875, 0.87):
        out[f"n_ge_{th}"] = int((post.Score >= th).sum())
    cols = [c for c in ("post_rank", "TeamName", "Score", "SubmissionCount", "LastSubmissionDate") if c in post]
    tb = post[cols].head(top).copy()
    if "LastSubmissionDate" in tb:
        tb["LastSubmissionDate"] = tb.LastSubmissionDate.astype(str)
    out["top"] = tb.to_dict("records")
    out["leak_era"] = df[df.leak_era][["TeamName", "Score"]].head(20).to_dict("records")
    return out
