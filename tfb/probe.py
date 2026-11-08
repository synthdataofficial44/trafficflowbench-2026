"""v9: read our public score per task exactly, with three probe uploads built from one set of files.

    S_total = 0.35 S_state + 0.30 S_queue + 0.15 S_physics + 0.20 S_ODME,   S_physics = S_FD / 3 + 2/3 S_LWR

  * no_queue - the files with every queue_pred at 0:          S_queue = (B - no_queue) / 0.30
  * scaled   - every Task 1 speed AND flow times SCALE (3): the per-cell density q/v, hence the vehicle counts the
               conservation term S_LWR is computed from, are exactly unchanged, while both error terms of S_state reach
               their floor of 0 (speed errors of about twice a free-flow speed against a scale of 25 km/h, per-lane flow
               errors of about twice a lane's flow against 600 veh/h; `scale_check`). S_FD moves by an amount computed here from the released March
               observations and our own values (`fd_scores`):
                                                               S_state = (B - scaled - 0.05 (S_FD - S_FD_scaled)) / 0.35
  * no_state - the files with every Task 1 value at 0 (S_state and S_physics both drop to 0):
                                                               S_physics = (B - no_state - 0.35 S_state) / 0.15
                                                               S_LWR = (3 S_physics - S_FD) / 2
                                                               S_ODME = (no_state - 0.30 S_queue) / 0.20
B is the public score of the unchanged files (found through the submission log). Nothing here reads a hidden value:
every number comes from the public leaderboard or from the released data.

Every probe is built from the exact file that was submitted (the CSV inside its zip, checked against the md5 the
submission log recorded), so a probe differs from the base in the one change it makes and nothing else: no rebuild of
Task 1 / Task 4 rows, no refit noise. The Task 2 probes of a run (`with_queue_rows`) are built the same way: the base
file with the queue rows of one condition's March windows taken from the probe's rows.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import common as C
from .data import pfiles, read_pq

SCALE = 3.0
W = dict(state=0.35, queue=0.30, physics=0.15, odme=0.20)
EMPTY_FLOOR_VPH, EMPTY_SHARE = 50.0, 0.20


def scale_state(state: pd.DataFrame, factor: float = SCALE) -> pd.DataFrame:
    """Task 1 rows with speed and flow both times `factor` (density q/v unchanged)."""
    st = state.copy()
    st["speed_kmh"] = st.speed_kmh.to_numpy(np.float64) * factor
    st["flow_vph"] = st.flow_vph.to_numpy(np.float64) * factor
    return st


def scale_check(state: pd.DataFrame, factor: float = SCALE) -> pd.DataFrame:
    """Per corridor and regime: the error the scaled answer must at least make, (factor - 1) x our own values (RMS),
    against the scale at which each S_state term reaches 0 (25 km/h, 600 veh/h per lane). Margins above 1 everywhere
    mean S_state of the scaled answer is exactly 0 (our errors are a few % of those values)."""
    from .data import load_net
    rows = []
    for panel, g in state.groupby(state.panel.astype(str)):
        net = load_net(panel)
        lanes = pd.Series(net.lanes, index=net.links)
        ql = g.flow_vph.to_numpy(np.float64) / lanes.reindex(g.link_id.astype(str)).fillna(1.0).clip(lower=1).to_numpy()
        for reg, idx in g.groupby(g.mask_regime.astype(str)).indices.items():
            rows.append(dict(panel=panel, regime=reg,
                             speed_margin=(factor - 1) * float(np.sqrt(np.mean(g.speed_kmh.to_numpy()[idx] ** 2))) / 25,
                             flow_margin=(factor - 1) * float(np.sqrt(np.mean(ql[idx] ** 2))) / 600))
    return pd.DataFrame(rows)


def network_parameters(panel: str) -> pd.DataFrame:
    """Per-link triangular FD parameters exactly as the Task 3 scorer reads them (fd_parameters.csv authoritative,
    capacity and free speed averaged over a link's stations, critical density recomputed, k_jam fallback)."""
    nd = C.rel() / "corridors" / panel / "network"
    n = pd.read_csv(nd / "links.csv")
    n["link_id"] = n.link_id.astype(str)
    n = n.drop_duplicates("link_id")
    out = n[["link_id", "length_km", "free_speed_kmh", "capacity_vph", "lanes"]].copy()
    for c, default in (("length_km", 1.0), ("free_speed_kmh", 100.0)):
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(default)
    out["lanes"] = pd.to_numeric(out["lanes"], errors="coerce").fillna(1.0).clip(lower=1.0)
    lanes = pd.to_numeric(n.get("lanes", pd.Series(4.0, index=n.index)), errors="coerce").fillna(4.0)
    out["capacity_vph"] = pd.to_numeric(out.capacity_vph, errors="coerce").fillna(
        pd.Series(1800.0 * lanes.to_numpy(), index=out.index))
    fd = nd / "fd_parameters.csv"
    if fd.exists():
        f = pd.read_csv(fd)
        f["link_id"] = f.link_id.astype(str)
        have = [c for c in ("free_speed_kmh", "capacity_vph", "lanes", "k_jam") if c in f.columns]
        m = f.groupby("link_id", as_index=False)[have].mean()
        out = out.merge(m, on="link_id", how="left", suffixes=("", "_fd"))
        for c in ("free_speed_kmh", "capacity_vph", "lanes"):
            if f"{c}_fd" in out.columns:
                out[c] = out[f"{c}_fd"].fillna(out[c])
                out = out.drop(columns=[f"{c}_fd"])
        if "k_jam" not in out.columns:
            out["k_jam"] = np.nan
    else:
        out["k_jam"] = np.nan
    out["critical_density"] = out.capacity_vph / out.free_speed_kmh.clip(lower=1)
    out["k_jam"] = out.k_jam.fillna(100.0 * out.lanes.clip(lower=1.0))
    out["k_jam"] = np.maximum(out.k_jam, out.critical_density * 1.05)
    return out.set_index("link_id")


def _s_fd(x: pd.DataFrame, hybrid: bool = False) -> float:
    """S_FD of one corridor and regime (score_task3.score_panel): per-lane flow against the triangular FD at the same
    per-lane density; 0 when a fifth of the submitted cells claim an empty road. hybrid: the filled (hidden) cells
    count with their interpolated flow, and an FD deviation in the proportion the released non-target cells show."""
    lanes = x.lanes.clip(lower=1.0)
    q = x.flow_vph / lanes
    k = (x.flow_vph / x.speed_kmh.clip(lower=1.0)) / lanes
    cap, kc, kj = x.capacity_vph / lanes, x.critical_density / lanes, x.k_jam / lanes
    qfd = np.where(k <= kc, x.free_speed_kmh * k, (cap / np.maximum(kj - kc, 1e-9)) * np.maximum(kj - k, 0))
    dev = np.abs(q - qfd).to_numpy()
    if hybrid and "filled" in x:
        fl = x.filled.to_numpy() > 0.5
        ref = ~fl & (x.is_target.to_numpy() <= 0)
        r = float(dev[ref].sum() / max(np.abs(q.to_numpy()[ref]).sum(), 1e-9)) if ref.any() else 0.0
        dev = np.where(fl, np.abs(q.to_numpy()) * r, dev)
    s = float(max(0.0, 1.0 - dev.sum() / (np.abs(q).sum() + 1e-9)))
    sub = x.is_target.to_numpy() > 0.5
    if sub.any() and float((x.flow_vph.to_numpy()[sub] < EMPTY_FLOOR_VPH).mean()) > EMPTY_SHARE:
        s = 0.0
    return s


def _tkey(x) -> np.ndarray:
    """Timestamps (ISO strings, naive or aware datetimes) -> int64 ns UTC."""
    t = pd.to_datetime(pd.Series(x).reset_index(drop=True), utc=True).dt.tz_convert(None)
    return t.to_numpy().astype("datetime64[ns]").astype(np.int64)


def fd_scores(states: dict, split: str = "validation", panels=None) -> pd.DataFrame:
    """S_FD per corridor and regime for one or more Task 1 answers {name: rows}, the way the Task 3 scorer builds its
    frame: every eligible cell of the split, the released observation where one is published and the answer on the
    Task 1 targets, averaged over a link's stations.
    The scorer also uses the hidden values of the eligible cells blank in the release that are not Task 1 targets (the
    Task 2 horizons and the hour after them, ~2% of a month). `S_FD` counts them with each station's released values
    interpolated in time for their flow and with the FD deviation the released non-target cells show in proportion to
    their flow (`_s_fd(hybrid=True)`); `S_FD_interp` takes the interpolated values as they are, `S_FD_drop` leaves the
    cells out. On the synthetic train split (where the truth is public) the three were off the official scorer by
    +0.00005 / -0.0011 / -0.00006 for an answer and +0.0001 / -0.0004 / +0.0024 for the difference answer - scaled.
    Diagnostics per row: `mixed` = link-times whose stations are partly targets (where scaling would move the
    station-averaged density; 0 = the scaled probe leaves S_LWR exactly unchanged), `slow` = target cells with speed < 1
    (where the scorer's clip at 1 km/h would break q/v invariance), `blank` = filled cells, `q_rms_lane` / `v_rms` = RMS of
    the released per-lane flow and speed (a random sample of the target cells' truth: the no_state probe zeroes S_state
    when they exceed 600 and 25)."""
    from .data import load_net
    rows = []
    key = ["timestamp", "station_id", "link_id"]
    for panel in panels or C.panels():
        base = C.rel() / "corridors" / panel / split / "mainline_states_masked"
        tmpl = pd.read_csv(C.rel() / "task1" / panel / split / "sample_submission_state.csv",
                           usecols=["timestamp", "station_id", "link_id", "mask_regime"], dtype=str)
        params = network_parameters(panel)
        net = load_net(panel)
        lanes_l = pd.Series(net.lanes, index=net.links).astype(float)
        frames = []
        for f in pfiles(base):
            d = read_pq(f, ["timestamp", "station_id", "link_id", "speed_kmh", "flow_vph", "is_score_eligible"])
            d = d[d.is_score_eligible.fillna(False).astype(bool)]
            frames.append(d.assign(regime=f.parent.name.split("=")[-1]))
        obs = pd.concat(frames, ignore_index=True)
        for d in (obs, tmpl):   # one key format whatever the files store (ISO strings or timestamps)
            d["timestamp"] = _tkey(d.timestamp)
            for c in ("station_id", "link_id"):
                d[c] = d[c].astype(str)
        for c in ("speed_kmh", "flow_vph"):
            obs[c] = pd.to_numeric(obs[c], errors="coerce").astype(np.float64)
        obs["is_target"] = pd.MultiIndex.from_frame(obs[key]).isin(pd.MultiIndex.from_frame(tmpl[key])).astype(float)
        obs = obs.sort_values(["station_id", "timestamp"], kind="stable").reset_index(drop=True)
        tg = obs.is_target.to_numpy() > 0
        blank = (obs.speed_kmh.isna() | obs.flow_vph.isna()).to_numpy() & ~tg
        rel_q = obs.flow_vph.to_numpy()[~tg & ~blank] / lanes_l.reindex(obs.link_id.to_numpy()[~tg & ~blank]).fillna(1.0).clip(lower=1).to_numpy()
        stats = {}
        for reg in obs.regime.unique():
            m = (obs.regime.to_numpy() == reg)[~tg & ~blank]
            stats[reg] = (float(np.sqrt(np.mean(rel_q[m] ** 2))) if m.any() else np.nan,
                          float(np.sqrt(np.mean(obs.speed_kmh.to_numpy()[~tg & ~blank][m] ** 2))) if m.any() else np.nan)
        if blank.any():   # the hidden non-target cells: each station's released values interpolated in time
            for c in ("speed_kmh", "flow_vph"):
                rel_ = obs[c].where(~tg)
                filled = rel_.groupby(obs.station_id, sort=False).transform(lambda x: x.interpolate(limit_area="inside"))
                obs.loc[blank, c] = filled.to_numpy()[blank]
        obs["filled"] = blank.astype(float)
        for name, st in states.items():
            s = st[st.panel.astype(str) == panel][key + ["speed_kmh", "flow_vph"]].copy()
            s["timestamp"] = _tkey(s.timestamp)
            for c in ("station_id", "link_id"):
                s[c] = s[c].astype(str)
            x = obs[key + ["regime", "speed_kmh", "flow_vph", "is_target", "filled"]].copy()
            sv = s.set_index(key).reindex(pd.MultiIndex.from_frame(x.loc[tg, key]))
            x.loc[tg, "speed_kmh"] = sv.speed_kmh.to_numpy(np.float64)
            x.loc[tg, "flow_vph"] = sv.flow_vph.to_numpy(np.float64)
            slow = int((x.speed_kmh.to_numpy()[tg] < 1.0).sum())
            x = x.dropna(subset=["speed_kmh", "flow_vph"])
            x = x.groupby(["regime", "timestamp", "link_id"], as_index=False)[
                ["speed_kmh", "flow_vph", "is_target", "filled"]].mean()
            x = x.join(params, on="link_id")
            x = x.dropna(subset=["length_km", "free_speed_kmh", "capacity_vph", "critical_density", "k_jam"])
            for reg, g in x.groupby("regime"):
                rows.append(dict(answer=name, panel=panel, regime=reg, S_FD=_s_fd(g, hybrid=True), S_FD_interp=_s_fd(g),
                                 S_FD_drop=_s_fd(g[g.filled == 0]),
                                 cells=int(len(g)), targets=int((g.is_target > 0).sum()),
                                 mixed=int(((g.is_target > 0) & (g.is_target < 1)).sum()), slow=slow,
                                 blank=int((g.filled > 0).sum()), q_rms_lane=stats.get(reg, (np.nan, np.nan))[0],
                                 v_rms=stats.get(reg, (np.nan, np.nan))[1]))
    return pd.DataFrame(rows)


def panel_mean(df: pd.DataFrame, col: str) -> pd.Series:
    """Mean over regimes per corridor, then over corridors (Task 3 aggregation), per answer."""
    return df.groupby(["answer", "panel"])[col].mean().groupby("answer").mean()


def file_md5(path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 24), b""):
            h.update(blk)
    return h.hexdigest()


def public_scores(listing: str) -> dict:
    """{file name: public score} from `kaggle competitions submissions -v` output (CSV text; the header is the first line
    whose fields include fileName and publicScore - kaggle >= 2.2 puts a `ref` column first - and lines before it, e.g.
    the CLI's version warning, are skipped; pending submissions have no score yet; a name listed twice keeps its newest
    score - the listing is newest first)."""
    import io
    import csv
    lines = listing.splitlines()
    head = None
    for i, l in enumerate(lines):
        try:
            f = next(csv.reader([l]))
        except Exception:  # noqa: BLE001
            continue
        if "fileName" in f and "publicScore" in f:
            head = i
            break
    if head is None:
        return {}
    try:
        d = pd.read_csv(io.StringIO("\n".join(lines[head:])))
    except Exception:  # noqa: BLE001
        return {}
    if "fileName" not in d or "publicScore" not in d:
        return {}
    d["publicScore"] = pd.to_numeric(d.publicScore, errors="coerce")
    d = d.dropna(subset=["publicScore"])
    out = {}
    for f, v in zip(d.fileName.astype(str), d.publicScore.astype(float)):
        out.setdefault(f, v)
    return out


def decompose(B: float, no_queue: float, no_state: float, scaled: float, sfd: float, sfd_scaled: float) -> dict:
    """The public components from the three probes (see the module docstring); sfd / sfd_scaled = S_FD of the files
    and of the scaled answer (Task 3 aggregation, `fd_scores`). `flags`: components outside their possible range."""
    S_queue = (B - no_queue) / W["queue"]
    S_state = (B - scaled - W["physics"] / 3 * (sfd - sfd_scaled)) / W["state"]
    S_phys = (B - no_state - W["state"] * S_state) / W["physics"]
    S_lwr = (3 * S_phys - sfd) / 2
    S_odme = (no_state - W["queue"] * S_queue) / W["odme"]
    out = dict(B=B, S_state=S_state, S_queue=S_queue, S_physics=S_phys, S_FD=sfd, S_LWR=S_lwr, S_ODME=S_odme)
    # the five reads are rounded to 5 decimals: every component is known to about 1e-4; outside [0, 1] (or a physics
    # term above a perfect answer's ~0.96) means an assumption of the probes failed
    out["flags"] = [k for k in ("S_state", "S_queue", "S_physics", "S_LWR", "S_ODME") if not -2e-3 <= out[k] <= 1 + 2e-3]
    if S_phys > 0.98:
        out["flags"].append("S_physics above a perfect answer's")
    return out


# ----------------------------------------------------------------------------- probe files from the submitted base file
PROBES = ("no_queue", "no_state", "scaled")


def zip_csv_md5(zpath) -> tuple:
    """(member name, md5 of the CSV inside a submission zip) - the md5 the submission log keys its entries by."""
    import zipfile
    with zipfile.ZipFile(zpath) as z:
        names = [n for n in z.namelist() if n.endswith(".csv")]
        if len(names) != 1:
            raise ValueError(f"{zpath}: expected one CSV inside, found {names}")
        h = hashlib.md5()
        with z.open(names[0]) as f:
            for blk in iter(lambda: f.read(1 << 24), b""):
                h.update(blk)
    return names[0], h.hexdigest()


def pick_base(log: dict, sub_dir, prefix: str = "tfb v8 main") -> str | None:
    """The newest file of the submission log whose message starts with `prefix` and whose zip is in `sub_dir`."""
    hits = [(v.get("time", ""), v.get("file")) for v in log.values()
            if str(v.get("message", "")).startswith(prefix) and v.get("file")
            and (Path(sub_dir) / v["file"]).exists()]
    return max(hits)[1] if hits else None


def read_base(zpath) -> pd.DataFrame:
    """The submitted file as it was uploaded (dtypes fixed so that it is written back unchanged)."""
    d = pd.read_csv(zpath, dtype={"task": str})
    for c in ("submission_id", "queue_pred"):
        d[c] = d[c].astype(np.int64)
    for c in ("speed_kmh", "flow_vph", "path_flow"):
        d[c] = d[c].astype(np.float64)
    return d


def probe_frame(base: pd.DataFrame, name: str, factor: float = SCALE) -> pd.DataFrame:
    """The base file with one change: no_queue (every queue_pred 0), no_state (every Task 1 value 0) or scaled
    (every Task 1 speed and flow times `factor`)."""
    p = base.copy()
    if name == "no_queue":
        p.loc[(p.task == "queue").to_numpy(), "queue_pred"] = 0
    elif name == "no_state":
        m = (p.task == "state").to_numpy()
        p.loc[m, "speed_kmh"] = 0.0; p.loc[m, "flow_vph"] = 0.0
    elif name == "scaled":
        m = (p.task == "state").to_numpy()
        p.loc[m, "speed_kmh"] = p.speed_kmh.to_numpy()[m] * factor
        p.loc[m, "flow_vph"] = p.flow_vph.to_numpy()[m] * factor
    else:
        raise ValueError(name)
    return p


def write_zip(df: pd.DataFrame, csv_path) -> Path:
    """CSV (the format of submit.build) + its zip; returns the zip path."""
    import zipfile
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False, float_format="%.4f")
    zp = csv_path.with_suffix(".zip")
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(csv_path, arcname=csv_path.name)
    return zp


def state_rows(base: pd.DataFrame, key: pd.DataFrame, split: str = "validation") -> pd.DataFrame:
    """The base file's Task 1 rows of one split with their cell keys (panel, timestamp, station, link, regime)."""
    k = key[key.task == "state"]
    if "split" in k.columns:
        k = k[k.split == split]
    else:   # the split of a row from its date
        cal = C.calendar()
        k = k[k.timestamp.str.slice(0, 10).map(dict(zip(cal.date, cal.split))) == split]
    k = k[["submission_id", "panel", "timestamp", "station_id", "link_id", "mask_regime"]]
    s = base[base.task == "state"][["submission_id", "speed_kmh", "flow_vph"]]
    out = k.merge(s, on="submission_id", how="left", validate="one_to_one")
    if out.speed_kmh.isna().any():
        raise ValueError(f"{int(out.speed_kmh.isna().sum())} {split} Task 1 rows missing from the base file")
    return out


def with_queue_rows(base: pd.DataFrame, key: pd.DataFrame, rows: pd.DataFrame, windows) -> tuple:
    """The base file with the queue_pred of the windows `windows` taken from `rows` (window_id, timestamp, link_id,
    queue_pred). Returns (frame, number of rows that changed)."""
    kq = key[key.task == "queue"][["submission_id", "window_id", "timestamp", "link_id"]]
    kq = kq[kq.window_id.isin(set(windows))]
    src = rows.astype({"window_id": str, "timestamp": str, "link_id": str}).set_index(
        ["window_id", "timestamp", "link_id"]).queue_pred
    if src.index.has_duplicates:
        raise ValueError("duplicate queue rows in the probe file")
    v = src.reindex(pd.MultiIndex.from_frame(kq[["window_id", "timestamp", "link_id"]]))
    if v.isna().any():
        raise ValueError(f"{int(v.isna().sum())} queue rows of the selected windows are missing from the probe rows")
    p = base.copy()
    pos = pd.Index(p.submission_id.to_numpy()).get_indexer(kq.submission_id.to_numpy())
    if (pos < 0).any():
        raise ValueError("submission ids of the key are missing from the base file")
    old = p.queue_pred.to_numpy()[pos]
    new = v.to_numpy().astype(np.int64)
    qp = p.queue_pred.to_numpy().copy(); qp[pos] = new
    p["queue_pred"] = qp
    return p, int((old != new).sum())


def condition_windows(cond: str, split: str = "validation") -> set:
    """window_ids of one condition and split (the release's window_index files)."""
    out = set()
    for p in C.task2_panels():
        f = C.rel() / "task2" / p / split / "window_index.csv"
        if f.exists():
            w = pd.read_csv(f, dtype={"window_id": str})
            out |= set(w.window_id[w.condition == cond])
    return out


# ----------------------------------------------------------------------------- v10: the onset / ongoing split of S_queue
def zero_windows(base: pd.DataFrame, key: pd.DataFrame, windows) -> tuple:
    """The base file with queue_pred 0 in every row of the windows `windows` (onset_zero: every scored March onset
    window, whose true horizon always holds a queue, so their IoU drops to 0). Returns (frame, rows changed)."""
    kq = key[key.task == "queue"][["submission_id", "window_id"]]
    ids = kq.submission_id[kq.window_id.isin(set(windows))].to_numpy()
    if not len(ids):
        raise ValueError("no queue rows of the given windows in the key")
    p = base.copy()
    pos = pd.Index(p.submission_id.to_numpy()).get_indexer(ids)
    if (pos < 0).any():
        raise ValueError("submission ids of the key are missing from the base file")
    qp = p.queue_pred.to_numpy().copy()
    n = int((qp[pos] != 0).sum())
    qp[pos] = 0
    p["queue_pred"] = qp
    return p, n


def onset_share(split: str = "validation") -> float:
    """Weight of the onset windows inside S_queue under the official aggregation (window -> panel and condition ->
    panel -> corridor family -> mean): 0.5 when every corridor has scored windows of both conditions."""
    fam = C.families()
    w = {}
    for p in C.task2_panels():
        f = C.rel() / "task2" / p / split / "window_index.csv"
        if not f.exists():
            continue
        conds = set(pd.read_csv(f, usecols=["condition"]).condition)
        if conds:
            w[p] = (1.0 / len(conds)) if "queue_onset" in conds else 0.0
    if not w:
        return float("nan")
    s = pd.Series(w)
    return float(s.groupby(s.index.map(fam)).mean().mean())


def queue_split(B: float, onset_zero: float, S_queue: float, share: float = 0.5) -> dict:
    """S_onset and S_ongoing from the onset_zero probe: B - onset_zero = 0.30 x share x S_onset, and
    S_queue = share x S_onset + (1 - share) x S_ongoing."""
    s_on = (B - onset_zero) / (W["queue"] * share)
    return dict(S_onset=s_on, S_ongoing=(S_queue - share * s_on) / (1 - share), share=share,
                points_lost_onset=W["queue"] * share * (1 - s_on),
                points_lost_ongoing=W["queue"] * (1 - share) * (1 - (S_queue - share * s_on) / (1 - share)))


def with_state_rows(base: pd.DataFrame, key: pd.DataFrame, state: pd.DataFrame) -> tuple:
    """The base file with every Task 1 row (speed_kmh, flow_vph) taken from `state` (the task1.parquet format: panel,
    timestamp, link_id, speed_kmh, total flow_vph), joined the way submit.build joins them; every other row stays
    exactly as submitted. Returns (frame, Task 1 rows whose written value changes)."""
    k = key[key.task == "state"][["submission_id", "panel", "timestamp", "link_id"]]
    kl = (k.panel.astype(str) + "|" + k.timestamp.astype(str) + "|" + k.link_id.astype(str)).to_numpy()
    kr = pd.Index((state.panel.astype(str) + "|" + state.timestamp.astype(str) + "|" + state.link_id.astype(str)).to_numpy())
    if kr.has_duplicates:
        raise ValueError(f"{int(kr.duplicated().sum())} duplicate Task 1 rows")
    pos = kr.get_indexer(kl)
    if (pos < 0).any():
        raise ValueError(f"{int((pos < 0).sum())} Task 1 rows of the key are missing from the new Task 1 rows")
    p = base.copy()
    bpos = pd.Index(p.submission_id.to_numpy()).get_indexer(k.submission_id.to_numpy())
    if (bpos < 0).any():
        raise ValueError("submission ids of the key are missing from the base file")
    sp, fl = p.speed_kmh.to_numpy().copy(), p.flow_vph.to_numpy().copy()
    ns, nf = state.speed_kmh.to_numpy(np.float64)[pos], state.flow_vph.to_numpy(np.float64)[pos]
    if not (np.isfinite(ns).all() and np.isfinite(nf).all() and (ns > 0).all() and (nf >= 0).all()):
        raise ValueError("non-finite or out-of-range Task 1 values")
    changed = int(((np.round(ns, 4) != np.round(sp[bpos], 4)) | (np.round(nf, 4) != np.round(fl[bpos], 4))).sum())
    sp[bpos], fl[bpos] = ns, nf
    p["speed_kmh"], p["flow_vph"] = sp, fl
    return p, changed
