"""Network loading and dense [time x link] caches for every panel.

Links are kept in *physical* order (index increases downstream, following the released topology),
so "upstream" is always index-1 whatever the corridor direction. Only links that carry detector data
are kept (the topology may also list detector-less links). One cache per panel holds every split of the
release on one global time axis (calendar day * 288 + slot).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from . import common as C


def pfiles(root) -> list[Path]:
    """Every parquet *file* under a directory, sorted (a part-file layout is read file by file, never twice)."""
    root = Path(root)
    if not root.is_dir():
        return []
    return sorted(q for q in root.rglob("*.parquet") if q.is_file())


def pq_columns(path) -> list[str]:
    path = Path(path)
    if path.is_dir():
        parts = pfiles(path)
        return pq.ParquetFile(str(parts[0])).schema_arrow.names if parts else []
    return pq.ParquetFile(str(path)).schema_arrow.names


def read_pq(path, columns=None) -> pd.DataFrame:
    """One parquet file read exactly as stored, without hive-partition discovery from the folder names.
    The release keeps `mask_regime` / `year_month` both as folder names (mask_regime=R1/) and as columns; pyarrow < 24
    (Colab ships 23) then refuses to merge the two in pq.read_table / pd.read_parquet:
    "Unable to merge: Field mask_regime has incompatible types: string vs dictionary<...>"."""
    path = Path(path)
    if path.is_dir():
        frames = [read_pq(q, columns) for q in pfiles(path)]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=list(columns or []))
    pf = pq.ParquetFile(str(path))
    have = pf.schema_arrow.names
    cols = None if columns is None else [c for c in columns if c in have]
    return pf.read(columns=cols).to_pandas()


def _split_ids(x) -> list[str]:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []
    return [s.strip() for s in str(x).split(";") if s.strip() and s.strip().lower() != "nan"]


@dataclass
class Net:
    panel: str
    links: list
    stations: list
    length: np.ndarray
    lanes: np.ndarray
    vf: np.ndarray
    cap: np.ndarray
    kjam: np.ndarray
    vcut: np.ndarray
    on_att: np.ndarray        # topology says an on-ramp is attached
    off_att: np.ndarray
    observable: np.ndarray    # every topology neighbour carries data (official Task 3 filter)
    ramps: pd.DataFrame       # ramp_link_id, kind (ON/OFF), li (mainline index)
    extra: dict = field(default_factory=dict)

    @property
    def L(self):
        return len(self.links)

    @property
    def kc(self):
        return self.cap / np.maximum(self.vf, 1.0)

    @property
    def w(self):  # congested-branch wave speed (km/h)
        return self.cap / np.maximum(self.kjam - self.kc, 1e-3)

    @property
    def mp(self):  # distance of each link's downstream end from the corridor start (km)
        return np.cumsum(self.length)

    @property
    def lid(self):
        return {l: i for i, l in enumerate(self.links)}


def _first_masked(panel: str):
    for split in C.SPLITS:
        fs = pfiles(C.rel() / "corridors" / panel / split / "mainline_states_masked")
        if fs:
            return fs[0]
    return None


@lru_cache(None)
def data_links(panel: str) -> tuple:
    """Links (and their stations) that appear in the mainline records."""
    f = _first_masked(panel)
    if f is None:
        return tuple(), tuple()
    have = pq_columns(f)
    cols = [c for c in ("link_id", "station_id") if c in have]
    d = read_pq(f, cols).drop_duplicates()
    st = d.groupby(d.link_id.astype(str)).station_id.first() if "station_id" in d else pd.Series(dtype=str)
    return tuple(st.index.tolist()) if len(st) else tuple(d.link_id.astype(str).unique()), tuple(st.astype(str).tolist())


@lru_cache(None)
def load_net(panel: str) -> Net:
    nd = C.rel() / "corridors" / panel / "network"
    topo = pd.read_csv(nd / "lwr_mainline_topology.csv", dtype=str)
    key = "link_id" if "link_id" in topo.columns else "mainline_link_id"
    topo["lk"] = topo[key].astype(str)
    topo = topo.drop_duplicates("lk").set_index("lk", drop=False)
    tl = topo.lk.tolist(); tset = set(tl)
    out = {l: [x for x in _split_ids(topo.at[l, "outgoing_link_ids"]) if x in tset] if "outgoing_link_ids" in topo else [] for l in tl}
    inc = {l: [x for x in _split_ids(topo.at[l, "incoming_link_ids"]) if x in tset] if "incoming_link_ids" in topo else [] for l in tl}
    # physical order: walk the outgoing chain from the head (no incoming); fall back to order_index
    heads = [l for l in tl if not inc[l]]
    order = []
    how = "outgoing chain"
    if len(heads) == 1:
        cur, seen = heads[0], set()
        while cur is not None and cur not in seen:
            order.append(cur); seen.add(cur)
            nx = out.get(cur, [])
            cur = nx[0] if nx else None
    if len(order) != len(tl):
        how = "order_index" if "order_index" in topo else "file order"
        oi = pd.to_numeric(topo.order_index, errors="coerce").to_numpy() if "order_index" in topo else np.arange(len(tl))
        order = [tl[i] for i in np.argsort(oi, kind="stable")]
        fwd = sum(order[i + 1] in out[order[i]] for i in range(len(order) - 1))
        bwd = sum(order[i + 1] in inc[order[i]] for i in range(len(order) - 1))
        if bwd > fwd:
            order = order[::-1]; how += ", reversed"
        C.log(f"{panel}: link order from {how} ({len(heads)} chain heads; {fwd} forward / {bwd} backward links agree)")
    dl, dst = data_links(panel)
    dset = set(dl)
    if dset:
        missing = [l for l in dl if l not in tset]
        if missing:
            C.log(f"WARNING {panel}: {len(missing)} data links not in the topology; appended at the end")
        latent = [l for l in order if l not in dset]
        if latent:
            C.log(f"{panel}: {len(latent)} topology links carry no detector data and are left out")
        order = [l for l in order if l in dset] + missing
    observable = np.array([set(inc.get(l, []) + out.get(l, [])).issubset(dset or tset) for l in order])
    fd = pd.read_csv(nd / "fd_parameters.csv", dtype={"link_id": str, "station_id": str})
    lk = pd.read_csv(nd / "links.csv", dtype={"link_id": str}).drop_duplicates("link_id").set_index("link_id")
    num = [c for c in ("free_speed_kmh", "capacity_vph", "lanes", "k_jam", "length_km", "v_cut") if c in fd.columns]
    g = fd.groupby("link_id")[num].mean()

    def col(name, default):
        a = g[name].reindex(order) if name in g.columns else pd.Series(np.nan, index=order)
        if name in lk.columns:
            a = a.fillna(pd.to_numeric(lk[name], errors="coerce").reindex(order))
        return a.fillna(default).to_numpy(np.float64)

    lanes = np.clip(col("lanes", 4.0), 1, 16)
    vf = col("free_speed_kmh", 105.0)
    cap = col("capacity_vph", np.nan)
    cap = np.where(np.isfinite(cap), cap, 1800.0 * lanes)
    kjam = col("k_jam", np.nan)
    kjam = np.where(np.isfinite(kjam), kjam, 100.0 * lanes)
    kjam = np.maximum(kjam, cap / np.maximum(vf, 1) * 1.05)
    length = col("length_km", 0.5)
    vc = g["v_cut"].reindex(order).to_numpy(np.float64) if "v_cut" in g.columns else np.full(len(order), np.nan)
    vcut = np.where(np.isfinite(vc), vc, 0.6 * vf)
    on_att = np.array([len(_split_ids(topo.at[l, "on_ramp_link_ids"])) > 0 if ("on_ramp_link_ids" in topo and l in tset) else False for l in order])
    off_att = np.array([len(_split_ids(topo.at[l, "off_ramp_link_ids"])) > 0 if ("off_ramp_link_ids" in topo and l in tset) else False for l in order])
    rm = pd.read_csv(nd / "ramp_attachment_map.csv", dtype=str) if (nd / "ramp_attachment_map.csv").exists() \
        else pd.DataFrame(columns=["ramp_link_id", "ramp_type", "nearest_mainline_link_id"])
    if "ramp_type" not in rm.columns or rm.ramp_type.isna().all():  # take the type from the ramp records
        fs = pfiles(C.rel() / "corridors" / panel / "train" / "ramp_states")
        if fs and "ramp_type" in pq_columns(fs[0]):
            rt = read_pq(fs[0], ["ramp_link_id", "ramp_type"]).drop_duplicates("ramp_link_id")
            rm = rm.drop(columns=["ramp_type"], errors="ignore").merge(rt.astype(str), on="ramp_link_id", how="left")
    lidx = {l: i for i, l in enumerate(order)}
    rm["li"] = rm.nearest_mainline_link_id.astype(str).map(lidx)
    typ = rm.get("ramp_type", pd.Series("", index=rm.index)).astype(str).str.upper()
    rm["kind"] = np.where(typ.isin(["OR", "ON", "ON_RAMP", "ONRAMP"]), "ON",
                          np.where(typ.isin(["FR", "OFF", "OFF_RAMP", "OFFRAMP"]), "OFF", "?"))
    if (rm.kind == "?").any():
        C.log(f"WARNING {panel}: {int((rm.kind == '?').sum())} ramps with unknown type {sorted(set(typ[rm.kind == '?']))[:5]}")
    rm = rm[rm.li.notna() & (rm.kind != "?")].copy()
    rm["li"] = rm.li.astype(int)
    stmap = dict(zip(dl, dst)) if dst else {}
    if not stmap and "station_id" in fd.columns:
        stmap = fd.groupby("link_id").station_id.first().to_dict()
    stations = [str(stmap.get(l, l)) for l in order]
    return Net(panel, order, stations, length, lanes, vf, cap, kjam, vcut, on_att, off_att, observable,
               rm[["ramp_link_id", "kind", "li"]].drop_duplicates("ramp_link_id").reset_index(drop=True))


def _read(path, cols):
    t = read_pq(path, cols)
    for c in cols:
        if c not in t.columns:
            t[c] = np.nan
    return t


def _rowcol(df: pd.DataFrame, lidx: dict, key="link_id"):
    li = df[key].astype(str).map(lidx)
    ok = li.notna().to_numpy()
    t = C.tindex(df.timestamp[ok]) if ok.any() else np.zeros(0, np.int64)
    return ok, t, li[ok].to_numpy(np.int64)


def _per_cell(t, li, df, num_cols, el_col=None, pct_col=None):
    """One value per (t, link): direct when unique (one station per link), else mean of finite values,
    eligibility = any, pct = max (several stations on one link)."""
    key = t * 100_000 + li
    if not pd.Index(key).has_duplicates:
        return t, li, df
    g = pd.DataFrame({"k": key, **{c: pd.to_numeric(df[c], errors="coerce").to_numpy() for c in num_cols}})
    agg = {c: "mean" for c in num_cols}
    if el_col:
        g[el_col] = df[el_col].fillna(False).astype(bool).to_numpy(); agg[el_col] = "max"
    if pct_col:
        g[pct_col] = pd.to_numeric(df[pct_col], errors="coerce").to_numpy(); agg[pct_col] = "max"
    r = g.groupby("k", sort=False).agg(agg).reset_index()
    return (r.k.to_numpy() // 100_000).astype(np.int64), (r.k.to_numpy() % 100_000).astype(np.int64), r


CACHE_KEYS = ("v", "q", "o", "pct", "el", "tv", "tq", "to", "tgt", "dayreg", "ron", "roff", "rv_on", "rv_off", "dark",
              "non", "noff")


def fingerprint() -> str:
    h = hashlib.md5()
    h.update(str(C.rel().resolve()).encode())
    h.update((C.rel() / "config" / "corridors.json").read_bytes())
    h.update(",".join(C.calendar().date).encode())
    return h.hexdigest()[:16]


def window_origins(panel: str) -> dict:
    """Official Task 2 origins per split (global slot index)."""
    out = {}
    for split in C.SPLITS:
        f = C.rel() / "task2" / panel / split / "window_index.csv"
        if f.exists():
            w = pd.read_csv(f)
            out[split] = C.tindex(w.forecast_origin)
    return out


def build_cache(panel: str, force: bool = False) -> Path:
    out = C.CACHE / f"{panel}.npz"
    if out.exists() and not force:
        return out
    C.CACHE.mkdir(parents=True, exist_ok=True)
    net = load_net(panel)
    L, NT, ND = net.L, C.n_slots(), C.n_days()
    lidx = net.lid
    A = {k: np.full((NT, L), np.nan, np.float32) for k in ("v", "q", "o", "tv", "tq", "to", "ron", "roff")}
    pct = np.full((NT, L), -1, np.int8)
    el = np.zeros((NT, L), bool)
    tgt = np.zeros((NT, L), np.int8)
    dayreg = np.zeros(ND, np.int8)
    base = C.rel() / "corridors" / panel
    cols = ["timestamp", "link_id", "speed_kmh", "flow_vph", "occupancy", "pct_observed", "is_score_eligible"]
    for split in C.SPLITS:
        for reg, rid in C.REG_ID.items():
            for f in pfiles(base / split / "mainline_states_masked" / f"mask_regime={reg}"):
                d = _read(f, cols)
                ok, t, li = _rowcol(d, lidx)
                t, li, d = _per_cell(t, li, d[ok].reset_index(drop=True), ["speed_kmh", "flow_vph", "occupancy"],
                                     "is_score_eligible", "pct_observed")
                A["v"][t, li] = pd.to_numeric(d.speed_kmh, errors="coerce").to_numpy(np.float32)
                A["q"][t, li] = pd.to_numeric(d.flow_vph, errors="coerce").to_numpy(np.float32)
                A["o"][t, li] = pd.to_numeric(d.occupancy, errors="coerce").to_numpy(np.float32)
                pct[t, li] = np.clip(pd.to_numeric(d.pct_observed, errors="coerce").fillna(-1).to_numpy(), -1, 100).astype(np.int8)
                el[t, li] = d.is_score_eligible.fillna(False).astype(bool).to_numpy()
                dayreg[np.unique(t // C.SLOTS)] = rid
        tp = C.rel() / "task1" / panel / split / "sample_submission_state.csv"
        if tp.exists():
            tm = pd.read_csv(tp, usecols=["timestamp", "link_id", "mask_regime"], dtype=str)
            ok, t, li = _rowcol(tm, lidx)
            tgt[t, li] = tm.mask_regime[ok].map(C.REG_ID).fillna(0).to_numpy(np.int8)
    for f in pfiles(base / "train" / "mainline_states"):
        d = _read(f, ["timestamp", "link_id", "speed_kmh", "flow_vph", "occupancy"])
        ok, t, li = _rowcol(d, lidx)
        t, li, d = _per_cell(t, li, d[ok].reset_index(drop=True), ["speed_kmh", "flow_vph", "occupancy"])
        A["tv"][t, li] = pd.to_numeric(d.speed_kmh, errors="coerce").to_numpy(np.float32)
        A["tq"][t, li] = pd.to_numeric(d.flow_vph, errors="coerce").to_numpy(np.float32)
        A["to"][t, li] = pd.to_numeric(d.occupancy, errors="coerce").to_numpy(np.float32)
    # ramps: per mainline link, sum of valid on / off ramp flows; Task 3 validity flags; number of valid on- and
    # off-ramps (v6/v7: a link whose ramps are only partly valid has a partial sum - `non` / `noff` tell it apart)
    rmap = net.ramps.set_index("ramp_link_id")
    anyon = np.zeros((NT, L), bool); anyoff = np.zeros((NT, L), bool)
    non = np.zeros((NT, L), np.int8); noff = np.zeros((NT, L), np.int8)
    for split in C.SPLITS:
        for f in pfiles(base / split / "ramp_states"):
            d = _read(f, ["timestamp", "ramp_link_id", "flow_vph", "pct_observed", "is_score_eligible"])
            d["ramp_link_id"] = d.ramp_link_id.astype(str)
            d = d[d.ramp_link_id.isin(rmap.index)]
            if d.empty:
                continue
            fl = pd.to_numeric(d.flow_vph, errors="coerce")
            valid = d.is_score_eligible.fillna(False).astype(bool) & (pd.to_numeric(d.pct_observed, errors="coerce") >= 75) & fl.notna()
            d = d[valid.to_numpy()]
            if d.empty:
                continue
            t = C.tindex(d.timestamp)
            li = rmap.li.reindex(d.ramp_link_id).to_numpy(np.int64)
            kind = rmap.kind.reindex(d.ramp_link_id).to_numpy()
            fv = pd.to_numeric(d.flow_vph).to_numpy(np.float32)
            for k, arr, anyf in (("ON", A["ron"], anyon), ("OFF", A["roff"], anyoff)):
                m = kind == k
                if not m.any():
                    continue
                tt, ll, vv = t[m], li[m], fv[m]
                cur = arr[tt, ll]
                arr[tt, ll] = np.where(np.isnan(cur), 0, cur)
                np.add.at(arr, (tt, ll), vv)
                anyf[tt, ll] = True
                # distinct valid ramps of this kind per (slot, link)
                u = pd.DataFrame(dict(t=tt, l=ll, r=d.ramp_link_id.to_numpy()[m])).drop_duplicates(["t", "r"])
                np.add.at(non if k == "ON" else noff, (u.t.to_numpy(), u.l.to_numpy()), 1)
    rv_on = anyon | ~net.on_att[None, :]
    rv_off = anyoff | ~net.off_att[None, :]
    # Task 2 blackout rows (horizon + 60-min buffer = T+1..T+18) of every official window, all splits
    dark = np.zeros(NT, bool)
    for split, Ts in window_origins(panel).items():
        for T in Ts:
            dark[T + 1:min(NT, T + 19)] = True
    detected = blackout_rows_detect(A["v"], A["q"], pct)
    if dark.any():
        agree = (dark & detected).sum() / max(dark.sum(), 1)
        C.log(f"{panel}: {int(dark.sum())} blackout rows from window_index; {agree:.0%} also blank in the release")
    np.savez(out, **A, pct=pct, el=el, tgt=tgt, dayreg=dayreg, rv_on=rv_on, rv_off=rv_off, dark=dark, non=non, noff=noff,
             links=np.array(net.links), stations=np.array(net.stations), fingerprint=np.array(fingerprint()))
    C.log(f"cache {panel}: L={L} targets={int((tgt > 0).sum()):,} train-truth={int(np.isfinite(A['tv']).sum()):,}")
    return out


def load_cache(panel: str, keys=None) -> dict:
    z = np.load(build_cache(panel), allow_pickle=True)
    stale = any(k not in z.files for k in CACHE_KEYS) or "fingerprint" not in z.files or str(z["fingerprint"]) != fingerprint()
    if stale:  # cache from an older version of this code or from another release
        z = np.load(build_cache(panel, force=True), allow_pickle=True)
    keys = keys or z.files
    return {k: z[k] for k in keys if k in z.files}


def build_all(n_jobs: int = 2, force: bool = False):
    from concurrent.futures import ProcessPoolExecutor
    ps = []
    for p in C.panels():
        f = C.CACHE / f"{p}.npz"
        if force or not f.exists():
            ps.append(p); continue
        z = np.load(f, allow_pickle=True)
        if any(k not in z.files for k in CACHE_KEYS) or "fingerprint" not in z.files or str(z["fingerprint"]) != fingerprint():
            ps.append(p)
    if not ps:
        return
    if n_jobs <= 1:
        for p in ps:
            build_cache(p, True)
        return
    with ProcessPoolExecutor(n_jobs) as ex:
        list(ex.map(build_cache, ps, [True] * len(ps)))


def blackout_rows_detect(v, q, pct) -> np.ndarray:
    has = np.isfinite(v).any(1) | np.isfinite(q).any(1)
    has_rec = (pct >= 0).any(1)
    return has_rec & ~has


def blackout_rows(c: dict) -> np.ndarray:
    """Task 2 horizon + buffer rows (from the window index; falls back to rows blank on every link)."""
    if "dark" in c:
        return c["dark"].copy()
    return blackout_rows_detect(c["v"], c["q"], c["pct"])


# ----------------------------------------------------------------------------- preflight
def milepost_order(panel: str) -> dict:
    """v8 diagnostic: our link order (the topology's outgoing chain; index increases downstream) against the released
    mileposts, by which the Task 3 scorer sorts links. mp_rho = Spearman correlation of link index and milepost
    (+1 or -1 when the two orders agree; -1 where mileposts fall in the direction of travel); mp_flips = adjacent links
    whose milepost step goes against the dominant direction; mp_missing = links without a milepost. Nothing is
    changed: the spatial features keep the topology order, which is also what the scorer's conservation terms use."""
    out = dict(mp_rho=np.nan, mp_flips=-1, mp_missing=-1)
    f = _first_masked(panel)
    if f is None or "milepost" not in pq_columns(f):
        return out
    links = load_net(panel).links
    d = read_pq(f, ["link_id", "milepost"])
    d = d.assign(link_id=d.link_id.astype(str), milepost=pd.to_numeric(d.milepost, errors="coerce"))
    mp = d.groupby("link_id").milepost.median().reindex(links).to_numpy(np.float64)
    ok = np.isfinite(mp)
    out["mp_missing"] = int((~ok).sum())
    if ok.sum() < 3:
        return out
    m = mp[ok]
    rho = pd.Series(np.arange(len(m), dtype=np.float64)).corr(pd.Series(m), method="spearman")
    if np.isfinite(rho):
        out["mp_rho"] = round(float(rho), 4)
        out["mp_flips"] = int((np.sign(np.diff(m)) == -(1.0 if rho >= 0 else -1.0)).sum())
    return out


def preflight(strict: bool = True) -> pd.DataFrame:
    """Read one file of every kind per corridor and check what the pipeline relies on (files, columns, link ids,
    timestamps) before any heavy stage. Returns a per-panel summary; raises with the full list of problems."""
    probs, warns, rows = [], [], []
    rel = C.rel()

    def need(cond, msg):
        if not cond:
            probs.append(msg)
        return bool(cond)

    for f in ("submission_key.csv", "sample_submission.csv"):
        need((rel / f).exists(), f"missing {f}")
    cal_days = set(C.calendar().date)
    t2p = set(C.task2_panels())
    for panel in C.panels():
        r = dict(panel=panel)
        nd = rel / "corridors" / panel / "network"
        for f in ("links.csv", "fd_parameters.csv", "lwr_mainline_topology.csv", "path_set.csv", "path_link_incidence.csv"):
            need((nd / f).exists(), f"{panel}: missing network/{f}")
        try:
            net = load_net(panel)
            r["links"] = net.L
            need(net.L > 0, f"{panel}: no mainline links with detector data")
            r["ramps"] = len(net.ramps)
        except Exception as e:  # noqa: BLE001
            probs.append(f"{panel}: network could not be loaded ({type(e).__name__}: {e})")
            rows.append(r)
            continue
        lid = net.lid
        try:   # v8 diagnostic only: never stops the pipeline
            mo = milepost_order(panel)
            r.update(mo)
            if np.isfinite(mo["mp_rho"]) and (abs(mo["mp_rho"]) < 0.95 or mo["mp_flips"] > 0):
                warns.append(f"{panel}: link order and mileposts disagree (Spearman {mo['mp_rho']:+.3f}, "
                             f"{mo['mp_flips']} adjacent pairs out of order); the features follow the topology chain")
        except Exception as e:  # noqa: BLE001
            warns.append(f"{panel}: milepost check failed ({type(e).__name__}: {e})")
        for split in C.SPLITS:
            fs = pfiles(rel / "corridors" / panel / split / "mainline_states_masked")
            if not need(fs, f"{panel}/{split}: no masked mainline files"):
                continue
            r[f"{split}_files"] = len(fs)
            have = pq_columns(fs[0])
            miss = [c for c in ("timestamp", "link_id", "speed_kmh", "flow_vph", "pct_observed", "is_score_eligible") if c not in have]
            need(not miss, f"{panel}/{split}: masked files lack columns {miss} (have {have})")
            if miss:
                continue
            d = read_pq(fs[0], ["timestamp", "link_id", "speed_kmh"])
            share = d.link_id.astype(str).isin(lid).mean() if len(d) else 0.0
            need(share > 0.9, f"{panel}/{split}: only {share:.0%} of masked-file link_ids match the network")
            try:
                days = set(pd.Series(d.timestamp).astype(str).str.slice(0, 10).unique())
                need(days <= cal_days, f"{panel}/{split}: timestamps outside the calendar, e.g. {sorted(days - cal_days)[:2]}")
                C.tindex(d.timestamp.iloc[:5])
            except Exception as e:  # noqa: BLE001
                probs.append(f"{panel}/{split}: timestamps not parseable ({e}); sample {d.timestamp.iloc[:2].tolist()}")
            if split != "train":
                tp = rel / "task1" / panel / split / "sample_submission_state.csv"
                if need(tp.exists(), f"{panel}/{split}: missing task1 template"):
                    cols = pd.read_csv(tp, nrows=5).columns.tolist()
                    need({"timestamp", "link_id", "mask_regime"} <= set(cols), f"{panel}/{split}: task1 template columns {cols}")
            rs = pfiles(rel / "corridors" / panel / split / "ramp_states")
            if rs:
                have = pq_columns(rs[0])
                miss = [c for c in ("timestamp", "ramp_link_id", "flow_vph", "pct_observed", "is_score_eligible") if c not in have]
                if miss:
                    warns.append(f"{panel}/{split}: ramp files lack {miss}; ramp features will be empty")
            elif split != "train" or panel in t2p:
                warns.append(f"{panel}/{split}: no ramp files")
            dt = rel / "task4" / panel / split
            if split != "train":
                for f, cols_ in (("synthetic_link_counts.csv", {"link_id", "count"}), ("synthetic_weak_prior.csv", {"path_id", "path_flow"}),
                                 ("sample_submission_path_flow.csv", {"path_id"})):
                    if need((dt / f).exists(), f"{panel}/{split}: missing task4/{f}"):
                        cols = set(pd.read_csv(dt / f, nrows=5).columns)
                        need(cols_ <= cols, f"{panel}/{split}: task4/{f} lacks {sorted(cols_ - cols)}")
            if panel in t2p and split != "train":
                d2 = rel / "task2" / panel / split
                if need((d2 / "window_index.csv").exists(), f"{panel}/{split}: missing task2 window_index.csv"):
                    wi = pd.read_csv(d2 / "window_index.csv")
                    need({"window_id", "forecast_origin", "condition"} <= set(wi.columns), f"{panel}/{split}: window_index columns {wi.columns.tolist()}")
                    r[f"{split}_windows"] = len(wi)
                if need((d2 / "window_history.parquet").exists(), f"{panel}/{split}: missing task2 window_history.parquet"):
                    have = pq_columns(d2 / "window_history.parquet")
                    miss = [c for c in ("window_id", "timestamp", "link_id", "speed_kmh") if c not in have]
                    need(not miss, f"{panel}/{split}: window_history lacks {miss}")
                    if "flow_vph" not in have:
                        warns.append(f"{panel}/{split}: window_history has no flow_vph; flow history features will be empty")
                need((d2 / "sample_submission_queue.csv").exists(), f"{panel}/{split}: missing task2 template")
        tr = pfiles(rel / "corridors" / panel / "train" / "mainline_states")
        if need(tr, f"{panel}/train: no unmasked mainline files"):
            have = pq_columns(tr[0])
            miss = [c for c in ("timestamp", "link_id", "speed_kmh", "flow_vph") if c not in have]
            need(not miss, f"{panel}/train: unmasked files lack {miss}")
            r["train_truth_files"] = len(tr)
        rows.append(r)
    out = pd.DataFrame(rows)
    for w in warns:
        C.log("preflight warning:", w)
    if probs:
        msg = "preflight found problems:\n  " + "\n  ".join(probs)
        if strict:
            raise RuntimeError(msg)
        C.log(msg)
    else:
        C.log(f"preflight ok: {len(out)} panels, every file kind readable")
    return out
