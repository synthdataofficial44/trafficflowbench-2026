"""Shared settings, calendar and small helpers for the TrafficFlowBench pipeline.

Paths come from environment variables so the same code runs in Colab, Kaggle or locally:
  TFB_DATA   release root (the unpacked kaggle_public/ directory)
  TFB_CACHE  dense per-panel caches (local disk, rebuilt in a few minutes)
  TFB_WORK   models, features, predictions (point this at Google Drive to survive disconnects)
"""
from __future__ import annotations

import datetime as _dt
import gc
import glob
import json
import os
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

SLOTS = 288
REG_ID = {"R1": 1, "R2": 2, "R3": 3}
REG_NAME = {1: "R1", 2: "R2", 3: "R3"}
REG_RATE = {1: 0.2, 2: 0.3, 3: 0.5}
SPLITS = ("train", "validation", "private")
TEST_SPLITS = ("validation", "private")
_T0 = time.time()


def log(*a):
    print(f"[{time.time() - _T0:7.0f}s]", *a, flush=True)


def find_release(hint: str | None = None) -> Path:
    cands = [hint, os.environ.get("TFB_DATA"), "/content/data/kaggle_public", "/content/kaggle_public",
             "/kaggle/input/2026-ieee-big-data-traffic-flow-bench/kaggle_public", "data/kaggle_public", "kaggle_public"]
    cands += glob.glob("/kaggle/input/**/kaggle_public", recursive=True)
    cands += glob.glob("/content/**/kaggle_public", recursive=True)
    for c in cands:
        if c and (Path(c) / "config" / "corridors.json").exists():
            return Path(c)
    raise FileNotFoundError("kaggle_public release root not found; set TFB_DATA")


REL = Path(os.environ.get("TFB_DATA", "")) if os.environ.get("TFB_DATA") else None
CACHE = Path(os.environ.get("TFB_CACHE", "/content/cache"))
WORK = Path(os.environ.get("TFB_WORK", "/content/work"))
THREADS = int(os.environ.get("TFB_THREADS", str(max(1, (os.cpu_count() or 2)))))


def rel() -> Path:
    global REL
    if REL is None or not (REL / "config" / "corridors.json").exists():
        REL = find_release()
    return REL


def set_paths(data: str | Path | None = None, cache: str | Path | None = None, work: str | Path | None = None):
    global REL, CACHE, WORK
    if data:
        REL = Path(data)
    if cache:
        CACHE = Path(cache)
    if work:
        WORK = Path(work)
    for p in (CACHE, WORK):
        Path(p).mkdir(parents=True, exist_ok=True)
    os.environ["TFB_DATA"] = str(rel()); os.environ["TFB_CACHE"] = str(CACHE); os.environ["TFB_WORK"] = str(WORK)


@lru_cache(None)
def panels() -> list[str]:
    m = json.loads((rel() / "config" / "corridors.json").read_text())
    return [p["corridor_id"] for p in m["panels"]]


@lru_cache(None)
def families() -> dict[str, str]:
    m = json.loads((rel() / "config" / "corridors.json").read_text())
    return {p["corridor_id"]: p.get("family_id", p["corridor_id"].rsplit("_", 1)[0]) for p in m["panels"]}


@lru_cache(None)
def task2_panels() -> list[str]:
    out = []
    for p in panels():
        if (rel() / "task2" / p / "validation" / "window_index.csv").exists():
            out.append(p)
    return out


@lru_cache(None)
def calendar() -> pd.DataFrame:
    """One row per calendar day of the release, in time order: date, split, day index, weekday."""
    rows = []
    cfg = rel() / "config" / "synthetic_release_v1.json"
    spans = None
    if cfg.exists():
        spans = json.loads(cfg.read_text()).get("synthetic_calendar")
    if spans:
        for s in SPLITS:
            a = pd.Timestamp(spans[s]["start"]).date(); b = pd.Timestamp(spans[s]["end_exclusive"]).date()
            d = a
            while d < b:
                rows.append((d.isoformat(), s)); d += _dt.timedelta(days=1)
    else:  # fall back to the files of the first panel
        p = panels()[0]
        for s in SPLITS:
            for f in sorted(glob.glob(str(rel() / "corridors" / p / s / "mainline_states_masked" / "**" / "*.parquet"), recursive=True)):
                st = Path(f).stem.split("_")[-3:]
                rows.append(("-".join(st), s))
        rows = sorted(set(rows))
    cal = pd.DataFrame(rows, columns=["date", "split"]).drop_duplicates("date").sort_values("date").reset_index(drop=True)
    cal["day"] = np.arange(len(cal))
    cal["weekday"] = pd.to_datetime(cal.date).dt.weekday.astype(int)
    return cal


@lru_cache(None)
def day_of_date() -> dict:
    c = calendar()
    return dict(zip(c.date, c.day))


def n_days() -> int:
    return len(calendar())


def n_slots() -> int:
    return n_days() * SLOTS


def split_days(split: str) -> np.ndarray:
    c = calendar()
    return c.day[c.split == split].to_numpy()


def split_rows(split: str) -> slice:
    d = split_days(split)
    return slice(int(d.min()) * SLOTS, (int(d.max()) + 1) * SLOTS)


def weekday_of_slot() -> np.ndarray:
    wd = calendar().weekday.to_numpy()
    return np.repeat(wd, SLOTS).astype(np.int8)


def split_of_day() -> np.ndarray:
    return calendar().split.to_numpy()


def tindex(ts) -> np.ndarray:
    """Timestamp text 'YYYY-MM-DDTHH:MM:SSZ' -> global slot index (vectorised, no datetime parsing)."""
    s = pd.Series(ts).astype(str)
    dmap = day_of_date()
    uniq = s.str.slice(0, 10)
    day = uniq.map(dmap)
    if day.isna().any():
        bad = uniq[day.isna()].iloc[0]
        raise KeyError(f"date {bad} not in calendar")
    hh = s.str.slice(11, 13).astype(int).to_numpy(); mm = s.str.slice(14, 16).astype(int).to_numpy()
    return day.to_numpy(np.int64) * SLOTS + hh * 12 + mm // 5


def ts_text(t: np.ndarray) -> np.ndarray:
    """Global slot index -> timestamp text in the release format."""
    c = calendar()
    t = np.asarray(t, np.int64)
    d = c.date.to_numpy()[t // SLOTS]
    s = t % SLOTS
    hh = (s // 12).astype(int); mm = (s % 12 * 5).astype(int)
    return np.array([f"{a}T{h:02d}:{m:02d}:00Z" for a, h, m in zip(d, hh, mm)], dtype=object)


def free():
    gc.collect()


def ffill_idx(ok: np.ndarray) -> np.ndarray:
    """Index of the last True at or before each row (axis 0); -1 if none."""
    T = ok.shape[0]
    idx = np.where(ok, np.arange(T, dtype=np.int32)[:, None], np.int32(-1))
    return np.maximum.accumulate(idx, axis=0)


def bfill_idx(ok: np.ndarray) -> np.ndarray:
    """Index of the first True at or after each row (axis 0); T if none."""
    T = ok.shape[0]
    idx = np.where(ok, np.arange(T, dtype=np.int32)[:, None], np.int32(T))
    return np.minimum.accumulate(idx[::-1], axis=0)[::-1]


def runs_1d(mask: np.ndarray):
    """Start/end (exclusive) of runs of True in a 1-d boolean array."""
    m = np.concatenate([[False], mask.astype(bool), [False]])
    d = np.diff(m.astype(np.int8))
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)
