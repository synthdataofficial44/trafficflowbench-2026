"""Task 2 (queue forecasting, online): only data timestamped <= T is used for a window at origin T.

Design (each element checked against the public leaderboard by other teams, see the notebook notes):
  * the official window selector is replicated on train to draw training/evaluation windows that look like
    the scored ones (coverage >= 0.7, condition from the visible history, queue in the horizon, persistence
    IoU <= 0.9, chronological greedy, 5 per condition, 360 min apart);
  * onset windows have their truth only at T+30 (the selector picks the earliest origin whose horizon touches
    the new queue; organizer-confirmed), so onset is a per-link classifier at T+30 and steps 1-5 stay empty;
  * ongoing windows: per (link, step) classifier on cells near queue activity, blended with a variant that
    drops the location priors (robust to the test months' different bottlenecks);
  * train labels: observed speed <= 0.6 v_f, missing cells filled in time, and near observed queues by a
    space-time imputer (the first queued slot of a new queue is sharp in time, smooth in space);
  * decoding: per window the top-m cells maximising the expected-IoU surrogate sum_S p / (|S| + sum_notS p).

v2 adds the scored months themselves as evidence (released masked view only, so nothing beyond the <= T rule):
  * pseudo-holdout: queue windows found in the March masked view with the selector's rules (hidden Task 1
    targets in the history filled causally, labels = observed eligible horizon cells, unknown cells left out of the
    IoU). It measures the models on the months that are scored instead of on train;
  * April adaptation: every April origin may use all of March, so the April models are refit on train + the March
    pseudo windows. It is adopted per condition only if the same recipe wins on March itself (fit on the first half
    of March, scored on the second half); otherwise April keeps the standard models. No April label is used (April
    pseudo windows are built only with the diagnostic switch `pseudo_april`, and nothing reads them then either).

v4 (after the v3 public probe: March rows with the recency features scored 0.86677 against 0.86840):
  * the scored windows are the first qualifying origins from the start of each month (chronological greedy, 5 + 5,
    360 min apart), and the release blanks every scored window's horizon and the hour after it. On those windows the
    recent-days evidence is missing exactly where it would matter (the same peak on the previous scored days) or
    belongs to the previous month, so the v3 recency features (rc_*) are dropped and the online adaptation is off;
  * validation and private were generated independently (organizer README: own demand draws, own incidents). For
    the first April days, "recent" means late March, another draw. Two consistent files are written:
      main (A)  - the 0.86840 features (v2, incl. the last-7-days `rec7_`), April models refit on train + March
                  (the March-gated adaptation): the bet that April resembles March;
      hedge (H) - train only and no recent-days features at all (`v2n`): the bet that April resembles train.
    Both use the same variants; the onset no-location blend is added only if it wins on the March pseudo-holdout
    (another draw) and does not lose on the week-blocked train CV (the training draw) by more than its SE. That choice
    uses March windows later than the scored March origins, so it applies to the April rows only; the March rows of
    both files keep the baseline variants.

v5 (after the v4 run: April gate +0.073 onset / +0.029 ongoing on later March, onset no-location blend +0.010 on March
and +0.004 on the train CV - positive on both draws but under the 1.5-SE bar):
  * variant choice by two draws: a candidate is taken when it wins on the March pseudo-holdout by `exp_z` (1.0) SEs and
    does not lose on the train CV, or wins on the train CV by `exp_cv_z` (2.0) SEs and does not lose on March (plus the
    v4 safety checks). The bar was lowered after v4 (whose onset blend stood at 1.2 SEs on March); its train-CV arm
    (2.7 SEs) would take that blend on its own. The March rows of both files keep the v4 recipe;
  * late windows: the selector takes each corridor's five onset windows first (the first peaks, April 1-3) and the
    ongoing ones after them, so April's scored windows from day 7 on (14 of 80, April 7-17) are ongoing windows a week
    into the month. They are predicted by models refitted with April's own ongoing pseudo windows of the days before
    (blocks start on day `late_blocks` = [6] of the month: the same-month effect the April gate measured), when the
    same recipe wins on March first (fit with the March windows before the block, scored on the March windows in it).
    April pseudo windows are built only for days before the last block start and enter only as training rows of
    models for April windows on later days: every label they carry ends before the origin it is used for. This shapes
    April rows only;
  * diagnostics (`report["diag"]`): train-vs-March shift statistics from the masked view (queue share per corridor
    segment, flow levels, a capacity proxy) and an error breakdown of the pseudo-holdout and train CV (site hit,
    predicted vs true size, time of day, queue size). Nothing in it is read back by the pipeline.

v6 (after the v5 diagnostics: the train -> March onset loss is mostly in the *site* - site hit 95% on the train CV,
80% on March, IoU when the site is right 0.91 vs 0.93; part of that gap is scoring, March counts known cells only - and
it sits in the 10-15 h onsets, 98% -> 72%. Each month's demand draw moves on-ramp demand by 10-20% on some corridors
while capacity and mainline flows stay put, so the active bottlenecks move, and the onset model leans on where queues
usually start):
  * `rel_*` features (group "r"): where each link stands among the corridor's links in the same window (rank and gap of
    the speed ratio at T, of its 30-min extrapolation, of the 30-min drop, of flow/capacity and density; local minimum
    within +-3 links). Location-free: ties share a rank and missing values have none, so the link order never enters;
  * `dr_*` features (group "d"), all from the released data <= T: today's demand against the train profile for the
    same weekday and time of day, from midnight to T and over the last hour - on-ramp inflow within 1.5 / 5 km upstream
    of each link and at the merges within 1.5 km downstream (in a merge bottleneck the queue starts on the link
    upstream of the ramp), the mainline flow, per corridor; train windows use the profile without their own week fold,
    test windows the full train profile; on-ramp sums only where every attached on-ramp is valid. Plus the demand
    arriving at the merges against the capacity behind them over the last 15 min (`dr_merge_*`);
  * `noloc2` variant: `noloc` without the static link attributes that act as link identifiers (length, capacity per
    lane, free speed); the capacity ratios to the neighbours (lane drops) stay;
  * candidates `v2n+r`, `v2n+rd`, `v2n+rd` with noloc2, through the two-draw gate with v6 safeguards (higher bars for
    three alternatives, March must also gain on the train-CV route for new feature groups, 3 of 4 corridor families
    must gain, a minimum of March windows). The March rows of the final files stay the v4 recipe (`march_rows`), and
    optional probe files (`probe_onset`: recipes fixed in advance, train-only models) carry other onset recipes on the
    March rows for public reads (the notebook sets v5's April recipe and the same + `rd`, so the two reads separate the
    no-location blend from the new features).

v7 (after the v6 run: +rd lost on March and on the public probe - 0.86208 against 0.86953 - while the train CV liked
it; per link, March re-deals the on-ramp demand - median deviation from the train profile 29-75% per corridor, train
4-week blocks within 2-3% - while mainline flow stays within 1-4%):
  * `ph_*` features (group "p"): a cell-transmission forecast 30 min ahead from the state at T (per-link FD from the
    release, ramp merges, FIFO diverges, the demand seen up to T flat and with its trend): forecast speed ratio,
    queue flags, first queued slot, largest merge ratio and its rank. It knows nothing of the training month, so it
    should travel to a new demand draw (`ctm_features`, `CTMNet`);
  * `_nr` variants: without the ramp-flow columns (link identifiers within one draw);
  * diagnostics: the forecast on its own on train and March onset windows (`report["diag"]["ctm"]`), how far the
    missed onset sites are from the true ones (`misses` in the window diagnostics), and whether the ramp records add
    up to the mainline (`ramp_coupling`);
  * the forecast runs three demand scenarios (flat, 30-min trend, the train time-of-day shape of each link's flow,
    `FlowProf`), takes its ramp terms from mainline flow differences and has a measured discharge limit at the corridor
    end;
  * the ongoing rows (unchanged since v5, ONGOING_RECIPE) are reused from the previous run's files when its settings
    match (`reuse_ongoing`, `reusable_ongoing`): only the onset part is refitted; the v6 candidates are gone.

v8 (after the v7 run: same-month evidence is the strongest signal found - train + the first half of March improved the
second half by +0.079 onset IoU, and the late-window refit +0.009 ongoing - while 29 of April's 40 ongoing windows fall
on April 1-5, before the late block):
  * early blocks (`late_early`, e.g. {"queue_ongoing": [1, 2]}: blocks [day 1, day 2) and [day 2, first late block),
    0 = the first day of the month): the official April windows of a condition in a block are predicted by models that
    also learn from April's own pseudo windows of the days before the block (main: + all of March when its April
    adaptation was adopted), when the same recipe wins on the same blocks of March first (`early_step`, gate =
    `late_gate` with role "egate", pooled over the blocks). Every label used is older than every origin it serves.
    Works on top of reused ongoing rows: only the early windows are recomputed; every run also writes the ongoing rows
    without its early windows (BASE_FILE), which a later run reuses, so the early windows always start from the same base.
  * `vs_previous` in the report: how many March / April cells and windows differ from the files this run replaced.

v9 (after the v8 run: the gap to the top is not closed by Task 1 or by more same-month refits of the same models):
  * 'm' feature group (`memory_features`, mm_*): where queues sat around this time of day on the earlier days of the
    same calendar month - the month's own demand draw - and today before the history window, from the released masked
    view. A month's official windows are its first queue onsets from the 1st, so most of them have earlier days whose
    same peak shows the active bottleneck; the window reaches past the blanked horizon + buffer of those earlier
    windows, and train months get the same blanks (`mem_prep`), so the features mean the same in train and in the test
    months (the v3 recency features failed exactly there: they read the blanked slots). Onset candidate v2n+pm against
    v8's v2n+p, through the two-draw gate; probes for the public read: `probe_onset` and, new, `probe_ongoing`
    (March ongoing rows of the main file from a recipe fixed in advance).
  * reuse_early: a previous v8+ run's early April windows are taken with its ongoing rows when the early settings are
    the same (EARLY_KEYS), instead of refitting them (about 3 h).
"""
from __future__ import annotations

import hashlib
import json
import re
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from . import common as C
from .data import load_cache, load_net, read_pq

warnings.filterwarnings("ignore", category=RuntimeWarning)

VERSION = "v9"
HH = 12       # history slots [T-12, T)
K = 6         # horizon steps T+1..T+6
CTX = 36      # masked-view context before the history (3 h)
SPACING = 72  # 360 min
BIG = 30.0
CONDS = ("queue_onset", "queue_ongoing")


def _shift(a, o):
    """Value at link l+o along the last axis, edge-held."""
    if o == 0:
        return a
    L = a.shape[-1]
    return a[..., np.clip(np.arange(L) + o, 0, L - 1)]


def _nearest_dist(q, mp, direction):
    """q [n, L] bool -> km to the nearest True link downstream (index >= l) / upstream (index <= l)."""
    n, L = q.shape
    out = np.full((n, L), BIG, np.float32)
    cur = np.full(n, np.nan)
    rng_ = range(L - 1, -1, -1) if direction == "dn" else range(L)
    for l in rng_:
        cur = np.where(q[:, l], mp[l], cur)
        d = (cur - mp[l]) if direction == "dn" else (mp[l] - cur)
        out[:, l] = np.where(np.isnan(d), BIG, np.minimum(np.abs(d), BIG))
    return out


def _ffill_time(x):
    """[n, h, L] forward fill along axis 1; returns filled values and age (slots since the last value)."""
    n, h, L = x.shape
    ok = np.isfinite(x)
    idx = np.where(ok, np.arange(h)[None, :, None], -1)
    idx = np.maximum.accumulate(idx, axis=1)
    f = np.take_along_axis(x, np.maximum(idx, 0), 1)
    f = np.where(idx >= 0, f, np.nan)
    age = np.where(idx >= 0, np.arange(h)[None, :, None] - idx, np.nan).astype(np.float32)
    return f.astype(np.float32), age


def _fill_space(x):
    """Fill NaN along the last axis by linear interpolation / edge hold (vectorised)."""
    sh = x.shape
    y = x.reshape(-1, sh[-1]).astype(np.float64)
    L = sh[-1]
    ok = np.isfinite(y)
    ar = np.arange(L)
    li = np.where(ok, ar[None, :], -1); li = np.maximum.accumulate(li, 1)
    ri = np.where(ok, ar[None, :], L); ri = np.minimum.accumulate(ri[:, ::-1], 1)[:, ::-1]
    rows = np.arange(y.shape[0])[:, None]
    lv = np.where(li >= 0, y[rows, np.clip(li, 0, L - 1)], np.nan)
    rv = np.where(ri < L, y[rows, np.clip(ri, 0, L - 1)], np.nan)
    wl = np.where((li >= 0) & (ri < L), (ri - ar) / np.maximum(ri - li, 1), np.nan)
    out = np.where(np.isfinite(wl), wl * lv + (1 - wl) * rv, np.where(np.isfinite(lv), lv, rv))
    out = np.where(ok, y, out)
    return out.reshape(sh).astype(np.float32)


# ----------------------------------------------------------------------------- panel
class P2:
    def __init__(self, panel: str):
        self.panel = panel
        self.net = net = load_net(panel)
        c = load_cache(panel, ["v", "q", "o", "tv", "tq", "to", "el", "pct", "ron", "roff", "tgt", "dark", "non", "noff"])
        self.L = net.L; self.NT = c["v"].shape[0]
        self.vf = net.vf.astype(np.float32); self.vcut = net.vcut.astype(np.float32)
        self.lanes = net.lanes.astype(np.float32); self.cap = net.cap.astype(np.float32)
        self.kc = net.kc.astype(np.float32); self.kjam = net.kjam.astype(np.float32)
        self.length = net.length.astype(np.float32); self.mp = net.mp.astype(np.float64)
        self.mv, self.mq, self.mo = c["v"], c["q"], c["o"]
        self.el, self.pct = c["el"], c["pct"]
        self.ron, self.roff = c["ron"], c["roff"]
        self.non = c.get("non")      # number of valid on-ramps per (slot, link) (v6)
        self.noff = c.get("noff")    # number of valid off-ramps per (slot, link) (v7)
        self.tgt = c["tgt"]          # Task 1 target cells (hidden in the masked view)
        self.dark = c["dark"]        # official Task 2 horizon + buffer rows (blank in the masked view)
        tr = C.split_days("train")
        self.ntr = int(tr.max() + 1) * C.SLOTS
        self.tv, self.tq, self.to = c["tv"][: self.ntr], c["tq"][: self.ntr], c["to"][: self.ntr]
        self.wd_day = C.calendar().weekday.to_numpy()
        on = np.zeros(self.L); off = np.zeros(self.L)
        for r in net.ramps.itertuples():
            (on if r.kind == "ON" else off)[r.li] += 1
        self.n_on, self.n_off = on, off
        self.Q = None
        self._ron_full = None
        self._ramp_est = {}

    def ron_full(self):
        """v6: on-ramp inflow per link only where every attached on-ramp is valid (NaN where only some are: a
        partial sum would read as a demand drop; a quarter of the released ramp cells are not score-eligible)."""
        if self._ron_full is None:
            if self.non is None:
                self._ron_full = self.ron
            else:
                need = self.n_on.astype(np.int16)[None, :]
                self._ron_full = np.where(self.non.astype(np.int16) >= need, self.ron, np.nan).astype(np.float32)
        return self._ron_full

    def ramp_est(self, kind):
        """v7: on- ("on") or off-ramp ("off") flow per link scaled to all attached ramps (sum of the valid ones x
        attached / valid); NaN where none is valid. 0 on links without ramps of that kind."""
        if kind not in self._ramp_est:
            arr, cnt, need = (self.ron, self.non, self.n_on) if kind == "on" else (self.roff, self.noff, self.n_off)
            need = np.asarray(need, np.float32)[None, :]
            if cnt is None:
                est = arr.astype(np.float32)
            else:
                c = cnt.astype(np.float32)
                est = np.where(c > 0, arr * need / np.maximum(c, 1), np.nan).astype(np.float32)
            self._ramp_est[kind] = np.where(need > 0, est, 0.0).astype(np.float32)
        return self._ramp_est[kind]

    # ------------------------------------------------------------------ recency (released masked view only)
    def recency_prep(self):
        """Per-day queue activity from the masked view: share of observed slots queued per day and per 30-min bin,
        cumulative sums for "today so far", calendar-month block and day type of every day."""
        if getattr(self, "q_bin", None) is not None:
            return
        q = np.where(np.isfinite(self.mv), (self.mv <= self.vcut[None, :]).astype(np.float32), np.nan)
        nd = self.NT // C.SLOTS
        qd = q[: nd * C.SLOTS].reshape(nd, C.SLOTS, self.L)
        with np.errstate(all="ignore"):
            self.q_day = np.nanmean(qd, 1).astype(np.float32)                                   # [ND, L]
            self.q_bin = np.nanmean(qd.reshape(nd, 48, 6, self.L), 2).astype(np.float32)        # [ND, 48, L]
        fin = np.isfinite(q)
        self.cq = np.vstack([np.zeros((1, self.L), np.float32), np.cumsum(np.where(fin, q, 0), 0, dtype=np.float32)])
        self.cn = np.vstack([np.zeros((1, self.L), np.int32), np.cumsum(fin, 0, dtype=np.int32)])
        cal = C.calendar()
        dt = pd.to_datetime(cal.date)
        self.mblock = (dt.dt.year * 12 + dt.dt.month).to_numpy(np.int64)                       # calendar month
        self.daytype = (cal.weekday.to_numpy() >= 5).astype(np.int8)                             # 1 = weekend
        del q, qd, fin

    def mem_prep(self):
        """v9: queued / known counts of the released masked view, cumulative over the global time axis, for the month
        memory (`memory_features`). In March / April the release blanks every official window's horizon and the hour
        after it; each train calendar month gets the same blanks, from the selector replica started on its first day
        (5 + 5 windows, 360 min apart), so the memory reads the same kind of record in train as in the test months.
        Needs the train labels (build_labels) for that replica; without them nothing extra is blanked."""
        if getattr(self, "mcq", None) is not None:
            return
        self.recency_prep()
        q = np.where(np.isfinite(self.mv), (self.mv <= self.vcut[None, :]).astype(np.float32), np.nan)
        self.mem_blank = 0
        if self.Q is not None:
            st = self.selector_stats()
            days = C.split_days("train")
            mb = self.mblock[days]
            blank = np.zeros(self.NT, bool)
            for m in np.unique(mb):
                dm = days[mb == m]
                for T, _ in self.greedy(st, int(dm.min()) * C.SLOTS, (int(dm.max()) + 1) * C.SLOTS):
                    blank[T + 1:T + 19] = True
            q[blank] = np.nan
            self.mem_blank = int(blank.sum())
        # the test months: the official windows' horizon + buffer rows read as blank on every link, as in the train
        # simulation above (the release blanks them on the eligible cells; this makes the two the same whatever it shows)
        trow = np.zeros(self.NT, bool)
        for split_ in C.TEST_SPLITS:
            if len(C.split_days(split_)):
                trow[C.split_rows(split_)] = True
        q[self.dark.astype(bool) & trow] = np.nan
        fin = np.isfinite(q)
        self.mcq = np.vstack([np.zeros((1, self.L), np.float32), np.cumsum(np.where(fin, q, 0), 0, dtype=np.float32)])
        self.mcn = np.vstack([np.zeros((1, self.L), np.int32), np.cumsum(fin, 0, dtype=np.int32)])
        del q, fin

    # ------------------------------------------------------------------ labels
    def ratio_train(self):
        return (self.tv / self.vcut[None, :]).astype(np.float32)

    def build_labels(self, imputer=None):
        """Queue truth on train: observed speed <= v_cut; gaps filled in time (<=3), space (<=2), time (<=12);
        missing cells next to observed queues are decided by the space-time imputer."""
        R = self.ratio_train()
        f = pd.DataFrame(R).interpolate(limit=3, limit_area="inside").to_numpy(np.float32)
        f = pd.DataFrame(f.T).interpolate(limit=2, limit_area="inside").to_numpy(np.float32).T
        f = pd.DataFrame(f).interpolate(limit=12, limit_area="inside").to_numpy(np.float32)
        Q = np.nan_to_num(f, nan=9.0) <= 1.0
        if imputer is not None:
            obsq = np.nan_to_num(R, nan=9.0) <= 1.0
            near = np.zeros_like(obsq)
            for a in (-1, 0, 1):
                for b in (-2, -1, 0, 1, 2):
                    near |= np.roll(np.roll(obsq, a, 0), b, 1)
            t, l = np.nonzero(np.isnan(R) & near)
            for i in range(0, len(t), 1_000_000):
                tt, ll = t[i:i + 1_000_000], l[i:i + 1_000_000]
                X = imp_features(R, tt, ll)
                Q[tt, ll] = imputer.predict(X, num_threads=C.THREADS) >= 0.5
        self.Q = Q
        return Q

    # ------------------------------------------------------------------ selector
    def selector_stats(self):
        if getattr(self, "_st", None) is not None and self._st[0] is self.Q:   # v9: also used by mem_prep
            return self._st[1]
        Q = self.Q; ntr = self.ntr; L = self.L
        E = self.el[:ntr].astype(np.int32)
        Qo = ((np.nan_to_num(self.tv, nan=999.0) <= self.vcut[None, :]) & self.el[:ntr]).astype(np.int32)
        cE = np.vstack([np.zeros((1, L), np.int32), np.cumsum(E, 0)])
        cQ = np.vstack([np.zeros((1, L), np.int32), np.cumsum(Qo, 0)])
        T = np.arange(HH, ntr - K - 1)
        tod = T % C.SLOTS
        T = T[(tod >= HH) & (tod < C.SLOTS - 18)]
        cov = (cE[T] - cE[T - HH]).sum(1) / (HH * L)
        per = cQ[T] - cQ[T - HH]
        nq = per.sum(1); mx = per.max(1)
        ongoing = (nq >= 2) & (mx >= 2)
        QT = Q[T]
        inter = np.zeros(len(T)); union = np.zeros(len(T)); nst = np.zeros((len(T), K), np.int32)
        for k in range(1, K + 1):
            Qk = Q[T + k]; nst[:, k - 1] = Qk.sum(1)
            inter += (QT & Qk).sum(1); union += (QT | Qk).sum(1)
        piou = np.where(union == 0, 1.0, inter / np.maximum(union, 1))
        df = pd.DataFrame(dict(T=T, cov=cov, nq=nq, ongoing=ongoing, piou=piou, nfut=nst.sum(1), nT=QT.sum(1)))
        for k in range(K):
            df[f"n{k + 1}"] = nst[:, k]
        df["cand"] = (df["cov"] >= 0.7) & (df.nfut > 0) & (df.piou <= 0.9)
        self._st = (Q, df)
        return df

    @staticmethod
    def greedy(st, start, end, quota=5):
        c = st[st.cand & (st["T"] >= start) & (st["T"] < end)]
        acc = []; cnt = {True: 0, False: 0}
        for T, og in zip(c["T"].to_numpy(), c.ongoing.to_numpy()):
            if cnt[og] >= quota:
                continue
            if all(abs(T - a) >= SPACING for a, _ in acc):
                acc.append((int(T), bool(og))); cnt[og] += 1
                if cnt[True] >= quota and cnt[False] >= quota:
                    break
        return acc

    def windows(self, rng, max_ongoing_cand=1500, horizon_days=40):
        """sim: the selector started on every train day; cand: training superset. Columns T, condition, src."""
        st = self.selector_stats()
        nd = self.ntr // C.SLOTS
        sim = {}
        for s in range(nd):
            for T, og in self.greedy(st, s * C.SLOTS, min(self.ntr, (s + horizon_days) * C.SLOTS)):
                sim[(T, og)] = sim.get((T, og), 0) + 1
        W = [dict(T=T, ongoing=og, src="sim", draws=n) for (T, og), n in sim.items()]
        s2 = st.set_index("T")
        cand = s2[s2.cand]
        on = cand[~cand.ongoing]
        prev = s2.cand & ~s2.ongoing
        prev = prev.shift(1, fill_value=False)
        first = on[~prev.reindex(on.index).fillna(False).to_numpy()]
        W += [dict(T=int(T), ongoing=False, src="cand", draws=0) for T in first.index if (int(T), False) not in sim]
        og = cand[cand.ongoing].index.to_numpy()
        og = np.array([T for T in og if (int(T), True) not in sim])
        if len(og) > max_ongoing_cand:
            og = np.sort(rng.choice(og, max_ongoing_cand, replace=False))
        W += [dict(T=int(T), ongoing=True, src="cand", draws=0) for T in og]
        W = pd.DataFrame(W, columns=["T", "ongoing", "src", "draws"])
        if W.empty:
            W["condition"] = []; W["day"] = []
            return W, np.zeros((0, K, self.L), bool)
        W["condition"] = np.where(W.ongoing, "queue_ongoing", "queue_onset")
        y = self.Q[W["T"].to_numpy()[:, None] + np.arange(1, K + 1)[None, :]]  # [n, K, L]
        # onset windows: the official truth sits at T+30 only; keep the windows that look like that
        keep = W.ongoing.to_numpy() | (~y[:, :K - 1].any((1, 2)) & y[:, K - 1].any(1))
        W = W[keep].reset_index(drop=True); y = y[keep]
        W["day"] = W["T"] // C.SLOTS
        return W, y

    # ------------------------------------------------------------------ inputs
    def inputs_train(self, Ts):
        Ts = np.asarray(Ts, np.int64)
        idx = Ts[:, None] + np.arange(-HH, 0)[None, :]
        return dict(hv=self.tv[idx], hq=self.tq[idx], ho=self.to[idx], he=self.el[idx])

    def inputs_official(self, split):
        d = C.rel() / "task2" / self.panel / split
        wi = pd.read_csv(d / "window_index.csv")
        wi["T"] = C.tindex(wi.forecast_origin)
        h = read_pq(d / "window_history.parquet")
        lid = self.net.lid
        n = len(wi); L = self.L
        A = {k: np.full((n, HH, L), np.nan, np.float32) for k in ("hv", "hq", "ho")}
        A["he"] = np.zeros((n, HH, L), bool)
        pos = {w: i for i, w in enumerate(wi.window_id)}
        h = h[h.window_id.isin(pos)]
        wi_i = h.window_id.map(pos).to_numpy()
        ti = C.tindex(h.timestamp) - (wi["T"].to_numpy()[wi_i] - HH)
        li = h.link_id.astype(str).map(lid)
        ok = li.notna().to_numpy() & (ti >= 0) & (ti < HH)
        wi_i, ti, li = wi_i[ok], ti[ok], li[ok].to_numpy(np.int64)
        hh = h[ok]
        A["hv"][wi_i, ti, li] = pd.to_numeric(hh.speed_kmh, errors="coerce").to_numpy(np.float32)
        if "flow_vph" in hh:
            A["hq"][wi_i, ti, li] = pd.to_numeric(hh.flow_vph, errors="coerce").to_numpy(np.float32)
        if "occupancy" in hh:
            A["ho"][wi_i, ti, li] = pd.to_numeric(hh.occupancy, errors="coerce").to_numpy(np.float32)
        if "is_score_eligible" in hh:
            A["he"][wi_i, ti, li] = hh.is_score_eligible.fillna(False).astype(bool).to_numpy()
        return wi, A


# ----------------------------------------------------------------------------- imputer for labels
OFFS = [(-1, 0), (1, 0), (-2, 0), (2, 0), (0, -1), (0, 1), (0, -2), (0, 2), (-1, -1), (-1, 1), (1, -1), (1, 1)]


def imp_features(R, t, l):
    Tn, L = R.shape
    X = np.full((len(t), len(OFFS) + 2), np.nan, np.float32)
    for j, (dt, dl) in enumerate(OFFS):
        tt = t + dt; ll = l + dl
        ok = (tt >= 0) & (tt < Tn) & (ll >= 0) & (ll < L)
        X[ok, j] = R[tt[ok], ll[ok]]
    with np.errstate(all="ignore"):
        X[:, -2] = np.nanmean(X[:, 0:2], 1); X[:, -1] = np.nanmean(X[:, 4:6], 1)
    return X


def fit_imputer(panels, n_per_panel=150_000, seed=0):
    import lightgbm as lgb
    rng = np.random.default_rng(seed)
    Xs, ys = [], []
    for p in panels:
        P = P2(p); R = P.ratio_train()
        slow = np.nan_to_num(R, nan=9.0) <= 1.3
        near = np.zeros_like(slow)
        for a in (-1, 0, 1):
            for b in (-1, 0, 1):
                near |= np.roll(np.roll(slow, a, 0), b, 1)
        t, l = np.nonzero(near & np.isfinite(R))
        if len(t) > n_per_panel:
            s = rng.choice(len(t), n_per_panel, replace=False); t, l = t[s], l[s]
        Xs.append(imp_features(R, t, l)); ys.append((R[t, l] <= 1).astype(np.float32))
    prm = dict(objective="binary", learning_rate=0.1, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.9,
               bagging_fraction=0.8, bagging_freq=1, num_threads=C.THREADS, verbose=-1, seed=seed)
    return lgb.train(prm, lgb.Dataset(np.vstack(Xs), np.concatenate(ys)), 300)


# ----------------------------------------------------------------------------- profiles
class Prof:
    """Weekday x time-of-day queue probability per link from train labels. Training windows get the
    profile without their own week-block fold (out-of-fold, no label leakage); test windows the full one."""

    NF = 4

    def __init__(self, P: P2, alpha=5.0):
        nd = P.ntr // C.SLOTS
        Qd = P.Q[: nd * C.SLOTS].reshape(nd, C.SLOTS, P.L).astype(np.float32)
        Qd = (np.roll(Qd, 1, 1) + 2 * Qd + np.roll(Qd, -1, 1)) / 4
        wd = P.wd_day[:nd]; fold = (np.arange(nd) // 7) % self.NF
        self.S = np.zeros((self.NF, 7, C.SLOTS, P.L), np.float32); self.N = np.zeros((self.NF, 7), np.float32)
        for f in range(self.NF):
            for w in range(7):
                m = (fold == f) & (wd == w)
                if m.any():
                    self.S[f, w] = Qd[m].sum(0); self.N[f, w] = m.sum()
        R = np.where(np.isfinite(P.tv), P.tv / P.vcut[None, :], np.nan)[: nd * C.SLOTS].reshape(nd, C.SLOTS, P.L)
        self.RS = np.stack([np.nansum(R[fold == f], 0) for f in range(self.NF)]).astype(np.float32)
        self.RN = np.stack([np.isfinite(R[fold == f]).sum(0) for f in range(self.NF)]).astype(np.float32)
        self.alpha = alpha

    def _tot(self, arr, fold):
        tot = arr.sum(0)
        return tot if fold < 0 else tot - arr[fold]

    def get(self, folds, wds, tods):
        """pq [n, L] at (weekday, tod) without the given fold (-1 = all train days), and the all-weekday rate."""
        folds = np.asarray(folds); wds = np.asarray(wds); tods = np.asarray(tods) % C.SLOTS
        out = np.zeros((len(folds), self.S.shape[-1]), np.float32); allp = np.zeros_like(out)
        for f in np.unique(folds):
            m = folds == f
            S = self._tot(self.S, f); N = self._tot(self.N, f)
            Sa = S.sum(0); Na = N.sum()
            a = Sa[tods[m]] / max(Na, 1)
            out[m] = (S[wds[m], tods[m]] + self.alpha * a) / (N[wds[m]][:, None] + self.alpha)
            allp[m] = a
        return out, allp

    def rmean(self, folds, tods):
        folds = np.asarray(folds); tods = np.asarray(tods) % C.SLOTS
        out = np.zeros((len(folds), self.RS.shape[-1]), np.float32)
        for f in np.unique(folds):
            m = folds == f
            out[m] = self._tot(self.RS, f)[tods[m]] / np.maximum(self._tot(self.RN, f)[tods[m]], 1)
        return out


class DemandProf:
    """v6. Today's demand against the train profile, from the released data only. For every day of the release:
    the train mean of the channel by weekday x 5-min slot x link - for a train day without that day's own week fold
    (the folds of Prof), for a test day over all train days - and within-day running sums of the channel and of that
    profile over the slots where both exist. A window at T then reads "since midnight" and "last hour" ratios in O(1).
    Channels: on-ramp inflow per mainline link (only slots where every attached on-ramp is valid: a partial sum would
    read as a demand drop) and the mainline flow."""

    def __init__(self, P: P2):
        L = P.L; nd_tr = P.ntr // C.SLOTS; nd = P.NT // C.SLOTS
        wd = np.asarray(P.wd_day)
        fold = (np.arange(nd_tr) // 7) % Prof.NF
        self.cX, self.cM = {}, {}
        for ch, arr in (("on", P.ron_full()), ("q", P.mq)):
            Xtr = arr[: nd_tr * C.SLOTS].reshape(nd_tr, C.SLOTS, L)
            S = np.zeros((Prof.NF, 7, C.SLOTS, L), np.float64); N = np.zeros_like(S)
            for f in range(Prof.NF):
                for w in range(7):
                    m = (fold == f) & (wd[:nd_tr] == w)
                    if m.any():
                        x = Xtr[m]; fin = np.isfinite(x)
                        S[f, w] = np.where(fin, x, 0).sum(0, dtype=np.float64); N[f, w] = fin.sum(0)
            X = arr[: nd * C.SLOTS].reshape(nd, C.SLOTS, L)
            cx = np.zeros((nd, C.SLOTS, L), np.float32); cm = np.zeros_like(cx)
            for d in range(nd):
                w = wd[d]
                others = [f for f in range(Prof.NF) if d >= nd_tr or f != fold[d]]   # a train day: not its own fold
                s_, n_ = S[others, w].sum(0), N[others, w].sum(0)
                mo = np.where(n_ > 0, s_ / np.maximum(n_, 1), np.nan)
                ok = np.isfinite(X[d]) & np.isfinite(mo)
                cx[d] = np.cumsum(np.where(ok, X[d], 0), 0, dtype=np.float64)
                cm[d] = np.cumsum(np.where(ok, mo, 0), 0, dtype=np.float64)
            self.cX[ch] = cx.reshape(-1, L); self.cM[ch] = cm.reshape(-1, L)
            del S, N, cx, cm
        # spatial windows (km along the corridor; mp = downstream end of each link): upstream ramps within 1.5 / 5 km
        # (the link itself included) and the merges just downstream (links starting within 1.5 km of this link's end,
        # at least the next link) - in a merge bottleneck the queue starts on the link upstream of the ramp
        mp = np.asarray(P.mp, np.float64); start = mp - np.asarray(P.length, np.float64)
        self.up_lo = {kt: np.minimum(np.searchsorted(mp, mp - km - 1e-9, side="left"), np.arange(L))
                      for kt, km in (("n", 1.5), ("w", 5.0))}
        self.dn_hi = np.clip(np.maximum(np.searchsorted(start, mp + 1.5 + 1e-9, side="right") - 1, np.arange(L) + 1), 0, L - 1)


def _ratio(a, b, floor):
    with np.errstate(all="ignore"):
        return np.where(b > floor, np.clip(a / np.maximum(b, 1e-9), 0, 5), np.nan).astype(np.float32)


def _range_up(a, lo):
    """[n, L] -> sum over links lo[l]..l (link order runs downstream)."""
    c = np.cumsum(a, 1, dtype=np.float64)
    return c - np.where(lo[None, :] > 0, c[:, np.maximum(lo - 1, 0)], 0)


def _range_dn(a, hi):
    """[n, L] -> sum over links l+1..hi[l] (empty, i.e. 0, when hi[l] <= l)."""
    c = np.cumsum(a, 1, dtype=np.float64)
    L = a.shape[1]
    return np.where(hi[None, :] > np.arange(L)[None, :], c[:, hi] - c, 0)


def demand_features(P: P2, Ts, use_T=True):
    """v6 dr_* features of windows at Ts (data <= T only; the origin slot itself only with use_T). Floors (in vph x
    slots; /12 = vehicles): 600 for on-ramp sums, 12000 for the mainline over 3 links, 3000 / 120000 per corridor."""
    dp = P.dprof
    Ts = np.asarray(Ts, np.int64)
    Te = Ts if use_T else Ts - 1
    prev = Te - 12
    same = (prev // C.SLOTS) == (Te // C.SLOTS)
    out = {}
    for ch in ("on", "q"):
        cX, cM = dp.cX[ch], dp.cM[ch]
        xt, mt = cX[Te].astype(np.float64), cM[Te].astype(np.float64)
        xp = np.where(same[:, None], cX[np.maximum(prev, 0)], 0); mp_ = np.where(same[:, None], cM[np.maximum(prev, 0)], 0)
        for tag, xs, ms in (("today", xt, mt), ("hour", xt - xp, mt - mp_)):
            if ch == "on":
                for kt, lo in dp.up_lo.items():
                    out[f"dr_on_{tag}_{kt}"] = _ratio(_range_up(xs, lo), _range_up(ms, lo), 600.0)
                out[f"dr_on_{tag}_d"] = _ratio(_range_dn(xs, dp.dn_hi), _range_dn(ms, dp.dn_hi), 600.0)
            else:
                sx = xs + _shift_np(xs, -1) + _shift_np(xs, 1); sm = ms + _shift_np(ms, -1) + _shift_np(ms, 1)
                out[f"dr_q_{tag}"] = _ratio(sx, sm, 12000.0)
            out[f"dr_{ch}_{tag}_panel"] = np.repeat(_ratio(xs.sum(1), ms.sum(1), 3000.0 if ch == "on" else 120000.0)[:, None],
                                                    xs.shape[1], 1)
    return out


def merge_features(P: P2, Ts, use_T=True):
    """v6 dr_merge_*: demand arriving at a merge against the capacity behind it, over the last 15 min up to T (or
    T-1): own = (flow on l-1 + on-ramps of l) / cap(l), dn1 = (flow on l + on-ramps of l+1) / cap(l+1) - when it
    nears 1 the queue starts on l - and dn2 = dn1 of the next link. A link without on-ramps adds 0; a link whose
    on-ramps are not all valid gives NaN."""
    Ts = np.asarray(Ts, np.int64)
    Te = Ts if use_T else Ts - 1
    idx = np.clip(Te[:, None] + np.arange(-2, 1)[None, :], 0, P.NT - 1)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        q = np.nanmean(P.mq[idx], 1)
        r = np.nanmean(P.ron_full()[idx], 1)
    r0 = np.where(P.n_on[None, :] > 0, r, 0.0)
    cap = np.maximum(P.cap[None, :].astype(np.float64), 1.0)
    own = (_shift_nan(q, -1) + r0) / cap
    dn1 = (q + _shift_nan(r0, 1)) / _shift_nan(cap, 1)
    return {"dr_merge_own": own.astype(np.float32), "dr_merge_dn1": dn1.astype(np.float32),
            "dr_merge_dn2": _shift_nan(dn1, 1).astype(np.float32)}


def _shift_np(a, o):
    """Value at link l+o, zero outside the corridor (for sums)."""
    L = a.shape[1]
    idx = np.arange(L) + o
    ok = (idx >= 0) & (idx < L)
    return np.where(ok[None, :], a[:, np.clip(idx, 0, L - 1)], 0)


def _shift_nan(a, o):
    """Value at link l+o, NaN outside the corridor."""
    L = a.shape[1]
    idx = np.arange(L) + o
    ok = (idx >= 0) & (idx < L)
    return np.where(ok[None, :], a[:, np.clip(idx, 0, L - 1)], np.nan)


def _rank_rows(a, ascending=True, cap=40):
    """Rank of every link within its window (row), 0 = first; ties share the smallest rank and a missing value has
    no rank (NaN), so the link order never enters (the no-location variants keep these columns); capped."""
    n, L = a.shape
    fin = np.isfinite(a)
    x = np.where(fin, a if ascending else -a, np.inf)
    o = np.argsort(x, axis=1, kind="stable")
    xs = np.take_along_axis(x, o, 1)
    brk = np.ones((n, L), bool); brk[:, 1:] = xs[:, 1:] != xs[:, :-1]
    first = np.maximum.accumulate(np.where(brk, np.arange(L)[None, :], 0), axis=1)   # first position of each tie group
    r = np.empty((n, L), np.int64)
    np.put_along_axis(r, o, first, axis=1)
    return np.where(fin, np.minimum(r, cap), np.nan).astype(np.float32)


# ----------------------------------------------------------------------------- v7: cell-transmission forecast
# The release was generated by a calibrated traffic-flow model (README): per-link free speed, capacity and jam density
# (fd_parameters.csv), demand by weekday and time of day, a new demand draw every test month. A queue starts where the
# demand arriving at a merge exceeds the capacity behind it - wherever this month's demand puts it. The forecast below
# runs a cell-transmission model (one cell per link, triangular FD, proportional merges, FIFO diverges, a measured
# discharge limit at the corridor end) 30 min ahead from the state at T and hands the result to the onset model as
# features (group "p", prefix ph_).
#   * Demand: the mainline flow of the last 15 min on every link, held flat ("" features), with its 30-min trend ("t")
#     and with the train time-of-day shape of that link's flow ("p"; out-of-fold for training windows). The net ramp
#     terms are the differences of consecutive links' flows, so an uncongested corridor reproduces the observed flow
#     profile exactly; the released ramp records are not needed (and the ramps of links without a detector count too).
#   * Speed of a link in a 5-min slot = its mean outflow / mean density, as a detector measures it; queued <= v_cut.
# Only data <= T (or < T without use_T) enters; the shape and the discharge limit come from train days only.
CTM_DT_S = (1.0, 10.0)      # bounds of the time step (s); shorter cells are lengthened to keep the CFL condition
CTM_SCEN = ("", "t", "p")   # flat, trend, time-of-day profile


class FlowProf:
    """v7. Weekday x 5-min slot mean of the mainline flow per link over train days (released masked view), each train
    day's own week fold left out (the folds of Prof): the time-of-day shape of the demand."""

    def __init__(self, P: P2):
        nd = P.ntr // C.SLOTS
        X = P.mq[: nd * C.SLOTS].reshape(nd, C.SLOTS, P.L)
        wd = np.asarray(P.wd_day)[:nd]; fold = (np.arange(nd) // 7) % Prof.NF
        self.S = np.zeros((Prof.NF, 7, C.SLOTS, P.L), np.float64); self.N = np.zeros_like(self.S)
        for f in range(Prof.NF):
            for w in range(7):
                m = (fold == f) & (wd == w)
                if m.any():
                    x = X[m]; fin = np.isfinite(x)
                    self.S[f, w] = np.where(fin, x, 0).sum(0); self.N[f, w] = fin.sum(0)
        self.St, self.Nt = self.S.sum(0), self.N.sum(0)

    def shape(self, folds, wds, tods):
        """[n, K + 1, L]: demand factor at T + 0, 5, ..., 30 min against T (3-slot smoothed means, clipped to
        0.6 .. 1.8; 1 where unknown)."""
        folds = np.asarray(folds); wds = np.asarray(wds); tods = np.asarray(tods) % C.SLOTS
        n, L = len(tods), self.S.shape[-1]
        idx = np.clip(tods[:, None] + np.arange(-1, K + 2)[None, :], 0, C.SLOTS - 1)       # [n, K + 3]
        out = np.ones((n, K + 1, L))
        for f in np.unique(folds):
            m = folds == f
            S = self.St if f < 0 else self.St - self.S[f]
            N = self.Nt if f < 0 else self.Nt - self.N[f]
            w_, i_ = wds[m][:, None], idx[m]
            with np.errstate(all="ignore"):
                mu = np.where(N[w_, i_] > 0, S[w_, i_] / np.maximum(N[w_, i_], 1), np.nan)       # [nm, K + 3, L]
                sm = (mu[:, :-2] + mu[:, 1:-1] + mu[:, 2:]) / 3.0
                r = sm / sm[:, :1]
            out[m] = np.where(np.isfinite(r), np.clip(r, 0.6, 1.8), 1.0)
        return out


class CTMNet:
    """Per-panel constants of the forecast."""

    def __init__(self, P: P2):
        self.L = P.L
        length = np.maximum(np.asarray(P.length, np.float64), 0.005)              # km
        self.vf = np.maximum(np.asarray(P.vf, np.float64), 30.0)                  # km/h
        self.Q = np.maximum(np.asarray(P.cap, np.float64), 300.0)                 # veh/h
        self.kc = self.Q / self.vf
        self.kj = np.maximum(np.asarray(P.kjam, np.float64), self.kc * 1.05)
        self.w = self.Q / np.maximum(self.kj - self.kc, 1e-6)                     # congested wave speed (km/h)
        self.vcut = np.clip(np.asarray(P.vcut, np.float64), 1.0, None)
        cmax = np.maximum(self.vf, self.w)                                         # fastest wave of each cell
        dt_s = float(np.clip(0.95 * float(np.min(length / cmax)) * 3600.0, *CTM_DT_S))
        self.steps = int(np.ceil(300.0 / dt_s))                                   # steps per 5-min slot
        self.dt_h = 300.0 / self.steps / 3600.0
        self.len = np.maximum(length, cmax * self.dt_h / 0.95)                    # CFL: lengthen very short cells
        self.stretched = int((self.len > length + 1e-12).sum())
        self.mp = np.asarray(P.mp, np.float64)
        self.prof = FlowProf(P)
        # discharge limit at the corridor end: the end link's flow while it is queued on train weekdays 06-20 h
        # (out-of-fold). None (free outflow) unless that queue recurs (>= 0.5% of the observed slots, on >= 5 days)
        # and the limit is at least 0.6 x the end link's capacity (incidents or bad records must not wall the end)
        nd = P.ntr // C.SLOTS
        v = P.mv[: nd * C.SLOTS, -1]; q = P.mq[: nd * C.SLOTS, -1]
        sl = np.arange(nd * C.SLOTS); day = sl // C.SLOTS; hr = (sl % C.SLOTS) / 12.0
        ok = np.isfinite(v) & np.isfinite(q) & (q > 0) & (np.asarray(P.wd_day)[day] < 5) & (hr >= 6) & (hr < 20)
        qd = ok & (v <= P.vcut[-1])
        fold = (day // 7) % Prof.NF
        def rout(sel):
            m = sel & qd
            if m.sum() < 50 or m.sum() < 0.005 * max((sel & ok).sum(), 1) or len(np.unique(day[m])) < 5:
                return np.inf
            r = float(np.median(q[m]))
            return r if r >= 0.6 * self.Q[-1] else np.inf
        self.rout_all = rout(np.ones_like(ok))
        self.rout_fold = np.array([rout(fold != f) for f in range(Prof.NF)])

    def r_out(self, folds):
        folds = np.asarray(folds)
        return np.where(folds < 0, self.rout_all, self.rout_fold[np.clip(folds, 0, Prof.NF - 1)])


def _ctm_run(net: CTMNet, k0, qfun, r_out):
    """One forecast. qfun(ts) -> [n, L] forecast mainline flow per link at ts slots after T (the demand: upstream
    inflow = link 0, net ramp terms = differences of consecutive links). Returns the slot speeds at T+5..T+30
    [K, n, L] (mean outflow / mean density), the largest merge ratio seen by each link over the horizon (demand
    arriving at the next link's entry / what it can receive; the end link: its sending / the discharge limit) and the
    largest k / k_crit."""
    n, L = k0.shape
    k = np.clip(k0, 0.0, net.kj[None, :])
    dt = net.dt_h
    ln, vf, Q, kj, w = (a[None, :] for a in (net.len, net.vf, net.Q, net.kj, net.w))
    rout = np.asarray(r_out, np.float64)
    spd = np.empty((K, n, L))
    mrg = np.zeros((n, L)); kx = k / net.kc[None, :]
    md = np.empty((n, L)); out = np.empty((n, L)); d_on = np.zeros((n, L)); keep = np.ones((n, L))
    upd = max(1, net.steps // 10)                       # demand refreshed 10 times a slot
    for s_ in range(K):
        so = np.zeros((n, L)); sk = np.zeros((n, L))
        for j in range(net.steps):
            if j % upd == 0:
                qh = np.maximum(qfun(s_ + (j + 0.5) / net.steps), 0.0)
                d_up = qh[:, 0]
                if L > 1:
                    dq = qh[:, 1:] - qh[:, :-1]
                    d_on[:, 1:] = np.maximum(dq, 0.0)
                    keep[:, :-1] = 1.0 - np.clip(np.maximum(-dq, 0.0) / np.maximum(qh[:, :-1], 100.0), 0.0, 0.95)
            S = np.minimum(vf * k, Q)
            R = np.minimum(Q, w * (kj - k))
            md[:, 0] = d_up
            if L > 1:
                md[:, 1:] = S[:, :-1] * keep[:, :-1]
            tot = md + d_on
            sc = np.where(tot > R, R / np.maximum(tot, 1e-9), 1.0)               # proportional merge
            fin = md * sc
            if L > 1:
                out[:, :-1] = S[:, :-1] * sc[:, 1:]                               # FIFO diverge (off-ramp share leaves too)
                mrg[:, :-1] = np.maximum(mrg[:, :-1], tot[:, 1:] / np.maximum(R[:, 1:], 1e-9))
            out[:, -1] = np.minimum(S[:, -1], rout)
            mrg[:, -1] = np.maximum(mrg[:, -1], S[:, -1] / np.maximum(rout, 1e-9))
            k = np.clip(k + dt / ln * (fin + d_on * sc - out), 0.0, kj)
            kx = np.maximum(kx, k / net.kc[None, :])
            so += out; sk += k
        spd[s_] = np.where(sk > 1e-9, np.minimum(so / np.maximum(sk, 1e-9), vf), vf)
    return spd, np.minimum(mrg, 10.0), kx


def ctm_features(P: P2, Ts, folds, last_s, q15, fq_slope30, use_T=True):
    """v7 ph_* features: the cell-transmission forecast from the state at T (speed ratio at T and the flow level at T:
    the 15-min mean plus one slot of trend, both spatially filled), three demand scenarios (flat, trend, time-of-day
    profile), and the profile's load forecast (flow / capacity at T+30 and its maximum) without the model."""
    net = getattr(P, "ctm", None)
    if net is None:
        net = P.ctm = CTMNet(P)
    Ts = np.asarray(Ts, np.int64); n, L = len(Ts), P.L
    folds = np.asarray(folds)
    v0 = np.nan_to_num(np.asarray(last_s, np.float64), nan=1.6) * net.vcut[None, :]
    slope = np.nan_to_num(np.asarray(fq_slope30, np.float64)) * net.Q[None, :]   # veh/h per slot
    # the flow level at T: the 15-min mean (centred one slot earlier) moved on by one slot of its 30-min trend
    qb = np.maximum(np.nan_to_num(np.asarray(q15, np.float64), nan=0.0) + slope, 0.0)
    k0 = qb / np.maximum(v0, 5.0)
    tod = Ts % C.SLOTS; wd = np.asarray(P.wd_day)[Ts // C.SLOTS]
    shp = net.prof.shape(folds, wd, tod)                                          # [n, K + 1, L]

    def q_prof(ts):
        i0 = min(int(ts), K - 1); fr = ts - i0
        return qb * ((1.0 - fr) * shp[:, i0] + fr * shp[:, i0 + 1])
    scen = {"": lambda ts: qb,
            "t": lambda ts: np.clip(qb + slope * ts, 0.7 * qb, 1.4 * qb),
            "p": q_prof}
    rout = net.r_out(folds)
    out = {}
    for tag in CTM_SCEN:
        spd, mrg, kx = _ctm_run(net, k0, scen[tag], rout)
        r = spd / net.vcut[None, None, :]                                         # [K, n, L]
        q = r <= 1.0
        first = np.where(q.any(0), q.argmax(0) + 1, 9)
        out[f"ph_r15{tag}"] = r[2].astype(np.float32); out[f"ph_r30{tag}"] = r[K - 1].astype(np.float32)
        out[f"ph_q30{tag}"] = q[K - 1].astype(np.float32); out[f"ph_tq{tag}"] = first.astype(np.float32)
        out[f"ph_mrg{tag}"] = mrg.astype(np.float32); out[f"ph_kx{tag}"] = kx.astype(np.float32)
        if tag == "p":
            out["ph_nq30p"] = np.repeat(q[K - 1].sum(1, keepdims=True).astype(np.float32), L, 1)
            out["ph_dq30p"] = np.minimum(np.minimum(_nearest_dist(q[K - 1], net.mp, "dn"),
                                                    _nearest_dist(q[K - 1], net.mp, "up")), 30.0).astype(np.float32)
            out["ph_mrg_rankp"] = _rank_rows(mrg, ascending=False)
    load = qb[:, None, :] * shp / net.Q[None, None, :]
    out["ph_load30"] = load[:, K].astype(np.float32); out["ph_loadmax"] = load.max(1).astype(np.float32)
    return out


def relative_features(F):
    """v6 rel_* features: where each link stands among the corridor's links in the same window."""
    last_s = F["r_last"]; ext = F["r_ext30"]; drop = F["d_r30"]; fl = F["f_last"]; dens = F["dens"]
    out = {}
    with np.errstate(all="ignore"):
        out["rel_r_rank"] = _rank_rows(last_s, True); out["rel_r_gap"] = (last_s - np.nanmin(last_s, 1, keepdims=True)).astype(np.float32)
        out["rel_ext_rank"] = _rank_rows(ext, True); out["rel_ext_gap"] = (ext - np.nanmin(ext, 1, keepdims=True)).astype(np.float32)
        out["rel_drop_rank"] = _rank_rows(drop, True)
        out["rel_f_rank"] = _rank_rows(fl, False); out["rel_dens_rank"] = _rank_rows(dens, False)
        loc3 = np.nanmin(np.stack([_shift(last_s, o) for o in range(-3, 4)], 0), 0)
        loc5 = np.nanmin(np.stack([_shift(last_s, o) for o in range(-5, 6)], 0), 0)
        out["rel_locmin3"] = (last_s <= loc3 + 1e-6).astype(np.float32)
        out["rel_loc_gap5"] = (last_s - loc5).astype(np.float32)
    return out


# ----------------------------------------------------------------------------- features
def window_features(P: P2, prof: Prof, Ts, A, folds, onset_prior=None, use_T=True, extra=False):
    """Shared per (window, link) features [n*L rows] and per-step arrays [n, K, L]. Only data <= T."""
    Ts = np.asarray(Ts, np.int64); n = len(Ts); L = P.L
    vcut = P.vcut[None, :]; cap = P.cap[None, :]; lanes = P.lanes[None, :]; mp = P.mp
    r = A["hv"] / P.vcut[None, None, :]
    # origin slot T from the released masked view (Task 1 targets there are hidden)
    rT = P.mv[Ts] / P.vcut[None, :]; qT = P.mq[Ts]
    if not use_T:  # strict variant: nothing from the origin slot itself
        rT = np.full_like(rT, np.nan); qT = np.full_like(qT, np.nan)
    rf, age = _ffill_time(r)
    qf, _ = _ffill_time(A["hq"])
    last = np.where(np.isfinite(rT), rT, rf[:, -1])
    lastq = np.where(np.isfinite(qT), qT, qf[:, -1])
    last_s = _fill_space(last); lastq_s = _fill_space(lastq)
    rfs = _fill_space(rf)
    with np.errstate(all="ignore"):
        r_m3 = np.nanmean(r[:, -3:], 1); r_m12 = np.nanmean(r, 1); r_min = np.nanmin(r, 1); r_max = np.nanmax(r, 1)
        q_m3 = np.nanmean(A["hq"][:, -3:], 1); o_m3 = np.nanmean(A["ho"][:, -3:], 1)
    F = {}
    F["r_last"] = last_s; F["r_last_raw"] = last; F["r_T_seen"] = np.isfinite(rT).astype(np.float32)
    F["age"] = np.where(np.isfinite(rT), 0, np.nan_to_num(age[:, -1], nan=HH) + 1).astype(np.float32)
    F["r_m3"] = r_m3; F["r_m12"] = r_m12; F["r_min"] = r_min; F["r_max"] = r_max
    F["d_r15"] = last_s - rfs[:, -3]; F["d_r30"] = last_s - rfs[:, -6]; F["d_r55"] = last_s - rfs[:, 0]
    F["nq12"] = np.nansum(r <= 1, 1).astype(np.float32); F["nq3"] = np.nansum(r[:, -3:] <= 1, 1).astype(np.float32)
    F["nobs"] = np.isfinite(r).sum(1).astype(np.float32)
    F["f_last"] = lastq_s / cap; F["f_m3"] = q_m3 / cap; F["occ_m3"] = o_m3
    F["d_f15"] = (lastq_s - _fill_space(qf[:, -3])) / cap
    F["dens"] = (lastq_s / np.maximum(last_s * vcut, 5.0)) / np.maximum(P.kc[None, :], 1e-3)
    # onset physics: demand against capacity and its trend over the last 30 min, extrapolated to T+30
    xs = np.arange(6, dtype=np.float32) - 2.5
    fq6 = np.concatenate([qf[:, -5:], lastq[:, None, :]], 1) / cap[:, None, :]
    r6 = np.concatenate([rf[:, -5:], last[:, None, :]], 1)
    for nm, arr in (("fq", fq6), ("r", r6)):
        ok = np.isfinite(arr)
        mu = np.where(ok.any(1), np.nanmean(arr, 1), np.nan)
        num = np.nansum(np.where(ok, (arr - mu[:, None, :]) * xs[None, :, None], 0), 1)
        den = np.sum(np.where(ok, xs[None, :, None] ** 2, 0), 1)
        slope = np.where(den > 0, num / np.maximum(den, 1e-6), np.nan)
        F[f"{nm}_slope30"] = slope.astype(np.float32)
        F[f"{nm}_ext30"] = (mu + slope * (2.5 + 6)).astype(np.float32)
    with np.errstate(all="ignore"):
        fq15 = np.stack([np.nanmean(A["hq"][:, i:i + 3], 1) for i in range(0, HH - 2, 3)], 1) / cap[:, None, :]
        F["fq_max15"] = np.nanmax(fq15, 1)
    fql = lastq_s / cap
    arr3 = np.nanmean(np.stack([_shift(fql, -1), _shift(fql, -2), _shift(fql, -3)], 0), 0)
    F["arr_dem"] = arr3; F["arr_dem_ratio"] = arr3 * cap / np.maximum(_shift(cap, 0), 1)
    F["arr_dem_slope"] = np.nanmean(np.stack([_shift(F["fq_slope30"], -1), _shift(F["fq_slope30"], -2), _shift(F["fq_slope30"], -3)], 0), 0)
    F["dens_kc_ext"] = F["dens"] + np.nan_to_num(F["fq_slope30"]) * 6
    for o in (-6, -4, -3, -2, -1, 1, 2, 3, 4, 6):
        F[f"r_o{o}"] = _shift(last_s, o)
    for o in (-2, -1, 1, 2):
        F[f"dr15_o{o}"] = _shift(F["d_r15"], o); F[f"f_o{o}"] = _shift(F["f_last"], o)
    bands = ((0, 1), (1, 3), (3, 6), (6, 12))
    for side, sgn in (("dn", 1), ("up", -1)):
        ms = {b: np.full((n, L), 9.0, np.float32) for b in bands}
        for o in range(1, L):
            idx = np.arange(L) + sgn * o
            okl = (idx >= 0) & (idx < L)
            if not okl.any():
                break
            idc = np.clip(idx, 0, L - 1)
            dist = np.where(okl, np.abs(mp[idc] - mp), np.inf)
            if np.all(dist > bands[-1][1]):
                break
            v = last_s[:, idc]
            for lo, hi in bands:
                sel = (dist > lo) & (dist <= hi)
                if sel.any():
                    ms[(lo, hi)] = np.where(sel[None, :], np.minimum(ms[(lo, hi)], v), ms[(lo, hi)])
        for lo, hi in bands:
            F[f"rmin_{side}_{lo}_{hi}"] = ms[(lo, hi)]
    q_now = last_s <= 1.0
    for tag, q in (("now", q_now), ("15", rfs[:, -3] <= 1), ("30", rfs[:, -6] <= 1), ("60", rfs[:, 0] <= 1)):
        F[f"ddn_{tag}"] = _nearest_dist(q, mp, "dn"); F[f"dup_{tag}"] = _nearest_dist(q, mp, "up")
    F["ddn_rate15"] = F["ddn_15"] - F["ddn_now"]; F["ddn_rate30"] = (F["ddn_30"] - F["ddn_now"]) / 2
    F["dup_rate15"] = F["dup_15"] - F["dup_now"]
    slow8 = last_s <= 0.8 / 0.6
    F["dslow_dn"] = _nearest_dist(slow8, mp, "dn"); F["dslow_up"] = _nearest_dist(slow8, mp, "up")
    tot_now = q_now.sum(1, keepdims=True).astype(np.float32)
    F["c_q_now"] = np.repeat(tot_now / L, L, 1)
    F["c_q_trend"] = np.repeat((tot_now - (rfs[:, -3] <= 1).sum(1, keepdims=True)) / L, L, 1)
    F["c_q_trend60"] = np.repeat((tot_now - (rfs[:, 0] <= 1).sum(1, keepdims=True)) / L, L, 1)
    F["c_rmin"] = np.repeat(np.nanmin(last_s, 1, keepdims=True), L, 1)
    F["h_cov"] = np.repeat(A["he"].mean((1, 2))[:, None], L, 1).astype(np.float32)
    # queue blocks at the origin and a Rankine-Hugoniot tail speed (km per 5 min, negative = growing upstream)
    kq = lastq_s / np.maximum(last_s * vcut, 3.0)
    tail_d = np.full((n, L), BIG, np.float32); head_d = np.full((n, L), BIG, np.float32)
    blen = np.zeros((n, L), np.float32); rh = np.zeros((n, L), np.float32)
    for i in range(n):
        s, e = C.runs_1d(q_now[i])
        for a, b in zip(s, e):
            tail, head = a, b - 1
            up = slice(max(0, tail - 3), tail)
            qq = np.nanmean(lastq_s[i, tail:min(b, tail + 3)]); kk = np.nanmean(kq[i, tail:min(b, tail + 3)])
            qu = np.nanmean(lastq_s[i, up]) if tail > 0 else np.nan; ku = np.nanmean(kq[i, up]) if tail > 0 else np.nan
            sp = (qq - qu) / (kk - ku) if np.isfinite(qu) and abs(kk - ku) > 1e-3 else 0.0
            sp = float(np.clip(sp, -30, 30)) * 5 / 60
            lo = max(0, tail - 20); hi = min(L, head + 8)
            seg = np.arange(lo, hi)
            dt_ = (P.mp[seg] - P.mp[tail]).astype(np.float32); dh_ = (P.mp[seg] - P.mp[head]).astype(np.float32)
            better = np.abs(dt_) < np.abs(tail_d[i, seg])
            tail_d[i, seg] = np.where(better, dt_, tail_d[i, seg]); head_d[i, seg] = np.where(better, dh_, head_d[i, seg])
            blen[i, seg] = np.where(better, P.mp[head] - P.mp[tail] + P.length[tail], blen[i, seg])
            rh[i, seg] = np.where(better, sp, rh[i, seg])
    F["tail_d"] = tail_d; F["head_d"] = head_d; F["blen"] = blen; F["rh_speed"] = rh
    # masked-view context before the history (3 h) and recent days at the same time of day
    cidx = Ts[:, None] + np.arange(-HH - CTX, -HH)[None, :]
    cr = P.mv[np.clip(cidx, 0, P.NT - 1)] / P.vcut[None, None, :]
    with np.errstate(all="ignore"):
        F["ctx_r_mean"] = np.nanmean(cr, 1); F["ctx_q_frac"] = np.nanmean(np.where(np.isfinite(cr), cr <= 1, np.nan), 1)
        F["ctx_r_min"] = np.nanmin(np.where(np.isfinite(cr), cr, 9), 1)
    for tag, off in (("T", 0), ("T30", K)):
        acc = np.zeros((n, L)); cnt = np.zeros((n, L))
        for dd in range(1, 8):
            for ds in (-2, -1, 0, 1, 2):
                tt = Ts - dd * C.SLOTS + off + ds
                ok = tt >= 0
                v = np.where(ok[:, None], P.mv[np.clip(tt, 0, P.NT - 1)], np.nan) / P.vcut[None, :]
                fin = np.isfinite(v); acc += np.where(fin, v <= 1, 0); cnt += fin
        F[f"rec7_{tag}"] = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan).astype(np.float32)
    F.update(recency_features(P, Ts))
    # ramps (released observations up to T)
    for nm, arr in (("on", P.ron), ("off", P.roff)):
        ridx = Ts[:, None] + np.arange(-HH, 1)[None, :]
        rr = arr[np.clip(ridx, 0, P.NT - 1)] / P.lanes[None, None, :]
        with np.errstate(all="ignore"):
            F[f"r{nm}_m"] = np.nanmean(rr, 1); F[f"r{nm}_last"] = np.nanmean(rr[:, -3:], 1)
    # static
    rep = lambda a: np.repeat(np.asarray(a, np.float32)[None, :], n, 0)
    F["vf"] = rep(P.vf); F["lanes"] = rep(P.lanes); F["cap_l"] = rep(P.cap / P.lanes); F["length"] = rep(P.length)
    F["relpos"] = rep(np.arange(L) / max(L - 1, 1)); F["dist_end"] = rep(P.mp[-1] - P.mp)
    capr_dn = P.cap / np.maximum(_shift(P.cap, 1), 1); capr_up = P.cap / np.maximum(_shift(P.cap, -1), 1)
    F["cap_ratio_dn"] = rep(capr_dn); F["cap_ratio_up"] = rep(capr_up)
    mincap = np.array([P.cap[l:min(L, l + 4)].min() for l in range(L)])
    F["mincap_dn"] = rep(mincap / P.cap)
    F["ramps_on"] = rep(np.convolve(P.n_on, np.ones(5), "same")); F["ramps_off"] = rep(np.convolve(P.n_off, np.ones(5), "same"))
    F["panel_id"] = np.full((n, L), C.panels().index(P.panel), np.float32)
    tod = (Ts % C.SLOTS).astype(np.int64); day = Ts // C.SLOTS; wd = P.wd_day[day]
    F["tod"] = np.repeat((tod / 12.0)[:, None], L, 1).astype(np.float32)
    F["dow"] = np.repeat(wd[:, None], L, 1).astype(np.float32)
    pqT, pqT_all = prof.get(folds, wd, tod)
    F["pq_T"] = pqT; F["pq_T_all"] = pqT_all; F["pq_Tm30"] = prof.get(folds, wd, tod - 6)[0]
    F["pq_up2"] = _shift(pqT, -2); F["pq_dn2"] = _shift(pqT, 2)
    F["rmean_T"] = prof.rmean(folds, tod)
    if onset_prior is not None:
        for k_, v_ in onset_prior.items():
            F[k_] = v_
    if extra:   # v6/v7 feature groups, appended after every v5 column so the v5 feature lists (and models) are unchanged
        groups = "rd" if extra is True else str(extra)
        if "r" in groups:
            F.update(relative_features(F))
        if "d" in groups:
            F.update(merge_features(P, Ts, use_T))
            if getattr(P, "dprof", None) is not None:
                F.update(demand_features(P, Ts, use_T))
        if "p" in groups:
            with np.errstate(all="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                q3 = np.stack([qf[:, -2], qf[:, -1], qT], 0) if use_T else qf[:, -3:].transpose(1, 0, 2)
                q15 = _fill_space(np.nanmean(q3, 0))
            F.update(ctm_features(P, Ts, folds, last_s, q15, F["fq_slope30"], use_T))
        if "m" in groups:   # v9: the month's own earlier days
            F.update(memory_features(P, Ts))
    step = {}
    pk = [prof.get(folds, wd, tod + k) for k in range(1, K + 1)]
    step["pq_k"] = np.stack([a for a, _ in pk], 1); step["pq_all_k"] = np.stack([b for _, b in pk], 1)
    step["rmean_k"] = np.stack([prof.rmean(folds, tod + k) for k in range(1, K + 1)], 1)
    df = pd.DataFrame({k: np.asarray(v, np.float32).reshape(-1) for k, v in F.items()})
    return df, step


def recency_features(P: P2, Ts):
    """Where queues formed recently, read from the released masked view before the history (all <= T).

    The test months are new demand draws: their active bottlenecks differ from train's (stable over the nine train
    months, shifted in March/April). Train-derived location priors therefore mislead, while the current month's own
    queue activity points at the bottlenecks that are active now. Day-type aware (weekday / weekend) and month aware
    (calendar month = the split for March / April), so the features mean the same thing in train and in the test
    months: `rc_m_*` only look at earlier days of the same calendar month, `rc_m_days` says how many there are.
    `rc_*` is not in NOLOC: the no-location variant keeps this current-month evidence."""
    P.recency_prep()
    Ts = np.asarray(Ts, np.int64); n = len(Ts); L = P.L
    D = Ts // C.SLOTS; tod = Ts % C.SLOTS
    bT = np.minimum(tod // 6, 47); b30 = np.minimum((tod + K) // 6, 47)
    dtyp, mb = P.daytype, P.mblock
    out = {}
    Dy = np.maximum(D - 1, 0); okY = (D - 1) >= 0
    out["rc_y_T"] = np.where(okY[:, None], P.q_bin[Dy, bT], np.nan)
    out["rc_y_T30"] = np.where(okY[:, None], P.q_bin[Dy, b30], np.nan)
    out["rc_y_any"] = np.where(okY[:, None], P.q_day[Dy], np.nan)
    out["rc_y_same"] = np.repeat((okY & (dtyp[Dy] == dtyp[D])).astype(np.float32)[:, None], L, 1)
    # the last (up to) five earlier days of the same day type, within two weeks
    sums = {b: np.zeros((n, L)) for b in ("T", "T30")}; cnts = {b: np.zeros((n, L)) for b in ("T", "T30")}
    taken = np.zeros(n, np.int64)
    for dd in range(1, 15):
        d = D - dd; dc = np.maximum(d, 0)
        ok = (d >= 0) & (taken < 5) & (dtyp[dc] == dtyp[D])
        if not ok.any():
            continue
        for tag, b in (("T", bT), ("T30", b30)):
            v = P.q_bin[dc, b]
            fin = np.isfinite(v) & ok[:, None]
            sums[tag] += np.where(fin, v, 0); cnts[tag] += fin
        taken += ok
    for tag in ("T", "T30"):
        out[f"rc_dt5_{tag}"] = np.where(cnts[tag] > 0, sums[tag] / np.maximum(cnts[tag], 1), np.nan)
    # month to date: earlier days of the same calendar month (same day type at T+30; any time of day)
    sm = np.zeros((n, L)); cm = np.zeros((n, L)); sa = np.zeros((n, L)); ca = np.zeros((n, L)); nday = np.zeros(n)
    for dd in range(1, 32):
        d = D - dd; dc = np.maximum(d, 0)
        ok = (d >= 0) & (mb[dc] == mb[D])
        if not ok.any():
            continue
        nday += ok
        v = P.q_day[dc]; fin = np.isfinite(v) & ok[:, None]
        sa += np.where(fin, v, 0); ca += fin
        ok2 = ok & (dtyp[dc] == dtyp[D])
        v = P.q_bin[dc, b30]; fin = np.isfinite(v) & ok2[:, None]
        sm += np.where(fin, v, 0); cm += fin
    out["rc_m_T30"] = np.where(cm > 0, sm / np.maximum(cm, 1), np.nan)
    out["rc_m_any"] = np.where(ca > 0, sa / np.maximum(ca, 1), np.nan)
    out["rc_m_days"] = np.repeat(nday[:, None], L, 1)
    # today before the history window
    lo = D * C.SLOTS; hi = np.maximum(Ts - HH, lo)
    s_ = P.cq[hi] - P.cq[lo]; c_ = P.cn[hi] - P.cn[lo]
    out["rc_today"] = np.where(c_ > 0, s_ / np.maximum(c_, 1), np.nan)
    # neighbourhood (+-2 links) of the recent same-type activity at T+30
    base = np.nan_to_num(out["rc_dt5_T30"], nan=0.0)
    out["rc_dt5_T30_nb"] = np.max(np.stack([_shift(base, o) for o in (-2, -1, 0, 1, 2)], 0), 0)
    return {k: np.asarray(v, np.float32) for k, v in out.items()}


# ----------------------------------------------------------------------------- v9: month memory ("m" group, mm_*)
MEM_W = (-6, 43)   # time of day on the earlier days: from 30 min before T's time to 215 min after it (slots)
MEM_TH = 0.2       # a link counts as queued on a day when at least this share of its known slots in MEM_W is queued


def _heads(prof, th=MEM_TH):
    """[n, L] queued share -> 1 where the link ends a queued stretch on its downstream side (the next known link
    downstream is not queued, or there is none), 0 where it is known and does not, NaN where it is unknown. Link
    index increases downstream, so the downstream end of a queue is its head, the bottleneck it grows from."""
    qd = np.where(np.isfinite(prof), (prof >= th).astype(np.float32), np.nan)
    n, L = qd.shape
    nxt = np.zeros((n, L), np.float32)
    cur = np.zeros(n, np.float32)            # queued state of the nearest known link downstream (0 past the end)
    for l in range(L - 1, -1, -1):
        nxt[:, l] = cur
        cur = np.where(np.isfinite(qd[:, l]), qd[:, l], cur)
    return np.where(np.isfinite(qd), qd * (1.0 - nxt), np.nan)


def memory_features(P: P2, Ts):
    """v9 'm' group: where queues sat around this time of day on the earlier days of the same calendar month - the
    month's own demand draw, which sets its bottlenecks - and today before the history window, read from the released
    masked view (every earlier day lies wholly before T; today only up to T - 60 min).

    The official windows of a month are its first queue onsets (and queues) from the 1st, so most of them have earlier
    days of the same month whose same peak points at the active bottleneck. The release blanks exactly the horizon and
    the following hour of those earlier windows; MEM_W reaches past that blank (to 215 min after the time of day), and
    `mem_prep` blanks train months the same way, so the features mean the same in train and in March / April. Only the
    same calendar month counts (train months stand for a month too): the previous month is another demand draw.
      mm_q_{dt,all}  queued share per link in MEM_W, averaged over the earlier days (dt: same day type only)
      mm_h_{dt,all}  share of those days on which the link was the head (downstream end) of a queued stretch
      mm_days_*      how many earlier days of the month carry data (0 on the 1st: then every other mm_ is NaN)
      neighbours, distance to the nearest frequent head, today's queued share and heads before the history."""
    P.mem_prep()
    Ts = np.asarray(Ts, np.int64); n = len(Ts); L = P.L
    D = Ts // C.SLOTS; tod = Ts % C.SLOTS
    a_off = np.maximum(tod + MEM_W[0], 0); b_off = np.minimum(tod + MEM_W[1], C.SLOTS)
    mb, dtyp = P.mblock, P.daytype
    keys = ("dt", "all")
    sq = {k: np.zeros((n, L)) for k in keys}; sh = {k: np.zeros((n, L)) for k in keys}
    cnt = {k: np.zeros((n, L)) for k in keys}; nday = {k: np.zeros(n) for k in keys}
    for dd in range(1, 32):
        d = D - dd; dc = np.maximum(d, 0)
        ok = (d >= 0) & (mb[dc] == mb[D])
        if not ok.any():
            continue
        a = dc * C.SLOTS + a_off; b = dc * C.SLOTS + b_off
        ks = (P.mcn[b] - P.mcn[a]).astype(np.float64)
        with np.errstate(all="ignore"):
            prof = np.where(ks > 0, (P.mcq[b] - P.mcq[a]) / np.maximum(ks, 1), np.nan)
        hd = _heads(prof)
        for key in keys:
            sel = ok if key == "all" else ok & (dtyp[dc] == dtyp[D])
            f2 = np.isfinite(prof) & sel[:, None]
            sq[key] += np.where(f2, prof, 0); sh[key] += np.where(f2, hd, 0); cnt[key] += f2
            nday[key] += sel & np.isfinite(prof).any(1)
    out = {}
    for key in keys:
        c = cnt[key]
        out[f"mm_q_{key}"] = np.where(c > 0, sq[key] / np.maximum(c, 1), np.nan)
        out[f"mm_h_{key}"] = np.where(c > 0, sh[key] / np.maximum(c, 1), np.nan)
        out[f"mm_days_{key}"] = np.repeat(nday[key][:, None], L, 1)
    out["mm_q_dt_dn1"] = _shift_nan(out["mm_q_dt"], 1); out["mm_q_dt_up1"] = _shift_nan(out["mm_q_dt"], -1)
    h = np.nan_to_num(out["mm_h_dt"], nan=0.0)
    out["mm_h_dt_nb2"] = np.where(nday["dt"][:, None] > 0,
                                  np.max(np.stack([_shift(h, o) for o in (-2, -1, 0, 1, 2)], 0), 0), np.nan)
    hb = np.nan_to_num(out["mm_h_all"], nan=0.0) >= 0.5
    out["mm_h_ddn"] = np.where(nday["all"][:, None] > 0, _nearest_dist(hb, P.mp, "dn"), np.nan)
    out["mm_h_dup"] = np.where(nday["all"][:, None] > 0, _nearest_dist(hb, P.mp, "up"), np.nan)
    # today, before the history window (the history itself is in the window's own features)
    lo = D * C.SLOTS; hi = np.maximum(Ts - HH, lo)
    ks = (P.mcn[hi] - P.mcn[lo]).astype(np.float64)
    with np.errstate(all="ignore"):
        tq = np.where(ks > 0, (P.mcq[hi] - P.mcq[lo]) / np.maximum(ks, 1), np.nan)
    out["mm_today"] = tq; out["mm_today_h"] = _heads(tq)
    return {k: np.asarray(v, np.float32) for k, v in out.items()}


def onset_prior(Wtr, ytr, query_folds, query_tods, L, win=18):
    """Share of training onset windows (same panel, |tod| <= win slots, other week-block folds; all of them for
    test windows, fold -1) queued at T+30 per link."""
    Y = ytr[:, K - 1].astype(np.float32)  # [m, L]
    dtod = np.abs(query_tods[:, None] - Wtr.tod.to_numpy()[None, :])
    dtod = np.minimum(dtod, C.SLOTS - dtod)
    other = (query_folds[:, None] != Wtr.fold.to_numpy()[None, :]) | (query_folds[:, None] < 0)
    out = {}
    for nm, m in (("op_near", (dtod <= win) & other), ("op_all", other)):
        w = m.astype(np.float32)
        out[nm] = ((w @ Y) / np.maximum(w.sum(1, keepdims=True), 1)).astype(np.float32)
        if nm == "op_near":
            out["op_n"] = np.repeat(w.sum(1, keepdims=True), L, 1).astype(np.float32)
    return out


def expand_steps(df, step, steps=range(1, K + 1)):
    parts = []
    for k in steps:
        d = df.copy()
        d["k"] = np.float32(k)
        for name, arr in step.items():
            d[name] = arr[:, k - 1].reshape(-1)
        d["tail_proj"] = d["tail_d"] - d["rh_speed"] * k
        d["ddn_proj"] = d["ddn_now"] - d["ddn_rate15"] * k / 3.0
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


# ----------------------------------------------------------------------------- decoding / scoring
def eiou_topm(p, min_m=1):
    o = np.argsort(-p)
    cp = np.cumsum(p[o]); tot = p.sum()
    m = np.arange(1, len(p) + 1)
    r = cp / (m + tot - cp)
    r[: max(0, min_m - 1)] = -1
    j = int(np.argmax(r))
    sel = np.zeros(len(p), bool); sel[o[: j + 1]] = True
    return sel


def iou(pred, true):
    u = np.logical_or(pred, true).sum()
    return 1.0 if u == 0 else float(np.logical_and(pred, true).sum() / u)


def aggregate(df, col="iou"):
    fam = C.families()
    pc = df.groupby(["panel", "condition"])[col].mean().reset_index()
    pm = pc.groupby("panel")[col].mean()
    res = {"overall": float(pm.groupby(pm.index.map(fam)).mean().mean())}
    for c in ("queue_onset", "queue_ongoing"):
        x = pc[pc.condition == c].set_index("panel")[col]
        res[c] = float(x.groupby(x.index.map(fam)).mean().mean()) if len(x) else np.nan
    return res


# ----------------------------------------------------------------------------- test-month pseudo windows
def _ffill_hidden(x, hidden, limit=3):
    """Hidden Task 1 target cells take the same link's last observed value at most `limit` slots back (causal).
    Official histories carry no Task 1 targets, so this makes a pseudo history look like an official one."""
    f = x.copy()
    for k in range(1, limit + 1):
        m = hidden & np.isnan(f)
        m[:k] = False
        src = np.full_like(x, np.nan)
        src[k:] = x[:-k]
        f = np.where(m, src, f)
    return f


def _hash01(T):
    """Deterministic pseudo-random number in [0, 1) per origin (multiplicative hash): window thinning that depends
    on nothing but the origin itself."""
    return ((np.asarray(T, np.uint64) * np.uint64(2654435761)) % np.uint64(2 ** 32)).astype(np.float64) / 2 ** 32


def pseudo_windows(P: P2, split: str, og_every=6, og_per_day=12):
    """Queue windows in a test month's masked view, chosen with the official selector's rules.

    Origins T with time of day in [1 h, 22.5 h): no blank (official horizon / buffer) row in [T-12, T+6]; eligible
    history coverage >= 0.7; condition from the history (hidden targets filled causally): ongoing iff >= 2 queued
    cells with >= 2 on one link; a queued, observed, eligible horizon cell; persistence IoU of slot T against the
    horizon <= 0.9; onset only when the first queued horizon cell is at T+30 (the organizer's ruling on the official
    onset windows). Two sets: `sim` = the chronological greedy (5 per condition, 360 min apart) started on every day
    of the month, like the official draw (`draws` = how many start days pick the window); `all` = every onset
    candidate that starts a run, plus ongoing candidates at slots T % og_every == 0, at most `og_per_day` per day
    (picked by a hash of T, so no random state and nothing from other days is involved).
    Labels are the queue status of the observed eligible horizon cells; `known` marks them (the IoU leaves the
    others out, which is unbiased for randomly hidden cells). Known limitation: a queue that starts in hidden cells
    before T+30 cannot be seen, so some onset windows are a few minutes later than the official selector's."""
    rs = C.split_rows(split)
    lo, hi = rs.start, rs.stop
    v, q, o, el = P.mv[lo:hi], P.mq[lo:hi], P.mo[lo:hi], P.el[lo:hi]
    hidden = (P.tgt[lo:hi] > 0) & np.isnan(v)
    vfl, qfl, ofl = (_ffill_hidden(a, hidden) for a in (v, q, o))
    drk = P.dark[lo:hi] | np.isnan(v).all(1)
    known = el & np.isfinite(v)
    vc = P.vcut[None, :]
    qk = known & (np.nan_to_num(v, nan=np.inf) <= vc)
    qf = el & (np.nan_to_num(vfl, nan=np.inf) <= vc)
    n, L = v.shape
    t = np.arange(n)
    tod = t % C.SLOTS
    t = t[(tod >= HH) & (tod < C.SLOTS - 18) & (t + K < n)]
    cd = np.concatenate([[0], np.cumsum(drk)])
    ok = (cd[t + K + 1] - cd[t - HH]) == 0
    cE = np.vstack([np.zeros((1, L), np.int32), np.cumsum(el, 0, dtype=np.int32)])
    cov = (cE[t] - cE[t - HH]).sum(1) / (HH * L)
    cQ = np.vstack([np.zeros((1, L), np.int32), np.cumsum(qf, 0, dtype=np.int32)])
    per = cQ[t] - cQ[t - HH]
    og = (per.sum(1) >= 2) & (per.max(1) >= 2)
    pT = qf[t]
    inter = np.zeros(len(t)); union = np.zeros(len(t)); early = np.zeros(len(t), bool)
    for k in range(1, K + 1):
        yk = qk[t + k]
        inter += (pT & yk).sum(1); union += ((pT & known[t + k]) | yk).sum(1)
        if k < K:
            early |= yk.any(1)
    last = qk[t + K].any(1)
    piou = inter / np.maximum(union, 1)
    cand = ok & (cov >= 0.7) & (early | last) & (piou <= 0.9) & (og | (~early & last))
    st = pd.DataFrame({"T": t[cand] + lo, "ongoing": og[cand], "cand": True})
    sim = {}
    for s in range(n // C.SLOTS):
        for T, o_ in P2.greedy(st, lo + s * C.SLOTS, hi):
            sim[(T, o_)] = sim.get((T, o_), 0) + 1
    Tc = st["T"].to_numpy(np.int64); oc = st.ongoing.to_numpy(bool)
    on_T = Tc[~oc]
    first = on_T[~np.isin(on_T - 1, on_T)]
    og_T = Tc[oc]
    og_T = og_T[(og_T % og_every) == 0]
    if og_per_day and len(og_T):
        d_ = og_T // C.SLOTS
        order = np.lexsort((_hash01(og_T), d_))
        og_T, d_ = og_T[order], d_[order]
        rank = np.arange(len(og_T)) - np.searchsorted(d_, d_, side="left")
        og_T = np.sort(og_T[rank < og_per_day])
    W = [dict(T=int(T), ongoing=bool(o_), src="sim", draws=c_) for (T, o_), c_ in sim.items()]
    W += [dict(T=int(T), ongoing=False, src="all", draws=0) for T in first if (int(T), False) not in sim]
    W += [dict(T=int(T), ongoing=True, src="all", draws=0) for T in og_T if (int(T), True) not in sim]
    W = pd.DataFrame(W, columns=["T", "ongoing", "src", "draws"]).astype({"T": np.int64, "ongoing": bool, "draws": np.int64})
    W = W.sort_values("T", kind="stable").reset_index(drop=True)
    W["condition"] = np.where(W.ongoing, "queue_ongoing", "queue_onset")
    W["day"] = W["T"] // C.SLOTS
    W["split"] = split
    Ts = W["T"].to_numpy(np.int64) - lo
    hidx = Ts[:, None] + np.arange(-HH, 0)[None, :]
    fidx = Ts[:, None] + np.arange(1, K + 1)[None, :]
    A = dict(hv=vfl[hidx], hq=qfl[hidx], ho=ofl[hidx], he=el[hidx])
    return W, A, qk[fidx], known[fidx]


def pseudo_rows(P: P2, prof: Prof, Wc, A, y, kn, cond, Won, yon, cfg):
    """Feature rows of pseudo windows, built exactly like the official windows' (full-train profile, fold -1).
    Onset: every link at T+30; ongoing: candidate cells x 6 steps. `y` = label, `kn` = label known."""
    Ts = Wc["T"].to_numpy(np.int64)
    folds = np.full(len(Ts), -1)
    parts = []
    for i in range(0, len(Ts), 400):
        sl = slice(i, i + 400)
        nW = len(Ts[sl])
        Ai = {k: v[sl] for k, v in A.items()}
        opr = onset_prior(Won, yon, folds[sl], Ts[sl] % C.SLOTS, P.L) if cond == "queue_onset" else None
        df, step = window_features(P, prof, Ts[sl], Ai, folds[sl], opr, cfg.get("use_T", True), extra=extra_groups(cfg, cond))
        df["wi"] = np.repeat(np.arange(i, i + nW), P.L); df["link"] = np.tile(np.arange(P.L), nW)
        if cond == "queue_onset":
            df["y"] = y[sl][:, K - 1].reshape(-1).astype(np.float32)
            df["kn"] = kn[sl][:, K - 1].reshape(-1)
            parts.append(df)
        else:
            cand = candidate_cells(df)
            st = expand_steps(df, step)
            st["y"] = np.transpose(y[sl], (1, 0, 2)).reshape(-1).astype(np.float32)
            st["kn"] = np.transpose(kn[sl], (1, 0, 2)).reshape(-1)
            parts.append(st[np.tile(cand, K)])
    return pd.concat(parts, ignore_index=True) if parts else None


# ----------------------------------------------------------------------------- dataset building
ONSET_P = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=100, feature_fraction=0.8,
               bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, max_bin=255, verbose=-1)
ONG_P = dict(ONSET_P, num_leaves=127, min_data_in_leaf=50)
NOLOC = ("pq_", "rec7_", "op_", "relpos", "dist_end", "panel_id", "panel_code", "rmean_")
NONF = ("y", "w", "wi", "link", "panel", "fold", "step_ok", "oof", "kn", "pseudo")
PANEL_KEY = 10 ** 8          # window key = panel_code * PANEL_KEY + wi
PSEUDO_WI = 10 ** 6          # pseudo windows of test split j get wi + PSEUDO_WI * (1 + j)


def candidate_cells(df_shared):
    """Ongoing rows worth modelling: queue activity nearby now or in the history, or a congested profile."""
    c = ((df_shared.nq12 > 0) | (df_shared.r_min <= 1.25) | (df_shared.ddn_now <= 12) |
         (df_shared.pq_T >= 0.03) | (df_shared.r_last <= 1.3) | (df_shared.dslow_dn <= 3))
    return c.to_numpy()


def pseudo_splits(cfg):
    """March: evaluation, the gates and training rows for April. April: training rows for the late-window adaptation
    (v5) / within-month online adaptation only, and only windows from days before the official windows they serve
    (never evaluated or used to choose anything)."""
    if not cfg.get("pseudo", True):
        return ()
    april = (cfg.get("online_adapt", False) or cfg.get("pseudo_april", False) or cfg.get("late_adapt", False)
             or any(early_blocks(cfg, "private", c) for c in (cfg.get("late_early") or {})))   # v8: early blocks too
    return ("validation", "private") if april else ("validation",)


def _private_conds(cfg):
    """Conditions whose April pseudo windows are built: the late-window adaptation's (default both; the official selector
    takes a corridor's five onset windows first, on the first peaks of the month, so the late official windows are
    mostly ongoing ones and the onset step usually finds nothing to refit); every condition with the diagnostic switch
    `pseudo_april`."""
    if cfg.get("pseudo_april", False):
        return set(CONDS)
    out = set()
    if cfg.get("late_adapt", False):
        out |= set(cfg.get("late_conds", list(CONDS)))
    out |= {c for c in (cfg.get("late_early") or {}) if early_blocks(cfg, "private", c)}
    if cfg.get("online_adapt", False):
        out.add("queue_onset")
    return out


def build_panel_sets(panel, imputer, rng, cfg, skip=(), skip_april=()):
    """Train windows (labels, fold ids), official windows (every split) and test-month pseudo windows ->
    feature tables. Conditions in `skip` get no training or pseudo tables (v7: ongoing rows reused); those in
    `skip_april` no April pseudo tables (v9: tables built only for the ongoing probes)."""
    P = P2(panel)
    P.build_labels(imputer)
    prof = Prof(P)
    P.dprof = DemandProf(P) if "d" in extra_groups(cfg) else None   # v6 dr_* features (and the ramp diagnostics)
    W, y = P.windows(rng, max_ongoing_cand=cfg.get("max_ongoing_cand", 1500))
    if len(W) == 0:
        C.log(f"WARNING {panel}: no training windows found by the selector replica")
        W = pd.DataFrame(columns=["T", "ongoing", "src", "draws", "condition", "day"]).astype({"T": int, "day": int})
    W["tod"] = W["T"] % C.SLOTS
    W["fold"] = (W.day // 7) % 4
    on = (W.condition == "queue_onset").to_numpy()
    Won, yon = W[on].reset_index(drop=True), y[on]
    out = {}
    for cond in CONDS:
        if cond in skip:
            continue
        m = (W.condition == cond).to_numpy()
        Wc, yc = W[m].reset_index(drop=True), y[m]
        if len(Wc) == 0:
            continue
        A = P.inputs_train(Wc["T"].to_numpy())
        opr = onset_prior(Won, yon, Wc.fold.to_numpy(), Wc.tod.to_numpy(), P.L) if cond == "queue_onset" else None
        parts = []
        for i in range(0, len(Wc), 400):
            sl = slice(i, i + 400)
            Ai = {k: v[sl] for k, v in A.items()}
            op_i = {k: v[sl] for k, v in opr.items()} if opr else None
            df, step = window_features(P, prof, Wc["T"].to_numpy()[sl], Ai, Wc.fold.to_numpy()[sl], op_i, cfg.get("use_T", True),
                                       extra=extra_groups(cfg, cond))
            nW = len(Wc.iloc[sl])
            df["wi"] = np.repeat(np.arange(i, i + nW), P.L); df["link"] = np.tile(np.arange(P.L), nW)
            if cond == "queue_onset":
                df["y"] = yc[sl][:, K - 1].reshape(-1).astype(np.float32)
                parts.append(df)
            else:
                cand = candidate_cells(df)
                st = expand_steps(df, step)
                yy = np.transpose(yc[sl], (1, 0, 2)).reshape(-1)  # expand_steps is step-major [K, n, L]
                st["y"] = yy.astype(np.float32)
                keep = np.tile(cand, K)
                parts.append(st[keep])
        D = pd.concat(parts, ignore_index=True)
        D["fold"] = Wc.fold.to_numpy()[D.wi.to_numpy()]
        out[cond] = (Wc, yc, D)
    # official windows of every split
    offi = {}
    for split in C.SPLITS:
        if not (C.rel() / "task2" / panel / split / "window_index.csv").exists():
            continue
        wi, A = P.inputs_official(split)
        Ts = wi["T"].to_numpy()
        folds = ((Ts // C.SLOTS) // 7) % Prof.NF if split == "train" else np.full(len(wi), -1)
        opr = onset_prior(Won, yon, folds, Ts % C.SLOTS, P.L)
        df, step = window_features(P, prof, Ts, A, folds, opr, cfg.get("use_T", True), extra=extra_groups(cfg))
        df["wi"] = np.repeat(np.arange(len(wi)), P.L); df["link"] = np.tile(np.arange(P.L), len(wi))
        offi[split] = (wi, df, step)
    # test-month pseudo windows (released masked view only)
    pseudo = {}
    for split in pseudo_splits(cfg):
        if len(C.split_days(split)) == 0:
            continue
        Wp, Ap, yp, kp = pseudo_windows(P, split, cfg.get("pseudo_og_every", 6), cfg.get("pseudo_og_per_day", 12))
        for cond in CONDS:
            if cond in skip:
                continue
            if split == "private" and (cond not in _private_conds(cfg) or cond in skip_april):
                continue   # April windows serve only the late-window adaptation (training rows, earlier days)
            m = (Wp.condition == cond).to_numpy()
            if split == "private" and not cfg.get("pseudo_april", False) and not cfg.get("online_adapt", False):
                # only the days before the last late block's first day can ever be used (training rows of models for
                # April windows on later days), so later April windows are not even built
                cuts = [lo for lo, _ in late_blocks(cfg, "private")]
                m = m & (Wp.day.to_numpy() < (max(cuts) if cuts else -1))
            if not m.any():
                continue
            Wc = Wp[m].reset_index(drop=True)
            X = pseudo_rows(P, prof, Wc, {k: v[m] for k, v in Ap.items()}, yp[m], kp[m], cond, Won, yon, cfg)
            if X is not None:
                pseudo.setdefault(split, {})[cond] = (Wc, yp[m], kp[m], X)
    return P, out, offi, pseudo


# ----------------------------------------------------------------------------- models
# v2  = the features of the 0.86840 submission (incl. rec7_, the last seven days at the same time of day);
# v2n = v2 without any recent-days evidence (no rec7_): train priors + the window's own history only;
# v3  = v2 + current-month recency (rc_*), dropped in v4 (it did not carry over to the scored windows).
FSETS = ("v2", "v2n", "v3")
RECENT = ("rc_", "rec7_")
EXTRA = {"r": "rel_", "d": "dr_", "p": "ph_", "m": "mm_"}   # v6/v7/v9 feature groups, switched on by a suffix: "v2n+rd" = v2n + rel_ + dr_
# v6 "noloc2": also drop the static link attributes that, combined, act as a link identifier (a tree can memorise
# where queues usually start through them). The lane count, the ramp counts and the capacity ratios to the neighbours
# (lane drops: where bottlenecks physically are, the same in every month) stay.
STATIC_ID = ("length", "cap_l", "vf")
# v7 "_nr" variants: without the ramp-flow columns. Inside one demand draw every link's ramp flow by time of day is
# nearly fixed, so these columns can act as link identifiers; each test month re-deals the ramp demand link by link.
# The ramp counts (road layout, the same every month) stay.
RAMP_F = ("ron_", "roff_")
VARIANTS = ("all", "noloc", "noloc2", "all_nr", "noloc_nr", "noloc2_nr")


def _variant_name(v):
    """"all" / "noloc" / "noloc2", optionally + "_nr" (the v5 booleans False / True are accepted); anything else is an
    error."""
    name = {False: "all", True: "noloc"}.get(v, v) if isinstance(v, (bool, np.bool_)) else v
    if name not in VARIANTS:
        raise ValueError(f"unknown variant {v!r}: one of {VARIANTS}")
    return name


def _check_fset(fset):
    """(base, groups) of a feature-set name: a base in FSETS, optionally '+' and distinct groups from EXTRA."""
    if not isinstance(fset, str):
        raise ValueError(f"feature set must be a string, got {fset!r}")
    base, sep, mods = fset.partition("+")
    if base not in FSETS or (sep and (not mods or any(m not in EXTRA for m in mods) or len(set(mods)) != len(mods))):
        raise ValueError(f"unknown feature set {fset!r}: a base in {FSETS}, optionally '+' and groups from {sorted(EXTRA)}")
    return base, mods


def _feats(D, noloc=False, fset="v3"):
    """Feature columns of a table for feature set `fset` ("v2", "v2n", "v3", optionally + "+r", "+d", "+rd") and variant
    `noloc` (False/"all", True/"noloc", "noloc2")."""
    base, mods = _check_fset(fset)
    cols = [c for c in D.columns if c not in NONF]
    if base == "v2":
        cols = [c for c in cols if not c.startswith("rc_")]
    elif base == "v2n":
        cols = [c for c in cols if not c.startswith(RECENT)]
    for m, pre in EXTRA.items():
        if m not in mods:
            cols = [c for c in cols if not c.startswith(pre)]
    v = _variant_name(noloc)
    nr = v.endswith("_nr")
    v = v[:-3] if nr else v
    if v in ("noloc", "noloc2"):
        cols = [c for c in cols if not c.startswith(NOLOC)]
    if v == "noloc2":
        cols = [c for c in cols if c not in STATIC_ID]
    if nr:
        cols = [c for c in cols if not c.startswith(RAMP_F)]
    return cols


def _fset_with(base, chosen):
    """The chosen experiment feature set's extra groups carried over to another base ("v2n+rd" on "v2" -> "v2+rd")."""
    b, _ = _check_fset(base)
    _, mods = _check_fset(chosen)
    return b + ("+" + mods if mods else "")


def _matrix(frames, feats, masks=None, chunk=200_000):
    """Feature columns of several frames (optionally row-masked) as one float32 C-order matrix, filled chunk by chunk:
    no full-size intermediate copy, and LightGBM takes the matrix as it is."""
    masks = masks if masks is not None else [None] * len(frames)
    n = sum(len(f) if m is None else int(np.count_nonzero(m)) for f, m in zip(frames, masks))
    X = np.empty((n, len(feats)), np.float32)
    o = 0
    for f, m in zip(frames, masks):
        for a in range(0, len(f), chunk):
            blk = f.iloc[a:a + chunk][feats].to_numpy(np.float32)
            if m is not None:
                blk = blk[m[a:a + chunk]]
            X[o:o + len(blk)] = blk
            o += len(blk)
    return X


def fit_lgb(frames, prm, rounds, feats, weight=None, seed=0, masks=None):
    import lightgbm as lgb
    frames = list(frames) if isinstance(frames, (list, tuple)) else [frames]
    p = dict(prm, num_threads=C.THREADS, seed=seed, bagging_seed=seed + 1, feature_fraction_seed=seed + 2)
    X = _matrix(frames, feats, masks)
    y = np.concatenate([f.y.to_numpy(np.float32) if masks is None else f.y.to_numpy(np.float32)[masks[j]]
                        for j, f in enumerate(frames)])
    ds = lgb.Dataset(X, y, weight=weight, feature_name=list(feats), params=p, free_raw_data=True)
    ds.construct()
    del X
    C.free()
    return lgb.train(p, ds, rounds)


def _wkey(f):
    return f.panel_code.to_numpy().astype(np.int64) * PANEL_KEY + f.wi.to_numpy().astype(np.int64)


def window_weights(frames):
    """Every window gets the same total weight (the metric averages windows); mean 1 over all rows."""
    frames = list(frames) if isinstance(frames, (list, tuple)) else [frames]
    _, inv, cnt = np.unique(np.concatenate([_wkey(f) for f in frames]), return_inverse=True, return_counts=True)
    w = 1.0 / cnt[inv]
    return w * len(w) / w.sum()


# model checkpoints: a disconnect in the middle of this stage does not lose the models already trained. The key is
# the code of the package, the release fingerprint and exactly the settings that shape that model's training rows
# (the model's own seed, rounds, variant, feature set, condition and role are added per file).
_CKPT = {"dir": None, "base": "", "cfg": {}}
CKPT_DEFAULTS = dict(seed=0, imputer=True, max_ongoing_cand=1500, use_T=True, pseudo_og_every=6, pseudo_og_per_day=12,
                     adapt_alpha=2.0, adapt_frac=0.5, online_alpha=2.0, late_alpha=2.0)
_STD_KEYS = ("seed", "imputer", "max_ongoing_cand", "use_T")
_PS_KEYS = _STD_KEYS + ("pseudo_og_every", "pseudo_og_per_day")
CKPT_KEYS = {"std": _STD_KEYS, "exp": _STD_KEYS,
             "adapt": _PS_KEYS + ("adapt_alpha",), "gate": _PS_KEYS + ("adapt_alpha", "adapt_frac"),
             "oneval": _PS_KEYS + ("online_alpha",), "online": _PS_KEYS + ("online_alpha",),
             "lgate": _PS_KEYS + ("late_alpha",), "late": _PS_KEYS + ("late_alpha",),
             "egate": _PS_KEYS + ("late_alpha",), "early": _PS_KEYS + ("late_alpha",)}   # the role names the block


CODE_FILES = ("common.py", "data.py", "task2.py")   # everything Task 2's models and rows depend on


def code_hash() -> str:
    """md5 of the sources Task 2 depends on: a saved Task 2 result or model is reused only if that code is the same
    (an edit to Task 1 or to the submission helpers does not invalidate the Task 2 checkpoints)."""
    h = hashlib.md5()
    for f in CODE_FILES:
        h.update((Path(__file__).parent / f).read_bytes())
    return h.hexdigest()[:16]


def set_checkpoints(cfg):
    from .data import fingerprint
    d = cfg.get("model_dir")
    _CKPT["dir"] = Path(d) if d else None
    if _CKPT["dir"] is not None:
        _CKPT["dir"].mkdir(parents=True, exist_ok=True)
    _CKPT["base"] = f"{VERSION}|{fingerprint()}|{code_hash()}"
    _CKPT["cfg"] = {k: cfg.get(k, v) for k, v in CKPT_DEFAULTS.items()}
    _CKPT["cfg"]["online_alpha"] = float(cfg.get("online_alpha", cfg.get("adapt_alpha", 2.0)))   # the alpha actually used
    _CKPT["cfg"]["late_alpha"] = float(cfg.get("late_alpha", cfg.get("adapt_alpha", 2.0)))


def _ckpt_path(cond, role, noloc, seed, rounds, fset, fs=None):
    """Checkpoint file of one model. The key holds the exact feature list (not the feature-set name), so two feature
    sets that give the same columns (the no-location model of the main and the hedge file) share one model."""
    noloc = _variant_name(noloc)
    if _CKPT["dir"] is None:
        return None
    base = role.split(":")[0]
    key = json.dumps({k: _CKPT["cfg"][k] for k in CKPT_KEYS[base]}, sort_keys=True, default=str)
    ftag = hashlib.md5("|".join(fs).encode()).hexdigest()[:10] if fs is not None else fset
    h = hashlib.md5(f"{_CKPT['base']}|{key}|{cond}|{role}|{noloc}|{seed}|{rounds}|{ftag}".encode()).hexdigest()[:12]
    return _CKPT["dir"] / f"t2_{cond}_{base}_{fset if noloc == 'all' else noloc}_s{seed}_{h}.txt"


def _save_ckpt(m, path):
    """Model text + its md5 (written last): a file that does not match its checksum is never parsed (LightGBM aborts
    the whole process on a truncated model file instead of raising)."""
    if path is None:
        return
    tmp = path.with_suffix(".tmp")
    m.save_model(str(tmp))
    digest = hashlib.md5(tmp.read_bytes()).hexdigest()
    tmp.replace(path)
    path.with_suffix(".md5").write_text(digest)


def _load_ckpt(path, fs, rounds):
    import lightgbm as lgb
    if path is None or not path.exists() or not path.with_suffix(".md5").exists():
        return None
    try:
        raw = path.read_bytes()
        if hashlib.md5(raw).hexdigest() != path.with_suffix(".md5").read_text().strip():
            C.log(f"T2 checkpoint {path.name} does not match its checksum; refitting")
            return None
        m = lgb.Booster(model_str=raw.decode())
    except Exception as e:
        C.log(f"T2 checkpoint {path.name} unreadable ({type(e).__name__}); refitting")
        return None
    return m if (m.feature_name() == fs and m.current_iteration() == rounds) else None


def fit_set(Dl, cond, prm, rounds, variants, seeds, extra=None, alpha=1.0, role="std", fset="v3"):
    """One model per (variant, seed) on the train windows (a list of per-panel frames). With `extra` (pseudo rows)
    the train rows keep exactly their standard weights and every extra window carries `alpha` times the mean total
    weight of a train window of its panel, so the adaptation changes nothing but the added windows.
    Models are checkpointed under `model_dir` (when set) and reloaded on a rerun with the same settings and data."""
    w = window_weights(Dl) if cond == "queue_ongoing" else None
    frames = list(Dl)
    if extra is not None and len(extra):
        w_tr = w if w is not None else np.ones(sum(len(f) for f in Dl))
        u, inv = np.unique(np.concatenate([_wkey(f) for f in Dl]), return_inverse=True)
        tot = np.bincount(inv, weights=w_tr)                       # total weight of each train window
        per_panel = pd.Series(tot).groupby(u // PANEL_KEY).mean()
        _, inv_e, cnt_e = np.unique(_wkey(extra), return_inverse=True, return_counts=True)
        base = per_panel.reindex(extra.panel_code.to_numpy().astype(np.int64)).fillna(float(tot.mean())).to_numpy()
        w = np.concatenate([w_tr, alpha * base / cnt_e[inv_e]])
        frames.append(extra)
    out = []
    for _, nl in variants:
        fs = _feats(Dl[0], nl, fset)
        for sd in seeds:
            path = _ckpt_path(cond, role, nl, sd, rounds, fset, fs)
            m = _load_ckpt(path, fs, rounds)
            if m is None:
                m = fit_lgb(frames, prm, rounds, fs, w, seed=sd)
                _save_ckpt(m, path)
            out.append((m, fs))
    return out


def _models_for(models, variants, seeds, names):
    """The (model, features) pairs of the variants `names` out of a fit_set result (variant-major, then seeds);
    None if one of them was not fitted."""
    vn = [v for v, _ in variants]
    if not set(names) <= set(vn):
        return None
    n = len(seeds)
    return [m for i, v in enumerate(vn) if v in names for m in models[i * n:(i + 1) * n]]


def _predict(models, X):
    return np.mean([m.predict(X[fs], num_threads=C.THREADS) for m, fs in models], 0)


def decode_window(cond, p, L):
    """p: onset [L] at T+30; ongoing [K, L]. Returns [K, L] bool."""
    out = np.zeros((K, L), bool)
    if cond == "queue_onset":
        out[K - 1] = eiou_topm(p, 1)
    else:
        out[:] = eiou_topm(p.reshape(-1), 1).reshape(K, L)
    return out


def _field(cond, X, pr, n, L):
    """Rows -> probability field: onset [n, L] at T+30, ongoing [n, K, L] (cells without a row stay 0)."""
    wi = X.wi.to_numpy().astype(np.int64); li = X.link.to_numpy().astype(np.int64)
    if cond == "queue_onset":
        F = np.zeros((n, L)); F[wi, li] = pr
    else:
        F = np.zeros((n, K, L)); F[wi, X.k.to_numpy().astype(np.int64) - 1, li] = pr
    return F


def predict_official(models, offis, pans, Ps, cond, probs, splits, day=None, day_min=None, day_max=None):
    """Probabilities of the official windows of `cond` (optionally only those whose origin is on calendar day `day`, or
    on a day in [day_min, day_max)). Returns the number of windows stored."""
    n_done = 0
    for i, p in enumerate(pans):
        for split, (wi, df, step) in offis[p].items():
            if split not in splits:
                continue
            L = Ps[p]["L"]
            sel = (wi.condition == cond).to_numpy()
            if day is not None:
                sel = sel & ((wi["T"].to_numpy() // C.SLOTS) == day)
            if day_min is not None:
                sel = sel & ((wi["T"].to_numpy() // C.SLOTS) >= day_min)
            if day_max is not None:
                sel = sel & ((wi["T"].to_numpy() // C.SLOTS) < day_max)
            if not sel.any():
                continue
            if cond == "queue_onset":
                X = df.copy(); X["panel_code"] = np.float32(i)
                pr = _predict(models, X).reshape(len(wi), L)
            else:
                X = expand_steps(df, step); X["panel_code"] = np.float32(i)
                pr = np.transpose(_predict(models, X).reshape(K, len(wi), L), (1, 0, 2))
                pr = pr * candidate_cells(df).reshape(len(wi), L)[:, None, :]
            store = probs.setdefault(p, {}).setdefault(split, {})
            for w in np.flatnonzero(sel):
                store[wi.window_id.iloc[w]] = (cond, pr[w])
                n_done += 1
    return n_done


# ----------------------------------------------------------------------------- evaluation
def _agg1(R, col="iou", wcol=None):
    """Official aggregation of one condition: window -> panel (optionally weighted) -> family -> mean."""
    if R is None or len(R) == 0:
        return float("nan")
    fam = C.families()
    if wcol is None:
        pm = R.groupby("panel")[col].mean()
    else:
        g = R.assign(_x=R[col] * R[wcol]).groupby("panel")
        pm = g._x.sum() / g[wcol].sum().clip(lower=1e-12)
    return float(pm.groupby(pm.index.map(fam)).mean().mean())


def _boot_se(R, col="d", wcol=None, B=400, seed=0):
    """Standard error of _agg1(R, col, wcol) by a block bootstrap: days are resampled within each panel (windows
    of one day share queue episodes, so they are not independent)."""
    if R is None or len(R) < 2:
        return float("nan")
    rng = np.random.default_rng(seed)
    fam = C.families()
    per = []
    for p, g in R.groupby("panel"):
        w = g[wcol].to_numpy(float) if wcol else np.ones(len(g))
        _, inv = np.unique(g.day.to_numpy(), return_inverse=True)
        per.append((p, np.bincount(inv, weights=w * g[col].to_numpy(float)), np.bincount(inv, weights=w)))
    vals = []
    for _ in range(B):
        pm = {}
        for p, sx, sw in per:
            s = rng.integers(0, len(sx), len(sx))
            pm[p] = sx[s].sum() / max(sw[s].sum(), 1e-12)
        pm = pd.Series(pm)
        vals.append(pm.groupby(pm.index.map(fam)).mean().mean())
    return float(np.std(vals))


def cv_score(Dl, sets, pans, Ps, cond, prm, rounds, variants, fset="v3"):
    """Week-blocked CV on the selector-drawn train windows (optimistic against the test months)."""
    wts = window_weights(Dl) if cond == "queue_ongoing" else None
    oof = [np.zeros(len(f)) for f in Dl]
    for k in range(4):
        tr = [(f.fold != k).to_numpy() for f in Dl]
        if not any(m.any() for m in tr) or all(m.all() for m in tr):
            continue
        wk = None if wts is None else wts[np.concatenate(tr)]
        pv = [np.zeros(int((~m).sum())) for m in tr]
        for _, nl in variants:
            fs = _feats(Dl[0], nl, fset)
            mdl = fit_lgb(Dl, prm, rounds, fs, wk, masks=tr)
            for j, f in enumerate(Dl):
                if (~tr[j]).any():
                    pv[j] += mdl.predict(_matrix([f], fs, [~tr[j]]), num_threads=C.THREADS) / len(variants)
        for j in range(len(Dl)):
            oof[j][~tr[j]] = pv[j]
    rows = []
    for j, f in enumerate(Dl):
        p = pans[int(f.panel_code.iat[0])]
        Wc, yc = sets[p][cond][0], sets[p][cond][1]
        F = _field(cond, f, oof[j], len(Wc), Ps[p]["L"])
        for w in np.flatnonzero((Wc.src == "sim").to_numpy()):
            rows.append(dict(panel=p, condition=cond, iou=iou(decode_window(cond, F[w], Ps[p]["L"]), yc[w])))
    R = pd.DataFrame(rows)
    return dict(S_sim=_agg1(R), windows=int(len(R)))


def cv_windows(Dl, sets, pans, Ps, cond, prm, rounds, configs, fset, seed=0):
    """Week-blocked train CV, per window: every variant is fitted once per fold (shared by all configurations that
    use it), out-of-fold probabilities are averaged per configuration and decoded like the official windows.
    Returns {tuple(variants): frame(panel, w, day, src, draws, iou)} over the selector-drawn (sim) train windows.
    The location priors and the onset prior are already out of fold (Prof / onset_prior use the same week folds)."""
    names = sorted({v for vn in configs for v in vn})
    wts = window_weights(Dl) if cond == "queue_ongoing" else None
    oof = {v: [np.full(len(f), np.nan) for f in Dl] for v in names}
    for k in range(4):
        tr = [(f.fold != k).to_numpy() for f in Dl]
        if not any(m.any() for m in tr) or all(m.all() for m in tr):
            continue
        wk = None if wts is None else wts[np.concatenate(tr)]
        for v in names:
            fs = _feats(Dl[0], v, fset)
            mdl = fit_lgb(Dl, prm, rounds, fs, wk, seed=seed, masks=tr)
            for j, f in enumerate(Dl):
                if (~tr[j]).any():
                    oof[v][j][~tr[j]] = mdl.predict(_matrix([f], fs, [~tr[j]]), num_threads=C.THREADS)
            del mdl
            C.free()
    out = {}
    for vn in configs:
        rows = []
        for j, f in enumerate(Dl):
            p = pans[int(f.panel_code.iat[0])]
            Wc, yc = sets[p][cond][0], sets[p][cond][1]
            pr = np.nan_to_num(np.mean([oof[v][j] for v in vn], 0))
            F = _field(cond, f, pr, len(Wc), Ps[p]["L"])
            mp = Ps[p].get("mp")
            for w in np.flatnonzero((Wc.src == "sim").to_numpy()):
                pred = decode_window(cond, F[w], Ps[p]["L"])
                ms = (_onset_miss(pred[K - 1], yc[w][K - 1], F[w], mp) if cond == "queue_onset" and mp is not None
                      else (np.nan, np.nan, np.nan))
                rows.append(dict(panel=p, w=int(w), day=int(Wc.day.iat[w]), src="sim", draws=int(Wc.draws.iat[w]),
                                 iou=iou(pred, yc[w]), n_pred=int(pred.sum()), n_true=int(yc[w].sum()),
                                 inter=int((pred & yc[w]).sum()), tod=float((int(Wc["T"].iat[w]) % C.SLOTS) / 12.0),
                                 miss_links=ms[0], miss_km=ms[1], true_rank=ms[2]))
        out[tuple(vn)] = pd.DataFrame(rows, columns=["panel", "w", "day", "src", "draws", "iou", "n_pred", "n_true",
                                                     "inter", "tod", "miss_links", "miss_km", "true_rank"])
    return out


def _weekday(days) -> np.ndarray:
    """Monday-Friday flag of calendar day indices (April's scored windows open on Tuesday 1 April: weekday peaks)."""
    wd = C.calendar().weekday.to_numpy()
    return wd[np.asarray(days, np.int64)] < 5


def pseudo_report(R):
    """pseudo_summary plus the same numbers on weekday and weekend windows."""
    out = pseudo_summary(R)
    if R is None or len(R) == 0:
        return out
    wd = _weekday(R.day.to_numpy())
    for tag, m in (("weekday", wd), ("weekend", ~wd)):
        s = R[m]
        out[f"n_{tag}"] = int(m.sum())
        out[f"S_sim_{tag}"] = _agg1(s[s.src == "sim"], "iou", "draws")
        out[f"S_all_{tag}"] = _agg1(s)
    return out


def pseudo_scores(models, pseudo, pans, Ps, split, cond, day_lo=None, day_hi=None):
    """IoU on the known cells of every pseudo window of a month (days in [day_lo, day_hi)), official decoding."""
    rows = []
    for p in pans:
        ent = pseudo.get(p, {}).get(split, {}).get(cond)
        if ent is None:
            continue
        Wc, y, kn, X = ent
        L = Ps[p]["L"]
        d = Wc.day.to_numpy()
        sel = np.ones(len(Wc), bool)
        if day_lo is not None:
            sel &= d >= day_lo
        if day_hi is not None:
            sel &= d < day_hi
        ws = np.flatnonzero(sel)
        if len(ws) == 0:
            continue
        Xs = X[np.isin(X.wi.to_numpy(), ws)]
        F = _field(cond, Xs, _predict(models, Xs), len(Wc), L)
        mp = Ps[p].get("mp")
        for w in ws:
            pred = decode_window(cond, F[w], L)
            u = int(((pred | y[w]) & kn[w]).sum()); it = int(((pred & y[w]) & kn[w]).sum())
            on = cond == "queue_onset"
            ms = (_onset_miss(pred[K - 1], y[w][K - 1] & kn[w][K - 1], F[w], mp) if on and mp is not None
                  else (np.nan, np.nan, np.nan))
            unk = bool(on and pred[K - 1].any() and not (pred[K - 1] & kn[w][K - 1]).any())   # predicted cells all hidden
            rows.append(dict(panel=p, condition=cond, split=split, w=int(w), src=Wc.src.iat[w], day=int(Wc.day.iat[w]),
                             draws=int(Wc.draws.iat[w]), iou=1.0 if u == 0 else it / u,
                             n_pred=int((pred & kn[w]).sum()), n_true=int((y[w] & kn[w]).sum()), inter=it,
                             tod=float((int(Wc["T"].iat[w]) % C.SLOTS) / 12.0),
                             miss_links=ms[0], miss_km=ms[1], true_rank=ms[2], pred_hidden=unk))
    return pd.DataFrame(rows, columns=["panel", "condition", "split", "w", "src", "day", "draws", "iou",
                                       "n_pred", "n_true", "inter", "tod", "miss_links", "miss_km", "true_rank",
                                       "pred_hidden"])


def ctm_diag(sets, pseudo, pans, Ps):
    """v7 diagnostics: the cell-transmission forecast on its own, on the onset windows of train (full labels) and of
    March (pseudo windows, known cells), for the profile ("p") and the trend ("t") demand. "queue": the links it
    forecasts queued at T+30; "top link": the link with the largest merge ratio (where a queue would start), a hit
    when it is within one link of a true queued cell. Nothing reads this back."""
    out = {}
    r4 = lambda x: None if x is None or not np.isfinite(x) else round(float(x), 4)
    for tag in ("train", "march"):
        rows = []
        for p in pans:
            if tag == "train":
                ent = sets.get(p, {}).get("queue_onset")
                if ent is None or ent[2] is None or "ph_q30p" not in ent[2]:
                    continue
                Wc, yc, D = ent
                t30 = yc[:, K - 1]; kn = np.ones_like(t30)
            else:
                ent = pseudo.get(p, {}).get("validation", {}).get("queue_onset")
                if ent is None or "ph_q30p" not in ent[3]:
                    continue
                Wc, y, kn_, D = ent
                t30 = y[:, K - 1]; kn = kn_[:, K - 1]
            L, n = Ps[p]["L"], len(Wc)
            wi = D.wi.to_numpy().astype(np.int64); li = D.link.to_numpy().astype(np.int64)
            fq, top = {}, {}
            for sc in ("p", "t"):
                q = np.zeros((n, L), bool); q[wi, li] = D[f"ph_q30{sc}"].to_numpy() > 0.5
                m = np.full((n, L), -np.inf); m[wi, li] = D[f"ph_mrg{sc}"].to_numpy()
                fq[sc], top[sc] = q, m.argmax(1)
            for w in range(n):
                tr = t30[w] & kn[w]
                if not tr.any():
                    continue
                r = dict(tod=float((int(Wc["T"].iat[w]) % C.SLOTS) / 12.0))
                for sc in ("p", "t"):
                    pr = fq[sc][w] & kn[w]
                    j = top[sc][w]
                    r[f"any_{sc}"] = bool(fq[sc][w].any()); r[f"hit_{sc}"] = bool((pr & tr).any())
                    r[f"iou_{sc}"] = float((pr & tr).sum() / max((pr | tr).sum(), 1))
                    r[f"top_{sc}"] = bool(tr[max(0, j - 1):j + 2].any())
                rows.append(r)
        if not rows:
            continue
        R = pd.DataFrame(rows)
        mid = (R.tod >= 10) & (R.tod < 15)
        res = dict(n=int(len(R)), n_10_15h=int(mid.sum()))
        for sc, nm in (("p", "profile"), ("t", "trend")):
            res.update({f"{nm}_forecasts_a_queue": r4(R[f"any_{sc}"].mean()), f"{nm}_site_hit": r4(R[f"hit_{sc}"].mean()),
                        f"{nm}_iou": r4(R[f"iou_{sc}"].mean()), f"{nm}_top_link_hit": r4(R[f"top_{sc}"].mean()),
                        f"{nm}_top_link_hit_10_15h": r4(R[f"top_{sc}"][mid].mean()) if mid.any() else None})
        out[tag] = res
    return out


def _onset_miss(pred, true, p, mp):
    """An onset window whose predicted cells (T+30, bool [L], all of them) miss every true (known) cell: how many links
    and km apart the nearest predicted and true cells are, and the rank of the best true link among the model's
    probabilities (0 = the model's first choice). NaNs for a hit, an empty truth or an empty prediction (diagnostics
    only; a March window whose predicted cells are all hidden scores 0 and is counted under `pred_hidden`)."""
    if mp is None or not true.any() or not pred.any() or (pred & true).any():
        return np.nan, np.nan, np.nan
    ip, it = np.flatnonzero(pred), np.flatnonzero(true)
    dl = float(np.min(np.abs(ip[:, None] - it[None, :])))
    dk = float(np.min(np.abs(mp[ip][:, None] - mp[it][None, :])))
    return dl, dk, float((p > p[it].max()).sum())


def window_diag(R, cond):
    """Where a model's window IoU is lost (plain means over windows; diagnostics only, never read by the pipeline).
    Onset: how often the predicted cells touch the true ones (site hit), sizes, IoU when the site is hit, by time of
    day; ongoing: IoU by true queue size. Both: weekday / weekend."""
    if R is None or len(R) == 0 or "n_pred" not in R:
        return {}
    R = R.dropna(subset=["n_pred"])
    if len(R) == 0:
        return {}
    r2 = lambda x: None if x is None or not np.isfinite(x) else round(float(x), 4)
    out = dict(n=int(len(R)), iou=r2(R.iou.mean()), n_pred=r2(R.n_pred.mean()), n_true=r2(R.n_true.mean()))
    wd = _weekday(R.day.to_numpy())
    out["weekday"] = [int(wd.sum()), r2(R.iou[wd].mean()) if wd.any() else None]
    out["weekend"] = [int((~wd).sum()), r2(R.iou[~wd].mean()) if (~wd).any() else None]
    if cond == "queue_onset":
        hit = R.inter > 0
        out["site_hit"] = r2(hit.mean())
        out["iou_if_hit"] = r2(R.iou[hit].mean()) if hit.any() else None
        out["pred_minus_true_if_hit"] = r2((R.n_pred - R.n_true)[hit].mean()) if hit.any() else None
        bins = ((0, 6, "00-06"), (6, 10, "06-10"), (10, 15, "10-15"), (15, 20, "15-20"), (20, 24, "20-24"))
        out["by_tod"] = {nm: [int(m.sum()), r2(R.iou[m].mean()), r2((R.inter[m] > 0).mean())]
                         for lo, hi, nm in bins for m in [(R.tod >= lo) & (R.tod < hi)] if m.any()}
        if "miss_km" in R and R.miss_km.notna().any():   # v7: how far off the misses are
            M = R[R.miss_km.notna()]
            out["misses"] = dict(n=int(len(M)), links_median=r2(M.miss_links.median()), km_median=r2(M.miss_km.median()),
                                 within_1km=r2((M.miss_km <= 1.0).mean()), within_3km=r2((M.miss_km <= 3.0).mean()),
                                 true_was_2nd=r2((M.true_rank <= 1).mean()), true_in_top5=r2((M.true_rank <= 4).mean()))
            if "pred_hidden" in R:
                out["misses"]["pred_hidden"] = int(R.pred_hidden.fillna(False).astype(bool).sum())
    else:
        bins = ((1, 6, "1-5"), (6, 13, "6-12"), (13, 25, "13-24"), (25, 10 ** 6, "25+"))
        out["by_true_cells"] = {nm: [int(m.sum()), r2(R.iou[m].mean()), r2(R.n_pred[m].mean()), r2(R.n_true[m].mean())]
                                for lo, hi, nm in bins for m in [(R.n_true >= lo) & (R.n_true < hi)] if m.any()}
    return out


def pseudo_summary(R):
    """S_sim: official-style windows weighted by how often the official draw would pick them; S_all: every window."""
    s = R[R.src == "sim"]
    return dict(windows=int(len(R)), sim_windows=int(len(s)), S_sim=_agg1(s, "iou", "draws"), S_all=_agg1(R))


def paired(Ra, Rb, early_day=None):
    """b - a on the same pseudo windows: official-style (draw-weighted sim) and all windows, day-block bootstrap SEs;
    with `early_day`, also the windows of the first days of the month (where the official windows sit)."""
    M = Ra.merge(Rb[["panel", "w", "iou"]], on=["panel", "w"], suffixes=("_a", "_b"))
    M["d"] = M.iou_b - M.iou_a
    sim = M[M.src == "sim"]
    out = dict(n_all=int(len(M)), n_sim=int(len(sim)), S_a_sim=_agg1(sim, "iou_a", "draws"), S_b_sim=_agg1(sim, "iou_b", "draws"),
               S_a_all=_agg1(M, "iou_a"), S_b_all=_agg1(M, "iou_b"),
               d_sim=_agg1(sim, "d", "draws"), se_sim=_boot_se(sim, "d", "draws"), d_all=_agg1(M, "d"), se_all=_boot_se(M, "d"),
               better=int((M.d > 1e-9).sum()), worse=int((M.d < -1e-9).sum()))
    if early_day is not None:
        E = M[M.day < early_day]
        out.update(n_early=int(len(E)), d_early=_agg1(E, "d"), se_early=_boot_se(E, "d"))
    return out, M


def _boot_gain(M, B=400, seed=0):
    """SE of 0.5 * (draw-weighted sim gain + all-window gain), both recomputed on the same day-block resample (the sim
    windows are a subset of all windows, so the two are correlated)."""
    if M is None or len(M) < 2:
        return float("nan")
    rng = np.random.default_rng(seed)
    fam = C.families()
    per = []
    for p, g in M.groupby("panel"):
        _, inv = np.unique(g.day.to_numpy(), return_inverse=True)
        w = g.draws.to_numpy(float) * (g.src == "sim").to_numpy()
        d = g.d.to_numpy(float)
        per.append((p, np.bincount(inv, weights=w * d), np.bincount(inv, weights=w), np.bincount(inv, weights=d),
                    np.bincount(inv)))
    vals = []
    for _ in range(B):
        ps, pa = {}, {}
        for p, swd, sw, sd, n in per:
            s = rng.integers(0, len(n), len(n))
            pa[p] = sd[s].sum() / max(n[s].sum(), 1)
            if sw[s].sum() > 0:
                ps[p] = swd[s].sum() / sw[s].sum()
        a = pd.Series(pa); a = a.groupby(a.index.map(fam)).mean().mean()
        b = a
        if ps:
            b = pd.Series(ps); b = b.groupby(b.index.map(fam)).mean().mean()
        vals.append(0.5 * (a + b))
    return float(np.std(vals))


def pseudo_train_rows(pseudo, pans, split, cond, day_hi=None):
    """Known-label rows of a month's pseudo windows (all, or those before day `day_hi`) as extra training rows."""
    parts = []
    j = C.TEST_SPLITS.index(split)
    for i, p in enumerate(pans):
        ent = pseudo.get(p, {}).get(split, {}).get(cond)
        if ent is None:
            continue
        Wc, _, _, X = ent
        m = X.kn.to_numpy().astype(bool)
        if day_hi is not None:
            m &= Wc.day.to_numpy()[X.wi.to_numpy().astype(np.int64)] < day_hi
        if m.any():
            Z = X.loc[m].drop(columns=["kn"])
            Z["wi"] = Z.wi.to_numpy() + PSEUDO_WI * (1 + j)
            Z["fold"] = -1
            Z["panel_code"] = np.float32(i)
            parts.append(Z)
    return pd.concat(parts, ignore_index=True) if parts else None


def _concat(parts):
    parts = [x for x in parts if x is not None and len(x)]
    return pd.concat(parts, ignore_index=True) if parts else None


# ----------------------------------------------------------------------------- experiments (March pseudo-holdout)
# v4: the variants are chosen with the hedge feature set (no recent-days evidence, so the March pseudo-holdout cannot
# favour a candidate through features the scored windows do not have); the main file uses the same variants.
# v6 tried rel_ (+r), rel_ + dr_ (+rd) and noloc2: none transferred to March (the dr_ group lost there and on the
# public probe). v7: the baseline stays v5's choice (all + noloc); the candidates add the cell-transmission forecast
# (+p), or drop the ramp-flow columns (_nr variants). Two alternatives: both bars rise by 0.5 SE.
DEFAULT_CONFIGS = {"queue_onset": [["v2n", ["all", "noloc"]], ["v2n+p", ["all", "noloc"]],
                                   ["v2n", ["all_nr", "noloc_nr"]]],
                   "queue_ongoing": [["v2n", ["all", "noloc"]]]}
# the March rows of the final files: the v4 recipe, fixed (v4/v5 public scores 0.86890 / 0.86791 reproduce exactly)
MARCH_ROWS = {"queue_onset": ["v2n", ["all"]], "queue_ongoing": ["v2n", ["all", "noloc"]]}


def probe_recipes(cfg, key="probe_onset"):
    """{name: [feature set, variants]} of the onset probe files (v6). `probe_onset` is one recipe [fset, variants]
    (-> "probe6") or a dict of named recipes {"probe<x>": [fset, variants]}; the feature set's base is replaced by the
    main file's ("v2n+rd" -> "v2+rd"). Malformed recipes are errors (a typo must not silently become another recipe).
    v9: key="probe_ongoing" reads the ongoing probes the same way (a dict only); their names must differ from the
    onset probes'."""
    pr = cfg.get(key)
    if not pr:
        return {}
    if key != "probe_onset" and not isinstance(pr, dict):
        raise ValueError(f"{key} must be a dict {{'probe<x>': [feature set, [variants]]}}, got {pr!r}")
    items = list(pr.items()) if isinstance(pr, dict) else [("probe6", pr)]
    out = {}
    for k, v in items:
        if not (isinstance(k, str) and re.fullmatch(r"probe[0-9a-z_]+", k)):
            raise ValueError(f"probe name {k!r}: 'probe' followed by letters/digits, e.g. 'probe6'")
        if not (isinstance(v, (list, tuple)) and len(v) == 2 and isinstance(v[1], (list, tuple)) and len(v[1])):
            raise ValueError(f"probe {k}: recipe must be [feature set, [variants]], got {v!r}")
        _check_fset(v[0])
        out[k] = [v[0], [_variant_name(x) for x in v[1]]]
    if key != "probe_onset" and set(out) & set(probe_recipes(cfg)):
        raise ValueError(f"{key}: probe names {sorted(set(out) & set(probe_recipes(cfg)))} are also onset probes")
    return out


def extra_groups(cfg, cond=None):
    """The v6/v7 feature groups (letters of EXTRA) the tables of `cond` must carry: every group of a "+" feature set
    among its candidates (onset: and the probes). cond=None: all conditions (the official tables are shared).
    Computing them only where needed keeps the large ongoing tables (and the run) as they were."""
    out = {}
    for c in CONDS:
        cands = cfg.get("configs", {}).get(c) or DEFAULT_CONFIGS[c]
        g = set()
        for f, _ in cands:
            g |= set(_check_fset(f)[1])
        if c == "queue_onset":
            for f, _ in probe_recipes(cfg).values():
                g |= set(_check_fset(f)[1])
        if c == "queue_ongoing":   # v9: ongoing probes
            for f, _ in probe_recipes(cfg, "probe_ongoing").values():
                g |= set(_check_fset(f)[1])
        out[c] = g
    g = set().union(*out.values()) if cond is None else out.get(cond, set())
    return "".join(sorted(g))


def extra_conds(cfg):
    """Conditions whose tables carry extra feature groups."""
    return {c for c in CONDS if extra_groups(cfg, c)}


def _variants(names):
    """[(name, variant)]: the variant is the name itself ("all", "noloc", "noloc2"); _feats reads it."""
    return [(v, v) for v in names]


def choose_config(Dl, cond, pseudo, pans, Ps, prm, rounds, seeds, cfg, sets=None):
    """Two draws decide. Each candidate (feature set, variants) is fitted once (seed 0) and scored on the March pseudo
    windows (another demand draw than train) against the first candidate (the baseline). A candidate is eligible when
    its paired gain over every window and over official-style windows is above -1 SE, it does not lose on the first
    days of March (where the scored windows sit) by more than 1 SE and - with `exp_cv` - it does not lose on the
    week-blocked train CV (the training draw) by more than `cv_z` SEs.
    v5 rule: an eligible candidate is taken when (a) its March gain exceeds `exp_z` SEs (joint day-block bootstrap)
    and its train-CV gain is not negative, or (b) its train-CV gain exceeds `exp_cv_z` SEs and its March gain is not
    negative - both draws must point the same way and one of them clearly; the largest March gain among those wins.
    v6 additions: with 2 / 3+ alternatives both bars rise by 0.5 / 0.75 SE; a candidate that adds feature groups needs a
    March gain of at least `exp_cv_march_z` SEs on route (b) as well (a train-CV gain is what new features always show
    on their own draw); at least `exp_fam_share` of the corridor families must gain on March (a gain carried by one
    corridor pair need not repeat in April); and March must offer `exp_min_windows` windows over 3+ days.
    The choice is taken on March windows later than the scored March origins, so it shapes the April rows only (run()
    predicts the March rows with the fixed v4 recipe, MARCH_ROWS)."""
    cands = cfg.get("configs", {}).get(cond) or DEFAULT_CONFIGS[cond]
    for f_, vn_ in cands:   # a typo must fail here, not silently become another recipe
        _check_fset(f_); [_variant_name(v_) for v_ in vn_]
    base = cands[0]
    rep = dict(baseline=base, candidates={})
    if not cfg.get("experiments", True) or not any(cond in pseudo[p].get("validation", {}) for p in pans) or len(cands) == 1:
        rep["chosen"] = base
        return (base[0], _variants(base[1])), rep
    exp_rounds = int(cfg.get("exp_rounds_ongoing", 300)) if cond == "queue_ongoing" else rounds
    cv_rounds = int(cfg.get("exp_cv_rounds", exp_rounds)) if cond == "queue_onset" else exp_rounds
    d0 = int(C.split_days("validation").min()); early = d0 + int(cfg.get("early_days", 7))
    R = {}
    role = "std" if exp_rounds == rounds else "exp"   # same recipe as the final seed-0 model: fitted once, reloaded later
    for fset, vn in cands:
        ms = fit_set(Dl, cond, prm, exp_rounds, _variants(vn), seeds[:1], role=role, fset=fset)
        R[(fset, tuple(vn))] = pseudo_scores(ms, pseudo, pans, Ps, "validation", cond)
        del ms
        C.free()
    Rb = R[(base[0], tuple(base[1]))]
    n_alt = len(cands) - 1
    inc = 0.0 if n_alt <= 1 else (0.5 if n_alt == 2 else 0.75)   # best of several alternatives: a higher bar
    z = float(cfg.get("exp_z", 1.0)) + inc
    zc = float(cfg.get("exp_cv_z", 2.0)) + inc
    min_w = int(cfg.get("exp_min_windows", 40)); fam_share = float(cfg.get("exp_fam_share", 0.75))
    rep.update(z=z, z_cv=zc, min_windows=min_w, fam_share=fam_share)
    rep["baseline_diag_march"] = window_diag(Rb, cond)
    # the training draw: week-blocked train CV of every candidate, per window (variants shared across candidates)
    cvR = {}
    if cfg.get("exp_cv", True) and sets is not None:
        for fset in dict.fromkeys(f for f, _ in cands):
            cfgs = [list(vn) for f, vn in cands if f == fset]
            try:
                for key, Rc in cv_windows(Dl, sets, pans, Ps, cond, prm, cv_rounds, cfgs, fset).items():
                    cvR[(fset, key)] = Rc
            except Exception as e:   # without the training-draw evidence the March-only bar applies (below)
                C.log(f"WARNING T2 train CV for {cond}/{fset} failed ({type(e).__name__}: {e}); March-only rule")
                rep.setdefault("cv_errors", []).append(f"{fset}: {type(e).__name__}: {str(e)[:200]}")
        if (base[0], tuple(base[1])) in cvR:
            rep["baseline_cv"] = dict(S_sim=_agg1(cvR[(base[0], tuple(base[1]))], "iou", "draws"),
                                      windows=int(len(cvR[(base[0], tuple(base[1]))])), rounds=cv_rounds)
            rep["baseline_diag_cv"] = window_diag(cvR[(base[0], tuple(base[1]))], cond)
    cv_z = float(cfg.get("cv_z", 1.0))
    fam = C.families()
    base_mods = set(_check_fset(base[0])[1])
    best, best_key = base, None
    for fset, vn in cands[1:]:
        st, M = paired(Rb, R[(fset, tuple(vn))], early)
        gain = 0.5 * (np.nan_to_num(st["d_sim"]) + st["d_all"])
        se = _boot_gain(M)
        wdm = _weekday(M.day.to_numpy())
        st.update(d_all_weekday=_agg1(M[wdm], "d"), d_all_weekend=_agg1(M[~wdm], "d"))
        early_bad = st.get("n_early", 0) >= 10 and st.get("d_early", 0) < -np.nan_to_num(st.get("se_early", 0))
        ok = (np.nan_to_num(st["d_sim"]) >= -np.nan_to_num(st["se_sim"]) and st["d_all"] >= -np.nan_to_num(st["se_all"])
              and not early_bad)
        # enough March evidence, and a gain that is not carried by one corridor family
        enough = bool(len(M) >= min_w and M.day.nunique() >= 3)
        pmd = M.groupby("panel").d.mean(); pn = M.groupby("panel").size()
        fd = pmd.groupby(pmd.index.map(fam)).mean(); fn = pn.groupby(pn.index.map(fam)).sum()
        fd = fd[fn.reindex(fd.index).fillna(0) >= 5]
        fam_ok = bool(len(fd) < 2 or float((fd >= -1e-12).mean()) >= fam_share)
        st.update(enough=enough, fam_ok=fam_ok, fam_d={str(k): round(float(v), 4) for k, v in fd.items()})
        kb, kc = (base[0], tuple(base[1])), (fset, tuple(vn))
        if kb in cvR and kc in cvR and len(cvR[kb]) and len(cvR[kc]):
            Mc = cvR[kb].merge(cvR[kc][["panel", "w", "iou"]], on=["panel", "w"], suffixes=("_a", "_b"))
            Mc["d"] = Mc.iou_b - Mc.iou_a
            wdc = _weekday(Mc.day.to_numpy())
            st.update(d_cv=_agg1(Mc, "d", "draws"), se_cv=_boot_se(Mc, "d", "draws"), d_cv_all=_agg1(Mc, "d"),
                      d_cv_weekday=_agg1(Mc[wdc], "d", "draws"), n_cv=int(len(Mc)))
            cv_ok = st["d_cv"] >= -cv_z * np.nan_to_num(st["se_cv"])
            st["cv_ok"] = bool(cv_ok)
            ok = ok and cv_ok
        has_cv = "d_cv" in st and np.isfinite(st.get("d_cv", np.nan))
        d_cv = st.get("d_cv", np.nan); se_cv = st.get("se_cv", np.nan)
        adds = bool(set(_check_fset(fset)[1]) - base_mods)          # new feature groups
        mz = float(cfg.get("exp_cv_march_z", 0.5)) if adds else 0.0
        if has_cv:
            strong_march = bool(gain > z * np.nan_to_num(se) and d_cv >= 0)
            march_ok = (gain > 0 and gain >= mz * np.nan_to_num(se)) if adds else gain >= 0   # new groups: March must gain
            strong_cv = bool(np.isfinite(se_cv) and se_cv > 0 and d_cv > zc * se_cv and march_ok)
        else:   # no train-CV evidence: March alone must clear the v4 bar
            z_alone = float(cfg.get("exp_z_alone", 1.5)) + inc
            strong_march = bool(gain > z_alone * np.nan_to_num(se)); strong_cv = False
        take = bool(ok and enough and fam_ok and (strong_march or strong_cv))
        st.update(gain=gain, se_gain=se, eligible=bool(ok), adds_groups=adds, strong_march=strong_march,
                  strong_cv=strong_cv, take=take)
        st["diag_march"] = window_diag(R[(fset, tuple(vn))], cond)
        if kc in cvR:
            st["diag_cv"] = window_diag(cvR[kc], cond)
        rep["candidates"][f"{fset}:{'+'.join(vn)}"] = st
        key_ = (gain, np.nan_to_num(st.get("d_cv", 0.0)))   # the largest March gain wins; ties: the larger train-CV gain
        if take and (best_key is None or key_ > best_key):
            best, best_key = [fset, vn], key_
    if cfg.get("exp_force", False):   # tests only: take the last candidate
        best = list(cands[-1])
    rep["baseline_level"] = pseudo_report(Rb)
    rep["chosen"] = best
    return (best[0], _variants(best[1])), rep


# ----------------------------------------------------------------------------- adaptation
def april_adapt(Dl, std_models, pseudo, pans, Ps, cond, prm, rounds, variants, seeds, cfg, fset="v3"):
    """April models refit on train + the March pseudo windows (all of March is <= T for every April origin).

    Gate, on March only: the same recipe fitted on train + March before `cut`, against the standard models (both
    single-seed), scored on the March pseudo windows from `cut` on, paired by window. Primary statistic: the
    official-style windows (`sim`, weighted by draws), whose gain must exceed `adapt_z` day-block bootstrap SEs over
    at least `adapt_min_windows` windows; the gain over every window must not be negative. Returns the report and
    the adapted models (None when not adopted)."""
    vd = C.split_days("validation")
    cut = int(vd.min() + int(round(len(vd) * cfg.get("adapt_frac", 0.5))))
    alpha = float(cfg.get("adapt_alpha", 2.0))
    z = float(cfg.get("adapt_z", 1.5))
    rep = dict(cut_day=cut, alpha=alpha, z=z, adopted=False)
    Dg = pseudo_train_rows(pseudo, pans, "validation", cond, day_hi=cut)
    Rs = pseudo_scores(std_models[:: len(seeds)], pseudo, pans, Ps, "validation", cond, day_lo=cut)
    if Dg is None or len(Rs) == 0:
        rep["reason"] = "no March pseudo windows on one side of the cut"
        return rep, None
    rep["gate_train_rows"] = int(len(Dg))
    gate = fit_set(Dl, cond, prm, rounds, variants, seeds[:1], Dg, alpha, role="gate", fset=fset)
    del Dg
    Ra = pseudo_scores(gate, pseudo, pans, Ps, "validation", cond, day_lo=cut)
    del gate
    C.free()
    st, M = paired(Rs, Ra)
    rep.update(n_eval_sim=st["n_sim"], n_eval_all=st["n_all"], S_std_sim=st["S_a_sim"], S_ad_sim=st["S_b_sim"],
               d_sim=st["d_sim"], se_sim=st["se_sim"], d_all=st["d_all"], se_all=st["se_all"],
               better=st["better"], worse=st["worse"])
    sim = M[M.src == "sim"]
    adopt = (len(sim) >= cfg.get("adapt_min_windows", 40) and sim.day.nunique() >= 3 and rep["se_sim"] > 0
             and rep["d_sim"] > z * rep["se_sim"] and rep["d_all"] >= 0)
    if cfg.get("adapt_force", False):
        adopt, rep["forced"] = True, True
    rep["adopted"] = bool(adopt)
    if not adopt:
        return rep, None
    Df = pseudo_train_rows(pseudo, pans, "validation", cond)
    rep["final_train_rows"] = int(len(Df))
    return rep, fit_set(Dl, cond, prm, rounds, variants, seeds, Df, alpha, role="adapt", fset=fset)


def late_blocks(cfg, split):
    """Day blocks [lo, hi) of a month for the late-window adaptation: block i starts on day `late_blocks[i]` of the month
    (0 = the first day) and ends where the next starts (the last one at the end of the month)."""
    days = C.split_days(split)
    if len(days) == 0:
        return []
    d0, end = int(days.min()), int(days.max()) + 1
    offs = sorted({int(o) for o in cfg.get("late_blocks", [6]) if int(o) > 0})
    cuts = [d0 + o for o in offs if d0 + o < end]
    return [(c, cuts[i + 1] if i + 1 < len(cuts) else end) for i, c in enumerate(cuts)]


def early_blocks(cfg, split, cond):
    """v8: day blocks [lo, hi) before the first late block for `cond` (`late_early` {cond: [start days]}, 0 = the first
    day of the month); the last early block ends where the first late block starts."""
    starts = sorted({int(o) for o in (cfg.get("late_early") or {}).get(cond, []) if int(o) > 0})
    days = C.split_days(split)
    if not starts or len(days) == 0:
        return []
    d0, end = int(days.min()), int(days.max()) + 1
    late = sorted({int(o) for o in cfg.get("late_blocks", [6]) if int(o) > 0})
    stop = d0 + late[0] if late else end
    cuts = [d0 + o for o in starts if d0 + o < stop]
    return [(c, cuts[i + 1] if i + 1 < len(cuts) else stop) for i, c in enumerate(cuts)]


def n_official(offis, pans, cond, split, lo, hi):
    """Number of official windows of `cond` in `split` whose origin day is in [lo, hi) (task metadata only)."""
    n = 0
    for p in pans:
        if split in offis[p]:
            wi = offis[p][split][0]
            d = wi["T"].to_numpy() // C.SLOTS
            n += int(((wi.condition == cond).to_numpy() & (d >= lo) & (d < hi)).sum())
    return n


def late_gate(Dl, std1, pseudo, pans, Ps, cond, prm, rounds, variants, cfg, fset="v2n", seed=0, blocks=None,
              role="lgate"):
    """v5. Same-month evidence for the official windows late in their month, tested on March first. For every block
    [lo, hi) of `late_blocks`, the recipe fitted on train + the March pseudo windows of the days before lo is scored
    against the standard models (single seed each) on the March pseudo windows of the block, paired by window; the
    blocks are pooled. Adopted when the pooled official-style gain exceeds `late_z` day-block bootstrap SEs over at
    least `late_min_windows` windows on at least three days and the gain over every window is not negative. Every
    training label of the gate ends before the days it is scored on, as it will for the April windows it decides
    about. It tests the hedge's recipe (train vs train + the month's earlier windows); the main file applies the same
    addition on top of its March-adapted training set, which March cannot test (March has no earlier test month)."""
    d0 = int(C.split_days("validation").min())
    alpha = float(cfg.get("late_alpha", cfg.get("adapt_alpha", 2.0)))
    z = float(cfg.get("late_z", 1.5))
    blocks = late_blocks(cfg, "validation") if blocks is None else blocks
    rep = dict(blocks=[lo - d0 for lo, _ in blocks], alpha=alpha, z=z, adopted=False, per_block={})
    Ms = []
    for lo, hi in blocks:
        extra = pseudo_train_rows(pseudo, pans, "validation", cond, day_hi=lo)
        Rs = pseudo_scores(std1, pseudo, pans, Ps, "validation", cond, day_lo=lo, day_hi=hi)
        if extra is None or len(Rs) == 0:
            rep["per_block"][str(lo - d0)] = dict(n=0)
            continue
        ms = fit_set(Dl, cond, prm, rounds, variants, [seed], extra, alpha, role=f"{role}:{lo - d0}", fset=fset)
        n_rows = int(len(extra))
        del extra
        Ra = pseudo_scores(ms, pseudo, pans, Ps, "validation", cond, day_lo=lo, day_hi=hi)
        del ms
        C.free()
        st, M = paired(Rs, Ra)
        rep["per_block"][str(lo - d0)] = dict(n=st["n_all"], n_sim=st["n_sim"], extra_rows=n_rows, d_all=st["d_all"],
                                              d_sim=st["d_sim"])
        Ms.append(M)
    if Ms:
        M = pd.concat(Ms, ignore_index=True)
        sim = M[M.src == "sim"]
        rep.update(n_eval_sim=int(len(sim)), n_eval_all=int(len(M)),
                   S_std_sim=_agg1(sim, "iou_a", "draws"), S_late_sim=_agg1(sim, "iou_b", "draws"),
                   d_sim=_agg1(sim, "d", "draws"), se_sim=_boot_se(sim, "d", "draws"), d_all=_agg1(M, "d"),
                   se_all=_boot_se(M, "d"), better=int((M.d > 1e-9).sum()), worse=int((M.d < -1e-9).sum()))
        adopt = (len(sim) >= cfg.get("late_min_windows", 30) and sim.day.nunique() >= 3
                 and np.isfinite(rep["se_sim"]) and rep["se_sim"] > 0
                 and rep["d_sim"] > z * rep["se_sim"] and rep["d_all"] >= 0)
    else:
        rep["reason"] = "no March pseudo windows in or before the blocks"
        adopt = False
    if cfg.get("late_force", False):   # tests only
        adopt, rep["forced"] = True, True
    rep["adopted"] = bool(adopt)
    return rep


def late_fit_predict(Dl, pseudo, offis, pans, Ps, cond, prm, rounds, variants, seeds, cfg, fset, with_march, probs, tag,
                     blocks=None, role="late"):
    """v5. Models for the official April windows of every late block [lo, hi): train + the April pseudo windows of the
    days before lo (+ all March pseudo windows when `with_march`), predicted for the official April windows whose origin
    falls in the block. The April pseudo windows of a block end before its first day, so every label used is older
    than every origin these models predict. Blocks without an official window of `cond` are skipped."""
    a0 = int(C.split_days("private").min())
    alpha = float(cfg.get("late_alpha", cfg.get("adapt_alpha", 2.0)))
    march = None
    rep = dict(with_march=bool(with_march), per_block={}, windows=0)
    for lo, hi in (late_blocks(cfg, "private") if blocks is None else blocks):
        n_off = n_official(offis, pans, cond, "private", lo, hi)
        if n_off == 0:
            continue
        apr = pseudo_train_rows(pseudo, pans, "private", cond, day_hi=lo)
        if apr is None or len(apr) == 0:
            rep["per_block"][str(lo - a0)] = dict(official=n_off, april_rows=0)
            continue
        if with_march and march is None:
            march = pseudo_train_rows(pseudo, pans, "validation", cond)
        extra = _concat([apr, march]) if march is not None else apr
        ms = fit_set(Dl, cond, prm, rounds, variants, seeds, extra, alpha,
                     role=f"{role}:private:{tag}:{int(bool(with_march))}:{lo - a0}", fset=fset)
        n = int(predict_official(ms, offis, pans, Ps, cond, probs, ("private",), day_min=lo, day_max=hi))
        rep["per_block"][str(lo - a0)] = dict(official=n_off, predicted=n, april_rows=int(len(apr)),
                                              extra_rows=int(len(extra)))
        rep["windows"] += n
        del ms, extra, apr
        C.free()
    return rep


def early_step(Dl, pseudo, offis, pans, Ps, cond, prm, rounds, variants, seeds, lseeds, cfg, fh, fa, adapted,
               early_a, early_h):
    """v8: the early blocks of `cond` (`early_blocks`). Gate on March with the hedge's recipe (train vs train + the March
    pseudo windows before each block, scored on the March windows in it; `late_z`, `late_min_windows`), then models for
    the official April windows of each block: hedge = train + April's pseudo windows before the block, main = the same
    + all of March when the main file's April adaptation was adopted (`adapted`). Predictions go to `early_h` /
    `early_a` and override every other source for those windows."""
    vb, pb = early_blocks(cfg, "validation", cond), early_blocks(cfg, "private", cond)
    a0 = int(C.split_days("private").min())
    n_off = sum(n_official(offis, pans, cond, "private", lo, hi) for lo, hi in pb)
    rep = dict(blocks=[lo - a0 for lo, _ in pb], block_ends=[hi - a0 for _, hi in pb], official=n_off, adopted=False)
    if n_off == 0:
        rep["reason"] = "no official April window of this condition in the early blocks"
        return rep
    if not any(cond in pseudo[p].get("validation", {}) for p in pans):
        rep["reason"] = "no March pseudo windows"
        return rep
    std1 = fit_set(Dl, cond, prm, rounds, variants, [seeds[0]], fset=fh)   # the hedge's standard models (checkpointed)
    g = late_gate(Dl, std1, pseudo, pans, Ps, cond, prm, rounds, variants, dict(cfg, late_force=cfg.get("early_force", False)),
                  fh, seed=seeds[0], blocks=vb, role="egate")
    del std1
    C.free()
    rep.update({k: v for k, v in g.items() if k not in ("blocks",)}, gate_blocks=g.get("blocks"))
    if rep["adopted"]:
        rep["hedge"] = late_fit_predict(Dl, pseudo, offis, pans, Ps, cond, prm, rounds, variants, lseeds, cfg, fh, False,
                                        early_h, "hedge", blocks=pb, role="early")
        rep["main"] = late_fit_predict(Dl, pseudo, offis, pans, Ps, cond, prm, rounds, variants, lseeds, cfg, fa,
                                       bool(adapted), early_a, "main", blocks=pb, role="early")
    return rep


def online_eval(Dl, std1, pseudo, pans, Ps, cond, prm, rounds, variants, cfg, fset="v3"):
    """Within-month online adaptation, simulated on March: for each block of days starting at offset o, the models
    are refitted on train + the March pseudo windows of the days before the block and scored on the block, against
    the standard models (single seed each). The early blocks are what the official windows (first days of the month)
    would see. Adopted if the official-style gain exceeds `online_z` SEs, the all-window gain is not negative and the
    blocks that start within the first five days (where the official windows sit) do not lose on average."""
    vd = C.split_days("validation"); d0 = int(vd.min()); nd = len(vd)
    offs = [o for o in cfg.get("online_blocks", [1, 2, 4, 7, 11, 16, 22]) if 0 < o < nd]
    alpha = float(cfg.get("online_alpha", cfg.get("adapt_alpha", 2.0)))
    z = float(cfg.get("online_z", 1.5))
    rep = dict(blocks=offs, alpha=alpha, z=z, adopted=False, per_block={})
    Ms = []
    for i, o in enumerate(offs):
        lo = d0 + o; hi = d0 + (offs[i + 1] if i + 1 < len(offs) else nd)
        extra = pseudo_train_rows(pseudo, pans, "validation", cond, day_hi=lo)
        Rs = pseudo_scores(std1, pseudo, pans, Ps, "validation", cond, day_lo=lo, day_hi=hi)
        if extra is None or len(Rs) == 0:
            continue
        ms = fit_set(Dl, cond, prm, rounds, variants, [0], extra, alpha, role=f"oneval:{o}", fset=fset)
        Ra = pseudo_scores(ms, pseudo, pans, Ps, "validation", cond, day_lo=lo, day_hi=hi)
        del ms, extra
        C.free()
        st, M = paired(Rs, Ra)
        rep["per_block"][str(o)] = dict(n=st["n_all"], d_all=st["d_all"], d_sim=st["d_sim"])
        Ms.append(M.assign(block=o))
    if not Ms:
        rep["reason"] = "no March pseudo windows to simulate with"
        return rep
    M = pd.concat(Ms, ignore_index=True)
    sim = M[M.src == "sim"]
    E = M[M.block <= 5]          # blocks that start within the days the official windows occupy
    rep.update(n_all=int(len(M)), n_sim=int(len(sim)), d_sim=_agg1(sim, "d", "draws"), se_sim=_boot_se(sim, "d", "draws"),
               d_all=_agg1(M, "d"), se_all=_boot_se(M, "d"), n_early=int(len(E)), d_early=_agg1(E, "d"), se_early=_boot_se(E, "d"),
               better=int((M.d > 1e-9).sum()), worse=int((M.d < -1e-9).sum()))
    early_ok = len(E) < 10 or rep["d_early"] >= 0
    adopt = (len(sim) >= cfg.get("online_min_windows", 40) and rep["se_sim"] > 0 and rep["d_sim"] > z * rep["se_sim"]
             and rep["d_all"] >= 0 and early_ok)
    if cfg.get("online_force", False):
        adopt, rep["forced"] = True, True
    rep["adopted"] = bool(adopt)
    return rep


def online_official(Dl, pseudo, offis, pans, Ps, cond, prm, rounds, variants, cfg, fset, splits, with_march, probs):
    """Per-day refits for the official windows: a window whose origin falls on day D is predicted by models trained
    on train + the pseudo windows of its own month from days before D (+ all of March for April when `with_march`).
    Every training label ends before D, so before the origin. Windows with no such data keep the standard models."""
    alpha = float(cfg.get("online_alpha", cfg.get("adapt_alpha", 2.0)))
    seeds = list(cfg.get("online_seeds", [0, 1]))
    days_done = {}
    for split in splits:
        days = sorted({int(T // C.SLOTS) for p in pans if split in offis[p]
                       for T, c in zip(offis[p][split][0]["T"], offis[p][split][0].condition) if c == cond})
        for D in days:
            parts = [pseudo_train_rows(pseudo, pans, split, cond, day_hi=D)]
            if split == "private" and with_march:
                parts.append(pseudo_train_rows(pseudo, pans, "validation", cond))
            extra = _concat(parts)
            if extra is None:
                continue
            ms = fit_set(Dl, cond, prm, rounds, variants, seeds, extra, alpha,
                         role=f"online:{split}:{D}:{int(with_march and split == 'private')}", fset=fset)
            predict_official(ms, offis, pans, Ps, cond, probs, (split,), day=D)
            days_done[f"{split}:{D}"] = int(len(extra))
            del ms, extra
            C.free()
    return days_done


# ----------------------------------------------------------------------------- driver
def _frames(sets, pans, cond):
    """Per-panel training tables of one condition (never concatenated: a copy of the largest table is the
    memory peak of this task)."""
    Dl = []
    for i, p in enumerate(pans):
        if cond in sets[p] and sets[p][cond][2] is not None:
            D = sets[p][cond][2]
            D["panel_code"] = np.float32(i)
            Dl.append(D)
    return Dl


def write_rows(prob_sets, offis, pans, Ps, quiet=frozenset()):
    """Template rows of every test split; each window takes its probabilities from the first set that has it (windows
    in `quiet` may have none: their rows are filled from elsewhere, v7 reused ongoing rows)."""
    outs = []
    for p in pans:
        L = Ps[p]["L"]; links = np.asarray(Ps[p]["links"])
        lid = {l: i for i, l in enumerate(links)}
        for split in C.TEST_SPLITS:
            tp = C.rel() / "task2" / p / split / "sample_submission_queue.csv"
            if not tp.exists():
                continue
            tm = pd.read_csv(tp, dtype=str)
            wi = offis[p][split][0]
            T0 = dict(zip(wi.window_id, wi["T"]))
            tm["h"] = C.tindex(tm.timestamp) - tm.window_id.map(T0).to_numpy()
            tm["li"] = tm.link_id.map(lid)
            pred = np.zeros(len(tm), np.int8)
            for wid, idx in tm.groupby("window_id").indices.items():
                hit = [ps[p][split][wid] for ps in prob_sets if wid in ps.get(p, {}).get(split, {})]
                if not hit:
                    if wid not in quiet:
                        C.log(f"WARNING T2: no prediction for window {wid}")
                    continue
                cond, pr = hit[0]
                sel = decode_window(cond, pr, L)
                h = tm.h.to_numpy()[idx]; li = tm.li.to_numpy()[idx]
                ok = (h >= 1) & (h <= K) & pd.notna(li)
                vals = np.zeros(len(idx), np.int8)
                vals[ok] = sel[h[ok].astype(int) - 1, li[ok].astype(int)]
                pred[idx] = vals
            tm["queue_pred"] = pred
            outs.append(tm[["window_id", "timestamp", "link_id", "queue_pred"]])
    return pd.concat(outs, ignore_index=True)


# v7: the ongoing rows of the previous Task 2 run are reused when that run had the same ongoing recipe (the ongoing
# code and settings did not change since v5; refitting them costs about two hours of the run)
ONGOING_RECIPE = "v5"      # bump whenever anything on the ongoing path changes (then nothing older is reused)
# v9: what the early April windows of a previous run depend on beyond REUSE_KEYS (its files are reused only when equal);
# EARLY_RECIPE: bump whenever the early-window code changes (then no earlier early windows are reused)
EARLY_RECIPE = "v8"
EARLY_KEYS = ("late_early", "late_blocks", "late_alpha", "late_z", "late_min_windows", "late_seeds", "early_force",
              "adapt_alpha", "april_adapt", "pseudo_og_every", "pseudo_og_per_day")
REUSE_KEYS = ("seed", "imputer", "max_ongoing_cand", "use_T", "ongoing_seeds", "ongoing_rounds", "pseudo",
              "pseudo_og_every", "pseudo_og_per_day", "fset_main", "fset_hedge", "april_adapt", "adapt_alpha",
              "adapt_frac", "adapt_z", "adapt_min_windows", "late_adapt", "late_blocks", "late_conds", "late_alpha",
              "late_z", "late_min_windows", "late_seeds", "adapt_force", "late_force")


def _md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 24), b""):
            h.update(blk)
    return h.hexdigest()


def _ongoing_keys():
    """(window_id, timestamp, link_id) of every official ongoing row of the test-split templates."""
    keys = []
    for p in C.task2_panels():
        for split in C.TEST_SPLITS:
            d = C.rel() / "task2" / p / split
            if not (d / "sample_submission_queue.csv").exists() or not (d / "window_index.csv").exists():
                continue
            wi = pd.read_csv(d / "window_index.csv", dtype={"window_id": str})
            og = set(wi.window_id[wi.condition == "queue_ongoing"])
            tm = pd.read_csv(d / "sample_submission_queue.csv", dtype=str, usecols=["window_id", "timestamp", "link_id"])
            keys.append(tm[tm.window_id.isin(og)])
    return pd.MultiIndex.from_frame(pd.concat(keys, ignore_index=True)) if keys else None


BASE_FILE = "task2_ongoing_base.csv"   # v8: the ongoing rows of both files before the early-window overlay


def reusable_ongoing(cfg):
    """{"main": rows, "hedge": rows, "info": ...} of the previous run's files in WORK/pred when `reuse_ongoing` is set
    and that run used the same ongoing settings (REUSE_KEYS), recipe (variants, feature sets, March rows) and release,
    and its files hold every official ongoing row once with a 0/1 value; None otherwise (logged). v8: the rows are taken
    from the previous run's base file (the ongoing rows before its early-window overlay) when it wrote one, so the early
    windows are always recomputed from the same base; a run without a base file is reused only if none of its early
    windows were refitted (v5-v7 runs never were)."""
    if not cfg.get("reuse_ongoing", False):
        return None
    d = C.WORK / "pred"
    rp, fa, fh = d / "task2_report.json", d / "task2.csv", d / "task2_hedge.csv"
    fb = d / BASE_FILE
    use_base = False
    why, old, ch = None, {}, {}
    try:
        if not (rp.exists() and fa.exists() and fh.exists()):
            why = "no previous Task 2 files"
        else:
            from .data import fingerprint
            old = json.load(open(rp))
            oc = old.get("cfg", {})
            same = lambda k: json.dumps(oc.get(k), sort_keys=True, default=str) == json.dumps(cfg.get(k), sort_keys=True, default=str)
            diff = [k for k in REUSE_KEYS if not same(k)]
            ch = (old.get("chosen") or {}).get("queue_ongoing") or {}
            cur = (cfg.get("configs") or {}).get("queue_ongoing") or DEFAULT_CONFIGS["queue_ongoing"]
            if old.get("version") not in ("v5", "v6", "v7", "v8", "v9"):
                why = f"previous run is {old.get('version')}"
            elif "fingerprint" in old and old["fingerprint"] != fingerprint():
                why = "the previous run used another release"
            elif diff:
                why = f"ongoing settings differ: {diff}"
            elif old.get("ongoing_recipe", ONGOING_RECIPE) != ONGOING_RECIPE:
                why = f"ongoing recipe {old.get('ongoing_recipe')} (now {ONGOING_RECIPE})"
            elif not (old.get("files") or {}).get(BASE_FILE) and ((old.get("early") or {}).get("queue_ongoing") or {}).get("adopted"):
                # v8: its early April windows carry that run's early models and there is no base file to start from
                why = "the previous run refitted its early April windows and left no base rows"
            elif len(cur) != 1:
                why = "ongoing candidates are configured"
            elif not (old.get("files") or {}).get("task2.csv") or not (old.get("files") or {}).get("task2_hedge.csv"):
                why = "the previous run did not finish its files"
            else:
                f0, v0 = cur[0]; fa_, fh_ = cfg.get("fset_main", "v2"), cfg.get("fset_hedge", "v2n")
                mr_f, mr_v = (cfg.get("march_rows") or {}).get("queue_ongoing") or MARCH_ROWS["queue_ongoing"]
                want = dict(variants=[_variant_name(v) for v in v0], main=_fset_with(fa_, f0), hedge=_fset_with(fh_, f0),
                            march_rows=dict(main=_fset_with(fa_, mr_f), hedge=_fset_with(fh_, mr_f),
                                            variants=[_variant_name(v) for v in mr_v]))
                if any(json.dumps(ch.get(k), sort_keys=True) != json.dumps(w_, sort_keys=True) for k, w_ in want.items()):
                    why = "ongoing recipe (variants / feature sets / March rows) differs"
                md5s = old.get("files_md5") or {}
                use_base = bool((old.get("files") or {}).get(BASE_FILE))
                if why is None and use_base and not (fb.exists() and md5s.get(BASE_FILE) == _md5(fb)):
                    why = f"{BASE_FILE} is missing or not the file its report describes"
                for fn, path in (("task2.csv", fa), ("task2_hedge.csv", fh)):
                    if why is None and fn in md5s and md5s[fn] != _md5(path):
                        why = f"{fn} is not the file its report describes"
    except Exception as e:
        why = f"previous report unreadable ({type(e).__name__})"
    if why is not None:
        C.log(f"T2 ongoing rows recomputed: {why}")
        return None
    key = ["window_id", "timestamp", "link_id"]; dt = {"window_id": str, "timestamp": str, "link_id": str}
    if use_base:
        B = pd.read_csv(fb, dtype=dt).set_index(key)
        main, hedge = B["main"].rename("queue_pred"), B["hedge"].rename("queue_pred")
    else:
        main = pd.read_csv(fa, dtype=dt).set_index(key).queue_pred
        hedge = pd.read_csv(fh, dtype=dt).set_index(key).queue_pred
    need = _ongoing_keys()
    for nm, rows in (("task2.csv", main), ("task2_hedge.csv", hedge)):   # every ongoing row, once, 0/1
        bad = rows.index.duplicated().any() or need is None
        if not bad:
            v = rows.reindex(need)
            bad = bool(v.isna().any() or not v.isin([0, 1]).all())
        if bad:
            C.log(f"T2 ongoing rows recomputed: {nm} of the previous run does not hold every ongoing row once")
            return None
    if old.get("reused_ongoing"):   # a chain of reuses: keep the run that fitted these rows as the source
        info = dict(old["reused_ongoing"]); info["via"] = dict(version=old.get("version"), code=old.get("code"))
    else:
        info = dict(from_version=old.get("version"), from_code=old.get("code"), chosen=ch,
                    adapt=(old.get("adapt") or {}).get("queue_ongoing"), late=(old.get("late") or {}).get("queue_ongoing"),
                    pseudo=(old.get("pseudo") or {}).get("queue_ongoing"))
    info["base_file"] = use_base
    out = dict(main=main, hedge=hedge, info=info, final=None)
    # v9: the previous run's early April windows too (its final rows), when it refitted them with these settings - the
    # early refit is the most expensive part of the ongoing path (about 3 h on the full data)
    why_e = None
    oc = old.get("cfg", {})
    same_e = lambda k: json.dumps(oc.get(k), sort_keys=True, default=str) == json.dumps(cfg.get(k), sort_keys=True, default=str)
    e_conds = [c for c in CONDS if early_blocks(cfg, "private", c)]
    if not cfg.get("reuse_early", True):
        why_e = "reuse_early is off"
    elif not e_conds:
        why_e = "no early blocks configured"
    elif not use_base:
        why_e = "the previous run left no base rows"
    elif [k for k in EARLY_KEYS if not same_e(k)]:
        why_e = f"early settings differ: {[k for k in EARLY_KEYS if not same_e(k)]}"
    elif old.get("early_recipe", "v8") != EARLY_RECIPE:
        why_e = f"early recipe {old.get('early_recipe', 'v8')} (now {EARLY_RECIPE})"
    elif any(c not in (old.get("early") or {}) or "adopted" not in (old.get("early") or {})[c] for c in e_conds):
        why_e = "the previous run has no early-window result"
    if why_e is None:
        fin = {}
        for nm, path in (("main", fa), ("hedge", fh)):
            rows = pd.read_csv(path, dtype=dt).set_index(key).queue_pred
            v = rows.reindex(need)
            if rows.index.duplicated().any() or v.isna().any() or not v.isin([0, 1]).all():
                why_e = f"the previous {nm} file does not hold every ongoing row once"
                break
            fin[nm] = v
        if why_e is None:
            out["final"] = fin
            out["old_early"] = {c: old["early"][c] for c in e_conds}
            # a chain of reuses keeps the run that fitted these windows as the source
            src_e = [old["early"][c].get("reused_from") for c in e_conds if old["early"][c].get("reused_from")]
            info["early_from"] = src_e[0] if src_e else dict(version=old.get("version"), code=old.get("code"))
    if why_e is not None:
        C.log(f"T2 early April windows recomputed: {why_e}")
    C.log(f"T2 ongoing rows reused from the previous run ({old.get('version')}, code {old.get('code')}"
          f"{', its base rows' if use_base else ''}{', with its early April windows' if out['final'] else ''})")
    return out


def fit_probes(Dl, cond, recipes, prm, rounds, seeds, fset_a, offis, pans, Ps, pseudo, probs_p, report):
    """Probe models (v6 onset, v9 ongoing): train-only models of a recipe fixed in advance, predicted for the March
    windows of `cond` only - the probe file is the main file with those rows replaced (a public read of the recipe on
    the scored March windows). The March pseudo-holdout of the same models is reported next to it."""
    has_march = any(cond in pseudo[p].get("validation", {}) for p in pans)
    for pname, (pf0, pv) in recipes.items():
        pf = _fset_with(fset_a, pf0)
        pm = fit_set(Dl, cond, prm, rounds, _variants(pv), seeds, fset=pf)
        predict_official(pm, offis, pans, Ps, cond, probs_p.setdefault(pname, {}), ("validation",))
        rpp = report.setdefault("probes", {})[pname] = dict(condition=cond, fset=pf, variants=list(pv))
        if has_march:
            Rp = pseudo_scores(pm, pseudo, pans, Ps, "validation", cond)
            rpp["pseudo_march"] = pseudo_report(Rp)
            rpp["diag_march"] = window_diag(Rp, cond)
        C.log(f"T2 probe {pname} ({cond}, {pf}, {list(pv)}): March pseudo-holdout {rpp.get('pseudo_march')}")
        del pm
        C.free()


def _previous_files():
    """The main and hedge rows in WORK/pred before this run writes its own (None if there are none), for the report."""
    d = C.WORK / "pred"
    fa, fh = d / "task2.csv", d / "task2_hedge.csv"
    if not (fa.exists() and fh.exists()):
        return None
    try:
        key = ["window_id", "timestamp", "link_id"]; dt = {"window_id": str, "timestamp": str, "link_id": str}
        info = {}
        if (d / "task2_report.json").exists():
            r = json.load(open(d / "task2_report.json"))
            info = dict(version=r.get("version"), code=r.get("code"))
        return dict(main=pd.read_csv(fa, dtype=dt).set_index(key).queue_pred,
                    hedge=pd.read_csv(fh, dtype=dt).set_index(key).queue_pred, info=info)
    except Exception as e:  # noqa: BLE001  (a report helper only)
        C.log(f"T2 previous files unreadable ({type(e).__name__}); no comparison")
        return None


def _overlay(frame, src, ids):
    """Rows of the windows `ids` take their queue_pred from `src` (indexed by window_id, timestamp, link_id)."""
    m = frame.window_id.isin(ids).to_numpy()
    if not m.any():
        return frame
    k = pd.MultiIndex.from_frame(frame.loc[m, ["window_id", "timestamp", "link_id"]])
    v = src.reindex(k)
    if v.isna().any():
        raise RuntimeError(f"T2: {int(v.isna().sum())} reused ongoing rows are missing from the previous files; "
                           f"rerun with reuse_ongoing=False")
    frame.loc[m, "queue_pred"] = v.to_numpy().astype(np.int8)
    return frame


def run(cfg: dict | None = None):
    """v5-v7. Datasets; per condition: the variants and (v6/v7, onset) the extra feature groups, chosen by the two-draw
    rule (March pseudo-holdout + train CV), then two model families with the same choice, written as two complete
    files:
      task2.csv        main (A): the 0.86840 features (v2, incl. the last-7-days rec7_); March rows from the standard
                       models; April rows from models refit on train + all March pseudo windows when the March gate
                       passes (all of March precedes every April origin), else the standard models. The bet that
                       April resembles March.
      task2_hedge.csv  hedge (H): train only, no recent-days features (v2n), both months. The bet that April
                       resembles train (for the first April days the "recent days" are late March, another demand draw).
    v5: in both files the official April windows of the late blocks (from day 7 of April on) are predicted by models
    that also learn from April's own pseudo windows of the days before their block (main: on top of all of March when
    the April gate passed), when adding a month's own earlier windows wins on March first (tested with the hedge's
    recipe: train vs train + early March).
    The variants, feature groups and the late adaptation are chosen on March pseudo windows (later than the scored
    March origins), so they shape the April rows only; the March rows of both files keep the v4 recipe (MARCH_ROWS,
    train-only models), so both files reproduce their v4 public scores. v6 probe files (`probe_onset`) re-predict only
    the March onset rows of the main file with recipes fixed in advance (public reads; never final picks).
    No April label is evaluated or used to choose anything; April pseudo windows only become training rows of models
    for April windows on later days."""
    cfg = cfg or {}
    for k_ in ("fset_main", "fset_hedge"):   # a base only: the extra groups come from the onset candidates
        if "+" in str(cfg.get(k_, "")) or (k_ in cfg and _check_fset(cfg[k_])[1]):
            raise ValueError(f"{k_}={cfg[k_]!r}: give a base feature set ({FSETS}); feature groups come from the candidates")
    probe_recipes(cfg)   # malformed probe recipes fail before hours of work
    for cond_, cands_ in (cfg.get("configs") or {}).items():
        for f_, vn_ in cands_:
            _check_fset(f_); [_variant_name(v_) for v_ in vn_]
    for cond_, (f_, vn_) in (cfg.get("march_rows") or {}).items():
        _check_fset(f_); [_variant_name(v_) for v_ in vn_]
    rng = np.random.default_rng(cfg.get("seed", 0))
    t0 = time.time()
    set_checkpoints(cfg)
    pans = C.task2_panels()
    imputer = fit_imputer(pans) if cfg.get("imputer", True) else None
    C.log(f"T2 label imputer ready ({time.time() - t0:.0f}s)")
    sets, offis, Ps, pseudo, shift = {}, {}, {}, {}, {}
    prev = _previous_files()        # v8: what the files were before this run (only to report what changed)
    reuse = reusable_ongoing(cfg)   # v7: the ongoing rows of the previous run, when its ongoing recipe is this one
    skip = ("queue_ongoing",) if reuse is not None else ()
    early_conds = [c for c in CONDS if early_blocks(cfg, "private", c)]   # v8
    final_reuse = reuse is not None and reuse.get("final") is not None   # v9: its early April windows as well
    need_tables = set() if final_reuse else set(early_conds)   # the early blocks need the tables of a reused condition
    if probe_recipes(cfg, "probe_ongoing"):
        need_tables.add("queue_ongoing")                       # v9: so do the ongoing probes
    skip_build = tuple(c for c in skip if c not in need_tables)
    skip_april = tuple(c for c in skip if final_reuse)   # nothing April-side is refitted for a fully reused condition
    for i, p in enumerate(pans):
        P, out, offi, ps = build_panel_sets(p, imputer, rng, cfg, skip=skip_build, skip_april=skip_april)
        for split in ps:
            for cond in ps[split]:
                ps[split][cond][3]["panel_code"] = np.float32(i)
        sets[p] = out; offis[p] = offi; pseudo[p] = ps; Ps[p] = dict(L=P.L, links=P.net.links, mp=np.asarray(P.mp, np.float64))
        npw = ", ".join(f"{s[:3]} {c.split('_')[1]} {len(v[0])}" for s in ps for c, v in ps[s].items())
        C.log(f"T2 {p}: train windows onset {len(out.get('queue_onset', ([],))[0])}, "
              f"ongoing {len(out.get('queue_ongoing', ([],))[0])}; pseudo windows: {npw or 'none'}")
        if cfg.get("diag", True):
            try:
                shift[p] = shift_stats(P)
            except Exception as e:   # diagnostics never stop the pipeline
                shift[p] = {"error": repr(e)[:300]}
        P.dprof = None
        del P
        C.free()
    fset_a = cfg.get("fset_main", "v2"); fset_h = cfg.get("fset_hedge", "v2n")
    from .data import fingerprint
    report = dict(version=VERSION, code=code_hash(), fingerprint=fingerprint(), ongoing_recipe=ONGOING_RECIPE,
                  early_recipe=EARLY_RECIPE, cfg=cfg,
                  fsets=dict(main=fset_a, hedge=fset_h),
                  cv={}, pseudo={}, experiments={}, chosen={}, adapt={}, late={}, early={},
                  windows=official_window_calendar(offis, pans), diag=dict(shift=shift, pseudo={}))
    if reuse is not None:
        report["reused_ongoing"] = reuse["info"]
        report["chosen"]["queue_ongoing"] = reuse["info"].get("chosen")
    if "p" in extra_groups(cfg, "queue_onset"):
        try:   # v7: the cell-transmission forecast on its own (diagnostics never stop the pipeline)
            report["diag"]["ctm"] = ctm_diag(sets, pseudo, pans, Ps)
            C.log(f"T2 CTM forecast alone (onset): {report['diag']['ctm']}")
        except Exception as e:
            report["diag"]["ctm"] = {"error": repr(e)[:300]}
    probs_a, probs_h, apr_a, late_a, late_h, probs_p = {}, {}, {}, {}, {}, {}
    early_a, early_h = {}, {}
    for cond in CONDS:
        if cond in skip:
            C.log(f"T2 {cond}: rows reused from the previous run ({reuse['info'].get('from_version')}), no refit")
            if cond in early_conds and final_reuse:   # v9: the previous run's early April windows, as they were
                report["early"][cond] = dict(reuse["old_early"][cond], reused_from=reuse["info"].get("early_from"))
                C.log(f"T2 early April windows {cond}: reused from the previous run "
                      f"(adopted {report['early'][cond].get('adopted')})")
            if cond == "queue_ongoing" and probe_recipes(cfg, "probe_ongoing"):   # v9: ongoing probes (March rows)
                Dl = _frames(sets, pans, cond)
                seeds = list(cfg.get("ongoing_seeds", [cfg.get("seed", 0)]))
                fit_probes(Dl, cond, probe_recipes(cfg, "probe_ongoing"), ONG_P, cfg.get("ongoing_rounds", 700), seeds,
                           fset_a, offis, pans, Ps, pseudo, probs_p, report)
                del Dl
                C.free()
            if cond in early_conds and not final_reuse:   # v8: only the early April windows are recomputed
                Dl = _frames(sets, pans, cond)
                ch_ = reuse["info"].get("chosen") or {}
                prm, rounds = (ONSET_P, cfg.get("onset_rounds", 500)) if cond == "queue_onset" else (ONG_P, cfg.get("ongoing_rounds", 700))
                seeds = list(cfg.get("onset_seeds" if cond == "queue_onset" else "ongoing_seeds", [cfg.get("seed", 0)]))
                variants = _variants(ch_.get("variants", ["all"]))
                ad_ = bool((reuse["info"].get("adapt") or {}).get("adopted", False))
                erep = early_step(Dl, pseudo, offis, pans, Ps, cond, prm, rounds, variants, seeds,
                                  list(cfg.get("late_seeds", seeds[:1])), cfg, ch_.get("hedge", fset_h), ch_.get("main", fset_a),
                                  ad_, early_a, early_h)
                report["early"][cond] = erep
                C.log(f"T2 early April windows {cond}: {erep}")
                del Dl
                for p in pans:
                    if cond in sets[p]:
                        sets[p][cond] = (sets[p][cond][0], sets[p][cond][1], None)
                C.free()
            continue
        Dl = _frames(sets, pans, cond)
        if not Dl:
            C.log(f"WARNING T2: no training windows for {cond}")
            continue
        C.log(f"T2 {cond}: {sum(len(f) for f in Dl):,} training rows")
        prm, rounds = (ONSET_P, cfg.get("onset_rounds", 500)) if cond == "queue_onset" else (ONG_P, cfg.get("ongoing_rounds", 700))
        seeds = list(cfg.get("onset_seeds" if cond == "queue_onset" else "ongoing_seeds", [cfg.get("seed", 0)]))
        lseeds = list(cfg.get("late_seeds", seeds[:1]))
        has_march = any(cond in pseudo[p].get("validation", {}) for p in pans)
        # ---- 1. variants and feature groups: March pseudo-holdout (another draw) and train CV (the training draw)
        (fset_c, variants), erep = choose_config(Dl, cond, pseudo, pans, Ps, prm, rounds, seeds, cfg, sets)
        report["experiments"][cond] = erep
        vn = [v for v, _ in variants]
        fa, fh = _fset_with(fset_a, fset_c), _fset_with(fset_h, fset_c)   # e.g. v2n+rd chosen -> main v2+rd, hedge v2n+rd
        report["chosen"][cond] = dict(variants=vn, main=fa, hedge=fh)
        C.log(f"T2 {cond}: chosen {fset_c} {vn} | {json.dumps(erep.get('candidates', {}), default=str)[:900]}")
        if cfg.get("cv", False):
            report["cv"][cond] = {fs: cv_score(Dl, sets, pans, Ps, cond, prm, rounds, variants, fs) for fs in (fa, fh)}
        # The choice is taken on March pseudo windows of the whole month, i.e. on data later than the scored March
        # origins: it shapes the April rows only. The March rows of both files keep the v4 recipe (MARCH_ROWS: no March
        # forecast depends on March data after its origin, and the v4 public scores reproduce exactly).
        mr_f, mr_v = (cfg.get("march_rows") or {}).get(cond) or MARCH_ROWS[cond]
        report["chosen"][cond]["march_rows"] = dict(main=_fset_with(fset_a, mr_f), hedge=_fset_with(fset_h, mr_f), variants=mr_v)

        def fit_both_months(fset, fset_march, probs, tag):
            ms = fit_set(Dl, cond, prm, rounds, variants, seeds, fset=fset)
            predict_official(ms, offis, pans, Ps, cond, probs, ("private",))
            mm = _models_for(ms, variants, seeds, mr_v) if fset == fset_march else None
            if mm is None:
                mm = fit_set(Dl, cond, prm, rounds, _variants(mr_v), seeds, fset=fset_march)
            predict_official(mm, offis, pans, Ps, cond, probs, ("validation",))
            if has_march:   # March pseudo-holdout of the April-row models (and of the March-row models when they differ)
                rp = report["pseudo"].setdefault(cond, {})
                R_ = pseudo_scores(ms, pseudo, pans, Ps, "validation", cond)
                rp[tag] = pseudo_report(R_)
                report["diag"]["pseudo"].setdefault(cond, {})[tag] = window_diag(R_, cond)
                if fset != fset_march or list(mr_v) != vn:
                    rp[f"{tag}_march_rows"] = pseudo_report(pseudo_scores(mm, pseudo, pans, Ps, "validation", cond))
            return ms
        # ---- 2. hedge (H): train only, no recent-days features, both months
        models = fit_both_months(fh, _fset_with(fset_h, mr_f), probs_h, "hedge")
        # ---- 2b. late April windows: same-month evidence, gated on March (the hedge's feature set decides for both)
        lrep = None
        n_late = sum(n_official(offis, pans, cond, "private", lo, hi) for lo, hi in late_blocks(cfg, "private"))
        if cfg.get("late_adapt", False) and cond in cfg.get("late_conds", list(CONDS)) and has_march and n_late == 0:
            report["late"][cond] = dict(adopted=False, reason="no official April window of this condition in the late blocks")
        elif cfg.get("late_adapt", False) and cond in cfg.get("late_conds", list(CONDS)) and has_march:
            lrep = late_gate(Dl, models[:: len(seeds)], pseudo, pans, Ps, cond, prm, rounds, variants, cfg, fh,
                             seed=seeds[0])
            lrep["official_late_windows"] = n_late
            report["late"][cond] = lrep
            C.log(f"T2 late-window gate {cond}: {lrep}")
            if lrep["adopted"]:
                lrep["hedge"] = late_fit_predict(Dl, pseudo, offis, pans, Ps, cond, prm, rounds, variants, lseeds, cfg,
                                                 fh, False, late_h, "hedge")
        del models
        C.free()
        # ---- 3. main (A): the 0.86840 features (+ the chosen groups); standard models (March rows: the v4 recipe;
        #      April when not adapted)
        models = fit_both_months(fa, _fset_with(fset_a, mr_f), probs_a, "main")
        # ---- 3b. probes (v6, optional): the main file with the March onset rows from a recipe fixed in advance
        #      (`probe_onset`), train-only models - public reads of those recipes on the scored March windows. A probe
        #      whose models equal ones already fitted reloads them from the checkpoints.
        if cond == "queue_onset":
            fit_probes(Dl, cond, probe_recipes(cfg), prm, rounds, seeds, fset_a, offis, pans, Ps, pseudo, probs_p, report)
        elif cond == "queue_ongoing" and probe_recipes(cfg, "probe_ongoing"):
            fit_probes(Dl, cond, probe_recipes(cfg, "probe_ongoing"), prm, rounds, seeds, fset_a, offis, pans, Ps, pseudo,
                       probs_p, report)
        if has_march:
            if cfg.get("pseudo_april", False):  # diagnostic only, never used to choose anything
                report["pseudo"][cond]["main_april"] = pseudo_report(pseudo_scores(models, pseudo, pans, Ps, "private", cond))
            C.log(f"T2 pseudo-holdout {cond}: {json.dumps(report['pseudo'][cond], default=str)[:600]}")
        # ---- 4. main (A), April: refit on train + March when the gate on March passes
        adapted = False
        if cfg.get("april_adapt", True) and has_march:
            rep_ad, ad = april_adapt(Dl, models, pseudo, pans, Ps, cond, prm, rounds, variants, seeds, cfg, fa)
            report["adapt"][cond] = rep_ad
            C.log(f"T2 April adaptation {cond}: {rep_ad}")
            if ad is not None:
                predict_official(ad, offis, pans, Ps, cond, apr_a, ("private",))
                adapted = True
            del ad
        # ---- 4b. main (A), late April windows: + April's own earlier pseudo windows (+ March when adapted)
        if lrep is not None and lrep["adopted"]:
            lrep["main"] = late_fit_predict(Dl, pseudo, offis, pans, Ps, cond, prm, rounds, variants, lseeds, cfg,
                                            fa, adapted, late_a, "main")
            C.log(f"T2 late windows {cond}: hedge {lrep.get('hedge')} | main {lrep.get('main')}")
        # ---- 4c. v8: early April windows (same-month evidence from the first days of April)
        if cond in early_conds and has_march:
            del models
            models = None
            C.free()
            erep = early_step(Dl, pseudo, offis, pans, Ps, cond, prm, rounds, variants, seeds, lseeds, cfg, fh, fa, adapted,
                              early_a, early_h)
            report["early"][cond] = erep
            C.log(f"T2 early April windows {cond}: {erep}")
        del Dl, models
        for p in pans:  # the training tables of this condition are not needed any more
            if cond in sets[p]:
                sets[p][cond] = (sets[p][cond][0], sets[p][cond][1], None)
        C.free()
    # ---- write the template rows
    (C.WORK / "pred").mkdir(parents=True, exist_ok=True)
    for f in [C.WORK / "pred" / "task2_std.csv"] + sorted((C.WORK / "pred").glob("task2_probe*.csv")):   # earlier runs
        if f.exists():
            f.unlink()
    og_ids = frozenset(w for p in pans for split, (wi, _, _) in offis[p].items() if split in C.TEST_SPLITS
                       for w in wi.window_id[wi.condition == "queue_ongoing"])
    ea_ids = frozenset(w for p in early_a for sp in early_a[p] for w in early_a[p][sp])
    eh_ids = frozenset(w for p in early_h for sp in early_h[p] for w in early_h[p][sp])
    quiet_a = (og_ids - ea_ids) if reuse is not None else frozenset()
    quiet_h = (og_ids - eh_ids) if reuse is not None else frozenset()

    def finish(frame, which, keep=frozenset()):
        """The ongoing rows of the previous run where this run did not predict them: its base rows except the
        recomputed early windows (v7/v8), or its final rows, early windows included (v9). `keep`: windows this file
        takes from its own probabilities (the ongoing probes' March windows)."""
        if reuse is None:
            return frame
        if final_reuse:
            return _overlay(frame, reuse["final"][which], og_ids - keep)
        return _overlay(frame, reuse[which], og_ids - (ea_ids if which == "main" else eh_ids) - keep)

    out = finish(write_rows([early_a, late_a, apr_a, probs_a], offis, pans, Ps, quiet_a), "main")
    hedge = finish(write_rows([early_h, late_h, probs_h], offis, pans, Ps, quiet_h), "hedge")
    # v8: the ongoing rows without the early-window overlay - what a later run reuses and overlays its own early windows on
    key = ["window_id", "timestamp", "link_id"]
    og_rows = out[out.window_id.isin(og_ids)][key].reset_index(drop=True)
    if reuse is not None:
        k_ = pd.MultiIndex.from_frame(og_rows)
        base_m, base_h = reuse["main"].reindex(k_).to_numpy(), reuse["hedge"].reindex(k_).to_numpy()
    else:
        bm = write_rows([late_a, apr_a, probs_a], offis, pans, Ps).set_index(key).queue_pred
        bh = write_rows([late_h, probs_h], offis, pans, Ps).set_index(key).queue_pred
        k_ = pd.MultiIndex.from_frame(og_rows)
        base_m, base_h = bm.reindex(k_).to_numpy(), bh.reindex(k_).to_numpy()
    if np.isnan(base_m.astype(float)).any() or np.isnan(base_h.astype(float)).any():
        raise RuntimeError("T2: base ongoing rows incomplete")
    og_rows["main"] = base_m.astype(np.int8); og_rows["hedge"] = base_h.astype(np.int8)
    og_rows.to_csv(C.WORK / "pred" / BASE_FILE, index=False)
    hedge.to_csv(C.WORK / "pred" / "task2_hedge.csv", index=False)
    out.to_csv(C.WORK / "pred" / "task2.csv", index=False)
    a = out.set_index(key).queue_pred; b = hedge.set_index(key).queue_pred.reindex(a.index)
    mar = a.index.get_level_values(0).str.contains("_validation_")
    report["files"] = {"task2.csv": True, "task2_hedge.csv": True, BASE_FILE: True}
    if prev is not None:   # v8: what changed against the files this run replaced (the public score reads the March part)
        report["vs_previous"] = dict(previous=prev["info"])
        for nm, new_ in (("main", a), ("hedge", b)):
            old_ = prev[nm].reindex(a.index)
            dif = (old_ != new_) | old_.isna()
            report["vs_previous"][nm] = {m: dict(cells_differ=int(dif[sel].sum()),
                                                 windows_differ=int(dif[sel].groupby(level=0).any().sum()))
                                         for m, sel in (("March", mar), ("April", ~mar))}
    for pname, pp in probs_p.items():   # probes: main with a probe recipe's March rows of one condition (April = main)
        own = frozenset(w for p in pp for sp in pp[p] for w in pp[p][sp])
        probe = finish(write_rows([pp, early_a, late_a, apr_a, probs_a], offis, pans, Ps, quiet_a), "main", own & og_ids)
        fn = f"task2_{pname}.csv"
        probe.to_csv(C.WORK / "pred" / fn, index=False)
        c = probe.set_index(key).queue_pred.reindex(a.index)
        report["files"][fn] = True
        report["probes"][pname].update(file=fn, vs_main=dict(
            March_differ=int((c[mar] != a[mar]).sum()), March_windows_differ=int((c[mar] != a[mar]).groupby(level=0).any().sum()),
            April_differ=int((c[~mar] != a[~mar]).sum())))
    report["agreement"] = {m: dict(cells_main=int(a[sel].sum()), cells_hedge=int(b[sel].sum()),
                                   differ=int((a[sel] != b[sel]).sum()),
                                   windows_differ=int((a[sel] != b[sel]).groupby(level=0).any().sum()))
                           for m, sel in (("March", mar), ("April", ~mar))}
    report["rows"] = int(len(out)); report["share_pred"] = float(out.queue_pred.mean())
    report["minutes"] = round((time.time() - t0) / 60, 1)
    report["files_md5"] = {fn: _md5(C.WORK / "pred" / fn) for fn in report["files"]}
    tmp = C.WORK / "pred" / "task2_report.json.part"
    json.dump(report, open(tmp, "w"), indent=1, default=str)
    tmp.replace(C.WORK / "pred" / "task2_report.json")   # written last and atomically: it vouches for the files
    json.dump(report["cv"], open(C.WORK / "pred" / "task2_cv.json", "w"), default=str)
    C.log(f"T2 done: {len(out):,} rows, share predicted {out.queue_pred.mean():.4f}, files main + hedge, "
          f"agreement {report['agreement']}")
    return out, report


def official_window_calendar(offis, pans):
    """Where the scored windows sit (dates, weekday / weekend, conditions), per split - metadata of the task only."""
    cal = C.calendar()
    rows = []
    for p in pans:
        for split, (wi, _, _) in offis[p].items():
            d = (wi["T"].to_numpy() // C.SLOTS).astype(np.int64)
            for dd, cnd in zip(d, wi.condition):
                rows.append(dict(split=split, date=cal.date.iat[dd], weekday=int(cal.weekday.iat[dd]), condition=cnd))
    if not rows:
        return {}
    R = pd.DataFrame(rows)
    out = {}
    for split, g in R.groupby("split"):
        out[split] = dict(windows=int(len(g)), first=str(g.date.min()), last=str(g.date.max()),
                          weekend_share=round(float((g.weekday >= 5).mean()), 3),
                          per_date=g.groupby("date").size().to_dict(),
                          per_date_condition={f"{d}|{c.split('_')[1]}": int(n) for (d, c), n in g.groupby(["date", "condition"]).size().items()},
                          weekend_share_by_condition={c.split("_")[1]: round(float((s.weekday >= 5).mean()), 3)
                                                      for c, s in g.groupby("condition")})
    return out


def shift_stats(P: P2, nseg: int = 10) -> dict:
    """v5 diagnostics: train vs March, both read from the released masked view (the same masking, so comparable).
    Where and how much the corridor queues (weekday 06-20 h; per tenth of the corridor in link order), how big the
    congestion gets, flow levels (night / AM / PM, weekdays), on-ramp demand, night speed and a capacity proxy (99.5th
    percentile of the observed link flow). April is never looked at. Nothing here is read back by the pipeline."""
    hrs = ((np.arange(P.NT) % C.SLOTS) / 12.0).astype(np.float32)
    wkd = C.weekday_of_slot()[: P.NT] < 5
    seg = np.minimum((np.arange(P.L) * nseg) // max(P.L, 1), nseg - 1)
    res, q995, lshare = {}, {}, {}
    r4 = lambda x: round(float(x), 4)
    for split, tag in (("train", "train"), ("validation", "march")):
        if len(C.split_days(split)) == 0:
            continue
        rs = C.split_rows(split)
        v, q, el, ron = P.mv[rs], P.mq[rs], P.el[rs], P.ron[rs]
        h, w = hrs[rs], wkd[rs]
        okv = el & np.isfinite(v)
        qu = okv & (np.nan_to_num(v, nan=np.inf) <= P.vcut[None, :])
        day = (h >= 6) & (h < 20)
        r = {}
        for nm, m in (("weekday", day & w), ("weekend", day & ~w)):
            r[f"queue_share_{nm}"] = r4(qu[m].sum() / max(okv[m].sum(), 1))
        m = day & w
        qn, on = qu[m].sum(0).astype(np.float64), okv[m].sum(0).astype(np.float64)
        lshare[tag] = qn / np.maximum(on, 1)
        num = np.bincount(seg, weights=qn, minlength=nseg); den = np.bincount(seg, weights=on, minlength=nseg)
        r["seg_queue_share"] = [r4(a / max(b, 1)) for a, b in zip(num, den)]
        nq = qu[m].sum(1)
        nz = nq[nq > 0]
        r["queued_links_p50_p90"] = [r4(np.percentile(nz, 50)), r4(np.percentile(nz, 90))] if len(nz) else [0.0, 0.0]
        okq = el & np.isfinite(q)
        for nm, hm in (("night", (h >= 1) & (h < 5)), ("am", (h >= 6) & (h < 10)), ("pm", (h >= 15) & (h < 19))):
            mm = hm & w
            r[f"flow_{nm}"] = r4(np.where(okq[mm], q[mm], 0).sum() / max(okq[mm].sum(), 1))
            fr = np.isfinite(ron[mm])
            r[f"onramp_{nm}"] = r4(np.where(fr, ron[mm], 0).sum() / max(fr.sum(), 1))
        nv = okv & ((h >= 1) & (h < 5))[:, None]
        r["speed_night"] = r4(np.where(nv, v, 0).sum() / max(nv.sum(), 1))
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            q995[tag] = np.nanpercentile(np.where(okq, q, np.nan), 99.5, axis=0)
        res[tag] = r
        del okv, qu, okq
    if "train" in q995 and "march" in q995:
        with np.errstate(all="ignore"):
            rat = q995["march"] / q995["train"]
        fin = np.isfinite(rat)
        if fin.any():
            res["capacity_proxy_ratio_march_over_train"] = dict(p10=r4(np.percentile(rat[fin], 10)),
                                                                p50=r4(np.percentile(rat[fin], 50)),
                                                                p90=r4(np.percentile(rat[fin], 90)))
        top_t = np.argsort(-lshare["train"])[:5]; top_m = np.argsort(-lshare["march"])[:5]
        res["top_train_bottlenecks"] = [dict(link=int(l), share_train=r4(lshare["train"][l]),
                                             share_march=r4(lshare["march"][l]),
                                             q995_ratio=r4(rat[l]) if np.isfinite(rat[l]) else None) for l in top_t]
        res["top_march_bottlenecks"] = [dict(link=int(l), share_train=r4(lshare["train"][l]),
                                             share_march=r4(lshare["march"][l]),
                                             q995_ratio=r4(rat[l]) if np.isfinite(rat[l]) else None) for l in top_m]
    if getattr(P, "dprof", None) is not None:
        res["daily_demand"] = daily_demand_stats(P)
    try:
        res["ramp_coupling"] = ramp_coupling(P)
    except Exception as e:   # diagnostics never stop the pipeline
        res["ramp_coupling"] = {"error": repr(e)[:200]}
    return res


def ramp_coupling(P: P2) -> dict:
    """v7 diagnostics: do the released ramp flows add up to the mainline? On weekday 06-20 h slots where two
    consecutive links both flow freely, regress the mainline flow change (link l+1 minus l) on the on-ramp flow into
    l+1 minus the off-ramp flow out of l (ramp pairs only). A slope near 1 means the ramp records drive the mainline;
    the cell-transmission forecast does not rely on them (it uses flow differences), so this only says whether they
    could add information. Train and March separately."""
    hrs = (np.arange(P.NT) % C.SLOTS) / 12.0
    wkd = C.weekday_of_slot()[: P.NT] < 5
    pair = (np.asarray(P.n_on)[1:] > 0) | (np.asarray(P.n_off)[:-1] > 0)
    out = {}
    if P.L < 2 or not pair.any():
        return out
    for split, tag in (("train", "train"), ("validation", "march")):
        if len(C.split_days(split)) == 0:
            continue
        rs = C.split_rows(split)
        sel = (wkd[rs] & (hrs[rs] >= 6) & (hrs[rs] < 20))
        q = P.mq[rs][sel]; v = P.mv[rs][sel]
        ron = P.ramp_est("on")[rs][sel]; roff = P.ramp_est("off")[rs][sel]
        y = (q[:, 1:] - q[:, :-1])[:, pair]
        x = (ron[:, 1:] - roff[:, :-1])[:, pair]
        free = ((v[:, 1:] > 0.8 * P.vf[None, 1:]) & (v[:, :-1] > 0.8 * P.vf[None, :-1]))[:, pair]
        m = free & np.isfinite(x) & np.isfinite(y)
        if m.sum() < 1000:
            continue
        def fit(xa, ya):
            xc, yc = xa - xa.mean(), ya - ya.mean()
            b = float((xc * yc).sum() / max((xc * xc).sum(), 1e-9))
            return b, float(1.0 - ((yc - b * xc) ** 2).sum() / max((yc * yc).sum(), 1e-9))
        xs, ys = x[m].astype(np.float64), y[m].astype(np.float64)
        b, r2 = fit(xs, ys)
        # the other attachment convention: on-ramps of link l and off-ramps of link l+1 between the two detectors
        x2 = (ron[:, :-1] - roff[:, 1:])[:, pair]
        m2 = free & np.isfinite(x2) & np.isfinite(y)
        b2, r22 = fit(x2[m2].astype(np.float64), y[m2].astype(np.float64)) if m2.sum() >= 1000 else (np.nan, np.nan)
        out[tag] = dict(n=int(m.sum()), slope=round(b, 4), r2=round(r2, 4), mean_ramp_net=round(float(xs.mean()), 1),
                        mean_flow_change=round(float(ys.mean()), 1), alt_slope=round(b2, 4), alt_r2=round(r22, 4))
    return out


def daily_demand_stats(P: P2) -> dict:
    """v6 diagnostics: day-to-day variation of weekday 06-20 h demand against the train profile (on-ramp inflow per link
    with ramps, and the mainline), on train days (out-of-fold profile) and the month mean in March. Tells whether train
    alone contains the kind of demand change the test months bring (then the dr_* features can be learned from train)."""
    dp = P.dprof
    nd = P.NT // C.SLOTS
    days = np.arange(nd)
    wk = np.asarray(P.wd_day)[:nd] < 5
    split = C.split_of_day()[:nd]
    lo, hi = 71, 239          # within-day running sums: slots 72..239 = 06:00-19:55
    r4 = lambda x: None if x is None or not np.isfinite(x) else round(float(x), 4)
    out = {}
    for ch, floor in (("on", 300.0), ("q", 50_000.0)):
        cx = dp.cX[ch].reshape(nd, C.SLOTS, -1); cm = dp.cM[ch].reshape(nd, C.SLOTS, -1)
        xs = cx[:, hi] - cx[:, lo]; ms = cm[:, hi] - cm[:, lo]
        with np.errstate(all="ignore"):
            r = np.where(ms > floor, xs / ms, np.nan)                       # [days, L]
        tr = (split == "train") & wk; mar = (split == "validation") & wk
        cols = np.isfinite(r[tr]).sum(0) >= 20                            # links with data on most train weekdays
        if cols.sum() < 2:
            out[ch] = dict(links=int(cols.sum()))
            continue
        rt, rm = r[tr][:, cols], r[mar][:, cols]
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sd_link = np.nanstd(rt, 0)
            z = rt - np.nanmean(rt, 0)
            zz = np.where(np.isfinite(z), z, 0); ok = np.isfinite(z).astype(float)
            cov = zz.T @ zz / np.maximum(ok.T @ ok, 1)
            sd = np.sqrt(np.diag(cov)); corr = cov / np.maximum(np.outer(sd, sd), 1e-12)
            iu = np.triu_indices(len(sd), 1)
            panel_tr = xs[tr][:, cols].sum(1) / np.maximum(ms[tr][:, cols].sum(1), 1e-9)   # same cells by construction
            mean_m = np.nanmean(rm, 0)
            # persistent shifts inside train: spread of 4-week block means (comparable with a month-long shift)
            bm = pd.DataFrame(rt).groupby(days[tr] // 28).mean()
            bsd = bm.std(0, ddof=1).to_numpy() if len(bm) >= 3 else np.full(rt.shape[1], np.nan)
            out[ch] = dict(links=int(cols.sum()), train_daily_sd_median=r4(np.nanmedian(sd_link)),
                           train_daily_sd_p90=r4(np.nanpercentile(sd_link, 90)),
                           train_block4w_sd_median=r4(np.nanmedian(bsd)), train_block4w_sd_p90=r4(np.nanpercentile(bsd, 90)),
                           train_link_corr_mean=r4(np.nanmean(corr[iu])) if len(iu[0]) else None,
                           train_panel_daily_sd=r4(np.nanstd(panel_tr)),
                           march_mean_abs_dev_median=r4(np.nanmedian(np.abs(mean_m - 1))),
                           march_mean_abs_dev_p90=r4(np.nanpercentile(np.abs(mean_m - 1), 90)),
                           march_mean_min=r4(np.nanmin(mean_m)), march_mean_max=r4(np.nanmax(mean_m)))
    return out


def _merge(dst, src):
    for p, sp in src.items():
        for split, d in sp.items():
            dst.setdefault(p, {}).setdefault(split, {}).update(d)
