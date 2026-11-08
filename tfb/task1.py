"""Task 1 (state reconstruction) and Task 3 (physics, scored on the same numbers).

Pooled LightGBM residual models over all panels:
  * regular cells: targets with same-time evidence; dark cells: targets inside the Task 2 blackout rows
    (horizon + buffer, every link blank), which are trained on simulated blackouts;
  * targets: speed, per-lane flow and per-lane density (q/v), each as a residual on a space-time
    interpolation baseline;
  * training rows: train-split targets (truth = unmasked layer) + transductive rows from the validation and
    private months (observed cells hidden and re-predicted; Task 1 is an offline task) + simulated
    test-month blackouts;
  * post-processing aimed at Task 3: density reconciliation in dense traffic, then total-variation smoothing
    of the density inside runs of consecutive target cells (L1-optimal against model jitter).
Local scoring: exact S_state and the Task 3 conservation proxy S_LWR = 1 - sum|dN_sub - dN_obs| / sum|dN_obs|
(the organizer boundary flux is projected onto the released observations, so the proxy tracks the leaderboard).
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd

from . import common as C
from .data import load_cache, load_net, blackout_rows

CH = ("v", "q", "o")
HOLD_DAYS = int(os.environ.get("TFB_T1_HOLD_DAYS", "30"))


# ----------------------------------------------------------------------------- panel
class P1:
    """One panel's arrays in physical link order. q and truth flow are per lane."""

    def __init__(self, panel: str):
        self.panel = panel
        self.net = net = load_net(panel)
        c = load_cache(panel)
        self.L = net.L
        self.NT = c["v"].shape[0]
        ln = net.lanes.astype(np.float32)
        self.lanes = ln
        self.X = {"v": c["v"], "q": c["q"] / ln[None, :], "o": c["o"]}
        self.pct = c["pct"]; self.el = c["el"]; self.tgt = c["tgt"]; self.dayreg = c["dayreg"]
        self.tv = c["tv"]; self.tq = c["tq"] / ln[None, :]; self.to = c["to"]
        self.ron = c["ron"] / ln[None, :]; self.roff = c["roff"] / ln[None, :]
        self.rv = c["rv_on"] & c["rv_off"]
        self.stations = list(c["stations"])
        self.length = net.length.astype(np.float32); self.vf = net.vf.astype(np.float32)
        self.cap_l = (net.cap / net.lanes).astype(np.float32)
        self.kj_l = (net.kjam / net.lanes).astype(np.float32)
        self.kc_l = self.cap_l / np.maximum(self.vf, 1)
        self.w = self.cap_l / np.maximum(self.kj_l - self.kc_l, 1e-3)
        self.vcut = net.vcut.astype(np.float32)
        self.observable = net.observable
        self.wd = C.weekday_of_slot()[: self.NT]
        self.tod = (np.arange(self.NT) % C.SLOTS).astype(np.int16)
        self.key = self.wd.astype(np.int32) * C.SLOTS + self.tod
        self.day_split = C.split_of_day()
        self.dark = blackout_rows(c)
        del c
        self._profiles()
        self.refresh()

    def _profiles(self):
        self.H = {}; self.Hs = {}; self.Hn = {}
        for ch in CH:
            X = self.X[ch]
            ok = np.isfinite(X)
            s = np.zeros((7 * C.SLOTS, self.L)); n = np.zeros((7 * C.SLOTS, self.L))
            np.add.at(s, self.key, np.where(ok, X, 0)); np.add.at(n, self.key, ok)
            self.Hs[ch] = s.astype(np.float32); self.Hn[ch] = n.astype(np.float32)
            prof = (s / np.maximum(n, 1)).reshape(7, C.SLOTS, self.L)
            cnt = n.reshape(7, C.SLOTS, self.L)
            allp = s.reshape(7, C.SLOTS, self.L).sum(0) / np.maximum(cnt.sum(0), 1)
            prof = np.where(cnt > 0, prof, allp[None])
            prof = (np.roll(prof, 1, 1) + 2 * prof + np.roll(prof, -1, 1)) / 4
            self.H[ch] = prof.reshape(7 * C.SLOTS, self.L).astype(np.float32)

    def refresh(self):
        """Recompute observation masks / gap indices after the view changed."""
        ok = np.isfinite(self.X["v"]) & np.isfinite(self.X["q"])
        self.ok = ok
        self.fi = C.ffill_idx(ok); self.bi = C.bfill_idx(ok)
        dark = self.dark
        pos = np.zeros(self.NT, np.int16); rem = np.zeros(self.NT, np.int16)
        s, e = C.runs_1d(dark)
        for a, b in zip(s, e):
            pos[a:b] = np.arange(1, b - a + 1); rem[a:b] = np.arange(b - a, 0, -1)
        self.dark_pos, self.dark_rem = pos, rem

    # ------------------------------------------------------------------ blackouts
    def origins(self, days, spacing: int, max_per_day: int = 4, rng=None):
        """Queue-like origins on days with truth (train): a truly queued cell in rows T+1..T+6, history rows
        mostly observed; greedy in time with the given spacing (slots)."""
        q = np.nan_to_num(self.tv, nan=999.0) <= self.vcut[None, :]
        anyq = q.any(1)
        fut = np.zeros(self.NT, bool)
        for k in range(1, 7):
            fut[: self.NT - k] |= anyq[k:]
        out = []
        for d in days:
            a = d * C.SLOTS + 13; b = (d + 1) * C.SLOTS - 20
            if b <= a:
                continue
            cand = np.flatnonzero(fut[a:b] & ~self.dark[a:b]) + a
            last = -10 ** 9; n = 0
            for T in cand:
                if T - last >= spacing and not self.dark[T - 12:T + 19].any():
                    out.append(int(T)); last = T; n += 1
                    if n >= max_per_day:
                        break
        return np.array(out, np.int64)

    def test_origins(self, split: str, max_n: int = 60, spacing: int = 72):
        """Queue-like origins in a test month from the released view (observed queued cells in rows T+1..T+6),
        away from the real blackouts; used to simulate blackouts for transductive dark rows."""
        sl = C.split_rows(split)
        vq = np.nan_to_num(self.X["v"], nan=999.0) <= self.vcut[None, :]
        anyq = vq.any(1)
        out = []; last = -10 ** 9
        for T in range(sl.start + 13, sl.stop - 20):
            if T - last < spacing or (T % C.SLOTS) > C.SLOTS - 20:
                continue
            if anyq[T + 1:T + 7].any() and not self.dark[T - 13:T + 20].any():
                out.append(T); last = T
        out = np.array(out, np.int64)
        if len(out) > max_n:
            out = np.sort(np.random.default_rng(0).choice(out, max_n, replace=False))
        return out

    def apply_blackouts(self, T0s, restore_history: bool = True):
        """Blank rows T+1..T+18 on every link (as the release does around a Task 2 window); the 12 history
        rows are never Task 1 targets in the release, so their truth is put back where available."""
        for T in T0s:
            if restore_history:
                h = slice(T - 12, T)
                m = self.el[h] & ~np.isfinite(self.X["v"][h]) & np.isfinite(self.tv[h])
                self.X["v"][h] = np.where(m, self.tv[h], self.X["v"][h])
                self.X["q"][h] = np.where(m, self.tq[h], self.X["q"][h])
                self.X["o"][h] = np.where(m, self.to[h], self.X["o"][h])
                self.tgt[h] = 0  # history cells are never Task 1 targets in the release
            d = slice(T + 1, T + 19)
            for ch in CH:  # the release removes the eligible cells of the horizon + buffer rows
                self.X[ch][d] = np.where(self.el[d], np.nan, self.X[ch][d])
            self.dark[T + 1:T + 19] = True
        self.refresh()

    # ------------------------------------------------------------------ features
    def feats(self, ti: np.ndarray, lj: np.ndarray, own: dict | None = None) -> pd.DataFrame:
        """own: values of these cells that were inside the profile sums (transductive rows) -> removed."""
        ti = np.asarray(ti, np.int64); lj = np.asarray(lj, np.int64)
        T, L = self.NT, self.L
        X = self.X
        F = {}
        key = self.key[ti]
        for ch in CH:
            h = self.H[ch][key, lj]
            if own is not None and ch in own:
                x = own[ch]; m = np.isfinite(x)
                s_ = self.Hs[ch][key, lj]; n_ = self.Hn[ch][key, lj]
                p_old = s_ / np.maximum(n_, 1); p_new = (s_ - np.where(m, x, 0)) / np.maximum(n_ - m, 1)
                h = h + 0.5 * np.where(m & (n_ > 1), p_new - p_old, 0)  # centre weight of the tod smoothing
            F[f"h_{ch}"] = h.astype(np.float32)

        def get(ch, dt, dl):
            t2 = ti + dt; l2 = lj + dl
            valid = (t2 >= 0) & (t2 < T) & (l2 >= 0) & (l2 < L)
            v = X[ch][np.clip(t2, 0, T - 1), np.clip(l2, 0, L - 1)]
            return np.where(valid, v, np.nan).astype(np.float32)

        for dt in (-6, -3, -2, -1, 1, 2, 3, 6):
            for ch in ("v", "q"):
                F[f"o_{ch}_{dt}"] = get(ch, dt, 0)
        for dt in (-1, 1):
            F[f"o_o_{dt}"] = get("o", dt, 0)
            t2 = np.clip(ti + dt, 0, T - 1)
            F[f"pct_{dt}"] = self.pct[t2, lj].astype(np.float32)
        F["pct_0"] = self.pct[ti, lj].astype(np.float32)
        fi = np.where(ti - 1 >= 0, self.fi[np.clip(ti - 1, 0, T - 1), lj], -1)
        bi = np.where(ti + 1 < T, self.bi[np.clip(ti + 1, 0, T - 1), lj], T)
        gp = (ti - fi).astype(np.float32); gn = (bi - ti).astype(np.float32)
        gp[fi < 0] = np.nan; gn[bi >= T] = np.nan
        F["gap_prev"] = gp; F["gap_next"] = gn
        wgt = np.where(np.isfinite(gp) & np.isfinite(gn), gp / (gp + gn), np.nan)
        fic = np.clip(fi, 0, T - 1); bic = np.clip(bi, 0, T - 1)
        for ch in CH:
            pv = np.where(fi >= 0, X[ch][fic, lj], np.nan)
            nv = np.where(bi < T, X[ch][bic, lj], np.nan)
            li = pv + wgt * (nv - pv)
            li = np.where(np.isfinite(li), li, np.where(np.isfinite(pv), pv, nv))
            F[f"pv_{ch}"] = pv.astype(np.float32); F[f"nv_{ch}"] = nv.astype(np.float32); F[f"li_{ch}"] = li.astype(np.float32)
            if ch != "o":
                hp = np.where(fi >= 0, self.H[ch][self.key[fic], lj], np.nan)
                hn = np.where(bi < T, self.H[ch][self.key[bic], lj], np.nan)
                dev = np.where(np.isfinite(pv - hp) & np.isfinite(nv - hn), (1 - np.nan_to_num(wgt)) * (pv - hp) + np.nan_to_num(wgt) * (nv - hn),
                               np.where(np.isfinite(pv - hp), pv - hp, nv - hn))
                F[f"lidev_{ch}"] = (F[f"h_{ch}"] + dev).astype(np.float32)
        for dl in (-3, -2, -1, 1, 2, 3):
            for ch in CH:
                F[f"n{dl}_{ch}_0"] = get(ch, 0, dl)
            if abs(dl) <= 2:
                for dt in (-1, 1):
                    for ch in ("v", "q"):
                        F[f"n{dl}_{ch}_{dt}"] = get(ch, dt, dl)
            l2 = np.clip(lj + dl, 0, L - 1)
            for ch in ("v", "q"):
                F[f"n{dl}_{ch}_dev"] = F[f"n{dl}_{ch}_0"] - self.H[ch][key, l2]
        for ch in ("v", "q"):
            up = np.full(len(ti), np.nan, np.float32); dn = up.copy()
            upd = np.full(len(ti), np.nan, np.float32); dnd = upd.copy()
            for dl in range(1, 7):
                v = get(ch, 0, -dl); m = np.isnan(up) & np.isfinite(v); up[m] = v[m]; upd[m] = dl
                v = get(ch, 0, dl); m = np.isnan(dn) & np.isfinite(v); dn[m] = v[m]; dnd[m] = dl
            F[f"near_up_{ch}"] = up; F[f"near_dn_{ch}"] = dn
            if ch == "v":
                F["near_up_d"] = upd; F["near_dn_d"] = dnd
        for dl in (-2, -1, 1, 2):
            l2 = np.clip(lj + dl, 0, L - 1); inb = (lj + dl >= 0) & (lj + dl < L)
            f2 = self.fi[np.clip(ti - 1, 0, T - 1), l2]; b2 = self.bi[np.clip(ti + 1, 0, T - 1), l2]
            for ch in ("v", "q"):
                F[f"n{dl}_pv_{ch}"] = np.where(inb & (f2 >= 0), X[ch][np.clip(f2, 0, T - 1), l2], np.nan).astype(np.float32)
                F[f"n{dl}_nv_{ch}"] = np.where(inb & (b2 < T), X[ch][np.clip(b2, 0, T - 1), l2], np.nan).astype(np.float32)
        # fundamental-diagram features (per lane)
        w, kj, vf = self.w[lj], self.kj_l[lj], self.vf[lj]
        F["fd_w"] = w; F["fd_kj"] = kj; F["fd_kc"] = self.kc_l[lj]
        for src in ("li", "pv", "nv"):
            v = F[f"{src}_v"]; q = F[f"{src}_q"]
            F[f"fd_qc_{src}"] = np.where(v < 0.9 * vf, v * (w * kj / (v + w)), np.nan).astype(np.float32)
            F[f"fd_vc_{src}"] = (q / np.maximum(kj - q / w, 1e-2)).astype(np.float32)
            F[f"fd_k_{src}"] = (q / np.maximum(v, 1)).astype(np.float32)
        for dl in (-1, 1):
            v = F[f"n{dl}_v_0"]
            F[f"fd_qc_n{dl}"] = np.where(v < 0.9 * vf, v * w * kj / (v + w), np.nan).astype(np.float32)
            F[f"fd_k_n{dl}"] = (F[f"n{dl}_q_0"] / np.maximum(v, 1)).astype(np.float32)
        F["vr_li"] = (F["li_v"] / vf).astype(np.float32)
        # ramps (per lane of the mainline link); released inside blackouts too
        for nm, arr in (("on", self.ron), ("off", self.roff)):
            for dt in (-1, 0, 1):
                t2 = np.clip(ti + dt, 0, T - 1)
                F[f"r{nm}_{dt}"] = arr[t2, lj]
            for side, rng_ in (("up", range(-3, 0)), ("dn", range(1, 4))):
                acc = np.zeros(len(ti), np.float32); cnt = np.zeros(len(ti), np.int16)
                for dl in rng_:
                    l2 = lj + dl; inb = (l2 >= 0) & (l2 < L)
                    v = np.where(inb, arr[ti, np.clip(l2, 0, L - 1)], np.nan)
                    m = np.isfinite(v); acc[m] += v[m]; cnt += m
                F[f"r{nm}_{side}3"] = np.where(cnt > 0, acc, np.nan).astype(np.float32)
        # blackout context: values at the blackout's edges are pv / nv already
        F["dark_pos"] = self.dark_pos[ti].astype(np.float32); F["dark_rem"] = self.dark_rem[ti].astype(np.float32)
        F["is_dark"] = self.dark[ti].astype(np.float32)
        # static & calendar
        F["lanes"] = self.lanes[lj]; F["length"] = self.length[lj]; F["vf"] = vf; F["cap_l"] = self.cap_l[lj]
        F["relpos"] = (lj / max(L - 1, 1)).astype(np.float32)
        F["tod"] = self.tod[ti].astype(np.float32); F["wd"] = self.wd[ti].astype(np.float32)
        F["regime"] = self.dayreg[ti // C.SLOTS].astype(np.float32)
        F["panel_id"] = np.full(len(ti), C.panels().index(self.panel), np.float32)
        return pd.DataFrame({k: np.asarray(v, np.float32) for k, v in F.items()})


def base(F: pd.DataFrame, what: str) -> np.ndarray:
    if what == "k":
        return base(F, "q") / np.maximum(base(F, "v"), 1.0)
    return F[f"li_{what}"].fillna(F[f"h_{what}"]).to_numpy(np.float64)


NONFEAT = ("y_v", "y_q", "t", "j", "kind", "src", "day", "panel")


# ----------------------------------------------------------------------------- row sets
def _sample(idx_t, idx_l, n, rng):
    if len(idx_t) > n:
        s = rng.choice(len(idx_t), n, replace=False)
        return idx_t[s], idx_l[s]
    return idx_t, idx_l


def _pseudo_plan(P: P1, tgt, cfg: dict, rng, excl=None) -> list:
    """The transductive sample of every test month: (split, t, j, per_round, dark) where dark = (T0s, a, b) are the
    simulated test-month blackouts and the cells taken inside them (or None). Draws from `rng` in exactly the order the
    rows are built in (v8 draws them all before any feature is built; nothing in between uses the generator, so model
    sets of earlier versions get the same rows). `excl` [NT, L] bool: cells no transductive row may use (v8: the
    test-month holdout); None for the model sets of earlier versions."""
    plans = []
    for split in C.TEST_SPLITS:
        sl = C.split_rows(split)
        cand = P.ok.copy(); cand[: sl.start] = False; cand[sl.stop:] = False
        cand &= P.el & ~tgt & ~P.dark[:, None]
        if excl is not None:
            cand &= ~excl
        ct, cl = np.nonzero(cand)
        n_tot = min(len(ct), cfg.get("n_pseudo", 120_000))
        sel = rng.permutation(len(ct))[:n_tot]
        per_round = max(1, int(cfg.get("pseudo_frac", 0.05) * len(ct)))
        dark = None
        if cfg.get("pseudo_dark", True):
            T0s = P.test_origins(split, max_n=cfg.get("pseudo_dark_n", 40))
            if len(T0s):
                rr = np.concatenate([np.arange(T + 1, T + 19) for T in T0s])
                m = np.zeros(P.NT, bool); m[rr] = True; m &= ~P.dark
                sub = np.zeros_like(cand)
                sub[m] = np.isfinite(P.X["v"][m]) & np.isfinite(P.X["q"][m]) & P.el[m] & ~tgt[m]
                if excl is not None:
                    sub &= ~excl
                a, b = np.nonzero(sub)
                a, b = _sample(a, b, cfg.get("n_pseudo_dark", 60_000), rng)
                dark = (T0s, a, b)
        plans.append((split, ct[sel], cl[sel], per_round, dark))
    return plans


def _exclusion(panel: str, cfg: dict, shape) -> np.ndarray | None:
    """v8: the cells of the test-month holdout (`tm_cells` file) as a mask, when this model set must not train on them."""
    d = cfg.get("tm_exclude_dir")
    if not d:
        return None
    from pathlib import Path
    z = np.load(Path(d) / f"{panel}_tm_cells.npz")
    m = np.zeros(shape, bool)
    m[z["t"], z["j"]] = True
    return m


def build_rows(panel: str, cfg: dict, out_dir, plan_only: bool = False):
    """Training rows (train targets, simulated train blackouts, transductive test-month rows), holdout rows
    (every target of the last HOLD_DAYS train days, with realistic blackouts) and the test target rows.
    plan_only (v8): return the transductive cells this configuration draws ({split: (t, j)} for regular rows and
    {split: (t, j)} for the simulated test-month blackout rows) without building any row."""
    rng = np.random.default_rng(cfg.get("seed", 0))
    P = P1(panel)
    tr_days = C.split_days("train")
    hold_days = tr_days[-HOLD_DAYS:] if cfg.get("holdout", True) else np.array([], int)
    fit_days = np.setdiff1d(tr_days, hold_days)
    # simulated blackouts on train: dense on fit days (dark-model data), realistic on holdout days
    o_fit = P.origins(fit_days, spacing=cfg.get("dark_spacing", 72), max_per_day=cfg.get("dark_per_day", 2))
    o_hold = P.origins(hold_days, spacing=72, max_per_day=1)
    if len(o_hold) > max(1, len(hold_days) // 3):
        o_hold = np.sort(rng.choice(o_hold, max(1, len(hold_days) // 3), replace=False))
    P.apply_blackouts(np.concatenate([o_fit, o_hold]))
    # a further model set (row_seed > 0) draws its own training sample; the holdout (days, blackouts, rows) and the
    # test rows stay exactly those of row_seed 0, so every model set is scored on the same holdout
    row_seed = int(cfg.get("row_seed", 0))
    if row_seed:
        rng = np.random.default_rng(10_000 + row_seed)
    ntr = (tr_days.max() + 1) * C.SLOTS
    day = np.arange(P.NT) // C.SLOTS
    is_hold = np.isin(day, hold_days)
    tgt = P.tgt > 0
    has_truth = np.isfinite(P.tv) & np.isfinite(P.tq)
    rows = {}
    # regular / dark train rows; holdout = every target of the last HOLD_DAYS train days (scored separately),
    # plus an early-stopping subsample drawn at the same density as the regular rows
    tt, ll = np.nonzero(tgt[:ntr] & has_truth[:ntr])
    dk = P.dark[tt]; hd = is_hold[tt]
    n_reg = cfg.get("n_reg", 250_000)
    cand_reg = int((~dk & ~hd).sum())
    rate = min(1.0, n_reg / max(cand_reg, 1))
    rt, rl = _sample(tt[~dk & ~hd], ll[~dk & ~hd], n_reg, rng)
    dt_, dl_ = _sample(tt[dk & ~hd], ll[dk & ~hd], cfg.get("n_dark", 120_000), rng)
    ht, hl = tt[hd], ll[hd]
    n_es_reg = int(rate * (~dk[hd]).sum())
    est_r, esl_r = _sample(ht[~dk[hd]], hl[~dk[hd]], n_es_reg, rng)
    est_d, esl_d = ht[dk[hd]], hl[dk[hd]]
    # transductive sample of the test months (drawn now: the feature building below never uses the generator)
    plans = _pseudo_plan(P, tgt, cfg, rng, _exclusion(panel, cfg, tgt.shape)) if cfg.get("pseudo", True) else []
    if plan_only:
        return dict(reg={s: (t, j) for s, t, j, _, _ in plans},
                    dark={s: (dk_[1], dk_[2]) for s, _, _, _, dk_ in plans if dk_ is not None})
    parts = []

    def add(a, b, kind_arr, src, sink):
        for i in range(0, len(a), 200_000):
            t2, l2 = a[i:i + 200_000], b[i:i + 200_000]
            F = P.feats(t2, l2)
            F["y_v"] = P.tv[t2, l2]; F["y_q"] = P.tq[t2, l2]
            F["t"] = t2.astype(np.float32); F["j"] = l2.astype(np.float32)
            F["kind"] = P.dark[t2].astype(np.float32) if kind_arr is None else np.float32(kind_arr)
            F["src"] = np.float32(src)
            sink.append(F)

    add(rt, rl, 0.0, 0, parts); add(dt_, dl_, 1.0, 0, parts)
    add(np.concatenate([est_r, est_d]), np.concatenate([esl_r, esl_d]), None, 2, parts)
    hold_parts = []
    if not row_seed:   # the holdout rows of a further model set are those of row_seed 0
        add(ht, hl, None, 2, hold_parts)
    if hold_parts:
        Hd = pd.concat(hold_parts, ignore_index=True)
        Hd[np.isfinite(Hd.y_v) & np.isfinite(Hd.y_q)].to_parquet(out_dir / f"{panel}_hold.parquet", index=False)
        del Hd, hold_parts
    # transductive rows from the test months (observed cells hidden in rounds)
    for split, ct_s, cl_s, per_round, dark in plans:
        for i in range(0, len(ct_s), per_round):
            t2, l2 = ct_s[i:i + per_round], cl_s[i:i + per_round]
            keep = {ch: P.X[ch][t2, l2].copy() for ch in CH}
            for ch in CH:
                P.X[ch][t2, l2] = np.nan
            P.refresh()
            F = P.feats(t2, l2, own=keep)
            F["y_v"] = keep["v"]; F["y_q"] = keep["q"]
            F["t"] = t2.astype(np.float32); F["j"] = l2.astype(np.float32); F["kind"] = 0.0; F["src"] = 1.0
            parts.append(F)
            for ch in CH:
                P.X[ch][t2, l2] = keep[ch]
        P.refresh()
        # simulated test-month blackouts -> transductive dark rows
        if dark is not None:
            T0s, a, b = dark
            saved = {ch: P.X[ch].copy() for ch in CH}; dsave = P.dark.copy()
            P.apply_blackouts(T0s, restore_history=False)
            if len(a):
                F = P.feats(a, b, own={ch: saved[ch][a, b] for ch in CH})
                F["y_v"] = saved["v"][a, b]; F["y_q"] = saved["q"][a, b]
                F["t"] = a.astype(np.float32); F["j"] = b.astype(np.float32); F["kind"] = 1.0; F["src"] = 1.0
                parts.append(F)
            for ch in CH:
                P.X[ch] = saved[ch]
            P.dark = dsave; P.refresh()
    D = pd.concat(parts, ignore_index=True)
    D = D[np.isfinite(D.y_v) & np.isfinite(D.y_q)]
    D.to_parquet(out_dir / f"{panel}_train.parquet", index=False)
    if row_seed:   # training rows only: holdout and test rows come from row_seed 0
        info = dict(panel=panel, row_seed=row_seed, n_train=int((D.src == 0).sum()), n_pseudo=int((D.src == 1).sum()),
                    n_dark=int(((D.kind == 1) & (D.src != 2)).sum()))
        json.dump(info, open(out_dir / f"{panel}_info.json", "w"))
        C.log(f"T1 rows {panel} (row seed {row_seed}): {info['n_train']:,} train, {info['n_pseudo']:,} pseudo, {info['n_dark']:,} dark")
        return info
    # test targets (validation + private), on the untouched released view
    P = P1(panel)
    a0 = C.split_rows("validation").start
    tt, ll = np.nonzero(P.tgt[a0:] > 0); tt = tt + a0
    parts = []
    for i in range(0, len(tt), 250_000):
        t2, l2 = tt[i:i + 250_000], ll[i:i + 250_000]
        F = P.feats(t2, l2)
        F["t"] = t2.astype(np.float32); F["j"] = l2.astype(np.float32); F["kind"] = P.dark[t2].astype(np.float32)
        parts.append(F)
    pd.concat(parts, ignore_index=True).to_parquet(out_dir / f"{panel}_test.parquet", index=False)
    info = dict(panel=panel, n_train=int((D.src == 0).sum()), n_pseudo=int((D.src == 1).sum()), n_hold=int(len(ht)),
                n_dark=int(((D.kind == 1) & (D.src != 2)).sum()), n_test=int(len(tt)), n_test_dark=int(P.dark[tt].sum()),
                hold_days=[int(x) for x in hold_days], hold_origins=[int(x) for x in o_hold], fit_origins=[int(x) for x in o_fit])
    json.dump(info, open(out_dir / f"{panel}_info.json", "w"))
    C.log(f"T1 rows {panel}: {info['n_train']:,} train, {info['n_pseudo']:,} pseudo, {info['n_hold']:,} holdout, "
          f"{info['n_dark']:,} dark, {info['n_test']:,} test ({info['n_test_dark']:,} dark)")
    return info


# ----------------------------------------------------------------------------- models
PARAMS = dict(objective="regression", metric="l2", learning_rate=0.06, num_leaves=255, min_data_in_leaf=80,
              feature_fraction=0.6, bagging_fraction=0.7, bagging_freq=1, lambda_l2=2.0, max_bin=127,
              verbose=-1)
DARK_PARAMS = dict(PARAMS, num_leaves=63, min_data_in_leaf=150, learning_rate=0.05)
TARGETS = ("v", "q", "k")


def _label(D: pd.DataFrame, what: str) -> np.ndarray:
    if what == "k":
        return D.y_q.to_numpy(np.float64) / np.maximum(D.y_v.to_numpy(np.float64), 1.0)
    return D[f"y_{what}"].to_numpy(np.float64)


def load_rows(rows_dir, kind: str, which: str):
    parts = []
    for p in C.panels():
        f = rows_dir / f"{p}_{which}.parquet"
        D = pd.read_parquet(f)
        if which == "train":
            D = D[D.kind == (1.0 if kind == "dark" else 0.0)] if kind != "all" else D
        parts.append(D)
    return pd.concat(parts, ignore_index=True)


def train_models(rows_dir, model_dir, cfg: dict, holdout: bool = True, rounds: dict | None = None) -> dict:
    import lightgbm as lgb
    model_dir.mkdir(parents=True, exist_ok=True)
    rep = {}
    seed = cfg.get("seed", 0)
    for kind in ("reg", "dark"):
        D = load_rows(rows_dir, kind, "train")
        feats = [c for c in D.columns if c not in NONFEAT]
        tr = (D.src != 2).to_numpy()
        va = (D.src == 2).to_numpy() & holdout
        if not holdout:
            tr = np.ones(len(D), bool)
        for what in TARGETS:
            name = f"{kind}_{what}"
            y = _label(D, what) - base(D, what)
            ok = np.isfinite(y)
            prm = dict(DARK_PARAMS if kind == "dark" else PARAMS, num_threads=C.THREADS, seed=seed,
                       bagging_seed=seed + 1, feature_fraction_seed=seed + 2, data_random_seed=seed + 3)
            prm.update(cfg.get("lgb_extra") or {})   # v8: e.g. deterministic training for the sets it adds
            if what == "k":
                prm.update(objective="huber", alpha=float(cfg.get("huber_k", 1.0)))
            w = (D.length * D.lanes).to_numpy() if what == "k" else None
            t0 = time.time()
            dtr = lgb.Dataset(D.loc[tr & ok, feats], y[tr & ok], weight=None if w is None else w[tr & ok],
                              categorical_feature=["panel_id"], free_raw_data=True)
            maxr = int(cfg.get("max_rounds", 4000))
            if holdout and (va & ok).sum() > 1000:
                dva = lgb.Dataset(D.loc[va & ok, feats], y[va & ok], weight=None if w is None else w[va & ok], reference=dtr)
                m = lgb.train(prm, dtr, maxr, valid_sets=[dva],
                              callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(0)])
                rep[name] = dict(best_iter=int(m.best_iteration or maxr), rmse=float(m.best_score["valid_0"].get("l2", np.nan) ** 0.5))
            else:
                if rounds and name in rounds:
                    n = int(rounds[name].get("best_iter", maxr) * 1.1)
                else:
                    n = int(cfg.get("rounds", min(maxr, 1500 if kind == "reg" else 800)))
                m = lgb.train(prm, dtr, max(50, n))
                rep[name] = dict(best_iter=n)
            _save_model(m, model_dir / f"{name}.txt")
            C.log(f"T1 model {name}: {rep[name]} ({time.time() - t0:.0f}s, {int((tr & ok).sum()):,} rows)")
            del dtr
        del D; C.free()
    json.dump(dict(row_seed=int(cfg.get("row_seed", 0)), seed=int(seed), max_rounds=int(cfg.get("max_rounds", 4000)),
                   rows_sig=rows_sig(cfg), lgb_extra=cfg.get("lgb_extra") or {}),
              open(model_dir / "meta.json", "w"))
    json.dump(rep, open(model_dir / "report.json", "w"), indent=1)   # written last: marks the set as complete
    return rep


ROWS_SIG_KEYS = ("n_reg", "n_dark", "n_pseudo", "n_pseudo_dark", "pseudo_frac", "pseudo_dark_n", "pseudo", "pseudo_dark",
                 "dark_spacing", "dark_per_day")


def rows_sig(cfg: dict) -> dict:
    """The row settings a model set was trained with (v8; sets of earlier versions carry none and are taken as they are)."""
    return dict({k: cfg.get(k) for k in ROWS_SIG_KEYS}, tm_exclude=tm_sig(cfg) if cfg.get("tm_exclude_dir") else False,
                lgb_extra=cfg.get("lgb_extra") or {})


def tm_sig(cfg: dict) -> dict:
    """v8: which test-month holdout cells a configuration draws (and a model set excludes)."""
    return dict(n=int(cfg.get("tm_n", 20_000)), seed=int(cfg.get("tm_seed", 7)),
                frac=float(cfg.get("pseudo_frac", 0.05)))


MODEL_FILES = tuple(f"{k}_{w}.txt" for k in ("reg", "dark") for w in TARGETS)


def _save_model(m, path):
    """Model text via a temporary file, then its md5 (a truncated copy is never mistaken for a model)."""
    import hashlib
    tmp = path.with_suffix(".tmp")
    m.save_model(str(tmp))
    digest = hashlib.md5(tmp.read_bytes()).hexdigest()
    tmp.replace(path)
    path.with_suffix(".md5").write_text(digest)


def _model_file_ok(path) -> bool:
    """A LightGBM model file that is complete: its md5 sidecar matches when there is one (sets written by this
    version), otherwise the file ends with the parameter block LightGBM writes last (sets of earlier versions).
    LightGBM aborts the whole process on a truncated model file instead of raising, so this is checked first."""
    import hashlib
    if not path.exists() or path.stat().st_size < 1000:
        return False
    side = path.with_suffix(".md5")
    if side.exists():
        return hashlib.md5(path.read_bytes()).hexdigest() == side.read_text().strip()
    with open(path, "rb") as f:
        f.seek(max(0, path.stat().st_size - 65536))
        tail = f.read()
    return b"end of parameters" in tail and b"pandas_categorical" in tail


def set_complete(model_dir, row_seed=None, sig=None) -> bool:
    """A model set can be reused: report.json present, every model file complete and (when recorded) trained on the
    expected row sample and (v8, when recorded) with the expected row settings."""
    if not (model_dir / "report.json").exists():
        return False
    if not all(_model_file_ok(model_dir / f) for f in MODEL_FILES):
        C.log(f"T1 model set {model_dir.name}: incomplete or damaged model file; retraining it")
        return False
    meta = model_dir / "meta.json"
    mt = json.load(open(meta)) if meta.exists() else {}
    if row_seed is not None and meta.exists() and int(mt.get("row_seed", 0)) != int(row_seed):
        C.log(f"T1 model set {model_dir.name}: trained on another row sample; retraining it")
        return False
    if sig is not None and "rows_sig" in mt and json.dumps(mt["rows_sig"], sort_keys=True, default=str) != json.dumps(sig, sort_keys=True, default=str):
        C.log(f"T1 model set {model_dir.name}: trained with other row settings; retraining it")
        return False
    return True


_BOOSTERS: dict = {}


def _booster(path):
    """Boosters are parsed once per session (a 2500-tree model file takes seconds to parse, and every panel and every
    model set used to parse all six again); the key includes size and mtime, so a retrained file is reloaded."""
    import lightgbm as lgb
    st = path.stat()
    key = (str(path), st.st_size, st.st_mtime_ns)
    if key not in _BOOSTERS:
        for k in [k for k in _BOOSTERS if k[0] == str(path)]:
            del _BOOSTERS[k]
        _BOOSTERS[key] = lgb.Booster(model_file=str(path))
    return _BOOSTERS[key]


def predict_frame(D: pd.DataFrame, model_dir) -> pd.DataFrame:
    out = pd.DataFrame({"t": D.t.to_numpy(np.int64), "j": D.j.to_numpy(np.int64), "kind": D.kind.to_numpy()})
    if "panel_id" in D:
        out["panel_id"] = D.panel_id.to_numpy()
    for what in TARGETS:
        out[what] = np.nan
    for kind, kv in (("reg", 0.0), ("dark", 1.0)):
        m = (D.kind == kv).to_numpy()
        if not m.any():
            continue
        Dm = D.loc[m]
        mats = {}
        for what in TARGETS:
            bst = _booster(model_dir / f"{kind}_{what}.txt")
            fn = tuple(bst.feature_name())
            if fn not in mats:   # the three targets share one feature list: one float32 matrix per kind
                mats[fn] = np.ascontiguousarray(Dm[list(fn)].to_numpy(np.float32))
            out.loc[m, what] = bst.predict(mats[fn], num_threads=C.THREADS) + base(Dm, what)
        del mats, Dm
    return out


# ----------------------------------------------------------------------------- v8: members (LightGBM sets, CatBoost)
CB_FILES = tuple(f"{k}_{w}.cbm" for k in ("reg", "dark") for w in TARGETS)
_CBMODELS: dict = {}


def _cb_model(path):
    from catboost import CatBoostRegressor
    st = path.stat()
    key = (str(path), st.st_size, st.st_mtime_ns)
    if key not in _CBMODELS:
        for k in [k for k in _CBMODELS if k[0] == str(path)]:
            del _CBMODELS[k]
        m = CatBoostRegressor()
        m.load_model(str(path))
        _CBMODELS[key] = m
    return _CBMODELS[key]


def predict_frame_cb(D: pd.DataFrame, model_dir) -> pd.DataFrame:
    """predict_frame for the CatBoost member."""
    out = pd.DataFrame({"t": D.t.to_numpy(np.int64), "j": D.j.to_numpy(np.int64), "kind": D.kind.to_numpy()})
    if "panel_id" in D:
        out["panel_id"] = D.panel_id.to_numpy()
    for what in TARGETS:
        out[what] = np.nan
    for kind, kv in (("reg", 0.0), ("dark", 1.0)):
        m = (D.kind == kv).to_numpy()
        if not m.any():
            continue
        Dm = D.loc[m]
        X, xf = None, None
        for what in TARGETS:
            mdl = _cb_model(model_dir / f"{kind}_{what}.cbm")
            fn = list(mdl.feature_names_)
            if xf != fn:
                X = np.ascontiguousarray(Dm[fn].to_numpy(np.float32)); xf = fn
            out.loc[m, what] = mdl.predict(X, thread_count=C.THREADS) + base(Dm, what)
        del X, Dm
    return out


def as_members(model_dirs) -> list:
    """[(kind, path, weight)] with weights summing to 1, from model-set directories (LightGBM, equal weights) and/or
    member triples (kind 'lgb' or 'cb', path, weight)."""
    out = [tuple(m) if isinstance(m, (tuple, list)) else ("lgb", m, 1.0) for m in model_dirs]
    s = float(sum(w for _, _, w in out))
    return [(k, p, float(w) / s) for k, p, w in out if w > 0]


def predict_member(D: pd.DataFrame, mem) -> pd.DataFrame:
    kind, path, _ = mem
    return predict_frame_cb(D, path) if kind == "cb" else predict_frame(D, path)


def blend_preds(preds: list, weights) -> pd.DataFrame:
    """Weighted mean of the members' raw predictions (weights sum to 1)."""
    out = preds[0].copy()
    for what in TARGETS:
        out[what] = np.sum([w * x[what].to_numpy() for x, w in zip(preds, weights)], 0)
    return out


def _cb_gpu() -> bool:
    try:
        from catboost.utils import get_gpu_device_count
        return int(get_gpu_device_count()) > 0
    except Exception:  # noqa: BLE001
        return False


def cb_signature(cfg: dict, gpu: bool) -> dict:
    return dict(gpu=bool(gpu), depth=int(cfg.get("cb_depth", 8)), lr=float(cfg.get("cb_lr", 0.06 if gpu else 0.1)),
                iters=int(cfg.get("cb_iters", 3000 if gpu else 1500)), seed=int(cfg.get("cb_seed", 0)),
                cpu_rows=None if gpu else int(cfg.get("cb_cpu_rows", 1_200_000)),
                cpu_rows_dark=None if gpu else int(cfg.get("cb_cpu_rows_dark", 500_000)),
                rows_tag=cfg.get("rows_tag", "r1"), rows=rows_sig(dict(cfg, row_seed=0, tm_exclude_dir=None, lgb_extra=None)))


def cb_complete(model_dir) -> bool:
    """A CatBoost member can be reused: report.json present and every model file matches its md5. The settings it was
    trained with are in meta.json (`cb_meta`); whether they are the wanted ones is the caller's decision."""
    import hashlib
    if not (model_dir / "report.json").exists():
        return False
    for f in CB_FILES:
        p = model_dir / f
        side = p.with_suffix(".md5")
        if not p.exists() or not side.exists() or hashlib.md5(p.read_bytes()).hexdigest() != side.read_text().strip():
            C.log(f"T1 CatBoost member {model_dir.name}: incomplete or damaged model file")
            return False
    return True


def cb_meta(model_dir) -> dict:
    f = model_dir / "meta.json"
    return json.load(open(f)) if f.exists() else {}


def train_cb(rows_dir, model_dir, cfg: dict) -> dict | None:
    """v8: a CatBoost member, another model family trained on the rows of model set 0 (the residual targets of the
    LightGBM sets), early-stopped on the same train-holdout rows. On the GPU when the runtime has one; without a GPU
    only when `cb_cpu` is set, and then on a row sample (`cb_cpu_rows`) with a higher learning rate. Returns None when
    skipped."""
    import hashlib
    from catboost import CatBoostRegressor, Pool
    gpu = _cb_gpu() if cfg.get("cb_gpu", "auto") == "auto" else bool(cfg.get("cb_gpu"))
    if not gpu and not cfg.get("cb_cpu", False):
        C.log("T1 CatBoost member: no GPU in this runtime and cb_cpu=False, skipped")
        return None
    sig = cb_signature(cfg, gpu)
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "report.json").unlink(missing_ok=True)
    rng = np.random.default_rng(sig["seed"])
    rep = {}
    for kind in ("reg", "dark"):
        D = load_rows(rows_dir, kind, "train")
        feats = [c for c in D.columns if c not in NONFEAT]
        tr = np.flatnonzero((D.src != 2).to_numpy()); va = np.flatnonzero((D.src == 2).to_numpy())
        cap = sig["cpu_rows"] if kind == "reg" else sig["cpu_rows_dark"]
        if cap is not None and len(tr) > cap:
            tr = np.sort(rng.choice(tr, cap, replace=False))
        X = D[feats].to_numpy(np.float32)
        for what in TARGETS:
            name = f"{kind}_{what}"
            y = _label(D, what) - base(D, what)
            ok = np.isfinite(y)
            w = (D.length * D.lanes).to_numpy(np.float64) if what == "k" else None
            trk, vak = tr[ok[tr]], va[ok[va]]
            prm = dict(depth=sig["depth"], learning_rate=sig["lr"], iterations=sig["iters"], loss_function="RMSE",
                       l2_leaf_reg=3.0, random_seed=sig["seed"], border_count=128, verbose=0, allow_writing_files=False,
                       task_type="GPU" if gpu else "CPU")
            if not gpu:
                prm["thread_count"] = C.THREADS
            t0 = time.time()
            pool = Pool(X[trk], y[trk], weight=None if w is None else w[trk], feature_names=feats)
            if len(vak) > 1000:
                vpool = Pool(X[vak], y[vak], weight=None if w is None else w[vak], feature_names=feats)
                m = CatBoostRegressor(**prm, od_type="Iter", od_wait=150, use_best_model=True)
                m.fit(pool, eval_set=vpool)
                bs = m.get_best_score().get("validation", {})
                rep[name] = dict(best_iter=int(m.get_best_iteration() or sig["iters"]),
                                 rmse=float(bs.get("RMSE", np.nan)), rows=int(len(trk)))
            else:
                m = CatBoostRegressor(**prm)
                m.fit(pool)
                rep[name] = dict(best_iter=sig["iters"], rows=int(len(trk)))
            path = model_dir / f"{name}.cbm"
            tmp = path.with_suffix(".tmp")
            m.save_model(str(tmp))
            digest = hashlib.md5(tmp.read_bytes()).hexdigest()
            tmp.replace(path)
            path.with_suffix(".md5").write_text(digest)
            C.log(f"T1 CatBoost {name}: {rep[name]} ({time.time() - t0:.0f}s, {'GPU' if gpu else 'CPU'})")
            del pool, m
        del D, X
        C.free()
    json.dump(dict(sig=sig), open(model_dir / "meta.json", "w"))
    json.dump(rep, open(model_dir / "report.json", "w"), indent=1)   # written last: marks the member as complete
    return rep


# ----------------------------------------------------------------------------- v8: test-month holdout
def build_tm(panel: str, cfg: dict, out_dir, avoid: list) -> dict:
    """v8 test-month holdout of one panel: `tm_n` observed, eligible, non-target cells outside the blackout rows of
    each test month, none of which a model set in `avoid` trained on (their transductive samples are re-drawn with
    plan_only), hidden in rounds the way the transductive rows are and predicted like a target. Model sets added in v8
    exclude these cells from their own samples, so the holdout measures every member on cells it never saw.
    Writes <panel>_tm_cells.npz (all chosen cells) and <panel>_tm.parquet (features and labels)."""
    f, fc, fm = out_dir / f"{panel}_tm.parquet", out_dir / f"{panel}_tm_cells.npz", out_dir / f"{panel}_tm_meta.json"
    want = json.loads(json.dumps(dict(tm=tm_sig(cfg), avoid=[dict(rows_sig(dict(c, tm_exclude_dir=None)),
                                                                 row_seed=int(c.get("row_seed", 0))) for c in avoid]),
                                 sort_keys=True, default=str))
    if f.exists() and fc.exists() and fm.exists() and json.load(open(fm)) == want:
        return dict(panel=panel, n=int(len(np.load(fc)["t"])), reused=True)
    fm.unlink(missing_ok=True)
    P = P1(panel)
    tgt = P.tgt > 0
    used = np.zeros((P.NT, P.L), bool)
    for c in avoid:
        pl = build_rows(panel, c, None, plan_only=True)
        for t, j in list(pl["reg"].values()) + list(pl["dark"].values()):
            used[t, j] = True
    rng = np.random.default_rng(int(cfg.get("tm_seed", 7)) + 1009 * C.panels().index(panel))
    parts, allt, allj, alls = [], [], [], []
    for si, split in enumerate(C.TEST_SPLITS):
        sl = C.split_rows(split)
        cand = P.ok.copy(); cand[: sl.start] = False; cand[sl.stop:] = False
        cand &= P.el & ~tgt & ~P.dark[:, None] & ~used
        ct, cl = np.nonzero(cand)
        n = min(len(ct), int(cfg.get("tm_n", 20_000)))
        if n == 0:
            continue
        s = np.sort(rng.choice(len(ct), n, replace=False))
        t_, j_ = ct[s], cl[s]
        per_round = max(1, int(cfg.get("pseudo_frac", 0.05) * len(ct)))
        order = rng.permutation(n)
        for i in range(0, n, per_round):
            o = order[i:i + per_round]
            t2, l2 = t_[o], j_[o]
            keep = {ch: P.X[ch][t2, l2].copy() for ch in CH}
            for ch in CH:
                P.X[ch][t2, l2] = np.nan
            P.refresh()
            F = P.feats(t2, l2, own=keep)
            F["y_v"] = keep["v"]; F["y_q"] = keep["q"]
            F["t"] = t2.astype(np.float32); F["j"] = l2.astype(np.float32); F["kind"] = 0.0; F["src"] = 3.0
            F["tm_split"] = np.float32(si)
            parts.append(F)
            for ch in CH:
                P.X[ch][t2, l2] = keep[ch]
        P.refresh()
        allt.append(t_); allj.append(j_); alls.append(np.full(n, si, np.int8))
    D = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if len(D):
        D = D[np.isfinite(D.y_v) & np.isfinite(D.y_q)]
    np.savez(fc, t=np.concatenate(allt) if allt else np.zeros(0, np.int64),
             j=np.concatenate(allj) if allj else np.zeros(0, np.int64),
             split=np.concatenate(alls) if alls else np.zeros(0, np.int8))
    D.to_parquet(f, index=False)
    json.dump(want, open(fm, "w"))   # written last: the holdout files of this draw are complete
    C.log(f"T1 test-month holdout {panel}: {len(D):,} cells ({int(used.sum()):,} cells of earlier model sets avoided)")
    return dict(panel=panel, n=int(len(D)), reused=False)


def _tm_worker(args):
    panel, cfg, out_dir, avoid = args
    from pathlib import Path
    return build_tm(panel, cfg, Path(out_dir), avoid)


def build_tm_all(cfg: dict, rows_dir, avoid: list) -> list:
    from concurrent.futures import ProcessPoolExecutor
    nj = int(cfg.get("row_jobs", 2))
    args = [(p, cfg, str(rows_dir), avoid) for p in C.panels()]
    if nj > 1:
        with ProcessPoolExecutor(nj) as ex:
            return list(ex.map(_tm_worker, args))
    return [_tm_worker(a) for a in args]


def _state_score(se_v, se_q, n):
    """S_state of one regime from summed squared errors (per-lane flow) and the cell count."""
    rv, rq = np.sqrt(se_v / np.maximum(n, 1)), np.sqrt(se_q / np.maximum(n, 1))
    return 0.54 * np.maximum(0, 1 - rv / 25) + 0.46 * np.maximum(0, 1 - rq / 600)


DECIDE_SPLIT = 0   # v8: index in C.TEST_SPLITS of the month whose held-out cells decide the blend (March)


def choose_blend(rows_dir, members_by_name: dict, cands: dict, cfg: dict) -> dict:
    """v8: pick the members' blend for the final Task 1 rows. `members_by_name` {name: (kind, path)}; `cands` {cand:
    {member: weight}} with a 'base' candidate (the result of earlier versions). Every candidate is scored
    (raw predictions, no post-processing)
      * on the test-month holdout: S_state per month and mask regime over cells no member trained on. Only March's
        cells decide (gain over 'base' and its day-block bootstrap SE, March days resampled); April's are reported and
        never used to choose anything;
      * on the train holdout (last HOLD_DAYS train days): S_state, S_LWR and J = 0.35 S_state + 0.10 S_LWR.
    A candidate replaces 'base' when its March gain exceeds `tm_z` SEs and its train-holdout J is not lower than
    base's by more than `hold_tol`; the best March score among those wins (run_task1 then checks the post-processed
    train-holdout J as well)."""
    z, tol = float(cfg.get("tm_z", 2.0)), float(cfg.get("hold_tol", 0.0002))
    B = int(cfg.get("tm_boot", 300))
    names = list(members_by_name)
    # ---- test-month holdout: squared errors per (panel, month, day, regime) for every candidate
    recs = []
    for p in C.panels():
        f = rows_dir / f"{p}_tm.parquet"
        if not f.exists():
            continue
        H = pd.read_parquet(f)
        if not len(H):
            continue
        preds = {n: predict_member(H, (*members_by_name[n], 1.0)) for n in names}
        dayreg = load_cache(p, ["dayreg"])["dayreg"]
        t = H.t.to_numpy(np.int64)
        day = t // C.SLOTS
        reg = dayreg[day]
        yv, yq = H.y_v.to_numpy(np.float64), H.y_q.to_numpy(np.float64)
        for c, wts in cands.items():
            v = np.sum([w * np.clip(preds[n].v.to_numpy(), 3, None) for n, w in wts.items()], 0) / sum(wts.values())
            q = np.sum([w * np.clip(preds[n].q.to_numpy(), 1, None) for n, w in wts.items()], 0) / sum(wts.values())
            g = pd.DataFrame(dict(panel=p, split=H.tm_split.to_numpy().astype(int), day=day, reg=reg,
                                  sv=(v - yv) ** 2, sq=(q - yq) ** 2, n=1))
            g = g.groupby(["panel", "split", "day", "reg"], as_index=False)[["sv", "sq", "n"]].sum()
            g["cand"] = c
            recs.append(g)
        del preds, H
        C.free()
    rep = dict(z=z, hold_tol=tol, candidates={}, weights=cands)
    if not recs:
        rep.update(chosen="base", reason="no test-month holdout cells")
        return rep
    R = pd.concat(recs, ignore_index=True)

    def score(g):
        a = g.groupby(["cand", "panel", "split", "reg"])[["sv", "sq", "n"]].sum().reset_index()
        a["S"] = _state_score(a.sv.to_numpy(), a.sq.to_numpy(), a.n.to_numpy())
        a = a.groupby(["cand", "panel", "split"]).S.mean().reset_index()      # regimes equally, as the metric
        return a.groupby(["cand", "split"]).S.mean().unstack("split")          # panels equally

    S = score(R)
    dec = DECIDE_SPLIT
    rep["decide"] = C.TEST_SPLITS[dec]
    if dec not in S.columns:
        rep.update(chosen="base", reason=f"no {C.TEST_SPLITS[dec]} holdout cells")
        return rep
    Rd = R[R.split == dec]
    days = Rd[["split", "day"]].drop_duplicates().to_numpy()
    rng = np.random.default_rng(int(cfg.get("tm_seed", 7)))
    diffs = {c: [] for c in cands if c != "base"}
    for _ in range(B):
        pick = days[rng.integers(0, len(days), len(days))]
        cnt = pd.DataFrame(pick, columns=["split", "day"]).value_counts().rename("rep").reset_index()
        Rb = Rd.merge(cnt, on=["split", "day"])
        for col in ("sv", "sq", "n"):
            Rb[col] = Rb[col] * Rb["rep"]
        Sb = score(Rb)[dec]
        for c in diffs:
            diffs[c].append(Sb[c] - Sb["base"])
    # ---- train holdout (raw, no post-processing)
    hold = holdout_raw(rows_dir, members_by_name, cands)
    for c in cands:
        d = dict(tm={str(k): round(float(v), 6) for k, v in S.loc[c].items()},
                 hold=hold.get(c, {}))
        if c != "base":
            d["tm_gain"] = round(float(S.loc[c, dec] - S.loc["base", dec]), 6)
            d["tm_se"] = round(float(np.std(diffs[c], ddof=1)), 6)
            for k in S.columns:   # the other month's gain: reported only
                if k != dec:
                    d[f"tm_gain_{C.TEST_SPLITS[k]}"] = round(float(S.loc[c, k] - S.loc["base", k]), 6)
            hb, hc = hold.get("base", {}).get("J"), hold.get(c, {}).get("J")
            d["hold_dJ"] = None if hb is None or hc is None else round(hc - hb, 6)
            d["ok"] = bool(d["tm_gain"] > z * max(d["tm_se"], 1e-9) and (d["hold_dJ"] is None or d["hold_dJ"] >= -tol))
        rep["candidates"][c] = d
    ok = [c for c, d in rep["candidates"].items() if d.get("ok")]
    rep["chosen"] = max(ok, key=lambda c: float(S.loc[c, dec])) if ok else "base"
    if cfg.get("blend_force") in cands:   # tests only: exercise a given blend end to end
        rep["chosen"], rep["forced"] = cfg["blend_force"], True
    return rep


_HOLD_PRED: dict = {}


def _hold_pred(panel: str, H: pd.DataFrame, mem) -> pd.DataFrame:
    """A member's raw predictions of a panel's train-holdout rows, computed once per run (the blend choice and the
    post-processing choice both need them; the key holds the model files' state, so a retrained member is redone)."""
    kind, path = mem[0], mem[1]
    files = CB_FILES if kind == "cb" else MODEL_FILES
    stamp = tuple((f, (path / f).stat().st_mtime_ns) for f in files if (path / f).exists())
    key = (panel, kind, str(path), stamp, len(H))
    if key not in _HOLD_PRED:
        _HOLD_PRED[key] = predict_member(H, (kind, path, 1.0))
    return _HOLD_PRED[key]


def holdout_raw(rows_dir, members_by_name: dict, cands: dict) -> dict:
    """Raw S_state, S_LWR and J of every candidate blend on the train holdout (averaged over panels)."""
    res = {c: [] for c in cands}
    for p in C.panels():
        f = rows_dir / f"{p}_hold.parquet"
        if not f.exists():
            continue
        H = pd.read_parquet(f)
        info = json.load(open(rows_dir / f"{p}_info.json"))
        P = P1(p)
        P.apply_blackouts(np.array(info["fit_origins"] + info["hold_origins"], np.int64))
        preds = {n: _hold_pred(p, H, members_by_name[n]) for n in members_by_name}
        t = H.t.to_numpy(np.int64); j = H.j.to_numpy(np.int64)
        for c, wts in cands.items():
            tw = sum(wts.values())
            v = np.clip(np.sum([w * preds[n].v.to_numpy() for n, w in wts.items()], 0) / tw, 3, None)
            q = np.clip(np.sum([w * preds[n].q.to_numpy() for n, w in wts.items()], 0) / tw, 1, None)
            s, _ = score_state(P, t, j, v, q)
            l, _ = score_lwr(P, t, j, v, q, info["hold_days"])
            res[c].append((s, l))
        del P, H, preds
        C.free()
    out = {}
    for c, v in res.items():
        if v:
            s, l = float(np.mean([a for a, _ in v])), float(np.mean([b for _, b in v]))
            out[c] = dict(S_state=round(s, 6), S_LWR=round(l, 6), J=round(0.35 * s + 0.10 * l, 6))
    return out


# ----------------------------------------------------------------------------- post-processing
def reconcile(v, q, k, vf, gate=0.6, a=0.75):
    """In dense traffic (v < gate*vf) move (v, q) so that q/v equals the density model, speed taking a share a."""
    v = v.copy(); q = q.copy()
    m = (v < gate * vf) & np.isfinite(k) & (k > 0)
    r = np.clip(k[m] / np.maximum(q[m] / np.maximum(v[m], 1.0), 1e-6), np.exp(-0.5), np.exp(0.5))
    v[m] = v[m] * r ** (-a); q[m] = q[m] * r ** (1 - a)
    return v, q


def tv_runs(k, kl, kr, lam, lam_a, iters=300):
    """Batched 1-D TV denoising with L1 anchors. k [n_runs, n]; kl, kr [n_runs] (NaN = no anchor);
    lam, lam_a [n_runs]. argmin 0.5||x-k||^2 + lam sum|x_{i+1}-x_i| + lam_a(|x_1-kl| + |kr-x_n|)."""
    R, n = k.shape
    wts = np.concatenate([np.where(np.isfinite(kl), lam_a, 0)[:, None], np.repeat(lam[:, None], n - 1, 1),
                          np.where(np.isfinite(kr), lam_a, 0)[:, None]], 1)
    e = np.zeros((R, n + 1)); e[:, 0] = -np.nan_to_num(kl); e[:, -1] = np.nan_to_num(kr)

    def D(x):  # (R, n) -> (R, n+1)
        out = np.zeros((R, n + 1)); out[:, 0] = x[:, 0]; out[:, 1:n] = x[:, 1:] - x[:, :-1]; out[:, n] = -x[:, -1]
        return out

    def DT(u):  # adjoint
        return u[:, :n] - u[:, 1:]

    u = np.zeros((R, n + 1)); c = D(k) + e; eta = 0.25
    for _ in range(iters):
        g = D(DT(u)) - c
        u = np.clip(u - eta * g, -wts, wts)
    return k - DT(u)


def tv_smooth(state_k, target, obs_k, cat, vfix, taus: dict, iters=300):
    """state_k [T, L] density (predicted at targets); target bool [T, L]; obs_k [T, L] observed density
    (NaN unobserved). Smooth every run of consecutive targets along time with each category's (tau, tau_a);
    each cell keeps the result of its own category (0 free, 1 gate, 2 dark)."""
    out = state_k.copy()
    T, L = target.shape
    runs = []
    for j in range(L):
        s, e = C.runs_1d(target[:, j])
        for a, b in zip(s, e):
            runs.append((j, a, b))
    if not runs:
        return out
    runs = np.array(runs)
    lens = runs[:, 2] - runs[:, 1]
    for n in np.unique(lens):
        sel = runs[lens == n]
        jj = sel[:, 0]; aa = sel[:, 1]
        idx_t = aa[:, None] + np.arange(n)[None, :]
        k = state_k[idx_t, jj[:, None]].astype(np.float64)
        kl = np.where(aa - 1 >= 0, obs_k[np.clip(aa - 1, 0, T - 1), jj], np.nan)
        kr = np.where(aa + n < T, obs_k[np.clip(aa + n, 0, T - 1), jj], np.nan)
        mk = np.nanmean(k, 1)
        res = {}
        for cid, (tau, tau_a) in taus.items():
            if tau <= 0 and tau_a <= 0:
                res[cid] = k
            else:
                res[cid] = tv_runs(k, kl, kr, tau * mk, tau_a * mk, iters)
        cc = cat[idx_t, jj[:, None]]
        newk = k.copy()
        for cid in res:
            m = cc == cid
            newk[m] = res[cid][m]
        out[idx_t, jj[:, None]] = newk
    return out


DEFAULT_PP = dict(gate=0.6, a=0.75, tv=True, tau_free=0.0075, tau_free_a=0.001, tau_gate=0.02, tau_gate_a=0.005,
                  tau_dark=0.05, tau_dark_a=0.0, a_out=0.0)


def postprocess_panel(P: P1, pred: pd.DataFrame, pp: dict):
    """pred: t, j, v, q (per lane), k (per lane), kind. Returns arrays v, q (per lane) at the pred rows."""
    t = pred.t.to_numpy(np.int64); j = pred.j.to_numpy(np.int64)
    v = pred.v.to_numpy(np.float64); q = pred.q.to_numpy(np.float64); k = pred.k.to_numpy(np.float64)
    vf = P.vf[j].astype(np.float64)
    v = np.clip(v, 3.0, 1.1 * vf); q = np.clip(q, 100.0 / P.lanes[j], None)
    v, q = reconcile(v, q, k, vf, pp["gate"], pp["a"])
    if pp.get("tv"):
        T, L = P.NT, P.L
        Kst = np.full((T, L), np.nan); Kst[t, j] = q / np.maximum(v, 1.0)
        tgt = np.zeros((T, L), bool); tgt[t, j] = True
        obs_k = np.where(P.ok, P.X["q"] / np.maximum(P.X["v"], 1.0), np.nan)
        cat = np.zeros((T, L), np.int8)
        cat[t, j] = np.where(pred.kind.to_numpy() == 1, 2, np.where(v < pp["gate"] * vf, 1, 0))
        taus = {0: (pp["tau_free"], pp["tau_free_a"]), 1: (pp["tau_gate"], pp["tau_gate_a"]), 2: (pp["tau_dark"], pp["tau_dark_a"])}
        Kn = tv_smooth(Kst, tgt, obs_k, cat, None, taus)
        knew = Kn[t, j]; kold = q / np.maximum(v, 1.0)
        r = np.clip(knew / np.maximum(kold, 1e-9), np.exp(-0.5), np.exp(0.5))
        a = np.where(cat[t, j] >= 1, pp["a"], pp.get("a_out", 0.0))
        v = v * r ** (-a); q = q * r ** (1 - a)
    v = np.clip(v, 3.0, 1.1 * vf); q = np.clip(q, 100.0 / P.lanes[j], None)
    return v, q


# ----------------------------------------------------------------------------- local scoring
def score_state(P: P1, t, j, v, q):
    """Exact S_state over the given target cells (per regime of their day)."""
    reg = P.dayreg[t // C.SLOTS]
    out = {}
    for r in (1, 2, 3):
        m = reg == r
        if not m.any():
            continue
        es = v[m] - P.tv[t[m], j[m]]; eq = q[m] - P.tq[t[m], j[m]]
        rs, rq = np.sqrt(np.mean(es ** 2)), np.sqrt(np.mean(eq ** 2))
        out[r] = (0.54 * max(0, 1 - rs / 25) + 0.46 * max(0, 1 - rq / 600), rs, rq)
    return float(np.mean([x[0] for x in out.values()])) if out else np.nan, out


def score_lwr(P: P1, t, j, v, q, days):
    """Task 3 conservation proxy on the given days (truth everywhere except the predicted cells)."""
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
    el = P.el[a:b] & np.isfinite(TV) & np.isfinite(TQ)
    rv = P.rv[a:b]
    rr = rows[a:b]
    reg = np.repeat(P.dayreg, C.SLOTS)[a:b]
    out = {}
    for r in (1, 2, 3):
        rm = rr & (reg == r)
        ok = el[:-1] & el[1:] & rm[:-1, None] & rm[1:, None] & rv[:-1] & P.observable[None, :]
        dNt = (Nt[1:] - Nt[:-1])[ok]; dNs = (Ns[1:] - Ns[:-1])[ok]
        den = np.abs(dNt).sum()
        if den > 0:
            out[r] = 1 - min(1.0, np.abs(dNs - dNt).sum() / den)
    return float(np.mean(list(out.values()))) if out else np.nan, out


# ----------------------------------------------------------------------------- driver
def run_holdout_eval(rows_dir, model_dir, pp_grid=None) -> pd.DataFrame:
    """Predict the holdout rows, try post-processing variants, score S_state, S_LWR and J per panel."""
    res = []
    pp_grid = pp_grid or [DEFAULT_PP]
    for p in C.panels():
        f = rows_dir / f"{p}_hold.parquet"
        if not f.exists():
            continue
        H = pd.read_parquet(f)
        info = json.load(open(rows_dir / f"{p}_info.json"))
        P = P1(p)
        P.apply_blackouts(np.array(info["fit_origins"] + info["hold_origins"], np.int64))
        pred = predict_frame(H, model_dir)
        t = pred.t.to_numpy(np.int64); j = pred.j.to_numpy(np.int64)
        vr = np.clip(pred.v.to_numpy(), 3, None); qr = np.clip(pred.q.to_numpy(), 1, None)
        s0, _ = score_state(P, t, j, vr, qr); l0, _ = score_lwr(P, t, j, vr, qr, info["hold_days"])
        res.append(dict(panel=p, variant="raw", S_state=s0, S_LWR=l0))
        for i, pp in enumerate(pp_grid):
            v, q = postprocess_panel(P, pred, pp)
            s, _ = score_state(P, t, j, v, q); l, _ = score_lwr(P, t, j, v, q, info["hold_days"])
            res.append(dict(panel=p, variant=f"pp{i}", S_state=s, S_LWR=l, **{k: pp[k] for k in pp}))
        C.log(f"T1 holdout {p}: raw S_state {s0:.4f} LWR {l0:.4f} | pp0 S_state {res[-len(pp_grid)]['S_state']:.4f} LWR {res[-len(pp_grid)]['S_LWR']:.4f}")
    R = pd.DataFrame(res)
    R["J"] = 0.35 * R.S_state + 0.10 * R.S_LWR
    return R


def predict_test(rows_dir, model_dirs, pp: dict, out_csv) -> pd.DataFrame:
    """Blend the members' raw predictions (model-set directories averaged, or (kind, path, weight) members), post-process
    per panel, write the Task 1 rows for both splits."""
    mems = as_members(model_dirs)
    outs = []
    for p in C.panels():
        D = pd.read_parquet(rows_dir / f"{p}_test.parquet")
        preds = [predict_member(D, m) for m in mems]
        pred = blend_preds(preds, [w for _, _, w in mems])
        del preds
        P = P1(p)
        v, q = postprocess_panel(P, pred, pp)
        t = pred.t.to_numpy(np.int64); j = pred.j.to_numpy(np.int64)
        o = pd.DataFrame({"panel": p, "timestamp": C.ts_text(t), "station_id": np.asarray(P.stations)[j],
                          "link_id": np.asarray(P.net.links)[j], "mask_regime": [C.REG_NAME[int(r)] for r in P.tgt[t, j]],
                          "speed_kmh": v, "flow_vph": q * P.lanes[j]})
        outs.append(o)
        C.log(f"T1 test {p}: {len(o):,} rows, mean speed {v.mean():.1f}")
        del P, D, pred
        C.free()
    out = pd.concat(outs, ignore_index=True)
    out.to_parquet(out_csv, index=False)
    return out


# ----------------------------------------------------------------------------- stage driver
def _rows_worker(args):
    panel, cfg, out_dir = args
    from pathlib import Path
    out_dir = Path(out_dir)
    need = [f"{panel}_info.json", f"{panel}_train.parquet"] + ([] if cfg.get("row_seed", 0) else [f"{panel}_test.parquet"])
    if all((out_dir / n).exists() for n in need):
        return json.load(open(out_dir / f"{panel}_info.json"))
    return build_rows(panel, cfg, out_dir)


def _build_rows_all(cfg: dict, rows_dir) -> list:
    from concurrent.futures import ProcessPoolExecutor
    rows_dir.mkdir(parents=True, exist_ok=True)
    nj = int(cfg.get("row_jobs", 2))
    args = [(p, cfg, str(rows_dir)) for p in C.panels()]
    if nj > 1:
        with ProcessPoolExecutor(nj) as ex:
            return list(ex.map(_rows_worker, args))
    return [_rows_worker(a) for a in args]


def pp_grid(base_pp=None):
    b = dict(base_pp or DEFAULT_PP)
    g = [b, dict(b, tv=False), dict(b, gate=0.5), dict(b, gate=0.7), dict(b, gate=0.0)]
    for f in (0.5, 1.5):
        g.append(dict(b, **{k: b[k] * f for k in ("tau_free", "tau_free_a", "tau_gate", "tau_gate_a", "tau_dark")}))
    return g


def plan_seeds(cfg: dict) -> list:
    """The LightGBM model sets of a plan: `seeds` followed by the `extra_seeds` not already in it."""
    seeds = [int(s) for s in cfg.get("seeds", [0])]
    return seeds + [int(s) for s in cfg.get("extra_seeds", []) if int(s) not in seeds]


def plan_signature(cfg: dict) -> dict:
    """What a Task 1 result is made of (v8): the LightGBM model sets (with the row settings of the sets that override
    them), the CatBoost member and the test-month holdout. A stored result is reused only for the same plan."""
    seeds = plan_seeds(cfg)
    scfg = {str(k): v for k, v in sorted((cfg.get("seed_cfg") or {}).items(), key=lambda kv: int(kv[0])) if int(k) in seeds}
    keys = ("rows_tag", "rows_per_seed", "full_refit", "n_reg", "n_dark", "n_pseudo", "n_pseudo_dark", "pseudo_frac",
            "max_rounds", "pp")
    sig = dict(seeds=seeds, seed_cfg=scfg, **{k: cfg.get(k) for k in keys})
    if cfg.get("cb", False):
        sig["cb"] = {k: cfg.get(k) for k in ("cb_gpu", "cb_cpu", "cb_depth", "cb_lr", "cb_iters", "cb_cpu_rows",
                                             "cb_cpu_rows_dark", "cb_seed", "cb_weights")}
    if cfg.get("tm_holdout", True) and (scfg or cfg.get("cb", False)):   # exactly when run_task1 uses it
        sig["tm"] = {k: cfg.get(k) for k in ("tm_n", "tm_seed", "tm_z", "hold_tol")}
    if cfg.get("blend_force"):   # tests only
        sig["blend_force"] = cfg["blend_force"]
    return json.loads(json.dumps(sig, sort_keys=True, default=str))


def run_task1(cfg: dict, on_set_done=None) -> dict:
    """Rows -> models (one or more seeds) -> holdout scoring and post-processing choice -> test rows.

    Model set s > 0 is trained on its own training sample (row seed s, `rows_per_seed`): the ensemble then averages
    models that saw different rows, not only different bagging seeds. Holdout and test rows are always those of row
    seed 0, so every set is scored on the same holdout. A set whose models exist (restored from Drive) is reused.
    v8: `seed_cfg` {seed: settings} lets a further set draw a larger transductive sample (and train deterministically);
    such sets never train on the test-month holdout (`tm_holdout`), which is drawn among the cells the sets without
    overrides did not train on. `cb` adds a CatBoost member. Members and weights are then chosen by `choose_blend`
    (test-month holdout + train holdout; the sets without overrides averaged, as earlier versions did, stay unless a
    blend beats them), and the post-processing is chosen on the train holdout for the chosen blend."""
    import shutil
    tag = cfg.get("rows_tag", "r1")
    rows_dir = C.WORK / "t1" / f"rows_{tag}"
    infos = _build_rows_all(dict(cfg, row_seed=0), rows_dir)
    C.log(f"T1 rows ready: {sum(i['n_train'] for i in infos):,} train, {sum(i['n_pseudo'] for i in infos):,} pseudo, "
          f"{sum(i['n_test'] for i in infos):,} test")
    seeds = plan_seeds(cfg)   # v8: seeds + extra_seeds (earlier notebooks passed the union as `seeds`)
    seed_cfg = {int(k): dict(v) for k, v in (cfg.get("seed_cfg") or {}).items()}
    legacy = [s for s in seeds if s not in seed_cfg]           # sets drawn like earlier versions
    if not legacy:
        raise ValueError("T1: at least one model set must use the base row settings (no seed_cfg entry)")
    # the test-month holdout decides whether anything beyond the sets of earlier versions is used
    use_tm = bool(cfg.get("tm_holdout", True)) and (any(s in seed_cfg for s in seeds) or bool(cfg.get("cb", False)))
    if use_tm:
        build_tm_all(cfg, rows_dir, avoid=[dict(cfg, row_seed=s) for s in legacy])
    lgb_dirs = {}
    for s in seeds:
        scfg = dict(cfg, **seed_cfg.get(s, {}))
        if s in seed_cfg and use_tm:
            scfg["tm_exclude_dir"] = str(rows_dir)
        md = C.WORK / "t1" / f"models_{tag}_s{s}"
        rd = rows_dir
        if s and cfg.get("rows_per_seed", True):
            rd = C.WORK / "t1" / f"rows_{tag}_rs{s}"
        rs = s if rd != rows_dir else 0
        sig = rows_sig(dict(scfg, row_seed=rs)) if s in seed_cfg else None
        if s in seed_cfg and rd == rows_dir:
            raise ValueError(f"T1: model set {s} overrides row settings, so it needs its own rows (rows_per_seed)")
        have = set_complete(md, rs, sig)
        if not have or (cfg.get("full_refit", False) and not set_complete(C.WORK / "t1" / f"models_{tag}_s{s}_full", rs, sig)):
            if rd != rows_dir:
                _build_rows_all(dict(scfg, row_seed=s), rd)
        if not have:
            (md / "report.json").unlink(missing_ok=True)
            train_models(rd, md, dict(scfg, seed=s, row_seed=rs), holdout=True)
            if on_set_done is not None:   # e.g. copy the finished set to Drive before the next one starts
                on_set_done()
        if cfg.get("full_refit", False):
            mf = C.WORK / "t1" / f"models_{tag}_s{s}_full"
            if not set_complete(mf, rs, sig):
                (mf / "report.json").unlink(missing_ok=True)
                train_models(rd, mf, dict(scfg, seed=s, row_seed=rs), holdout=False, rounds=json.load(open(md / "report.json")))
                if on_set_done is not None:
                    on_set_done()
            lgb_dirs[s] = (md, mf)
        else:
            lgb_dirs[s] = (md, md)
        if rd != rows_dir and rd.exists():   # a further set's training rows are not needed after its models exist
            shutil.rmtree(rd, ignore_errors=True)
    # ---- v8: CatBoost member (rows of set 0)
    cb_dir, cb_error = None, None
    if cfg.get("cb", False):
        cbd = C.WORK / "t1" / f"cb_{tag}"
        gpu = _cb_gpu() if cfg.get("cb_gpu", "auto") == "auto" else bool(cfg.get("cb_gpu"))
        want = cb_signature(cfg, gpu)
        have = cb_complete(cbd) and cb_meta(cbd).get("sig") is not None
        if have and json.dumps(cb_meta(cbd)["sig"], sort_keys=True) != json.dumps(want, sort_keys=True):
            # a member trained elsewhere (e.g. on a GPU runtime) is kept when only the device-dependent settings differ
            old = cb_meta(cbd)["sig"]
            have = old.get("rows") == want["rows"] and old.get("rows_tag") == want["rows_tag"] and old.get("seed") == want["seed"] \
                and old.get("depth") == want["depth"]
            C.log(f"T1 CatBoost member: trained with {old}; {'kept' if have else 'retrained'} (wanted {want})")
        if not have and not gpu and not cfg.get("cb_cpu", False):
            C.log("T1 CatBoost member: no GPU in this runtime and cb_cpu=False, skipped")
        elif not have:
            att = cbd / "attempts.json"   # copied to Drive before training: survives a kernel that is killed mid-way
            try:
                n_att = int(json.load(open(att)).get("n", 0)) if att.exists() else 0
            except Exception:  # noqa: BLE001
                n_att = 0
            if n_att >= 2:
                cb_error = (f"{n_att} earlier attempts did not finish (the kernel was killed or disconnected); delete "
                            f"t1/{cbd.name} in Drive and add 'task1' to RERUN to try again")
                C.log(f"T1 CatBoost member skipped: {cb_error}")
            else:
                cbd.mkdir(parents=True, exist_ok=True)
                json.dump({"n": n_att + 1}, open(att, "w"))
                if on_set_done is not None:
                    on_set_done()
                try:
                    if train_cb(rows_dir, cbd, cfg) is not None:
                        have = True
                except Exception as e:  # noqa: BLE001  a GPU / driver / install problem must not stop the Task 1 upgrade
                    cb_error = f"{type(e).__name__}: {e}"[:400]
                    C.log(f"T1 CatBoost member failed ({cb_error}); the blend is chosen without it")
                    C.free()
                json.dump({"n": 0}, open(att, "w"))   # this attempt ended (trained, skipped or a caught error)
                if on_set_done is not None:
                    on_set_done()
        if have:
            cb_dir = cbd
    # ---- members and their blend
    mem = {f"lgb{s}": ("lgb", lgb_dirs[s][0]) for s in seeds}          # holdout models (scored on the train holdout)
    cands = {"base": {f"lgb{s}": 1.0 for s in legacy}}
    newer = [s for s in seeds if s in seed_cfg]
    if newer:
        cands["lgb_all"] = {f"lgb{s}": 1.0 for s in seeds}
    if cb_dir is not None:
        mem["cb"] = ("cb", cb_dir)
        for w in cfg.get("cb_weights", [0.25, 0.4]):
            for nm in [c for c in ("base", "lgb_all") if c in cands]:
                g = cands[nm]
                cands[f"{nm}+cb{w}"] = dict({k: (1 - w) / len(g) for k in g}, cb=float(w))
    if len(cands) > 1:
        blend = choose_blend(rows_dir, mem, cands, cfg)
        C.log(f"T1 blend: chosen {blend['chosen']} | " + json.dumps(blend["candidates"], default=str)[:900])
    else:
        blend = dict(chosen="base", candidates={}, weights=cands)
    # ---- holdout: score raw and post-processed variants of the chosen blend (and of base, when another blend won)
    grid = pp_grid(cfg.get("pp"))

    def pp_table(name):
        R_ = run_holdout_eval_multi(rows_dir, [(mem[n][0], mem[n][1], w) for n, w in cands[name].items()], grid)
        return R_, R_.groupby("variant")[["S_state", "S_LWR", "J"]].mean().sort_values("J", ascending=False)

    R, summ = pp_table(blend["chosen"])
    if blend["chosen"] != "base":
        Rb, sb = pp_table("base")
        jc, jb = float(summ.J.iloc[0]), float(sb.J.iloc[0])
        blend["pp_check"] = dict(chosen_best_J=round(jc, 6), base_best_J=round(jb, 6))
        if jc < jb - float(cfg.get("hold_tol", 0.0002)) and not blend.get("forced"):
            C.log(f"T1 blend {blend['chosen']}: post-processed train-holdout J {jc:.6f} < base's {jb:.6f}; keeping base")
            blend["reverted_from"], blend["chosen"] = blend["chosen"], "base"
            R, summ = Rb, sb
    wts = cands[blend["chosen"]]
    final_members = [(mem[n][0], lgb_dirs[int(n[3:])][1] if n.startswith("lgb") else mem[n][1], w) for n, w in wts.items()]
    R.to_csv(C.WORK / "t1" / "holdout_scores.csv", index=False)
    print(summ.round(5).to_string())
    best = summ.index[0]
    pp = dict(DEFAULT_PP, gate=0.0, tv=False) if best == "raw" else grid[int(best[2:])]
    C.log(f"T1 chosen post-processing: {best} {pp}")
    (C.WORK / "pred").mkdir(parents=True, exist_ok=True)
    out = predict_test(rows_dir, final_members, pp, C.WORK / "pred" / "task1.parquet")
    reps = {}
    for s in seeds:
        f = lgb_dirs[s][0] / "report.json"
        reps[lgb_dirs[s][0].name] = json.load(open(f)) if f.exists() else {}
    if cb_dir is not None:
        reps[cb_dir.name] = json.load(open(cb_dir / "report.json"))
    _HOLD_PRED.clear()
    json.dump(dict(pp=pp, holdout=summ.round(6).reset_index().to_dict("records"), seeds=seeds,
                   rows_tag=tag, rows_per_seed=bool(cfg.get("rows_per_seed", True)), full_refit=bool(cfg.get("full_refit", False)),
                   models=reps, plan=plan_signature(cfg), blend=blend,
                   members={n: dict(kind=mem[n][0], dir=(lgb_dirs[int(n[3:])][1] if n.startswith("lgb") else mem[n][1]).name,
                                    weight=round(float(w) / sum(wts.values()), 6)) for n, w in wts.items()},
                   cb_trained=cb_dir is not None, cb_error=cb_error,
                   task1_md5=file_md5(C.WORK / "pred" / "task1.parquet")),
              open(C.WORK / "t1" / "choice.json", "w"), indent=1, default=str)
    return dict(pp=pp, holdout=summ, blend=blend)


def result_up_to_date(cfg: dict, work=None) -> tuple:
    """(True, why) when the Task 1 result on disk was built for this plan (v8) and is consistent; a plan that wanted a
    CatBoost member which could not be trained (no GPU) is redone once a GPU (or cb_cpu) makes it possible."""
    st = result_state(work)
    if not st["exists"]:
        return False, "no Task 1 result yet"
    if not st["consistent"]:
        return False, "task1.parquet does not match t1/choice.json"
    if st.get("plan") != plan_signature(cfg):
        return False, "the result was built for another plan (model sets / members)"
    if cfg.get("cb", False):
        ch = json.load(open((work or C.WORK) / "t1" / "choice.json"))
        can = _cb_gpu() if cfg.get("cb_gpu", "auto") == "auto" else bool(cfg.get("cb_gpu"))
        # a member that failed to train (cb_error) is not retried by itself: RERUN 'task1' tries again
        if not ch.get("cb_trained", False) and not ch.get("cb_error") and (can or cfg.get("cb_cpu", False)):
            return False, "the CatBoost member can be trained now"
    return True, "up to date"


def file_md5(path) -> str:
    import hashlib
    h = hashlib.md5()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 24), b""):
            h.update(blk)
    return h.hexdigest()


def result_state(work=None) -> dict:
    """What the Task 1 result on disk is: the model sets of choice.json, and whether task1.parquet is the file that
    choice.json describes (a run interrupted between copying the two can leave them out of step; choice.json files of
    earlier versions carry no checksum and are taken as consistent). v8: `plan` is the plan signature the result was
    built for (None for earlier versions)."""
    work = work or C.WORK
    cp, tp = work / "t1" / "choice.json", work / "pred" / "task1.parquet"
    if not (cp.exists() and tp.exists()):
        return dict(exists=False, seeds=None, consistent=False, plan=None)
    ch = json.load(open(cp))
    md = ch.get("task1_md5")
    return dict(exists=True, seeds=list(ch.get("seeds", [0])), consistent=(md is None or md == file_md5(tp)),
                plan=ch.get("plan"))


def run_holdout_eval_multi(rows_dir, model_dirs, pp_grid_):
    """Like run_holdout_eval, but blends several members' raw predictions first (model-set directories averaged, or
    (kind, path, weight) members)."""
    mems = as_members(model_dirs)
    res = []
    for p in C.panels():
        f = rows_dir / f"{p}_hold.parquet"
        if not f.exists():
            continue
        H = pd.read_parquet(f)
        info = json.load(open(rows_dir / f"{p}_info.json"))
        P = P1(p)
        P.apply_blackouts(np.array(info["fit_origins"] + info["hold_origins"], np.int64))
        preds = [_hold_pred(p, H, m) for m in mems]
        pred = blend_preds(preds, [w for _, _, w in mems])
        t = pred.t.to_numpy(np.int64); j = pred.j.to_numpy(np.int64)
        vr = np.clip(pred.v.to_numpy(), 3, None); qr = np.clip(pred.q.to_numpy(), 1, None)
        s0, d0 = score_state(P, t, j, vr, qr); l0, _ = score_lwr(P, t, j, vr, qr, info["hold_days"])
        res.append(dict(panel=p, variant="raw", S_state=s0, S_LWR=l0,
                        rmse_v=float(np.mean([x[1] for x in d0.values()])), rmse_q=float(np.mean([x[2] for x in d0.values()]))))
        for i, pp in enumerate(pp_grid_):
            v, q = postprocess_panel(P, pred, pp)
            s, d = score_state(P, t, j, v, q); l, _ = score_lwr(P, t, j, v, q, info["hold_days"])
            res.append(dict(panel=p, variant=f"pp{i}", S_state=s, S_LWR=l,
                            rmse_v=float(np.mean([x[1] for x in d.values()])), rmse_q=float(np.mean([x[2] for x in d.values()]))))
        C.log(f"T1 holdout {p}: raw S_state {s0:.4f} S_LWR {l0:.4f}")
        del P, H, preds, pred
        C.free()
    R = pd.DataFrame(res)
    R["J"] = 0.35 * R.S_state + 0.10 * R.S_LWR
    return R
