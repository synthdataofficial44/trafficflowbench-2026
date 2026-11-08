"""v13 Task 2 (onset only): a listwise ranker and a location-light weighting of the onset models behind the two-draw gate,
on top of the v12 run's probabilities - nothing else of Task 2 is refitted, and every v10-v12 checkpoint reloads.

Why. The onset loss is the site. On March (another demand draw than train) the v12 onset models put the queue on the
right links in 87 % of the pseudo windows against 95 % on the training draw; when they miss, the true site is ~17 km
away and outside their top five in 58 % of the misses. The organizers: each month is generated with its own demand
draw and incident schedule, and "what will separate the leaderboard is which links queue". Two changes aimed at that,
both fixed before any March number was looked at:
  rk50  a listwise ranker (LightGBM rank_xendcg, one query per window, the onset recipe's features and variants): it
        learns which link of a window queues *relative to the other links of the same window*, while the per-row
        classifier also spends capacity on window-level calibration (time of day, demand level) that does not separate
        the links. Its softmax over the window's links gets a temperature fitted out of fold on the training draw (the
        softmax of rank_xendcg is deliberately flat - its target mixes the labels with noise - so its raw scale says
        nothing about how sure it is); times the classifier's expected number of queued links it gives per-link
        probabilities, blended 50/50 with the classifier;
  nl75  the classifier's location-free variant weighted 0.75 instead of 0.5 (the location columns encode where queues
        started on the training draw).
Gate: the v5/v6 two-draw rule - March pseudo windows (another draw; train-only models) and the week-blocked train CV
(official rule: eligible cells at T+30, empty union = 1) - with the bars for two alternatives. An adopted candidate
changes the onset rows of March (train-only models: the public board reads it) and of April in both files: the hedge
with the train-only fields; the main file on top of the v12 April refit (the ranker refitted on train + March like the
classifier was), and only if the same change does not lose within March (fitted on the first half, scored on the
second, like the April gate) - otherwise the main file keeps v12's April onset rows. The changes are applied to v12's
stored probabilities (the reloaded models must reproduce them closely, else the step stops), so nothing else moves.
Only data timestamped <= T feeds a window at origin T; no April label is evaluated or used to choose anything.

v13.1 (after the v13 run: nl75 +0.022 onset IoU on March, 2.3 SEs, 21 windows better / 8 worse, 3 of 4 corridor
families, no loss on the first-week windows - but -0.0055 on the training-draw CV, so the two-draw rule did not take it;
the ranker lost on all three reads, public -0.00062): `rule13="cross_draw"` decides on the draw that is not the training
draw. The organizers generate every month from its own demand draw, so April is a new draw like March, and the training
CV measures how well a model remembers where queues start on the training draw itself. Twice before the two reads
disagreed (v6 `+rd`: train CV for, March against; v10 month memory: March for, train CV against) and both times the
public read of the scored March windows sided with March. Under the cross-draw rule a candidate is taken when its March
gain exceeds `z13x` (2.0, both alternatives counted) bootstrap SEs, both March reads are positive, and the v13 safety
checks pass (no loss beyond one SE on the official-style or first-week windows, 3 of 4 corridor families, enough
windows); the training CV is reported, not a veto. The rule was set after the v13 gate numbers were known, which the
report says (`gate["rule_note"]`); the public read of the 40 scored March onset windows is its first out-of-sample test.
The main file's April refit keeps the v13 within-March check (it is not forced). Files: task2_v131*.
"""
from __future__ import annotations

import copy
import hashlib
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import common as C
from . import task2 as T
from . import task2m as TM
from . import task2x as X
from . import task2y as Y

VERSION = "v13.1"
MAIN_FILE, HEDGE_FILE = "task2_v131.csv", "task2_v131_hedge.csv"
PROBE_FILE = "task2_v131_probe_rk50.csv"   # only with probe13 and nothing adopted: v12 main + the ranker's March onset rows
REPORT_FILE = "task2_v131_report.json"
PROBS_FILE = "task2_v131_probs.pkl"
COND = "queue_onset"
KEY = TM.KEY
QDT = TM.QDT
RANK_P = dict(objective="rank_xendcg", learning_rate=0.05, num_leaves=63, min_data_in_leaf=100, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, max_bin=255, verbose=-1)
RK_TAG = hashlib.md5(json.dumps(RANK_P, sort_keys=True).encode()).hexdigest()[:6]
CANDS = ("rk50", "nl75")
# rank_seeds: the ranker ensemble of the files is the one the calibration is fitted for (one seed x the two variants, as
# every out-of-fold fold); more seeds would average the raw scores into a smoother scale than the calibration saw
DEFAULTS = dict(z13=1.0, z13_cv=2.0, z13_alt=0.5, cv13_z=1.0, rank_seeds=[0], rank_rounds=None, nl_w=0.75,
                rk_w=0.5, main_check=True, min_windows13=40, fam_share13=0.75, reload_tol=0.02, force13=None,
                rule13="two_draw", z13x=2.0, probe13=True)
RULES = ("two_draw", "cross_draw")
RULE_NOTE = ("cross_draw (v13.1): set after the v13 gate results were known, on the record of earlier runs (v6 +rd and "
             "v10 month memory: when March pseudo and the train CV disagreed, the public March read sided with March "
             "both times) and the organizers' statement that every month is a new demand draw")


def install():
    Y.install()


def code_hash() -> str:
    h = hashlib.md5((Y.code_hash() + "|").encode())
    h.update(Path(__file__).read_bytes())
    return h.hexdigest()[:16]


def settings(cfg13: dict | None) -> dict:
    s = dict(DEFAULTS)
    s.update({k: v for k, v in (cfg13 or {}).items() if v is not None or k == "force13"})
    unknown = set(s) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"v13 settings: unknown keys {sorted(unknown)}")
    if s["force13"] not in (None,) + CANDS:
        raise ValueError(f"v13 force13 must be None or one of {CANDS}")
    if s["rule13"] not in RULES:
        raise ValueError(f"v13 rule13 must be one of {RULES}")
    return s


# ----------------------------------------------------------------------------- the ranker
def _groups(key: np.ndarray):
    """(starts, sizes) of the runs of equal keys (the rows of one window are contiguous in every onset table)."""
    if len(key) == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    brk = np.flatnonzero(np.diff(key) != 0) + 1
    starts = np.r_[0, brk].astype(np.int64)
    sizes = np.diff(np.r_[starts, len(key)]).astype(np.int64)
    return starts, sizes


def fit_rank(frames, feats, rounds, seed=0, masks=None, weight=None):
    """LightGBM listwise ranker (one query per window; windows without a queued known link carry no ranking signal and
    are left out). `weight`: per row (a window's rows share it)."""
    import lightgbm as lgb
    frames = list(frames) if isinstance(frames, (list, tuple)) else [frames]
    p = dict(RANK_P, num_threads=C.THREADS, seed=seed, bagging_seed=seed + 1, feature_fraction_seed=seed + 2,
             objective_seed=seed + 3)
    Xm = T._matrix(frames, feats, masks)
    y = np.concatenate([f.y.to_numpy(np.float32) if masks is None else f.y.to_numpy(np.float32)[masks[j]]
                        for j, f in enumerate(frames)])
    key = np.concatenate([T._wkey(f) if masks is None else T._wkey(f)[masks[j]] for j, f in enumerate(frames)])
    starts, sizes = _groups(key)
    if len(np.unique(key)) != len(sizes):
        raise RuntimeError("v13 ranker: the rows of a window are not contiguous")
    pos = np.add.reduceat(y, starts) > 0 if len(starts) else np.zeros(0, bool)
    if not pos.any():
        raise RuntimeError("v13 ranker: no window with a queued link")
    if pos.all():   # every selector-drawn onset window holds a queued link: no copy of the matrix
        Xk, yk, w = Xm, y, None if weight is None else np.asarray(weight, np.float64)
    else:
        keep = np.repeat(pos, sizes)
        Xk, yk, w = Xm[keep], y[keep], None if weight is None else np.asarray(weight, np.float64)[keep]
    ds = lgb.Dataset(Xk, yk.astype(np.float32), group=sizes[pos], weight=w, feature_name=list(feats),
                     params=p, free_raw_data=True)
    ds.construct()
    del Xm, Xk
    C.free()
    return lgb.train(p, ds, int(rounds))


def _rank_cached(frames, feats, rounds, fset, variant, seed, role, masks=None, weight=None):
    path = T._ckpt_path(COND, role, variant, seed, rounds, fset, feats)
    m = T._load_ckpt(path, feats, rounds)
    if m is None:
        m = fit_rank(frames, feats, rounds, seed=seed, masks=masks, weight=weight)
        T._save_ckpt(m, path)
    return m


def rank_set(frames, fset, vnames, seeds, rounds, role, weight=None):
    """[(booster, feats)] of the ranker for every variant and seed (variant-major), checkpointed."""
    out = []
    for v in vnames:
        fs = T._feats(frames[0], v, fset)
        for sd in seeds:
            out.append((_rank_cached(frames, fs, rounds, fset, v, sd, f"{role}:rk{RK_TAG}", None, weight), fs))
    return out


def _lse(z, axis=-1):
    m = np.max(z, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    return (m + np.log(np.sum(np.exp(z - m), axis=axis, keepdims=True))).squeeze(axis)


def softmax_t(scores, tau):
    """Softmax of ranker scores over a window's links (last axis) at temperature tau."""
    z = np.asarray(scores, np.float64) / float(tau)
    z = z - z.max(-1, keepdims=True)
    e = np.exp(np.clip(z, -700.0, 0.0))
    return e / e.sum(-1, keepdims=True)


TAU_GRID = np.exp(np.linspace(np.log(0.005), np.log(200.0), 161))


def fit_calib(S: dict, sets, pans) -> dict:
    """The ranker's temperature, fitted out of fold on the train onset windows: tau maximises the within-window
    likelihood that the softmax(s / tau) mass falls on the links truly queued at T+30 (a conditional logit - every window
    is normalised on its own, so corridors with 20 or 250 links weigh alike and the link count plays no part). The
    ranker's per-link probability is then the classifier's expected number of queued links times that softmax.
    `informative`: the fitted softmax beats the uniform one by more than 0.01 nats per window (a ranker without signal
    has its optimum at the uniform limit and is left out of the gate)."""
    Zs, Ys = [], []
    for p in pans:
        if p not in S or COND not in sets.get(p, {}):
            continue
        yc = np.asarray(sets[p][COND][1][:, T.K - 1, :], bool)
        sp = np.asarray(S[p], np.float64)
        ok = yc.any(1)
        if ok.any():
            Zs.append(sp[ok]); Ys.append(yc[ok])
    if not Zs:
        return dict(tau=1.0, windows=0, informative=False, reason="no train onset window with a queued link")
    n = sum(len(z) for z in Zs)
    ll_unif = float(sum(np.sum(np.log(y.sum(1) / y.shape[1])) for y in Ys))

    def ll(tau):
        tot = 0.0
        for z, y in zip(Zs, Ys):
            zt = z / tau
            tot += float(np.sum(_lse(np.where(y, zt, -np.inf)) - _lse(zt)))
        return tot

    vals = np.array([ll(t) for t in TAU_GRID])
    i = int(np.argmax(vals))
    lo, hi = TAU_GRID[max(i - 1, 0)], TAU_GRID[min(i + 1, len(TAU_GRID) - 1)]
    fine = np.exp(np.linspace(np.log(lo), np.log(hi), 41))
    fv = np.array([ll(t) for t in fine])
    j = int(np.argmax(fv))
    tau, best = float(fine[j]), float(fv[j])
    gain = (best - ll_unif) / n
    return dict(tau=tau, windows=int(n), loglik_per_window=best / n, loglik_uniform_per_window=ll_unif / n,
                gain_per_window=gain, at_grid_edge=bool(i in (0, len(TAU_GRID) - 1)),
                informative=bool(gain > 0.01 and i < len(TAU_GRID) - 1))


def rank_prob(base, scores, cal: dict):
    """The ranker's per-link probabilities: the classifier field's expected number of queued links (its sum over the
    window) times the ranker's softmax at the fitted temperature, capped at 1."""
    mass = np.asarray(base, np.float64).sum(-1, keepdims=True)
    return np.clip(mass * softmax_t(scores, cal["tau"]), 0.0, 1.0)


def combine(name, fa, fn, s, st, cal=None):
    """A candidate field from the classifier's 'all' (fa) and 'noloc' (fn) fields and the ranker scores (s)."""
    base = 0.5 * (np.asarray(fa, np.float64) + np.asarray(fn, np.float64))
    if name == "base":
        return base
    return apply_cand(name, base, fa, fn, s, st, cal=cal)


def apply_cand(name, b, fa, fn, s, st, scale=1.0, cal=None):
    """Candidate `name` applied to a field `b` whose classifier part averages the variants 'all' (fa) and 'noloc' (fn)
    with weight `scale` (0.5 when b is the 50/50 blend with the location-free April refit): nl75 moves the variant weights,
    rk50 blends in the ranker's probabilities (b's expected count times the tempered softmax). Written as changes of b, so the stored v12 fields are the starting point
    (a recomputation that differs from them in the last digits changes nothing else)."""
    b = np.asarray(b, np.float64)
    if name == "nl75":
        d = (float(st["nl_w"]) - 0.5) * (np.asarray(fn, np.float64) - np.asarray(fa, np.float64))
        return np.clip(b + float(scale) * d, 0.0, 1.0)
    if name == "rk50":
        if cal is None:
            raise ValueError("rk50 needs the ranker calibration")
        w = float(st["rk_w"])
        return np.clip((1.0 - w) * b + w * rank_prob(b, s, cal), 0.0, 1.0)
    raise ValueError(name)


# ----------------------------------------------------------------------------- fields
def official_fields(models, offis, pans, splits) -> dict:
    """{panel: {split: {window_id: [L]}}}: the mean prediction of `models` on the onset official windows (classifier:
    probabilities, ranker: raw scores)."""
    out = {}
    for i, p in enumerate(pans):
        for split, (wi, df, step) in offis[p].items():
            if split not in splits:
                continue
            sel = (wi.condition == COND).to_numpy()
            if not sel.any():
                continue
            Xo = df.copy(); Xo["panel_code"] = np.float32(i)
            pr = T._predict(models, Xo).reshape(len(wi), -1)
            for w in np.flatnonzero(sel):
                out.setdefault(p, {}).setdefault(split, {})[wi.window_id.iloc[w]] = pr[w].astype(np.float64)
            del Xo
    return out


def pseudo_fields(models, pseudo, pans, Ps, split="validation") -> dict:
    """{panel: [n windows, L]} of `models` on the onset pseudo windows of a month."""
    out = {}
    for p in pans:
        ent = pseudo.get(p, {}).get(split, {}).get(COND)
        if ent is None:
            continue
        Wc, _, _, Xp = ent
        out[p] = T._field(COND, Xp, T._predict(models, Xp), len(Wc), Ps[p]["L"])
    return out


def oof_fields(Dl, sets, pans, Ps, fset, vnames, fit, seed=0) -> dict:
    """{variant: {panel: [n windows, L]}} out of fold on the train onset windows (week-block folds 0-3, as the v12 decoder
    CV): fit(k, v, fs, tr) returns the model of fold k for variant v."""
    out = {}
    with_e = "e" in T._check_fset(fset)[1]   # as task2y.onset_oof: an "e" recipe trains without the windows it drops
    for v in vnames:
        fs = T._feats(Dl[0], v, fset)
        oof = [np.zeros(len(f)) for f in Dl]
        seen = [np.zeros(len(f), bool) for f in Dl]
        for k in range(4):
            tr = [((f.fold != k) & ((f.elok > 0.5) if (with_e and "elok" in f) else True)).to_numpy() for f in Dl]
            te = [(f.fold == k).to_numpy() for f in Dl]
            if not any(m.any() for m in tr) or not any(m.any() for m in te):
                continue
            mdl = fit(k, v, fs, tr)
            for j, f in enumerate(Dl):
                if te[j].any():
                    oof[j][te[j]] = mdl.predict(T._matrix([f], fs, [te[j]]), num_threads=C.THREADS)
                    seen[j] |= te[j]
            del mdl
            C.free()
        fv = {}
        for j, f in enumerate(Dl):
            p = pans[int(f.panel_code.iat[0])]
            fv[p] = T._field(COND, f, np.where(seen[j], oof[j], 0.0), len(sets[p][COND][0]), Ps[p]["L"])
        out[v] = fv
    return out


def _mean_fields(fields: list) -> dict:
    return {p: np.mean([f[p] for f in fields], 0) for p in fields[0]}


# ----------------------------------------------------------------------------- scoring
def cv_rows(F: dict, sets, pans) -> pd.DataFrame:
    """Per selector-drawn train onset window: the official IoU (eligible cells at T+30; empty union = 1) of field F
    decoded with the run's onset decoder."""
    rows = []
    for p in pans:
        if p not in F or p not in X._EL30 or COND not in sets.get(p, {}):
            continue
        Wc, yc = sets[p][COND][0], sets[p][COND][1]
        el30, _ = X._EL30[p]
        for w in np.flatnonzero((Wc.src == "sim").to_numpy()):
            sel = T.decode_window(COND, F[p][w], len(F[p][w]))[T.K - 1]
            pe = sel & el30[w]
            tru = np.asarray(yc[w][T.K - 1], bool) & el30[w]
            u = int((pe | tru).sum()); it = int((pe & tru).sum())
            rows.append(dict(panel=p, w=int(w), day=int(Wc.day.iat[w]), src="sim", draws=int(Wc.draws.iat[w]),
                             iou=1.0 if u == 0 else it / u, n_pred=int(pe.sum()), n_true=int(tru.sum()), inter=it,
                             tod=float((int(Wc["T"].iat[w]) % C.SLOTS) / 12.0)))
    return pd.DataFrame(rows, columns=["panel", "w", "day", "src", "draws", "iou", "n_pred", "n_true", "inter", "tod"])


def top1(fields: dict, rankp: dict, sets, pans) -> dict:
    """Diagnostic: share of the selector-drawn train windows whose most likely link (out of fold) is truly queued at
    T+30 - per candidate field and for the calibrated ranker alone."""
    out = {}
    for name, F in list(fields.items()) + [("ranker_alone", rankp)]:
        hit = n = 0
        for p in pans:
            if p not in F or COND not in sets.get(p, {}):
                continue
            Wc, yc = sets[p][COND][0], sets[p][COND][1]
            for w in np.flatnonzero((Wc.src == "sim").to_numpy()):
                tru = np.asarray(yc[w][T.K - 1], bool)
                if tru.any():
                    hit += int(tru[int(np.argmax(F[p][w]))]); n += 1
        out[name] = round(hit / n, 4) if n else None
    return out


def pseudo_rows(F: dict, pseudo, pans, Ps, split="validation", day_lo=None) -> pd.DataFrame:
    """task2.pseudo_scores for given fields: IoU on the known cells of every pseudo window (days >= day_lo)."""
    rows = []
    for p in pans:
        ent = pseudo.get(p, {}).get(split, {}).get(COND)
        if ent is None or p not in F:
            continue
        Wc, y, kn, _ = ent
        L = Ps[p]["L"]
        mp = Ps[p].get("mp")
        d = Wc.day.to_numpy()
        for w in range(len(Wc)):
            if day_lo is not None and d[w] < day_lo:
                continue
            pred = T.decode_window(COND, F[p][w], L)
            u = int(((pred | y[w]) & kn[w]).sum()); it = int(((pred & y[w]) & kn[w]).sum())
            ms = (T._onset_miss(pred[T.K - 1], y[w][T.K - 1] & kn[w][T.K - 1], F[p][w], mp) if mp is not None
                  else (np.nan, np.nan, np.nan))
            unk = bool(pred[T.K - 1].any() and not (pred[T.K - 1] & kn[w][T.K - 1]).any())
            rows.append(dict(panel=p, condition=COND, split=split, w=int(w), src=Wc.src.iat[w], day=int(d[w]),
                             draws=int(Wc.draws.iat[w]), iou=1.0 if u == 0 else it / u,
                             n_pred=int((pred & kn[w]).sum()), n_true=int((y[w] & kn[w]).sum()), inter=it,
                             tod=float((int(Wc["T"].iat[w]) % C.SLOTS) / 12.0), miss_links=ms[0], miss_km=ms[1],
                             true_rank=ms[2], pred_hidden=unk))
    return pd.DataFrame(rows, columns=["panel", "condition", "split", "w", "src", "day", "draws", "iou", "n_pred",
                                       "n_true", "inter", "tod", "miss_links", "miss_km", "true_rank", "pred_hidden"])


def two_draw(Rb, Rc, Cb, Cc, n_alt, st, early_day) -> dict:
    """The v5/v6 rule (task2.choose_config) for one candidate against the baseline: Rb/Rc March pseudo rows, Cb/Cc train
    CV rows (official rule)."""
    out, M = T.paired(Rb, Rc, early_day)
    gain = 0.5 * (np.nan_to_num(out["d_sim"]) + out["d_all"])
    se = T._boot_gain(M)
    early_bad = out.get("n_early", 0) >= 10 and out.get("d_early", 0) < -np.nan_to_num(out.get("se_early", 0))
    ok = (np.nan_to_num(out["d_sim"]) >= -np.nan_to_num(out["se_sim"]) and out["d_all"] >= -np.nan_to_num(out["se_all"])
          and not early_bad)
    enough = bool(len(M) >= int(st["min_windows13"]) and M.day.nunique() >= 3)
    fam = C.families()
    pmd = M.groupby("panel").d.mean(); pn = M.groupby("panel").size()
    fd = pmd.groupby(pmd.index.map(fam)).mean(); fn = pn.groupby(pn.index.map(fam)).sum()
    fd = fd[fn.reindex(fd.index).fillna(0) >= 5]
    fam_ok = bool(len(fd) < 2 or float((fd >= -1e-12).mean()) >= float(st["fam_share13"]))
    Mc = Cb.merge(Cc[["panel", "w", "iou"]], on=["panel", "w"], suffixes=("_a", "_b"))
    Mc["d"] = Mc.iou_b - Mc.iou_a
    d_cv = T._agg1(Mc, "d", "draws") if len(Mc) else float("nan")
    se_cv = T._boot_se(Mc, "d", "draws") if len(Mc) else float("nan")
    cv_ok = bool(np.isfinite(d_cv) and d_cv >= -float(st["cv13_z"]) * np.nan_to_num(se_cv))
    inc = 0.0 if n_alt <= 1 else (float(st["z13_alt"]) if n_alt == 2 else 1.5 * float(st["z13_alt"]))
    z, zc = float(st["z13"]) + inc, float(st["z13_cv"]) + inc
    strong_march = bool(gain > z * np.nan_to_num(se) and np.isfinite(d_cv) and d_cv >= 0)
    strong_cv = bool(np.isfinite(se_cv) and se_cv > 0 and d_cv > zc * se_cv and gain >= 0)
    take_two = bool(ok and cv_ok and enough and fam_ok and (strong_march or strong_cv))
    # v13.1 cross-draw rule: the March draw decides (z13x SEs, both March reads positive, the same safety checks); the
    # training-draw CV is reported only
    zx = float(st["z13x"])
    strong_x = bool(np.isfinite(se) and se > 0 and gain > zx * se and np.nan_to_num(out["d_sim"]) > 0
                    and out["d_all"] > 0)
    take_x = bool(ok and enough and fam_ok and strong_x)
    take = take_x if st["rule13"] == "cross_draw" else take_two
    out.update(gain=gain, se_gain=se, z=z, z_cv=zc, d_cv=d_cv, se_cv=se_cv, n_cv=int(len(Mc)), cv_ok=cv_ok,
               eligible=bool(ok), enough=enough, fam_ok=fam_ok,
               fam_d={str(k): round(float(v), 4) for k, v in fd.items()}, strong_march=strong_march,
               strong_cv=strong_cv, take_two_draw=take_two, z_x=zx, strong_x=strong_x, take_cross_draw=take_x,
               rule=st["rule13"], take=take)
    return out


# ----------------------------------------------------------------------------- settings and files
def _norm(d) -> str:
    return json.dumps({k: v for k, v in (d or {}).items() if k not in ("model_dir",)}, sort_keys=True, default=str)


def up_to_date(cfg, cfg13, work=None) -> tuple:
    """(True, report) when the v13 files of this code, these settings, this release and these v12 probabilities exist."""
    d = (work or C.WORK) / "pred"
    rp = d / REPORT_FILE
    if not rp.exists():
        return False, None
    try:
        from .data import fingerprint
        install()
        rep = json.load(open(rp))
        fp = fingerprint()
    except Exception:  # noqa: BLE001
        return False, None
    fm = rep.get("files_md5") or {}
    ok = (rep.get("version") == VERSION and rep.get("code") == code_hash() and _norm(rep.get("cfg")) == _norm(cfg)
          and _norm(rep.get("cfg13")) == _norm(settings(cfg13)) and rep.get("fingerprint") == fp
          and (d / Y.PROBS_FILE).exists() and rep.get("v12_probs_md5") == T._md5(d / Y.PROBS_FILE)
          and set(fm) >= {MAIN_FILE, HEDGE_FILE} | ({PROBE_FILE} if rep.get("probe") else set())
          and all((d / fn).exists() and T._md5(d / fn) == h for fn, h in fm.items()))
    return ok, rep


def load_v12(work=None) -> tuple:
    """(probabilities, report) of the v12 run, checked against each other."""
    d = (work or C.WORK) / "pred"
    pf, rf = d / Y.PROBS_FILE, d / Y.REPORT_FILE
    if not (pf.exists() and rf.exists()):
        raise FileNotFoundError(f"v13 needs pred/{Y.PROBS_FILE} and pred/{Y.REPORT_FILE} from the v12 run")
    rep = json.load(open(rf))
    if rep.get("probs_md5") and rep["probs_md5"] != T._md5(pf):
        raise RuntimeError(f"pred/{Y.PROBS_FILE} is not the file the v12 report describes (an interrupted copy?)")
    with open(pf, "rb") as f:
        pk = pickle.load(f)
    return pk, rep


def _save_probs(path: Path, obj):
    tmp = path.with_suffix(".part")
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=4)
    tmp.replace(path)


def _maxdiff(a: dict, b: dict) -> float:
    """Largest absolute difference between two {panel: {split: {wid: [L]}}} onset field sets over their common windows
    (inf when one misses a window of the other)."""
    m = 0.0
    for p, d in a.items():
        for s, ws in d.items():
            for wid, v in ws.items():
                other = b.get(p, {}).get(s, {}).get(wid)
                if other is None:
                    return float("inf")
                m = max(m, float(np.max(np.abs(np.asarray(v, np.float64) - np.asarray(other, np.float64)))))
    return m


def _onset_part(probs_set: dict) -> dict:
    """The onset windows of a v12 probability set as {panel: {split: {wid: [L]}}}."""
    out = {}
    for p, d in probs_set.items():
        for s, ws in d.items():
            for wid, (c, pr) in ws.items():
                if c == COND:
                    out.setdefault(p, {}).setdefault(s, {})[wid] = np.asarray(pr, np.float64)
    return out


# ----------------------------------------------------------------------------- the run
def run_main(cfg: dict, cfg13: dict | None = None, device: dict | None = None) -> dict:
    """cfg: the v12 run's Task 2 settings (so its tables and checkpoints are rebuilt / reloaded exactly); cfg13: the v13
    settings (DEFAULTS). Writes pred/task2_v13.csv, pred/task2_v13_hedge.csv, pred/task2_v13_probs.pkl and
    pred/task2_v13_report.json (last). When no candidate is adopted the v13 files equal the v12 files."""
    try:
        return _run_main(cfg, cfg13, device)
    finally:
        Y.DEC["mode"] = "topm"


def _run_main(cfg: dict, cfg13: dict | None, device: dict | None) -> dict:
    from .data import fingerprint
    install()
    t0 = time.time()
    st = settings(cfg13)
    pk, rep12 = load_v12()
    probs12 = pk["probs"]
    if _norm(rep12.get("cfg")) != _norm(cfg):
        raise RuntimeError("v13 must run with the v12 run's Task 2 settings (CFG['T2']), so its tables and models reload")
    Y.DEC.update(mode=pk.get("mode", "topm"), rho=pk.get("rho", Y.DEC["rho"]), n=pk.get("n", Y.DEC["n"]),
                 M=pk.get("M", Y.DEC["M"]), seed=pk.get("seed", Y.DEC["seed"]))
    tcfg = Y.task_cfg(cfg)
    TM.use_device(device)
    T.set_checkpoints(tcfg)
    # the onset tables only (the April pseudo windows and the ongoing tables are not needed): the window draw consumes the
    # generator the same way, so the tables are v12's and every checkpoint (standard, out-of-fold, adapted) reloads
    bcfg = dict(tcfg, late_adapt=False, late_early={}, pseudo_april=False, online_adapt=False)
    pans = C.task2_panels()
    rng = np.random.default_rng(tcfg.get("seed", 0))
    imputer = T.fit_imputer(pans) if tcfg.get("imputer", True) else None
    sets, offis, Ps, pseudo = {}, {}, {}, {}
    for i, p in enumerate(pans):
        P, out, offi, ps = T.build_panel_sets(p, imputer, rng, bcfg, skip=("queue_ongoing",), skip_april=(COND,))
        for split in ps:
            for cond in ps[split]:
                ps[split][cond][3]["panel_code"] = np.float32(i)
        sets[p] = out; offis[p] = offi; pseudo[p] = ps
        Ps[p] = dict(L=P.L, links=P.net.links, mp=np.asarray(P.mp, np.float64))
        P.dprof = None
        del P
        C.free()
    C.log(f"T2 {VERSION} onset tables ready ({(time.time() - t0) / 60:.1f} min)")
    report = dict(version=VERSION, code=code_hash(), fingerprint=fingerprint(), cfg=cfg, cfg13=st,
                  v12_probs_md5=T._md5(C.WORK / "pred" / Y.PROBS_FILE), decoder=Y.DEC["mode"], checks={}, minutes={})
    Dl = T._frames(sets, pans, COND)
    if not Dl:
        raise RuntimeError("T2 v13: no onset training windows")
    prm, rounds = T.ONSET_P, int(tcfg.get("onset_rounds", 500))
    rrounds = int(st["rank_rounds"] or rounds)
    seeds = list(tcfg.get("onset_seeds", [tcfg.get("seed", 0)]))
    rseeds = list(st["rank_seeds"])
    fset, vnames = rep12["chosen"][COND][0], list(rep12["chosen"][COND][1])
    if sorted(vnames) != ["all", "noloc"]:
        raise RuntimeError(f"T2 v13 expects the onset variants all + noloc (v12 chose {vnames})")
    variants = T._variants(vnames)
    report["recipe"] = [fset, vnames]
    has_march = any(COND in pseudo[p].get("validation", {}) for p in pans)
    if not has_march:
        raise RuntimeError("T2 v13: no March onset pseudo windows")
    # ---- the v12 standard models (reloaded), per variant, checked against v12's stored onset probabilities
    n_fit0 = TM.DEVICE["fits_cpu"] + TM.DEVICE["fits_gpu"]
    std = T.fit_set(Dl, COND, prm, rounds, variants, seeds, fset=fset)
    if TM.DEVICE["fits_cpu"] + TM.DEVICE["fits_gpu"] > n_fit0:
        raise RuntimeError("T2 v13: v12's standard onset models were not all found in t2_models (some were refitted); "
                           "is CFG['DRIVE_WORK'] the v12 run's folder?")
    by_var = {v: T._models_for(std, variants, seeds, [v]) for v in vnames}
    ofa = official_fields(by_var["all"], offis, pans, C.TEST_SPLITS)
    ofn = official_fields(by_var["noloc"], offis, pans, C.TEST_SPLITS)
    obase = {p: {s: {w: 0.5 * (ofa[p][s][w] + ofn[p][s][w]) for w in ws} for s, ws in d.items()} for p, d in ofa.items()}
    std12 = _onset_part(probs12["std"])
    rep_std = max(_maxdiff(std12, obase), _maxdiff(obase, std12))
    report["checks"]["standard onset fields vs v12 (max abs diff)"] = rep_std
    report["reload"] = "exact" if rep_std <= 1e-6 else ("close" if rep_std <= float(st["reload_tol"]) else "mismatch")
    if report["reload"] == "mismatch":
        raise RuntimeError(f"T2 v13: the reloaded onset models do not reproduce v12's probabilities (max diff {rep_std}); "
                           "were the t2_models checkpoints or the v12 settings changed?")
    if report["reload"] == "close":   # e.g. another thread count; the gate compares recomputed fields with each other
        C.log(f"T2 v13: the reloaded onset fields differ from v12's by up to {rep_std:.2e} (changes are applied to the "
              "stored v12 fields)")
    report["minutes"]["standard"] = round((time.time() - t0) / 60, 1)
    # ---- out of fold on train (seed 0): the classifier per variant (v12's checkpoints) and the ranker
    sd0 = int(tcfg.get("seed", 0))
    oof_c = oof_fields(Dl, sets, pans, Ps, fset, vnames,
                       lambda k, v, fs, tr: Y._fit_cached(Dl, COND, prm, rounds, fs, fset, v, sd0, f"exp:oof{k}", tr), sd0)
    oof_r = oof_fields(Dl, sets, pans, Ps, fset, vnames,
                       lambda k, v, fs, tr: _rank_cached(Dl, fs, rrounds, fset, v, sd0, f"exp:rk{RK_TAG}oof{k}", tr), sd0)
    s_oof = _mean_fields([oof_r[v] for v in vnames])
    cal = fit_calib(s_oof, sets, pans)   # the ranker's scores -> probabilities, on the training draw only
    report["rank_calibration"] = cal
    C.log(f"T2 {VERSION} ranker temperature (out of fold, train): {cal}")
    report["minutes"]["oof"] = round((time.time() - t0) / 60, 1)
    # ---- train-only ranker (March rows, the hedge's April rows) and the March pseudo fields
    rk_std = rank_set(Dl, fset, vnames, rseeds, rrounds, "exp")
    pfa = pseudo_fields(by_var["all"], pseudo, pans, Ps)
    pfn = pseudo_fields(by_var["noloc"], pseudo, pans, Ps)
    ps_ = pseudo_fields(rk_std, pseudo, pans, Ps)
    report["minutes"]["ranker"] = round((time.time() - t0) / 60, 1)
    # ---- the gate
    cvF = {c: {p: combine(c, oof_c["all"][p], oof_c["noloc"][p], s_oof[p], st, cal) for p in s_oof} for c in ("base",) + CANDS}
    mF = {c: {p: combine(c, pfa[p], pfn[p], ps_[p], st, cal) for p in pfa} for c in ("base",) + CANDS}
    Cv = {c: cv_rows(cvF[c], sets, pans) for c in cvF}
    Rm = {c: pseudo_rows(mF[c], pseudo, pans, Ps) for c in mF}
    d0 = int(C.split_days("validation").min())
    early = d0 + int(tcfg.get("early_days", 7))
    gate = dict(baseline=dict(march=T.pseudo_report(Rm["base"]), cv=T._agg1(Cv["base"], "iou", "draws"),
                              diag_march=T.window_diag(Rm["base"], COND), cv_windows=int(len(Cv["base"]))),
                candidates={}, top1_cv=top1(cvF, {p: softmax_t(s_oof[p], cal["tau"]) for p in s_oof}, sets, pans))
    best, best_key = "base", None
    if not cal.get("informative"):
        gate["rk50_excluded"] = "the ranker does not separate the queued links on the training draw (calibration)"
    gate["rule"] = st["rule13"]
    if st["rule13"] == "cross_draw":
        gate["rule_note"] = RULE_NOTE
    for c in CANDS:
        g = two_draw(Rm["base"], Rm[c], Cv["base"], Cv[c], len(CANDS), st, early)
        if c == "rk50" and not cal.get("informative"):
            g["take"] = g["take_two_draw"] = g["take_cross_draw"] = False
        g["march"] = T.pseudo_report(Rm[c]); g["cv"] = T._agg1(Cv[c], "iou", "draws")
        g["diag_march"] = T.window_diag(Rm[c], COND)
        gate["candidates"][c] = g
        key_ = (g["gain"], np.nan_to_num(g["d_cv"]))
        if g["take"] and (best_key is None or key_ > best_key):
            best, best_key = c, key_
    if st["force13"]:   # tests only
        best, gate["forced"] = st["force13"], True
    gate["adopted"] = best
    report["gate"] = gate
    C.log(f"T2 {VERSION} onset gate ({st['rule13']}): adopted {best} | " + " | ".join(
        f"{c}: gain {g['gain']:+.4f} se {g['se_gain']:.4f} cv {g['d_cv']:+.4f} se {g['se_cv']:.4f} "
        f"two-draw {g['take_two_draw']} cross-draw {g['take_cross_draw']}"
        for c, g in gate["candidates"].items()))
    report["minutes"]["gate"] = round((time.time() - t0) / 60, 1)
    # ---- the new probability sets (unchanged when nothing is adopted)
    probs = copy.deepcopy(probs12)
    main_rep = dict(applied=False)
    ors = official_fields(rk_std, offis, pans, C.TEST_SPLITS)
    if best != "base":
        n_std = 0
        for p, d in std12.items():
            for s, ws in d.items():
                for wid, b12 in ws.items():
                    new = apply_cand(best, b12, ofa[p][s][wid], ofn[p][s][wid], ors[p][s][wid], st, cal=cal)
                    probs["std"][p][s][wid] = (COND, new)
                    n_std += 1
        report["checks"]["onset windows recomputed (standard: March + April hedge)"] = n_std
        apr12 = _onset_part(probs12["april"])
        if apr12 and st["main_check"]:
            main_rep = main_april(best, Dl, sets, pseudo, offis, pans, Ps, prm, rounds, rrounds, variants, vnames, seeds,
                                  rseeds, tcfg, fset, rep12, st, apr12, probs, cal)
        elif apr12:
            main_rep = dict(applied=False, reason="main_check=False: the main file keeps v12's April onset rows")
        else:
            main_rep = dict(applied=False, reason="v12 adopted no April onset refit: the main file's April onset rows "
                                                  "are the standard ones (changed above)")
        for k in ("late", "early"):
            if _onset_part(probs12[k]):
                main_rep.setdefault("kept_v12", []).append(k)   # v12 adopted none on the real data; left as v12's
    report["main_april"] = main_rep
    report["minutes"]["probs"] = round((time.time() - t0) / 60, 1)
    # ---- files
    d = C.WORK / "pred"
    _save_probs(d / PROBS_FILE, dict(pk, probs=probs, v13=dict(adopted=best, settings=st)))
    probe = None
    with Y.decoder(Y.DEC["mode"]):
        main = T.write_rows([probs["early"], probs["late"], probs["april"], probs["std"]], offis, pans, Ps)
        hedge = T.write_rows([probs["std"]], offis, pans, Ps)
        if best == "base" and cal.get("informative") and st["probe13"]:   # the one public read left: the ranker
            pr_set = {p: {"validation": {wid: (COND, apply_cand("rk50", b12, None, None, ors[p]["validation"][wid], st, cal=cal))
                                         for wid, b12 in dd.get("validation", {}).items()}} for p, dd in std12.items()}
            probe = T.write_rows([pr_set, probs["early"], probs["late"], probs["april"], probs["std"]], offis, pans, Ps)
    a = main.set_index(KEY).queue_pred
    h = hedge.set_index(KEY).queue_pred
    TM.check_rows(a, "the v13 main file"); TM.check_rows(h, "the v13 hedge file")
    ids = X._ids(offis, pans)
    mar = a.index.get_level_values(0).str.contains("_validation_")
    if not (a[mar] == h.reindex(a.index)[mar]).all():
        raise RuntimeError("T2 v13: the hedge's March rows must be the main file's")
    checks = {}
    for fn, nm, new in ((Y.MAIN_FILE, "v12 main", a), (Y.HEDGE_FILE, "v12 hedge", h)):
        f = d / fn
        if not f.exists():
            raise FileNotFoundError(f"T2 v13: pred/{fn} (the v12 file) is needed for the checks")
        old = pd.read_csv(f, dtype=QDT).set_index(KEY).queue_pred
        for c in T.CONDS:
            for s, m in (("validation", "March"), ("private", "April")):
                checks[f"{m} {c} rows ({nm.split()[1]}) vs the {nm} file"] = TM._compare(new, old, ids[c][s])
        og = sum(checks[f"{m} queue_ongoing rows ({nm.split()[1]}) vs the {nm} file"]["cells_differ"] for m in ("March", "April"))
        if og:
            raise RuntimeError(f"T2 v13: {og} ongoing cells differ from the {nm} file (v13 changes onset rows only)")
        if best == "base":
            on = sum(checks[f"{m} queue_onset rows ({nm.split()[1]}) vs the {nm} file"]["cells_differ"] for m in ("March", "April"))
            if on:
                raise RuntimeError(f"T2 v13: nothing was adopted, yet {on} onset cells differ from the {nm} file")
    if probe is not None:
        pb = probe.set_index(KEY).queue_pred
        TM.check_rows(pb, "the v13 ranker probe")
        on_mar = pb.index.get_level_values(0).isin(list(ids[COND]["validation"]))
        if not (pb[~on_mar] == a.reindex(pb.index)[~on_mar]).all():
            raise RuntimeError("T2 v13: the ranker probe may differ from the main file on the March onset windows only")
        checks["March queue_onset rows (ranker probe) vs the v12 main file"] = TM._compare(
            pb, pd.read_csv(d / Y.MAIN_FILE, dtype=QDT).set_index(KEY).queue_pred, ids[COND]["validation"])
    report["checks"].update(checks)
    report["onset_cells"] = {nm: {m: Y.onset_cells(r, ids[COND][s]) for s, m in (("validation", "March"), ("private", "April"))}
                             for nm, r in (("v13 main", a), ("v13 hedge", h))}
    report["agreement_main_hedge"] = {m: dict(cells_main=int(a[sel].sum()), cells_hedge=int(h.reindex(a.index)[sel].sum()),
                                              differ=int((a[sel] != h.reindex(a.index)[sel]).sum()))
                                      for m, sel in (("March", mar), ("April", ~mar))}
    for fr, fn in ((main, MAIN_FILE), (hedge, HEDGE_FILE), (probe, PROBE_FILE)):
        if fr is not None:
            X._write(fr, d, fn)
    if probe is None:
        (d / PROBE_FILE).unlink(missing_ok=True)
    report["probe"] = PROBE_FILE if probe is not None else None
    report["changed"] = best != "base"
    report["device"] = dict(params=dict(device or {}), failed=TM.DEVICE["failed"], fits_gpu=TM.DEVICE["fits_gpu"],
                            fits_cpu=TM.DEVICE["fits_cpu"])
    report["minutes"]["total"] = round((time.time() - t0) / 60, 1)
    report["files_md5"] = {fn: T._md5(d / fn) for fn in (MAIN_FILE, HEDGE_FILE) + ((PROBE_FILE,) if probe is not None else ())}
    report["probs_md5"] = T._md5(d / PROBS_FILE)
    tmp = d / (REPORT_FILE + ".part")
    json.dump(report, open(tmp, "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    tmp.replace(d / REPORT_FILE)   # written last: it vouches for the files
    C.log(f"T2 {VERSION} done in {report['minutes']['total']} min: adopted {best} | main April {main_rep} | "
          f"checks {json.dumps({k: v for k, v in checks.items() if 'onset' in k})[:600]}")
    return report


def main_april(best, Dl, sets, pseudo, offis, pans, Ps, prm, rounds, rrounds, variants, vnames, seeds, rseeds, tcfg,
               fset, rep12, st, apr12, probs, cal) -> dict:
    """The adopted candidate on top of the v12 April onset refit of the main file (as a change of v12's stored April
    fields) - applied only when, within March, the same change on the gate models (train + March before the cut) does not
    lose on the March windows after the cut."""
    out = dict(applied=False)
    alpha = float(tcfg.get("adapt_alpha", 2.0))
    vd = C.split_days("validation")
    cut = int(vd.min() + int(round(len(vd) * tcfg.get("adapt_frac", 0.5))))
    lf = bool((rep12.get("locfree") or {}).get(COND, {}).get("adopted"))
    vlf = T._variants([(rep12.get("locfree") or {}).get(COND, {}).get("variant", X.LOCFREE)]) if lf else None
    out.update(cut_day=cut, locfree=lf)
    # within-March check on the gate models (single seed, v12's checkpoints) and a gate ranker
    Dg = T.pseudo_train_rows(pseudo, pans, "validation", COND, day_hi=cut)
    if Dg is None:
        out["reason"] = "no March pseudo windows before the cut"
        return out
    nfit = lambda: TM.DEVICE["fits_cpu"] + TM.DEVICE["fits_gpu"]   # v13.1: refits (missing checkpoints) are reported
    n0 = nfit()
    g_full = T.fit_set(Dl, COND, prm, rounds, variants, seeds[:1], Dg, alpha, role="gate", fset=fset)
    gfa = pseudo_fields(T._models_for(g_full, variants, seeds[:1], ["all"]), pseudo, pans, Ps)
    gfn = pseudo_fields(T._models_for(g_full, variants, seeds[:1], ["noloc"]), pseudo, pans, Ps)
    glf = None
    if lf:
        g_lf = T.fit_set(Dl, COND, prm, rounds, vlf, seeds[:1], Dg, alpha, role="gate", fset=fset)
        glf = pseudo_fields(g_lf, pseudo, pans, Ps)
    out["gate_models_refitted"] = nfit() - n0
    if best == "rk50":   # nl75 needs no ranker scores (v13.1: no gate ranker is fitted for it)
        wg = np.r_[np.ones(sum(len(f) for f in Dl)), np.full(len(Dg), alpha)]
        rk_g = rank_set(list(Dl) + [Dg], fset, vnames, rseeds[:1], rrounds, "gate", weight=wg)
        gs = pseudo_fields(rk_g, pseudo, pans, Ps)
    else:
        gs = {p: None for p in gfa}
    del Dg
    C.free()

    def main_field(c, fa, fn, flf, s):
        full = combine("base", fa, fn, s, st)
        b = 0.5 * full + 0.5 * flf if flf is not None else full
        return b if c == "base" else apply_cand(c, b, fa, fn, s, st, scale=0.5 if flf is not None else 1.0, cal=cal)

    Fb = {p: main_field("base", gfa[p], gfn[p], None if glf is None else glf[p], gs[p]) for p in gfa}
    Fc = {p: main_field(best, gfa[p], gfn[p], None if glf is None else glf[p], gs[p]) for p in gfa}
    Rb = pseudo_rows(Fb, pseudo, pans, Ps, day_lo=cut)
    Rc = pseudo_rows(Fc, pseudo, pans, Ps, day_lo=cut)
    if len(Rb) == 0:
        out["reason"] = "no March pseudo windows after the cut"
        return out
    pst, _ = T.paired(Rb, Rc)
    out["march_second_half"] = {k: pst[k] for k in ("n_sim", "n_all", "S_a_sim", "S_b_sim", "d_sim", "se_sim", "d_all",
                                                     "se_all", "better", "worse")}
    ok = bool(np.nan_to_num(pst["d_sim"]) >= -np.nan_to_num(pst["se_sim"]) and pst["d_all"] >= -np.nan_to_num(pst["se_all"]))
    if st["force13"]:
        ok, out["forced"] = True, True
    out["passes"] = ok
    if not ok:
        out["reason"] = "the change loses within March on the April-refit models: the main file keeps v12's April onset rows"
        return out
    # the April fields, as changes of v12's stored April onset fields: the ranker refitted on train + all of March (rk50),
    # or v12's April refit per variant (reloaded) for the variant weights (nl75; half of the field when v12 blended it
    # with the location-free refit)
    Df = T.pseudo_train_rows(pseudo, pans, "validation", COND)
    afa = afn = ars = None
    if best == "nl75":
        n1 = nfit()
        ad = T.fit_set(Dl, COND, prm, rounds, variants, seeds, Df, alpha, role="adapt", fset=fset)
        out["adapt_models_refitted"] = nfit() - n1   # 0 = v12's April refit reloaded from its checkpoints
        afa = official_fields(T._models_for(ad, variants, seeds, ["all"]), offis, pans, ("private",))
        afn = official_fields(T._models_for(ad, variants, seeds, ["noloc"]), offis, pans, ("private",))
        del ad
    else:
        wf = np.r_[np.ones(sum(len(f) for f in Dl)), np.full(len(Df), alpha)]
        rk_a = rank_set(list(Dl) + [Df], fset, vnames, rseeds, rrounds, "adapt", weight=wf)
        ars = official_fields(rk_a, offis, pans, ("private",))
        del rk_a
    del Df
    C.free()
    n = 0
    for p, dd in apr12.items():
        for s, ws in dd.items():
            for wid, b12 in ws.items():
                if best == "nl75":
                    new = apply_cand(best, b12, afa[p][s][wid], afn[p][s][wid], None, st, scale=0.5 if lf else 1.0)
                else:
                    new = apply_cand(best, b12, None, None, ars[p][s][wid], st, cal=cal)
                probs["april"][p][s][wid] = (COND, new)
                n += 1
    out.update(applied=True, windows=n)
    return out
