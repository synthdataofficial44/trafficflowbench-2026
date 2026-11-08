"""v11 Task 2: two new feature groups behind the two-draw gate, a location-free April refit, a hedge that keeps the
month memory, and a public probe of the onset queue size.

Everything here sits on top of task2.py without editing it (the v10 model checkpoints in Drive stay valid and are
reloaded); `install()` adds the new pieces to the module:
  * group "e" (av_*): per-link detector availability up to T - the origin slot, the hour of history, the 3 h before
    it, the day so far and the previous 24 h, the trailing run of the current status, neighbours. Only eligible cells
    are scored, and an official window needs an eligible queued cell in its horizon (the scoring spec rules out
    empty-empty free marks), so a site whose detectors are down is unlikely to be the scored onset site. With "e" the
    onset training windows are also aligned with that rule: a training window whose T+30 queue sits only on
    ineligible cells is dropped (it could not be an official window). The availability flags of horizon cells are never
    read for a window being forecast; the filter reads them on train only, as a training label.
  * group "c" (ecap_*): load against an empirical capacity - the 99.5th percentile of each link's free-flow flow on
    train (capacity is the same in every month while each month re-deals the ramp demand) - now, extrapolated to T+30,
    arriving from upstream, against the tightest capacity within three links downstream, and their ranks within the
    window. Only ratios enter (no static per-link value that could act as a link identifier).
  * onset: the April refit adds a location-free variant (no link-identifying columns, no ramp-flow levels) blended
    50/50 with the full refit. The v10 gate (early March -> late March) measures a same-month gain; April is another
    demand draw, and only the location-free part can't have memorised March's bottlenecks. The blend replaces the full
    refit in the main file when it passes the same gate.
Files: main (as v10: refits override the standard rows), hedge = the standard (train-only) models with the month
memory for both months (its March rows are the main file's, so the two picks differ only on the April refit bet),
size probe = the main file with every March onset window cut to its single most likely cell.
Only data timestamped <= T feeds a window at origin T; no April label is evaluated or used to choose anything.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from . import common as C
from . import task2 as T
from . import task2m as TM

VERSION = "v11"
MAIN_FILE, HEDGE_FILE, SIZE_FILE = "task2_v11.csv", "task2_v11_hedge.csv", "task2_v11_size.csv"
MAIN_TOP1, HEDGE_TOP1 = "task2_v11_top1.csv", "task2_v11_hedge_top1.csv"   # every onset window at its top cell (both months)
REPORT_FILE = "task2_v11_report.json"
V10_MAIN, V10_HEDGE = "task2_v10.csv", "task2_v10_hedge.csv"
NEW_GROUPS = {"e": "av_", "c": "ecap_"}
NEW_PREFIXES = tuple(NEW_GROUPS.values())
LOCFREE = "noloc2_nr"
AV_CTX = 36           # masked-view slots before the history (3 h) for the availability context
ECAP_Q = 99.5         # percentile of the free-flow flow taken as a link's empirical capacity
ECAP_FREE = 0.85      # free flow: speed >= this share of the free speed
DEFAULT_CANDIDATES = {
    "queue_onset": [["v2+pm", ["all", "noloc"]], ["v2+pme", ["all", "noloc"]], ["v2+pmc", ["all", "noloc"]],
                    ["v2+pmce", ["all", "noloc"]]],
    "queue_ongoing": [["v2+m", ["all", "noloc"]], ["v2+me", ["all", "noloc"]]],
}
KEY = TM.KEY
QDT = TM.QDT
_ORIG: dict = {}
_EL30: dict = {}     # panel -> (eligible at T+30 [n, L], window has an eligible queued cell at T+30 [n]) of the onset train windows


# ----------------------------------------------------------------------------- installation into task2
def _unwrap(f):
    return getattr(f, "_v11_orig", f)


def _installed() -> bool:
    return (T.window_features is window_features and T.build_panel_sets is build_panel_sets and T.fit_set is fit_set
            and T._ckpt_path is _ckpt_path and T.cv_windows is cv_windows
            and all(T.EXTRA.get(k) == v for k, v in NEW_GROUPS.items()) and "elok" in T.NONF)


def install():
    """Add the v11 groups to task2 (idempotent, and safe after a reload of task2 and/or of this module: the originals
    are found by unwrapping whatever task2 holds). Feature sets without "e"/"c" get exactly their v10 columns, so their
    checkpoints keep their keys."""
    if _installed():
        return
    _ORIG.update(wf=_unwrap(T.window_features), bps=_unwrap(T.build_panel_sets), fs=_unwrap(T.fit_set),
                 ck=_unwrap(T._ckpt_path), cv=_unwrap(T.cv_windows))
    T.EXTRA = dict(T.EXTRA, **NEW_GROUPS)
    if "elok" not in T.NONF:
        T.NONF = tuple(T.NONF) + ("elok",)
    for name, f, key in (("window_features", window_features, "wf"), ("build_panel_sets", build_panel_sets, "bps"),
                         ("fit_set", fit_set, "fs"), ("_ckpt_path", _ckpt_path, "ck"), ("cv_windows", cv_windows, "cv")):
        f._v11_orig = _ORIG[key]
        setattr(T, name, f)


def x_hash() -> str:
    return hashlib.md5(Path(__file__).read_bytes()).hexdigest()[:12]


def code_hash() -> str:
    """Task 2's code hash + this file's + task2m's (a stored v11 result is reused only for the same code)."""
    h = hashlib.md5((T.code_hash() + "|").encode())
    h.update(Path(__file__).read_bytes())
    h.update((Path(__file__).parent / "task2m.py").read_bytes())
    return h.hexdigest()[:16]


def _groups(extra) -> str:
    if not extra:
        return ""
    return "rd" if extra is True else str(extra)


def _has_new(fset) -> bool:
    return bool(set(T._check_fset(fset)[1]) & set(NEW_GROUPS))


def _ckpt_path(cond, role, noloc, seed, rounds, fset, fs=None):
    """v10's checkpoint path; models that use a v11 column also carry this file's hash (a change of the new
    features or of the onset filter never reloads a stale model)."""
    p = _ORIG["ck"](cond, role, noloc, seed, rounds, fset, fs)
    if p is not None and fs is not None and any(f.startswith(NEW_PREFIXES) for f in fs):
        tag = hashlib.md5(f"{x_hash()}|{p.name}".encode()).hexdigest()[:8]
        p = p.with_name(f"{p.stem}_x{tag}{p.suffix}")
    return p


def fit_set(Dl, cond, prm, rounds, variants, seeds, extra=None, alpha=1.0, role="std", fset="v3"):
    """task2.fit_set; with group "e" the onset training windows whose T+30 queue has no eligible cell are left out
    (they could not be official windows). Extra (pseudo) rows are labelled on eligible cells already."""
    if cond == "queue_onset" and "e" in T._check_fset(fset)[1] and Dl and all("elok" in f for f in Dl):
        Dl = [f[f.elok.to_numpy() > 0.5] for f in Dl]
        Dl = [f for f in Dl if len(f)]
    return _ORIG["fs"](Dl, cond, prm, rounds, variants, seeds, extra, alpha, role, fset)


def build_panel_sets(panel, imputer, rng, cfg, skip=(), skip_april=()):
    """task2.build_panel_sets + the onset training windows' `elok` flag (an eligible cell queued at T+30)."""
    P, out, offi, pseudo = _ORIG["bps"](panel, imputer, rng, cfg, skip, skip_april)
    _EL30.pop(panel, None)
    if "queue_onset" in out:
        Wc, yc, D = out["queue_onset"]
        if len(Wc):
            t30 = np.clip(Wc["T"].to_numpy(np.int64) + T.K, 0, P.NT - 1)
            el30 = np.asarray(P.el[t30], bool)
            ok = (np.asarray(yc[:, T.K - 1, :], bool) & el30).any(1)
            D["elok"] = ok[D.wi.to_numpy().astype(np.int64)].astype(np.float32)
            Wc["elok"] = ok
            _EL30[panel] = (el30, ok)
    return P, out, offi, pseudo


def cv_windows(Dl, sets, pans, Ps, cond, prm, rounds, configs, fset, seed=0):
    """task2.cv_windows; for the onset aligned with the official scoring: only train windows with an eligible queued
    cell at T+30 are scored, and only on the cells eligible at T+30 (for every candidate alike, so the comparison stays
    paired), and the "e" recipes are fitted without the windows they drop. Ongoing: unchanged."""
    if cond != "queue_onset" or not all(p in _EL30 for p in pans if cond in sets.get(p, {})):
        return _ORIG["cv"](Dl, sets, pans, Ps, cond, prm, rounds, configs, fset, seed)
    K_ = T.K
    names = sorted({v for vn in configs for v in vn})
    with_e = "e" in T._check_fset(fset)[1]
    oof = {v: [np.full(len(f), np.nan) for f in Dl] for v in names}
    for k in range(4):
        tr = [((f.fold != k) & ((f.elok > 0.5) if (with_e and "elok" in f) else True)).to_numpy() for f in Dl]
        te = [(f.fold == k).to_numpy() for f in Dl]
        if not any(m.any() for m in tr) or not any(m.any() for m in te):
            continue
        for v in names:
            fs = T._feats(Dl[0], v, fset)
            mdl = T.fit_lgb(Dl, prm, rounds, fs, None, seed=seed, masks=tr)
            for j, f in enumerate(Dl):
                if te[j].any():
                    oof[v][j][te[j]] = mdl.predict(T._matrix([f], fs, [te[j]]), num_threads=C.THREADS)
            del mdl
            C.free()
    out = {}
    for vn in configs:
        rows = []
        for j, f in enumerate(Dl):
            p = pans[int(f.panel_code.iat[0])]
            Wc, yc = sets[p][cond][0], sets[p][cond][1]
            el30, ok = _EL30[p]
            pr = np.nan_to_num(np.mean([oof[v][j] for v in vn], 0))
            F = T._field(cond, f, pr, len(Wc), Ps[p]["L"])
            mp = Ps[p].get("mp")
            for w in np.flatnonzero((Wc.src == "sim").to_numpy() & ok):
                pred = T.decode_window(cond, F[w], Ps[p]["L"])[K_ - 1] & el30[w]
                tru = np.asarray(yc[w][K_ - 1], bool) & el30[w]
                ms = T._onset_miss(pred, tru, F[w], mp) if mp is not None else (np.nan, np.nan, np.nan)
                u = int((pred | tru).sum())
                rows.append(dict(panel=p, w=int(w), day=int(Wc.day.iat[w]), src="sim", draws=int(Wc.draws.iat[w]),
                                 iou=1.0 if u == 0 else float((pred & tru).sum() / u), n_pred=int(pred.sum()),
                                 n_true=int(tru.sum()), inter=int((pred & tru).sum()),
                                 tod=float((int(Wc["T"].iat[w]) % C.SLOTS) / 12.0),
                                 miss_links=ms[0], miss_km=ms[1], true_rank=ms[2]))
        out[tuple(vn)] = pd.DataFrame(rows, columns=["panel", "w", "day", "src", "draws", "iou", "n_pred", "n_true",
                                                     "inter", "tod", "miss_links", "miss_km", "true_rank"])
    return out


def window_features(P, prof, Ts, A, folds, onset_prior=None, use_T=True, extra=False):
    """task2.window_features + the v11 groups (appended after every v10 column)."""
    df, step = _ORIG["wf"](P, prof, Ts, A, folds, onset_prior, use_T, extra)
    g = _groups(extra)
    n = len(np.asarray(Ts))
    new = {}
    if "e" in g:
        new.update(av_features(P, Ts, A))
    if "c" in g:
        new.update(ecap_features(P, df, n, folds))
    if new:
        df = pd.concat([df, pd.DataFrame({k: np.asarray(v, np.float32).reshape(-1) for k, v in new.items()},
                                         index=df.index)], axis=1)
    return df, step


# ----------------------------------------------------------------------------- group "e": detector availability
def _av_prep(P):
    """Cumulative counts of available (eligible) cells and of known rows; the rows of the official windows' horizon +
    buffer are unknown in every split (the release blanks them in the test months)."""
    if getattr(P, "_av", None) is None:
        known = ~np.asarray(P.dark, bool)
        e = (np.asarray(P.el, bool) & known[:, None]).astype(np.int32)
        P._av = (np.vstack([np.zeros((1, P.L), np.int32), np.cumsum(e, 0, dtype=np.int32)]),
                 np.concatenate([[0], np.cumsum(known.astype(np.int32))]))
    return P._av


def av_features(P, Ts, A) -> dict:
    """[n, L] availability features of windows at origins Ts; history from A['he'] (the window history), the origin
    slot and older context from the released masked view. Nothing after T."""
    Ts = np.asarray(Ts, np.int64)
    n, L = len(Ts), P.L
    ce, cn = _av_prep(P)
    dark = np.asarray(P.dark, bool)
    he = np.asarray(A["he"], np.float32)
    eT = np.where(dark[Ts][:, None], np.nan, np.asarray(P.el[Ts], np.float32))

    def frac(a, b):
        a = np.clip(a, 0, P.NT); b = np.clip(np.maximum(b, a), 0, P.NT)
        kn = (cn[b] - cn[a]).astype(np.float32)[:, None]
        return np.where(kn > 0, (ce[b] - ce[a]).astype(np.float32) / np.maximum(kn, 1), np.nan)

    out = {"av_T": eT, "av_h": he.mean(1), "av_h3": he[:, -3:].mean(1)}
    out["av_c36"] = frac(Ts - T.HH - AV_CTX, Ts - T.HH)
    day0 = (Ts // C.SLOTS) * C.SLOTS
    out["av_day"] = frac(day0, Ts - T.HH)
    out["av_24h"] = frac(Ts - C.SLOTS, Ts - T.HH)
    last = np.where(np.isnan(eT), he[:, -1, :], eT)
    seq = np.concatenate([he, last[:, None, :]], 1)
    rev = (seq == last[:, None, :])[:, ::-1, :]
    run = np.where(rev.all(1), seq.shape[1], np.argmin(rev, axis=1)).astype(np.float32)
    out["av_run"] = np.where(last > 0.5, run, -run)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out["av_nb"] = np.nanmean(np.stack([T._shift_nan(out["av_h3"], o) for o in (-2, -1, 1, 2)], 0), 0)
    out["av_up1"] = T._shift_nan(out["av_h3"], -1)
    out["av_dn1"] = T._shift_nan(out["av_h3"], 1)
    return {k: np.asarray(v, np.float32) for k, v in out.items()}


# ----------------------------------------------------------------------------- group "c": empirical capacity
def emp_capacity(P, fold: int = -1) -> np.ndarray:
    """Per link: the ECAP_Q percentile of its free-flow flow on train (total vph) - for a training window of week-block
    fold f (0-3) without that fold's days (the location priors are out of fold the same way), for every other window
    (fold -1) on all train days; the released capacity where a link has fewer than a day of free-flow cells."""
    cache = getattr(P, "_ecap", None)
    if cache is None:
        cache = P._ecap = {}
    fold = int(fold)
    if fold not in cache:
        v, q = P.tv, P.tq
        free = np.isfinite(v) & np.isfinite(q) & (v >= ECAP_FREE * P.vf[None, :])
        if fold >= 0:
            day = np.arange(len(v)) // C.SLOTS
            free &= ((day // 7) % T.Prof.NF != fold)[:, None]
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            e = np.nanpercentile(np.where(free, q, np.nan).astype(np.float32), ECAP_Q, axis=0)
        e = np.where((free.sum(0) >= C.SLOTS) & np.isfinite(e) & (e > 0), e, P.cap)
        cache[fold] = np.asarray(e, np.float32)
    return cache[fold]


def ecap_features(P, df: pd.DataFrame, n: int, folds=None) -> dict:
    """[n, L] load against the empirical capacity (from v10 columns of `df`: flows relative to the released capacity);
    each window takes the capacity of its fold (train windows) or of all train days (-1: every other window)."""
    L = P.L
    fo = np.full(n, -1, np.int64) if folds is None else np.asarray(folds, np.int64).reshape(-1)
    e = np.stack([emp_capacity(P, f) for f in fo], 0) if len(fo) else np.zeros((0, L), np.float32)
    cap = P.cap[None, :].astype(np.float32)
    g = lambda c: df[c].to_numpy(np.float32).reshape(n, L)
    fl = g("f_last") * cap
    out = {"ecap_f": fl / e, "ecap_ext": g("fq_ext30") * cap / e, "ecap_mx": g("fq_max15") * cap / e,
           "ecap_m3": g("f_m3") * cap / e}
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        up = np.nanmean(np.stack([T._shift_nan(fl, o) for o in (-1, -2, -3)], 0), 0)
        out["ecap_arr"] = up / e
        dnmin = np.min(np.stack([T._shift(e, o) for o in range(4)], 0), 0)
        out["ecap_dn"] = fl / dnmin
        out["ecap_ext_rk"] = T._rank_rows(out["ecap_ext"], ascending=False)
        out["ecap_ext_gap"] = out["ecap_ext"] - np.nanmax(out["ecap_ext"], 1, keepdims=True)
        out["ecap_dn_rk"] = T._rank_rows(out["ecap_dn"], ascending=False)
    return out


# ----------------------------------------------------------------------------- settings
def candidates(cfg) -> dict:
    c = {k: [list(x) for x in v] for k, v in DEFAULT_CANDIDATES.items()}
    c.update({k: [list(x) for x in v] for k, v in (cfg.get("candidates") or {}).items()})
    for cond, cands in c.items():
        if cond not in T.CONDS:
            raise ValueError(f"candidates: unknown condition {cond!r}")
        if not cands:
            raise ValueError(f"candidates[{cond}]: empty")
        for f, vn in cands:
            T._check_fset(f)
            [T._variant_name(v) for v in vn]
    return c


def task_cfg(cfg) -> dict:
    """The settings the task2 building blocks read: the candidates (the first is the baseline = the v10 recipe; their
    feature groups decide which columns the tables carry), no probes, no online adaptation."""
    return dict(cfg, configs=candidates(cfg), probe_onset=None, probe_ongoing=None, experiments=True,
                online_adapt=False, pseudo_april=False)


def _norm(d) -> str:
    return json.dumps({k: v for k, v in (d or {}).items() if k not in ("model_dir",)}, sort_keys=True, default=str)


def up_to_date(cfg, work=None) -> tuple:
    """(True, report) when the v11 files of this code, these settings and this release are on disk."""
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
    ok = (rep.get("version") == VERSION and rep.get("code") == code_hash() and _norm(rep.get("cfg")) == _norm(cfg)
          and rep.get("fingerprint") == fp
          and set(rep.get("files_md5") or {}) >= {MAIN_FILE, HEDGE_FILE, SIZE_FILE, MAIN_TOP1, HEDGE_TOP1}
          and all((d / fn).exists() and T._md5(d / fn) == h for fn, h in (rep.get("files_md5") or {}).items()))
    return ok, rep


# ----------------------------------------------------------------------------- April: location-free refit (onset)
def _adopt(st, M, cfg) -> bool:
    sim = M[M.src == "sim"]
    return bool(len(sim) >= cfg.get("adapt_min_windows", 40) and sim.day.nunique() >= 3 and st["se_sim"] > 0
                and np.isfinite(st["d_sim"]) and st["d_sim"] > float(cfg.get("adapt_z", 1.5)) * st["se_sim"]
                and st["d_all"] >= 0)


def locfree_gate(Dl, std, pseudo, pans, Ps, cond, prm, rounds, variants, seeds, cfg, fset) -> dict:
    """v10's April gate (train + March before the cut, scored on March from the cut, against the standard models) for
    the full refit, the location-free refit and their 50/50 blend; the blend is adopted by the v10 rule."""
    vlf = T._variants([cfg.get("locfree_variant", LOCFREE)])
    vd = C.split_days("validation")
    cut = int(vd.min() + int(round(len(vd) * cfg.get("adapt_frac", 0.5))))
    alpha = float(cfg.get("adapt_alpha", 2.0))
    rep = dict(variant=vlf[0][0], cut_day=cut, adopted=False)
    Dg = T.pseudo_train_rows(pseudo, pans, "validation", cond, day_hi=cut)
    Rs = T.pseudo_scores(std[:: len(seeds)], pseudo, pans, Ps, "validation", cond, day_lo=cut)
    if Dg is None or len(Rs) == 0:
        rep["reason"] = "no March pseudo windows on one side of the cut"
        return rep
    g_full = T.fit_set(Dl, cond, prm, rounds, variants, seeds[:1], Dg, alpha, role="gate", fset=fset)
    g_lf = T.fit_set(Dl, cond, prm, rounds, vlf, seeds[:1], Dg, alpha, role="gate", fset=fset)
    del Dg
    Mb = None
    for name, ms in (("full", g_full), ("locfree", g_lf), ("blend", list(g_full) + list(g_lf) * len(variants))):
        Ra = T.pseudo_scores(ms, pseudo, pans, Ps, "validation", cond, day_lo=cut)
        st, M = T.paired(Rs, Ra)
        rep[name] = {k: st[k] for k in ("n_sim", "n_all", "S_a_sim", "S_b_sim", "d_sim", "se_sim", "d_all", "se_all",
                                        "better", "worse")}
        if name == "blend":
            Mb, stb = M, st
    del g_full, g_lf
    C.free()
    adopt = _adopt(stb, Mb, cfg)
    if cfg.get("locfree_force", False):   # tests only
        adopt, rep["forced"] = True, True
    rep["adopted"] = bool(adopt)
    return rep


# ----------------------------------------------------------------------------- diagnostics
def panel_tod(R, cond) -> dict:
    """Pseudo-holdout IoU by corridor x time of day (onset: also the site hit), over all windows and weekdays."""
    if R is None or len(R) == 0:
        return {}
    R = R.dropna(subset=["n_pred"]).copy()
    bins = [0, 6, 10, 15, 20, 24]
    names = ["00-06", "06-10", "10-15", "15-20", "20-24"]
    R["tb"] = pd.cut(R.tod, bins, right=False, labels=names)
    R["wd"] = T._weekday(R.day.to_numpy())
    out = {}
    for (p, tb), g in R.groupby(["panel", "tb"], observed=True):
        d = dict(n=int(len(g)), iou=round(float(g.iou.mean()), 4), n_weekday=int(g.wd.sum()))
        if cond == "queue_onset":
            d["site_hit"] = round(float((g.inter > 0).mean()), 4)
        out.setdefault(str(p), {})[str(tb)] = d
    return out


# ----------------------------------------------------------------------------- the run
def _ids(offis, pans):
    ids = {c: {s: set() for s in C.TEST_SPLITS} for c in T.CONDS}
    for p in pans:
        for s in C.TEST_SPLITS:
            if s in offis[p]:
                wi = offis[p][s][0]
                for c in T.CONDS:
                    ids[c][s] |= set(wi.window_id[wi.condition == c])
    return ids


def size_probs(probs, cond="queue_onset", splits=("validation",)) -> dict:
    """The onset windows of `splits` with a one-hot field at their most likely link (decoded: that single cell)."""
    out = {}
    for p, d in probs.items():
        for split in splits:
            for wid, (c, pr) in (d.get(split) or {}).items():
                if c != cond:
                    continue
                oh = np.zeros_like(np.asarray(pr, np.float64))
                oh[int(np.argmax(pr))] = 1.0
                out.setdefault(p, {}).setdefault(split, {})[wid] = (c, oh)
    return out


def effective(prob_sets) -> dict:
    """Per window the probabilities write_rows decodes: the first set that has the window."""
    out = {}
    for ps in reversed(prob_sets):
        for p, d in ps.items():
            for split, ws in d.items():
                out.setdefault(p, {}).setdefault(split, {}).update(ws)
    return out


def _write(frame: pd.DataFrame, d: Path, name: str):
    tmp = d / (name + ".part")
    frame.to_csv(tmp, index=False)
    tmp.replace(d / name)


def run_main(cfg: dict, device: dict | None = None) -> dict:
    """Writes pred/task2_v11.csv (main), pred/task2_v11_hedge.csv, pred/task2_v11_size.csv (probe: March onset windows
    at their top cell), pred/task2_v11_top1.csv / task2_v11_hedge_top1.csv (every onset window of both months at its top
    cell, for the final picks if the probe reads positive) and pred/task2_v11_report.json (written last).
    `device`: extra LightGBM training parameters ({} / None = the CPU)."""
    from .data import fingerprint
    install()
    t0 = time.time()
    tcfg = task_cfg(cfg)
    cands = tcfg["configs"]
    TM.use_device(device)
    T.set_checkpoints(tcfg)
    pans = C.task2_panels()
    rng = np.random.default_rng(tcfg.get("seed", 0))
    imputer = T.fit_imputer(pans) if tcfg.get("imputer", True) else None
    C.log(f"T2 v11 label imputer ready ({time.time() - t0:.0f}s); table groups onset "
          f"{T.extra_groups(tcfg, 'queue_onset')!r}, ongoing {T.extra_groups(tcfg, 'queue_ongoing')!r}")
    sets, offis, Ps, pseudo = {}, {}, {}, {}
    elok = {}
    for i, p in enumerate(pans):
        P, out, offi, ps = T.build_panel_sets(p, imputer, rng, tcfg)
        for split in ps:
            for cond in ps[split]:
                ps[split][cond][3]["panel_code"] = np.float32(i)
        sets[p] = out; offis[p] = offi; pseudo[p] = ps
        Ps[p] = dict(L=P.L, links=P.net.links, mp=np.asarray(P.mp, np.float64))
        if "queue_onset" in out and "elok" in out["queue_onset"][0]:
            w = out["queue_onset"][0]
            elok[p] = dict(windows=int(len(w)), dropped=int((~w.elok.astype(bool)).sum()))
        npw = ", ".join(f"{s[:3]} {c.split('_')[1]} {len(v[0])}" for s in ps for c, v in ps[s].items())
        C.log(f"T2 v11 {p}: train windows onset {len(out.get('queue_onset', ([],))[0])}, "
              f"ongoing {len(out.get('queue_ongoing', ([],))[0])}; pseudo windows: {npw or 'none'}")
        P.dprof = None
        del P
        C.free()
    report = dict(version=VERSION, code=code_hash(), task2_code=T.code_hash(), x_code=x_hash(), fingerprint=fingerprint(),
                  cfg=cfg, candidates=cands, chosen={}, choose={}, pseudo={}, diag={}, panel_tod={}, adapt={}, locfree={},
                  late={}, early={}, minutes_by_step={}, onset_train_windows_without_eligible_queue=elok,
                  windows=T.official_window_calendar(offis, pans))
    probs_m, apr_m, late_m, early_m = {}, {}, {}, {}
    april_src = {}
    for cond in T.CONDS:
        tc = time.time()
        Dl = T._frames(sets, pans, cond)
        if not Dl:
            raise RuntimeError(f"T2 v11: no training windows for {cond}")
        on = cond == "queue_onset"
        prm, rounds = (T.ONSET_P, tcfg.get("onset_rounds", 500)) if on else (T.ONG_P, tcfg.get("ongoing_rounds", 700))
        seeds = list(tcfg.get("onset_seeds" if on else "ongoing_seeds", [tcfg.get("seed", 0)]))
        lseeds = list(tcfg.get("late_seeds", seeds[:1]))
        has_march = any(cond in pseudo[p].get("validation", {}) for p in pans)
        C.log(f"T2 v11 {cond}: {sum(len(f) for f in Dl):,} training rows; candidates {cands[cond]}")
        # ---- the recipe: the v10 recipe (first candidate) unless a candidate wins on both draws (v5/v6 rules)
        ccfg = dict(tcfg, exp_cv=bool(tcfg.get("exp_cv", True)) and (on or bool(tcfg.get("exp_cv_ongoing", False))))
        if len(cands[cond]) > 1 and has_march and tcfg.get("choose", True):
            (fset, variants), crep = T.choose_config(Dl, cond, pseudo, pans, Ps, prm, rounds, seeds, ccfg, sets=sets)
            report["choose"][cond] = crep
            C.log(f"T2 v11 recipe {cond}: chosen {crep.get('chosen')} | "
                  + " | ".join(f"{k}: gain {v.get('gain', float('nan')):+.4f} se {v.get('se_gain', float('nan')):.4f} "
                               f"cv {v.get('d_cv', float('nan')):+.4f} take {v.get('take')}" for k, v in crep.get("candidates", {}).items()))
        else:
            fset, variants = cands[cond][0][0], T._variants(cands[cond][0][1])
        report["chosen"][cond] = [fset, [v for v, _ in variants]]
        # ---- standard models (train only): the March rows of both files and the April rows of the hedge
        std = T.fit_set(Dl, cond, prm, rounds, variants, seeds, fset=fset)
        T.predict_official(std, offis, pans, Ps, cond, probs_m, ("validation", "private"))
        if has_march:
            R_ = T.pseudo_scores(std, pseudo, pans, Ps, "validation", cond)
            report["pseudo"][cond] = dict(main=T.pseudo_report(R_))
            report["diag"][cond] = T.window_diag(R_, cond)
            report["panel_tod"][cond] = panel_tod(R_, cond)
            del R_
        # ---- April refit on train + all March pseudo windows (gate: first half of March -> second half)
        adapted, ad = False, None
        if tcfg.get("april_adapt", True) and has_march:
            rep_ad, ad = T.april_adapt(Dl, std, pseudo, pans, Ps, cond, prm, rounds, variants, seeds, tcfg, fset)
            report["adapt"][cond] = rep_ad
            C.log(f"T2 v11 April adaptation {cond}: {rep_ad}")
        april_src[cond] = "standard"
        if on and tcfg.get("locfree", True) and has_march and tcfg.get("april_adapt", True):
            lrep = locfree_gate(Dl, std, pseudo, pans, Ps, cond, prm, rounds, variants, seeds, tcfg, fset)
            report["locfree"][cond] = lrep
            C.log(f"T2 v11 location-free April refit {cond}: {lrep}")
            if lrep["adopted"]:
                Df = T.pseudo_train_rows(pseudo, pans, "validation", cond)
                alpha = float(tcfg.get("adapt_alpha", 2.0))
                if ad is None:
                    ad = T.fit_set(Dl, cond, prm, rounds, variants, seeds, Df, alpha, role="adapt", fset=fset)
                vlf = T._variants([lrep["variant"]])
                ad_lf = T.fit_set(Dl, cond, prm, rounds, vlf, seeds, Df, alpha, role="adapt", fset=fset)
                del Df
                ad = list(ad) + list(ad_lf) * len(variants)
                april_src[cond] = "blend of the full and the location-free refit"
        if ad is not None:
            T.predict_official(ad, offis, pans, Ps, cond, apr_m, ("private",))
            adapted = True
            if april_src[cond] == "standard":
                april_src[cond] = "full refit (v10 rule)"
        del ad
        C.free()
        # ---- late April windows: + April's own earlier pseudo windows (+ March when adapted)
        if tcfg.get("late_adapt", False) and cond in tcfg.get("late_conds", list(T.CONDS)) and has_march:
            n_late = sum(T.n_official(offis, pans, cond, "private", lo, hi) for lo, hi in T.late_blocks(tcfg, "private"))
            if n_late == 0:
                report["late"][cond] = dict(adopted=False, reason="no official April window of this condition in the late blocks")
            else:
                lrep = T.late_gate(Dl, std[:: len(seeds)], pseudo, pans, Ps, cond, prm, rounds, variants, tcfg, fset,
                                   seed=seeds[0])
                lrep["official_late_windows"] = n_late
                if lrep["adopted"]:
                    lrep["main"] = T.late_fit_predict(Dl, pseudo, offis, pans, Ps, cond, prm, rounds, variants, lseeds,
                                                      tcfg, fset, adapted, late_m, "main")
                report["late"][cond] = lrep
                C.log(f"T2 v11 late windows {cond}: {lrep}")
        # ---- early April windows (v8 blocks)
        if T.early_blocks(tcfg, "private", cond) and has_march:
            erep = TM.early_main(Dl, std[:: len(seeds)], pseudo, offis, pans, Ps, cond, prm, rounds, variants, seeds,
                                 lseeds, tcfg, fset, adapted, early_m)
            report["early"][cond] = erep
            C.log(f"T2 v11 early April windows {cond}: {erep}")
        del Dl, std
        for p in pans:
            if cond in sets[p]:
                sets[p][cond] = (sets[p][cond][0], sets[p][cond][1], None)
        C.free()
        report["minutes_by_step"][cond] = round((time.time() - tc) / 60, 1)
    report["april_source"] = april_src
    # ---- files
    d = C.WORK / "pred"
    main_sets = [early_m, late_m, apr_m, probs_m]
    main = T.write_rows(main_sets, offis, pans, Ps)
    hedge = T.write_rows([probs_m], offis, pans, Ps)
    size = T.write_rows([size_probs(probs_m)] + main_sets, offis, pans, Ps)
    top1 = T.write_rows([size_probs(effective(main_sets), splits=C.TEST_SPLITS)] + main_sets, offis, pans, Ps)
    htop1 = T.write_rows([size_probs(probs_m, splits=C.TEST_SPLITS), probs_m], offis, pans, Ps)
    a = main.set_index(KEY).queue_pred
    hb = hedge.set_index(KEY).queue_pred
    sz = size.set_index(KEY).queue_pred
    for what, rows in (("the v11 main file", a), ("the v11 hedge file", hb), ("the v11 size probe", sz),
                       ("the v11 top-1 main file", top1.set_index(KEY).queue_pred),
                       ("the v11 top-1 hedge file", htop1.set_index(KEY).queue_pred)):
        TM.check_rows(rows, what)
    ids = _ids(offis, pans)
    mar = a.index.get_level_values(0).str.contains("_validation_")
    if not (a[mar] == hb.reindex(a.index)[mar]).all():
        raise RuntimeError("T2 v11: the hedge's March rows must be the main file's")
    on_mar = a.index.get_level_values(0).isin(list(ids["queue_onset"]["validation"]))
    if not (a[~on_mar] == sz.reindex(a.index)[~on_mar]).all():
        raise RuntimeError("T2 v11: the size probe may differ from the main file only on the March onset windows")
    t1a = top1.set_index(KEY).queue_pred.reindex(a.index)
    if not (t1a[on_mar] == sz.reindex(a.index)[on_mar]).all():
        raise RuntimeError("T2 v11: the top-1 main file's March onset rows must be the size probe's")
    _write(main, d, MAIN_FILE); _write(hedge, d, HEDGE_FILE); _write(size, d, SIZE_FILE)
    _write(top1, d, MAIN_TOP1); _write(htop1, d, HEDGE_TOP1)
    # ---- what changed: against the v10 main and hedge files
    checks = {}
    for fn, nm in ((V10_MAIN, "v10 main"), (V10_HEDGE, "v10 hedge")):
        f = d / fn
        if f.exists():
            old = pd.read_csv(f, dtype=QDT).set_index(KEY).queue_pred
            new = a if nm == "v10 main" else hb
            for c in T.CONDS:
                for s, m in (("validation", "March"), ("private", "April")):
                    checks[f"{m} {c} rows ({'main' if nm == 'v10 main' else 'hedge'}) vs the {nm} file"] = TM._compare(new, old, ids[c][s])
    report["checks"] = checks
    report["agreement_main_hedge"] = {m: dict(cells_main=int(a[sel].sum()), cells_hedge=int(hb.reindex(a.index)[sel].sum()),
                                              differ=int((a[sel] != hb.reindex(a.index)[sel]).sum()))
                                      for m, sel in (("March", mar), ("April", ~mar))}
    report["size_probe"] = dict(march_onset_cells_main=int(a[on_mar].sum()), march_onset_cells_probe=int(sz.reindex(a.index)[on_mar].sum()),
                                windows=len(ids["queue_onset"]["validation"]))
    report["device"] = dict(params=dict(device or {}), failed=TM.DEVICE["failed"], fits_gpu=TM.DEVICE["fits_gpu"],
                            fits_cpu=TM.DEVICE["fits_cpu"])
    report["rows"] = int(len(main)); report["share_pred"] = float(main.queue_pred.mean())
    report["minutes"] = round((time.time() - t0) / 60, 1)
    report["files_md5"] = {fn: T._md5(d / fn) for fn in (MAIN_FILE, HEDGE_FILE, SIZE_FILE, MAIN_TOP1, HEDGE_TOP1)}
    tmp = d / (REPORT_FILE + ".part")
    json.dump(report, open(tmp, "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    tmp.replace(d / REPORT_FILE)   # written last: it vouches for the files
    C.log(f"T2 v11 done in {report['minutes']} min: chosen {report['chosen']} | April {april_src} | "
          f"checks {json.dumps(checks)[:600]}")
    return report
