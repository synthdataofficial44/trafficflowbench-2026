"""v22: the Task 2 as-of-origin self-check (inputs) and the reproducibility manifest. Nothing here changes a file the
pipeline or a submission reads.

The organizers' award review (discussion 749601): "for Task 2 we will rerun it with every row stamped after each
window's origin removed and check that the predictions do not change". With the models frozen, a window's prediction
can only change if its inputs change, so `check_inputs` rebuilds the inputs of a sample of official windows twice -
from the full panel arrays, and from a copy in which every released row stamped after the window's origin T is blank
(every mainline channel, the quality flags, the ramp flows and the ramp validity counts) - and compares them column by
column. The columns compared are the ones the final Task 2 models read: onset `v2+pm` and ongoing `v2+m` (variants
all / noloc / the location-free onset refit) plus every other column of the table; the onset prior (`op_*`, train
windows only) is left out of both copies. Train-derived parts (profiles, capacities, the CTM constants) are rebuilt
from the same train rows in both copies, so they are compared too.

What the models were fitted on (the other half of the review) is recorded by `provenance`: which data each Task 2
role used, by construction of the pipeline (std: train; adapt: train + March; late block: train + March + the April
days before the block's first day; gates and selection: March pseudo windows for April rows only; March rows: train
only).
"""
from __future__ import annotations

import copy
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import common as C
from . import task2 as T

VERSION = "v22"
REPORT = "asof_report.json"
MANIFEST = "manifest_v22.json"
# the released per-slot arrays of a panel (P2 attributes); everything else of P2 is static or train-only
TIME_ARRAYS = ("mv", "mq", "mo", "el", "pct", "ron", "roff", "non", "noff", "tgt", "dark")
LAZY = ("q_day", "q_bin", "cq", "cn", "mcq", "mcn", "mem_blank", "_ron_full", "_ecap", "_av", "ctm")
RECIPES = {"queue_onset": ("v2+pm", ("all", "noloc", "noloc2_nr")), "queue_ongoing": ("v2+m", ("all", "noloc"))}


def _blank_after(P: "T.P2", t_last: int) -> "T.P2":
    """A shallow copy of P whose released per-slot arrays are blank after slot t_last (NaN for float arrays, 0 / False
    otherwise) and whose lazily built caches are dropped, so they are rebuilt from the truncated arrays."""
    Q = copy.copy(P)
    for name in TIME_ARRAYS:
        a = getattr(P, name, None)
        if a is None:
            continue
        b = np.array(a, copy=True)
        if b.dtype.kind == "f":
            b[t_last + 1:] = np.nan
        else:
            b[t_last + 1:] = 0
        setattr(Q, name, b)
    for name in LAZY:
        if hasattr(Q, name):
            setattr(Q, name, None)
    Q._ramp_est = {}
    return Q


def _cols_used() -> dict:
    """{condition: feature columns of the final recipes} (resolved on an empty frame later)."""
    return RECIPES


def _compare(a: pd.DataFrame, b: pd.DataFrame, cols) -> dict:
    out = {}
    for c in cols:
        x = a[c].to_numpy(np.float64); y = b[c].to_numpy(np.float64)
        same_nan = np.isnan(x) == np.isnan(y)
        fin = np.isfinite(x) & np.isfinite(y)
        d = float(np.max(np.abs(x[fin] - y[fin]))) if fin.any() else 0.0
        if not same_nan.all() or d > 0.0:
            out[c] = dict(max_abs_diff=d, nan_mismatch=int((~same_nan).sum()))
    return out


def check_inputs(cfg: dict, panels=None, per_split: int = 2, seed: int = 0, splits=("validation", "private")) -> dict:
    """For `per_split` official windows per (panel, split) - one onset and one ongoing window when both exist, drawn
    with `seed` - compare the feature rows built from the full arrays with those built from arrays blank after the
    window's origin. Returns a report: windows checked, columns compared, and every difference."""
    t0 = time.time()
    rng = np.random.default_rng(seed)
    groups = "".join(sorted(set("".join(T._check_fset(f)[1] for f, _ in RECIPES.values()))))
    rows, diffs = [], []
    used = {}
    for panel in (panels or C.panels()):
        if not any((C.rel() / "task2" / panel / s / "window_index.csv").exists() for s in splits):
            continue
        P = T.P2(panel)
        P.build_labels(None)          # train labels: the train profiles (pq_*) of both copies
        prof = T.Prof(P)
        for split in splits:
            if not (C.rel() / "task2" / panel / split / "window_index.csv").exists() or len(C.split_days(split)) == 0:
                continue
            wi, A = P.inputs_official(split)
            pick = []
            for cond in T.CONDS:
                idx = np.flatnonzero(wi.condition.to_numpy() == cond)
                if len(idx):
                    pick.append(int(rng.choice(idx)))
            pick = pick[:max(per_split, 1)]
            for i in pick:
                Ts = wi["T"].to_numpy()[i:i + 1]
                Ai = {k: v[i:i + 1] for k, v in A.items()}
                folds = np.full(1, -1)
                full, _ = T.window_features(P, prof, Ts, Ai, folds, None, cfg.get("use_T", True), extra=groups)
                Pt = _blank_after(P, int(Ts[0]))
                trunc, _ = T.window_features(Pt, prof, Ts, Ai, folds, None, cfg.get("use_T", True), extra=groups)
                cond = str(wi.condition.iloc[i])
                fset, variants = RECIPES[cond]
                cols_model = sorted(set().union(*[T._feats(full, v, fset) for v in variants]))
                used[cond] = len(cols_model)
                d_model = _compare(full, trunc, cols_model)
                d_all = _compare(full, trunc, [c for c in full.columns if c in trunc.columns])
                rows.append(dict(panel=panel, split=split, window_id=str(wi.window_id.iloc[i]), condition=cond,
                                 origin=str(wi.forecast_origin.iloc[i]), cols_model=len(cols_model), cols_all=full.shape[1],
                                 diff_model=len(d_model), diff_all=len(d_all)))
                if d_all:
                    diffs.append(dict(panel=panel, split=split, window_id=str(wi.window_id.iloc[i]), model_columns=d_model,
                                      other_columns={k: v for k, v in d_all.items() if k not in d_model}))
                del Pt, full, trunc
        del P, prof
    rep = dict(version=VERSION, windows=len(rows), model_columns=used, rows=rows, differences=diffs,
               passed=bool(rows) and not any(r["diff_model"] for r in rows),
               all_columns_equal=bool(rows) and not any(r["diff_all"] for r in rows),
               minutes=round((time.time() - t0) / 60, 1),
               note="inputs of each window rebuilt with every released row after its origin blanked; equal model "
                    "columns + frozen models = unchanged predictions")
    return rep


def provenance() -> dict:
    """What each Task 2 model role was fitted / selected on (by construction of task2.py / task2m / task2v15)."""
    return {
        "std (March rows, hedge April rows)": "train split only (selector-replica windows, train labels)",
        "adapt (main April rows)": "train + every March window (pseudo windows from the March masked view); "
                                   "every March row is stamped before every April origin",
        "late:private:main:1:6 (10 April ongoing windows from 6 April)": "train + March + April pseudo windows of the days "
                                                                       "before 6 April (labels read up to the block start)",
        "gate / selection (cross-draw rule, nl75, v15 gates)": "March pseudo windows - they shape April rows only",
        "March rows": "std models and v12 decisions only (v15 put the March onset rows back): no March data after an "
                      "origin feeds a March row",
        "decoder": "top-m expected-IoU surrogate of the window's own probabilities (no fitted threshold)",
    }


def _sha(path: Path, algo="md5") -> str:
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def _versions() -> dict:
    out = dict(python=sys.version.split()[0], platform=platform.platform())
    for m in ("numpy", "pandas", "pyarrow", "lightgbm", "catboost", "scipy", "sklearn", "torch"):
        try:
            mod = __import__(m)
            out[m] = getattr(mod, "__version__", "?")
        except Exception:  # noqa: BLE001
            out[m] = None
    return out


def _tree(d: Path, pattern="*") -> dict:
    """{relative path: size} of the files under d (sorted); empty when d does not exist."""
    if not d.exists():
        return {}
    return {str(p.relative_to(d)): p.stat().st_size for p in sorted(d.rglob(pattern)) if p.is_file()}


def manifest(picks: dict, files: dict, cfg: dict, pkg_dir: Path, extra_files=()) -> dict:
    """The reproducibility manifest of the current picks: every upload file's content hash and its components (Task 1
    parquet, Task 2 csv, Task 4 csv) with their hashes now on disk, the hashes of the code that built them, package
    versions, the settings, and an inventory of the model folders the pipeline reloads (names and sizes)."""
    W = C.WORK
    comp = {}
    for name, f in files.items():
        parts = {}
        for key in ("t1", "t2"):
            fn = f.get(key)
            if fn and (W / "pred" / fn).exists():
                parts[fn] = _sha(W / "pred" / fn)
        if (W / "pred" / "task4.csv").exists():
            parts["task4.csv"] = _sha(W / "pred" / "task4.csv")
        comp[name] = dict(zip=f.get("zip"), csv_md5=f.get("csv_md5"), components=parts)
    code = {p.name: _sha(p, "sha256") for p in sorted(Path(pkg_dir).glob("*.py"))}
    inv = {k: dict(files=len(v), bytes=int(sum(v.values()))) for k, v in (
        ("t2_models", _tree(Path(cfg.get("model_dir", W / "t2_models")))),
        ("t1", _tree(W / "t1")), ("t1lab", _tree(W / "t1lab")), ("pred", _tree(W / "pred")))}
    pk = {p.name: _sha(p) for p in sorted((W / "pred").glob("*.pkl"))} if (W / "pred").exists() else {}
    ex = {str(p): _sha(Path(p)) for p in extra_files if Path(p).exists()}
    return dict(version=VERSION, built=time.strftime("%Y-%m-%d %H:%M"), picks=picks, files=comp, code_sha256=code,
                packages=_versions(), settings=cfg, inventory=inv, prob_pickles_md5=pk, extra=ex,
                provenance=provenance(),
                rules=dict(task2="for a window with origin T only released rows stamped <= T feed its inputs, fitting and "
                                 "selection (discussions 742068, 748789, 749601)",
                           april_labels="never used", march_for_april="allowed (all of March precedes April)"))
