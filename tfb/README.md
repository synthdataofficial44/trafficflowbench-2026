# tfb — the TrafficFlowBench 2026 pipeline

Byte-identical to the code that produced the selected submission (SHA-256 in `../docs/manifest_v22.json`; the `check`
stage of `run_all.py` verifies it). Each later module sits on top of the earlier ones without editing them, so every
stored model and report of an earlier version stays valid.

| Module | Version | Role |
|---|---|---|
| `common.py`, `data.py` | — | paths, calendar, release reader, dense per-panel caches, preflight |
| `task4.py` | v2 | Task 4: Euclidean projection of the prior onto the link counts |
| `task1.py` | v2–v8 | Task 1: rows, LightGBM residual sets, blend choice, post-processing, test predictions |
| `physics.py`, `t1lab.py` | v10 | Task 3 conservation analysis; physics lab stage 2 |
| `t1lab11.py` | v11 | stage 3 heads |
| `t1pre13.py`, `t1v14.py`, `t1v15.py` | v13–v15 | fourth and fifth sets, their blends and the lab reruns |
| `task2.py` | v2–v9 | Task 2: selector replica, window features (history, CTM forecast, month memory), models, decoder |
| `task2m.py`, `task2x.py`, `task2y.py`, `task2z.py`, `task2o.py`, `task2v15.py` | v10–v15 | Task 2 versions (month memory, location-free refit, v12 models, nl75, ongoing weighting diagnostic, v15 files) |
| `submit.py` | — | the upload file: template order, checks, zip |
| `asof.py` | v22 | as-of-origin input check and manifest |
| `probe.py`, `diag12.py`, `gpu.py`, `research13.py` | — | leaderboard probes, diagnostics, LightGBM GPU benchmark, research dump (not used by `run_all.py`) |
