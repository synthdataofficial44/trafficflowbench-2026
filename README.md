# TrafficFlowBench 2026 — final package of team "Emre Cirak"

Code and report for the **2026 IEEE Big Data Cup – Traffic Flow Bench** (Kaggle competition
`2026-ieee-big-data-traffic-flow-bench`), team **Emre Cirak** (Kaggle user `biohack44`).

| | |
|---|---|
| Selected final submission | `submission_20261025_1742_v15b_main.zip` (Kaggle auto-selected the two best public files; this one counts) |
| Private leaderboard | **0.88253**, rank **55 of 802** |
| Public leaderboard | 0.87514 |
| Report | [`docs/TrafficFlowBench2026_EmreCirak_report.pdf`](docs/TrafficFlowBench2026_EmreCirak_report.pdf) |
| Single entry script | [`run_all.py`](run_all.py) |
| Settings | [`configs/final.json`](configs/final.json) |
| Environment | [`requirements.txt`](requirements.txt), section *Environment* below |

The selected file is the merge of three component files, each produced by the code in `tfb/`:

| Task | Component | md5 of the component file |
|---|---|---|
| Task 1 (state reconstruction; Task 3 is scored on it) | `pred/task1_v15.parquet` | `c0327df4e41373dd733c70ae64cb6de9` |
| Task 2 (queue forecasting) | `pred/task2_v15_0.csv` | `9be1bef82420ac144af4020832e3420b` |
| Task 4 (ODME path flows) | `pred/task4.csv` | `d514ac203fa8489c6d4956dd77ec0e20` |
| Upload file (`submission.csv` inside the zip) | | `6d196f7870a9e58d13edb4c183cef020` |

`tfb/` is **byte-identical** to the code that produced the submission: the `check` stage of `run_all.py` compares the
SHA-256 of every module with [`docs/manifest_v22.json`](docs/manifest_v22.json), the manifest written by the run that
built the selected file.

## Quick start

```bash
git clone https://github.com/synthdataofficial44/trafficflowbench-2026.git tfb && cd tfb
pip install -r requirements.txt

# the competition data (needs a Kaggle account that accepted the competition rules)
kaggle competitions download -c 2026-ieee-big-data-traffic-flow-bench -p data && unzip -q data/*.zip -d data

python run_all.py --data data            # train everything, predict, assemble, check  -> output/submission.csv
```

`--data` is the unpacked competition folder (`kaggle_public`, or any folder above it). The run writes
`output/submission.csv` (+ `.zip`, the file to upload), `output/run_report.json` (what each stage decided, md5 of every
component, timings) and `output/asof_report.json` (the Task 2 as-of-origin input check).

Other ways to run it:

```bash
python run_all.py --data data --stages task4,task2,files      # a subset of the stages
python run_all.py --data data --stages asof                    # only the Task 2 as-of-origin input check (~5 min)
python run_all.py --data data --work /path/to/tfb_work         # reuse trained artifacts (see below): reload, no training
python tests/test_run_all_flow.py                              # control-flow test of run_all.py (no data needed)
```

Every stage stores its result in `--work` together with the code hash, the settings and a fingerprint of the data, and
is skipped when such a result is already there — an interrupted run resumes where it stopped.

## What `run_all.py` does

The submission came out of a sequence of Colab runs (v8 → v22), each adding one step on top of the stored results of
the previous ones. `run_all.py` runs the same stage functions in the same order with the same settings:

| Stage | Original run | What it does | Writes |
|---|---|---|---|
| `check` | — | code hashes vs the manifest, package versions | — |
| `caches` | all | data preflight; dense per-panel arrays of every split on one time axis | `cache/` |
| `task4` | v8 | per panel and split, the Euclidean projection of the released prior onto the released link counts, min ‖f − b‖² s.t. A f = c, f ≥ 0 (exact, semismooth Newton on the dual) | `pred/task4.csv` |
| `task1` | v8 | rows; three LightGBM residual sets (s0, s1, s2; s2 with a 2× transductive sample); blend chosen on held-out March cells; post-processing (physics reconciliation + TV smoothing) chosen on the train holdout | `pred/task1.parquet` |
| | v10 | physics lab, stage 2 (`s2`): a LightGBM model of the density error from the released cells around each target, gated on the train holdout J = 0.35·S_state + 0.10·S_LWR and on held-out March cells | `pred/task1_v10.parquet` |
| | v11 | stage 3 (`s3vq`): log-speed and log-flow heads on neighbour fluctuations, ramp flows and conservation estimates, on top of stage 2; same gates | `pred/task1_v11.parquet` |
| | v13 | fourth set s3 (3× transductive sample, up to 3 500 rounds) | `t1/models_r1_s3` |
| | v14 | blend `new_x2` (s3 at twice the average weight) + the lab rerun on it | `pred/task1_v14.parquet` |
| | v14 (step 12) | fifth set s4 (4× transductive sample, up to 5 000 rounds), scored against the v14 blend | `t1/t1next_report.json` |
| | v22 ("Task 1 v15") | blend `new_x2` with s4 (weights s0–s4 = 2/15, 2/15, 2/15, 4/15, 5/15) + the lab rerun | **`pred/task1_v15.parquet`** |
| `task2` | v12 | onset and ongoing LightGBM window models (7 / 3 seeds, full + location-free variants), month memory and cell-transmission features, March pseudo windows, the April refit on train + March, the late-block refit (from 6 April), the top-m expected-IoU decoder | `pred/task2_v12*.csv`, `t2_models/` |
| | v13.1 | onset: the location-free variant weighted 0.75 instead of 0.5 (`nl75`), adopted by the cross-draw rule on March pseudo windows; in the final file it shapes the April rows only | `pred/task2_v131*.csv` |
| | v15 | v15 files; the **fallback main** = v13.1's April rows with v12's March rows | **`pred/task2_v15_0.csv`** |
| `files` | v22 | the upload file in template order, format checks, zip; md5 against the selected submission | `output/submission.csv` |
| `asof` | v22 | Task 2 inputs rebuilt with every row after each window's origin blanked, compared column by column with the full-data inputs | `output/asof_report.json` |

Decisions inside the stages (blend choice, lab candidates, Task 2 April refits, the onset weighting) are made by the
code's own gates on training data and March pseudo windows, exactly as in the original runs; none uses an April label.
Two choices came from public-leaderboard reads and are pinned in `configs/final.json`:

1. **Task 2 file.** The v15 ongoing reweighting (location-free weight 0.75) was checked with a probe file on the public
   board; the probe read −0.00196, below the pre-set bar of −0.0015, so the pre-registered rule switched to the fallback
   pair (v13.1's April rows). `run_all.py` therefore assembles `task2_v15_0.csv`.
2. **Task 1 file.** The v22 Task 1 answer (`task1_v15.parquet`) was adopted by its gates and confirmed on the public board
   (+0.00022). If a rerun's gates keep an earlier answer instead, `run_all.py` uses that one and says so in the log and in
   `run_report.json`.

The CatBoost member of Task 1 is trained only on a GPU (`cb='auto'`). The original v8 and v14 runtimes had no GPU, and in
v22 the CatBoost blends were evaluated and rejected, so `run_all.py` hides the GPU from CatBoost by default: the blends
are then chosen among the same members as originally on any machine. `--catboost` restores the v22 behaviour (it does
not change the final answer).

## Environment

The submission was produced on **Google Colab Pro+, A100 GPU runtime with High-RAM** (Linux 6.6, glibc 2.39; the GPU
was used only by the rejected CatBoost candidates). Versions recorded by that run (`docs/manifest_v22.json`):

| | version |
|---|---|
| Python | 3.13 (Colab image of October 2026) |
| numpy | 2.1.3 |
| pandas | 2.2.3 |
| pyarrow | 23.0.1 |
| lightgbm | 4.6.0 (CPU) |
| scipy | 1.16.3 |
| scikit-learn | 1.6.1 (not used by the pipeline) |
| catboost | 1.2.10 (optional; only for the rejected CatBoost candidates) |

Requirements:
- **RAM ≥ 50 GB** (Task 1 rows and the lab; a Colab High-RAM runtime), **disk ≈ 100 GB** free for `--work` and `--cache`.
- **No GPU is needed**: every model of the final answer is a CPU LightGBM model.
- No external data, no pretrained weights, no internet access after the data download.

Approximate run times on that runtime (from the stage reports):

| Stage | time |
|---|---|
| caches | ≈ 25 min |
| task4 | ≈ 2 min |
| task1 (five LightGBM sets, three blend choices, three lab runs) | ≈ 20–26 h |
| task2 (v12 ≈ 3.2 h, v13.1 ≈ 1 h, v15 ≈ 25 min) | ≈ 5 h |
| files + asof | ≈ 10 min |
| everything reused from `--work` (reload and assemble only) | ≈ 1 h |

## Reproducibility

- **Reusing the trained artifacts** gives the selected file **bit for bit**: every stage finds its result (models,
  probabilities, reports) and only reloads it; the Task 2 stages also re-predict the official windows with the reloaded
  checkpoints and stop if they do not reproduce the stored probabilities. The artifacts (`pred/`, `t1/`, `t1lab/`,
  `t2_models/`, `w14/`, `w15/`, ≈ 12 GB) are too large for GitHub; they are available from the author on request
  (Google Drive). Pass the folder holding them as `--work`.
- **A full retrain** from the data follows the same steps with the same seeds. The LightGBM sets s2, s3 and s4 train with
  `deterministic=True, force_row_wise=True`; the other models are multithreaded CPU fits, which can differ in the last
  digits across machines and thread counts. The award rule asks for the private score within 1 % (±0.0088 here); the
  spread between all our April-scored files of the same method family (v12–v15b main: 0.88164–0.88253) is ten times
  smaller.
- **Task 2 time rule.** For a window with origin T, only released rows stamped ≤ T feed its inputs, model fitting and
  model or threshold selection; March data may serve April windows (all of March precedes April), never the reverse; no
  April label is ever used. The `asof` stage rebuilds the inputs of official windows (one onset and one ongoing window
  per panel and month, 32 in all) from arrays in which **every released row after the window's origin is blank**
  (mainline channels, quality flags, ramp flows, ramp validity counts; the month-memory, recency and CTM caches
  rebuilt from the truncated copy) and compares them column by column with the full-data inputs. The run that built the
  submission passed: 32 windows, all 141 onset / 118 ongoing model columns and every other column equal
  ([`docs/asof_report_v22.json`](docs/asof_report_v22.json)). Equal inputs and frozen models give unchanged predictions.
- What each Task 2 model role was fitted on is listed in `docs/manifest_v22.json` (`provenance`).

## Repository layout

```
run_all.py                 single entry script (all stages)
configs/final.json         settings of every stage (copied from the original notebooks; do not edit)
tfb/                       the pipeline package (byte-identical to the submission's code)
  common.py data.py        paths, calendar, release reader, dense caches, preflight
  task1.py                 Task 1 rows, LightGBM sets, blend choice, post-processing, test predictions
  physics.py t1lab.py      Task 3 conservation analysis (v10); physics lab with candidates kfree / rts / s2 (v10)
  t1lab11.py               stage 3 (v11)
  t1pre13.py t1v14.py t1v15.py   further sets and blends (v13, v14, v15)
  task2.py                 Task 2 windows, features (history, CTM forecast, month memory), models, decoder
  task2m.py task2x.py task2y.py task2z.py task2o.py task2v15.py   Task 2 versions v10, v11, v12, v13.1, v14 diag, v15
  task4.py                 Task 4 L2 projection
  submit.py                upload file: template order, checks, zip
  asof.py                  as-of-origin input check and manifest
  probe.py diag12.py gpu.py research13.py   leaderboard probes, diagnostics, device benchmark, research dump (not used by run_all)
docs/                      report (PDF), manifest of the selected submission, as-of report, all Kaggle scores
tests/                     control-flow test of run_all.py
```

## Rules

Only the competition data were used: no external data, no pretrained models, no hand labelling. Task 1 is offline
(organizer ruling): the test months' own observed cells are legitimate training rows. Task 2 follows the as-of-origin
rule above. Kaggle's two final selections were automatic (the two best public files: v15b main and v15b hedge, both
0.87514 public); the main file is the one whose private score counts.

## Author

Emre Cirak — MSc student, Gazi University (Ankara, Türkiye). Kaggle: `biohack44`.

## License

MIT (see `LICENSE`).
