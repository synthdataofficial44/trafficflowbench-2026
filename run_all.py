#!/usr/bin/env python3
"""TrafficFlowBench 2026 (IEEE BigData Cup) - team "Emre Cirak" (Kaggle: biohack44) - single entry script.

Rebuilds the selected final Kaggle submission

    submission_20261025_1742_v15b_main.zip     private 0.88253 (rank 55 of 802), public 0.87514
    = Task 1 (+ Task 3) task1_v15.parquet  +  Task 2 task2_v15_0.csv  +  Task 4 task4.csv

from the competition data, by running the stages of the original Colab runs (v8 -> v22) in their order, with their
settings (configs/final.json) and the package in tfb/ (byte-identical to the code that produced the submission; the
`check` stage verifies this against docs/manifest_v22.json).

    python run_all.py --data /path/to/kaggle_public                   # everything: train, predict, assemble, checks
    python run_all.py --data ... --work /path/to/tfb_work              # reuse trained artifacts: every finished stage reloads
    python run_all.py --data ... --stages task4,task2,files            # a subset of the stages
    python run_all.py --data ... --stages asof                         # only the Task 2 as-of-origin input check

Stages (default: all, in this order):
    check   code hashes of tfb/ against the manifest of the selected submission
    caches  data preflight + dense per-panel arrays
    task4   L2 projection of the ODME prior onto the link counts                 -> pred/task4.csv
    task1   v8 sets -> v10 lab -> v11 lab -> v13 set -> v14 blend -> v14 set -> v15 blend + lab rerun
                                                                                -> pred/task1_v15.parquet
    task2   v12 models -> v13.1 onset weighting -> v15 files (fallback main)    -> pred/task2_v15_0.csv
    files   the upload file (template order), checks, zip, md5 against the selected submission
    asof    Task 2 inputs rebuilt with every row after each window's origin blanked (must equal the full-data inputs)

Every stage writes its result to --work and is skipped when a result for the same code, settings and data is
already there, so an interrupted run resumes. Outputs: <out>/submission.csv (+ .zip) and <out>/run_report.json.
"""
from __future__ import annotations

import argparse
import contextlib
import glob
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
STAGES = ("check", "caches", "task4", "task1", "task2", "files", "asof")
QDT = {"window_id": str, "timestamp": str, "link_id": str}
ODT = {"panel": str, "departure_time": str, "path_id": str}


# ----------------------------------------------------------------------------- helpers
def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def md5(path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 24), b""):
            h.update(blk)
    return h.hexdigest()


def jdump(obj, path):
    path = str(path)
    with open(path + ".part", "w") as f:
        json.dump(obj, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    os.replace(path + ".part", path)


def jload(path) -> dict:
    p = Path(path)
    return json.load(open(p)) if p.exists() else {}


def load_cfg(path) -> dict:
    cfg = json.load(open(path))
    # JSON keys are strings; the notebooks passed the Task 1 per-seed overrides with int keys
    cfg["T1"]["seed_cfg"] = {int(k): v for k, v in cfg["T1"].get("seed_cfg", {}).items()}
    return cfg


def release_root(data: str) -> Path:
    """The `kaggle_public` folder (the one holding config/corridors.json), given it or any folder above it."""
    d = Path(data)
    if (d / "config" / "corridors.json").exists():
        return d
    hits = sorted(glob.glob(str(d / "**" / "config" / "corridors.json"), recursive=True))
    if not hits:
        raise FileNotFoundError(f"no config/corridors.json under {d}: pass the unpacked competition data (kaggle_public)")
    return Path(hits[0]).parent.parent


@contextlib.contextmanager
def catboost_gpu(allowed: bool):
    """The CatBoost member of Task 1 is trained only when a GPU is visible (cb='auto'). The original v8 and v14 runtimes
    had none, and in v15 the CatBoost blends were evaluated and rejected, so by default the GPU is hidden from CatBoost:
    the chosen blends are then the original ones whatever the runtime. --catboost lets it see the GPU (as in v15)."""
    import tfb.task1 as T1
    orig = T1._cb_gpu
    if not allowed:
        T1._cb_gpu = lambda: False
    try:
        yield
    finally:
        T1._cb_gpu = orig


def link_work_dirs(work: Path):
    """A copy of the Colab Drive folder tfb_work keeps the v14 / v15 lab folders as tfb_work/w14 and tfb_work/w15, while
    the code reads them as work14 / work15 next to the work folder: link them there when only the Drive copy exists."""
    for drive_name, local_name in (("w14", "work14"), ("w15", "work15")):
        src, dst = work / drive_name, work.parent / local_name
        if src.is_dir() and not dst.exists():
            os.symlink(src, dst, target_is_directory=True)
            log(f"linked {dst} -> {src}")


# ----------------------------------------------------------------------------- stages
def stage_check(cfg, a, R):
    man = jload(HERE / "docs" / "manifest_v22.json")
    want = man.get("code_sha256") or {}
    got = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((HERE / "tfb").glob("*.py"))}
    diff = sorted(f for f in want if got.get(f) != want[f])
    R["code"] = dict(files=len(got), identical_to_submission=not diff and bool(want), differ=diff)
    log("code check:", "tfb/ is byte-identical to the code of the selected submission" if not diff and want
        else f"files that differ from the manifest: {diff}")
    import numpy, pandas, lightgbm, pyarrow   # noqa: F401
    R["packages"] = {m.__name__: m.__version__ for m in (numpy, pandas, lightgbm, pyarrow)}
    R["packages"]["python"] = sys.version.split()[0]
    used = man.get("packages") or {}
    log("packages:", R["packages"], "| used for the submission:", {k: used.get(k) for k in ("python", "numpy", "pandas",
                                                                                              "lightgbm", "pyarrow")})


def stage_caches(cfg, a, R):
    import tfb.common as C
    import tfb.data as D
    t0 = time.time()
    pf = D.preflight()
    log("preflight:\n" + pf.to_string())
    D.build_all(n_jobs=a.cache_jobs)
    log("calendar:\n" + C.calendar().groupby("split").day.agg(["min", "max", "count"]).to_string())
    R["minutes"]["caches"] = round((time.time() - t0) / 60, 1)


def stage_task4(cfg, a, R):
    import tfb.common as C
    import tfb.task4 as T4
    t0 = time.time()
    rep = T4.run()
    f = C.WORK / "pred" / "task4.csv"
    R["task4"] = dict(file=f.name, md5=md5(f), S_link_min=float(rep.S_link.min()),
                      dev_over_prior_mean=float(rep.dev_over_prior.mean()))
    R["minutes"]["task4"] = round((time.time() - t0) / 60, 1)
    log("Task 4:", R["task4"])


def _lab10(cfg1, labc, R):
    """v10 notebook, step 7: stage 2 (s2) on the v8 answer, the held-out March check, the test rows."""
    import tfb.common as C
    import tfb.task1 as T1
    import tfb.t1lab as LB
    lab_path = C.WORK / "t1lab" / "lab_report.json"
    lab = jload(lab_path)
    if lab.get("key") != LB.lab_key(cfg1, labc):
        lab = LB.run_lab(cfg1, labc)
    t1_v10 = C.WORK / "pred" / "task1_v10.parquet"
    app_path = C.WORK / "t1lab" / "applied.json"
    app = jload(app_path)
    same = app.get("key") == lab.get("key") and app.get("candidate") == lab.get("candidate")
    if not lab.get("adopted"):
        src = "v8"
    elif same and app.get("tm") and not app["tm"].get("ok"):
        src = "v8"
    elif same and app.get("tm") and t1_v10.exists() and app.get("md5") == T1.file_md5(t1_v10):
        src = "v10"
    else:
        LB.prepare_rows(cfg1)
        tm = app["tm"] if same and app.get("tm") else LB.tm_check(lab, cfg1, labc)
        jdump(dict(key=lab["key"], candidate=lab["candidate"], tm=tm, md5=None), app_path)
        if tm.get("ok"):
            LB.apply_test(lab, cfg1, labc)
            jdump(dict(key=lab["key"], candidate=lab["candidate"], tm=tm, md5=T1.file_md5(t1_v10)), app_path)
            src = "v10"
        else:
            src = "v8"
    R["task1"]["v10_lab"] = dict(candidate=lab.get("candidate"), adopted=bool(lab.get("adopted")), answer=src)
    log("Task 1 v10 lab:", R["task1"]["v10_lab"])


def _lab11(cfg1, labc, R):
    """v11 notebook, step 8: stage 3 on top of the v10 answer (or the v8 answer when v10 was not applied)."""
    import tfb.common as C
    import tfb.task1 as T1
    import tfb.t1lab as LB
    import tfb.t1lab11 as L11
    lab10 = jload(C.WORK / "t1lab" / "lab_report.json")
    app10 = jload(C.WORK / "t1lab" / "applied.json")
    t1_v10 = C.WORK / "pred" / "task1_v10.parquet"
    v10_t1 = bool(lab10.get("adopted") and (app10.get("tm") or {}).get("ok") and t1_v10.exists()
                  and app10.get("md5") == T1.file_md5(t1_v10))
    base_chain = [[f, p] for f, p in (lab10.get("candidate_chain") or [])] if v10_t1 else []
    if not all(f == "s2" for f, _ in base_chain):
        raise RuntimeError(f"the v10 Task 1 chain {base_chain} has steps the v11 lab does not rebuild")
    base = "v10" if v10_t1 else "v8"
    lab_path = C.WORK / "t1lab" / "lab11_report.json"
    lab = jload(lab_path)
    if lab.get("key") != L11.lab_key11(cfg1, labc, base_chain):
        v10_dj = ((lab10.get("variants") or {}).get(lab10.get("candidate")) or {}).get("d_J") if v10_t1 else None
        lab = L11.run_lab11(cfg1, labc, base_chain, v10_dj=v10_dj)
    t1_v11 = C.WORK / "pred" / "task1_v11.parquet"
    app_path = C.WORK / "t1lab" / "applied11.json"
    app = jload(app_path)
    same = app.get("key") == lab.get("key") and app.get("candidate") == lab.get("candidate")
    if not lab.get("adopted"):
        src = base
    elif same and app.get("tm") and not app["tm"].get("ok"):
        src = base
    elif same and app.get("tm") and t1_v11.exists() and app.get("md5") == T1.file_md5(t1_v11):
        src = "v11"
    else:
        LB.prepare_rows(cfg1)
        tm = app["tm"] if same and app.get("tm") else L11.tm_check11(lab, cfg1, labc)
        jdump(dict(key=lab["key"], candidate=lab["candidate"], tm=tm, md5=None), app_path)
        if tm.get("ok"):
            chk = L11.apply_test11(lab, cfg1)
            jdump(dict(key=lab["key"], candidate=lab["candidate"], tm=tm, md5=T1.file_md5(t1_v11), apply=chk), app_path)
            src = "v11"
        else:
            src = base
    R["task1"]["v11_lab"] = dict(base_chain=base_chain, candidate=lab.get("candidate"), adopted=bool(lab.get("adopted")),
                                 answer=src)
    log("Task 1 v11 lab:", R["task1"]["v11_lab"])


def stage_task1(cfg, a, R):
    import tfb.common as C
    import tfb.task1 as T1
    import tfb.t1pre13 as P13
    import tfb.t1v14 as V
    import tfb.t1v15 as V5
    t0 = time.time()
    cfg1 = cfg["T1"]
    R["task1"] = {}
    with catboost_gpu(a.catboost):
        # v8 notebook (steps 7 and 10): rows, the three LightGBM sets, the blend, the post-processing -> pred/task1.parquet
        ok, why = T1.result_up_to_date(cfg1)
        log("Task 1 v8:", "found - not recomputed" if ok else f"running ({why})")
        if not ok:
            T1.run_task1(cfg1)
        R["task1"]["v8_choice"] = {k: jload(C.WORK / "t1" / "choice.json").get(k) for k in ("members", "pp")}
        # v10 and v11 notebooks: the physics lab (stage 2, then stage 3) on the v8 answer
        _lab10(cfg1, cfg["LAB10"], R)
        _lab11(cfg1, cfg["LAB11"], R)
        # v13 notebook (step 11): the fourth set (s3)
        R["task1"]["v13_set"] = P13.train(cfg1, cfg["T1_NEW"])
        # v14 notebook (step 7): the blend with s3 and the lab rerun on it -> pred/task1_v14.parquet
        ok, rep14 = V.up_to_date(cfg1, cfg["T1_14"])
        if not ok:
            rep14 = V.run(cfg1, cfg["T1_14"])
        R["task1"]["v14"] = dict(adopted=rep14.get("adopted"), blend=(rep14.get("choice") or {}).get("chosen"),
                                 reason=rep14.get("reason"))
        log("Task 1 v14:", R["task1"]["v14"])
        # v14 notebook (step 12): the fifth set (s4) and its score against the v14 blend -> t1/t1next_report.json
        R["task1"]["v14_next_set"] = P13.train(cfg1, cfg["T1_NEXT"])
        nx = C.WORK / "t1" / "t1next_report.json"
        if not (nx.exists() and jload(nx).get("new_set") == P13.new_set(cfg["T1_NEXT"])):
            V.score_next(cfg1, cfg["T1_NEXT"], cfg["T1_14"])
        # v22 notebook (step 11, "Task 1 v15"): the blend with s4 and the lab rerun in work15 -> pred/task1_v15.parquet
        ok, rep15 = V5.up_to_date(cfg1, cfg["T1_15"])
        if not ok:
            rep15 = V5.run(cfg1, cfg["T1_15"])
        R["task1"]["v15"] = dict(adopted=rep15.get("adopted"), blend=(rep15.get("choice") or {}).get("chosen"),
                                 weights=(rep15.get("choice") or {}).get("weights"), cb=rep15.get("cb"),
                                 reason=rep15.get("reason"))
        log("Task 1 v15:", R["task1"]["v15"])
    R["minutes"]["task1"] = round((time.time() - t0) / 60, 1)


def task1_answer(R) -> str:
    """The Task 1 file of the selected submission (task1_v15.parquet); when a rerun's own gates kept an earlier answer,
    that answer is used and reported."""
    import tfb.common as C
    d = C.WORK / "pred"
    order = [("task1_v15.parquet", jload(C.WORK / "t1" / "t1v15_report.json").get("adopted")),
             ("task1_v14.parquet", jload(C.WORK / "t1" / "t1v14_report.json").get("adopted")),
             ("task1_v11.parquet", bool((jload(C.WORK / "t1lab" / "applied11.json").get("tm") or {}).get("ok"))),
             ("task1_v10.parquet", bool((jload(C.WORK / "t1lab" / "applied.json").get("tm") or {}).get("ok"))),
             ("task1.parquet", True)]
    for f, ok in order:
        if ok and (d / f).exists():
            if f != "task1_v15.parquet":
                log(f"WARNING: Task 1 v15 was not adopted in this run; the submission uses {f}")
            R.setdefault("task1", {})["answer"] = f
            return f
    raise FileNotFoundError("no Task 1 answer in pred/: run the task1 stage")


def stage_task2(cfg, a, R):
    import tfb.task2y as Y
    import tfb.task2z as Z
    import tfb.task2v15 as Z15
    t0 = time.time()
    cfg2 = dict(cfg["T2"], model_dir=str(a.model_dir))
    # v12 notebook (step 9): every Task 2 model (standard, gate, April refit, late blocks) + probabilities + files
    ok, rep12 = Y.up_to_date(cfg2)
    log("Task 2 v12:", "found - not recomputed" if ok else "running")
    if not ok:
        rep12 = Y.run_main(cfg2, device={})
    # v13.1 notebook (step 7): the location-light onset weighting (nl75), cross-draw rule on the March pseudo windows
    ok, rep13 = Z.up_to_date(cfg2, cfg["T2_13"])
    log("Task 2 v13.1:", "found - not recomputed" if ok else "running")
    if not ok:
        rep13 = Z.run_main(cfg2, cfg["T2_13"], device={})
    # v15 notebook (step 7): the v15 files; the fallback main task2_v15_0.csv = v13.1's April rows + v12's March rows
    ok, rep15 = Z15.up_to_date(cfg2, cfg["T2_15"])
    log("Task 2 v15:", "found - not recomputed" if ok else "running")
    if not ok:
        rep15 = Z15.run(cfg2, cfg["T2_15"], device={})
    R["task2"] = dict(v12_chosen=rep12.get("chosen"), v12_decoder=(rep12.get("decoder") or {}).get("adopted"),
                      v12_april_adapt={c: (d or {}).get("adopted") for c, d in (rep12.get("adapt") or {}).items()},
                      v12_late={c: (d or {}).get("adopted") for c, d in (rep12.get("late") or {}).items()},
                      v131_adopted=(rep13.get("gate") or {}).get("adopted"),
                      v15_changed=rep15.get("changed"), file=Z15.MAIN0_FILE)
    R["minutes"]["task2"] = round((time.time() - t0) / 60, 1)
    log("Task 2:", R["task2"])


def stage_files(cfg, a, R):
    import pandas as pd
    import tfb.common as C
    import tfb.submit as S
    t0 = time.time()
    fin = cfg["FINAL"]
    d = C.WORK / "pred"
    t1f, t2f, t4f = task1_answer(R), fin["task2_file"], fin["task4_file"]
    for f in (t1f, t2f, t4f):
        if not (d / f).exists():
            raise FileNotFoundError(f"pred/{f} is missing: run the stage that writes it")
    comp = {f: md5(d / f) for f in (t1f, t2f, t4f)}
    st = pd.read_parquet(d / t1f)
    qq = pd.read_csv(d / t2f, dtype=QDT)
    od = pd.read_csv(d / t4f, dtype=ODT)
    out_csv = a.out / "submission.csv"
    S.build(st, qq, od, out_csv)
    got = md5(out_csv)
    same = {f: comp[f] == fin["components_md5"].get(f) for f in comp}
    R["files"] = dict(csv=str(out_csv), zip=str(out_csv.with_suffix(".zip")), csv_md5=got,
                      identical_to_selected=got == fin["selected_csv_md5"], components=comp, components_identical=same,
                      selected=dict(zip=fin["selected_zip"], csv_md5=fin["selected_csv_md5"],
                                    private=fin["private_score"], public=fin["public_score"]))
    R["minutes"]["files"] = round((time.time() - t0) / 60, 1)
    if got == fin["selected_csv_md5"]:
        log("SUBMISSION: byte-identical to the selected final submission", fin["selected_zip"])
    else:
        log("SUBMISSION written:", out_csv, "| md5", got, "| components identical to the selected one's:", same,
            "| (a full retrain is not bit-exact; the award rule asks for the private score within 1 %)")


def stage_asof(cfg, a, R):
    import tfb.task2z as Z
    import tfb.asof as AS
    t0 = time.time()
    Z.install()
    rep = AS.check_inputs(cfg["T2"], per_split=cfg["ASOF"]["per_split"], seed=cfg["ASOF"]["seed"])
    jdump(rep, a.out / "asof_report.json")
    R["asof"] = {k: rep.get(k) for k in ("passed", "windows", "model_columns", "all_columns_equal", "minutes")}
    R["minutes"]["asof"] = round((time.time() - t0) / 60, 1)
    log("AS-OF-ORIGIN INPUT CHECK:", "PASSED" if rep.get("passed") else "FAILED", R["asof"])
    if rep.get("differences"):
        log("differences:", json.dumps(rep["differences"][:4], default=str)[:3000])


RUN = dict(check=stage_check, caches=stage_caches, task4=stage_task4, task1=stage_task1, task2=stage_task2,
           files=stage_files, asof=stage_asof)


# ----------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="the unpacked competition data (kaggle_public, or a folder above it)")
    ap.add_argument("--work", default="./work", help="work folder: models, predictions, reports (default ./work)")
    ap.add_argument("--cache", default="./cache", help="dense per-panel arrays (default ./cache)")
    ap.add_argument("--out", default="./output", help="submission.csv, run_report.json, asof_report.json")
    ap.add_argument("--model-dir", default=None, help="Task 2 checkpoints (default <work>/t2_models)")
    ap.add_argument("--stages", default=",".join(STAGES), help=f"comma-separated subset of {','.join(STAGES)}")
    ap.add_argument("--config", default=str(HERE / "configs" / "final.json"))
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 2, help="LightGBM / numpy threads")
    ap.add_argument("--cache-jobs", type=int, default=None, help="processes for the dense caches (default cpus/2, max 4)")
    ap.add_argument("--catboost", action="store_true", help="let the Task 1 CatBoost member use a GPU (see catboost_gpu)")
    a = ap.parse_args(argv)
    stages = [s.strip() for s in a.stages.split(",") if s.strip()]
    bad = [s for s in stages if s not in STAGES]
    if bad:
        ap.error(f"unknown stages {bad}; choose from {STAGES}")
    a.work, a.cache, a.out = Path(a.work).resolve(), Path(a.cache).resolve(), Path(a.out).resolve()
    a.model_dir = Path(a.model_dir).resolve() if a.model_dir else a.work / "t2_models"
    a.cache_jobs = a.cache_jobs or min(4, max(1, (os.cpu_count() or 2) // 2))
    for p in (a.work, a.cache, a.out, a.model_dir):
        p.mkdir(parents=True, exist_ok=True)
    os.environ["TFB_THREADS"] = str(a.threads)
    rel = release_root(a.data)
    import tfb.common as C
    C.THREADS = a.threads
    C.set_paths(data=rel, cache=a.cache, work=a.work)
    link_work_dirs(a.work)
    cfg = load_cfg(a.config)
    rp = a.out / "run_report.json"
    R = dict(started=time.strftime("%Y-%m-%d %H:%M:%S"), data=str(rel), work=str(a.work), stages=stages,
             threads=a.threads, catboost_gpu=bool(a.catboost), minutes={})
    log(f"TrafficFlowBench final package | data {rel} | work {a.work} | stages {stages} | threads {a.threads}")
    t0 = time.time()
    try:
        for s in [s for s in STAGES if s in stages]:
            log(f"===== stage {s}")
            RUN[s](cfg, a, R)
            jdump(R, rp)
    except Exception as e:
        R["error"] = f"{type(e).__name__}: {e}"
        R["traceback"] = traceback.format_exc()[-4000:]
        jdump(R, rp)
        raise
    R["minutes"]["total"] = round((time.time() - t0) / 60, 1)
    R["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    jdump(R, rp)
    log("done in", R["minutes"]["total"], "min | report:", rp)
    return R


if __name__ == "__main__":
    main()
