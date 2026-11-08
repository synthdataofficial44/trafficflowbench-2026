"""Control-flow test of run_all.py without data: every heavy stage function of tfb/ is replaced by a stub that checks
its call against the real function's signature and writes the files the real one writes. Run: python tests/test_run_all_flow.py
"""
from __future__ import annotations

import inspect
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import run_all  # noqa: E402
import tfb.common as C  # noqa: E402
import tfb.data as D  # noqa: E402
import tfb.task1 as T1  # noqa: E402
import tfb.task4 as T4  # noqa: E402
import tfb.t1lab as LB  # noqa: E402
import tfb.t1lab11 as L11  # noqa: E402
import tfb.t1pre13 as P13  # noqa: E402
import tfb.t1v14 as V  # noqa: E402
import tfb.t1v15 as V5  # noqa: E402
import tfb.task2y as Y  # noqa: E402
import tfb.task2z as Z  # noqa: E402
import tfb.task2v15 as Z15  # noqa: E402
import tfb.submit as S  # noqa: E402
import tfb.asof as AS  # noqa: E402

CALLS: list[str] = []
STATE: dict = {}


def stub(mod, name, body):
    real = getattr(mod, name)
    sig = inspect.signature(real)

    def f(*a, **k):
        sig.bind(*a, **k)                       # raises TypeError on a wrong call
        CALLS.append(f"{mod.__name__.split('.')[-1]}.{name}")
        return body(*a, **k)
    setattr(mod, name, f)


def wjson(p: Path, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    json.dump(obj, open(p, "w"))


def install_stubs():
    W = lambda: C.WORK  # noqa: E731
    pq = lambda f: pd.DataFrame({"panel": ["P"], "timestamp": ["t"], "link_id": ["L"], "speed_kmh": [90.0],  # noqa: E731
                                 "flow_vph": [1000.0]}).to_parquet(f)
    stub(D, "preflight", lambda strict=True: pd.DataFrame({"ok": [True]}))
    stub(D, "build_all", lambda n_jobs=2, force=False: None)
    stub(C, "calendar", lambda: pd.DataFrame({"split": ["train"], "day": [0]}))

    def t4_run(splits=C.TEST_SPLITS):
        (W() / "pred").mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"panel": ["P"], "departure_time": ["d"], "path_id": ["p"], "path_flow": [1.0]}).to_csv(
            W() / "pred" / "task4.csv", index=False)
        return pd.DataFrame({"S_link": [1.0], "dev_over_prior": [0.1]})
    stub(T4, "run", t4_run)

    stub(T1, "result_up_to_date", lambda cfg, work=None: (STATE.get("v8", False), "stub"))

    def run_task1(cfg, on_set_done=None):
        assert isinstance(cfg["seed_cfg"], dict) and 2 in cfg["seed_cfg"], "seed_cfg keys must be ints"
        (W() / "pred").mkdir(parents=True, exist_ok=True)
        pq(W() / "pred" / "task1.parquet")
        wjson(W() / "t1" / "choice.json", dict(members={"lgb0": {"weight": 1 / 3}}, pp={"gate": 0.6}))
        STATE["v8"] = True
    stub(T1, "run_task1", run_task1)
    stub(LB, "lab_key", lambda cfg1, lab=None, work=None: {"k": "lab10"})
    stub(LB, "run_lab", lambda cfg1, lab=None, device=None: (wjson(W() / "t1lab" / "lab_report.json", d := dict(
        key={"k": "lab10"}, adopted=True, candidate="s2_x1.0", candidate_chain=[["s2", {"shrink": 1.0}]],
        variants={"s2_x1.0": {"d_J": 0.0006}})), d)[1])
    stub(LB, "prepare_rows", lambda cfg1: W() / "t1" / "rows_r1")
    stub(LB, "tm_check", lambda rep, cfg1, lab=None, device=None, s2m=None: {"ok": True, "d": 0.001})
    stub(LB, "apply_test", lambda rep, cfg1, lab=None, device=None, s2m=None: pq(W() / "pred" / "task1_v10.parquet"))
    stub(L11, "lab_key11", lambda cfg1, lab, base_chain, work=None: {"k": "lab11", "chain": base_chain})

    def run_lab11(cfg1, lab=None, base_chain=(), device=None, v10_dj=None):
        assert base_chain == [["s2", {"shrink": 1.0}]], base_chain
        assert v10_dj == 0.0006, v10_dj
        d = dict(key={"k": "lab11", "chain": base_chain}, adopted=True, candidate="s3vq")
        wjson(W() / "t1lab" / "lab11_report.json", d)
        return d
    stub(L11, "run_lab11", run_lab11)
    stub(L11, "tm_check11", lambda rep, cfg1, lab=None: {"ok": True})
    stub(L11, "apply_test11", lambda rep, cfg1: (pq(W() / "pred" / "task1_v11.parquet"), {"compared": False})[1])
    stub(P13, "train", lambda cfg1, new=None, on_done=None: {"status": "trained", "dir": f"models_r1_s{new['seed']}"})
    stub(V, "up_to_date", lambda cfg1, st=None: (STATE.get("v14", False), jload_or_none(W() / "t1" / V.REPORT)))

    def v14_run(cfg1, st=None, on_step=None):
        pq(W() / "pred" / "task1_v14.parquet")
        d = dict(adopted=True, choice={"chosen": "new_x2"})
        wjson(W() / "t1" / V.REPORT, d)
        STATE["v14"] = True
        return d
    stub(V, "run", v14_run)

    def score_next(cfg1, new, st=None):
        wjson(W() / "t1" / "t1next_report.json", dict(new_set=P13.new_set(new)))
        STATE["next"] = STATE.get("next", 0) + 1
        return {}
    stub(V, "score_next", score_next)
    stub(V5, "up_to_date", lambda cfg1, st=None: (STATE.get("v15", False), jload_or_none(W() / "t1" / V5.REPORT)))

    def v15_run(cfg1, st=None, on_step=None):
        assert T1._cb_gpu() is False, "CatBoost must not see a GPU by default"
        pq(W() / "pred" / "task1_v15.parquet")
        d = dict(adopted=True, choice={"chosen": "new_x2", "weights": {"lgb4": 0.83}}, cb="no GPU in this runtime")
        wjson(W() / "t1" / V5.REPORT, d)
        STATE["v15"] = True
        return d
    stub(V5, "run", v15_run)
    stub(Y, "up_to_date", lambda cfg, work=None: (STATE.get("t2v12", False), {"chosen": {}} if STATE.get("t2v12") else None))

    def y_run(cfg, device=None):
        assert cfg["model_dir"].endswith("t2_models") and device == {}
        STATE["t2v12"] = True
        return dict(chosen={"queue_onset": ["v2+pm", ["all", "noloc"]]}, decoder={"adopted": "topm"},
                    adapt={"queue_onset": {"adopted": True}}, late={"queue_ongoing": {"adopted": True}})
    stub(Y, "run_main", y_run)
    stub(Z, "up_to_date", lambda cfg, cfg13, work=None: (STATE.get("t2v131", False), {} if STATE.get("t2v131") else None))

    def z_run(cfg, cfg13=None, device=None):
        assert cfg13["rule13"] == "cross_draw"
        STATE["t2v131"] = True
        return dict(gate={"adopted": "nl75"})
    stub(Z, "run_main", z_run)
    stub(Z15, "up_to_date", lambda cfg, st, work=None: (STATE.get("t2v15", False), {} if STATE.get("t2v15") else None))

    def z15_run(cfg, st=None, device=None):
        pd.DataFrame({"window_id": ["w"], "timestamp": ["t"], "link_id": ["L"], "queue_pred": [1]}).to_csv(
            W() / "pred" / Z15.MAIN0_FILE, index=False)
        STATE["t2v15"] = True
        return dict(changed={"ongoing": True, "onset": False})
    stub(Z15, "run", z15_run)

    def build(state, queue, odme, out_csv, drop_tasks=()):
        assert len(state) and len(queue) and len(odme)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"submission_id": [0]}).to_csv(out_csv, index=False)
        return pd.DataFrame()
    stub(S, "build", build)
    stub(AS, "check_inputs", lambda cfg, panels=None, per_split=2, seed=0, splits=("validation", "private"): dict(
        passed=True, windows=32, model_columns={}, all_columns_equal=True, minutes=0.1, differences=[]))


def jload_or_none(p: Path):
    return json.load(open(p)) if p.exists() else None


def main():
    tmp = Path(tempfile.mkdtemp())
    rel = tmp / "data" / "kaggle_public"
    (rel / "config").mkdir(parents=True)
    (rel / "config" / "corridors.json").write_text("{}")
    install_stubs()
    args = ["--data", str(tmp / "data"), "--work", str(tmp / "work"), "--cache", str(tmp / "cache"),
            "--out", str(tmp / "out"), "--threads", "2"]
    R = run_all.main(args)
    first = list(CALLS)
    assert R["task1"]["v15"]["adopted"] and R["task1"]["answer"] == "task1_v15.parquet", R["task1"]
    assert R["task2"]["file"] == "task2_v15_0.csv" and R["task2"]["v131_adopted"] == "nl75", R["task2"]
    assert R["asof"]["passed"] and R["code"]["identical_to_submission"], (R["asof"], R["code"])
    assert (tmp / "out" / "run_report.json").exists()
    for f in ("task1.run_task1", "t1lab.run_lab", "t1lab.tm_check", "t1lab.apply_test", "t1lab11.run_lab11",
              "t1lab11.tm_check11", "t1lab11.apply_test11", "t1pre13.train", "t1v14.run", "t1v14.score_next",
              "t1v15.run", "task2y.run_main", "task2z.run_main", "task2v15.run", "submit.build", "asof.check_inputs",
              "task4.run", "data.preflight", "data.build_all"):
        assert f in first, f"{f} was not called"
    # a second run reuses every finished stage
    CALLS.clear()
    R2 = run_all.main(args)
    heavy = {"task1.run_task1", "t1lab.run_lab", "t1lab11.run_lab11", "t1v14.run", "t1v14.score_next", "t1v15.run",
             "task2y.run_main", "task2z.run_main", "task2v15.run", "t1lab.apply_test", "t1lab11.apply_test11"}
    again = sorted(set(CALLS) & heavy)
    assert not again, f"recomputed on the second run: {again}"
    assert R2["files"]["components"], R2["files"]
    print("first run calls:", first)
    print("second run calls:", CALLS)
    print("OK: control flow, signatures and resume behaviour")
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
