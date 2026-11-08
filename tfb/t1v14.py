"""v14 Task 1: the blend of the four LightGBM sets (and a CatBoost member when the runtime has a GPU), with the stage-2/3
lab of v10/v11 rerun on top of it, compared with the v11 answer before it replaces it.

Why. The fourth set (v13: a 3x transductive sample of the test months, up to 3500 rounds) is the best single set on
held-out flow and density, and in blends it gained +0.0006-0.0007 S_state on the held-out observed March cells *and* on
the held-out April cells (April: the private month itself; Task 1 is offline, so its own observed cells are a legitimate
ruler) - small, but it is the one gain in this competition that is measured on the month that decides. It was not
applied in v13: the v11 answer is the v8 blend (sets 0-2) + the v10 stage 2 + the v11 stage 3, and those stages are
fitted on the blend's own errors, so a new blend needs them refitted.

Steps (all on the rows and the test-month holdout cells of the v8 plan - same code, same settings, same cells):
  1. members: the plan's sets (0-2), the v13 set (3) and, with a GPU, a CatBoost member (v8 support, never trained on the
     real data before because the runtimes had no GPU);
  2. blend: every candidate (base = the v11 blend; + set 3 at an average weight, at twice that, sets 2 + 3 alone; with
     the CatBoost member at 25 / 40 %) scored by `task1.choose_blend` (raw: March held-out cells decide - gain > 2 SEs -
     and the train-holdout J must not fall by more than 2e-4); among those that pass, the one with the best
     *post-processed* train-holdout J (the post-processing grid of v8 per candidate) - J = 0.35 S_state + 0.10 S_LWR is
     the part of the total score Task 1 moves, and only the train holdout measures S_LWR;
  3. the lab: in its own work folder (work14: links to the member sets and rows, its own choice.json / task1.parquet /
     t1lab), the v11 lab on top of the new answer with the v10 stage 2 as its base chain - stage 2 out of fold, the
     stage-3 heads out of fold, the v11 gate (J > 2 day-bootstrap SEs on both halves of the holdout days, S_state not
     down), then S_state on the held-out March cells (tm_check11), then the chain on the test rows (apply_test11);
  4. against the v11 answer: the final chain's train-holdout J must not be below v11's (lab11_report: B1 J + the stage-3
     gain) and its S_state on the held-out March cells not below v11's (applied11.json) by more than `final_tol_tm`.
     Only then pred/task1_v14.parquet is written. April is never used to decide; nothing in WORK/t1 or WORK/pred of the
     earlier versions is changed.
Also: `score_next` scores a further set against the v14 blend (report only, for the next version).
"""
from __future__ import annotations

import contextlib
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import common as C
from . import task1 as T1
from . import t1lab as L
from . import t1lab11 as L11

VERSION = "v14"
OUT_FILE = "task1_v14.parquet"
REPORT = "t1v14_report.json"
W14 = "work14"
BASE_CHAIN = [["s2", {"shrink": 1.0}]]          # the v10 chain, the base of the v11 lab
LAB11 = dict(z=2.0, tol_state=0.0005, tol_tm=0.0003, boot=400, s2_rounds=400, s2_folds=3, s2_max_rows=3_000_000,
             s3_rounds=400, s3_max_rows=3_000_000, tm_check=True, force=None, candidates=list(L11.CANDS))
DEFAULTS = dict(new_seeds=[3], cb="auto", cb_weights=[0.25, 0.4], tm_z=2.0, hold_tol=0.0002, pp_top=4, final_tol_J=0.0,
                final_tol_tm=0.0001, force_blend=None)


def settings(st: dict | None) -> dict:
    s = dict(DEFAULTS)
    s.update({k: v for k, v in (st or {}).items() if v is not None or k == "force_blend"})
    unknown = set(s) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"t1v14 settings: unknown keys {sorted(unknown)}")
    if s["cb"] not in ("auto", True, False):
        raise ValueError("t1v14 cb must be 'auto', True or False")
    return s


def work14() -> Path:
    return C.WORK.parent / W14


@contextlib.contextmanager
def in_work(path: Path):
    """Run the lab modules (which read C.WORK) in another work folder; the old one is restored afterwards."""
    old = C.WORK
    C.set_paths(work=path)
    try:
        yield
    finally:
        C.set_paths(work=old)


def _link(src: Path, dst: Path):
    """dst -> src (a symlink); an existing link is replaced, an existing real directory is left alone."""
    if dst.is_symlink():
        if Path(os.readlink(dst)) == src:
            return
        dst.unlink()
    elif dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(src, dst, target_is_directory=src.is_dir())


# ----------------------------------------------------------------------------- members and candidates
def lgb_members(cfg1: dict, st: dict) -> dict:
    """{name: ('lgb', dir)} of the complete LightGBM sets: the plan's and the newer ones (`new_seeds`)."""
    tag = cfg1.get("rows_tag", "r1")
    out = {}
    for s in list(T1.plan_seeds(cfg1)) + [int(x) for x in st["new_seeds"]]:
        d = C.WORK / "t1" / f"models_{tag}_s{s}"
        if T1.set_complete(d):
            out[f"lgb{s}"] = ("lgb", d)
    return out


def cb_member(cfg1: dict, st: dict, on_done=None) -> tuple:
    """(dir or None, note): the CatBoost member of the v8 plan, trained here when the runtime has a GPU (`cb` 'auto') or
    when forced (`cb` True: also on the CPU with cfg1's cb_cpu settings). A complete member with the same signature is
    reused."""
    mode = st["cb"]
    if mode is False:
        return None, "off"
    gpu = T1._cb_gpu()
    if mode == "auto" and not gpu:
        return None, "no GPU in this runtime"
    cfg = dict(cfg1, cb_cpu=True) if mode is True else cfg1
    cbd = C.WORK / "t1" / f"cb_{cfg1.get('rows_tag', 'r1')}"
    want = T1.cb_signature(cfg, gpu)
    if T1.cb_complete(cbd) and json.dumps(T1.cb_meta(cbd).get("sig"), sort_keys=True) == json.dumps(want, sort_keys=True):
        return cbd, "reused"
    rows_dir = L.prepare_rows(cfg1)
    t0 = time.time()
    try:
        rep = T1.train_cb(rows_dir, cbd, cfg)
    except Exception as e:  # noqa: BLE001  a GPU / driver / install problem must not stop the Task 1 step
        C.log(f"T1 v14 CatBoost member failed: {type(e).__name__}: {e}")
        C.free()
        return None, f"failed: {type(e).__name__}: {str(e)[:200]}"
    if rep is None:
        return None, "skipped"
    if on_done is not None:
        on_done()
    return cbd, f"trained in {(time.time() - t0) / 60:.0f} min ({'GPU' if gpu else 'CPU'})"


def current_blend(work=None) -> dict:
    """{member name: weight} of the v11 answer (WORK/t1/choice.json)."""
    ch = json.load(open((work or C.WORK) / "t1" / "choice.json"))
    cur = {n: float(m["weight"]) for n, m in (ch.get("members") or {}).items()}
    if not cur:
        raise RuntimeError("t1v14: t1/choice.json lists no members")
    return cur


def candidates(cur: dict, mem: dict, st: dict) -> dict:
    """The blends: base (v11's), + each newer set at an average member weight and at twice that, sets 2 + the newest
    alone (the two enlarged transductive samples), and with a CatBoost member at its v8 weights on top of those."""
    w1 = float(np.mean(list(cur.values())))
    out = {"base": dict(cur)}
    new = [f"lgb{int(s)}" for s in st["new_seeds"] if f"lgb{int(s)}" in mem and f"lgb{int(s)}" not in cur]
    if new:
        out["add_new"] = dict(cur, **{n: w1 for n in new})
        out["new_x2"] = dict(cur, **{n: 2.0 * w1 for n in new})
        if "lgb2" in cur:
            out["new_s2"] = dict({"lgb2": 1.0}, **{n: 1.0 for n in new})
    if "cb" in mem:
        for nm in list(out):
            g = out[nm]
            if "cb" in g:   # the blend has the member already (an earlier answer with it)
                continue
            tot = sum(g.values())
            for w in st["cb_weights"]:
                out[f"{nm}+cb{w}"] = dict({k: (1.0 - float(w)) * v / tot for k, v in g.items()}, cb=float(w))
    return out


# ----------------------------------------------------------------------------- 2. the blend
def choose(cfg1: dict, mem: dict, cands: dict, st: dict) -> dict:
    """task1.choose_blend (raw: March held-out cells decide, the train holdout must not lose), then the post-processing
    grid per passing candidate on the train holdout: the best post-processed J wins (and must not be below base's by
    more than hold_tol). Returns the choice with the post-processing of the chosen blend."""
    rows_dir = L.prepare_rows(cfg1)
    t0 = time.time()
    rep = T1.choose_blend(rows_dir, mem, cands, dict(cfg1, tm_z=float(st["tm_z"]), hold_tol=float(st["hold_tol"])))
    rep.pop("chosen", None)
    ok = [c for c, d in rep["candidates"].items() if d.get("ok")]
    # the post-processing grid for the (at most `pp_top`) passing candidates with the best March scores, and for base
    ok = sorted(ok, key=lambda c: -float(rep["candidates"][c]["tm"].get("0", -1)))[: int(st["pp_top"])]
    grid = T1.pp_grid(cfg1.get("pp"))
    pp_tab = {}
    for c in ["base"] + ok:
        R_ = T1.run_holdout_eval_multi(rows_dir, [(mem[n][0], mem[n][1], w) for n, w in cands[c].items()], grid)
        summ = R_.groupby("variant")[["S_state", "S_LWR", "J"]].mean().sort_values("J", ascending=False)
        pp_tab[c] = dict(best=str(summ.index[0]), **{k: round(float(v), 6) for k, v in summ.iloc[0].items()})
    chosen = max(ok, key=lambda c: pp_tab[c]["J"]) if ok else "base"
    if chosen != "base" and pp_tab[chosen]["J"] < pp_tab["base"]["J"] - float(st["hold_tol"]):
        chosen = "base"
    if st.get("force_blend") in cands:   # tests only
        chosen = st["force_blend"]
        if chosen not in pp_tab:
            R_ = T1.run_holdout_eval_multi(rows_dir, [(mem[n][0], mem[n][1], w) for n, w in cands[chosen].items()], grid)
            summ = R_.groupby("variant")[["S_state", "S_LWR", "J"]].mean().sort_values("J", ascending=False)
            pp_tab[chosen] = dict(best=str(summ.index[0]), **{k: round(float(v), 6) for k, v in summ.iloc[0].items()})
    best = pp_tab[chosen]["best"]
    pp = dict(T1.DEFAULT_PP, gate=0.0, tv=False) if best == "raw" else dict(grid[int(best[2:])])
    T1._HOLD_PRED.clear()
    return dict(blend=rep, pp_table=pp_tab, passing=ok, chosen=chosen, weights=cands[chosen], pp=pp,
                forced=bool(st.get("force_blend") in cands), minutes=round((time.time() - t0) / 60, 1))


# ----------------------------------------------------------------------------- 3. the answer and the lab in work14
def build_answer(cfg1: dict, mem: dict, weights: dict, pp: dict) -> dict:
    """work14: links to the members' sets and to the rows, task1.parquet of the blend (post-processed) and its
    choice.json (the lab modules read both from their work folder)."""
    tag = cfg1.get("rows_tag", "r1")
    w14 = work14()
    rows_dir = C.WORK / "t1" / f"rows_{tag}"
    if not rows_dir.exists():
        raise FileNotFoundError(f"t1v14: {rows_dir} is missing - prepare_rows first")
    t1d = w14 / "t1"
    t1d.mkdir(parents=True, exist_ok=True)
    (w14 / "pred").mkdir(parents=True, exist_ok=True)
    _link(rows_dir, t1d / rows_dir.name)
    for n in weights:
        _link(mem[n][1], t1d / mem[n][1].name)
    tot = sum(weights.values())
    members = [(mem[n][0], t1d / mem[n][1].name, float(w) / tot) for n, w in weights.items()]
    f = w14 / "pred" / "task1.parquet"
    ch_path = t1d / "choice.json"
    want = dict(pp=pp, members={n: dict(kind=mem[n][0], dir=mem[n][1].name, weight=round(float(w) / tot, 6))
                                for n, w in weights.items()})
    if f.exists() and ch_path.exists():
        old = json.load(open(ch_path))
        if old.get("pp") == pp and old.get("members") == want["members"] and old.get("task1_md5") == T1.file_md5(f):
            return dict(rows=None, reused=True, task1_md5=old["task1_md5"])
    out = T1.predict_test(rows_dir, members, pp, f)
    ch = dict(want, version=VERSION, rows_tag=tag, seeds=sorted(int(n[3:]) for n in weights if n.startswith("lgb")),
              plan=T1.plan_signature(cfg1), task1_md5=T1.file_md5(f))
    json.dump(ch, open(ch_path, "w"), indent=1, default=str)
    return dict(rows=int(len(out)), reused=False, task1_md5=ch["task1_md5"])


def _summ(rep: dict) -> dict:
    c = rep.get("candidate")
    d = (rep.get("variants") or {}).get(c) or {}
    return dict(B0=rep.get("B0"), B1=rep.get("B1"), base_vs_B0=rep.get("base_vs_B0"), candidate=c,
                cand=dict((k, d.get(k)) for k in ("d_J", "se_J", "d_S_state", "d_S_LWR", "dJ_first_half", "dJ_second_half")),
                passes_holdout=rep.get("passes_holdout"), adopted=rep.get("adopted"),
                s3_heads={h: {k: v for k, v in dd.items() if k != "top_features"}
                          for h, dd in ((rep.get("s3") or {}).get("heads") or {}).items()},
                minutes=rep.get("minutes"))


def run_chain(cfg1: dict, on_step=None) -> dict:
    """The v11 lab (v10 stage 2 as its base chain) on the work14 answer; a stored lab result of the same key is reused.
    Returns the lab summary, the March check and the test-row application."""
    w14 = work14()
    out = {}
    with in_work(w14):
        key_now = L11.lab_key11(cfg1, LAB11, BASE_CHAIN)
        rp = w14 / "t1lab" / "lab11_report.json"
        rep = json.load(open(rp)) if rp.exists() else {}
        if rep.get("key") != key_now or (rep.get("adopted") and L11.load_models(rep) is None):
            rep = L11.run_lab11(cfg1, LAB11, BASE_CHAIN, v10_dj=None)
            if on_step is not None:
                on_step()
        out["lab"] = _summ(rep)
        out["lab_key"] = rep.get("key")
        if not rep.get("adopted"):
            out["reason"] = "the v11 stage 3 does not pass the holdout gate on the new answer"
            return out
        L.prepare_rows(cfg1)   # the test-month holdout files (linked rows: reused)
        app_p = w14 / "t1lab" / "applied14.json"
        app = json.load(open(app_p)) if app_p.exists() else {}
        same = app.get("key") == rep.get("key") and app.get("candidate") == rep.get("candidate")
        tm = app["tm"] if same and app.get("tm") else L11.tm_check11(rep, cfg1, LAB11)
        json.dump(dict(key=rep.get("key"), candidate=rep.get("candidate"), tm=tm), open(app_p, "w"), indent=1, default=str)
        out["tm"] = tm
        if not tm.get("ok"):
            out["reason"] = "the held-out March cells lose S_state against the new B1"
            return out
        f11 = w14 / "pred" / "task1_v11.parquet"
        if same and app.get("md5") and f11.exists() and app["md5"] == T1.file_md5(f11):
            out["apply"] = app.get("apply")
        else:
            chk = L11.apply_test11(rep, cfg1)
            out["apply"] = chk
            json.dump(dict(key=rep.get("key"), candidate=rep.get("candidate"), tm=tm, apply=chk, md5=T1.file_md5(f11)),
                      open(app_p, "w"), indent=1, default=str)
        if on_step is not None:
            on_step()
        out["file"] = str(f11)
    return out


# ----------------------------------------------------------------------------- 4. against the v11 answer
def old_final(work=None) -> dict:
    """The v11 answer's train-holdout J (B1 + its stage-3 gain) and its S_state on the held-out March cells."""
    work = work or C.WORK
    lr = json.load(open(work / "t1lab" / "lab11_report.json"))
    ap = json.load(open(work / "t1lab" / "applied11.json"))
    c = lr.get("candidate")
    d = (lr.get("variants") or {}).get(c) or {}
    return dict(candidate=c, J=float(lr["B1"]["J"]) + float(d.get("d_J", 0.0)),
                S_LWR=float(lr["B1"]["S_LWR"]) + float(d.get("d_S_LWR", 0.0)),
                S_state=float(lr["B1"]["S_state"]) + float(d.get("d_S_state", 0.0)),
                tm=float(ap["tm"]["S_cand"]), tm_B1=float(ap["tm"]["S_B1"]))


def compare(chain: dict, st: dict, work=None) -> dict:
    old = old_final(work)
    lab = chain["lab"]
    new = dict(candidate=lab["candidate"], J=float(lab["B1"]["J"]) + float(lab["cand"]["d_J"]),
               S_LWR=float(lab["B1"]["S_LWR"]) + float(lab["cand"]["d_S_LWR"]),
               S_state=float(lab["B1"]["S_state"]) + float(lab["cand"]["d_S_state"]),
               tm=float(chain["tm"]["S_cand"]), tm_B1=float(chain["tm"]["S_B1"]))
    d = dict(J=new["J"] - old["J"], S_state=new["S_state"] - old["S_state"], S_LWR=new["S_LWR"] - old["S_LWR"],
             tm=new["tm"] - old["tm"])
    ok = bool(d["J"] >= -float(st["final_tol_J"]) and d["tm"] >= -float(st["final_tol_tm"]))
    return dict(old=old, new=new, d=d, ok=ok,
                est_total=round(0.35 * d["tm"] + 0.10 * d["S_LWR"], 6))   # rough: March cells for S_state, holdout for S_LWR


def test_diff(new_f: Path, old_f: Path) -> dict:
    a, b = pd.read_parquet(new_f), pd.read_parquet(old_f)
    if len(a) != len(b) or not (a.timestamp.to_numpy() == b.timestamp.to_numpy()).all():
        return dict(comparable=False)
    lv = np.log(np.maximum(a.speed_kmh.to_numpy(), 1) / np.maximum(b.speed_kmh.to_numpy(), 1))
    lq = np.log(np.maximum(a.flow_vph.to_numpy(), 1) / np.maximum(b.flow_vph.to_numpy(), 1))
    return dict(comparable=True, rows=int(len(a)), mean_abs_dlog_v=round(float(np.mean(np.abs(lv))), 5),
                mean_abs_dlog_q=round(float(np.mean(np.abs(lq))), 5), max_abs_dv=round(float(np.max(np.abs(
                    a.speed_kmh.to_numpy() - b.speed_kmh.to_numpy()))), 3))


# ----------------------------------------------------------------------------- the step
def up_to_date(cfg1: dict, st: dict | None = None) -> tuple:
    """(True, report) when pred/task1_v14.parquet and its report belong to this code and these settings."""
    rp = C.WORK / "t1" / REPORT
    if not rp.exists():
        return False, None
    rep = json.load(open(rp))
    f = C.WORK / "pred" / OUT_FILE
    ok = (rep.get("version") == VERSION and rep.get("code") == code_hash() and rep.get("settings") == json.loads(
        json.dumps(settings(st), default=str)) and rep.get("cfg1") == json.loads(json.dumps(
            {k: v for k, v in cfg1.items() if k != "row_jobs"}, default=str))
          and (not rep.get("adopted") or (f.exists() and T1.file_md5(f) == rep.get("md5"))))
    return ok, rep


def code_hash() -> str:
    import hashlib
    h = hashlib.md5()
    for fn in ("task1.py", "physics.py", "t1lab.py", "t1lab11.py", "t1v14.py", "common.py", "data.py"):
        h.update((Path(__file__).parent / fn).read_bytes())
    return h.hexdigest()[:16]


def run(cfg1: dict, st: dict | None = None, on_step=None) -> dict:
    """Steps 1-4. Writes WORK/t1/t1v14_report.json (last) and, when adopted, WORK/pred/task1_v14.parquet."""
    st = settings(st)
    t0 = time.time()
    rep = dict(version=VERSION, code=code_hash(), settings=json.loads(json.dumps(st, default=str)),
               cfg1=json.loads(json.dumps({k: v for k, v in cfg1.items() if k != "row_jobs"}, default=str)),
               minutes={}, adopted=False)
    mem = lgb_members(cfg1, st)
    cur = current_blend()
    cbd, note = cb_member(cfg1, st, on_done=on_step)
    rep["cb"] = note
    if cbd is not None:
        mem["cb"] = ("cb", cbd)
    elif "cb" in cur:   # the v11 answer has a CatBoost member: keep it as it is
        ch11 = json.load(open(C.WORK / "t1" / "choice.json"))
        d = C.WORK / "t1" / ch11["members"]["cb"]["dir"]
        if T1.cb_complete(d):
            mem["cb"] = ("cb", d)
    missing = [n for n in cur if n not in mem]
    if missing:
        raise FileNotFoundError(f"t1v14: the v11 blend's members {missing} are not complete in WORK/t1")
    rep["members"] = {n: [k, d.name] for n, (k, d) in mem.items()}
    rep["minutes"]["members"] = round((time.time() - t0) / 60, 1)
    cands = candidates(cur, mem, st)
    rep["candidates"] = cands
    ch = choose(cfg1, mem, cands, st)
    rep["choice"] = ch
    rep["minutes"]["choice"] = round((time.time() - t0) / 60, 1)
    C.log(f"T1 v14 blend: {ch['chosen']} {ch['weights']} pp {ch['pp_table'].get(ch['chosen'])} | passing {ch['passing']}")
    if ch["chosen"] == "base":
        rep["reason"] = "no blend beats the v11 blend (March held-out cells and the post-processed train holdout)"
        return _finish(rep, t0)
    rep["answer"] = build_answer(cfg1, mem, ch["weights"], ch["pp"])
    if on_step is not None:
        on_step()
    rep["minutes"]["answer"] = round((time.time() - t0) / 60, 1)
    chain = run_chain(cfg1, on_step=on_step)
    rep["chain"] = chain
    rep["minutes"]["chain"] = round((time.time() - t0) / 60, 1)
    if "file" not in chain:
        rep["reason"] = chain.get("reason", "the chain was not applied")
        return _finish(rep, t0)
    cmpr = compare(chain, st)
    rep["vs_v11"] = cmpr
    new_f = Path(chain["file"])
    old_f = C.WORK / "pred" / "task1_v11.parquet"
    if old_f.exists():
        rep["test_vs_v11"] = test_diff(new_f, old_f)
    if not cmpr["ok"]:
        rep["reason"] = "the new chain does not beat the v11 answer (train-holdout J or held-out March cells)"
        return _finish(rep, t0)
    dst = C.WORK / "pred" / OUT_FILE
    tmp = dst.with_suffix(".part")
    shutil.copy2(new_f, tmp)
    tmp.replace(dst)
    rep["adopted"] = True
    rep["md5"] = T1.file_md5(dst)
    return _finish(rep, t0)


def _finish(rep: dict, t0: float) -> dict:
    rep["minutes"]["total"] = round((time.time() - t0) / 60, 1)
    d = C.WORK / "t1"
    tmp = d / (REPORT + ".part")
    json.dump(rep, open(tmp, "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    tmp.replace(d / REPORT)   # written last: it vouches for task1_v14.parquet
    C.log(f"T1 v14 done in {rep['minutes']['total']} min: adopted {rep['adopted']} | {rep.get('reason', '')} "
          f"| vs v11 {json.dumps(rep.get('vs_v11', {}).get('d'), default=str)}")
    return rep


# ----------------------------------------------------------------------------- the next set (report only)
def score_next(cfg1: dict, new: dict, st: dict | None = None) -> dict:
    """A further LightGBM set (t1pre13.train with `new`) scored against the v14 blend (or v11's when v14 kept it) on
    the test-month holdout and the train holdout - report only (WORK/t1/t1next_report.json)."""
    from . import t1pre13 as P13
    st = settings(st)
    new = P13.new_set(new)
    rp = C.WORK / "t1" / REPORT
    v14 = json.load(open(rp)) if rp.exists() else {}
    mem = lgb_members(cfg1, dict(st, new_seeds=list(st["new_seeds"]) + [int(new["seed"])]))
    if "cb" in v14.get("members", {}):
        cbd = C.WORK / "t1" / v14["members"]["cb"][1]
        if T1.cb_complete(cbd):
            mem["cb"] = ("cb", cbd)
    cur = dict(v14["choice"]["weights"]) if v14.get("adopted") else current_blend()
    nn = f"lgb{int(new['seed'])}"
    if nn not in mem:
        raise FileNotFoundError(f"t1v14: model set {nn} is not complete")
    w1 = float(np.mean(list(cur.values())))
    cands = {"base": dict(cur), "add_next": dict(cur, **{nn: w1}), "next_x2": dict(cur, **{nn: 2.0 * w1})}
    rows_dir = L.prepare_rows(cfg1)
    t0 = time.time()
    rep = T1.choose_blend(rows_dir, {n: mem[n] for n in set().union(*[set(c) for c in cands.values()])}, cands,
                          dict(cfg1, tm_z=float(st["tm_z"]), hold_tol=float(st["hold_tol"])))
    rep.pop("chosen", None)
    rep["new_set"] = new
    rep["base_is"] = "v14" if v14.get("adopted") else "v11"
    rep["minutes"] = round((time.time() - t0) / 60, 1)
    T1._HOLD_PRED.clear()
    d = C.WORK / "t1"
    json.dump(rep, open(d / "t1next_report.json.part", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    (d / "t1next_report.json.part").replace(d / "t1next_report.json")
    return rep
