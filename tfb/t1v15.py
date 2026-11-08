"""v15 Task 1: the fifth LightGBM set (v14 step 12: 4x the v8 transductive sample, up to 5000 rounds) and, on a GPU
runtime, the CatBoost member added to the v14 blend, with the v10/v11 lab rerun on top - the v14 procedure (t1v14) with
the v14 answer as the base, in its own work folder (work15), compared with the v14 answer before it replaces it.

Runs only when there is something new to add: the step-12 report (t1/t1next_report.json) shows a passing candidate (the
fifth set beats the v14 blend on the held-out March cells by > 2 SEs and does not lose on the train holdout), or a
CatBoost member can be trained / reused. Same gates as v14 at every step: `task1.choose_blend` (March held-out cells
decide, the train holdout must not lose), the post-processed train-holdout J among the passing blends, the v11 lab gate,
the held-out March cells, and finally the v14 answer itself (train-holdout J not lower, held-out March S_state not lower
by more than 1e-4). April is never used to decide (its held-out cells are reported). Nothing of v8-v14 is changed.
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np

from . import common as C
from . import task1 as T1
from . import t1lab as L
from . import t1lab11 as L11
from . import t1v14 as V

VERSION = "v15"
OUT_FILE = "task1_v15.parquet"
REPORT = "t1v15_report.json"
W15 = "work15"
DEFAULTS = dict(new_seeds=[4], cb="auto", cb_weights=[0.25, 0.4], tm_z=2.0, hold_tol=0.0002, pp_top=4, final_tol_J=0.0,
                final_tol_tm=0.0001, force_blend=None, need_next=True)


def settings(st: dict | None) -> dict:
    s = dict(DEFAULTS)
    s.update({k: v for k, v in (st or {}).items() if v is not None or k == "force_blend"})
    unknown = set(s) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"t1v15 settings: unknown keys {sorted(unknown)}")
    if s["cb"] not in ("auto", True, False):
        raise ValueError("t1v15 cb must be 'auto', True or False")
    return s


def work15() -> Path:
    return C.WORK.parent / W15


def v14_report(work=None) -> dict:
    rp = (work or C.WORK) / "t1" / V.REPORT
    return json.load(open(rp)) if rp.exists() else {}


def base_answer(work=None) -> dict:
    """The answer v15 builds on: v14's when it was adopted (its blend, file and final numbers), else v11's."""
    work = work or C.WORK
    r14 = v14_report(work)
    if r14.get("adopted") and (work / "pred" / V.OUT_FILE).exists():
        new = (r14.get("vs_v11") or {}).get("new")
        if not new:
            raise RuntimeError("t1v15: the v14 report has no final numbers (vs_v11.new)")
        return dict(name="v14", weights=dict(r14["choice"]["weights"]), file=work / "pred" / V.OUT_FILE, final=dict(new),
                    members=r14.get("members") or {})
    return dict(name="v11", weights=V.current_blend(work), file=work / "pred" / "task1_v11.parquet",
                final=V.old_final(work), members={})


def next_passes(work=None) -> tuple:
    """(bool, summary) from the step-12 report: a fifth-set candidate passes the v8 blend rule against the v14 blend."""
    rp = (work or C.WORK) / "t1" / "t1next_report.json"
    if not rp.exists():
        return False, dict(reason="t1/t1next_report.json is missing (v14 step 12 not finished)")
    r = json.load(open(rp))
    cands = r.get("candidates") or {}
    summ = {c: dict(tm_gain=d.get("tm_gain"), tm_se=d.get("tm_se"), tm_gain_April=d.get("tm_gain_private"),
                    hold_dJ=d.get("hold_dJ"), ok=d.get("ok")) for c, d in cands.items() if c != "base"}
    return any(bool(d.get("ok")) for d in summ.values()), dict(base_is=r.get("base_is"), new_set=r.get("new_set"),
                                                               candidates=summ)


def candidates(cur: dict, mem: dict, st: dict) -> dict:
    """base (the v14 blend), + each newer set at an average member weight and at twice that, and with a CatBoost member at
    its v8 weights on top of those (when the base has none)."""
    w1 = float(np.mean(list(cur.values())))
    out = {"base": dict(cur)}
    new = [f"lgb{int(s)}" for s in st["new_seeds"] if f"lgb{int(s)}" in mem and f"lgb{int(s)}" not in cur]
    if new:
        out["add_new"] = dict(cur, **{n: w1 for n in new})
        out["new_x2"] = dict(cur, **{n: 2.0 * w1 for n in new})
    if "cb" in mem:
        for nm in list(out):
            g = out[nm]
            if "cb" in g:
                continue
            tot = sum(g.values())
            for w in st["cb_weights"]:
                out[f"{nm}+cb{w}"] = dict({k: (1.0 - float(w)) * v / tot for k, v in g.items()}, cb=float(w))
    return out


def build_answer(cfg1: dict, mem: dict, weights: dict, pp: dict) -> dict:
    """work15: links to the members' sets and to the rows, task1.parquet of the blend (post-processed) and its choice.json."""
    tag = cfg1.get("rows_tag", "r1")
    w15 = work15()
    rows_dir = C.WORK / "t1" / f"rows_{tag}"
    if not rows_dir.exists():
        raise FileNotFoundError(f"t1v15: {rows_dir} is missing - prepare_rows first")
    t1d = w15 / "t1"
    t1d.mkdir(parents=True, exist_ok=True)
    (w15 / "pred").mkdir(parents=True, exist_ok=True)
    V._link(rows_dir, t1d / rows_dir.name)
    for n in weights:
        V._link(mem[n][1], t1d / mem[n][1].name)
    tot = sum(weights.values())
    members = [(mem[n][0], t1d / mem[n][1].name, float(w) / tot) for n, w in weights.items()]
    f = w15 / "pred" / "task1.parquet"
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


def run_chain(cfg1: dict, on_step=None) -> dict:
    """The v11 lab (v10 stage 2 as its base chain) on the work15 answer (t1v14.run_chain in work15)."""
    w15 = work15()
    out = {}
    with V.in_work(w15):
        key_now = L11.lab_key11(cfg1, V.LAB11, V.BASE_CHAIN)
        rp = w15 / "t1lab" / "lab11_report.json"
        rep = json.load(open(rp)) if rp.exists() else {}
        if rep.get("key") != key_now or (rep.get("adopted") and L11.load_models(rep) is None):
            rep = L11.run_lab11(cfg1, V.LAB11, V.BASE_CHAIN, v10_dj=None)
            if on_step is not None:
                on_step()
        out["lab"] = V._summ(rep)
        out["lab_key"] = rep.get("key")
        if not rep.get("adopted"):
            out["reason"] = "the v11 stage 3 does not pass the holdout gate on the new answer"
            return out
        L.prepare_rows(cfg1)
        app_p = w15 / "t1lab" / "applied15.json"
        app = json.load(open(app_p)) if app_p.exists() else {}
        same = app.get("key") == rep.get("key") and app.get("candidate") == rep.get("candidate")
        tm = app["tm"] if same and app.get("tm") else L11.tm_check11(rep, cfg1, V.LAB11)
        json.dump(dict(key=rep.get("key"), candidate=rep.get("candidate"), tm=tm), open(app_p, "w"), indent=1, default=str)
        out["tm"] = tm
        if not tm.get("ok"):
            out["reason"] = "the held-out March cells lose S_state against the new B1"
            return out
        f11 = w15 / "pred" / "task1_v11.parquet"
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


def compare(chain: dict, base: dict, st: dict) -> dict:
    old = dict(base["final"])
    lab = chain["lab"]
    new = dict(candidate=lab["candidate"], J=float(lab["B1"]["J"]) + float(lab["cand"]["d_J"]),
               S_LWR=float(lab["B1"]["S_LWR"]) + float(lab["cand"]["d_S_LWR"]),
               S_state=float(lab["B1"]["S_state"]) + float(lab["cand"]["d_S_state"]),
               tm=float(chain["tm"]["S_cand"]), tm_B1=float(chain["tm"]["S_B1"]))
    d = dict(J=new["J"] - float(old["J"]), S_state=new["S_state"] - float(old["S_state"]),
             S_LWR=new["S_LWR"] - float(old["S_LWR"]), tm=new["tm"] - float(old["tm"]))
    ok = bool(d["J"] >= -float(st["final_tol_J"]) and d["tm"] >= -float(st["final_tol_tm"]))
    return dict(base=base["name"], old=old, new=new, d=d, ok=ok, est_total=round(0.35 * d["tm"] + 0.10 * d["S_LWR"], 6))


def code_hash() -> str:
    import hashlib
    h = hashlib.md5()
    for fn in ("task1.py", "physics.py", "t1lab.py", "t1lab11.py", "t1v14.py", "t1v15.py", "common.py", "data.py"):
        h.update((Path(__file__).parent / fn).read_bytes())
    return h.hexdigest()[:16]


def up_to_date(cfg1: dict, st: dict | None = None) -> tuple:
    """(True, report) when the v15 Task 1 report belongs to this code, these settings and the current inputs."""
    rp = C.WORK / "t1" / REPORT
    if not rp.exists():
        return False, None
    rep = json.load(open(rp))
    f = C.WORK / "pred" / OUT_FILE
    nx = C.WORK / "t1" / "t1next_report.json"
    ok = (rep.get("version") == VERSION and rep.get("code") == code_hash()
          and rep.get("settings") == json.loads(json.dumps(settings(st), default=str))
          and rep.get("cfg1") == json.loads(json.dumps({k: v for k, v in cfg1.items() if k != "row_jobs"}, default=str))
          and rep.get("next_md5") == (T1.file_md5(nx) if nx.exists() else None)
          and (not rep.get("adopted") or (f.exists() and T1.file_md5(f) == rep.get("md5"))))
    return ok, rep


def run(cfg1: dict, st: dict | None = None, on_step=None) -> dict:
    """Writes WORK/t1/t1v15_report.json (last) and, when adopted, WORK/pred/task1_v15.parquet."""
    st = settings(st)
    t0 = time.time()
    nx = C.WORK / "t1" / "t1next_report.json"
    rep = dict(version=VERSION, code=code_hash(), settings=json.loads(json.dumps(st, default=str)),
               cfg1=json.loads(json.dumps({k: v for k, v in cfg1.items() if k != "row_jobs"}, default=str)),
               next_md5=T1.file_md5(nx) if nx.exists() else None, minutes={}, adopted=False)
    base = base_answer()
    rep["base"] = dict(name=base["name"], weights=base["weights"], final=base["final"])
    nok, nsum = next_passes()
    rep["next"] = nsum
    mem = V.lgb_members(cfg1, dict(V.DEFAULTS, new_seeds=sorted({3, *[int(s) for s in st["new_seeds"]]})))
    if not nok:
        mem = {n: m for n, m in mem.items() if n not in {f"lgb{int(s)}" for s in st["new_seeds"]}}
    cbd, note = V.cb_member(cfg1, dict(V.DEFAULTS, cb=st["cb"]), on_done=on_step)
    rep["cb"] = note
    if cbd is not None:
        mem["cb"] = ("cb", cbd)
    elif "cb" in base["weights"]:   # the base answer has a CatBoost member: keep it as it is
        d = C.WORK / "t1" / (base["members"].get("cb") or ["cb", f"cb_{cfg1.get('rows_tag', 'r1')}"])[1]
        if T1.cb_complete(d):
            mem["cb"] = ("cb", d)
    missing = [n for n in base["weights"] if n not in mem]
    if missing:
        raise FileNotFoundError(f"t1v15: the {base['name']} blend's members {missing} are not complete in WORK/t1")
    rep["members"] = {n: [k, d.name] for n, (k, d) in mem.items()}
    rep["minutes"]["members"] = round((time.time() - t0) / 60, 1)
    cands = candidates(base["weights"], mem, st)
    rep["candidates"] = cands
    if st["need_next"] and len(cands) == 1:
        rep["reason"] = ("nothing new to add: " + (nsum.get("reason") or "the fifth set does not pass against the v14 blend")
                         + f"; CatBoost member: {note}")
        return _finish(rep, t0)
    ch = V.choose(cfg1, mem, cands, st)
    rep["choice"] = ch
    rep["minutes"]["choice"] = round((time.time() - t0) / 60, 1)
    C.log(f"T1 v15 blend: {ch['chosen']} {ch['weights']} pp {ch['pp_table'].get(ch['chosen'])} | passing {ch['passing']}")
    if ch["chosen"] == "base":
        rep["reason"] = f"no blend beats the {base['name']} blend (March held-out cells and the post-processed train holdout)"
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
    cmpr = compare(chain, base, st)
    rep["vs_base"] = cmpr
    new_f = Path(chain["file"])
    if Path(base["file"]).exists():
        rep["test_vs_base"] = V.test_diff(new_f, Path(base["file"]))
    if not cmpr["ok"]:
        rep["reason"] = f"the new chain does not beat the {base['name']} answer (train-holdout J or held-out March cells)"
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
    tmp.replace(d / REPORT)
    C.log(f"T1 v15 done in {rep['minutes']['total']} min: adopted {rep['adopted']} | {rep.get('reason', '')} "
          f"| vs {rep['base']['name']} {json.dumps(rep.get('vs_base', {}).get('d'), default=str)}")
    return rep
