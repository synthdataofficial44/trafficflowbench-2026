"""v13: Kaggle research dump, run in its own process so the notebook's kaggle CLI (used for the submissions) is untouched:

    PYTHONPATH=<folder with kaggle==2.2.4>:<folder holding tfb> python -m tfb.research13 <out_dir> [since YYYY-MM-DD]

kaggle 2.2.x is the first client that reads discussion topics. Writes into <out_dir>: leaderboard.csv, discussions.json
and discussions.md (every topic with all its messages), kernels.json (metadata of the public notebooks), kernels_new.md
(the code of the notebooks run on or after `since`, newest first), our_submissions.json and summary.json (what worked).
Nothing is submitted and nothing here feeds the pipeline: it is reading material for the analysis.
"""
from __future__ import annotations

import glob
import json
import os
import shutil
import sys
import time
import traceback
import zipfile

COMP = "2026-ieee-big-data-traffic-flow-bench"


def _plain(o):
    if hasattr(o, "to_dict"):
        try:
            return o.to_dict()
        except Exception:  # noqa: BLE001
            pass
    if isinstance(o, (list, tuple)):
        return [_plain(x) for x in o]
    if isinstance(o, dict):
        return {str(k): _plain(v) for k, v in o.items()}
    if isinstance(o, (str, int, float, bool)) or o is None:
        return o
    return str(o)


def _g(o, *names, default=None):
    for n in names:
        v = o.get(n) if isinstance(o, dict) else getattr(o, n, None)
        if v not in (None, ""):
            return v
    return default


def main(out: str, since: str = "2026-10-10", max_kernels: int = 40, max_chars: int = 60_000) -> dict:
    os.makedirs(out, exist_ok=True)
    summary = dict(started=time.strftime("%Y-%m-%d %H:%M:%S"), competition=COMP, since=since)

    def save_json(name, obj):
        with open(os.path.join(out, name + ".part"), "w") as f:
            json.dump(obj, f, indent=1, default=str, ensure_ascii=False)
        os.replace(os.path.join(out, name + ".part"), os.path.join(out, name))

    def save_text(name, text):
        with open(os.path.join(out, name + ".part"), "w") as f:
            f.write(text)
        os.replace(os.path.join(out, name + ".part"), os.path.join(out, name))

    def step(name, fn):
        t0 = time.time()
        try:
            r = fn()
            summary[name] = "ok (%.0f s)" % (time.time() - t0)
            print(f"{name}: ok ({time.time() - t0:.0f} s)", flush=True)
            return r
        except Exception as e:  # noqa: BLE001  one failed part must not stop the others
            summary[name] = f"FAILED {type(e).__name__}: {str(e)[:300]}"
            print(f"{name}: FAILED {type(e).__name__}: {e}", flush=True)
            traceback.print_exc(limit=2)
            return None

    import kaggle as _kg
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    summary["kaggle_version"] = getattr(_kg, "__version__", "?")

    def leaderboard():
        tmp = os.path.join(out, "_lb")
        os.makedirs(tmp, exist_ok=True)
        api.competition_leaderboard_download(COMP, tmp)
        for z in glob.glob(os.path.join(tmp, "*.zip")):
            with zipfile.ZipFile(z) as zf:
                zf.extractall(tmp)
        csvs = glob.glob(os.path.join(tmp, "**", "*.csv"), recursive=True)
        if not csvs:
            raise RuntimeError("no leaderboard csv in the download")
        shutil.copy2(csvs[0], os.path.join(out, "leaderboard.csv"))
        shutil.rmtree(tmp, ignore_errors=True)
        return True

    def ours():
        subs = api.competition_submissions(COMP, page_size=200) or []
        save_json("our_submissions.json", _plain(subs))
        return len(subs)

    def _flat(msgs, depth=0):
        rows = []
        for m in msgs or []:
            rows.append(dict(id=_g(m, "id"), author=_g(m, "author_name", "authorName"), date=str(_g(m, "post_date", "postDate")),
                             votes=_g(m, "votes", default=0), depth=depth,
                             text=_g(m, "raw_markdown", "rawMarkdown", "content", default="")))
            rows += _flat(_g(m, "replies", default=[]), depth + 1)
        return rows

    def discussions():
        topics, page = [], 1
        while page <= 50:
            r = api.competition_list_topics(COMP, sort_by="new", page=page)
            ts = list(getattr(r, "topics", None) or [])
            if not ts:
                break
            topics += ts
            total = getattr(r, "total_count", None) or 0
            if total and len(topics) >= total:
                break
            page += 1
            time.sleep(0.3)
        seen, rows = set(), []
        for t in topics:
            tid = _g(t, "id")
            if tid in seen:
                continue
            seen.add(tid)
            row = dict(id=tid, title=_g(t, "title"), author=_g(t, "author_name", "authorName"),
                       date=str(_g(t, "post_date", "postDate")),
                       last=str(_g(t, "last_comment_post_date", "lastCommentPostDate")),
                       comments=_g(t, "comment_count", "commentCount", default=0), votes=_g(t, "votes", default=0))
            try:
                m = api.competition_list_topic_messages(COMP, int(tid), sort_by="old", page_size=-1)
                row["messages"] = _flat(getattr(m, "messages", None) or [])
            except Exception as e:  # noqa: BLE001
                row["messages"] = []
                row["error"] = f"{type(e).__name__}: {e}"[:300]
            rows.append(row)
            time.sleep(0.3)
        save_json("discussions.json", rows)
        md = []
        for r in rows:
            md.append(f"# [{r['id']}] {r['title']}\n{r['author']} | {r['date']} | last {r['last']} | {r['comments']} comments"
                      f" | {r['votes']} votes\n")
            for m in r["messages"]:
                md.append(f"{'  ' * m['depth']}- **{m['author']}** ({m['date']}, {m['votes']} votes):\n"
                          + "\n".join(("  " * m["depth"]) + "  " + ln for ln in str(m["text"]).splitlines()))
            md.append("")
        save_text("discussions.md", "\n".join(md))
        return len(rows)

    def kernels():
        found = {}
        for sort in ("dateRun", "voteCount", "scoreDescending", "hotness"):
            for page in range(1, 6):
                try:
                    ks = api.kernels_list(competition=COMP, sort_by=sort, page=page, page_size=100) or []
                except Exception as e:  # noqa: BLE001
                    print("kernels_list", sort, page, type(e).__name__, e)
                    break
                for k in ks:
                    d = _plain(k)
                    ref = _g(k, "ref") or d.get("ref")
                    if ref and ref not in found:
                        d["_found_by"] = sort
                        found[ref] = d
                if len(ks) < 100:
                    break
                time.sleep(0.3)
        meta = list(found.values())
        save_json("kernels.json", meta)
        run = lambda d: str(d.get("lastRunTime") or d.get("last_run_time") or "")
        new = sorted([d for d in meta if run(d)[:10] >= since], key=run, reverse=True)[:max_kernels]
        kdir = os.path.join(out, "_kernels")
        os.makedirs(kdir, exist_ok=True)
        parts = []
        for d in new:
            ref = d["ref"]
            dst = os.path.join(kdir, ref.replace("/", "__"))
            head = (f"# {d.get('title')}\nref: {ref} | votes: {d.get('totalVotes') or d.get('total_votes')} | last run: {run(d)}\n")
            try:
                os.makedirs(dst, exist_ok=True)
                api.kernels_pull(ref, path=dst)
                txt = []
                for f in sorted(glob.glob(os.path.join(dst, "*"))):
                    if f.endswith(".ipynb"):
                        nb = json.load(open(f))
                        for c in nb.get("cells", []):
                            src = "".join(c.get("source", [])) if isinstance(c.get("source"), list) else str(c.get("source", ""))
                            if src.strip():
                                txt.append(("```python\n" + src + "\n```") if c.get("cell_type") == "code" else src)
                    elif f.endswith((".py", ".r", ".R", ".md", ".txt")):
                        txt.append("```\n" + open(f, errors="ignore").read() + "\n```")
                body = "\n\n".join(txt)
                if len(body) > max_chars:
                    body = body[:max_chars] + f"\n\n[... cut at {max_chars} characters]"
                parts.append(head + "\n" + body)
            except Exception as e:  # noqa: BLE001
                parts.append(head + f"\n(pull failed: {type(e).__name__}: {str(e)[:200]})")
            time.sleep(0.3)
        save_text("kernels_new.md", "\n\n---\n\n".join(parts))
        shutil.rmtree(kdir, ignore_errors=True)
        summary["kernels_new"] = [dict(ref=d.get("ref"), title=d.get("title"), last_run=run(d),
                                       votes=d.get("totalVotes") or d.get("total_votes")) for d in new]
        return len(meta)

    summary["leaderboard_ok"] = step("leaderboard", leaderboard)
    summary["our_submissions"] = step("our_submissions", ours)
    summary["topics"] = step("discussions", discussions)
    summary["kernels"] = step("kernels", kernels)
    summary["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save_json("summary.json", summary)
    return summary


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a:
        raise SystemExit("usage: python -m tfb.research13 <out_dir> [since YYYY-MM-DD]")
    main(a[0], a[1] if len(a) > 1 else "2026-10-10")
