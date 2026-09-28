"""Replay 2: the semantic anti-leak over the 19 hyp/*
branches, against today's literal check re-run on each branch's prompt files (the verdict the
loop recorded at the time is kept as `old_reason`, information only).
Needs the hyp/* branches of the working repo, which are not published.
Writes evals/results/replay_leak_<ts>.json.

Usage: uv run python -m scripts.replay_leak
"""

import asyncio
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

from git import Repo

from evals.run import RESULTS, load_cases
from src import jev
from src.llm import load_env
from src.optimizer.gates import leaks_expected, leaks_semantic
from src.optimizer.loop import ROOT, paths

PROMPT_FILES = ("system.md", "tools.yaml")


def hypotheses(loops: list[dict]) -> list[dict]:
    """Each hypothesis with the results file of the base it was proposed from: the loop's own
    acceptances advance the base, a rejection does not."""
    out = []
    for loop in loops:
        before = None
        for it in loop["iterations"]:
            if it.get("hyp_id") is None:
                before = it.get("results_file")
                continue
            out.append({"hyp": it["hyp_id"], "n": it["n"],
                        "old_reason": it.get("reason"), "before_file": before})  # fmt: skip
            if it.get("reason") is None and it.get("results_file"):
                before = it["results_file"]
    return out


def classify(regex_now: str | None, jev_leak: str | None) -> str:
    if regex_now and jev_leak:
        return "agree_leak"
    if regex_now:
        return "regex_only"
    if jev_leak:
        return "jev_only"
    return "agree_clean"


def failing_visible(results_file: str, cases_by_id: dict) -> list[dict]:
    doc = json.loads((ROOT / results_file).read_text())
    return [{"id": r["id"], "input": cases_by_id.get(r["id"], {}).get("input", r["input"])}
            for r in doc["cases"] if r["split"] == "visible" and not r["passed"]]  # fmt: skip


def branch_prompts(repo: Repo, branch: str, prompts_dir: Path) -> dict[str, str]:
    rel = prompts_dir.relative_to(ROOT)
    return {f: repo.git.show(f"{branch}:{rel}/{f}") for f in PROMPT_FILES}


async def main_async() -> Path:
    load_env()
    if not jev.enabled():
        sys.exit("TYPESAFE_API_KEY missing: the replay needs Jev")
    repo, fs = Repo(ROOT), paths()
    cases = load_cases()
    cases_by_id = {c["id"]: c for c in cases}
    loops = [json.loads(p.read_text()) for p in sorted(fs["loops"].glob("*.json"))]
    threshold = float(os.environ.get("JEV_LEAK_MIN", "0.7"))
    rows, cost = [], 0.0

    async def ask(state, questions):
        nonlocal cost
        r = await jev.ask(state, questions)
        cost += r.cost_usd
        return r

    for h in hypotheses(loops):
        branch = h["hyp"]
        diff = repo.git.diff(f"{branch}~1", branch, "--", str(fs["prompts"]))
        failing = failing_visible(h["before_file"], cases_by_id) if h["before_file"] else []
        failing_ids = {f["id"] for f in failing}
        # today's literal check, re-run on the branch's own prompt files
        regex_now = None
        for _name, content in branch_prompts(repo, branch, fs["prompts"]).items():
            regex_now = regex_now or leaks_expected(content, cases, failing_ids)
        try:
            res = await leaks_semantic(diff, failing, ask, threshold) if failing else None
            verdict, error = classify(regex_now, res), None
        except Exception as e:  # noqa: BLE001 — one branch's Jev error is a row, not a lost run
            res, verdict, error = None, "jev_error", str(e)[:200]
        rows.append({**h, "failing": sorted(failing_ids), "regex_now": regex_now, "jev": res,
                     "verdict": verdict, "error": error})  # fmt: skip
        print(f"{branch}: old={h['old_reason']!r} regex_now={regex_now!r} jev={res!r} "
              f"-> {rows[-1]['verdict']}", file=sys.stderr)  # fmt: skip
    verdicts = ("agree_leak", "agree_clean", "regex_only", "jev_only", "jev_error")
    counts = {k: sum(1 for r in rows if r["verdict"] == k) for k in verdicts}
    ts = datetime.now(UTC).isoformat(timespec="seconds")
    out = {"timestamp": ts, "replay": "leak", "models": {"jev": jev.model()},
           "threshold": threshold, "cost": {"jev_usd": round(cost, 4)}, "counts": counts,
           "disagreements": [r["hyp"] for r in rows if r["verdict"] in ("regex_only", "jev_only")],
           "hypotheses": rows}  # fmt: skip
    path = RESULTS / f"replay_leak_{ts.replace(':', '')}.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"{counts}\ndisagreements: {out['disagreements']}\n-> {path}")
    return path


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
