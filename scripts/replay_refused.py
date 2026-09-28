"""Replay 1: re-read every stored answer with the Jev refusal Choice,
regrade the `refused` checks, and re-decide every gate verdict in the loop files. Writes
evals/results/replay_refused_<ts>.json.

Usage: uv run python -m scripts.replay_refused [--concurrency 8]
"""

import argparse
import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from evals.checks import check
from evals.refusal import refused_from_text
from evals.run import RESULTS, load_cases, summarize
from src import jev
from src.llm import load_env
from src.optimizer.gates import gate, tolerances

LOOPS = RESULTS / "loops"


def _agent_flag(row: dict) -> bool:
    """The agent's own refusal flag. A row already regraded once carries `refused_flag`
    separately from `refused` (which may already be Jev-decided): prefer it so a second regrade
    never mistakes yesterday's Jev verdict for the agent's flag."""
    return row.get("refused_flag", row["refused"])


def regrade(doc: dict, decisions: dict[str, bool]) -> dict:
    """decisions: case id -> refused as read from the text. Only `refused` checks are regraded:
    number/contains checks read the answer text, judge cases read it themselves. Rows must carry
    `expected` (the committed results rows don't: the caller merges it in from the dataset)."""
    rows = []
    for r in doc["cases"]:
        r = dict(r, refused_flag=_agent_flag(r), refused=decisions.get(r["id"], r["refused"]))
        if r["check"] == "refused":
            r["passed"], r["detail"] = check(r, r["answer"], r["refused"])
        rows.append(r)
    return {**doc, "cases": rows, "pass_rate": summarize(rows)}


def regrade_file(doc: dict, pairs: list[tuple[str, dict]]) -> dict:
    """pairs: (case id, refused_from_text() result) for every case in doc, in the same order.
    A Jev error on one case is `refused_from_text`'s own fallback to the flag (never raises); it
    is carried here into `summary.errors` and counted, not lost (same rule as replay 2's
    jev_error row)."""
    decisions = {k: v["refused"] for k, v in pairs}
    new = regrade(doc, decisions)
    by_id = dict(pairs)
    disagree = sum(
        1
        for r in doc["cases"]
        if by_id[r["id"]]["refused_by"] == "jev" and by_id[r["id"]]["refused"] != r["refused"]
    )
    errors = [{"id": k, "error": v["error"]} for k, v in pairs if v.get("error")]
    cost = sum(v["cost_usd"] for _, v in pairs)
    summary = {
        "old": doc["pass_rate"].get("total"),
        "new": new["pass_rate"]["total"],
        "disagree": disagree,
        "errors": errors,
        "error_count": len(errors),
        "cost_usd": round(cost, 6),
    }
    return {"doc": new, "summary": summary}


def _with_expected(rows: list[dict], expected_by_id: dict) -> list[dict]:
    """Merge in `expected` from today's dataset for rows that don't already carry it. A `refused`
    row whose case id isn't in today's dataset can't be graded at all (check() needs `expected`)
    — fail loudly instead of silently regrading it against `expected=None`."""
    out = []
    for r in rows:
        if "expected" in r:
            out.append(r)
            continue
        if r["id"] not in expected_by_id:
            if r["check"] == "refused":
                raise ValueError(
                    f"case {r['id']!r} (check=refused) is missing from today's dataset: "
                    "cannot regrade it without an expected value"
                )
            out.append(dict(r, expected=None))
            continue
        out.append(dict(r, expected=expected_by_id[r["id"]]))
    return out


def _rates(pr: dict) -> dict:
    sp = pr["per_split"]
    return {"visible": sp.get("visible", 0.0), "holdout": sp.get("holdout", 0.0),
            "per_category_visible": pr["per_category_visible"]}  # fmt: skip


def replay_gates(
    loops: list[dict], new_rates: dict[str, dict], tol_by_file: dict[str, dict[str, float]]
) -> list[dict]:
    """Walk each loop in order; `before` advances on the loop's own (old) acceptances so every
    hypothesis is re-decided against the same base it was decided against. Each gate() call uses
    the tolerance of `before`'s own results file — the dataset in force when that comparison was
    actually made (src/optimizer/loop.py computes `tol` once per loop, from the dataset current at
    that time; tolerances have changed since: 40->44 cases moved refusal/reasoning from 1/8=0.125
    to 1/10=0.1). A `holdout_confirm` rejection is treated as an old acceptance for `old_rejected`:
    that reason comes from the second (holdout-only) confirmation run, not from the gate()
    decision replayed here."""
    out = []
    for loop in loops:
        before, before_file = None, None
        for it in loop["iterations"]:
            f = it.get("results_file")
            if not f or f not in new_rates:
                continue
            after = new_rates[f]
            if it.get("hyp_id") is None:
                before, before_file = after, f
                continue
            old = it.get("reason")
            tol = tol_by_file.get(before_file, {}) if before_file else {}
            new = gate(before, after, tol) if before else None
            old_rejected = old is not None and not str(old).startswith("holdout_confirm")
            changed = old_rejected != (new is not None)
            out.append({"loop": loop["loop_id"], "hyp": it["hyp_id"], "old_reason": old,
                        "new_reason": new, "changed": changed})  # fmt: skip
            if old is None:
                before, before_file = after, f
    return out


async def main_async(concurrency: int) -> Path:
    load_env()
    if not jev.enabled():
        sys.exit("TYPESAFE_API_KEY missing: the replay needs Jev")
    # committed results rows don't carry `expected` (evals.run writes it only for the judge
    # path); the dataset (ids don't get their `expected` changed, only added to) supplies it.
    expected_by_id = {c["id"]: c["expected"] for c in load_cases()}
    files = sorted(p for p in RESULTS.glob("*.json") if not p.name.startswith("replay_"))
    sem = asyncio.Semaphore(concurrency)
    cost = 0.0
    per_file, new_rates, tol_by_file = {}, {}, {}

    async def decide(row):
        async with sem:
            return row["id"], await refused_from_text(row["answer"], _agent_flag(row))

    for p in files:
        doc = json.loads(p.read_text())
        if "cases" not in doc or not doc["cases"]:
            continue
        doc["cases"] = _with_expected(doc["cases"], expected_by_id)
        pairs = await asyncio.gather(*(decide(r) for r in doc["cases"]))
        result = regrade_file(doc, pairs)
        cost += result["summary"]["cost_usd"]
        rel = f"evals/results/{p.name}"
        per_file[rel] = result["summary"]
        new_rates[rel] = _rates(result["doc"]["pass_rate"])
        tol_by_file[rel] = tolerances(doc["cases"])
        print(f"{p.name}: {per_file[rel]}", file=sys.stderr)
    loops = [json.loads(p.read_text()) for p in sorted(LOOPS.glob("*.json"))]
    decisions = replay_gates(loops, new_rates, tol_by_file)
    ts = datetime.now(UTC).isoformat(timespec="seconds")
    out = {"timestamp": ts, "replay": "refused", "models": {"jev": jev.model()},
           "cost": {"jev_usd": round(cost, 4)}, "files": per_file, "decisions": decisions,
           "changed": [d for d in decisions if d["changed"]]}  # fmt: skip
    path = RESULTS / f"replay_refused_{ts.replace(':', '')}.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"changed decisions: {[d['hyp'] for d in out['changed']]}\n→ {path}")
    return path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--concurrency", type=int, default=8)
    asyncio.run(main_async(p.parse_args().concurrency))


if __name__ == "__main__":
    main()
