"""Variance over N results files of the same prompt.
Usage: uv run python scripts/variance.py evals/results/A.json evals/results/B.json ..."""

import json
import sys
from statistics import mean

CATS = ("lookup", "aggregation", "reasoning", "refusal")


def main(files: list[str]) -> None:
    runs = [json.loads(open(f).read()) for f in files]
    rows = [
        ("total", [r["pass_rate"]["total"] for r in runs]),
        ("visible", [r["pass_rate"]["per_split"]["visible"] for r in runs]),
        ("holdout", [r["pass_rate"]["per_split"]["holdout"] for r in runs]),
    ]
    rows += [(c, [r["pass_rate"]["per_category"][c] for r in runs]) for c in CATS]
    print("| metric | min | mean | max |\n|---|---|---|---|")
    for name, vals in rows:
        print(f"| {name} | {min(vals):.0%} | {mean(vals):.1%} | {max(vals):.0%} |")
    outcomes = {}
    for r in runs:
        for c in r["cases"]:
            outcomes.setdefault(c["id"], []).append(c["passed"])
    flips = sorted(i for i, v in outcomes.items() if len(set(v)) > 1)
    print(f"\ncases whose outcome differs across the {len(runs)} runs: {len(flips)} {flips}")
    print(
        "shas:", [r["git_sha"] for r in runs], "cost usd:", [r["cost"]["total_usd"] for r in runs]
    )


if __name__ == "__main__":
    main(sys.argv[1:])
