"""Static SVG charts from the loop files. Usage: uv run python scripts/plot.py

Reads every evals/results/loops/*.json in order, concatenates the iterations (baselines included,
so a new loop's re-measured baseline shows as its own point) and writes docs/img/pass_rate.svg and
docs/img/cost.svg. Every point is traceable to a results file in the loop file."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("svg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LOOPS = ROOT / "evals" / "results" / "loops"
OUT = ROOT / "docs" / "img"
CATS = ("lookup", "aggregation", "reasoning", "refusal")


def points() -> list[dict]:
    pts, x, cumulative = [], 0, 0.0
    for f in sorted(LOOPS.glob("*.json")):
        doc = json.loads(f.read_text())
        for it in doc["iterations"]:
            cumulative += it["cost_eur"]["total"]
            pts.append({**it, "x": x, "loop": f.stem, "cum": cumulative, "stop": doc["stop"]})
            x += 1
    return pts


def main() -> None:
    pts = points()
    OUT.mkdir(parents=True, exist_ok=True)
    evald = [p for p in pts if p["visible"] is not None]
    best = []
    b = None
    for p in pts:  # best-so-far visible = what main holds after each step
        if p["verdict"] in ("baseline", "accepted"):
            b = p["visible"]
        best.append(b)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.step(
        [p["x"] for p in pts],
        [v * 100 for v in best],
        where="post",
        lw=2,
        label="visible (on main)",
    )
    ax.plot(
        [p["x"] for p in evald],
        [p["visible"] * 100 for p in evald],
        "o",
        ms=5,
        label="visible, each evaluated hypothesis",
    )
    ax.plot(
        [p["x"] for p in evald],
        [p["holdout"] * 100 for p in evald],
        "s",
        ms=5,
        label="holdout, each evaluated hypothesis",
    )
    for p in evald:
        if p["verdict"].startswith("rejected"):
            ax.annotate(
                "rejected",
                (p["x"], p["holdout"] * 100),
                xytext=(0, -12),
                textcoords="offset points",
                ha="center",
                fontsize=7,
            )
    for p in pts:
        if p["verdict"] == "baseline" and p["x"] > 0:
            ax.axvline(p["x"] - 0.5, color="grey", ls=":", lw=1)
    ax.set_xlabel("iteration (all loops, in order; dotted line = new loop, re-measured baseline)")
    ax.set_ylabel("pass rate %")
    ax.set_ylim(0, 105)
    ax.set_xticks([p["x"] for p in pts], [str(p["n"]) for p in pts])
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title("Pass rate per iteration")
    fig.tight_layout()
    fig.savefig(OUT / "pass_rate.svg")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot([p["cum"] for p in pts], [v * 100 for v in best], "-o", ms=4)
    for p in pts:
        if p["stop"] and p["stop"]["at"] == p["n"] and p["verdict"] != "baseline":
            ax.annotate(
                f"stop: {p['stop']['reason']}",
                (p["cum"], best[p["x"]] * 100),
                xytext=(4, -12),
                textcoords="offset points",
                fontsize=7,
            )
    ax.set_xlabel("cumulative cost € (evals + optimizer, rejected hypotheses included)")
    ax.set_ylabel("visible pass rate % on main")
    ax.set_ylim(0, 105)
    ax.set_title("Cost vs pass rate, with budget stops")
    fig.tight_layout()
    fig.savefig(OUT / "cost.svg")
    print(f"{len(pts)} points → {OUT}/pass_rate.svg, cost.svg")


if __name__ == "__main__":
    main()
