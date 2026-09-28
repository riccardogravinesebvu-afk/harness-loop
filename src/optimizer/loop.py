"""Optimizer loop as a LangGraph graph: baseline → propose → apply → evaluate → decide → … → END.

One hypothesis per iteration, one prompt file edited on branch hyp/<n>, kept (fast-forward main)
or rolled back (branch stays as evidence). Every iteration leaves on main a results file, a
CHANGELOG row and the loop summary file. Why: DESIGN.md.

Usage: uv run python -m src.optimizer.loop [--max-iterations N] [--max-eur X] [--base BRANCH]
"""

import argparse
import difflib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, TypedDict

import yaml
from git import Repo
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from evals.run import RUNS_LOG, load_cases, summarize
from evals.run import main_async as run_evals
from src import jev
from src import observability as obs
from src.llm import load_env, make_chat, model_for
from src.optimizer.budget import should_stop
from src.optimizer.gates import CAPS, delta_cases, gate, leaks_expected, leaks_semantic, tolerances
from src.pricing import cost_usd, usd_to_eur

ROOT = Path(__file__).resolve().parents[2]


def paths(root: Path = ROOT) -> dict[str, Path | list[Path]]:
    agent = root / "src" / "agent_under_test"
    return {
        "root": root,
        "prompts": agent / "prompts",
        "code": [agent / f for f in ("ledger.py", "tools.py", "graph.py")],
        "changelog": root / "CHANGELOG.md",
        "proposed": root / "evals" / "proposed.yaml",
        "loops": root / "evals" / "results" / "loops",
    }


def prompt_diff(before: str, after: str, name: str) -> str:
    return "".join(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
                                        fromfile=f"a/{name}", tofile=f"b/{name}"))  # fmt: skip


TOOL_KEYS = {"run_sql", "lookup_customer", "compute", "final_answer"}
CATS = ("lookup", "aggregation", "reasoning", "refusal")
WINDOW = 3
HEADER = (
    "| n | hyp | sha | target | title | verdict | visible Δ | lookup/aggr/reas/ref Δ (cases) "
    "| holdout Δ | cost € | cumul € | results |\n" + "|---" * 12 + "|\n"
)


class ProposedCase(BaseModel):
    category: str
    input: str
    expected: str
    check: str
    why: str


class Hypothesis(BaseModel):
    title: str = Field(description="one line")
    rationale: str = Field(description="which failures, which mechanism; three sentences max")
    target: Literal["system.md", "tools.yaml"]
    content: str = Field(description="the complete new content of the target file")
    parent: str = Field(
        default="main",
        description="branch to start from: main, or an eligible hyp/<k> listed under Archive",
    )
    proposed_cases: list[ProposedCase] = Field(default_factory=list)


class LoopState(TypedDict, total=False):
    n: int  # branch number of the current hypothesis
    done: int  # iterations completed in this loop run
    best: dict  # summary of current main: visible, holdout, per_category_visible, results_file
    hyp: dict | None
    reason: str | None  # rejection reason, set before or after eval
    leak_regex: str | None  # the literal-leak verdict, recorded but not applied when Jev is on
    sha: str
    opt_eur: float
    results: dict | None
    history: list[dict]
    cumulative_eur: float
    archive: list[dict]  # rejected-with-eval since the last acceptance; eligible parents
    parent: str  # branch the current hypothesis was cut from
    parent_fallback: bool  # the optimizer named a parent that was not eligible
    stop: str | None


def summary(res: dict, root: Path) -> dict:
    pr = res["pass_rate"]
    return {
        "passed_ids": sorted(c["id"] for c in res["cases"] if c["passed"]),
        "visible": pr["per_split"].get("visible", 0.0),
        "holdout": pr["per_split"].get("holdout", 0.0),
        "per_category_visible": pr["per_category_visible"],
        "cost_eur": res["cost"]["total_eur"],
        "results_file": str(Path(res["_path"]).resolve().relative_to(root.resolve())),
        "sha": res["git_sha"],
    }


def failures(results_file: str, cases: list[dict]) -> list[dict]:
    """Visible failures of one results file, with tool args/outputs, from the local run log.
    Cases flagged by feedback carry their weight and the human note."""
    fb = {c["id"]: c for c in cases if c.get("feedback")}
    out = []
    for line in RUNS_LOG.read_text().splitlines() if RUNS_LOG.exists() else []:
        r = json.loads(line)
        if r["results_file"] == results_file and r["split"] == "visible" and not r["passed"]:
            detail = r["detail"]
            out.append(
                {
                    "id": r["id"],
                    "category": r["category"],
                    "check": r["check"],
                    "input": r["input"],
                    "expected": r.get("reference") or r["expected"],
                    "answer": r["answer"],
                    "refused": r["refused"],
                    "error": r["error"],
                    "tool_calls": [{**c, "output": c["output"][:500]} for c in r["tool_calls"]],
                    "detail": detail.get("rationale") if isinstance(detail, dict) else detail,
                    **(
                        {
                            "weight": fb[r["id"]]["weight"],
                            "human_feedback": fb[r["id"]]["feedback"]["note"],
                        }
                        if r["id"] in fb
                        else {}
                    ),
                }
            )
    return sorted(out, key=lambda r: r["id"])


def context(
    best: dict, cases: list[dict], fs: dict, fails: list[dict], archive: list[dict] | None = None
) -> str:
    n = {c: sum(1 for x in cases if x["category"] == c and x["split"] == "visible") for c in CATS}
    rates = "\n".join(
        f"- {c}: {best['per_category_visible'].get(c, 0.0):.0%} of {n[c]} visible cases"
        for c in CATS
    )
    log = fs["changelog"].read_text().splitlines()[4:] if fs["changelog"].exists() else []
    parts = [
        "## Current prompt files",
        *(f"### {f.name}\n```\n{f.read_text()}\n```" for f in sorted(fs["prompts"].iterdir())),
        "## Agent code (read-only)",
        *(f"### {f.name}\n```python\n{f.read_text()}\n```" for f in fs["code"]),
        "## Visible pass rate per category\n" + rates,
        f"## Failing visible cases ({len(fails)})\n```json\n{json.dumps(fails, indent=1)}\n```",
        "## Previous hypotheses (CHANGELOG rows, oldest first)\n" + ("\n".join(log[-8:]) or "none"),
    ]
    if archive is not None:
        rows = [
            f'### {a["branch"]}: "{a["title"]}" edited {a["target"]}; rejected: {a["reason"]}; '
            f"visible {a['visible']:.0%}, holdout {a['holdout']:.0%}; cases "
            + (", ".join(f"{c} {d:+d}" for c, d in a["delta_cases"].items() if d) or "unchanged")
            + f"\n```diff\n{a['diff'][:3000]}\n```"
            for a in archive
        ]
        parts.append(
            "## Archive (rejected since the last acceptance; set `parent` to one of these to "
            "build on it)\n" + ("\n".join(rows) or "empty")
        )
    return "\n\n".join(parts)


def changelog_row(it: dict) -> str:
    d = it["delta_cases"]
    cats = "/".join(f"{d.get(c, 0):+d}" if d.get(c) else "0" for c in CATS) if d else "—"
    vis = f"{it['visible_delta'] * 100:+.1f}pp" if it["visible_delta"] is not None else "—"
    hold = f"{it['holdout_delta'] * 100:+.1f}pp" if it["holdout_delta"] is not None else "—"
    if it.get("holdout_confirm"):
        hold += f" (confirm {it['holdout_confirm']['holdout'] * 100:.0f}%)"
    return (
        f"| {it['n']} | {it['hyp_id']} | {it['sha']} | {it['target']} | {it['title']} | "
        f"{it['verdict']} | {vis} | {cats} | {hold} | {it['cost_eur']['total']:.2f} | "
        f"{it['cumulative_eur']:.2f} | {it['results_file'] or '—'} |\n"
    )


def build_loop(cfg: dict, root: Path = ROOT):
    """cfg: max_iterations, max_eur, min_gain, seed, loop_id, base (default main).
    root: repo to operate on."""
    fs = paths(root)
    base = cfg.get("base", "main")
    use_archive = cfg.get("archive", True)
    repo = Repo(root)
    cases = load_cases()
    tol = tolerances(cases)
    loop_file = fs["loops"] / f"{cfg['loop_id']}.json"
    doc = {
        "loop_id": cfg["loop_id"],
        "started": cfg["loop_id"],
        "git_sha_start": repo.head.commit.hexsha[:7],
        "params": {
            "max_iterations": cfg["max_iterations"],
            "max_total_eur": cfg["max_eur"],
            "min_gain_per_eur": cfg["min_gain"],
            "window": WINDOW,
            "feedback_weight": float(os.environ.get("FEEDBACK_WEIGHT", "2")),
            "tolerance": tol,
            "seed": cfg["seed"],
            "base": base,
            "archive": use_archive,
        },
        "models": {r: model_for(r) for r in ("agent", "judge", "optimizer")},
        "iterations": [],
        "stop": None,
        "feedback": {},
    }

    def commit(paths: list[Path], msg: str) -> str:
        repo.index.add([str(p) for p in paths])
        return repo.index.commit(msg).hexsha[:7]

    async def evaluate_now(n: int | None, hyp_id: str | None, split: str = "all") -> dict:
        ns = argparse.Namespace(
            seed=cfg["seed"],
            split=split,
            limit=0,
            ids="",
            concurrency=4,
            out=None,
            iteration=n,
            hypothesis=hyp_id,
        )
        path = await run_evals(ns)
        return {**json.loads(path.read_text()), "_path": str(path)}

    def record(it: dict) -> None:
        doc["iterations"].append(it)
        fs["loops"].mkdir(parents=True, exist_ok=True)
        loop_file.write_text(json.dumps(doc, indent=2, ensure_ascii=False))

    async def baseline(state: LoopState) -> dict:
        if repo.is_dirty() or repo.untracked_files:
            raise SystemExit("working tree not clean: commit or stash first")
        if repo.active_branch.name != base:
            raise SystemExit(f"start the loop from {base}")
        res = await evaluate_now(0, None)
        best = summary(res, root)
        it = {
            "n": 0,
            "hyp_id": None,
            "sha": best["sha"],
            "target": None,
            "title": "baseline",
            "verdict": "baseline",
            "reason": None,
            "visible": best["visible"],
            "holdout": best["holdout"],
            "per_category_visible": best["per_category_visible"],
            "visible_delta": None,
            "holdout_delta": None,
            "delta_cases": {},
            "cost_eur": {"eval": best["cost_eur"], "optimizer": 0.0, "total": best["cost_eur"]},
            "cumulative_eur": best["cost_eur"],
            "results_file": best["results_file"],
            "best_visible": best["visible"],
        }
        record(it)
        commit(
            [root / best["results_file"], loop_file],
            f"loop {cfg['loop_id']}: baseline eval, iteration 0",
        )
        nums = [int(b.name.split("/")[1]) for b in repo.branches if b.name.startswith("hyp/")]
        return {
            "best": best,
            "n": max(nums, default=0) + 1,
            "done": 0,
            "hyp": None,
            "history": [{"pass_rate": best["visible"], "cost_eur": best["cost_eur"]}],
            "cumulative_eur": best["cost_eur"],
            "stop": None,
            "archive": [],
        }

    def reweigh(best: dict) -> dict:
        """Feedback can change case weights between iterations: recompute the current best's
        pass rates from its results file under today's weights, so the gate compares like
        with like."""
        res = json.loads((root / best["results_file"]).read_text())
        weights = {c["id"]: c["weight"] for c in load_cases()}
        rows = [{**r, "weight": weights.get(r["id"], r["weight"])} for r in res["cases"]]
        if all(a["weight"] == b["weight"] for a, b in zip(rows, res["cases"], strict=True)):
            return best  # no weight changed since that eval
        res = {**res, "pass_rate": summarize(rows), "_path": str(root / best["results_file"])}
        return summary(res, root)

    async def propose(state: LoopState) -> dict:
        n = state["n"]
        cases[:] = load_cases()  # feedback.yaml may have changed since the last iteration
        state["best"] = reweigh(state["best"])
        tid = obs.trace_id(f"{cfg['loop_id']}:{n}")
        meta = {"iteration": n, "hypothesis_id": f"hyp/{n}", "loop_id": cfg["loop_id"]}
        with obs.trace(f"iteration:{n}", tid, ["optimizer"], meta, span_name="optimizer") as (
            span,
            callbacks,
        ):
            llm = make_chat("optimizer", max_tokens=8000).with_structured_output(
                Hypothesis, method="function_calling", include_raw=True
            )
            fails = failures(state["best"]["results_file"], cases)
            msgs = [
                ("system", (Path(__file__).parent / "prompt.md").read_text()),
                (
                    "human",
                    context(
                        state["best"],
                        cases,
                        fs,
                        fails,
                        state.get("archive", []) if use_archive else None,
                    ),
                ),  # fmt: skip
            ]
            try:
                res = await llm.ainvoke(msgs, config={"callbacks": callbacks})
                hyp: Hypothesis = res["parsed"]
                if not use_archive:
                    hyp.parent = base
                u = res["raw"].usage_metadata or {}
            except Exception as e:  # noqa: BLE001 — provider errors are rows, not crashes
                span.update(output={"error": str(e)})
                return {"hyp": None, "reason": f"provider_error: {str(e)[:80]}", "opt_eur": 0.0,
                        "best": state["best"], "leak_regex": None}  # fmt: skip
            usd, _ = cost_usd(
                model_for("optimizer"), u.get("input_tokens", 0), u.get("output_tokens", 0)
            )
            span.update(
                output=hyp.model_dump(exclude={"content"}), metadata={**meta, "cost_usd": usd}
            )
        reason, leak_regex = None, leaks_expected(hyp.content, cases, {f["id"] for f in fails})
        if len(hyp.content) > CAPS[hyp.target]:
            reason = "too_long"
        elif hyp.target == "tools.yaml" and _tool_keys(hyp.content) != TOOL_KEYS:
            reason = "too_long"  # same bucket, the file is structurally invalid
        elif jev.enabled():  # the literal check is information; the semantic one is the veto
            current = (fs["prompts"] / hyp.target).read_text()
            leak_cost = 0.0

            async def ask_leak(state, questions):
                nonlocal leak_cost
                r = await jev.ask(state, questions)
                leak_cost += r.cost_usd
                return r

            try:
                leak = await leaks_semantic(
                    prompt_diff(current, hyp.content, hyp.target),
                    fails,
                    ask_leak,
                    float(os.environ.get("JEV_LEAK_MIN", "0.7")),
                )
            except Exception as e:  # noqa: BLE001 — unverified anti-leak rejects, like provider errors
                leak, reason = None, f"jev_error: {str(e)[:80]}"
            usd += leak_cost  # the leak check's Jev cost is optimizer cost too
            if leak:
                reason = f"leaks_expected {leak}"
        elif leak_regex:
            reason = f"leaks_expected {leak_regex}"
        return {"hyp": hyp.model_dump(), "reason": reason, "opt_eur": usd_to_eur(usd),
                "best": state["best"], "leak_regex": leak_regex}  # fmt: skip

    def apply(state: LoopState) -> dict:
        if not state["hyp"]:
            return {"sha": "—", "parent": base, "parent_fallback": False}
        n, hyp = state["n"], state["hyp"]
        eligible = {a["branch"] for a in state.get("archive", [])}
        parent = hyp["parent"] if hyp["parent"] in eligible else base
        fallback = parent != hyp["parent"]
        # cut from the base tip so acceptance stays a fast-forward (base moves on every
        # rejection with an evidence commit); inherit the parent's prompt files instead
        repo.git.checkout("-b", f"hyp/{n}", base)
        if parent != base:
            repo.git.checkout(parent, "--", str(fs["prompts"]))
        target = fs["prompts"] / hyp["target"]
        target.write_text(hyp["content"])
        origin = f" (from {parent})" if parent != base else ""
        sha = commit([target], f"hyp/{n}{origin}: {hyp['title']}\n\n{hyp['rationale']}")
        return {"sha": sha, "parent": parent, "parent_fallback": fallback}

    def after_apply(state: LoopState) -> Literal["evaluate", "decide"]:
        return "decide" if state["reason"] else "evaluate"

    async def evaluate(state: LoopState) -> dict:
        n, before = state["n"], state["best"]
        res = await evaluate_now(n, f"hyp/{n}")
        after = summary(res, root)
        reason = gate(before, after, tol)
        if reason is None:  # passed on one run: confirm the holdout on a second run
            conf = await evaluate_now(n, f"hyp/{n}", "holdout")
            hold2 = conf["pass_rate"]["per_split"]["holdout"]
            after["confirm"] = {
                "holdout": hold2,
                "results_file": str(Path(conf["_path"]).resolve().relative_to(root.resolve())),
                "cost_eur": conf["cost"]["total_eur"],
            }
            after["cost_eur"] = round(after["cost_eur"] + conf["cost"]["total_eur"], 4)
            if hold2 + 1e-9 < before["holdout"]:
                reason = f"holdout_confirm {hold2:.0%} vs {before['holdout']:.0%}"
            else:
                after["holdout"] = min(after["holdout"], hold2)  # the lower bound becomes the bar
        return {"results": after, "reason": reason}

    def decide(state: LoopState) -> dict:
        n, hyp, best, after = state["n"], state["hyp"], state["best"], state.get("results")
        accepted = after is not None and state["reason"] is None
        repo.git.checkout(base)
        if accepted:
            repo.git.merge("--ff-only", f"hyp/{n}")
            best = after
        for c in cases:
            f = c.get("feedback")
            if not f:
                continue
            rec = doc["feedback"].setdefault(
                f["id"],
                {
                    "ref": c["id"],
                    "note": f["note"],
                    "flagged_before_iteration": n,
                    "recovered_at_iteration": None,
                },  # fmt: skip
            )
            if (
                accepted
                and rec["recovered_at_iteration"] is None
                and c["id"] in after["passed_ids"]
            ):
                rec["recovered_at_iteration"] = n
                rec["iterations_to_recover"] = n - rec["flagged_before_iteration"] + 1
        eval_eur = after["cost_eur"] if after else 0.0
        total = eval_eur + state["opt_eur"]
        cumulative = state["cumulative_eur"] + total
        it = {
            "n": n,
            "hyp_id": f"hyp/{n}",
            "sha": state["sha"],
            "target": hyp["target"] if hyp else None,
            "title": (
                (f"(from {state['parent']}) " if state.get("parent", base) != base else "")
                + (hyp["title"] if hyp else "(no hypothesis)")
            ),
            "parent": state.get("parent", base),
            "parent_fallback": state.get("parent_fallback", False),
            "verdict": "accepted" if accepted else f"rejected: {state['reason']}",
            "reason": state["reason"],
            "leak_regex": state.get("leak_regex"),
            "visible": after["visible"] if after else None,
            "holdout": after["holdout"] if after else None,
            "per_category_visible": after["per_category_visible"] if after else None,
            "visible_delta": after["visible"] - state["best"]["visible"] if after else None,
            "holdout_delta": after["holdout"] - state["best"]["holdout"] if after else None,
            "holdout_confirm": after.get("confirm") if after else None,
            "delta_cases": delta_cases(
                state["best"]["per_category_visible"], after["per_category_visible"], tol
            )
            if after
            else {},
            "cost_eur": {
                "eval": eval_eur,
                "optimizer": round(state["opt_eur"], 4),
                "total": round(total, 4),
            },
            "cumulative_eur": round(cumulative, 4),
            "results_file": after["results_file"] if after else None,
            "best_visible": best["visible"],
        }
        record(it)
        if accepted:
            archive: list[dict] = []
        elif after:  # rejected with an eval behind it: a legitimate parent
            archive = state.get("archive", []) + [
                {
                    "branch": f"hyp/{n}",
                    "title": hyp["title"],
                    "target": hyp["target"],
                    "reason": state["reason"],
                    "visible": after["visible"],
                    "holdout": after["holdout"],
                    "delta_cases": it["delta_cases"],
                    "diff": repo.git.diff(base, f"hyp/{n}", "--", str(fs["prompts"])),
                }
            ]
        else:  # static rejection: nothing measured, nothing to inherit
            archive = state.get("archive", [])
        changelog = fs["changelog"]
        if not changelog.exists() or "| n | hyp |" not in changelog.read_text():
            changelog.write_text(
                "# Changelog\n\nOne row per optimizer hypothesis. "
                "Δ in pass-rate points (pp) or visible cases.\n\n" + HEADER
            )
        with changelog.open("a") as f:
            f.write(changelog_row(it))
        to_add = [changelog, loop_file] + ([root / after["results_file"]] if after else [])
        if after and after.get("confirm"):
            to_add.append(root / after["confirm"]["results_file"])
        if hyp and hyp["proposed_cases"]:
            pf = fs["proposed"]
            prev = yaml.safe_load(pf.read_text()) if pf.exists() else {"cases": []}
            prev["cases"] += [{"hypothesis": f"hyp/{n}", **c} for c in hyp["proposed_cases"]]
            pf.write_text(yaml.safe_dump(prev, allow_unicode=True, sort_keys=False))
            to_add.append(pf)
        history = state["history"] + [{"pass_rate": best["visible"], "cost_eur": total}]
        done = state["done"] + 1
        stop = None
        if cumulative >= cfg["max_eur"]:
            stop = "max_eur"
        elif done >= cfg["max_iterations"]:
            stop = "max_iterations"
        elif should_stop(history, cfg["min_gain"], WINDOW):
            stop = "budget"
        if stop:
            doc["stop"] = {"reason": stop, "at": n}
            loop_file.write_text(json.dumps(doc, indent=2, ensure_ascii=False))
            with changelog.open("a") as f:
                f.write(
                    f"\nstop: {stop} after hyp/{n} (loop {cfg['loop_id']}, {cumulative:.2f} €)\n\n"
                )
        commit(
            to_add, f"hyp/{n} {it['verdict']}: {it['title']}" + (f"; stop: {stop}" if stop else "")
        )
        print(changelog_row(it), end="")
        return {
            "best": best,
            "history": history,
            "cumulative_eur": cumulative,
            "done": done,
            "stop": stop,
            "archive": archive,
            "n": n + 1,
            "hyp": None,
            "reason": None,
            "results": None,
            "sha": "",
        }

    def after_decide(state: LoopState) -> Literal["propose", "__end__"]:
        return END if state["stop"] else "propose"

    g = StateGraph(LoopState)
    for name, fn in (
        ("baseline", baseline),
        ("propose", propose),
        ("apply", apply),
        ("evaluate", evaluate),
        ("decide", decide),
    ):
        g.add_node(name, fn)
    g.add_edge(START, "baseline")
    g.add_edge("baseline", "propose")
    g.add_edge("propose", "apply")
    g.add_conditional_edges("apply", after_apply)
    g.add_edge("evaluate", "decide")
    g.add_conditional_edges("decide", after_decide)
    return g.compile()


def _tool_keys(content: str) -> set:
    try:
        d = yaml.safe_load(content)
        return set(d) if isinstance(d, dict) else set()
    except yaml.YAMLError:
        return set()


def main() -> None:
    load_env()
    p = argparse.ArgumentParser()
    p.add_argument(
        "--max-iterations", type=int, default=int(os.environ.get("MAX_ITERATIONS", "12"))
    )
    p.add_argument("--max-eur", type=float, default=float(os.environ.get("MAX_TOTAL_EUR", "10")))
    p.add_argument(
        "--min-gain", type=float, default=float(os.environ.get("BUDGET_MIN_GAIN_PER_EUR", "0.03"))
    )
    p.add_argument("--seed", type=int, default=int(os.environ.get("EVAL_SEED", "42")))
    p.add_argument("--base", default="main", help="branch the loop runs on and fast-forwards")
    p.add_argument("--archive", action="store_true",
                   help="unmeasured: start from a rejected branch")  # fmt: skip
    a = p.parse_args()
    loop_id = datetime.now(UTC).strftime("%Y-%m-%dT%H%M%S+0000")
    cfg = {
        "max_iterations": a.max_iterations,
        "max_eur": a.max_eur,
        "min_gain": a.min_gain,
        "seed": a.seed,
        "loop_id": loop_id,
        "base": a.base,
        "archive": a.archive,
    }
    import asyncio

    final = asyncio.run(
        build_loop(cfg).ainvoke({}, config={"recursion_limit": 10 * a.max_iterations + 10})
    )
    obs.flush()
    print(
        f"stop: {final['stop']}  best visible {final['best']['visible']}  "
        f"spent {final['cumulative_eur']:.2f} €"
    )
    print(f"→ {paths()['loops'] / (loop_id + '.json')}")


if __name__ == "__main__":
    main()
