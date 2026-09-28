"""The loop's git, changelog and stop mechanics on a throwaway clone, with fake evals and LLM.
No provider key, no Langfuse keys, no network."""

import json

import pytest
from git import Repo

from src.optimizer import loop as lp

ROOT = lp.ROOT
VIS = {"lookup": 6 / 7, "aggregation": 5 / 7, "reasoning": 3 / 8, "refusal": 0.0}


def fake_results(path, visible, holdout, per_cat):
    out = {"git_sha": "fake", "pass_rate": {"total": visible, "per_split": {"visible": visible, "holdout": holdout},
           "per_category": per_cat, "per_category_visible": per_cat, "n": 40},
           "cost": {"total_usd": 0.5, "total_eur": 0.4}, "cases": []}  # fmt: skip
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out))
    return path


@pytest.fixture
def clone(tmp_path, monkeypatch):
    r = Repo.clone_from(ROOT, tmp_path / "r")
    r.git.config("user.name", "t")
    r.git.config("user.email", "t@example.com")
    monkeypatch.setenv("AGENT_MODEL", "a")
    monkeypatch.setenv("JUDGE_MODEL", "j")
    monkeypatch.setenv("OPTIMIZER_MODEL", "claude-sonnet-4-6")
    monkeypatch.setattr(lp, "RUNS_LOG", tmp_path / "runs.jsonl")
    monkeypatch.setattr(lp.jev, "enabled", lambda: False)
    return tmp_path / "r"


async def test_accept_then_reject_then_stop(clone, monkeypatch):
    # baseline 47%/40%; hyp 1: visible up, holdout 0.5 then confirmed at 0.4 → accepted, bar = 0.4;
    # hyp 2: no gain → rejected without confirmation; hyp 3: gain, holdout 0.4 on the first run
    # but 0.3 on the confirmation → rejected: holdout_confirm
    up = dict(VIS, refusal=4 / 8)
    plan = iter([(0.4667, 0.4, VIS), (0.6, 0.5, up), (None, 0.4, up), (0.6, 0.5, up),
                 (0.7, 0.4, up), (None, 0.3, up)])  # fmt: skip
    calls = {"n": 0, "splits": []}

    async def run_evals(ns):
        calls["n"] += 1
        calls["splits"].append(ns.split)
        v, h, c = next(plan)
        assert (ns.split == "holdout") == (v is None)
        return fake_results(clone / "evals" / "results" / f"fake{calls['n']}.json", v or 0, h, c)

    class Raw:
        usage_metadata = {"input_tokens": 1000, "output_tokens": 500}

    class LLM:
        def with_structured_output(self, *a, **k):
            return self

        async def ainvoke(self, msgs, config=None):
            hyp = lp.Hypothesis(title="always call final_answer", rationale="r", target="system.md",
                               content="Always end with final_answer. Today is {as_of}.",
                               proposed_cases=[lp.ProposedCase(category="refusal", input="q", expected="True", check="refused", why="w")])  # fmt: skip
            return {"parsed": hyp, "raw": Raw()}

    monkeypatch.setattr(lp, "run_evals", lambda ns: run_evals(ns))
    monkeypatch.setattr(lp, "make_chat", lambda *a, **k: LLM())
    cfg = {"max_iterations": 3, "max_eur": 10, "min_gain": 0.03, "seed": 42, "loop_id": "T"}
    final = await lp.build_loop(cfg, root=clone).ainvoke({}, config={"recursion_limit": 40})

    repo = Repo(clone)
    assert final["stop"] == "max_iterations" and final["best"]["visible"] == 0.6
    assert repo.active_branch.name == "main" and not repo.is_dirty() and not repo.untracked_files
    assert {b.name for b in repo.branches} >= {"main", "hyp/1", "hyp/2"}
    assert "final_answer" in (clone / "src/agent_under_test/prompts/system.md").read_text()
    rows = [ln for ln in (clone / "CHANGELOG.md").read_text().splitlines() if ln.startswith("| ")]
    rows = rows[-3:]  # the clone carries the real CHANGELOG; only the new rows matter
    assert "| accepted |" in rows[0] and "rejected: no_gain" in rows[1]
    assert "rejected: holdout_confirm 30% vs 40%" in rows[2] and "(confirm 30%)" in rows[2]
    assert "(confirm 40%)" in rows[0]
    assert calls["splits"] == ["all", "all", "holdout", "all", "all", "holdout"]
    assert "+13.3pp" in rows[0] and "| 0/0/0/+" in rows[0]  # refusal delta = 4/8 ÷ (1/n_refusal)
    doc = json.loads((clone / "evals/results/loops/T.json").read_text())
    assert [i["verdict"] for i in doc["iterations"]] == [
        "baseline", "accepted", "rejected: no_gain", "rejected: holdout_confirm 30% vs 40%",
    ]  # fmt: skip
    assert doc["stop"] == {"reason": "max_iterations", "at": 3}
    assert doc["iterations"][1]["holdout_confirm"]["holdout"] == 0.4
    assert doc["iterations"][1]["holdout"] == 0.4  # min of the two runs is the new bar
    assert (clone / doc["iterations"][3]["holdout_confirm"]["results_file"]).exists()
    assert doc["iterations"][1]["cost_eur"]["optimizer"] > 0  # optimizer tokens are charged
    assert "hyp/1" in (clone / "evals/proposed.yaml").read_text()
    msgs = [c.message.splitlines()[0] for c in repo.iter_commits("main", max_count=5)]
    assert msgs[0].startswith("hyp/3 rejected: holdout_confirm")
    assert msgs[1].startswith("hyp/2 rejected: no_gain") and msgs[2].startswith("hyp/1 accepted")
    assert msgs[3].startswith("hyp/1: always call final_answer")
    assert msgs[4].startswith("loop T: baseline")


TOOLS = "run_sql: a\nlookup_customer: b\ncompute: c\nfinal_answer: d\n"


def fake_llm(parents):
    """Fake optimizer: one hypothesis per call; each item is a parent branch, or a
    (parent, target) pair. Only one iterator: make_chat is called once per iteration."""
    parents = iter(parents)

    class Raw:
        usage_metadata = {"input_tokens": 1000, "output_tokens": 500}

    class LLM:
        def with_structured_output(self, *a, **k):
            return self

        async def ainvoke(self, msgs, config=None):
            item = next(parents)
            parent, target = item if isinstance(item, tuple) else (item, "system.md")
            content = TOOLS if target == "tools.yaml" else "Always call final_answer. {as_of}"
            hyp = lp.Hypothesis(title="t", rationale="r", target=target, content=content,
                                parent=parent)  # fmt: skip
            return {"parsed": hyp, "raw": Raw()}

    return LLM()


async def test_base_branch(clone, monkeypatch):
    # the loop runs on exp/t, not main: starts there, commits there, leaves main untouched
    repo = Repo(clone)
    repo.git.checkout("-b", "exp/t")
    main_sha = repo.commit("main").hexsha
    plan = iter([(0.4667, 0.4, VIS), (0.6, 0.5, dict(VIS, refusal=4 / 8)), (None, 0.5, VIS)])
    calls = {"n": 0}

    async def run_evals(ns):
        calls["n"] += 1
        v, h, c = next(plan)
        return fake_results(clone / "evals" / "results" / f"b{calls['n']}.json", v or 0, h, c)

    monkeypatch.setattr(lp, "run_evals", lambda ns: run_evals(ns))
    llm = fake_llm(["exp/t"])
    monkeypatch.setattr(lp, "make_chat", lambda *a, **k: llm)
    cfg = {"max_iterations": 1, "max_eur": 10, "min_gain": 0.03, "seed": 42, "loop_id": "B",
           "base": "exp/t"}  # fmt: skip
    final = await lp.build_loop(cfg, root=clone).ainvoke({}, config={"recursion_limit": 40})
    assert final["stop"] == "max_iterations"
    assert repo.active_branch.name == "exp/t" and not repo.is_dirty()
    assert repo.commit("main").hexsha == main_sha  # nothing landed on main
    assert repo.commit("exp/t").message.startswith("hyp/1 accepted")


async def test_archive_parent(clone, monkeypatch):
    # hyp 1 (tools.yaml): gains visible, holdout 0.4 → 0.3: rejected holdout, enters the archive
    # hyp 2 (system.md, parent hyp/1): inherits hyp/1's tools.yaml, gains, holdout 0.5
    #   confirmed 0.5: accepted → archive emptied
    # hyp 3: parent hyp/1 again, but hyp/1 is no longer eligible: fallback to main; no_gain
    up = dict(VIS, refusal=4 / 8)
    plan = iter([(0.4667, 0.4, VIS), (0.6, 0.3, up), (0.7, 0.5, up), (None, 0.5, up),
                 (0.7, 0.5, up)])  # fmt: skip
    calls = {"n": 0}

    async def run_evals(ns):
        calls["n"] += 1
        v, h, c = next(plan)
        return fake_results(clone / "evals" / "results" / f"a{calls['n']}.json", v or 0, h, c)

    monkeypatch.setattr(lp, "run_evals", lambda ns: run_evals(ns))
    llm = fake_llm([("main", "tools.yaml"), "hyp/1", "hyp/1"])
    monkeypatch.setattr(lp, "make_chat", lambda *a, **k: llm)
    cfg = {"max_iterations": 3, "max_eur": 10, "min_gain": 0.03, "seed": 42, "loop_id": "A"}
    final = await lp.build_loop(cfg, root=clone).ainvoke({}, config={"recursion_limit": 40})

    repo = Repo(clone)
    assert final["stop"] == "max_iterations"
    assert [a["branch"] for a in final["archive"]] == ["hyp/3"]  # emptied at hyp/2, refilled
    doc = json.loads((clone / "evals/results/loops/A.json").read_text())
    its = doc["iterations"]
    assert [i["verdict"] for i in its] == [
        "baseline", "rejected: holdout", "accepted", "rejected: no_gain",
    ]  # fmt: skip
    assert [i.get("parent") for i in its] == [None, "main", "hyp/1", "main"]
    assert [i.get("parent_fallback") for i in its] == [None, False, False, True]
    # every branch is cut from the main tip (hyp/1's evidence commit, then hyp/2's), so
    # acceptance is a fast-forward; the parent's prompt files are inherited, not its commit
    assert repo.commit("hyp/2").parents[0].message.startswith("hyp/1 rejected")
    assert repo.commit("hyp/3").parents[0].message.startswith("hyp/2 accepted")
    tools = "src/agent_under_test/prompts/tools.yaml"
    assert repo.git.show(f"hyp/2:{tools}") == TOOLS.rstrip("\n")  # inherited from hyp/1
    assert repo.git.show(f"main:{tools}") == TOOLS.rstrip("\n")  # and fast-forwarded
    # hyp/3 rewrote system.md to what hyp/2 had already landed: an empty diff, on record
    assert final["archive"][0]["branch"] == "hyp/3" and final["archive"][0]["diff"] == ""
    assert repo.commit("hyp/2").message.startswith("hyp/2 (from hyp/1): t")
    assert repo.commit("hyp/3").message.startswith("hyp/3: t")
    rows = [ln for ln in (clone / "CHANGELOG.md").read_text().splitlines() if ln.startswith("| ")]
    assert "| (from hyp/1) t |" in rows[-2] and "| t |" in rows[-1]


async def test_no_archive_forces_base(clone, monkeypatch):
    # same shape as test_archive_parent, archive off: hyp 2 names hyp/1 but starts from main
    up = dict(VIS, refusal=4 / 8)
    plan = iter([(0.4667, 0.4, VIS), (0.6, 0.3, up), (0.7, 0.5, up), (None, 0.5, up)])
    calls = {"n": 0}

    async def run_evals(ns):
        calls["n"] += 1
        v, h, c = next(plan)
        return fake_results(clone / "evals" / "results" / f"c{calls['n']}.json", v or 0, h, c)

    monkeypatch.setattr(lp, "run_evals", lambda ns: run_evals(ns))
    llm = fake_llm([("main", "tools.yaml"), "hyp/1"])
    monkeypatch.setattr(lp, "make_chat", lambda *a, **k: llm)
    cfg = {"max_iterations": 2, "max_eur": 10, "min_gain": 0.03, "seed": 42, "loop_id": "C",
           "archive": False}  # fmt: skip
    await lp.build_loop(cfg, root=clone).ainvoke({}, config={"recursion_limit": 40})
    repo = Repo(clone)
    its = json.loads((clone / "evals/results/loops/C.json").read_text())["iterations"]
    assert [i.get("parent") for i in its] == [None, "main", "main"]
    assert [i.get("parent_fallback") for i in its] == [None, False, False]
    assert repo.commit("hyp/2").message.startswith("hyp/2: t")  # no "(from hyp/1)"
    tools = "src/agent_under_test/prompts/tools.yaml"
    assert repo.git.show(f"hyp/2:{tools}") != TOOLS.rstrip("\n")  # nothing inherited
