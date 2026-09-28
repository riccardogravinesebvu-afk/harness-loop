"""Offline checks for the gate, the anti-leak filter, the changelog row and the loop graph shape."""

from src.optimizer import loop as lp
from src.optimizer.gates import delta_cases, gate, leaks_expected, leaks_semantic, tolerances
from src.optimizer.loop import CATS, HEADER, changelog_row
from tests.fakes import FakeJev, noul

CASES = [
    {"id": "L01", "category": "lookup", "split": "visible", "expected": 8},
    {"id": "L05", "category": "lookup", "split": "visible", "expected": "Helios"},
    {"id": "L04", "category": "lookup", "split": "holdout", "expected": "2026-09-09"},
    {"id": "A01", "category": "aggregation", "split": "visible", "expected": 139520},
    {"id": "A06", "category": "aggregation", "split": "visible", "expected": 5813.33},
    {
        "id": "R01",
        "category": "reasoning",
        "split": "visible",
        "expected": "judge",
        "reference": "Outstanding is 24,890.00 across 3 invoices.",
    },  # fmt: skip
    {"id": "F01", "category": "refusal", "split": "visible", "expected": True},
]


def test_tolerance_is_one_visible_case_per_category():
    assert tolerances(CASES) == {
        "lookup": 0.5,
        "aggregation": 0.5,
        "reasoning": 1.0,
        "refusal": 1.0,
    }


def _s(visible, holdout, **cats):
    return {"visible": visible, "holdout": holdout, "per_category_visible": cats}


def test_gate_reasons_in_order():
    tol = {"lookup": 1 / 7, "aggregation": 1 / 7, "reasoning": 1 / 8, "refusal": 1 / 8}
    before = _s(0.47, 0.4, lookup=6 / 7, aggregation=5 / 7, reasoning=3 / 8, refusal=0.0)
    assert gate(before, _s(0.47, 0.5, **before["per_category_visible"]), tol) == "no_gain"
    two_down = dict(before["per_category_visible"], lookup=4 / 7, refusal=6 / 8)
    assert gate(before, _s(0.6, 0.4, **two_down), tol) == "gate lookup -2"
    one_down = dict(before["per_category_visible"], lookup=5 / 7, refusal=6 / 8)
    assert gate(before, _s(0.6, 0.3, **one_down), tol) == "holdout"
    assert gate(before, _s(0.6, 0.4, **one_down), tol) is None
    assert delta_cases(before["per_category_visible"], one_down, tol) == {
        "lookup": -1, "aggregation": 0, "reasoning": 0, "refusal": 6
    }  # fmt: skip


def test_leak_filter_catches_visible_literals_only():
    assert leaks_expected("Answer with numbers.", CASES) is None
    assert leaks_expected("Total invoiced is 139,520 EUR", CASES) == "A01:139520"
    assert leaks_expected("Helios Energy owes the most", CASES) == "L05:Helios"
    assert leaks_expected("outstanding is 24890.00", CASES) == "R01:24,890.00"
    assert leaks_expected("due 2026-09-09", CASES) is None  # holdout: not visible to the optimizer
    assert leaks_expected("Today is 2026-09-01", CASES) is None  # a year in a date is not an answer
    assert leaks_expected("There are 8 customers", CASES) is None  # short literals are noise
    # only failing cases count: a passing case's accepted word may be general knowledge
    assert leaks_expected("'CH' for Switzerland... wait, Helios", CASES, failing={"A01"}) is None
    assert leaks_expected("Helios owes most", CASES, failing={"L05"}) == "L05:Helios"


def test_changelog_row_matches_header():
    it = {"n": 1, "hyp_id": "hyp/1", "sha": "abc1234", "target": "system.md", "title": "t",
          "verdict": "accepted", "visible_delta": 0.1, "holdout_delta": 0.0,
          "delta_cases": {c: 1 for c in CATS}, "cost_eur": {"total": 0.45},
          "cumulative_eur": 0.9, "results_file": "evals/results/x.json"}  # fmt: skip
    assert changelog_row(it).count("|") == HEADER.splitlines()[0].count("|")
    assert "+10.0pp" in changelog_row(it) and "+1/+1/+1/+1" in changelog_row(it)


async def test_semantic_leak_flags_the_worst_case_above_threshold():
    fake = FakeJev(lambda name, state: noul(0.9 if name == "leak_L02" else 0.1))

    async def ask(state, questions):
        from src import jev

        return await jev.ask(state, questions, client=fake)

    failing = [
        {"id": "L02", "input": "Which country is Fortuna in?"},
        {"id": "L09", "input": "Customers in Italy?"},
    ]
    assert await leaks_semantic("+ Fortuna is in CH", failing, ask, 0.7) == "L02:0.90"
    assert await leaks_semantic("+ country is an ISO code", [failing[1]], ask, 0.7) is None
    assert await leaks_semantic("", failing, ask, 0.7) is None and len(fake.calls) == 2
    state, names = fake.calls[0]
    assert state == {"prompt_diff": "+ Fortuna is in CH"} and names == ["leak_L02", "leak_L09"]


async def test_semantic_leak_cost_is_accumulated_into_optimizer_cost():
    """loop.py's `propose` wraps `ask` the same way scripts/replay_leak.py does, so the leak
    check's Jev cost is added to `usd` before it becomes `opt_eur` — it must not be silently
    dropped."""
    from src.jev import Answer, JevResult

    leak_cost = 0.0

    async def ask_leak(state, questions):
        nonlocal leak_cost
        r = JevResult({n: Answer("noul", None, {}, None, 0.9) for n in questions},
                     "jev-1.13.0", 100, 0.0042)  # fmt: skip
        leak_cost += r.cost_usd
        return r

    failing = [{"id": "L02", "input": "Which country is Fortuna in?"}]
    usd = 0.01
    leak = await leaks_semantic("+ Fortuna is in CH", failing, ask_leak, 0.7)
    usd += leak_cost
    assert leak == "L02:0.90" and leak_cost == 0.0042 and usd > 0.01


def test_prompt_diff_is_unified():
    d = lp.prompt_diff("a\nb\n", "a\nc\n", "system.md")
    assert "-b" in d and "+c" in d and "system.md" in d


def test_context_archive_section(tmp_path):
    fs = {"prompts": tmp_path / "p", "code": [], "changelog": tmp_path / "CHANGELOG.md"}
    fs["prompts"].mkdir()
    (fs["prompts"] / "system.md").write_text("s")
    best = {"per_category_visible": {c: 0.5 for c in lp.CATS}}
    cases = [{"category": c, "split": "visible"} for c in lp.CATS]
    arc = [{"branch": "hyp/7", "title": "Teach refusals", "target": "system.md",
            "reason": "holdout", "visible": 0.6, "holdout": 0.3,
            "delta_cases": {"refusal": 2}, "diff": "+refuse forecasts"}]  # fmt: skip
    on = lp.context(best, cases, fs, [], archive=arc)
    assert "## Archive" in on and "hyp/7" in on and "rejected: holdout" in on
    assert "refusal +2" in on and "+refuse forecasts" in on
    empty = lp.context(best, cases, fs, [], archive=[])
    assert "## Archive" in empty and "empty" in empty
    off = lp.context(best, cases, fs, [])
    assert "## Archive" not in off
