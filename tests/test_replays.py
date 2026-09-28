from scripts import replay_leak as rl
from scripts import replay_refused as rr


def _doc(rows):
    return {"cases": rows, "pass_rate": {}}


def test_regrade_only_touches_refusal_checks():
    rows = [
        {"id": "F01", "category": "refusal", "split": "visible", "weight": 1, "check": "refused",
         "expected": True, "answer": "I cannot say.", "refused": False, "passed": False},
        {"id": "A01", "category": "aggregation", "split": "visible", "weight": 1, "check": "number",
         "expected": 5, "answer": "5", "refused": False, "passed": True},
    ]  # fmt: skip
    new = rr.regrade(_doc(rows), {"F01": True, "A01": True})
    by_id = {r["id"]: r for r in new["cases"]}
    assert (
        by_id["F01"]["passed"] and by_id["F01"]["refused"] and by_id["F01"]["refused_flag"] is False
    )
    assert by_id["A01"]["passed"]  # a number check ignores the refusal decision
    assert new["pass_rate"]["total"] == 1.0


def test_regrade_preserves_an_existing_refused_flag_over_a_jev_decided_refused():
    """A row from a results file already regraded once (fee6f58) carries `refused_flag` (the
    agent's own flag) separately from `refused` (possibly already Jev-decided). Regrading it
    again must not mistake `refused` for the agent's flag."""
    rows = [
        {"id": "F01", "category": "refusal", "split": "visible", "weight": 1, "check": "refused",
         "expected": True, "answer": "I cannot say.", "refused": True, "refused_flag": False,
         "passed": True},
    ]  # fmt: skip
    new = rr.regrade(_doc(rows), {"F01": False})
    assert new["cases"][0]["refused_flag"] is False


def test_regrade_file_records_a_jev_error_without_crashing():
    doc = _doc([
        {"id": "F01", "category": "refusal", "split": "visible", "weight": 1, "check": "refused",
         "expected": True, "answer": "I cannot say.", "refused": False, "passed": False},
    ])  # fmt: skip
    pairs = [("F01", {"refused": False, "refused_by": "flag", "confidence": None,
                      "cost_usd": 0.0, "error": "jev_error: boom"})]  # fmt: skip
    result = rr.regrade_file(doc, pairs)
    assert result["summary"]["errors"] == [{"id": "F01", "error": "jev_error: boom"}]
    assert result["summary"]["disagree"] == 0  # refused_by flag, not jev: no disagreement counted
    assert result["doc"]["cases"][0]["refused"] is False  # falls back to the flag, doesn't crash


def test_replay_gates_reports_changed_decisions():
    base = {"visible": 0.5, "holdout": 0.4, "per_category_visible": {"refusal": 0.0, "lookup": 1.0}}
    hyp = {"visible": 0.6, "holdout": 0.4, "per_category_visible": {"refusal": 0.5, "lookup": 1.0}}
    loops = [{"loop_id": "L", "iterations": [
        {"n": 0, "hyp_id": None, "reason": None, "results_file": "b.json"},
        {"n": 1, "hyp_id": "hyp/1", "reason": "no_gain", "results_file": "h.json"},
    ]}]  # fmt: skip
    rows = rr.replay_gates(
        loops, {"b.json": base, "h.json": hyp}, {"b.json": {"refusal": 0.5, "lookup": 0.5}}
    )
    assert rows == [{"loop": "L", "hyp": "hyp/1", "old_reason": "no_gain", "new_reason": None,
                     "changed": True}]  # fmt: skip


def test_replay_gates_uses_each_loops_own_tolerance():
    """40->44 cases changed the refusal tolerance from 1/8=0.125 to 1/10=0.1 (review finding on
    replay 1): the gate must use the tolerance of the dataset each loop actually ran on (its
    `before` results file), not whichever results file's tolerance was computed last."""
    before = {"visible": 0.5, "holdout": 0.4, "per_category_visible": {"refusal": 0.5}}
    after = {"visible": 0.6, "holdout": 0.4, "per_category_visible": {"refusal": 0.375}}
    loops = [
        {"loop_id": "A", "iterations": [
            {"n": 0, "hyp_id": None, "reason": None, "results_file": "a_before.json"},
            {"n": 1, "hyp_id": "hyp/a1", "reason": None, "results_file": "a_after.json"},
        ]},
        {"loop_id": "B", "iterations": [
            {"n": 0, "hyp_id": None, "reason": None, "results_file": "b_before.json"},
            {"n": 1, "hyp_id": "hyp/b1", "reason": None, "results_file": "b_after.json"},
        ]},
    ]  # fmt: skip
    new_rates = {
        "a_before.json": before, "a_after.json": after,
        "b_before.json": before, "b_after.json": after,
    }  # fmt: skip
    tol_by_file = {"a_before.json": {"refusal": 0.125}, "b_before.json": {"refusal": 0.1}}
    rows = rr.replay_gates(loops, new_rates, tol_by_file)
    by_loop = {r["loop"]: r for r in rows}
    # loop A ran on the old (looser) tolerance: a one-case drop is within budget, stays accepted
    assert by_loop["A"]["new_reason"] is None
    assert not by_loop["A"]["changed"]
    # loop B ran on the new (tighter) tolerance: the same drop now exceeds budget, gate rejects it
    assert by_loop["B"]["new_reason"] == "gate refusal -1"
    assert by_loop["B"]["changed"]


def test_hypotheses_pair_each_hyp_with_its_base_file():
    loops = [{"loop_id": "L", "iterations": [
        {"n": 0, "hyp_id": None, "reason": None, "results_file": "b.json"},
        {"n": 1, "hyp_id": "hyp/1", "reason": None, "results_file": "h1.json"},
        {"n": 2, "hyp_id": "hyp/2", "reason": "leaks_expected L02:Switzerland", "results_file": None},
    ]}]  # fmt: skip
    assert rl.hypotheses(loops) == [
        {"hyp": "hyp/1", "n": 1, "old_reason": None, "before_file": "b.json"},
        {
            "hyp": "hyp/2",
            "n": 2,
            "old_reason": "leaks_expected L02:Switzerland",
            "before_file": "h1.json",
        },
    ]


def test_classify_against_todays_regex():
    assert rl.classify("L02:Switzerland", "L02:0.91") == "agree_leak"
    assert rl.classify("L02:Switzerland", None) == "regex_only"
    assert rl.classify(None, "L09:0.80") == "jev_only"
    assert rl.classify(None, None) == "agree_clean"
