"""Refusal read from the answer text: Jev decides above the confidence floor, the
agent's flag below it, and an empty answer or no key never calls Jev. A Jev error also falls
back to the flag and is recorded, never crashing the run (errors never disappear)."""

import hashlib

import pytest

from evals import refusal
from evals import run as er
from src import jev
from tests.fakes import CHOICE_ANSWERED, CHOICE_REFUSED, CHOICE_UNSURE, FakeJev


@pytest.fixture(autouse=True)
def key(monkeypatch):
    # same pattern as tests/test_jev.py — stub load_env so the real .env (which does
    # carry a key on this machine) can't repopulate what a test just deleted.
    monkeypatch.setattr(jev, "load_env", lambda: None)
    monkeypatch.setenv("TYPESAFE_API_KEY", "k")
    monkeypatch.setenv("JEV_REFUSED_MIN_CONF", "0.6")


async def test_jev_overrides_flag_when_confident():
    r = await refusal.refused_from_text("I cannot determine that from the ledger.", False,
                                        client=FakeJev({"refused": CHOICE_REFUSED}))  # fmt: skip
    assert r == {"refused": True, "refused_by": "jev", "confidence": 0.9, "cost_usd": 0.0}
    r = await refusal.refused_from_text("Total is 139,520 EUR.", True,
                                        client=FakeJev({"refused": CHOICE_ANSWERED}))  # fmt: skip
    assert r["refused"] is False and r["refused_by"] == "jev"


async def test_flag_wins_below_confidence_floor():
    r = await refusal.refused_from_text("Hmm.", True, client=FakeJev({"refused": CHOICE_UNSURE}))
    assert r["refused"] is True and r["refused_by"] == "flag" and r["confidence"] == 0.2


async def test_empty_answer_or_no_key_uses_flag(monkeypatch):
    fake = FakeJev({"refused": CHOICE_REFUSED})
    r = await refusal.refused_from_text("", False, client=fake)
    assert r == {"refused": False, "refused_by": "flag", "confidence": None, "cost_usd": 0.0}
    monkeypatch.delenv("TYPESAFE_API_KEY")
    r = await refusal.refused_from_text("anything", True, client=fake)
    assert r["refused_by"] == "flag" and fake.calls == []


async def test_jev_error_falls_back_to_flag_and_is_recorded():
    def boom(name, state):
        raise RuntimeError("system_one timed out")

    r = await refusal.refused_from_text("I cannot determine that.", True, client=FakeJev(boom))
    assert r["refused"] is True and r["refused_by"] == "flag" and r["confidence"] is None
    assert r["cost_usd"] == 0.0 and "system_one timed out" in r["error"]


def test_dataset_sha_is_sha256_of_dataset_file():
    path = er.HERE / "dataset.yaml"
    assert er.dataset_sha() == hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def test_refused_version_follows_jev(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    assert er.refused_version() == 1
    monkeypatch.setenv("TYPESAFE_API_KEY", "k")
    assert er.refused_version() == 2


async def test_eval_case_records_jev_decision(monkeypatch):
    class Res:
        answer = "I cannot answer that from the ledger."
        refused = False  # the agent's flag disagrees with its own text
        error = None
        tool_calls = []
        input_tokens = output_tokens = 10
        cost_usd = 0.001
        latency_s = 0.1
        model = "a"

    async def fake_run(*a, **k):
        return Res()

    async def fake_refusal(answer, flag, client=None):
        return {"refused": True, "refused_by": "jev", "confidence": 0.9, "cost_usd": 0.0002}

    monkeypatch.setattr(er, "run", fake_run)
    monkeypatch.setattr(er, "refused_from_text", fake_refusal)
    case = {"id": "F01", "category": "refusal", "split": "visible", "weight": 1, "source": "seed",
            "check": "refused", "expected": True, "input": "q"}  # fmt: skip
    row, _ = await er._eval_case(case, graph=None, seed=42, callbacks=[])
    assert row["passed"] and row["refused"] and row["refused_flag"] is False
    assert row["refused_by"] == "jev" and row["jev_cost_usd"] == 0.0002
    assert er.errors_summary([row]) == {"no_final_answer": 0, "max_steps": 0,
                                        "provider_error": 0, "refused_disagree": 1,
                                        "refused_error": 0}  # fmt: skip


async def test_errors_summary_counts_refused_error(monkeypatch):
    class Res:
        answer = "I cannot answer that from the ledger."
        refused = True
        error = None
        tool_calls = []
        input_tokens = output_tokens = 10
        cost_usd = 0.001
        latency_s = 0.1
        model = "a"

    async def fake_run(*a, **k):
        return Res()

    async def fake_refusal(answer, flag, client=None):
        return {"refused": flag, "refused_by": "flag", "confidence": None, "cost_usd": 0.0,
                "error": "jev_error: boom"}  # fmt: skip

    monkeypatch.setattr(er, "run", fake_run)
    monkeypatch.setattr(er, "refused_from_text", fake_refusal)
    case = {"id": "F02", "category": "refusal", "split": "visible", "weight": 1, "source": "seed",
            "check": "refused", "expected": True, "input": "q"}  # fmt: skip
    row, _ = await er._eval_case(case, graph=None, seed=42, callbacks=[])
    assert row["refused_error"] == "jev_error: boom"
    assert er.errors_summary([row])["refused_error"] == 1
