"""src/jev.py: normalised answers, cost, question loading. No network: FakeJev only."""

import os

from src import jev
from src.pricing import cost_usd
from tests.fakes import CHOICE_REFUSED, FakeJev, noul


def test_jev_price_input_only():
    usd, known = cost_usd("jev-1.13.0", 1_000_000, 500)
    assert known and usd == 0.042


async def test_ask_normalises_answers_and_cost():
    fake = FakeJev({"r": CHOICE_REFUSED, "n": noul(0.3),
                    "s": {"type": "score", "score": 1.0, "confidence": 0.8, "legend": {0: "a", 1: "b"},
                          "probabilities": {0: 0.2, 1: 0.8}}})  # fmt: skip
    qs = {"r": {"type": "choice", "instructions": "x", "criteria": {"refused": "a", "answered": "b"}},
          "n": {"type": "noul", "instructions": "y"},
          "s": {"type": "score", "instructions": "z", "criteria": ["a", "b"]}}  # fmt: skip
    res = await jev.ask({"answer": "no"}, qs, client=fake)
    assert res.answers["r"].value == "refused" and res.answers["r"].confidence == 0.9
    assert res.answers["n"].noul == 0.3 and res.answers["n"].confidence is None
    assert res.answers["s"].value == 1.0 and res.answers["s"].probabilities == {"0": 0.2, "1": 0.8}
    assert res.model == "jev-fake" and res.input_tokens == 100 and res.cost_usd == 0.0
    assert fake.calls == [({"answer": "no"}, ["r", "n", "s"])]


async def test_ask_tolerates_none_usage_counts():
    """SDK types usage counts as optional; None → 0 tokens and 0 cost."""
    fake = FakeJev({"q": CHOICE_REFUSED}, usage={"input_tokens": None, "output_tokens": None})
    qs = {
        "q": {"type": "choice", "instructions": "x", "criteria": {"refused": "a", "answered": "b"}}
    }
    res = await jev.ask({"answer": "no"}, qs, client=fake)
    assert res.input_tokens == 0
    assert res.cost_usd == 0.0


def test_enabled_reads_env(monkeypatch):
    # same pattern as tests/test_observability.py's no_keys fixture — stub load_env so
    # the real .env (which does carry a key on this machine) can't repopulate what we just deleted.
    monkeypatch.setattr(jev, "load_env", lambda: None)
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    assert not jev.enabled()
    monkeypatch.setenv("TYPESAFE_API_KEY", "k")
    assert jev.enabled()
    monkeypatch.setenv("JEV_MODEL", "jev-9.9.9")
    assert jev.model() == "jev-9.9.9"


def test_questions_load_and_format():
    qs = jev.load_questions()
    for name in ("refused", "leak", "gen_category", "gen_answerable", "gen_one_reading",
                 "gen_sql_matches", "gen_expected_in_outputs", "gen_duplicate"):  # fmt: skip
        assert qs[name]["type"] in ("choice", "noul")
    assert set(qs["refused"]["criteria"]) == {"refused", "answered"}
    assert set(qs["gen_category"]["criteria"]) == {"lookup", "aggregation", "reasoning", "refusal"}
    assert (
        "customers in Italy" in jev.question("leak", question="customers in Italy")["instructions"]
    )
    assert "existing.L02" in jev.question("gen_duplicate", other_id="L02")["instructions"]


def test_default_client_loads_env_before_reading_key(monkeypatch):
    """Regression: _default_client() used to read os.environ["TYPESAFE_API_KEY"] without calling
    load_env() first, so a key that only lives in .env raised KeyError (found by the real-call
    smoke test)."""
    monkeypatch.setattr(jev, "_client", None)
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.setattr(jev, "load_env", lambda: os.environ.__setitem__("TYPESAFE_API_KEY", "k"))
    assert jev._default_client() is not None
