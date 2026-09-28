"""evals/generate.py: proposals become candidates with an expected computed by SQL, never by a
model; verification and outcomes are pure functions over Jev answers; --apply appends accepted
cases and writes provenance; --stage brings proposals into review."""

import argparse
import sqlite3
import sys

import pytest
import yaml

from evals import dataset as ds
from evals import generate as g
from src.agent_under_test import ledger
from tests.fakes import FakeJev, noul

CAT_OK = {"type": "choice", "choice": "aggregation", "confidence": 0.95,
          "probabilities": {"aggregation": 0.97, "lookup": 0.02, "reasoning": 0.01, "refusal": 0.0}}  # fmt: skip
EXISTING = [{"id": "A01", "input": "What is the total amount invoiced?"},
            {"id": "L01", "input": "How many customers are in the ledger?"}]  # fmt: skip


def _table(dup_a01=0.1):
    def f(name, state):
        if name == "category":
            return CAT_OK
        if name == "dup_A01":
            return noul(dup_a01)
        if name.startswith("dup_"):
            return noul(0.1)
        return noul(0.9)

    return f


@pytest.fixture(autouse=True)
def optimizer_model(monkeypatch):
    # a fresh clone has no .env: the tests name the model themselves
    monkeypatch.setenv("OPTIMIZER_MODEL", "claude-sonnet-4-6")


@pytest.fixture(scope="module")
def con(tmp_path_factory):
    return sqlite3.connect(ledger.build_db(tmp_path_factory.mktemp("db") / "ledger.sqlite"))


def test_expected_from_sql_scalar_and_list(con):
    c = g.Candidate(category="aggregation", input="Total invoiced?", check="number",
                    sql="SELECT SUM(amount_eur) FROM invoices", reference=None, why="w")  # fmt: skip
    assert g.compute_expected(c, con) == (139520.0, None)
    c = g.Candidate(category="lookup", input="Italian customers?", check="contains_any",
                    sql="SELECT name FROM customers WHERE country='IT'", reference=None, why="w")  # fmt: skip
    exp, err = g.compute_expected(c, con)
    assert err is None and sorted(exp) == ["Bruno Serramenti Srl", "Gallo Ristorazione Srl"]


def test_expected_rejects_check_not_valid_for_category(con):
    c = g.Candidate(category="aggregation", input="q", check="contains_any",
                    sql="SELECT 1", reference=None, why="w")  # fmt: skip
    exp, err = g.compute_expected(c, con)
    assert exp is None and "contains_any" in err and "aggregation" in err
    c = g.Candidate(category="refusal", input="q", check="number", sql="SELECT 1", why="w")
    exp, err = g.compute_expected(c, con)
    assert exp is None and err is not None


def test_expected_rejects_bad_shapes(con):
    c = g.Candidate(category="lookup", input="q", check="number",
                    sql="SELECT name FROM customers", reference=None, why="w")  # fmt: skip
    assert g.compute_expected(c, con)[1] == "number check needs one numeric value"
    c = g.Candidate(category="lookup", input="q", check="contains",
                    sql="DELETE FROM customers", reference=None, why="w")  # fmt: skip
    assert g.compute_expected(c, con)[1].startswith("only SELECT")
    c = g.Candidate(
        category="refusal", input="Will X pay?", check="refused", sql=None, reference=None, why="w"
    )
    assert g.compute_expected(c, con) == (True, None)
    c = g.Candidate(
        category="reasoning", input="q", check="judge", sql=None, reference="ref", why="w"
    )
    assert g.compute_expected(c, con) == ("judge", None)


async def test_propose_uses_structured_output():
    class Raw:
        usage_metadata = {"input_tokens": 1000, "output_tokens": 200}

    class LLM:
        def with_structured_output(self, *a, **k):
            return self

        async def ainvoke(self, msgs, config=None):
            assert "lookup" in msgs[1][1] and "L01" in msgs[1][1]
            return {
                "parsed": g.Proposal(
                    candidates=[
                        g.Candidate(
                            category="lookup",
                            input="How many invoices?",
                            check="number",
                            sql="SELECT COUNT(*) FROM invoices",
                            reference=None,
                            why="coverage",
                        )
                    ]
                ),
                "raw": Raw(),
            }

    cands, usd = await g.propose(
        "lookup", 1, [{"id": "L01", "input": "x", "category": "lookup"}], llm=LLM()
    )
    assert cands[0].sql.startswith("SELECT") and usd > 0


async def test_propose_raises_a_clear_error_when_parsing_fails():
    class Raw:
        usage_metadata = {"input_tokens": 1000, "output_tokens": 200}

    class LLM:
        def with_structured_output(self, *a, **k):
            return self

        async def ainvoke(self, msgs, config=None):
            return {"parsed": None, "raw": Raw(), "parsing_error": ValueError("bad json")}

    with pytest.raises(ValueError, match="bad json"):
        await g.propose("lookup", 1, [], llm=LLM())


async def test_verify_fans_out_one_request():
    c = g.Candidate(category="aggregation", input="Sum of payments?", check="number",
                    sql="SELECT SUM(amount_eur) FROM payments", reference=None, why="w")  # fmt: skip
    fake = FakeJev(_table(dup_a01=0.8))
    v, usd = await g.verify(c, 83140.0, EXISTING, client=fake)
    assert len(fake.calls) == 1
    state, names = fake.calls[0]
    assert (
        state["candidate"]["input"] == "Sum of payments?"
        and state["existing"]["A01"] == EXISTING[0]["input"]
    )
    assert set(names) == {
        "category",
        "answerable",
        "one_reading",
        "sql_matches",
        "dup_A01",
        "dup_L01",
    }
    assert v["category"] == {"value": "aggregation", "confidence": 0.95}
    assert v["duplicate"] == {"id": "A01", "p": 0.8} and v["expected_in_outputs"] is None


def test_outcome_thresholds(monkeypatch):
    monkeypatch.setenv("JEV_GEN_MIN_CONF", "0.8")
    monkeypatch.setenv("JEV_GEN_MIN_NOUL", "0.8")
    monkeypatch.setenv("JEV_GEN_MAX_DUP", "0.3")
    c = g.Candidate(
        category="aggregation", input="q", check="number", sql="SELECT 1", reference=None, why="w"
    )
    good = {"category": {"value": "aggregation", "confidence": 0.9}, "answerable": 0.9,
            "one_reading": 0.85, "sql_matches": 0.9, "expected_in_outputs": None,
            "duplicate": {"id": "A01", "p": 0.1}}  # fmt: skip
    assert g.outcome(c, good, None, all_review=False) == ("auto_add", "all checks clear")
    assert g.outcome(c, good, None, all_review=True)[0] == "review"
    lookup = g.Candidate(category="lookup", input="q", check="contains_any", sql="SELECT 1",
                         reference=None, why="w")  # fmt: skip
    lookup_good = dict(good, category={"value": "lookup", "confidence": 0.9})
    assert g.outcome(lookup, lookup_good, None, all_review=False)[0] == "review"
    assert g.outcome(c, dict(good, duplicate={"id": "A01", "p": 0.75}), None, False) == (
        "discard",
        "duplicate of A01 (0.75)",
    )
    assert g.outcome(c, dict(good, answerable=0.15), None, False) == ("discard", "answerable 0.15")
    assert g.outcome(c, dict(good, one_reading=0.5), None, False) == ("review", "one_reading 0.50")
    assert g.outcome(
        c, dict(good, category={"value": "lookup", "confidence": 0.9}), None, False
    ) == ("review", "category lookup != aggregation")
    assert g.outcome(c, good, "sql error: x", False) == ("discard", "sql error: x")
    judge = g.Candidate(
        category="reasoning", input="q", check="judge", sql=None, reference="r", why="w"
    )
    assert g.outcome(
        judge,
        dict(good, category={"value": "reasoning", "confidence": 0.9}, sql_matches=None),
        None,
        False,
    ) == ("review", "judge and refused cases are never auto-added")


def test_next_id_and_dataset_line():
    assert g.next_id([{"id": "L01"}, {"id": "G03"}]) == "G04"
    rec = {"category": "lookup", "check": "contains", "expected": "Helios Energy Ltd",
           "input": 'Who is "the" energy customer?', "reference": None}  # fmt: skip
    line = g.dataset_line(rec, "G01", "visible")
    assert line.startswith(
        "  - {id: G01, category: lookup, split: visible, check: contains, expected: "
    )
    assert "source: generated" in line and "weight: 1" in line
    parsed = yaml.safe_load("cases:\n" + line)["cases"][0]
    assert (
        parsed["input"] == 'Who is "the" energy customer?'
        and parsed["expected"] == "Helios Energy Ltd"
    )


def _cand(inp, sql, **kw):
    base = {
        "category": "lookup",
        "input": inp,
        "check": "number",
        "sql": sql,
        "reference": None,
        "why": "w",
        "source": "generated",
        "tables": ds.tables_of(sql),
        "expected": None,
        "sql_error": None,
        "jev": None,
        "outcome": "review",
        "reason": "x",
        "decision": None,
    }
    return {**base, **kw}


async def test_apply_writes_provenance_and_keeps_undecided(tmp_path, con, monkeypatch):
    monkeypatch.setattr(g.jev, "enabled", lambda: False)  # no key, no network: verifiers null
    dsp = tmp_path / "dataset.yaml"
    dsp.write_text(
        "cases:\n  - {id: L01, category: lookup, split: visible, check: number, "
        'expected: 8, input: "n?"}\n'
    )
    cand, prov = tmp_path / "candidates.yaml", tmp_path / "provenance.yaml"
    recs = [
        _cand(
            "q1",
            "SELECT COUNT(*) FROM invoices",
            expected=24.0,
            outcome="auto_add",
            jev={"category": {"value": "lookup", "confidence": 0.9}},
        ),
        _cand("q2", "SELECT COUNT(*) FROM customers", expected=8.0, decision="reject"),
        _cand("q3", "SELECT 3", expected=3.0),  # undecided, stays
        _cand(
            "How many payments?",
            "SELECT COUNT(*) FROM payments",
            source="hand",
            split="holdout",
            decision="accept",
        ),
    ]
    g.write_candidates(recs, cand, batch="b1")
    doc = yaml.safe_load(cand.read_text())
    doc["review"]["minutes"] = 6
    cand.write_text(yaml.safe_dump(doc, sort_keys=False))
    added = await g.apply(cand, dsp, prov, con=con)
    assert added == ["G01", "G02"]
    cases = yaml.safe_load(dsp.read_text())["cases"]
    assert [(c["id"], c["split"], c["source"]) for c in cases[1:]] == [
        ("G01", "visible", "generated"),
        ("G02", "holdout", "hand"),
    ]
    assert cases[2]["expected"] == 13  # computed from the SQL at apply time, never typed
    left = yaml.safe_load(cand.read_text())
    assert [r["input"] for r in left["candidates"]] == ["q3"] and left["review"]["batch"] == "b1"
    records = ds.load_provenance(prov)
    # G01 is an undecided auto_add: it enters the dataset but carries no human confirmation.
    assert records[0]["id"] == "G01" and records[0]["human"] is None
    assert [(r["id"], r["human"]["decision"]) for r in records[1:]] == [
        (None, "reject"),
        ("G02", "accept"),
    ]
    assert records[0]["derivation"] == {
        "kind": "sql",
        "sql": "SELECT COUNT(*) FROM invoices",
        "tables": ["invoices"],
    }
    assert records[0]["trace"] == {"generator": g.model_for("optimizer"), "batch": "b1"}
    # minutes are divided only among the records a human actually decided: q2 and the hand case
    assert (
        records[1]["human"]["minutes_per_case"] == 3.0
        and records[2]["human"]["minutes_per_case"] == 3.0
    )
    assert records[0]["dataset_sha"] == ds.dataset_sha(dsp) and records[1]["dataset_sha"] is None
    assert records[2]["derivation"]["tables"] == ["payments"] and records[2]["source"] == "hand"


async def test_apply_moves_undecided_discards_into_provenance_and_drops_them(
    tmp_path, con, monkeypatch
):
    monkeypatch.setattr(g.jev, "enabled", lambda: False)
    dsp = tmp_path / "dataset.yaml"
    dsp.write_text("cases:\n")
    cand, prov = tmp_path / "candidates.yaml", tmp_path / "provenance.yaml"
    recs = [
        _cand("q1", "SELECT 1", outcome="discard", reason="dup"),  # decision: None, never reviewed
        _cand("q2", "SELECT 2", outcome="review"),  # still undecided, stays in candidates.yaml
    ]
    g.write_candidates(recs, cand, batch="b1")
    added = await g.apply(cand, dsp, prov, con=con)
    assert added == []
    assert not yaml.safe_load(dsp.read_text())["cases"]  # the discard never entered the dataset
    left = yaml.safe_load(cand.read_text())
    assert [r["input"] for r in left["candidates"]] == ["q2"]
    records = ds.load_provenance(prov)
    assert len(records) == 1
    assert records[0]["input"] == "q1"
    assert records[0]["outcome"] == "discard" and records[0]["human"] is None


async def test_apply_refuses_generated_holdout_and_bad_decisions(tmp_path, con, monkeypatch):
    monkeypatch.setattr(g.jev, "enabled", lambda: False)
    dsp = tmp_path / "dataset.yaml"
    dsp.write_text("cases:\n")
    cand = tmp_path / "candidates.yaml"
    g.write_candidates(
        [_cand("q", "SELECT 1", expected=1.0, split="holdout", decision="accept")], cand
    )
    with pytest.raises(ValueError, match="only source: hand"):
        await g.apply(cand, dsp, tmp_path / "p.yaml", con=con)
    g.write_candidates([_cand("q", "SELECT 1", expected=1.0, decision="holdout")], cand)
    with pytest.raises(ValueError, match="accept or reject"):
        await g.apply(cand, dsp, tmp_path / "p.yaml", con=con)
    g.write_candidates([_cand("q", None, decision="accept")], cand)
    with pytest.raises(ValueError, match="only SELECT"):
        await g.apply(cand, dsp, tmp_path / "p.yaml", con=con)


def test_stage_moves_proposed_rows_into_candidates(tmp_path):
    sel = tmp_path / "selection.yaml"
    sel.write_text(
        yaml.safe_dump(
            {
                "rows": [
                    {"reason": "empty_cell", "target": "a:b:c"},
                    {
                        "reason": "disagreement",
                        "proposed": {
                            "hypothesis": "hyp/2",
                            "category": "lookup",
                            "input": "Largest invoice?",
                            "expected": "INV-014",
                            "check": "contains",
                            "why": "w",
                        },
                    },
                ]
            }
        )
    )
    cand = tmp_path / "candidates.yaml"
    assert g.stage(sel, cand) == 1 and g.stage(sel, cand) == 0  # idempotent
    review, recs = g.read_candidates(cand)
    r = recs[0]
    assert r["source"] == "proposed" and r["sql"] is None and r["expected"] is None
    assert r["trace"] == {"hypothesis": "hyp/2"} and r["proposed_expected"] == "INV-014"
    assert r["outcome"] == "review" and r["reason"].startswith("proposed by hyp/2: write the sql")


async def test_apply_rejects_holdout_when_record_carries_a_jev_verdict(tmp_path, con, monkeypatch):
    monkeypatch.setattr(g.jev, "enabled", lambda: False)
    dsp = tmp_path / "dataset.yaml"
    dsp.write_text("cases:\n")
    cand, prov = tmp_path / "candidates.yaml", tmp_path / "provenance.yaml"
    g.write_candidates(
        [
            _cand(
                "q",
                "SELECT 1",
                source="hand",
                split="holdout",
                decision="accept",
                jev={"category": {"value": "lookup", "confidence": 0.9}},
            )
        ],
        cand,
    )
    with pytest.raises(ValueError, match="Jev verdict"):
        await g.apply(cand, dsp, prov, con=con)


async def test_apply_minimal_hand_record_gets_why_and_outcome_defaults(tmp_path, con, monkeypatch):
    monkeypatch.setattr(g.jev, "enabled", lambda: False)
    dsp = tmp_path / "dataset.yaml"
    dsp.write_text("cases:\n")
    cand, prov = tmp_path / "candidates.yaml", tmp_path / "provenance.yaml"
    minimal = {
        "category": "lookup",
        "input": "How many customers?",
        "check": "number",
        "sql": "SELECT COUNT(*) FROM customers",
        "source": "hand",
        "split": "visible",
        "decision": "accept",
    }
    g.write_candidates([minimal], cand)
    added = await g.apply(cand, dsp, prov, con=con)
    assert added == ["G01"]
    cases = yaml.safe_load(dsp.read_text())["cases"]
    assert cases[0]["expected"] == 8


async def test_apply_recomputes_expected_for_proposed_and_hand_never_trusts_typed(
    tmp_path, con, monkeypatch
):
    monkeypatch.setattr(g.jev, "enabled", lambda: False)
    dsp = tmp_path / "dataset.yaml"
    dsp.write_text("cases:\n")
    cand, prov = tmp_path / "candidates.yaml", tmp_path / "provenance.yaml"
    # proposed record: typed (wrong) expected but no SQL yet -> must recompute and fail
    g.write_candidates(
        [
            _cand(
                "q",
                None,
                source="proposed",
                expected=999.0,
                decision="accept",
            )
        ],
        cand,
    )
    with pytest.raises(ValueError, match="only SELECT"):
        await g.apply(cand, dsp, prov, con=con)
    # nothing written: dataset, candidates and provenance are untouched
    assert dsp.read_text() == "cases:\n"
    assert yaml.safe_load(cand.read_text())["candidates"][0]["decision"] == "accept"
    assert not prov.exists()


async def test_apply_hand_record_ignores_typed_expected_uses_sql(tmp_path, con, monkeypatch):
    monkeypatch.setattr(g.jev, "enabled", lambda: False)
    dsp = tmp_path / "dataset.yaml"
    dsp.write_text("cases:\n")
    cand, prov = tmp_path / "candidates.yaml", tmp_path / "provenance.yaml"
    g.write_candidates(
        [
            _cand(
                "How many customers?",
                "SELECT COUNT(*) FROM customers",
                source="hand",
                expected=-1.0,  # wrong typed value, must never be trusted
                decision="accept",
            )
        ],
        cand,
    )
    added = await g.apply(cand, dsp, prov, con=con)
    assert added == ["G01"]
    cases = yaml.safe_load(dsp.read_text())["cases"]
    assert cases[0]["expected"] == 8  # the SQL result, not the typed -1.0


def test_stage_preserves_recorded_review_minutes(tmp_path):
    cand = tmp_path / "candidates.yaml"
    g.write_candidates([], cand, batch="b1")
    doc = yaml.safe_load(cand.read_text())
    doc["review"]["minutes"] = 12
    cand.write_text(yaml.safe_dump(doc, sort_keys=False))
    sel = tmp_path / "selection.yaml"
    sel.write_text(
        yaml.safe_dump(
            {
                "rows": [
                    {
                        "reason": "disagreement",
                        "proposed": {
                            "hypothesis": "hyp/1",
                            "category": "lookup",
                            "input": "New question?",
                            "expected": "x",
                            "check": "contains",
                            "why": "w",
                        },
                    }
                ]
            }
        )
    )
    assert g.stage(sel, cand) == 1
    review, _ = g.read_candidates(cand)
    assert review["minutes"] == 12 and review["batch"] == "b1"


def test_main_rejects_malformed_cell(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["generate", "--cell", "bad"])
    with pytest.raises(SystemExit):
        g.main()
    assert "category:check:tables" in capsys.readouterr().err

    for cell in ("lookup:bogus:customers", "typo:number:customers"):
        monkeypatch.setattr(sys, "argv", ["generate", "--cell", cell])
        with pytest.raises(SystemExit):
            g.main()
        assert "category:check:tables" in capsys.readouterr().err


async def test_main_async_refuses_to_overwrite_pending_candidates(tmp_path, monkeypatch):
    # belt and suspenders: never let this test reach a real Jev/LLM call even if the guard
    # under test is missing or broken.
    monkeypatch.setattr(g.jev, "enabled", lambda: False)
    cand = tmp_path / "candidates.yaml"
    g.write_candidates([_cand("q", "SELECT 1")], cand)
    monkeypatch.setattr(g, "CANDIDATES", cand)
    args = argparse.Namespace(
        apply=False, stage=False, cell=None, per_category=10, all_review=False
    )
    with pytest.raises(SystemExit, match="pending"):
        await g.main_async(args)


async def test_main_async_records_a_jev_error_as_a_discard_row_and_continues(tmp_path, monkeypatch):
    cand = tmp_path / "candidates.yaml"
    monkeypatch.setattr(g, "CANDIDATES", cand)
    # write_candidates(records) is called with no path in main_async, so its own default
    # argument (bound to the real CANDIDATES at import time) must be redirected too, or this
    # test would write into the repo's real evals/candidates.yaml.
    orig_write_candidates = g.write_candidates
    monkeypatch.setattr(
        g, "write_candidates", lambda records, *a, **kw: orig_write_candidates(records, cand)
    )
    monkeypatch.setattr(g.jev, "enabled", lambda: True)

    async def fake_propose(cat, n, existing, cell=None):
        return [
            g.Candidate(category=cat, input="q1", check="number", sql="SELECT 1", why="w"),
            g.Candidate(category=cat, input="q2", check="number", sql="SELECT 1", why="w"),
        ], 0.001

    async def fake_verify(c, expected, existing):
        if c.input == "q1":
            raise RuntimeError("boom")
        return {"category": {"value": c.category, "confidence": 0.9}, "answerable": 0.9,
                "one_reading": 0.9, "sql_matches": 0.9, "expected_in_outputs": None,
                "duplicate": {"id": "", "p": 0.0}}, 0.001  # fmt: skip

    monkeypatch.setattr(g, "propose", fake_propose)
    monkeypatch.setattr(g, "verify", fake_verify)
    args = argparse.Namespace(
        apply=False, stage=False, cell="lookup:number:customers", per_category=2, all_review=False
    )
    path = await g.main_async(args)
    recs = g.read_candidates(path)[1]
    by_input = {r["input"]: r for r in recs}
    assert by_input["q1"]["outcome"] == "discard"
    assert by_input["q1"]["reason"].startswith("jev_error:")
    assert by_input["q2"]["outcome"] in ("auto_add", "review")  # the batch continued past q1


async def test_apply_rejects_invalid_split(tmp_path, con, monkeypatch):
    monkeypatch.setattr(g.jev, "enabled", lambda: False)
    dsp = tmp_path / "dataset.yaml"
    dsp.write_text("cases:\n")
    cand, prov = tmp_path / "candidates.yaml", tmp_path / "provenance.yaml"
    g.write_candidates(
        [_cand("q", "SELECT 1", split="typo", decision="accept")],
        cand,
    )
    with pytest.raises(ValueError, match="visible or holdout"):
        await g.apply(cand, dsp, prov, con=con)
    assert dsp.read_text() == "cases:\n"
    assert not prov.exists()


def test_provenance_records_who_reviewed():
    r = {"category": "lookup", "input": "q", "check": "number", "sql": "SELECT 1"}
    assert g.provenance_record(r, "G01", "accept", "b", 1.0)["human"]["by"] == "author"
    rec = g.provenance_record(r, "G01", "reject", "b", None, reviewer="claude-opus-5-5")
    assert rec["human"]["by"] == "claude-opus-5-5"
