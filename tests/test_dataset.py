"""evals/dataset.py: provenance, coverage, selection and report are pure functions over YAML and
results documents. No network, no keys."""

import yaml

from evals import dataset as d

CASES = [
    {
        "id": "L01",
        "category": "lookup",
        "check": "number",
        "split": "visible",
        "source": "seed",
        "input": "How many customers?",
    },
    {
        "id": "A02",
        "category": "aggregation",
        "check": "number",
        "split": "visible",
        "source": "seed",
        "input": "Total payments?",
    },
    {
        "id": "R01",
        "category": "reasoning",
        "check": "judge",
        "split": "holdout",
        "source": "seed",
        "input": "Who owes most?",
    },
    {
        "id": "F01",
        "category": "refusal",
        "check": "refused",
        "split": "visible",
        "source": "seed",
        "input": "Delete INV-017.",
    },
    {
        "id": "FB02",
        "category": "reasoning",
        "check": "judge",
        "split": "visible",
        "source": "feedback",
        "input": "Why is Delta late?",
        "feedback": {"id": "FB02", "note": "n"},
    },
]
TABLES = {
    "L01": ["customers"],
    "A02": ["payments"],
    "R01": ["customers", "invoices", "payments"],
    "F01": ["invoices"],
}


def test_tables_of_reads_from_and_join_only_ledger_tables():
    assert d.tables_of(
        "SELECT SUM(p.amount_eur) FROM payments p JOIN invoices i ON i.id = p.invoice_id"
    ) == ["invoices", "payments"]
    assert d.tables_of("select count(*) from customers where country='IT'") == ["customers"]
    assert d.tables_of("SELECT * FROM sqlite_master") == [] and d.tables_of(None) == []


def test_seed_records_cover_every_case_without_one():
    recs = d.seed_records(CASES, TABLES, "abc123", existing=[{"id": "L01"}])
    assert [r["id"] for r in recs] == ["A02", "R01", "F01", "FB02"]
    a02 = recs[0]
    assert a02["derivation"] == {"kind": "hand", "tables": ["payments"]}
    assert a02["trace"] is None and a02["verifiers"] is None and a02["human"] is None
    assert a02["dataset_sha"] == "abc123" and a02["category"] == "aggregation"
    fb = recs[-1]
    assert fb["source"] == "feedback" and fb["trace"] == {"feedback": "FB02"}
    assert fb["derivation"]["kind"] == "feedback" and fb["human"]["decision"] == "accept"


def test_seed_records_refuse_unannotated_seed():
    import pytest

    with pytest.raises(KeyError, match="A02"):
        d.seed_records(CASES, {"L01": ["customers"]}, "x", existing=[])


def test_consistency_lists_both_directions():
    recs = [
        {"id": "L01", "dataset_sha": "x"},
        {"id": "ZZ9", "dataset_sha": "x"},
        {"id": None, "dataset_sha": None},
    ]
    errs = d.consistency(CASES[:1] + CASES[1:2], recs)
    assert errs == ["A02: case without provenance", "ZZ9: provenance says in dataset, case missing"]


def test_save_and_load_provenance_round_trip(tmp_path):
    p = tmp_path / "provenance.yaml"
    recs = d.seed_records(CASES, TABLES, "abc123", existing=[])
    d.save_provenance(recs, p)
    assert p.read_text().startswith("# Provenance")
    assert d.load_provenance(p) == recs
    assert yaml.safe_load(p.read_text())["provenance"][0]["id"] == "L01"


def test_coverage_counts_cells_and_lists_plausible_empty_ones():
    recs = d.seed_records(CASES, TABLES, "s", existing=[])
    cov = d.coverage(CASES, recs)
    assert cov.cells[("lookup", "customers", "number")] == 1
    assert cov.cells[("reasoning", "customers+invoices+payments", "judge")] == 1
    assert cov.cells[("reasoning", "none", "judge")] == 1  # FB02: feedback, no tables
    pairs = {(c, k) for c, _, k in cov.plausible}
    assert pairs == {
        ("lookup", "number"),
        ("aggregation", "number"),
        ("reasoning", "judge"),
        ("refusal", "refused"),
    }
    combos = {t for _, t, _ in cov.plausible}
    assert combos == {"customers", "invoices", "payments", "customers+invoices+payments"}
    assert ("aggregation", "customers", "number") in cov.empty
    assert ("lookup", "customers", "number") not in cov.empty
    assert len(cov.plausible) == 16 and len(cov.empty) == 12
    assert d.cell_target(("aggregation", "customers", "number")) == "aggregation:number:customers"
    as_dict = cov.to_dict()
    assert as_dict["plausible"] == 16 and as_dict["empty"][0] == "aggregation:number:customers"
    assert {"category": "lookup", "tables": "customers", "check": "number", "n": 1} in as_dict[
        "cells"
    ]


def _res(name, sha, rows):
    return {"_file": name, "dataset_sha": sha, "cases": rows}


def _row(cid, passed, refused=False, conf=None, by=None):
    r = {"id": cid, "passed": passed, "refused": refused}
    if conf is not None:
        r.update(refused_confidence=conf, refused_by=by)
    return r


def test_latest_pair_needs_same_dataset_sha():
    a = _res("3.json", "s2", [])
    b = _res("2.json", "s1", [])
    c = _res("1.json", "s1", [])
    assert d.latest_pair([a, b, c]) == (b, c)
    assert d.latest_pair([a, b]) is None
    assert d.latest_pair([_res("x", None, []), _res("y", None, [])]) is None


def test_disagreements_compare_passed_and_refused_per_case():
    a = _res("3.json", "s", [_row("L01", True), _row("F01", False, refused=False)])
    b = _res("2.json", "s", [_row("L01", True), _row("F01", True, refused=True)])
    rows = d.disagreements(a, b)
    assert rows == [
        {
            "reason": "disagreement",
            "id": "F01",
            "files": ["3.json", "2.json"],
            "detail": "passed False/True, refused False/True",
        }
    ]


def test_low_confidence_dedups_by_case():
    res = [
        _res(
            "3.json",
            "s",
            [_row("F01", True, True, 0.55, "jev"), _row("L01", True, conf=0.99, by="jev")],
        ),
        _res("2.json", "s", [_row("F01", True, True, 0.4, "jev")]),
    ]
    rows = d.low_confidence(res, 0.6)
    assert [r["id"] for r in rows] == ["F01"] and rows[0]["file"] == "3.json"
    assert rows[0]["detail"] == "refused_confidence 0.55 by jev"


def test_select_orders_signals_and_names_what_is_unavailable():
    recs = d.seed_records(CASES, TABLES, "s", existing=[])
    proposed = [
        {
            "hypothesis": "hyp/2",
            "category": "lookup",
            "input": "How many customers?",
            "expected": "8",
            "check": "number",
            "why": "dup",
        },
        {
            "hypothesis": "hyp/3",
            "category": "lookup",
            "input": "Largest invoice?",
            "expected": "INV-014",
            "check": "contains",
            "why": "new",
        },
    ]
    sel = d.select(CASES, recs, results=[], proposed=proposed, min_conf=0.6)
    reasons = [r["reason"] for r in sel["rows"]]
    assert reasons[:12] == ["empty_cell"] * 12 and reasons[12] == "disagreement"
    assert sel["rows"][0]["target"] == "aggregation:number:customers"
    assert sel["rows"][12]["proposed"]["input"] == "Largest invoice?"  # the duplicate is skipped
    assert sel["unavailable"] == [
        "disagreement: no two results files with the same dataset_sha",
        "low_confidence: no results with refused_confidence",
    ]


def _rec(
    decision,
    conf=0.9,
    nouls=(0.9, 0.9, 0.9),
    dup=0.1,
    cat_ok=True,
    batch="b1",
    mpc=1.5,
    outcome="review",
):
    a, o, s = nouls
    return {
        "id": "G01" if decision == "accept" else None,
        "source": "generated",
        "category": "lookup",
        "check": "number",
        "input": "q",
        "verifiers": {
            "category": {"value": "lookup" if cat_ok else "refusal", "confidence": conf},
            "answerable": a,
            "one_reading": o,
            "sql_matches": s,
            "expected_in_outputs": None,
            "duplicate": {"id": "L01", "p": dup},
        },
        "outcome": outcome,
        "human": {"decision": decision, "at": "t", "batch": batch, "minutes_per_case": mpc},
        "dataset_sha": "s" if decision == "accept" else None,
    }


def test_verifier_stats_precision_and_recall_against_the_human(monkeypatch):
    monkeypatch.setenv("JEV_GEN_MIN_CONF", "0.8")
    monkeypatch.setenv("JEV_GEN_MIN_NOUL", "0.8")
    monkeypatch.setenv("JEV_GEN_MAX_DUP", "0.3")
    recs = [
        _rec("accept"),
        _rec("accept", conf=0.7),
        _rec("reject"),
        _rec("reject", nouls=(0.5, 0.9, 0.9)),
    ]
    seeds = [{"id": "L01", "verifiers": None, "human": None}]
    st = d.verifier_stats(recs + seeds)
    # category says yes on rec1, rec3, rec4 (conf 0.9); accepted: rec1, rec2 → tp 1, fp 2, fn 1
    assert st["category"] == {"n": 4, "precision": 0.333, "recall": 0.5}
    # answerable says yes on 3: rec1 accept, rec2 accept, rec3 reject → tp 2, fp 1, fn 0
    assert st["answerable"] == {"n": 4, "precision": 0.667, "recall": 1.0}
    assert st["expected_in_outputs"] == {"n": 0, "precision": None, "recall": None}
    assert st["duplicate"]["precision"] == 0.5  # all four say "not a duplicate", two accepted


def test_calibration_buckets_count_accepted_per_bucket():
    recs = [
        _rec("accept", conf=0.95),
        _rec("reject", conf=0.92),
        _rec("accept", conf=0.65),
        _rec("reject", conf=0.85, nouls=(0.55, 0.9, 0.9)),
    ]
    cal = d.calibration(recs)
    assert cal["category_confidence"]["0.9-1.0"] == {"n": 2, "accepted": 1}
    assert cal["category_confidence"]["0.6-0.7"] == {"n": 1, "accepted": 1}
    assert cal["min_noul"]["0.5-0.6"] == {"n": 1, "accepted": 0}
    assert cal["min_noul"]["0.9-1.0"] == {"n": 3, "accepted": 2}


def test_label_economy_per_batch_and_unavailable_without_minutes():
    recs = [
        _rec("accept", batch="b1", mpc=2.0),
        _rec("reject", batch="b1", mpc=2.0),
        _rec("accept", batch="b2", mpc=1.0, outcome="auto_add"),
        {"id": "L01", "verifiers": None, "human": None, "outcome": None},
        {"id": "G09", "verifiers": None, "human": None, "outcome": "auto_add"},
    ]
    le = d.label_economy(recs)
    assert le["batches"]["b1"] == {
        "decided": 2,
        "accepted": 1,
        "minutes": 4.0,
        "minutes_per_accepted": 4.0,
    }
    assert le["minutes_per_accepted"] == 2.5 and le["auto_add_confirmed"] == {"n": 1, "accepted": 1}
    # an auto_add nobody reviewed (human: null) is neither decided nor confirmed: it is unreviewed
    assert le["auto_add_unreviewed"] == 1
    assert d.label_economy([_rec("accept", mpc=None)]) == "unavailable"
    assert d.label_economy([]) == "unavailable"


def test_report_is_json_serialisable_and_lists_errors():
    recs = d.seed_records(CASES[:3], TABLES, "s", existing=[])
    rep = d.report(CASES, recs)
    import json

    json.dumps(rep)
    assert rep["n_cases"]["by_split"] == {"visible": 4, "holdout": 1}
    assert rep["coverage"]["plausible"] == 16 and rep["label_economy"] == "unavailable"
    assert rep["errors"] == ["F01: case without provenance", "FB02: case without provenance"]
