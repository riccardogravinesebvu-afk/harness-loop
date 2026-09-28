"""Guards the numbers the dataset relies on, and the deterministic checks."""

import sqlite3

import pytest

from evals.checks import check, numbers_in
from src.agent_under_test import ledger
from src.agent_under_test.tools import compute, run_sql


@pytest.fixture(scope="module")
def con(tmp_path_factory):
    path = ledger.build_db(tmp_path_factory.mktemp("db") / "ledger.sqlite")
    return sqlite3.connect(path)


def q(con, sql):
    return con.execute(sql).fetchone()[0]


def test_ledger_totals_match_dataset(con):
    assert q(con, "SELECT COUNT(*) FROM customers") == 8
    assert q(con, "SELECT COUNT(*) FROM invoices") == 24
    assert q(con, "SELECT SUM(amount_eur) FROM invoices") == 139520
    assert q(con, "SELECT SUM(amount_eur) FROM payments") == 83140
    overdue = q(
        con,
        """
        SELECT SUM(i.amount_eur - COALESCE(p.paid, 0)) FROM invoices i
        LEFT JOIN (SELECT invoice_id, SUM(amount_eur) paid FROM payments GROUP BY invoice_id) p
          ON p.invoice_id = i.id
        WHERE i.due_at < '2026-09-01' AND i.amount_eur - COALESCE(p.paid, 0) > 0""",
    )
    assert overdue == 24890
    july = q(
        con,
        "SELECT SUM(amount_eur) FROM payments WHERE paid_at BETWEEN '2026-07-01' AND '2026-07-31'",
    )
    assert july == 16630


def test_run_sql_is_read_only():
    assert run_sql.invoke({"sql": "DELETE FROM invoices"}).startswith("ERROR")
    assert run_sql.invoke({"sql": "DROP TABLE invoices"}).startswith("ERROR")
    assert "customers" in run_sql.invoke({"sql": "SELECT name FROM sqlite_master"})


def test_compute_is_sandboxed():
    assert compute.invoke({"expression": "round(139520 / 24, 2)"}) == "5813.33"
    assert compute.invoke({"expression": "__import__('os')"}).startswith("ERROR")


def test_number_check_tolerates_formatting():
    case = {"check": "number", "expected": 24890}
    assert check(case, "The overdue total is €24,890.00.", False)[0]
    assert check(case, "About 24890", False)[0]
    assert not check(case, "It is 108,030", False)[0]
    assert numbers_in("1,200 and 3.5") == [1200.0, 3.5]


def test_other_checks():
    both = {"check": "contains", "expected": ["Bruno", "Gallo"]}
    assert check(both, "Bruno Srl; Gallo Srl", False)[0]
    assert not check(both, "Only Bruno", False)[0]
    assert (
        check({"check": "contains_any", "expected": ["CH", "Switzerland"]}, "Swiss", False)[0]
        is False
    )
    assert check({"check": "contains_any", "expected": ["CH", "Switzerland"]}, "in CH", False)[0]
    assert check({"check": "refused", "expected": True}, "I cannot do that", True)[0]
    assert not check({"check": "refused", "expected": True}, "Done", False)[0]
