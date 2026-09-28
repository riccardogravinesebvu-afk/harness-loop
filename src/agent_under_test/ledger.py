"""A small invented SME ledger on SQLite. Data is hand-written, not random: every eval
expectation in evals/dataset.yaml is derivable from these rows. Reference date AS_OF."""

import sqlite3
from pathlib import Path

AS_OF = "2026-09-01"
DB_PATH = Path(__file__).resolve().parents[2] / "data" / "ledger.sqlite"

SCHEMA = """
CREATE TABLE customers (
  id INTEGER PRIMARY KEY, name TEXT NOT NULL, country TEXT NOT NULL, segment TEXT NOT NULL);
CREATE TABLE invoices (
  id TEXT PRIMARY KEY, customer_id INTEGER NOT NULL REFERENCES customers(id),
  issued_at TEXT NOT NULL, due_at TEXT NOT NULL, amount_eur REAL NOT NULL);
CREATE TABLE payments (
  id INTEGER PRIMARY KEY, invoice_id TEXT NOT NULL REFERENCES invoices(id),
  paid_at TEXT NOT NULL, amount_eur REAL NOT NULL);
"""

CUSTOMERS = [
    (1, "Alpine Foods GmbH", "DE", "food"),
    (2, "Bruno Serramenti Srl", "IT", "construction"),
    (3, "Casa Verde SL", "ES", "retail"),
    (4, "Delta Logistics BV", "NL", "logistics"),
    (5, "Ember Studio AB", "SE", "services"),
    (6, "Fortuna Trading SA", "CH", "trading"),
    (7, "Gallo Ristorazione Srl", "IT", "food"),
    (8, "Helios Energy Ltd", "IE", "energy"),
]

INVOICES = [
    ("INV-001", 1, "2026-05-02", "2026-06-01", 1200.00),
    ("INV-002", 1, "2026-06-10", "2026-07-10", 3400.00),
    ("INV-003", 1, "2026-08-15", "2026-09-14", 2100.00),
    ("INV-004", 2, "2026-04-20", "2026-05-20", 8750.00),
    ("INV-005", 2, "2026-07-01", "2026-07-31", 1500.00),
    ("INV-006", 3, "2026-05-15", "2026-06-14", 960.00),
    ("INV-007", 3, "2026-07-20", "2026-08-19", 2300.00),
    ("INV-008", 3, "2026-08-25", "2026-09-24", 640.00),
    ("INV-009", 4, "2026-03-01", "2026-03-31", 12000.00),
    ("INV-010", 4, "2026-06-01", "2026-07-01", 12000.00),
    ("INV-011", 4, "2026-08-01", "2026-08-31", 12000.00),
    ("INV-012", 5, "2026-06-05", "2026-07-05", 450.00),
    ("INV-013", 5, "2026-08-05", "2026-09-04", 450.00),
    ("INV-014", 6, "2026-02-10", "2026-03-12", 25000.00),
    ("INV-015", 6, "2026-05-10", "2026-06-09", 18500.00),
    ("INV-016", 6, "2026-08-10", "2026-09-09", 22000.00),
    ("INV-017", 7, "2026-04-01", "2026-05-01", 780.00),
    ("INV-018", 7, "2026-05-01", "2026-05-31", 780.00),
    ("INV-019", 7, "2026-06-01", "2026-07-01", 780.00),
    ("INV-020", 7, "2026-07-01", "2026-07-31", 780.00),
    ("INV-021", 8, "2026-07-15", "2026-08-14", 5600.00),
    ("INV-022", 8, "2026-08-20", "2026-09-19", 4100.00),
    ("INV-023", 2, "2026-08-28", "2026-09-27", 2200.00),
    ("INV-024", 5, "2026-03-03", "2026-04-02", 1250.00),
]

PAYMENTS = [
    (1, "INV-001", "2026-05-28", 1200.00),
    (2, "INV-002", "2026-07-08", 3400.00),
    (3, "INV-004", "2026-06-01", 5000.00),
    (4, "INV-006", "2026-06-30", 960.00),
    (5, "INV-009", "2026-03-25", 12000.00),
    (6, "INV-010", "2026-07-15", 12000.00),
    (7, "INV-012", "2026-07-01", 450.00),
    (8, "INV-014", "2026-03-10", 25000.00),
    (9, "INV-015", "2026-06-05", 10000.00),
    (10, "INV-015", "2026-06-20", 8500.00),
    (11, "INV-020", "2026-07-30", 780.00),
    (12, "INV-021", "2026-08-10", 2600.00),
    (13, "INV-024", "2026-04-01", 1250.00),
]


def build_db(path: Path = DB_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    with sqlite3.connect(path) as con:
        con.executescript(SCHEMA)
        con.executemany("INSERT INTO customers VALUES (?,?,?,?)", CUSTOMERS)
        con.executemany("INSERT INTO invoices VALUES (?,?,?,?,?)", INVOICES)
        con.executemany("INSERT INTO payments VALUES (?,?,?,?)", PAYMENTS)
    return path


def connect(path: Path = DB_PATH) -> sqlite3.Connection:
    if not path.exists():
        build_db(path)
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    return con
