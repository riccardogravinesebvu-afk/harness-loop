"""Three tools over the ledger plus final_answer. Descriptions are loaded from prompts/tools.yaml
so the optimizer can edit them without touching this file. Tools never raise: errors are strings."""

import ast
import json
import operator
import sqlite3
from pathlib import Path

import yaml
from langchain_core.tools import BaseTool, tool

from src.agent_under_test import ledger

PROMPTS = Path(__file__).parent / "prompts"
MAX_ROWS = 50


@tool
def run_sql(sql: str) -> str:
    """Placeholder, replaced from tools.yaml."""
    head = sql.lstrip().lower()
    if not (head.startswith("select") or head.startswith("with")):
        return "ERROR: only SELECT queries are allowed; the ledger is read-only."
    try:
        with ledger.connect() as con:
            cur = con.execute(sql)
            cols = [d[0] for d in cur.description or []]
            rows = cur.fetchmany(MAX_ROWS + 1)
    except sqlite3.Error as e:
        return f"ERROR: {e}"
    truncated = len(rows) > MAX_ROWS
    out = [dict(zip(cols, r, strict=True)) for r in rows[:MAX_ROWS]]
    return json.dumps(out) + (f"\n(truncated to {MAX_ROWS} rows)" if truncated else "")


@tool
def lookup_customer(name: str) -> str:
    """Placeholder, replaced from tools.yaml."""
    q = """
    SELECT c.id, c.name, c.country, c.segment,
      COUNT(i.id) AS invoices,
      COALESCE(SUM(i.amount_eur), 0) AS invoiced_eur,
      COALESCE((SELECT SUM(p.amount_eur) FROM payments p JOIN invoices i2 ON i2.id = p.invoice_id
                WHERE i2.customer_id = c.id), 0) AS paid_eur
    FROM customers c LEFT JOIN invoices i ON i.customer_id = c.id
    WHERE lower(c.name) LIKE lower(?) GROUP BY c.id"""
    with ledger.connect() as con:
        rows = [dict(r) for r in con.execute(q, (f"%{name}%",))]
    if not rows:
        return f"No customer matching '{name}'."
    for r in rows:
        r["outstanding_eur"] = round(r["invoiced_eur"] - r["paid_eur"], 2)
    return json.dumps(rows)


_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Mod: operator.mod, ast.Pow: operator.pow,
    ast.USub: operator.neg, ast.UAdd: operator.pos,
}  # fmt: skip
_FUNCS = {"round": round, "abs": abs, "min": min, "max": max, "sum": sum}


def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS:
        return _FUNCS[node.func.id](*[_eval(a) for a in node.args])
    if isinstance(node, ast.List | ast.Tuple):
        return [_eval(e) for e in node.elts]  # type: ignore[return-value]
    raise ValueError(f"unsupported expression: {ast.dump(node)[:60]}")


@tool
def compute(expression: str) -> str:
    """Placeholder, replaced from tools.yaml."""
    try:
        return str(_eval(ast.parse(expression, mode="eval").body))
    except (ValueError, SyntaxError, ZeroDivisionError, TypeError) as e:
        return f"ERROR: {e}"


@tool
def final_answer(answer: str, refused: bool = False) -> str:
    """Placeholder, replaced from tools.yaml."""
    return answer  # never executed: the graph routes final_answer to END


def load_tools() -> list[BaseTool]:
    descriptions = yaml.safe_load((PROMPTS / "tools.yaml").read_text())
    tools = [run_sql, lookup_customer, compute, final_answer]
    for t in tools:
        t.description = descriptions[t.name].strip()
    return tools
