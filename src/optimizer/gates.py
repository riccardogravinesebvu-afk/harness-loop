"""Regression gate: a hypothesis is kept only if the visible pass rate rises,
no category drops by more than one visible case, and holdout does not fall. Plus two checks that
reject a hypothesis before spending an eval: file too long, or a visible expected value written
literally into the prompt."""

import re

CAPS = {"system.md": 6000, "tools.yaml": 3000}
EPS = 1e-9


def tolerances(cases: list[dict]) -> dict[str, float]:
    """One visible case per category, as a pass-rate fraction, computed from the dataset."""
    n: dict[str, int] = {}
    for c in cases:
        if c.get("split", "visible") == "visible":
            n[c["category"]] = n.get(c["category"], 0) + 1
    return {cat: 1 / k for cat, k in n.items()}


def delta_cases(before: dict[str, float], after: dict[str, float], tol: dict[str, float]) -> dict:
    return {c: round((after.get(c, 0.0) - before.get(c, 0.0)) / tol[c]) for c in tol}


def gate(before: dict, after: dict, tol: dict[str, float]) -> str | None:
    """before/after: {visible, holdout, per_category_visible}. Rejection reason or None."""
    if after["visible"] <= before["visible"] + EPS:
        return "no_gain"
    for cat, t in tol.items():
        b, a = (
            before["per_category_visible"].get(cat, 0.0),
            after["per_category_visible"].get(cat, 0.0),
        )
        if a + t + EPS < b:
            return f"gate {cat} {round((a - b) / t):+d}"
    if after["holdout"] + EPS < before["holdout"]:
        return "holdout"
    return None


def _literals(case: dict) -> list[str]:
    out: list[str] = []
    exp = case.get("expected")
    vals = exp if isinstance(exp, list) else [exp]
    for v in vals:
        if isinstance(v, bool) or v is None:
            continue
        if isinstance(v, int | float):
            digits = str(int(v)) if float(v).is_integer() else str(v)
            if len(digits) >= 4:
                out.append(digits)
        elif isinstance(v, str) and len(v) >= 4 and v != "judge":
            out.append(v)
    for m in re.findall(r"\d[\d,.]*\d", case.get("reference") or ""):
        # amounts and counts are answers; a year inside a date (2026-09-01) is not
        if len(m) >= 4 and not (len(m) == 4 and m.isdigit() and 1900 <= int(m) <= 2100):
            out.append(m)
    return out


def leaks_expected(content: str, cases: list[dict], failing: set[str] | None = None) -> str | None:
    """Returns the leaked literal if the content spells out an expected answer of a visible case.

    `failing` restricts the check to those case ids: the optimizer only sees `expected` for
    failing cases, and a passing case's accepted answer can be a general-knowledge word
    (loop 2026-09-17: "Switzerland" in an ISO-code list blocked three hypotheses in a row)."""
    flat = re.sub(r"[,.\s]", "", content.lower())
    for c in cases:
        if c.get("split", "visible") != "visible":
            continue
        if failing is not None and c["id"] not in failing:
            continue
        for lit in _literals(c):
            if re.sub(r"[,.\s]", "", lit.lower()) in flat:
                return f"{c['id']}:{lit}"
    return None


async def leaks_semantic(diff: str, failing: list[dict], ask, threshold: float) -> str | None:
    """one Noul per failing visible case, on the prompt diff, in one fan-out.
    `ask(state, questions)` is src.jev.ask or a test double. Returns "<id>:<p>" or None.
    Replaces the literal check as the veto when Jev is configured."""
    from src import jev

    if not diff.strip() or not failing:
        return None
    qs = {f"leak_{c['id']}": jev.question("leak", question=c["input"]) for c in failing}
    res = await ask({"prompt_diff": diff}, qs)
    cid, p = max(
        ((c["id"], res.answers[f"leak_{c['id']}"].noul) for c in failing), key=lambda x: x[1]
    )
    return f"{cid}:{p:.2f}" if p >= threshold else None
