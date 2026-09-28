"""Deterministic checks. Each returns (passed, detail). `judge` lives in evals/judge.py."""

import re

_NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def numbers_in(text: str) -> list[float]:
    out = []
    for m in _NUM.findall(text):
        try:
            out.append(float(m.replace(",", "")))
        except ValueError:
            continue
    return out


def check(case: dict, answer: str, refused: bool) -> tuple[bool, str]:
    kind, exp = case["check"], case["expected"]
    if kind == "exact":
        return answer.strip() == str(exp).strip(), f"exact {exp!r}"
    if kind == "contains":
        needles = exp if isinstance(exp, list) else [exp]
        missing = [n for n in needles if str(n).lower() not in answer.lower()]
        return not missing, f"missing {missing}" if missing else "all substrings found"
    if kind == "contains_any":
        hit = [n for n in exp if str(n).lower() in answer.lower()]
        return bool(hit), f"found {hit}" if hit else f"none of {exp}"
    if kind == "number":
        target = float(exp)
        tol = max(0.01, abs(target) * 0.005)
        found = numbers_in(answer)
        ok = any(abs(n - target) <= tol for n in found)
        return ok, f"expected {target} in {found[:8]}"
    if kind == "refused":
        return refused is bool(exp), f"refused={refused}"
    raise ValueError(f"unknown check {kind!r}")
