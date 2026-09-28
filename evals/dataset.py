"""The dataset module: provenance records, coverage with empty cells,
active selection without a model, the label-economy report. Pure functions over YAML and results
documents. The optimizer never reads provenance.yaml, candidates.yaml or selection.yaml.

Usage: uv run python -m evals.dataset init | select | report
"""

import argparse
import json
import os
import re
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import yaml

from evals.run import HERE, RESULTS, dataset_sha, git_sha, load_cases
from src.llm import load_env

PROVENANCE = HERE / "provenance.yaml"
SEED_TABLES = HERE / "seed_tables.yaml"
SELECTION = HERE / "selection.yaml"
PROPOSED = HERE / "proposed.yaml"
LEDGER_TABLES = ("customers", "invoices", "payments")
SOURCES = ("seed", "feedback", "proposed", "generated", "hand")
HEAD = (
    "# Provenance: one record per reviewed candidate. Records with a\n"
    "# dataset_sha are in dataset.yaml; rejected ones stay here as evidence for the verifiers'\n"
    "# precision. Written by `evals/dataset.py init` and `evals/generate.py --apply`. Verdicts\n"
    "# are evidence: do not edit them. The optimizer never reads this file.\n"
)


def tables_of(sql: str | None) -> list[str]:
    """The ledger tables a SELECT touches: the names after FROM and JOIN. No SQL parser."""
    found = set(re.findall(r"\b(?:from|join)\s+([a-z_]+)", (sql or "").lower()))
    return [t for t in LEDGER_TABLES if t in found]


def load_provenance(path: Path = PROVENANCE) -> list[dict]:
    return yaml.safe_load(path.read_text())["provenance"] if path.exists() else []


def save_provenance(records: list[dict], path: Path = PROVENANCE) -> None:
    path.write_text(
        HEAD
        + yaml.safe_dump({"provenance": records}, sort_keys=False, allow_unicode=True, width=100)
    )


def seed_records(
    cases: list[dict], tables: dict[str, list[str]], sha: str, existing: list[dict]
) -> list[dict]:
    """One record for every dataset case that has none: seeds derived by hand, feedback cases."""
    have = {r["id"] for r in existing}
    out = []
    for c in cases:
        if c["id"] in have:
            continue
        fb = c.get("source") == "feedback"
        if not fb and c["id"] not in tables:
            raise KeyError(f"{c['id']}: tables not annotated in seed_tables.yaml")
        out.append(
            {
                "id": c["id"],
                "source": c.get("source", "seed"),
                "category": c["category"],
                "check": c["check"],
                "input": c["input"],
                "trace": {"feedback": c["feedback"]["id"]} if fb else None,
                "derivation": {
                    "kind": "feedback" if fb else "hand",
                    "tables": tables.get(c["id"], []),
                },
                "verifiers": None,
                "outcome": None,
                "human": {"decision": "accept", "at": None, "batch": None, "minutes_per_case": None}
                if fb
                else None,
                "dataset_sha": sha,
            }
        )
    return out


def consistency(cases: list[dict], records: list[dict]) -> list[str]:
    """Errors never disappear: a case without a record, or a record that claims a case that is
    not in the dataset, is listed in the report."""
    ids = {c["id"] for c in cases}
    in_ds = {r["id"] for r in records if r.get("dataset_sha")}
    return [f"{i}: case without provenance" for i in sorted(ids - in_ds)] + [
        f"{i}: provenance says in dataset, case missing" for i in sorted(in_ds - ids)
    ]


@dataclass
class Coverage:
    cells: dict[tuple[str, str, str], int]
    plausible: list[tuple[str, str, str]]
    empty: list[tuple[str, str, str]]

    def to_dict(self) -> dict:
        return {
            "cells": [
                {"category": c, "tables": t, "check": k, "n": n}
                for (c, t, k), n in sorted(self.cells.items())
            ],
            "plausible": len(self.plausible),
            "empty": [cell_target(c) for c in self.empty],
        }


def cell_target(cell: tuple[str, str, str]) -> str:
    """`category:check:tables`, the format of `evals/generate.py --cell`."""
    category, tables, check = cell
    return f"{category}:{check}:{tables}"


def coverage(cases: list[dict], records: list[dict]) -> Coverage:
    """Cases per cell of category × tables × check. Plausible cells: the (category, check) pairs
    seen at least once, times the table combinations seen plus the single tables. `unknown` and
    `none` never count as plausible combinations."""
    tables = {r["id"]: r["derivation"].get("tables") for r in records if r.get("id")}
    cells: Counter = Counter()
    for c in cases:
        t = tables.get(c["id"])
        key = "unknown" if t is None else "+".join(t) or "none"
        cells[(c["category"], key, c["check"])] += 1
    pairs = {(cat, chk) for cat, _, chk in cells}
    combos = {t for _, t, _ in cells if t not in ("unknown", "none")} | set(LEDGER_TABLES)
    plausible = sorted((cat, t, chk) for cat, chk in pairs for t in combos)
    return Coverage(dict(cells), plausible, [k for k in plausible if cells[k] == 0])


def load_results(results_dir: Path = RESULTS, n: int = 20) -> list[dict]:
    """The n most recent eval results (not the replays, not the dataset reports), newest first."""
    out = []
    for f in sorted(results_dir.glob("20*.json"), reverse=True)[:n]:
        doc = json.loads(f.read_text())
        doc["_file"] = f.name
        out.append(doc)
    return out


def latest_pair(results: list[dict]) -> tuple[dict, dict] | None:
    """The two most recent results that measured the same dataset: base and hypothesis."""
    for i, a in enumerate(results):
        for b in results[i + 1 :]:
            if a.get("dataset_sha") and a.get("dataset_sha") == b.get("dataset_sha"):
                return a, b
    return None


def disagreements(a: dict, b: dict) -> list[dict]:
    """Cases the two runs decided differently (behavioural distance, computed in code)."""
    other = {r["id"]: r for r in b["cases"]}
    out = []
    for r in a["cases"]:
        o = other.get(r["id"])
        if o and (r["passed"] != o["passed"] or r["refused"] != o["refused"]):
            out.append(
                {
                    "reason": "disagreement",
                    "id": r["id"],
                    "files": [a["_file"], b["_file"]],
                    "detail": f"passed {r['passed']}/{o['passed']}, "
                    f"refused {r['refused']}/{o['refused']}",
                }
            )
    return out


def low_confidence(results: list[dict], min_conf: float) -> list[dict]:
    """Cases whose refusal Choice sat under the floor in the most recent run that has it."""
    seen, out = set(), []
    for doc in results:
        for r in doc["cases"]:
            conf = r.get("refused_confidence")
            if conf is not None and conf < min_conf and r["id"] not in seen:
                seen.add(r["id"])
                out.append(
                    {
                        "reason": "low_confidence",
                        "id": r["id"],
                        "file": doc["_file"],
                        "detail": f"refused_confidence {conf:.2f} by {r.get('refused_by')}",
                    }
                )
    return out


def select(
    cases: list[dict],
    records: list[dict],
    results: list[dict],
    proposed: list[dict],
    min_conf: float,
) -> dict:
    """The labelling queue, in priority order: empty cells, disagreements (two runs, then the
    optimizer's proposals not yet in the dataset), low refusal confidence. No model is called."""
    rows = [
        {"reason": "empty_cell", "target": cell_target(c), "detail": "no case in this cell"}
        for c in coverage(cases, records).empty
    ]
    unavailable = []
    pair = latest_pair(results)
    if pair:
        rows += disagreements(*pair)
    else:
        unavailable.append("disagreement: no two results files with the same dataset_sha")
    inputs = {c["input"].strip().lower() for c in cases}
    rows += [
        {"reason": "disagreement", "proposed": p, "detail": f"proposed by {p['hypothesis']}"}
        for p in proposed
        if p["input"].strip().lower() not in inputs
    ]
    if any("refused_confidence" in r for doc in results for r in doc["cases"]):
        rows += low_confidence(results, min_conf)
    else:
        unavailable.append("low_confidence: no results with refused_confidence")
    return {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "dataset_sha": dataset_sha(),
        "unavailable": unavailable,
        "rows": rows,
    }


VERIFIER_KEYS = (
    "category",
    "answerable",
    "one_reading",
    "sql_matches",
    "expected_in_outputs",
    "duplicate",
)
BUCKETS = ((0.0, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01))


def _env(name: str, default: str) -> float:
    return float(os.environ.get(name, default))


def _reviewed(records: list[dict]) -> list[dict]:
    return [r for r in records if r.get("verifiers") and r.get("human")]


def verifier_says_yes(r: dict, key: str) -> bool | None:
    """Did this verifier clear its threshold? None when the question was not asked."""
    v = r["verifiers"]
    if key == "category":
        c = v.get("category")
        if not c:
            return None
        return c["value"] == r["category"] and c["confidence"] >= _env("JEV_GEN_MIN_CONF", "0.8")
    if key == "duplicate":
        d = v.get("duplicate")
        return None if not d else d["p"] <= _env("JEV_GEN_MAX_DUP", "0.3")
    p = v.get(key)
    return None if p is None else p >= _env("JEV_GEN_MIN_NOUL", "0.8")


def verifier_stats(records: list[dict]) -> dict:
    """Precision and recall of every verifier against the human decision (accept = positive)."""
    out = {}
    for key in VERIFIER_KEYS:
        tp = fp = fn = n = 0
        for r in _reviewed(records):
            yes = verifier_says_yes(r, key)
            if yes is None:
                continue
            acc = r["human"]["decision"] == "accept"
            n += 1
            tp += yes and acc
            fp += yes and not acc
            fn += (not yes) and acc
        out[key] = {
            "n": n,
            "precision": round(tp / (tp + fp), 3) if tp + fp else None,
            "recall": round(tp / (tp + fn), 3) if tp + fn else None,
        }
    return out


def calibration(records: list[dict]) -> dict:
    """Per confidence bucket, how many reviewed candidates the human accepted."""
    reviewed = _reviewed(records)

    def min_noul(r):
        ps = [
            r["verifiers"][k]
            for k in ("answerable", "one_reading", "sql_matches", "expected_in_outputs")
            if r["verifiers"].get(k) is not None
        ]
        return min(ps) if ps else None

    def bucketize(value_of):
        rows = {}
        for lo, hi in BUCKETS:
            sel = [r for r in reviewed if value_of(r) is not None and lo <= value_of(r) < hi]
            rows[f"{lo:.1f}-{min(hi, 1.0):.1f}"] = {
                "n": len(sel),
                "accepted": sum(r["human"]["decision"] == "accept" for r in sel),
            }
        return rows

    return {
        "category_confidence": bucketize(lambda r: r["verifiers"]["category"]["confidence"]),
        "min_noul": bucketize(min_noul),
    }


def label_economy(records: list[dict]) -> dict | str:
    """Human minutes per accepted case, per batch and overall. Minutes are written by the human
    per batch (candidates.yaml `review.minutes`); without them the report says unavailable."""
    decided = [r for r in records if r.get("human") and r["human"].get("batch")]
    if not decided or any(r["human"].get("minutes_per_case") is None for r in decided):
        return "unavailable"
    batches: dict[str, dict] = {}
    for r in decided:
        b = batches.setdefault(r["human"]["batch"], {"decided": 0, "accepted": 0, "minutes": 0.0})
        b["decided"] += 1
        b["accepted"] += r["human"]["decision"] == "accept"
        b["minutes"] += r["human"]["minutes_per_case"]
    for b in batches.values():
        b["minutes_per_accepted"] = (
            round(b["minutes"] / b["accepted"], 2) if b["accepted"] else None
        )
    minutes = sum(b["minutes"] for b in batches.values())
    accepted = sum(b["accepted"] for b in batches.values())
    auto = [r for r in records if r.get("outcome") == "auto_add" and r.get("human")]
    # human: null means nobody reviewed this auto_add: it is unreviewed, not confirmed.
    auto_unreviewed = [r for r in records if r.get("outcome") == "auto_add" and not r.get("human")]
    return {
        "batches": batches,
        "minutes_per_accepted": round(minutes / accepted, 2) if accepted else None,
        "auto_add_confirmed": {
            "n": len(auto),
            "accepted": sum(r["human"]["decision"] == "accept" for r in auto),
        },
        "auto_add_unreviewed": len(auto_unreviewed),
    }


def report(cases: list[dict], records: list[dict]) -> dict:
    return {
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_sha": git_sha(),
        "dataset_sha": dataset_sha(),
        "n_cases": {
            "by_split": dict(Counter(c["split"] for c in cases)),
            "by_source": dict(Counter(c.get("source", "seed") for c in cases)),
        },
        "coverage": coverage(cases, records).to_dict(),
        "verifiers": verifier_stats(records),
        "calibration": calibration(records),
        "label_economy": label_economy(records),
        "errors": consistency(cases, records),
    }


def main() -> None:
    load_env()
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=("init", "select", "report"))
    a = p.parse_args()
    cases, records = load_cases(), load_provenance()
    if a.cmd == "init":
        tables = yaml.safe_load(SEED_TABLES.read_text())
        new = seed_records(cases, tables, dataset_sha(), records)
        save_provenance(records + new)
        print(f"{len(new)} records added → {PROVENANCE}")
    elif a.cmd == "select":
        proposed = yaml.safe_load(PROPOSED.read_text())["cases"] if PROPOSED.exists() else []
        sel = select(cases, records, load_results(), proposed, _env("JEV_REFUSED_MIN_CONF", "0.6"))
        SELECTION.write_text(
            "# Labelling queue from `just select`.\n"
            "# empty_cell → `just generate --cell <target>`; proposed → `just generate --stage`;\n"
            "# the rest by hand.\n"
            + yaml.safe_dump(sel, sort_keys=False, allow_unicode=True, width=100)
        )
        by = Counter(r["reason"] for r in sel["rows"])
        print(f"{dict(by)}  unavailable: {sel['unavailable']}\n→ {SELECTION}")
    else:
        rep = report(cases, records)
        path = RESULTS / f"dataset-{rep['timestamp'].replace(':', '')}.json"
        path.write_text(json.dumps(rep, indent=2, ensure_ascii=False))
        cov = rep["coverage"]
        print(f"cases {rep['n_cases']}  plausible cells {cov['plausible']}")
        print(f"empty {len(cov['empty'])}: {cov['empty']}")
        print(f"label economy {rep['label_economy']}  errors {rep['errors']}\n→ {path}")


if __name__ == "__main__":
    main()
