"""Generated eval cases, verified before a human sees them.

propose: an LLM writes candidates per category with the SQL that computes the expected value.
compute_expected: the code runs that SQL on the ledger — Jev does no arithmetic.
verify: one Jev fan-out per candidate (category, answerable, one reading, SQL matches, expected
in outputs, one duplicate Noul per existing case). outcome: pure sort into auto_add | review |
discard by .env thresholds. write_candidates: evals/candidates.yaml for human review. main/--apply:
generate a batch, or fold reviewed decisions into dataset.yaml and write the provenance record to
evals/provenance.yaml. --stage: proposed rows of evals/selection.yaml enter review.

Usage: just generate --all-review | --apply | --stage | --cell category:check:tables |
       --per-category 10
"""

import argparse
import asyncio
import json
import sqlite3
import sys
from datetime import UTC, datetime
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from evals.dataset import PROVENANCE, SELECTION, _env, load_provenance, save_provenance, tables_of
from evals.run import HERE, dataset_sha, load_cases
from src import jev
from src.agent_under_test import ledger
from src.llm import load_env, make_chat, model_for
from src.pricing import cost_usd

CANDIDATES = HERE / "candidates.yaml"
DATASET = HERE / "dataset.yaml"
CATS = ("lookup", "aggregation", "reasoning", "refusal")
CHECKS = {"lookup": "number|contains|contains_any", "aggregation": "number",
          "reasoning": "judge", "refusal": "refused"}  # fmt: skip
DATASET_CHECKS = {k for v in CHECKS.values() for k in v.split("|")}

PROMPT = """You write evaluation cases for a data agent that answers questions over a small
accounting ledger (SQLite). Reference date: {as_of}. Schema:
{schema}

Write {n} new cases of category `{category}` (allowed checks: {checks}). Each case must be
answerable exactly from the ledger (or, for `refusal`, clearly NOT answerable: a prediction, an
opinion, outside data). Do not repeat or paraphrase the existing cases below. Vary entities,
measures and filters; prefer questions a bookkeeper would ask.

For `number`, `contains` and `contains_any` cases write the SQL that returns the expected value:
one numeric value for number, one text value for contains, a list of text values for
contains_any. For `judge` cases write a `reference` answer with the figures and how they follow
from the data. For `refused` cases leave sql and reference empty.

Existing cases of this category:
{existing}
"""


class Candidate(BaseModel):
    category: str
    input: str
    check: str = Field(description="number | contains | contains_any | judge | refused")
    sql: str | None = Field(default=None, description="SELECT computing the expected value")
    reference: str | None = Field(default=None, description="judge cases only")
    why: str = Field(default="", description="what this case covers that the existing ones do not")
    source: str = Field(default="generated", description="generated | proposed | hand")
    tables: list[str] = Field(default_factory=list, description="ledger tables the SQL touches")


class Proposal(BaseModel):
    candidates: list[Candidate]


async def propose(
    category: str, n: int, existing: list[dict], llm=None, cell: tuple[str, str] | None = None
) -> tuple[list[Candidate], float]:
    llm = llm or make_chat("optimizer", max_tokens=6000).with_structured_output(
        Proposal, method="function_calling", include_raw=True
    )
    prompt = PROMPT.format(
        as_of=ledger.AS_OF,
        schema=ledger.SCHEMA.strip(),
        n=n,
        category=category,
        checks=CHECKS[category],
        existing="\n".join(f"- {c['id']}: {c['input']}" for c in existing) or "none",
    )
    if cell:
        prompt += (
            f"\nEvery case must use check `{cell[0]}` and its SQL must read exactly these tables: "
            f"{cell[1].replace('+', ', ')}. This cell of the coverage table is empty."
        )
    msgs = [
        ("system", "You are precise, you only write SELECT queries, you never invent data."),
        ("human", prompt),
    ]
    res = await llm.ainvoke(msgs)
    if res["parsed"] is None:
        raise ValueError(f"propose: structured output parsing failed: {res.get('parsing_error')}")
    u = res["raw"].usage_metadata or {}
    usd, _ = cost_usd(model_for("optimizer"), u.get("input_tokens", 0), u.get("output_tokens", 0))
    return list(res["parsed"].candidates)[:n], usd


def compute_expected(c: Candidate, con: sqlite3.Connection) -> tuple[object | None, str | None]:
    """The expected value comes from the ledger, not from a model."""
    if c.check not in CHECKS.get(c.category, "").split("|"):
        return None, f"check {c.check!r} is not valid for category {c.category!r}"
    if c.check == "refused":
        return True, None
    if c.check == "judge":
        return ("judge", None) if c.reference else (None, "judge case needs a reference")
    if not c.sql or not c.sql.lstrip().lower().startswith(("select", "with")):
        return None, "only SELECT queries compute an expected value"
    try:
        rows = con.execute(c.sql).fetchall()
    except sqlite3.Error as e:
        return None, f"sql error: {e}"
    if c.check == "contains_any":
        vals = [r[0] for r in rows if r and r[0] is not None]
        return (
            (vals, None)
            if vals and all(isinstance(v, str) for v in vals)
            else (None, "contains_any needs a list of text values")
        )
    if len(rows) != 1 or len(rows[0]) != 1 or rows[0][0] is None:
        return (
            None,
            f"{c.check} check needs one {'numeric' if c.check == 'number' else 'text'} value",
        )
    v = rows[0][0]
    if c.check == "number":
        return (
            (float(v), None)
            if isinstance(v, int | float)
            else (None, "number check needs one numeric value")
        )
    return (str(v), None) if isinstance(v, str) else (None, "contains check needs one text value")


async def verify(c: Candidate, expected, existing: list[dict], client=None) -> tuple[dict, float]:
    """One fan-out: every check plus one duplicate Noul per existing case, same state."""
    state = {
        "candidate": {"input": c.input, "category": c.category, "check": c.check,
                      "sql": c.sql, "expected": expected if c.check != "judge" else c.reference},
        "schema": ledger.SCHEMA.strip(),
        "existing": {e["id"]: e["input"] for e in existing},
    }  # fmt: skip
    qs = {"category": jev.question("gen_category"), "answerable": jev.question("gen_answerable"),
          "one_reading": jev.question("gen_one_reading")}  # fmt: skip
    if c.sql:
        qs["sql_matches"] = jev.question("gen_sql_matches", sql=c.sql)
    if c.check == "contains" and expected is not None:
        qs["expected_in_outputs"] = jev.question("gen_expected_in_outputs", expected=expected)
    for e in existing:
        qs[f"dup_{e['id']}"] = jev.question("gen_duplicate", other_id=e["id"])
    res = await jev.ask(state, qs, client=client)
    a = res.answers
    dups = [(e["id"], a[f"dup_{e['id']}"].noul) for e in existing]
    worst = max(dups, key=lambda x: x[1]) if dups else ("", 0.0)
    return {
        "category": {"value": a["category"].value, "confidence": a["category"].confidence},
        "answerable": a["answerable"].noul,
        "one_reading": a["one_reading"].noul,
        "sql_matches": a["sql_matches"].noul if "sql_matches" in a else None,
        "expected_in_outputs": a["expected_in_outputs"].noul
        if "expected_in_outputs" in a
        else None,
        "duplicate": {"id": worst[0], "p": worst[1]},
    }, res.cost_usd


def outcome(c: Candidate, v: dict, sql_error: str | None, all_review: bool) -> tuple[str, str]:
    """auto_add | review | discard, with the reason. Thresholds from .env."""
    min_conf, min_noul, max_dup = (
        _env("JEV_GEN_MIN_CONF", "0.8"),
        _env("JEV_GEN_MIN_NOUL", "0.8"),
        _env("JEV_GEN_MAX_DUP", "0.3"),
    )
    if sql_error:
        return "discard", sql_error
    if v["duplicate"]["p"] >= 0.7:
        return "discard", f"duplicate of {v['duplicate']['id']} ({v['duplicate']['p']:.2f})"
    nouls = {
        k: v[k]
        for k in ("answerable", "one_reading", "sql_matches", "expected_in_outputs")
        if v[k] is not None
    }
    for k, p in nouls.items():
        if p <= 0.2:
            return "discard", f"{k} {p:.2f}"
    if v["category"]["value"] != c.category:
        return "review", f"category {v['category']['value']} != {c.category}"
    for k, p in nouls.items():
        if p < min_noul:
            return "review", f"{k} {p:.2f}"
    if v["category"]["confidence"] < min_conf:
        return "review", f"category confidence {v['category']['confidence']:.2f}"
    if v["duplicate"]["p"] > max_dup:
        return "review", f"near duplicate of {v['duplicate']['id']} ({v['duplicate']['p']:.2f})"
    if c.check not in ("number", "contains"):
        reason = (
            "judge and refused cases are never auto-added"
            if c.check in ("judge", "refused")
            else f"{c.check} cases are never auto-added"
        )
        return "review", reason
    if all_review:
        return "review", "all-review batch"
    return "auto_add", "all checks clear"


def write_candidates(
    records: list[dict],
    path: Path = CANDIDATES,
    batch: str | None = None,
    minutes: float | None = None,
) -> None:
    head = (
        "# Generated candidates awaiting review (evals/generate.py). Fill `decision` with\n"
        "# accept | reject and `review.minutes` with the minutes spent on this batch, then\n"
        "# `just generate --apply`. Jev verdicts are evidence: do not edit them. Only\n"
        "# `source: hand` records may carry `split: holdout`. The optimizer never reads this\n"
        "# file.\n"
    )
    batch = batch or datetime.now(UTC).strftime("%Y-%m-%d")
    doc = {"review": {"batch": batch, "minutes": minutes}, "candidates": records}
    path.write_text(head + yaml.safe_dump(doc, sort_keys=False, allow_unicode=True, width=100))


def read_candidates(path: Path = CANDIDATES) -> tuple[dict, list[dict]]:
    doc = yaml.safe_load(path.read_text()) or {}
    return doc.get("review") or {"batch": None, "minutes": None}, doc.get("candidates") or []


def next_id(cases: list[dict]) -> str:
    n = max(
        (int(c["id"][1:]) for c in cases if c["id"].startswith("G") and c["id"][1:].isdigit()),
        default=0,
    )
    return f"G{n + 1:02d}"


def dataset_line(rec: dict, cid: str, split: str) -> str:
    """One flow-style case line, appended as text so dataset.yaml keeps its comments and layout."""
    parts = [
        f"id: {cid}",
        f"category: {rec['category']}",
        f"split: {split}",
        f"check: {rec['check']}",
        f"expected: {json.dumps(rec['expected'], ensure_ascii=False)}",
        f"input: {json.dumps(rec['input'], ensure_ascii=False)}",
        "weight: 1",
        f"source: {rec.get('source', 'generated')}",
    ]
    if rec.get("reference"):
        parts.append(f"reference: {json.dumps(rec['reference'], ensure_ascii=False)}")
    return "  - {" + ", ".join(parts) + "}\n"


def provenance_record(
    r: dict,
    cid: str | None,
    human: str | None,
    batch: str | None,
    mpc: float | None,
    reviewer: str = "author",
) -> dict:
    """The provenance record; `dataset_sha` is filled after the append."""
    source = r.get("source", "generated")
    if r.get("sql"):
        derivation = {
            "kind": "sql",
            "sql": r["sql"],
            "tables": r.get("tables") or tables_of(r["sql"]),
        }
    elif r["check"] == "judge":
        derivation = {
            "kind": "reference" if source != "hand" else "hand",
            "tables": r.get("tables") or [],
        }
    else:
        derivation = {
            "kind": "none" if source != "hand" else "hand",
            "tables": r.get("tables") or [],
        }
    trace = r.get("trace") or (
        {"generator": model_for("optimizer"), "batch": batch} if source == "generated" else None
    )
    return {
        "id": cid,
        "source": source,
        "category": r["category"],
        "check": r["check"],
        "input": r["input"],
        "trace": trace,
        "derivation": derivation,
        "verifiers": r.get("jev"),
        "outcome": r.get("outcome", "review"),
        "human": (
            {
                "decision": human,
                "by": reviewer,  # "author" unless review.reviewer names someone (or a model)
                "at": datetime.now(UTC).isoformat(timespec="seconds"),
                "batch": batch,
                "minutes_per_case": mpc,
            }
            if human is not None
            else None  # an auto_add nobody reviewed: no decision, no timestamp, no minutes share
        ),
        "dataset_sha": None,
    }


async def apply(
    candidates_path: Path = CANDIDATES,
    dataset_path: Path = DATASET,
    provenance_path: Path = PROVENANCE,
    con=None,
    client=None,
) -> list[str]:
    """accept/reject decisions, undecided auto_adds and undecided discards leave candidates.yaml:
    accepted cases (and undecided auto_adds) go into the dataset, every decided record — plus
    undecided auto_adds and discards — goes into provenance.yaml with the human verdict next to
    Jev's (null for the undecided ones). A hand or proposed record gets its expected from its
    SQL here, never from a model. Only still-undecided `review` rows stay in candidates.yaml."""
    review, recs = read_candidates(candidates_path)
    cases = yaml.safe_load(dataset_path.read_text())["cases"] or []
    existing = [{"id": c["id"], "input": c["input"]} for c in cases]
    records = load_provenance(provenance_path)
    # minutes are shared only among records a human actually decided; an undecided auto_add
    # (decision: null) is not one of them, even though it still enters the dataset below.
    decided = [r for r in recs if r.get("decision")]
    minutes = review.get("minutes")
    mpc = round(minutes / len(decided), 2) if minutes and decided else None
    keep, added, lines, new = [], [], [], []
    for r in recs:
        decision = r.get("decision")
        auto = decision is None and r.get("outcome") == "auto_add"
        # an undecided discard is never going to be reviewed: it leaves candidates.yaml into
        # provenance with human: null instead of blocking every propose run after it forever.
        auto_discard = decision is None and r.get("outcome") == "discard"
        if decision is None and not auto and not auto_discard:
            keep.append(r)
            continue
        label = r["input"][:40]
        split = r.get("split")
        if split not in (None, "visible", "holdout"):
            raise ValueError(f"{label}: split must be visible or holdout, not {split}")
        if not auto and not auto_discard and decision not in ("accept", "reject"):
            raise ValueError(f"{label}: decision must be accept or reject, not {decision}")
        if split == "holdout" and r.get("source") != "hand":
            raise ValueError(f"{label}: only source: hand may enter the holdout")
        if split == "holdout" and r.get("jev") is not None and r.get("source") == "hand":
            # a generator/proposed trace with source edited to hand is not a hand record
            raise ValueError(f"{label}: a record with a Jev verdict cannot enter the holdout")
        cid = None
        if auto or decision == "accept":
            if r.get("expected") is None or r.get("source") in ("hand", "proposed"):
                # hand and proposed records never carry a trustworthy typed expected: the SQL,
                # run here, is the only source of truth (generated ones were already verified).
                c = Candidate(**{k: r.get(k) for k in Candidate.model_fields if k in r})
                r["expected"], err = compute_expected(c, con or ledger.connect())
                if err:
                    raise ValueError(f"{label}: {err}")
                r["tables"] = tables_of(r.get("sql"))
                if r.get("jev") is None and (client or jev.enabled()):
                    r["jev"], _ = await verify(c, r["expected"], existing, client=client)
            cid = next_id(cases + [{"id": a} for a in added])
            lines.append(dataset_line(r, cid, r.get("split") or "visible"))
            added.append(cid)
        human = None if (auto or auto_discard) else decision
        new.append(
            provenance_record(
                r,
                cid,
                human,
                review.get("batch"),
                None if (auto or auto_discard) else mpc,
                review.get("reviewer") or "author",
            )
        )
    if lines:
        with dataset_path.open("a") as f:
            f.write(
                "  # ---- reviewed candidates (evals/generate.py --apply): "
                "see evals/provenance.yaml ----\n"
            )
            f.writelines(lines)
    sha = dataset_sha(dataset_path)
    for rec in new:
        if rec["id"] in added:
            rec["dataset_sha"] = sha
    write_candidates(keep, candidates_path, batch=review.get("batch"))
    save_provenance(records + new, provenance_path)
    return added


def stage(selection_path: Path = SELECTION, candidates_path: Path = CANDIDATES) -> int:
    """Proposed cases from the selection enter candidates.yaml for review. The proposed expected
    is a model's guess and is never used: the human writes the SQL, --apply runs it."""
    sel = yaml.safe_load(selection_path.read_text())
    review, recs = read_candidates(candidates_path) if candidates_path.exists() else ({}, [])
    have = {r["input"] for r in recs}
    n = 0
    for row in sel["rows"]:
        p = row.get("proposed")
        if not p or p["input"] in have:
            continue
        recs.append(
            {
                "category": p["category"],
                "input": p["input"],
                "check": p["check"],
                "sql": None,
                "reference": p.get("expected") if p["check"] == "judge" else None,
                "why": p.get("why", ""),
                "source": "proposed",
                "tables": [],
                "expected": None,
                "sql_error": None,
                "jev": None,
                "outcome": "review",
                "reason": f"proposed by {p['hypothesis']}: write the sql, "
                "the expected comes from the ledger",
                "trace": {"hypothesis": p["hypothesis"]},
                "proposed_expected": p.get("expected"),
                "decision": None,
            }
        )
        have.add(p["input"])
        n += 1
    write_candidates(
        recs, candidates_path, batch=review.get("batch"), minutes=review.get("minutes")
    )
    return n


async def main_async(args: argparse.Namespace) -> Path:
    load_env()
    if args.apply:
        added = await apply()
        print(f"added {added} to {DATASET}; provenance in {PROVENANCE}")
        print("run `just evals` and commit the results file")
        return DATASET
    if args.stage:
        print(f"staged {stage()} proposed cases → {CANDIDATES}")
        return CANDIDATES
    if CANDIDATES.exists():
        _, pending = read_candidates(CANDIDATES)
        if pending:
            sys.exit(
                f"{CANDIDATES} has {len(pending)} records pending: run --apply or move the file"
            )
    if not jev.enabled():
        sys.exit("TYPESAFE_API_KEY missing: generation needs Jev (errors never disappear)")
    ledger.build_db()
    existing = load_cases()
    con = ledger.connect()
    records, opt_usd, jev_usd = [], 0.0, 0.0
    cell = tuple(args.cell.split(":")) if args.cell else None  # (category, check, tables)
    cats = [cell[0]] if cell else CATS
    for cat in cats:
        cands, usd = await propose(
            cat, args.per_category, [c for c in existing if c["category"] == cat],
            cell=(cell[1], cell[2]) if cell else None,
        )  # fmt: skip
        opt_usd += usd
        for c in cands:
            expected, err = compute_expected(c, con)
            c.tables = tables_of(c.sql)
            try:
                v, cost = await verify(
                    c, expected, [{"id": e["id"], "input": e["input"]} for e in existing]
                )
            except Exception as e:  # noqa: BLE001 — one candidate's Jev error is a row, not a lost batch
                v, out, why = None, "discard", f"jev_error: {str(e)[:80]}"
            else:
                jev_usd += cost
                out, why = outcome(c, v, err, args.all_review)
            if cell and out == "auto_add" and (c.check != cell[1] or "+".join(c.tables) != cell[2]):
                got = f"{c.check}:{'+'.join(c.tables)}"
                out, why = "review", f"cell: got {got}, asked {cell[1]}:{cell[2]}"
            records.append({**c.model_dump(), "expected": expected, "sql_error": err, "jev": v,
                            "outcome": out, "reason": why, "decision": None})  # fmt: skip
            print(f"{out:<8} {cat:<12} {why:<40} {c.input[:60]}", file=sys.stderr)
    write_candidates(records)
    counts = {
        k: sum(1 for r in records if r["outcome"] == k) for k in ("auto_add", "review", "discard")
    }
    print(
        f"{counts}  cost optimizer ${opt_usd:.3f} jev ${jev_usd:.4f} model {jev.model()}\n"
        f"→ {CANDIDATES}"
    )
    return CANDIDATES


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--per-category", type=int, default=10)
    p.add_argument("--all-review", action="store_true", help="first batch: nothing is auto-added")
    p.add_argument("--apply", action="store_true")
    p.add_argument(
        "--cell",
        help="category:check:tables from evals/selection.yaml, e.g. aggregation:number:customers",
    )
    p.add_argument(
        "--stage",
        action="store_true",
        help="proposed rows of evals/selection.yaml → candidates.yaml",
    )
    args = p.parse_args()
    if args.cell:
        parts = args.cell.split(":")
        cat, check, tables = (parts + ["", "", ""])[:3]
        if len(parts) != 3 or cat not in CATS or check not in DATASET_CHECKS or not tables:
            p.error(f"--cell must be category:check:tables, check one of {sorted(DATASET_CHECKS)}")
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
