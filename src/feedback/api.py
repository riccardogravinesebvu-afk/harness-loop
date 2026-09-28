"""POST /feedback {trace_id, verdict, note, category?} → evals/feedback.yaml.

Local development tool, no auth. The run is rebuilt from evals/results/runs.jsonl by trace_id.
Run: uv run uvicorn src.feedback.api:app --port 8765
"""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from evals.run import HERE as EVALS
from evals.run import RUNS_LOG

FEEDBACK = EVALS / "feedback.yaml"
app = FastAPI(title="harness-loop-py feedback")


class Feedback(BaseModel):
    trace_id: str
    verdict: Literal["bad", "good"]
    note: str
    category: str | None = None  # required only for runs that are not dataset cases


def find_run(trace_id: str, log: Path = RUNS_LOG) -> dict | None:
    if not log.exists():
        return None
    hits = [json.loads(ln) for ln in log.read_text().splitlines() if trace_id in ln]
    hits = [r for r in hits if r.get("trace_id") == trace_id]
    return hits[-1] if hits else None


def ingest(run: dict, fb: Feedback, path: Path = FEEDBACK) -> dict:
    doc = yaml.safe_load(path.read_text()) if path.exists() else None
    doc = doc or {"feedback": []}
    entry = {
        "id": f"FB{len(doc['feedback']) + 1:02d}",
        "verdict": fb.verdict,
        "note": fb.note,
        "trace_id": run["trace_id"],
        "ts": datetime.now(UTC).isoformat(timespec="seconds"),
        "results_file": run.get("results_file"),
    }
    if run.get("id"):  # a dataset case: reweight it, do not duplicate the input
        entry["ref"] = run["id"]
    else:
        if not fb.category:
            raise HTTPException(422, "category is required for a run that is not a dataset case")
        entry.update(category=fb.category, input=run["input"])
    doc["feedback"].append(entry)
    path.write_text(yaml.safe_dump(doc, allow_unicode=True, sort_keys=False))
    return entry


@app.post("/feedback")
def post_feedback(fb: Feedback) -> dict:
    run = find_run(fb.trace_id, RUNS_LOG)  # globals looked up at call time: tests patch them
    if run is None:
        raise HTTPException(404, f"no run with trace_id {fb.trace_id} in {RUNS_LOG}")
    return ingest(run, fb, FEEDBACK)
