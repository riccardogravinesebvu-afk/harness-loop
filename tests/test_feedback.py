"""Offline: feedback ingest, weighted load_cases, recovery bookkeeping fields."""

import yaml
from fastapi.testclient import TestClient

from evals.run import load_cases
from src.feedback import api

RUN = {"trace_id": "abc", "id": "L09", "input": "List the customers based in Italy.",
       "results_file": "evals/results/x.json"}  # fmt: skip


def test_ingest_dataset_case_reweights_instead_of_duplicating(tmp_path, monkeypatch):
    monkeypatch.setenv("FEEDBACK_WEIGHT", "3")
    fb_path = tmp_path / "feedback.yaml"
    e = api.ingest(RUN, api.Feedback(trace_id="abc", verdict="bad", note="missed Gallo"), fb_path)
    assert e["id"] == "FB01" and e["ref"] == "L09" and "input" not in e
    cases = load_cases(feedback=fb_path)
    l09 = next(c for c in cases if c["id"] == "L09")
    assert l09["weight"] == 3 and l09["feedback"]["note"] == "missed Gallo"
    assert len(cases) == len(load_cases())  # no duplicate input
    # a good verdict is recorded but changes nothing
    api.ingest(
        {**RUN, "id": "A01"}, api.Feedback(trace_id="abc", verdict="good", note="ok"), fb_path
    )
    assert next(c for c in load_cases(feedback=fb_path) if c["id"] == "A01")["weight"] == 1
    assert len(yaml.safe_load(fb_path.read_text())["feedback"]) == 2


def test_ingest_ad_hoc_run_becomes_judge_case(tmp_path):
    fb_path = tmp_path / "feedback.yaml"
    run = {"trace_id": "zzz", "id": None, "input": "Who pays late?", "results_file": None}
    e = api.ingest(run, api.Feedback(trace_id="zzz", verdict="bad", note="Delta pays late",
                                     category="reasoning"), fb_path)  # fmt: skip
    new = next(c for c in load_cases(feedback=fb_path) if c["id"] == e["id"])
    assert new["check"] == "judge" and new["reference"] == "Delta pays late"
    assert new["source"] == "feedback" and new["weight"] == 2 and new["split"] == "visible"


def test_endpoint_404_without_run_and_writes_with_run(tmp_path, monkeypatch):
    log = tmp_path / "runs.jsonl"
    log.write_text('{"trace_id": "abc", "id": "L09", "input": "q", "results_file": "r"}\n')
    monkeypatch.setattr(api, "RUNS_LOG", log)
    monkeypatch.setattr(api, "FEEDBACK", tmp_path / "feedback.yaml")
    client = TestClient(api.app)
    assert (
        client.post(
            "/feedback", json={"trace_id": "nope", "verdict": "bad", "note": "n"}
        ).status_code
        == 404
    )
    r = client.post("/feedback", json={"trace_id": "abc", "verdict": "bad", "note": "n"})
    assert r.status_code == 200 and r.json()["ref"] == "L09"
