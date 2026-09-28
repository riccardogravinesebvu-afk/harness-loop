"""Offline: no Langfuse keys, no provider key."""

import json
import os

import pytest

from src import observability as obs
from src.agent_under_test.graph import AgentResult, ToolCall


@pytest.fixture(autouse=True)
def no_keys(monkeypatch):
    for k in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(obs, "load_env", lambda: None)


def test_trace_id_is_deterministic_and_silent_without_keys():
    assert obs.trace_id("ts:A01") == obs.trace_id("ts:A01") != obs.trace_id("ts:A02")
    assert len(obs.trace_id("x")) == 32
    with obs.trace("eval:A01", obs.trace_id("ts:A01"), ["lookup"], {"seed": 42}) as (span, cbs):
        span.update(input={"a": 1}, output={"b": 2})
        obs.client().score_current_trace(name="passed", value=1)
    assert cbs == []  # no keys → no callback, no network
    assert not obs.enabled()


def test_runs_jsonl_line_rebuilds_the_case(tmp_path):
    from evals.run import log_run

    res = AgentResult(
        answer="42",
        tool_calls=[ToolCall(name="run_sql", args={"sql": "s"}, output="o" * 3000)],
        model="m",
    )
    row = {"id": "A01", "category": "aggregation", "check": "number", "input": "q", "answer": "42",
           "refused": False, "error": None, "passed": True, "detail": "ok", "cost_usd": 0.001,
           "trace_id": obs.trace_id("ts:A01")}  # fmt: skip
    case = {"id": "A01", "expected": 42}
    ctx = {"git_sha": "abc", "results_file": "evals/results/x.json", "iteration": None,
           "hypothesis_id": None, "seed": 42}  # fmt: skip
    log_run(row, res, case, ctx, path=tmp_path / "runs.jsonl")
    line = json.loads((tmp_path / "runs.jsonl").read_text())
    for k in ("trace_id", "ts", "id", "category", "check", "expected", "input", "answer",
              "refused", "error", "tool_calls", "passed", "detail", "cost_usd", "model",
              "git_sha", "results_file", "iteration"):  # fmt: skip
        assert k in line, k
    assert len(line["tool_calls"][0]["output"]) == 1500
    assert "LLM_API_KEY" not in os.environ or "sk-" not in json.dumps(line)
