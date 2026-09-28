"""Langfuse, optional: silent without keys, one trace per agent run with a deterministic trace_id.

The trace_id is derived locally from a seed string, so runs.jsonl and Langfuse share the same id
whether or not keys are set. Trace name and tags go through propagate_attributes; the LangChain
callback records every LLM call and tool call under the current span."""

import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager

from langfuse import Langfuse, propagate_attributes
from langfuse.langchain import CallbackHandler

from src.llm import load_env

_client: Langfuse | None = None


def enabled() -> bool:
    load_env()
    return bool(os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY"))


def client() -> Langfuse:
    global _client  # noqa: PLW0603 — one client per process, Langfuse batches and flushes it
    if _client is None:
        on = enabled()
        if not on:
            logging.getLogger("langfuse").setLevel(logging.CRITICAL)  # no "auth error" noise
        _client = Langfuse(tracing_enabled=on)
    return _client


def trace_id(seed: str) -> str:
    """32-hex id, deterministic from the seed string (e.g. '<results ts>:<case id>')."""
    return Langfuse.create_trace_id(seed=seed)


@contextmanager
def trace(
    name: str, tid: str, tags: list[str], metadata: dict, span_name: str = "eval_context"
) -> Iterator[tuple]:
    """Yields (root span, callbacks). The root span is `span_name`; the trace is named `name`."""
    root = client().start_as_current_observation(name=span_name, trace_context={"trace_id": tid})
    with root as span, propagate_attributes(trace_name=name, tags=tags, metadata=metadata):
        yield span, ([CallbackHandler()] if enabled() else [])


def flush() -> None:
    if _client is not None:
        _client.flush()
