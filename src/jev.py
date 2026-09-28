"""One TypeSafe (System One) client. Questions live in evals/jev_questions.yaml, not in code.
Without TYPESAFE_API_KEY nothing here is called: consumers check enabled() and fall back loudly
(results say models.jev = null). Model pinned by JEV_MODEL: aliases move on every release."""

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

from src.llm import load_env
from src.pricing import cost_usd

QUESTIONS = Path(__file__).resolve().parents[1] / "evals" / "jev_questions.yaml"
DEFAULT_MODEL = "jev-1.13.0"


@dataclass
class Answer:
    type: str  # choice | score | noul
    value: str | float | None  # choice name, score level, None for noul
    probabilities: dict[str, float]
    confidence: float | None  # None for noul: the SDK does not report one
    noul: float | None


@dataclass
class JevResult:
    answers: dict[str, Answer]
    model: str
    input_tokens: int
    cost_usd: float


def enabled() -> bool:
    load_env()
    return bool(os.environ.get("TYPESAFE_API_KEY"))


def model() -> str:
    load_env()
    return os.environ.get("JEV_MODEL", DEFAULT_MODEL)


def load_questions(path: Path = QUESTIONS) -> dict[str, dict]:
    return yaml.safe_load(path.read_text())["questions"]


def question(name: str, **fmt) -> dict:
    """A question template with its {placeholders} filled. Criteria are formatted too."""
    q = dict(load_questions()[name])
    q["instructions"] = q["instructions"].format(**fmt)
    c = q.get("criteria")
    if isinstance(c, dict):
        q["criteria"] = {k: (v.format(**fmt) if isinstance(v, str) else v) for k, v in c.items()}
    elif isinstance(c, list):
        q["criteria"] = [v.format(**fmt) if isinstance(v, str) else v for v in c]
    return q


def _build(q: dict):
    from typesafe_sdk import Choice, Noul, NoulCriteria, Score

    t = q["type"]
    if t == "choice":
        return Choice(instructions=q["instructions"], criteria=q["criteria"])
    if t == "score":
        return Score(instructions=q["instructions"], criteria=q["criteria"])
    if t == "noul":
        c = q.get("criteria")
        return Noul(instructions=q["instructions"], criteria=NoulCriteria(**c) if c else None)
    raise ValueError(f"unknown question type {t!r}")


_client = None


def _default_client():
    global _client
    if _client is None:
        from typesafe_sdk import AsyncTypeSafeClient

        load_env()
        _client = AsyncTypeSafeClient(api_key=os.environ["TYPESAFE_API_KEY"], model=model())
    return _client


async def ask(state, questions: dict[str, dict], client=None) -> JevResult:
    """One request, every question evaluated in parallel against the same state."""
    client = client or _default_client()
    res = await client.system_one(state, {k: _build(q) for k, q in questions.items()})
    out: dict[str, Answer] = {}
    for k, a in res.answers.items():
        if a.type == "choice":
            out[k] = Answer("choice", a.choice, dict(a.probabilities), a.confidence, None)
        elif a.type == "score":
            probs = {str(lvl): p for lvl, p in a.probabilities.items()}
            out[k] = Answer("score", a.score, probs, a.confidence, None)
        else:
            out[k] = Answer("noul", None, {}, None, a.noul)
    # SDK types usage optional; None → 0 tokens (real calls always report it)
    input_tokens = res.usage.input_tokens or 0
    output_tokens = res.usage.output_tokens or 0
    usd, _ = cost_usd(res.model, input_tokens, output_tokens)
    return JevResult(out, res.model, input_tokens, usd)
