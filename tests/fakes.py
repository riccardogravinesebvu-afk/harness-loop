"""Fakes shared by tests. FakeJev answers from a table (or a function) and validates through the
SDK's own response model, so a wrong raw dict fails here, not in production."""

from typesafe_sdk import SystemOneResponse


class FakeJev:
    def __init__(self, table, usage=None):
        self.table = table  # {question name: raw answer dict} or callable(name, state) -> dict
        self.calls: list[tuple[object, list[str]]] = []
        self.usage = usage if usage is not None else {"input_tokens": 100, "output_tokens": 0}

    async def system_one(self, state, questions, **kw):
        self.calls.append((state, list(questions)))
        raw = {
            n: (self.table(n, state) if callable(self.table) else self.table[n]) for n in questions
        }
        return SystemOneResponse.model_validate(
            {
                "model": "jev-fake",
                "answers": raw,
                "usage": self.usage,
            }
        )


CHOICE_REFUSED = {"type": "choice", "choice": "refused", "confidence": 0.9,
                  "probabilities": {"refused": 0.95, "answered": 0.05}}  # fmt: skip
CHOICE_ANSWERED = {"type": "choice", "choice": "answered", "confidence": 0.9,
                   "probabilities": {"refused": 0.05, "answered": 0.95}}  # fmt: skip
CHOICE_UNSURE = {"type": "choice", "choice": "refused", "confidence": 0.2,
                 "probabilities": {"refused": 0.6, "answered": 0.4}}  # fmt: skip


def noul(p: float) -> dict:
    return {"type": "noul", "noul": p}
