"""One OpenAI-compatible chat client, configured from env (.env loaded if present)."""

import os
from pathlib import Path

from langchain_openai import ChatOpenAI

ROOT = Path(__file__).resolve().parent.parent


def load_env(path: Path = ROOT / ".env") -> None:
    # 6-line .env loader instead of a python-dotenv dependency
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def model_for(role: str) -> str:
    load_env()
    return os.environ[f"{role.upper()}_MODEL"]


def make_chat(role: str, *, seed: int | None = None, max_tokens: int = 2000) -> ChatOpenAI:
    """role: agent | judge | optimizer. Temperature 0 always: the loop must be replayable."""
    load_env()
    return ChatOpenAI(
        model=model_for(role),
        base_url=os.environ.get("LLM_BASE_URL", "https://api.anthropic.com/v1/"),
        api_key=os.environ["LLM_API_KEY"],
        temperature=0,
        max_tokens=max_tokens,
        seed=seed,  # ignored by the Anthropic endpoint, honoured by OpenAI-style providers
    )
