import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    api_key: str
    base_url: str
    model: str
    max_steps: int
    loop_limit: int
    workspace: Path


def load_settings() -> Settings:
    api_key = os.getenv("GEMINI_API_KEY", "")
    model = os.getenv("MODEL", "")
    if not api_key or not model:
        raise SystemExit("Set GEMINI_API_KEY and MODEL in .env (see .env.example)")
    return Settings(
        api_key=api_key,
        base_url=os.getenv(
            "BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/"
        ),
        model=model,
        max_steps=int(os.getenv("MAX_STEPS", "15")),
        loop_limit=int(os.getenv("LOOP_LIMIT", "3")),
        workspace=Path(os.getenv("WORKSPACE", ".")).resolve(),
    )
