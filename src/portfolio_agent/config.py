from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class AppConfig:
    data_dir: Path
    sqlite_path: Path
    openai_api_key: str | None


def load_config() -> AppConfig:
    load_dotenv(override=False)

    data_dir = Path(os.getenv("AGENTIC_BRIEF_DATA_DIR", ".agentic-brief")).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    sqlite_path = Path(os.getenv("AGENTIC_BRIEF_SQLITE_PATH", str(data_dir / "state.sqlite3"))).resolve()

    openai_api_key = os.getenv("OPENAI_API_KEY")

    return AppConfig(
        data_dir=data_dir,
        sqlite_path=sqlite_path,
        openai_api_key=openai_api_key,
    )
