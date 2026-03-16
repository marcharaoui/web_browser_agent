from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACTS_DIR = ROOT_DIR / "artifacts" / "runs"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def slugify(text: str, max_length: int = 48) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower()).strip("-")
    return slug[:max_length] or "run"


def make_run_id(goal: str) -> str:
    return f"{timestamp_slug()}_{slugify(goal)}"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def truncate_text(text: str, max_chars: int) -> str:
    compact = re.sub(r"\s+", " ", text or "").strip()
    return compact[:max_chars]


def normalized_to_pixels(x: int, y: int, viewport_width: int, viewport_height: int) -> tuple[int, int]:
    return (
        int((x / 1000.0) * viewport_width),
        int((y / 1000.0) * viewport_height),
    )


def to_jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


def dump_json(path: Path, value: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(to_jsonable(value), indent=2), encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))
