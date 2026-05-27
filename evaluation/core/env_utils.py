"""환경 변수 로딩 모듈.

이 모듈은 평가 CLI가 `OPENAI_API_KEY` 같은 값을 `.env`에서 읽을 수 있게 한다.
`python-dotenv`가 설치되어 있지 않아도 동작하도록, evaluation 내부에서 필요한
최소한의 `.env` 파서를 제공한다.
"""

from __future__ import annotations

import os
from pathlib import Path


EVAL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EVAL_DIR.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent


def load_project_env(*, override: bool = False) -> list[Path]:
    """평가 실행에 필요한 `.env` 후보들을 읽고, 실제 로드한 경로 목록을 반환한다."""
    candidates = _env_candidates()
    loaded: list[Path] = []
    for path in candidates:
        if path.exists() and path.is_file():
            _load_env_file(path, override=override)
            loaded.append(path)
    return loaded


def require_env(name: str) -> str:
    """환경 변수를 `.env`에서 로드한 뒤 필수 값이 있는지 확인한다."""
    load_project_env()
    value = os.getenv(name)
    if not value:
        loaded = ", ".join(str(p) for p in _env_candidates() if p.exists()) or "없음"
        raise RuntimeError(f"{name}가 필요합니다. 확인한 .env 경로: {loaded}")
    return value


def _env_candidates() -> list[Path]:
    """실행 위치가 달라도 찾을 수 있도록 `.env` 후보 경로를 넓게 잡는다."""
    cwd = Path.cwd().resolve()
    candidates = [
        cwd / ".env",
        cwd.parent / ".env",
        EVAL_DIR / ".env",
        PROJECT_ROOT / ".env",
        WORKSPACE_ROOT / ".env",
    ]
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        if path not in seen:
            unique.append(path)
            seen.add(path)
    return unique


def _load_env_file(path: Path, *, override: bool) -> None:
    """간단한 KEY=VALUE 형식의 `.env` 파일을 환경 변수로 반영한다."""
    with path.open(encoding="utf-8") as f:
        for line in f:
            parsed = _parse_env_line(line)
            if parsed is None:
                continue
            key, value = parsed
            if override or key not in os.environ:
                os.environ[key] = value


def _parse_env_line(line: str) -> tuple[str, str] | None:
    """주석과 빈 줄을 제외하고 `.env` 한 줄을 key/value로 해석한다."""
    stripped = line.strip()
    if not stripped or stripped.startswith("#") or "=" not in stripped:
        return None
    key, value = stripped.split("=", 1)
    key = key.strip()
    if key.startswith("export "):
        key = key[len("export ") :].strip()
    if not key:
        return None
    value = _strip_inline_comment(value.strip())
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    return key, value


def _strip_inline_comment(value: str) -> str:
    """따옴표 밖의 인라인 주석을 제거한다."""
    quote: str | None = None
    for i, ch in enumerate(value):
        if ch in {"'", '"'}:
            quote = None if quote == ch else ch
        elif ch == "#" and quote is None:
            if i == 0 or value[i - 1].isspace():
                return value[:i].strip()
    return value

