"""평가 데이터 구조 모듈.

이 모듈은 평가셋, baseline 실행 로그, metric 점수 결과에서 공통으로 쓰는
dataclass와 JSONL 입출력 함수를 정의한다. 외부 의존성 없이 표준 라이브러리만
사용해서 테스트와 실행 환경을 가볍게 유지한다.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


@dataclass
class DialogueTurn:
    """평가 문항에 포함되는 이전 대화 한 턴."""

    role: str
    content: str


@dataclass
class RetrievedDocument:
    """baseline이 실제로 검색한 문서 한 건."""

    rank: int
    doc_key: str
    metadata: dict[str, Any]
    text: str


@dataclass
class EvalItem:
    """평가셋의 문항 한 건."""

    eval_id: str
    item_type: str
    question: str
    dialogue_history: list[dict[str, str]] = field(default_factory=list)
    answerable: bool = True
    gold_answer: str | None = None
    gold_doc_keys: list[str] = field(default_factory=list)
    acceptable_doc_keys: list[str] = field(default_factory=list)
    gold_evidence: str | None = None
    source_metadata: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    created_by: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """필수 필드와 기본 타입을 검사한다."""
        if not self.eval_id:
            raise ValueError("eval_id가 비어 있습니다.")
        if not self.question:
            raise ValueError(f"{self.eval_id}: question이 비어 있습니다.")
        if self.answerable and not self.gold_doc_keys:
            raise ValueError(f"{self.eval_id}: answerable 문항에는 gold_doc_keys가 필요합니다.")
        _validate_history(self.dialogue_history, self.eval_id)


@dataclass
class RunLog:
    """baseline 실행 결과 한 건."""

    run_id: str
    baseline: str
    eval_id: str
    question: str
    dialogue_history: list[dict[str, str]] = field(default_factory=list)
    plan: dict[str, Any] = field(default_factory=dict)
    retrieved_context: list[dict[str, Any]] = field(default_factory=list)
    generated_answer: str | None = None
    error: str | None = None
    timing_ms: float | None = None

    def validate(self) -> None:
        """실행 로그의 필수 식별자를 검사한다."""
        if not self.run_id:
            raise ValueError("run_id가 비어 있습니다.")
        if not self.baseline:
            raise ValueError(f"{self.eval_id}: baseline이 비어 있습니다.")
        if not self.eval_id:
            raise ValueError("eval_id가 비어 있습니다.")
        _validate_history(self.dialogue_history, self.eval_id)


@dataclass
class MetricResult:
    """하나의 metric 결과. N/A인 경우 value는 None이다."""

    value: float | None
    na: bool = False
    reason: str = ""

    @classmethod
    def of(cls, value: float) -> "MetricResult":
        """실수 값을 가진 metric 결과를 만든다."""
        return cls(value=float(value), na=False, reason="")

    @classmethod
    def not_applicable(cls, reason: str) -> "MetricResult":
        """평가 대상이 아닌 metric 결과를 만든다."""
        return cls(value=None, na=True, reason=reason)


@dataclass
class ScoreRecord:
    """한 eval item에 대한 전체 채점 결과."""

    run_id: str
    baseline: str
    eval_id: str
    metrics: dict[str, dict[str, Any]]
    judge_reasons: dict[str, str] = field(default_factory=dict)
    error: str | None = None


def dataclass_to_dict(obj: Any) -> dict[str, Any]:
    """dataclass 객체를 JSON 직렬화 가능한 dict로 변환한다."""
    return asdict(obj)


def eval_item_from_dict(obj: Mapping[str, Any]) -> EvalItem:
    """dict에서 `EvalItem`을 만들고 검증한다."""
    item = EvalItem(
        eval_id=str(obj.get("eval_id", "")),
        item_type=str(obj.get("item_type", "grounded_single")),
        question=str(obj.get("question", "")),
        dialogue_history=list(obj.get("dialogue_history") or []),
        answerable=bool(obj.get("answerable", True)),
        gold_answer=obj.get("gold_answer"),
        gold_doc_keys=list(obj.get("gold_doc_keys") or []),
        acceptable_doc_keys=list(obj.get("acceptable_doc_keys") or []),
        gold_evidence=obj.get("gold_evidence"),
        source_metadata=dict(obj.get("source_metadata") or {}),
        tags=list(obj.get("tags") or []),
        created_by=dict(obj.get("created_by") or {}),
    )
    item.validate()
    return item


def run_log_from_dict(obj: Mapping[str, Any]) -> RunLog:
    """dict에서 `RunLog`를 만들고 검증한다."""
    log = RunLog(
        run_id=str(obj.get("run_id", "")),
        baseline=str(obj.get("baseline", "")),
        eval_id=str(obj.get("eval_id", "")),
        question=str(obj.get("question", "")),
        dialogue_history=list(obj.get("dialogue_history") or []),
        plan=dict(obj.get("plan") or {}),
        retrieved_context=list(obj.get("retrieved_context") or []),
        generated_answer=obj.get("generated_answer"),
        error=obj.get("error"),
        timing_ms=obj.get("timing_ms"),
    )
    log.validate()
    return log


def metric_to_dict(metric: MetricResult) -> dict[str, Any]:
    """`MetricResult`를 JSON용 dict로 바꾼다."""
    return dataclass_to_dict(metric)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """JSONL 파일을 읽어 dict 목록으로 반환한다."""
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no} JSON 파싱 실패: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """dict 목록을 JSONL 파일로 저장한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _validate_history(history: list[dict[str, str]], owner: str) -> None:
    """대화 기록이 role/content 구조인지 확인한다."""
    for i, turn in enumerate(history):
        role = turn.get("role")
        content = turn.get("content")
        if role not in {"user", "assistant", "system"}:
            raise ValueError(f"{owner}: dialogue_history[{i}].role 값이 올바르지 않습니다.")
        if not isinstance(content, str):
            raise ValueError(f"{owner}: dialogue_history[{i}].content는 문자열이어야 합니다.")

