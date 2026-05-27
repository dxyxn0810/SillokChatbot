"""평가셋 생성 모듈.

이 모듈은 읽기 전용 입력인 `data/data.jsonl`에서 article chunk와 summary 문서를
뽑고, LLM에게 해당 문서로 답할 수 있는 질문과 모범답안을 역생성하게 한다.
생성 결과는 `evaluation/data/` 아래 JSONL로 저장한다.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
from typing import Any

from .core.doc_ids import make_article_group_key, make_doc_key
from .core.env_utils import require_env
from .core.progress import progress_bar
from .core.schemas import EvalItem, dataclass_to_dict, write_jsonl


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT_DIR / "data" / "data.jsonl"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "data" / "eval_set.jsonl"
SUPPORTED_ITEM_TYPES = ("grounded_single", "multi_turn_followup", "multi_chunk", "unanswerable")
DEFAULT_SOURCE_TYPES = ("article", "daily_summary", "monthly_summary", "yearly_summary")


GROUNDED_SINGLE_PROMPT_TEMPLATE = """\
너는 조선왕조실록 기반 RAG 챗봇 평가셋을 만드는 사람이다.
아래 실록 source 문서만 근거로, 이 문서를 검색해야 답할 수 있는 평가 질문과 모범답안을 만들어라.

요구사항:
- JSON 객체 하나만 출력한다.
- question은 한국어 사용자 질문이어야 한다.
- gold_answer는 단종 1인칭 말투가 아니라, 평가 기준이 되는 중립적 사실 답변이어야 한다.
- gold_evidence는 chunk에서 답변 근거가 되는 핵심 문장을 짧게 요약한다.
- source 문서에 없는 사실은 넣지 않는다.

[메타데이터]
{metadata}

[source 문서]
{page_content}

[출력 JSON 형식]
{{
  "question": "...",
  "gold_answer": "...",
  "gold_evidence": "..."
}}
"""

MULTI_TURN_PROMPT_TEMPLATE = """\
너는 조선왕조실록 기반 대화형 RAG 챗봇 평가셋을 만드는 사람이다.
아래 실록 source 문서만 근거로, 이전 대화 1쌍과 follow-up 질문을 만들어라.

요구사항:
- JSON 객체 하나만 출력한다.
- previous_question은 source 문서의 핵심 사건을 묻는 독립 질문이어야 한다.
- previous_answer는 previous_question에 대한 짧은 중립 답변이어야 한다.
- followup_question은 대명사나 생략 표현을 포함해야 한다. 예: "그 일은 왜 문제가 되었나요?", "그 뒤 처분은 어떻게 되었나요?"
- followup_question은 dialogue_history 없이는 의미가 불완전해야 한다.
- gold_answer는 followup_question의 모범답안이며, source 문서에 있는 사실만 사용한다.
- gold_evidence는 근거를 짧게 요약한다.

[메타데이터]
{metadata}

[source 문서]
{page_content}

[출력 JSON 형식]
{{
  "previous_question": "...",
  "previous_answer": "...",
  "followup_question": "...",
  "gold_answer": "...",
  "gold_evidence": "..."
}}
"""

MULTI_CHUNK_PROMPT_TEMPLATE = """\
너는 조선왕조실록 기반 RAG 챗봇 평가셋을 만드는 사람이다.
아래 관련 source 문서 여러 개를 모두 근거로 사용해야 답할 수 있는 평가 질문과 모범답안을 만들어라.

요구사항:
- JSON 객체 하나만 출력한다.
- question은 여러 source 문서의 정보를 종합해야 답할 수 있어야 한다.
- gold_answer는 각 source 문서의 핵심 근거를 연결한 중립적 사실 답변이어야 한다.
- gold_evidence는 어떤 근거들이 연결되는지 짧게 요약한다.
- 제공된 chunk에 없는 사실은 넣지 않는다.

[관련 source 문서들]
{chunks}

[출력 JSON 형식]
{{
  "question": "...",
  "gold_answer": "...",
  "gold_evidence": "..."
}}
"""

UNANSWERABLE_PROMPT_TEMPLATE = """\
너는 조선왕조실록 기반 RAG 챗봇의 negative rejection 평가셋을 만드는 사람이다.
아래 source chunk를 참고하되, 수집된 문종·단종·세조 시기 실록 corpus만으로는
답할 수 없어야 하는 그럴듯한 질문을 만들어라.

요구사항:
- JSON 객체 하나만 출력한다.
- question은 한국어 사용자 질문이어야 한다.
- question은 챗봇이 지어내기 쉬울 정도로 그럴듯해야 한다.
- 하지만 현재 corpus에는 답을 뒷받침할 근거가 없어야 한다.
- 문종·단종·세조 시기를 벗어난 사건, 후대 사건, 현대 개념, 가짜 날짜/사건 조합 등을 활용할 수 있다.
- gold_evidence에는 왜 이 질문이 corpus 기준으로 answerable하지 않은지 짧게 써라.
- gold_answer는 만들지 않는다.

[참고 source metadata]
{metadata}

[참고 source chunk]
{page_content}

[출력 JSON 형식]
{{
  "question": "...",
  "gold_evidence": "..."
}}
"""


def load_source_records(
    path: Path,
    *,
    source_type: str | None = None,
    source_types: list[str] | tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    """평가셋 생성에 사용할 source record를 읽는다."""
    allowed_types = set(source_types or parse_source_types(source_type or ",".join(DEFAULT_SOURCE_TYPES)))
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            obj = json.loads(line)
            metadata = obj.get("metadata") or {}
            if metadata.get("type") not in allowed_types:
                continue
            if not obj.get("page_content"):
                continue
            obj["_line_no"] = line_no
            obj["_doc_key"] = make_doc_key(metadata, obj["page_content"])
            obj["_group_key"] = make_article_group_key(metadata)
            records.append(obj)
    return records


def build_acceptable_doc_map(records: list[dict[str, Any]]) -> dict[str, list[str]]:
    """같은 기사에 속한 chunk들을 relaxed 정답 후보로 묶는다."""
    groups: dict[str, list[str]] = {}
    for record in records:
        groups.setdefault(record["_group_key"], []).append(record["_doc_key"])
    return groups


def parse_item_types(raw: str) -> list[str]:
    """CLI 문자열에서 생성할 item_type 목록을 파싱한다."""
    item_types = [part.strip() for part in raw.split(",") if part.strip()]
    unknown = [item_type for item_type in item_types if item_type not in SUPPORTED_ITEM_TYPES]
    if unknown:
        allowed = ", ".join(SUPPORTED_ITEM_TYPES)
        raise ValueError(f"알 수 없는 item_type입니다: {unknown}. 사용 가능: {allowed}")
    if not item_types:
        raise ValueError("item_types가 비어 있습니다.")
    return item_types


def parse_source_types(raw: str) -> list[str]:
    """CLI 문자열에서 평가셋 source 문서 type 목록을 파싱한다."""
    if raw.strip().lower() == "all":
        return list(DEFAULT_SOURCE_TYPES)
    source_types = [part.strip() for part in raw.split(",") if part.strip()]
    if not source_types:
        raise ValueError("source_types가 비어 있습니다.")
    return source_types


def allocate_item_counts(total: int, item_types: list[str]) -> dict[str, int]:
    """전체 문항 수를 item_type별로 최대한 균등하게 나눈다."""
    if total < 0:
        raise ValueError("n은 0 이상이어야 합니다.")
    base, remainder = divmod(total, len(item_types))
    return {
        item_type: base + (1 if index < remainder else 0)
        for index, item_type in enumerate(item_types)
    }


def build_multi_chunk_groups(records: list[dict[str, Any]], *, max_chunks: int = 4) -> list[dict[str, Any]]:
    """metadata를 이용해 관련 있는 chunk 묶음을 만든다."""
    groups: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()

    def add_group(group_type: str, rows: list[dict[str, Any]]) -> None:
        selected = _unique_records(_sort_records(rows))[:max_chunks]
        if len(selected) < 2:
            return
        key = tuple(record["_doc_key"] for record in selected)
        if key in seen:
            return
        seen.add(key)
        groups.append({"group_type": group_type, "records": selected})

    by_article: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_date: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    by_title: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for record in records:
        metadata = record.get("metadata") or {}
        by_article[record["_group_key"]].append(record)
        by_date[
            (
                metadata.get("king"),
                metadata.get("solar_year"),
                metadata.get("month"),
                metadata.get("day"),
            )
        ].append(record)
        title = metadata.get("title")
        if title:
            by_title[str(title)].append(record)

    for rows in by_article.values():
        add_group("same_article", rows)

    for rows in by_date.values():
        idx_values = {((record.get("metadata") or {}).get("idx")) for record in rows}
        if len(idx_values) >= 2:
            add_group("same_date", rows)

    for rows in by_title.values():
        if len(rows) >= 2:
            add_group("same_title", rows)

    return groups


def call_openai_json(model: str, prompt: str) -> dict[str, Any]:
    """OpenAI Chat Completions API를 호출하고 JSON 객체를 파싱한다."""
    from openai import OpenAI

    client = OpenAI(api_key=require_env("OPENAI_API_KEY"))
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"},
    )
    content = resp.choices[0].message.content or "{}"
    return json.loads(content)


def make_eval_item(
    record: dict[str, Any],
    generated: dict[str, Any],
    *,
    index: int,
    model: str,
    acceptable_doc_keys: list[str],
) -> EvalItem:
    """source record와 LLM 생성 결과를 하나의 `EvalItem`으로 합친다."""
    metadata = dict(record.get("metadata") or {})
    eval_id = f"eval_{index:05d}"
    item = EvalItem(
        eval_id=eval_id,
        item_type="grounded_single",
        question=str(generated.get("question", "")).strip(),
        dialogue_history=[],
        answerable=True,
        gold_answer=str(generated.get("gold_answer", "")).strip(),
        gold_doc_keys=[record["_doc_key"]],
        acceptable_doc_keys=acceptable_doc_keys,
        gold_evidence=str(generated.get("gold_evidence", "")).strip(),
        source_metadata=metadata,
        tags=["llm_generated", metadata.get("type", "unknown")],
        created_by={"model": model, "source": "data.jsonl", "line_no": record.get("_line_no")},
    )
    item.validate()
    return item


def make_multi_turn_eval_item(
    record: dict[str, Any],
    generated: dict[str, Any],
    *,
    index: int,
    model: str,
    acceptable_doc_keys: list[str],
) -> EvalItem:
    """source record와 LLM 생성 결과를 multi-turn 평가 문항으로 합친다."""
    metadata = dict(record.get("metadata") or {})
    previous_question = str(generated.get("previous_question", "")).strip()
    previous_answer = str(generated.get("previous_answer", "")).strip()
    followup_question = str(generated.get("followup_question", "") or generated.get("question", "")).strip()
    item = EvalItem(
        eval_id=f"eval_{index:05d}",
        item_type="multi_turn_followup",
        question=followup_question,
        dialogue_history=[
            {"role": "user", "content": previous_question},
            {"role": "assistant", "content": previous_answer},
        ],
        answerable=True,
        gold_answer=str(generated.get("gold_answer", "")).strip(),
        gold_doc_keys=[record["_doc_key"]],
        acceptable_doc_keys=acceptable_doc_keys,
        gold_evidence=str(generated.get("gold_evidence", "")).strip(),
        source_metadata=metadata,
        tags=["llm_generated", "multi_turn", "followup", metadata.get("type", "unknown")],
        created_by={"model": model, "source": "data.jsonl", "line_no": record.get("_line_no")},
    )
    item.validate()
    return item


def make_multi_chunk_eval_item(
    group: dict[str, Any],
    generated: dict[str, Any],
    *,
    index: int,
    model: str,
) -> EvalItem:
    """관련 chunk 묶음과 LLM 생성 결과를 multi-chunk 평가 문항으로 합친다."""
    records = group["records"]
    gold_doc_keys = [record["_doc_key"] for record in records]
    source_metadata = {
        "group_type": group.get("group_type"),
        "sources": [dict(record.get("metadata") or {}) for record in records],
    }
    item = EvalItem(
        eval_id=f"eval_{index:05d}",
        item_type="multi_chunk",
        question=str(generated.get("question", "")).strip(),
        dialogue_history=[],
        answerable=True,
        gold_answer=str(generated.get("gold_answer", "")).strip(),
        gold_doc_keys=gold_doc_keys,
        acceptable_doc_keys=gold_doc_keys,
        gold_evidence=str(generated.get("gold_evidence", "")).strip(),
        source_metadata=source_metadata,
        tags=["llm_generated", "multi_chunk", str(group.get("group_type", "related"))],
        created_by={
            "model": model,
            "source": "data.jsonl",
            "line_no": [record.get("_line_no") for record in records],
        },
    )
    item.validate()
    return item


def make_unanswerable_eval_item(
    record: dict[str, Any],
    generated: dict[str, Any],
    *,
    index: int,
    model: str,
) -> EvalItem:
    """LLM 생성 결과를 corpus 기준 unanswerable 평가 문항으로 합친다."""
    metadata = dict(record.get("metadata") or {})
    item = EvalItem(
        eval_id=f"eval_{index:05d}",
        item_type="unanswerable",
        question=str(generated.get("question", "")).strip(),
        dialogue_history=[],
        answerable=False,
        gold_answer=None,
        gold_doc_keys=[],
        acceptable_doc_keys=[],
        gold_evidence=str(
            generated.get("gold_evidence")
            or "수집된 문종·단종·세조 시기 실록 corpus 기준으로 답할 근거가 없어야 한다."
        ).strip(),
        source_metadata=metadata,
        tags=["llm_generated", "unanswerable", "negative_rejection"],
        created_by={"model": model, "source": "data.jsonl", "line_no": record.get("_line_no")},
    )
    item.validate()
    return item


def build_dataset(
    input_path: Path,
    output_path: Path,
    *,
    n: int,
    model: str,
    seed: int,
    source_types: list[str],
    item_types: list[str],
) -> list[EvalItem]:
    """평가셋을 생성하고 JSONL로 저장한다."""
    require_env("OPENAI_API_KEY")

    records = load_source_records(input_path, source_types=source_types)
    if not records:
        raise RuntimeError(f"source_types={source_types!r}에 해당하는 문서가 없습니다.")

    rng = random.Random(seed)
    acceptable_map = build_acceptable_doc_map(records)
    counts = allocate_item_counts(n, item_types)
    multi_chunk_groups = build_multi_chunk_groups(records) if counts.get("multi_chunk", 0) > 0 else []
    if counts.get("multi_chunk", 0) > 0 and not multi_chunk_groups:
        raise RuntimeError("multi_chunk 문항을 만들 관련 chunk 묶음이 없습니다.")
    progress_total = _planned_item_total(counts, records, multi_chunk_groups)

    items: list[EvalItem] = []
    next_index = 1

    with progress_bar(total=progress_total, desc="평가셋 생성", unit="문항") as pbar:
        for item_type in item_types:
            count = counts[item_type]
            if count <= 0:
                continue
            if item_type == "grounded_single":
                selected = rng.sample(records, k=min(count, len(records)))
                for record in selected:
                    prompt = GROUNDED_SINGLE_PROMPT_TEMPLATE.format(
                        metadata=json.dumps(record.get("metadata") or {}, ensure_ascii=False),
                        page_content=record.get("page_content", ""),
                    )
                    generated = call_openai_json(model, prompt)
                    acceptable = acceptable_map.get(record["_group_key"], [record["_doc_key"]])
                    items.append(
                        make_eval_item(
                            record,
                            generated,
                            index=next_index,
                            model=model,
                            acceptable_doc_keys=acceptable,
                        )
                    )
                    next_index += 1
                    pbar.update(1)
            elif item_type == "multi_turn_followup":
                selected = rng.sample(records, k=min(count, len(records)))
                for record in selected:
                    prompt = MULTI_TURN_PROMPT_TEMPLATE.format(
                        metadata=json.dumps(record.get("metadata") or {}, ensure_ascii=False),
                        page_content=record.get("page_content", ""),
                    )
                    generated = call_openai_json(model, prompt)
                    acceptable = acceptable_map.get(record["_group_key"], [record["_doc_key"]])
                    items.append(
                        make_multi_turn_eval_item(
                            record,
                            generated,
                            index=next_index,
                            model=model,
                            acceptable_doc_keys=acceptable,
                        )
                    )
                    next_index += 1
                    pbar.update(1)
            elif item_type == "multi_chunk":
                selected_groups = rng.sample(multi_chunk_groups, k=min(count, len(multi_chunk_groups)))
                for group in selected_groups:
                    prompt = MULTI_CHUNK_PROMPT_TEMPLATE.format(chunks=_format_chunk_group(group["records"]))
                    generated = call_openai_json(model, prompt)
                    items.append(
                        make_multi_chunk_eval_item(
                            group,
                            generated,
                            index=next_index,
                            model=model,
                        )
                    )
                    next_index += 1
                    pbar.update(1)
            elif item_type == "unanswerable":
                selected = rng.sample(records, k=min(count, len(records)))
                for record in selected:
                    prompt = UNANSWERABLE_PROMPT_TEMPLATE.format(
                        metadata=json.dumps(record.get("metadata") or {}, ensure_ascii=False),
                        page_content=record.get("page_content", ""),
                    )
                    generated = call_openai_json(model, prompt)
                    items.append(
                        make_unanswerable_eval_item(
                            record,
                            generated,
                            index=next_index,
                            model=model,
                        )
                    )
                    next_index += 1
                    pbar.update(1)

    write_jsonl(output_path, [dataclass_to_dict(item) for item in items])
    return items


def _planned_item_total(
    counts: dict[str, int],
    records: list[dict[str, Any]],
    multi_chunk_groups: list[dict[str, Any]],
) -> int:
    """실제로 생성을 시도할 문항 수를 계산한다."""
    total = 0
    for item_type, count in counts.items():
        if count <= 0:
            continue
        if item_type == "multi_chunk":
            total += min(count, len(multi_chunk_groups))
        else:
            total += min(count, len(records))
    return total


def _format_chunk_group(records: list[dict[str, Any]]) -> str:
    """multi-chunk 생성 프롬프트에 넣을 관련 chunk 묶음을 문자열로 만든다."""
    blocks = []
    for index, record in enumerate(records, start=1):
        metadata = json.dumps(record.get("metadata") or {}, ensure_ascii=False)
        blocks.append(
            f"[chunk {index}]\n"
            f"metadata: {metadata}\n"
            f"content:\n{record.get('page_content', '')}"
        )
    return "\n\n".join(blocks)


def _sort_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """metadata 순서로 record 목록을 안정적으로 정렬한다."""
    return sorted(
        records,
        key=lambda record: (
            str((record.get("metadata") or {}).get("king") or ""),
            str((record.get("metadata") or {}).get("solar_year") or ""),
            str((record.get("metadata") or {}).get("month") or ""),
            str((record.get("metadata") or {}).get("day") or ""),
            str((record.get("metadata") or {}).get("idx") or ""),
            str((record.get("metadata") or {}).get("chunk_id") or ""),
            record.get("_line_no") or 0,
        ),
    )


def _unique_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """doc_key 기준으로 중복 record를 제거한다."""
    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in records:
        key = record["_doc_key"]
        if key in seen:
            continue
        seen.add(key)
        unique.append(record)
    return unique


def main() -> None:
    """CLI 인자를 받아 평가셋을 생성한다."""
    parser = argparse.ArgumentParser(description="SillokChatbot 평가셋 생성기")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="읽을 data.jsonl 경로")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="저장할 eval JSONL 경로")
    parser.add_argument("--n", type=int, default=100, help="생성할 문항 수")
    parser.add_argument("--model", default="gpt-4o", help="평가셋 생성에 사용할 OpenAI 모델")
    parser.add_argument("--seed", type=int, default=42, help="source chunk 샘플링 seed")
    parser.add_argument(
        "--source-types",
        "--source-type",
        dest="source_types",
        default=",".join(DEFAULT_SOURCE_TYPES),
        help="쉼표로 구분한 source metadata type 목록. 예: article,daily_summary,monthly_summary,yearly_summary",
    )
    parser.add_argument(
        "--item-types",
        default="grounded_single",
        help="쉼표로 구분한 문항 유형 목록: grounded_single,multi_turn_followup,multi_chunk,unanswerable",
    )
    args = parser.parse_args()
    item_types = parse_item_types(args.item_types)
    source_types = parse_source_types(args.source_types)

    items = build_dataset(
        args.input,
        args.output,
        n=args.n,
        model=args.model,
        seed=args.seed,
        source_types=source_types,
        item_types=item_types,
    )
    print(f"완료: {len(items)}개 문항을 {args.output}에 저장했습니다.")


if __name__ == "__main__":
    main()
