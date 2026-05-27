"""문서 식별자 생성 모듈.

이 모듈은 `data.jsonl`의 각 문서를 평가에서 안정적으로 비교할 수 있도록
`doc_key`를 만든다. 현재 데이터의 `chunk_id`는 기사 안에서만 고유하므로,
메타데이터와 본문 해시를 함께 묶어 전역 식별자로 사용한다.
"""

from __future__ import annotations

import hashlib
from typing import Any, Mapping


DOC_KEY_FIELDS = (
    "type",
    "king",
    "solar_year",
    "month",
    "day",
    "idx",
    "chunk_id",
)


def content_hash(page_content: str, length: int = 12) -> str:
    """본문 문자열에서 짧은 SHA1 해시를 만든다."""
    digest = hashlib.sha1((page_content or "").encode("utf-8")).hexdigest()
    return digest[:length]


def _safe_value(value: Any) -> str:
    """doc_key 안에서 구분자를 깨뜨리지 않도록 값을 안전한 문자열로 바꾼다."""
    if value is None:
        return ""
    return str(value).replace("|", "/").replace("\n", " ").strip()


def make_doc_key(metadata: Mapping[str, Any] | None, page_content: str) -> str:
    """메타데이터와 본문을 조합해 평가용 문서 키를 만든다."""
    metadata = metadata or {}
    parts = [f"{field}={_safe_value(metadata.get(field))}" for field in DOC_KEY_FIELDS]
    parts.append(f"sha1={content_hash(page_content)}")
    return "|".join(parts)


def make_article_group_key(metadata: Mapping[str, Any] | None) -> str:
    """같은 기사에 속한 chunk들을 묶기 위한 느슨한 그룹 키를 만든다."""
    metadata = metadata or {}
    fields = ("type", "king", "solar_year", "month", "day", "idx", "title")
    return "|".join(f"{field}={_safe_value(metadata.get(field))}" for field in fields)


def attach_doc_key(record: Mapping[str, Any]) -> dict:
    """`page_content`와 `metadata`를 가진 JSONL record에 `doc_key`를 추가한다."""
    obj = dict(record)
    metadata = dict(obj.get("metadata") or {})
    page_content = obj.get("page_content", "")
    metadata["doc_key"] = make_doc_key(metadata, page_content)
    obj["metadata"] = metadata
    return obj

