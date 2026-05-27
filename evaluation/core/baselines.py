"""baseline 설정 모듈.

이 모듈은 평가에서 비교할 챗봇 실행 조건을 한곳에 모아 둔다. 기존
`DanjongBot`의 CLI 옵션과 같은 의미를 쓰되, 검색이 없는 baseline은
retrieval metric을 N/A 처리할 수 있도록 `uses_retrieval`을 명시한다.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BaselineConfig:
    """하나의 baseline 실행 설정."""

    name: str
    data_level: str
    use_cot: bool
    retrieval_k: int
    history_turns: int = 6
    uses_retrieval: bool = True
    use_query_rewrite: bool = False
    description: str = ""


BASELINES: dict[str, BaselineConfig] = {
    "api_only": BaselineConfig(
        name="api_only",
        data_level="prompt",
        use_cot=False,
        retrieval_k=0,
        uses_retrieval=False,
        description="검색 없이 단종 페르소나 프롬프트만 사용하는 baseline",
    ),
    "rag_article": BaselineConfig(
        name="rag_article",
        data_level="article",
        use_cot=False,
        retrieval_k=10,
        description="article 문서만 순수 의미검색으로 사용하는 RAG baseline",
    ),
    "rag_rewrite": BaselineConfig(
        name="rag_rewrite",
        data_level="article",
        use_cot=False,
        retrieval_k=10,
        use_query_rewrite=True,
        description="질문을 검색용 질의로 재작성한 뒤 article 문서를 검색하는 RAG baseline",
    ),
    "rag_article_rewrite": BaselineConfig(
        name="rag_article_rewrite",
        data_level="article",
        use_cot=False,
        retrieval_k=10,
        use_query_rewrite=True,
        description="rag_rewrite의 이전 이름. 예전 실험 호환용 alias",
    ),
    "rag_article_cot": BaselineConfig(
        name="rag_article_cot",
        data_level="article",
        use_cot=True,
        retrieval_k=10,
        description="CoT 메타데이터 계획을 사용해 article 문서를 검색하는 baseline",
    ),
    "rag_full": BaselineConfig(
        name="rag_full",
        data_level="wikipedia",
        use_cot=True,
        retrieval_k=10,
        description="article, summary, 위키 보조근거를 모두 사용하는 전체 baseline",
    ),
    "rag_full_rewrite": BaselineConfig(
        name="rag_full_rewrite",
        data_level="wikipedia",
        use_cot=True,
        retrieval_k=10,
        use_query_rewrite=True,
        description="검색 질의 재작성 후 article, summary, 위키 보조근거를 모두 사용하는 전체 baseline",
    ),
}


def get_baseline(name: str) -> BaselineConfig:
    """이름으로 baseline 설정을 조회한다."""
    try:
        return BASELINES[name]
    except KeyError as exc:
        available = ", ".join(sorted(BASELINES))
        raise ValueError(f"알 수 없는 baseline입니다: {name}. 사용 가능: {available}") from exc


def list_baselines() -> list[BaselineConfig]:
    """등록된 baseline 설정을 이름순으로 반환한다."""
    return [BASELINES[name] for name in sorted(BASELINES)]
