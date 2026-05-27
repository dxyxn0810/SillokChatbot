"""baseline 실행 로그 수집 모듈.

이 모듈은 기존 `doyoon_project` 코드를 수정하지 않고 import해서 baseline을 실행한다.
평가 adapter가 `DanjongBot.ask()`와 같은 흐름을 재현하며, 중간 산출물인 검색 계획,
검색 문서, 생성 답변, 실행 시간을 `evaluation/runs/` 아래 JSONL로 저장한다.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
from typing import Any
from uuid import uuid4

from .core.baselines import get_baseline
from .core.doc_ids import make_doc_key
from .core.env_utils import require_env
from .core.progress import progress_iter
from .core.schemas import RunLog, dataclass_to_dict, eval_item_from_dict, read_jsonl, write_jsonl


EVAL_DIR = Path(__file__).resolve().parent
ROOT_DIR = EVAL_DIR.parent
DEFAULT_EVAL_SET = EVAL_DIR / "data" / "eval_set.jsonl"
DEFAULT_RUN_DIR = EVAL_DIR / "runs"


def _prepare_import_path() -> None:
    """기존 doyoon_project 모듈의 상대 import가 동작하도록 경로를 보강한다."""
    project_dir = ROOT_DIR / "doyoon_project"
    for path in (str(project_dir), str(ROOT_DIR)):
        if path not in sys.path:
            sys.path.insert(0, path)


class DanjongEvalAdapter:
    """기존 DanjongBot 흐름을 평가용으로 감싸는 adapter."""

    def __init__(self, baseline_name: str, *, faiss_dir: Path | None = None, verbose: bool = False):
        _prepare_import_path()
        from chatbot import CHAT_MODEL, build_messages  # type: ignore
        from cot_planner import make_plan  # type: ignore
        from langchain_openai import ChatOpenAI
        from retriever import format_context, load_vectorstore, retrieve  # type: ignore

        self.config = get_baseline(baseline_name)
        self.verbose = verbose
        self.chat_model = CHAT_MODEL
        self.build_messages = build_messages
        self.make_plan = make_plan
        self.retrieve = retrieve
        self.format_context = format_context
        self.llm = ChatOpenAI(model=CHAT_MODEL, temperature=0.7)
        self.rewrite_llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)
        self.vs = None
        if self.config.uses_retrieval:
            self.vs = load_vectorstore(faiss_dir or (ROOT_DIR / "faiss"))

    def run_item(self, item: dict[str, Any], run_id: str) -> RunLog:
        """평가 문항 하나를 실행하고 로그를 반환한다."""
        question = item["question"]
        history = list(item.get("dialogue_history") or [])
        start = time.perf_counter()
        plan: dict[str, Any] = {}
        docs = []
        answer: str | None = None
        error: str | None = None

        try:
            wiki_evidence = ""
            if self.config.uses_retrieval:
                allowed_types = _allowed_types(self.config.data_level)
                search_query = question
                rewritten_query = None
                if self.config.use_query_rewrite:
                    rewritten_query = self._rewrite_search_query(question, history)
                    search_query = rewritten_query or question
                if self.config.use_cot:
                    plan = self.make_plan(search_query, verbose=self.verbose)
                    planned = plan.get("doc_types") or []
                    intersect = [t for t in planned if t in allowed_types]
                    plan["doc_types"] = intersect or allowed_types
                    if self.config.data_level != "wikipedia":
                        plan["wiki_evidence"] = ""
                else:
                    plan = {
                        "search_query": search_query,
                        "king": None,
                        "solar_year": None,
                        "solar_year_range": None,
                        "month": None,
                        "month_range": None,
                        "day": None,
                        "day_range": None,
                        "doc_types": allowed_types,
                        "wiki_evidence": "",
                    }
                plan["original_question"] = question
                plan["query_rewrite_enabled"] = self.config.use_query_rewrite
                if self.config.use_query_rewrite:
                    plan["rewritten_query"] = rewritten_query
                fetch_k = _overfetch_k(self.config.retrieval_k)
                raw_docs = self.retrieve(self.vs, plan, k=fetch_k)
                docs = _filter_docs_by_allowed_types(
                    raw_docs,
                    plan.get("doc_types") or allowed_types,
                    limit=self.config.retrieval_k,
                )
                wiki_evidence = plan.get("wiki_evidence", "") if self.config.data_level == "wikipedia" else ""

            context = self.format_context(docs) if docs else ""
            msgs = self.build_messages(
                history,
                question,
                context,
                wiki_evidence,
                history_turns=self.config.history_turns,
            )
            answer = self.llm.invoke(msgs).content
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"

        timing_ms = (time.perf_counter() - start) * 1000
        retrieved_context = [_document_to_log(rank, doc) for rank, doc in enumerate(docs, start=1)]
        return RunLog(
            run_id=run_id,
            baseline=self.config.name,
            eval_id=item["eval_id"],
            question=question,
            dialogue_history=history,
            plan=plan,
            retrieved_context=retrieved_context,
            generated_answer=answer,
            error=error,
            timing_ms=timing_ms,
        )

    def _rewrite_search_query(self, question: str, history: list[dict[str, str]]) -> str:
        """대화 맥락과 현재 질문을 검색에 적합한 독립 질의로 재작성한다."""
        history_text = _format_history_for_rewrite(history)
        prompt = f"""\
너는 조선왕조실록 RAG 검색 질의 재작성기다.
사용자의 현재 질문을 FAISS 검색에 적합한 독립 한국어 검색 질의 한 줄로 바꾸어라.

규칙:
- 대명사나 생략된 표현은 대화 기록을 이용해 구체화한다.
- 실록 검색에 도움이 되는 핵심 인물, 사건, 날짜, 왕명을 포함한다.
- 답변을 쓰지 말고 검색 질의만 출력한다.
- 근거에 없는 사실을 새로 만들지 않는다.

[대화 기록]
{history_text}

[현재 질문]
{question}

[검색 질의]
"""
        response = self.rewrite_llm.invoke(prompt)
        rewritten = getattr(response, "content", str(response)).strip()
        return rewritten or question


def _allowed_types(data_level: str) -> list[str]:
    """DanjongBot의 data level 규칙을 evaluation 내부에서 동일하게 적용한다."""
    mapping = {
        "prompt": [],
        "article": ["article"],
        "summary": ["article", "daily_summary", "monthly_summary", "yearly_summary"],
        "wikipedia": ["article", "daily_summary", "monthly_summary", "yearly_summary"],
    }
    return mapping.get(data_level, [])


def _overfetch_k(retrieval_k: int) -> int:
    """type 후처리로 빠질 문서를 고려해 내부 검색 개수를 넉넉히 잡는다."""
    return max(retrieval_k, retrieval_k * 3)


def _filter_docs_by_allowed_types(docs: list[Any], allowed_types: list[str], *, limit: int) -> list[Any]:
    """baseline이 허용한 문서 type만 남기고 최종 검색 개수로 자른다."""
    if limit <= 0:
        return []
    if not allowed_types:
        return docs[:limit]

    allowed = set(allowed_types)
    filtered = []
    for doc in docs:
        metadata = getattr(doc, "metadata", {}) or {}
        if metadata.get("type") in allowed:
            filtered.append(doc)
        if len(filtered) >= limit:
            break
    return filtered


def _format_history_for_rewrite(history: list[dict[str, str]]) -> str:
    """query rewriting 프롬프트에 넣을 대화 기록 문자열을 만든다."""
    if not history:
        return "(없음)"
    lines = []
    for turn in history:
        role = turn.get("role", "")
        content = turn.get("content", "")
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _document_to_log(rank: int, doc: Any) -> dict[str, Any]:
    """LangChain Document를 실행 로그용 dict로 변환한다."""
    metadata = dict(getattr(doc, "metadata", {}) or {})
    text = getattr(doc, "page_content", "") or ""
    doc_key = make_doc_key(metadata, text)
    return {"rank": rank, "doc_key": doc_key, "metadata": metadata, "text": text}


def run_baseline(
    eval_set_path: Path,
    output_path: Path,
    *,
    baseline: str,
    faiss_dir: Path | None,
    limit: int | None,
    verbose: bool,
) -> list[RunLog]:
    """평가셋 전체를 하나의 baseline으로 실행한다."""
    require_env("OPENAI_API_KEY")

    rows = read_jsonl(eval_set_path)
    items = [dataclass_to_dict(eval_item_from_dict(row)) for row in rows]
    if limit is not None:
        items = items[:limit]

    adapter = DanjongEvalAdapter(baseline, faiss_dir=faiss_dir, verbose=verbose)
    run_id = f"{baseline}_{uuid4().hex[:8]}"
    logs = [
        adapter.run_item(item, run_id)
        for item in progress_iter(items, total=len(items), desc=f"{baseline} 실행", unit="문항")
    ]
    write_jsonl(output_path, [dataclass_to_dict(log) for log in logs])
    return logs


def main() -> None:
    """CLI 인자를 받아 baseline 실행 로그를 생성한다."""
    parser = argparse.ArgumentParser(description="SillokChatbot baseline 실행기")
    parser.add_argument("--eval-set", type=Path, default=DEFAULT_EVAL_SET, help="평가셋 JSONL")
    parser.add_argument("--baseline", required=True, help="실행할 baseline 이름")
    parser.add_argument("--out", type=Path, default=None, help="저장할 run JSONL")
    parser.add_argument("--faiss", type=Path, default=ROOT_DIR / "faiss", help="FAISS 인덱스 폴더")
    parser.add_argument("--limit", type=int, default=None, help="앞에서부터 일부 문항만 실행")
    parser.add_argument("--verbose", action="store_true", help="기존 챗봇의 verbose 출력을 켠다")
    args = parser.parse_args()

    out = args.out or (DEFAULT_RUN_DIR / f"{args.baseline}.jsonl")
    logs = run_baseline(
        args.eval_set,
        out,
        baseline=args.baseline,
        faiss_dir=args.faiss,
        limit=args.limit,
        verbose=args.verbose,
    )
    print(f"완료: {len(logs)}개 실행 로그를 {out}에 저장했습니다.")


if __name__ == "__main__":
    main()
