"""
FAISS 인덱스(faiss_kfa) 를 로드하고 사용자의 질문에 답하는 대화형 RAG.

사용법:
    export OPENAI_API_KEY=...
    python rag.py
    # 또는 옵션 지정
    python rag.py --index-dir ./faiss_kfa --k 5 --model gpt-4o-mini
"""

import argparse
import os
import re
import sys
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --------------------------------------------------------------------------- #
# 1) 인덱스 로드
# --------------------------------------------------------------------------- #
def load_vectorstore(index_dir: Path, embedding_model: str) -> FAISS:
    embeddings = OpenAIEmbeddings(model=embedding_model)
    vs = FAISS.load_local(
        str(index_dir),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    return vs


# --------------------------------------------------------------------------- #
# 2) 검색 결과 -> 프롬프트용 컨텍스트
# --------------------------------------------------------------------------- #
def _format_period(meta: dict) -> str:
    """metadata 의 king/year/month/day 를 사람이 읽기 좋게 합친다."""
    parts = [meta.get("king", ""), meta.get("year", "")]
    sy = meta.get("solar_year")
    if sy:
        parts.append(f"({sy}년)")
    if meta.get("month"):
        parts.append(meta["month"])
    if meta.get("day"):
        parts.append(meta["day"])
    return " ".join(p for p in parts if p).strip()


def format_context(docs: List[Document]) -> str:
    """검색된 Document 들을 LLM 에 넘길 컨텍스트 문자열로 합친다."""
    blocks = []
    for i, d in enumerate(docs, start=1):
        meta = d.metadata or {}
        typ = meta.get("type", "")
        title = meta.get("title", "")
        period = _format_period(meta)
        blocks.append(
            f"[문서 {i}] (type={typ}) {title} / {period}\n"
            f"{d.page_content}"
        )
    return "\n\n---\n\n".join(blocks)


# --------------------------------------------------------------------------- #
# 3) LLM 호출
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT = """당신은 조선왕조실록 관련 자료(article / daily_summary / monthly_summary / yearly_summary)를
바탕으로 사용자의 질문에 답하는 한국사 전문가입니다.

규칙:
- 반드시 아래 [참고 문서] 안의 정보만 근거로 답하세요.
- 참고 문서에 없는 사실은 추측하지 말고, "참고 문서에서 찾을 수 없습니다." 라고 답하세요.
- 답변은 한국어로, 핵심을 먼저 2~5 문장 정도로 명료하게 답합니다.
- 답변 끝에 사용한 근거 문서 번호를 [문서 N, 문서 M] 형태로 명시하세요."""


def answer(client: OpenAI, model: str, query: str, context: str) -> str:
    user_prompt = (
        f"[참고 문서]\n{context}\n\n"
        f"[질문]\n{query}\n\n"
        f"[지침] 참고 문서를 바탕으로 답하고, 마지막에 사용한 근거 문서 번호를 표시하세요."
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
        max_tokens=1024,
    )
    return resp.choices[0].message.content.strip()


# --------------------------------------------------------------------------- #
# 4) 필터 파싱 (대화형용)
# --------------------------------------------------------------------------- #
def parse_filters(query: str):
    """
    질문 끝에 붙은 옵션을 파싱한다. 예:
        "단종 즉위 --type yearly_summary"
        "지진 발생 --king 단종 --k 3"
    반환: (cleaned_query, filter_dict, k_override)
    """
    filters = {}
    k_override = None

    def take(flag, cast=str):
        nonlocal query
        pat = re.compile(rf"\s--{flag}\s+(\S+)")
        m = pat.search(query)
        if not m:
            return None
        query = pat.sub("", query)
        return cast(m.group(1))

    for flag in ("type", "king", "year", "month", "day"):
        v = take(flag)
        if v is not None:
            filters[flag] = v
    k_override = take("k", int)
    return query.strip(), filters, k_override


# --------------------------------------------------------------------------- #
# 5) 대화형 루프
# --------------------------------------------------------------------------- #
def interactive(vs: FAISS, llm_client: OpenAI, model: str, k: int):
    print("=" * 60)
    print("조선왕조실록 RAG  (종료: 'exit' / 'quit' / 빈 입력 두 번)")
    print("옵션 예시: '계유정난  --type yearly_summary --k 3'")
    print("           '지진      --king 단종 --month 5월'")
    print("=" * 60)

    empty_count = 0
    while True:
        try:
            raw = input("\n질문> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n종료합니다.")
            return

        if raw.lower() in {"exit", "quit"}:
            print("종료합니다.")
            return
        if not raw:
            empty_count += 1
            if empty_count >= 2:
                print("종료합니다.")
                return
            continue
        empty_count = 0

        query, filters, k_override = parse_filters(raw)
        cur_k = k_override or k

        kwargs = {"k": cur_k}
        if filters:
            kwargs["filter"] = filters
            print(f"  (filter={filters}, k={cur_k})")
        else:
            print(f"  (k={cur_k})")

        try:
            docs = vs.similarity_search(query, **kwargs)
        except Exception as e:
            print(f"  ! 검색 실패: {e}")
            continue

        if not docs:
            print("  검색 결과가 없습니다.")
            continue

        print(f"\n[검색된 문서 {len(docs)}건]")
        for i, d in enumerate(docs, start=1):
            meta = d.metadata or {}
            print(f"  {i}. [{meta.get('type','')}] "
                  f"{meta.get('title','')} ({_format_period(meta)})")

        ctx = format_context(docs)
        try:
            ans = answer(llm_client, model, query, ctx)
        except Exception as e:
            print(f"  ! LLM 호출 실패: {e}")
            continue

        print("\n[답변]")
        print(ans)


# --------------------------------------------------------------------------- #
# 6) 엔트리 포인트
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-dir", default="faiss_kfa", type=Path,
                        help="FAISS 인덱스 폴더")
    parser.add_argument("--embedding-model", default="text-embedding-3-small",
                        help="OpenAI 임베딩 모델 (인덱스 빌드 시와 동일해야 함)")
    parser.add_argument("--model", default="gpt-4o-mini",
                        help="답변 생성에 사용할 OpenAI 모델")
    parser.add_argument("--k", default=5, type=int,
                        help="기본 top-k 검색 개수")
    args = parser.parse_args()

    if not args.index_dir.exists():
        print(f"인덱스 폴더 없음: {args.index_dir}")
        sys.exit(1)

    print(f"FAISS 인덱스 로드 중: {args.index_dir}")
    vs = load_vectorstore(args.index_dir, args.embedding_model)
    print(f"  - 문서 수: {vs.index.ntotal}")

    interactive(vs, client, args.model, args.k)


if __name__ == "__main__":
    main()