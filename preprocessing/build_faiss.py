"""
kfa.jsonl (article + daily/monthly/yearly summary 모두 합쳐진 파일) 을 읽어
LangChain Document 로 변환하고, OpenAI text-embedding-3-small 로 임베딩하여
FAISS 벡터스토어를 구축한 뒤 절대 경로의 faiss 폴더에 저장한다.

사용법:
    export OPENAI_API_KEY=...
    cd preprocessing
    python build_faiss.py
    # 또는 명시적으로 절대 경로 지정:
    # python build_faiss.py --input-file D:\\path\\to\\data.jsonl --index-dir D:\\path\\to\\faiss
"""

import argparse
import json
import os
from pathlib import Path
from typing import List

# 스크립트 위치 기반 프로젝트 루트 경로 계산
SCRIPT_DIR = Path(__file__).parent.resolve()      # preprocessing 폴더의 절대 경로
PROJECT_ROOT = SCRIPT_DIR.parent                   # SillokChatbot 폴더
DEFAULT_INPUT_FILE = PROJECT_ROOT / "data" / "data.jsonl"
DEFAULT_INDEX_DIR = PROJECT_ROOT / "faiss"

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

# --------------------------------------------------------------------------- #
# 1) jsonl -> Document 리스트
# --------------------------------------------------------------------------- #
def load_documents(path: Path) -> List[Document]:
    """kfa.jsonl 의 각 줄을 LangChain Document 로 변환."""
    docs: List[Document] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  ! {line_no}번째 줄 JSON 파싱 실패, 스킵: {e}")
                continue
            page_content = obj.get("page_content", "")
            metadata = obj.get("metadata", {}) or {}
            docs.append(Document(page_content=page_content, metadata=metadata))
    return docs


def summarize_types(docs: List[Document]) -> None:
    """type 별 개수를 출력한다."""
    from collections import Counter
    c = Counter(d.metadata.get("type", "unknown") for d in docs)
    print("  - Document type 분포:")
    for k, v in c.most_common():
        print(f"      {k}: {v}")


# --------------------------------------------------------------------------- #
# 2) FAISS 인덱스 구축
# --------------------------------------------------------------------------- #
def build_faiss(
    docs: List[Document],
    embedding_model: str,
    batch_size: int,
) -> FAISS:
    """배치 단위로 Document 를 임베딩하여 FAISS 벡터스토어를 만든다."""
    embeddings = OpenAIEmbeddings(model=embedding_model)

    if not docs:
        raise ValueError("입력 Document 가 없습니다.")

    # 첫 배치로 vectorstore 초기화 후 나머지를 add_documents 로 누적
    first = docs[: batch_size]
    print(f"  - 첫 배치({len(first)})로 FAISS 초기화...")
    vs = FAISS.from_documents(first, embeddings)

    rest = docs[batch_size:]
    if rest:
        total_batches = (len(rest) + batch_size - 1) // batch_size
        for bi in range(total_batches):
            batch = rest[bi * batch_size : (bi + 1) * batch_size]
            vs.add_documents(batch)
            done = batch_size + (bi + 1) * batch_size
            done = min(done, len(docs))
            print(f"    · batch {bi + 2}/{total_batches + 1} 추가 ({done}/{len(docs)})")

    return vs


# --------------------------------------------------------------------------- #
# 3) 엔트리 포인트
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", default=str(DEFAULT_INPUT_FILE), type=Path,
                        help="합쳐진 입력 jsonl 파일")
    parser.add_argument("--index-dir", default=str(DEFAULT_INDEX_DIR), type=Path,
                        help="FAISS 인덱스 저장 폴더")
    parser.add_argument("--embedding-model", default="text-embedding-3-large",
                        help="OpenAI 임베딩 모델명")
    parser.add_argument("--batch-size", default=200, type=int,
                        help="임베딩 요청 배치 크기")
    args = parser.parse_args()

    if not args.input_file.exists():
        print(f"입력 파일 없음: {args.input_file}")
        return

    print(f"[{args.input_file}] 로드 중...")
    docs = load_documents(args.input_file)
    print(f"  - 총 {len(docs)}개 Document 로드됨")
    summarize_types(docs)

    print("\n[FAISS 인덱스 구축]")
    vs = build_faiss(docs, args.embedding_model, args.batch_size)

    args.index_dir.parent.mkdir(parents=True, exist_ok=True)
    vs.save_local(str(args.index_dir))
    print(f"\n완료 -> {args.index_dir}")
    print("  파일:")
    for p in sorted(args.index_dir.glob('*')):
        print(f"    {p.name}  ({p.stat().st_size:,} bytes)")

    # 간단한 동작 확인용 샘플 검색 (원하지 않으면 주석 처리 가능)
    # sample_query = "단종 즉위"
    # print(f"\n[샘플 검색] query={sample_query!r}")
    # hits = vs.similarity_search(sample_query, k=3)
    # for i, d in enumerate(hits, start=1):
    #     title = d.metadata.get("title", "")
    #     typ = d.metadata.get("type", "")
    #     period = (f"{d.metadata.get('king','')} {d.metadata.get('year','')} "
    #               f"{d.metadata.get('month','') or ''} {d.metadata.get('day','') or ''}").strip()
    #     print(f"  {i}. [{typ}] {title} ({period})")


if __name__ == "__main__":
    main()