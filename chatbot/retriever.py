"""
retriever.py
------------
FAISS 벡터스토어 로드 + CoT plan 기반 메타데이터 필터 검색.

핵심: '점진적 완화(progressive relaxation)' 전략.
  좁은 필터(연-월-일 + type) 로 먼저 찾고, 결과가 부족하면
  단계적으로 조건을 풀어 최종적으로는 순수 의미검색까지 폴백한다.
이렇게 하면 날짜 추정이 약간 어긋나도 빈 결과가 나오지 않는다.
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from metadata_utils import (
    build_filter, months_in_range, day_num, lunar_month_key,
)

EMBEDDING_MODEL = "text-embedding-3-large"

SCRIPT_DIR = Path(__file__).parent.resolve()      # doyoon_project 폴더
ROOT_DIR = SCRIPT_DIR.parent                       # SillokChatbot 루트 폴더
DEFAULT_INDEX_DIR = ROOT_DIR / "faiss"             # 루트의 faiss 폴더

# 연도별로 실제 존재하는 음력 월 목록 캐시 (월 범위 검색에 사용)
_YEAR_MONTHS_CACHE: Dict[int, List[str]] = {}


def _year_months(vs: FAISS, year: int) -> List[str]:
    """해당 양력연도에 존재하는 음력 월 목록(윤달 포함)을 인덱스에서 수집."""
    if not _YEAR_MONTHS_CACHE:
        # docstore 전체를 한 번 훑어 연->월 목록을 만든다 (최초 1회)
        try:
            store = vs.docstore._dict  # langchain-community FAISS 내부 docstore
            tmp: Dict[int, set] = {}
            for doc in store.values():
                m = doc.metadata
                sy, mo = m.get("solar_year"), m.get("month")
                if sy is not None and mo:
                    tmp.setdefault(int(sy), set()).add(mo)
            for y, mset in tmp.items():
                _YEAR_MONTHS_CACHE[y] = sorted(mset, key=lunar_month_key)
        except Exception:
            pass
    return _YEAR_MONTHS_CACHE.get(int(year), [])


def load_vectorstore(faiss_dir=None) -> FAISS:
    # faiss_dir 미지정 시 스크립트 위치 기준 faiss 폴더 사용
    faiss_dir = Path(faiss_dir) if faiss_dir else DEFAULT_INDEX_DIR
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    return FAISS.load_local(
        str(faiss_dir), embeddings, allow_dangerous_deserialization=True
    )


def _search(vs: FAISS, query: str, k: int, flt: Dict[str, Any],
            fetch_k: int = 5000) -> List[Document]:
    """
    FAISS 메타데이터 필터 검색.

    중요: LangChain-FAISS의 similarity_search 는 먼저 벡터 기준 상위 fetch_k 개를
    가져온 뒤 그 안에서 filter 를 적용한다. fetch_k 기본값(20)이 너무 작으면,
    필터에 맞는 문서가 상위 20개 안에 없어 결과가 0건이 되고 만다.
    그래서 필터를 쓸 때는 fetch_k 를 크게 잡아 충분한 후보를 확보한다.
    """
    flt = {k_: v for k_, v in (flt or {}).items() if v is not None}
    if flt:
        # 필터가 있으면 넉넉히 prefetch 한 뒤 필터링
        return vs.similarity_search(query, k=k, filter=flt, fetch_k=fetch_k)
    return vs.similarity_search(query, k=k)


def retrieve(vs: FAISS, plan: Dict[str, Any], k: int = 6) -> List[Document]:
    """plan에 따라 점진적으로 필터를 완화하며 문서를 검색."""
    query = plan.get("search_query") or ""
    king = plan.get("king")
    solar_year = plan.get("solar_year")
    solar_year_range = plan.get("solar_year_range")
    month = plan.get("month")
    month_range = plan.get("month_range")   # 같은 연도 내 음력 월 범위 [시작월, 끝월]
    day = plan.get("day")
    day_range = plan.get("day_range")       # 같은 월 내 일 범위 [시작일, 끝일]
    doc_types = plan.get("doc_types") or []
    primary_type = doc_types[0] if doc_types else None

    seen = set()
    collected: List[Document] = []

    def add(docs: List[Document]):
        for d in docs:
            key = (
                d.metadata.get("title"),
                d.metadata.get("solar_year"),
                d.metadata.get("month"),
                d.metadata.get("day"),
                d.metadata.get("idx"),
                d.metadata.get("chunk_id"),
            )
            if key not in seen:
                seen.add(key)
                collected.append(d)

    # 0-a) 같은 연도 내 '월 범위' 또는 '일 범위' 질문 처리.
    #      FAISS filter 는 범위 비교를 못 하므로, 범위에 드는 (월[,일]) 조합을
    #      구체값으로 풀어 각각 검색한다. 윤달도 자동 포함된다.
    if solar_year and (month_range or (month and day_range)):
        # 검색 대상 월 목록 결정
        if month_range:
            avail = _year_months(vs, solar_year)
            target_months = months_in_range(avail, month_range[0], month_range[1])
            if not target_months:  # 인덱스에서 못 받으면 숫자 범위로라도
                lo, hi = sorted(int(x) for x in month_range)
                target_months = [f"{n}월" for n in range(lo, hi + 1)]
        else:
            target_months = [month]

        # 일 범위가 있으면 각 월 안에서 해당 일들만, 없으면 월 전체
        day_list = None
        if day_range:
            lo, hi = sorted(int(x) for x in day_range)
            day_list = [f"{d}일" for d in range(lo, hi + 1)]

        per_slot = max(3, k)  # 월별로 넉넉히 받아두고 라운드로빈으로 병합
        per_month_docs: Dict[str, List[Document]] = {}
        for mo in target_months:
            mdocs: List[Document] = []
            try:
                if day_list:
                    for dd in day_list:
                        mdocs += _search(vs, query, k=per_slot,
                                         flt=build_filter(king=king, solar_year=solar_year,
                                                          month=mo, day=dd,
                                                          doc_type=primary_type))
                else:
                    mdocs += _search(vs, query, k=per_slot,
                                     flt=build_filter(king=king, solar_year=solar_year,
                                                      month=mo, doc_type=primary_type))
                    # king 없이도 보강 (양위 전후 혼재 대비)
                    mdocs += _search(vs, query, k=per_slot,
                                     flt=build_filter(solar_year=solar_year, month=mo))
            except Exception:
                pass
            per_month_docs[mo] = mdocs
        # 라운드로빈으로 월별 균형 있게 병합
        idx = 0
        while len(collected) < k and any(idx < len(per_month_docs[m]) for m in target_months):
            for mo in target_months:
                if idx < len(per_month_docs[mo]):
                    add([per_month_docs[mo][idx]])
                if len(collected) >= k:
                    break
            idx += 1
        if len(collected) >= k:
            return collected[:k]

    # 0-b) 연도 '범위' 질문: 범위 내 각 연도를 따로 검색해 합친다.
    #    (FAISS filter 는 연도 범위 비교를 직접 지원하지 않으므로 연도별로 나눠 검색)
    if solar_year_range and not solar_year:
        start, end = solar_year_range
        years = list(range(start, end + 1))
        per_year = max(3, k)  # 연도별로 넉넉히 받아 두고 아래서 골고루 병합
        per_year_docs: Dict[int, List[Document]] = {}
        for yr in years:
            ydocs: List[Document] = []
            try:
                if king:
                    ydocs += _search(vs, query, k=per_year,
                                     flt=build_filter(king=king, solar_year=yr,
                                                      doc_type=primary_type))
                ydocs += _search(vs, query, k=per_year,
                                 flt=build_filter(solar_year=yr, doc_type=primary_type))
                ydocs += _search(vs, query, k=per_year,
                                 flt=build_filter(solar_year=yr))
            except Exception:
                pass
            per_year_docs[yr] = ydocs
        # 라운드로빈으로 연도별 균형 있게 병합
        idx = 0
        while len(collected) < k and any(idx < len(per_year_docs[y]) for y in years):
            for yr in years:
                if idx < len(per_year_docs[yr]):
                    add([per_year_docs[yr][idx]])
                if len(collected) >= k:
                    break
            idx += 1
        if len(collected) >= k:
            return collected[:k]

    # 완화 단계들: 좁은 것 -> 넓은 것
    attempts: List[Dict[str, Any]] = []

    # 1) 연-월-일 + 타입
    if solar_year and month and day:
        attempts.append(build_filter(king=king, solar_year=solar_year,
                                     month=month, day=day, doc_type=primary_type))
        attempts.append(build_filter(king=king, solar_year=solar_year,
                                     month=month, day=day))
        # king 없이 날짜만 (king 추정이 틀렸을 경우 대비)
        attempts.append(build_filter(solar_year=solar_year, month=month, day=day))
    # 2) 연-월 + 타입
    if solar_year and month:
        attempts.append(build_filter(king=king, solar_year=solar_year,
                                     month=month, doc_type=primary_type))
        attempts.append(build_filter(king=king, solar_year=solar_year, month=month))
        attempts.append(build_filter(solar_year=solar_year, month=month))
    # 3) 연 + 타입
    if solar_year:
        attempts.append(build_filter(king=king, solar_year=solar_year,
                                     doc_type=primary_type))
        attempts.append(build_filter(king=king, solar_year=solar_year))
        # king 없이 연도만 (양위 전후로 왕대가 갈리는 사건 대비)
        attempts.append(build_filter(solar_year=solar_year))
    # 4) 왕만
    if king:
        attempts.append(build_filter(king=king))
    # 5) 무필터(순수 의미검색)
    attempts.append({})

    for flt in attempts:
        if len(collected) >= k:
            break
        try:
            add(_search(vs, query, k=k, flt=flt))
        except Exception:
            # 이 필터 조합이 실패하면 조용히 다음(더 느슨한) 단계로 넘어간다.
            continue

    return collected[:k]


def format_context(docs: List[Document]) -> str:
    """검색 문서들을 LLM 프롬프트용 컨텍스트 문자열로 변환."""
    blocks = []
    for i, d in enumerate(docs, 1):
        m = d.metadata
        date = f"{m.get('king','')} {m.get('year','')}({m.get('solar_year','')}) " \
               f"{m.get('month','') or ''} {m.get('day','') or ''}".strip()
        blocks.append(
            f"[자료 {i}] (유형:{m.get('type')}, 날짜:{date}, 제목:{m.get('title')})\n"
            f"{d.page_content}"
        )
    return "\n\n".join(blocks)