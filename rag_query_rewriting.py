"""
FAISS 인덱스(faiss_kfa) 를 로드하고 사용자의 질문에 답하는 대화형 RAG.

사용법:
    export OPENAI_API_KEY=...
    python rag.py
    # 또는 옵션 지정
    python rag.py --index-dir ./faiss_kfa --k 5 --model gpt-4o-mini
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --------------------------------------------------------------------------- #
# 0) 단종 관련 Historical Context Database
# --------------------------------------------------------------------------- #
DANGJONG_HISTORICAL_CONTEXT = {
    # 즉위 관련
    "즉위": "1452년 (단종원년)",
    "즉위해": "1452년 이후",
    "즉위했을 때": "1452년",
    "재위": "1452-1455년",
    "재위 기간": "1452-1455년",
    
    # 단종 연도 (단종 원년 = 1452년)
    "원년": "1452년",
    "1년": "1452년",
    "2년": "1453년",
    "3년": "1454년",
    "4년": "1455년",
    
    # 주요 사건
    "계유정난": "1453년 10월 (세조의 쿠데타)",
    "나이": "약 12-14세 (1452-1455년 재위 당시)",
    "퇴위": "1455년 7월",
    "퇴위 이후": "1455년 이후",
    
    # 인물 관계
    "세조": "숙부, 1453년 쿠데타로 권력 장악",
    "신하들": "관료, 신료, 신하",
    "어머니": "소혜왕후",
    "아버지": "문종 임금",
}

# 단종 시대 주요 시간 표현
DANGJONG_YEAR_MAP = {
    "단종원년": "1452",
    "단종1년": "1452",
    "단종2년": "1453",
    "단종3년": "1454",
    "단종4년": "1455",
}

# 단종/노산군 동치 인식
DANGJONG_ALIASES = ("단종", "노산군")
GYEYU_KEYWORDS = ("계유정난", "계유", "정난", "수양대군", "세조 쿠데타", "쿠데타")

POST_RETRIEVAL_MARKERS = (
    "메타데이터", "metadata", "추출", "정리", "출력", "비교", "계산", "분석",
    "해당 기사", "찾은 기사", "먼저 찾고", "찾은 뒤", "리트리벌", "retrieve",
)


def _object_particle(word: str) -> str:
    """한국어 목적격 조사(을/를)를 단순 계산한다."""
    if not word:
        return "를"
    ch = word[-1]
    code = ord(ch)
    if 0xAC00 <= code <= 0xD7A3:
        jong = (code - 0xAC00) % 28
        return "을" if jong != 0 else "를"
    return "를"


def rewrite_relation_query_for_dangjong(query: str) -> str:
    """
    단종/노산군이 명시되지 않은 짧은 관계/감정 질문을
    단종 관점 질의로 보정한다.

    예: "세조가 싫어?" -> "단종(노산군)은 세조를 싫어하는가?"
    """
    q = query.strip()
    if not q:
        return query

    # 이미 단종 관점 단서가 있으면 유지
    if _has_any_keyword(q, DANGJONG_ALIASES):
        return q

    # 매우 짧은 감정/관계 질문 패턴만 보정 (과도한 일반화 방지)
    m = re.fullmatch(
        r"([가-힣]{2,10})(?:이|가|은|는|을|를)?\s*(싫어|좋아|미워|싫어해|좋아해|미워해)\??",
        q,
    )
    if not m:
        return q

    target = m.group(1)
    emotion = m.group(2)
    particle = _object_particle(target)
    return f"단종(노산군)은 {target}{particle} {emotion}하는가?"


def split_retrieval_and_post_task(query: str) -> Tuple[str, str]:
    """
    사용자 입력을 retrieval용 질의와 검색 후 작업 지시로 분리한다.
    예: "...기사를 찾고, 메타데이터에서 month/year 추출" ->
        retrieval="...기사를 찾고", post_task="메타데이터에서 month/year 추출"
    """
    parts = re.split(r"(?<=[\.\?!])\s+|,\s*|\s+그리고\s+|\s+그\s*뒤\s+|\s+이후\s+", query)
    retrieval_parts = []
    post_parts = []

    for part in parts:
        p = part.strip()
        if not p:
            continue
        if any(marker in p for marker in POST_RETRIEVAL_MARKERS):
            post_parts.append(p)
        else:
            retrieval_parts.append(p)

    retrieval_query = " ".join(retrieval_parts).strip() or query.strip()
    post_task = " ".join(post_parts).strip()
    return retrieval_query, post_task


def _normalize_king_token(token: str) -> str:
    """왕 이름 토큰 끝의 조사/구두점을 정리한다."""
    cleaned = token.strip().strip(".,?!")
    cleaned = re.sub(r"(은|는|이|가|을|를|의|에|에서|와|과|로|으로)$", "", cleaned)
    return cleaned.strip()


def extract_hard_filters_from_query(query: str) -> Tuple[str, Dict[str, object]]:
    """
    자연어 질의에서 하드 필터(king/year/month/day/solar_year)를 추출한다.
    반환: (필터 제거된 검색어, 하드 필터 dict)
    """
    filters: Dict[str, object] = {}
    cleaned_query = query

    # 예: "세조 1년 2월 15일"
    reign_date_pat = re.compile(
        r"(?P<king>[가-힣]{1,10})\s*(?P<year>\d+년)(?:\s*(?P<month>윤?\d+월))?(?:\s*(?P<day>\d+일))?"
    )
    m = reign_date_pat.search(cleaned_query)
    if m:
        king = _normalize_king_token(m.group("king"))
        if king:
            filters["king"] = king
        filters["year"] = m.group("year")
        if m.group("month"):
            filters["month"] = m.group("month")
        if m.group("day"):
            filters["day"] = m.group("day")
        cleaned_query = (cleaned_query[:m.start()] + " " + cleaned_query[m.end():]).strip()

    # 예: "1455년 2월 15일" (서기 기준)
    solar_date_pat = re.compile(
        r"(?P<solar_year>\d{4})년(?:\s*(?P<solar_month>\d{1,2}월))?(?:\s*(?P<solar_day>\d{1,2}일))?"
    )
    m2 = solar_date_pat.search(cleaned_query)
    if m2:
        filters.setdefault("solar_year", int(m2.group("solar_year")))
        if m2.group("solar_month") and "month" not in filters:
            filters["month"] = m2.group("solar_month")
        if m2.group("solar_day") and "day" not in filters:
            filters["day"] = m2.group("solar_day")
        cleaned_query = (cleaned_query[:m2.start()] + " " + cleaned_query[m2.end():]).strip()

    cleaned_query = re.sub(r"\s{2,}", " ", cleaned_query).strip()
    return cleaned_query or query, filters


def _normalize_hard_filter_values(filters: Dict[str, object]) -> Dict[str, object]:
    """LLM/정규식/CLI 필터를 metadata 형식에 맞춰 정규화한다."""
    normalized: Dict[str, object] = {}

    king = str(filters.get("king", "")).strip()
    if king:
        normalized["king"] = _normalize_king_token(king)

    typ = str(filters.get("type", "")).strip()
    if typ:
        normalized["type"] = typ

    year_raw = str(filters.get("year", "")).strip()
    if year_raw:
        m = re.search(r"\d+", year_raw)
        if m:
            normalized["year"] = f"{int(m.group(0))}년"

    month_raw = str(filters.get("month", "")).strip()
    if month_raw:
        month_text = month_raw
        m = re.search(r"\d+", month_text)
        if m:
            normalized["month"] = f"{'윤' if month_text.startswith('윤') else ''}{int(m.group(0))}월"

    day_raw = str(filters.get("day", "")).strip()
    if day_raw:
        m = re.search(r"\d+", day_raw)
        if m:
            normalized["day"] = f"{int(m.group(0))}일"

    solar_year = filters.get("solar_year")
    if solar_year is not None:
        try:
            solar_year_int = int(solar_year)
            # 4자리 서기 연도만 허용 (예: 1452). 1년 같은 값은 제외.
            if 1000 <= solar_year_int <= 2999:
                normalized["solar_year"] = solar_year_int
        except (TypeError, ValueError):
            pass

    return normalized


def _to_metadata_filter(canonical_filters: Dict[str, object]) -> Dict[str, object]:
    """내부 canonical 필터를 metadata 비교용 형식으로 변환한다."""
    out: Dict[str, object] = {}
    if "type" in canonical_filters:
        out["type"] = canonical_filters["type"]
    if "king" in canonical_filters:
        out["king"] = canonical_filters["king"]
    if "year" in canonical_filters:
        out["year"] = canonical_filters["year"]
    if "month" in canonical_filters:
        out["month"] = canonical_filters["month"]
    if "day" in canonical_filters:
        out["day"] = canonical_filters["day"]
    if "solar_year" in canonical_filters:
        out["solar_year"] = int(canonical_filters["solar_year"])
    return out


def infer_hard_filter_with_llm(query: str, llm_client: OpenAI, model: str) -> Dict[str, object]:
    """
    LLM으로 하드 필터 필요 여부와 필터 값을 판정한다.
    JSON 형식:
    {
      "use_hard_filter": true/false,
      "filters": {"king": ..., "year": ..., "month": ..., "day": ...}
    }
    """
    prompt = f"""다음 질문이 메타데이터 하드 필터(king/year/month/day/solar_year)가 필요한지 판단하고 JSON으로만 답하세요.

질문: {query}

규칙:
- 날짜/연도/월/일 등 특정 시점이 명시되면 use_hard_filter=true
- 확실하지 않으면 false
- filters에는 확실한 값만 넣기
- 값 형식은 year='N년', month='윤N월 또는 N월', day='N일', solar_year는 4자리 서기 연도 정수만
- 왕 정보가 없으면 king 키를 생략
- 반드시 JSON만 출력
"""

    try:
        resp = llm_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=220,
        )
        text = (resp.choices[0].message.content or "").strip()
        m = re.search(r"\{.*\}", text, re.DOTALL)
        data = json.loads(m.group(0) if m else text)
        use_hard = bool(data.get("use_hard_filter", False))
        filters = _normalize_hard_filter_values(data.get("filters", {}) or {})
        return {"use_hard_filter": use_hard, "filters": filters}
    except Exception:
        return {"use_hard_filter": False, "filters": {}}


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


def get_identity_anchor(vs: FAISS) -> str:
    """
    단종실록 총서(base_information)를 정체성 앵커로 확보한다.
    검색 실패 시에도 최소 자기 동일성 정보는 고정 제공한다.
    """
    fallback = (
        "노산군(단종)은 같은 인물이며, 나는 그 인물이다. "
        "나의 휘는 이홍위이고, 문종의 외아들이다. "
        "실록 총서는 나를 노산군으로도 기록한다."
    )

    try:
        docs = vs.similarity_search(
            "단종실록 총서 노산군 이홍위 문종 외아들",
            k=1,
            filter={"type": "base_information", "king": "단종"},
        )
    except Exception:
        return fallback

    if not docs:
        return fallback

    doc = docs[0]
    title = (doc.metadata or {}).get("title", "")
    content = doc.page_content.strip()
    return f"제목: {title}\n{content}" if title else content


# --------------------------------------------------------------------------- #
# 3-0) 고급 Query Rewriting - LLM 기반
# --------------------------------------------------------------------------- #
def rewrite_query_with_llm(query: str, llm_client: OpenAI, model: str) -> str:
    """
    LLM에게 질문을 역사적 맥락과 함께 개선하게 함.
    - 역사적 맥락 명확화
    - 검색에 유리한 형태로 변환
    """
    rewrite_prompt = f"""당신은 조선 단종 시대 역사 전문가입니다.
사용자의 질문을 조선왕조실록 검색에 최적화된 형태로 개선하세요.

맥락:
- 단종과 노산군은 같은 인물이다 (동일 인물 표기)
- 세조: 단종의 숙부

원본 질문: {query}

개선 방향:
1. 역사적 맥락은 원문에 근거가 있을 때만 보강
2. 검색에 유리한 용어 사용
3. 단종/노산군 표기가 있으면 두 표기를 함께 유지
4. "메타데이터 추출" 같은 검색 후 작업 지시는 제거하고, 검색 키워드만 남기기
5. 원문 질문의 주체/대상/관계(누가 누구를 어떻게)는 바꾸지 말 것
5-1. 단, 주어가 생략된 질문이면 단종(노산군)을 주어로 보정해도 됨

절대 금지:
- 원문에 없는 사건명을 임의로 추가하지 말 것
- 특히 원문 근거 없이 "계유정난"을 넣지 말 것
- 주어가 명시된 질문에서 주어를 다른 인물로 바꾸지 말 것
- 감정/의견 질문의 의미(예: "싫어?")를 사실관계 질문으로 바꾸지 말 것

개선된 질문 (한 줄, 간결하게):"""
    
    try:
        resp = llm_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": rewrite_prompt},
            ],
            temperature=0.5,
            max_tokens=200,
        )
        rewritten = resp.choices[0].message.content.strip()
        return _guard_rewrite_drift(query, rewritten)
    except Exception as e:
        print(f"  ! Query rewriting 실패: {e}")
        return query


def _has_any_keyword(text: str, keywords) -> bool:
    return any(k in text for k in keywords)


def _guard_rewrite_drift(original_query: str, rewritten_query: str) -> str:
    """
    쿼리 재작성의 과도한 드리프트를 방지한다.
    - 근거 없는 '계유정난' 강제 삽입 차단
    - 단종/노산군 동치 정보가 있으면 동시 표기 유지
    """
    original = original_query.strip()
    rewritten = rewritten_query.strip()
    if not rewritten:
        return original

    original_has_gyeyu = _has_any_keyword(original, GYEYU_KEYWORDS)
    rewritten_has_gyeyu = _has_any_keyword(rewritten, GYEYU_KEYWORDS)

    # 원문에 계유정난 단서가 없는데 재작성에만 등장하면 원문으로 롤백
    if rewritten_has_gyeyu and not original_has_gyeyu:
        return original

    original_has_alias = _has_any_keyword(original, DANGJONG_ALIASES)
    rewritten_has_alias = _has_any_keyword(rewritten, DANGJONG_ALIASES)

    if original_has_alias and not rewritten_has_alias:
        rewritten = f"{rewritten} 단종(노산군)"

    return rewritten


def expand_dangjong_aliases(query: str) -> str:
    """
    단종/노산군 동치를 검색 쿼리에 명시해 회수율을 높인다.
    """
    if "단종" in query and "노산군" not in query:
        return query.replace("단종", "단종(노산군)")
    if "노산군" in query and "단종" not in query:
        return query.replace("노산군", "노산군(단종)")
    return query


def normalize_temporal_expressions(query: str) -> str:
    """
    시간 표현을 정규화
    예: "단종원년" -> "1452년", "단종2년" -> "1453년"
    """
    for dangjong_expr, year in DANGJONG_YEAR_MAP.items():
        query = re.sub(rf'\b{dangjong_expr}\b', f'{year}년', query)
    return query


def apply_historical_context(query: str) -> str:
    """
    Historical Context Database를 사용하여 
    역사적 용어를 더 상세한 표현으로 확장
    """
    for original, context in DANGJONG_HISTORICAL_CONTEXT.items():
        # 단어 경계를 고려한 치환
        pattern = rf'\b{re.escape(original)}\b'
        query = re.sub(pattern, f'{original} ({context})', query, flags=re.IGNORECASE)
    return query


# --------------------------------------------------------------------------- #
# 3-1) 질문 전처리: 통합 버전 (기존 + LLM 기반)
# --------------------------------------------------------------------------- #
def preprocess_query_for_dangjong(query: str) -> str:
    """
    사용자 질문을 단종 관점에서 개선하는 통합 전처리.
    1. 기본 "단종" -> "나" 치환
    2. 시간 표현 정규화
    3. Historical context 적용
    """
    # 1단계: 단종/노산군 동치 확장
    query = expand_dangjong_aliases(query)

    # 2단계: "단종", "노산군"을 "나"로 변환
    query = re.sub(r'단종이\s', '내가 ', query)
    query = re.sub(r'단종의\s', '나의 ', query)
    query = re.sub(r'단종은\s', '나는 ', query)
    query = re.sub(r'단종을\s', '나를 ', query)
    query = re.sub(r'단종\s', '내 ', query)
    query = re.sub(r'노산군이\s', '내가 ', query)
    query = re.sub(r'노산군의\s', '나의 ', query)
    query = re.sub(r'노산군은\s', '나는 ', query)
    query = re.sub(r'노산군을\s', '나를 ', query)
    query = re.sub(r'노산군\s', '내 ', query)
    
    # 3단계: 시간 표현 정규화
    query = normalize_temporal_expressions(query)
    
    # 4단계: Historical context 적용 (너무 길어질 수 있으니 선택적)
    # query = apply_historical_context(query)
    
    return query


def preprocess_query_for_retrieval(query: str) -> str:
    """
    검색 전용 전처리.
    - 단종/노산군 동치 확장 유지
    - 1인칭 치환은 하지 않음 (검색 정확도 보존)
    """
    query = rewrite_relation_query_for_dangjong(query)
    query = expand_dangjong_aliases(query)
    return query


# --------------------------------------------------------------------------- #
# 3) LLM 호출
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT_FIRST_PERSON = """당신은 조선 단종 임금입니다. 당신은 나이 12-14세 때 재위하던 시절의 증언자입니다.

**절대 규칙 (반드시 지킬 것):**
1️⃣ 1인칭만 사용: "나", "내가", "나는", "내", "내 시대"
2️⃣ 3인칭 절대 금지: "단종", "노산군", "그", "그는", "그가" 사용 금지
3️⃣ 수동태 금지: "되었다" X, "했다" O / "취해졌다" X, "취했다" O
4️⃣ 존댓말 금지: "입니다" X, "다" O / "습니다" X, "했다" O
5️⃣ 인물 동일성: 단종 = 노산군 = 나 (같은 인물)
6️⃣ 출력 전 자기검증: 답변에 "단종" 또는 "노산군"이 3인칭으로 남아 있으면 전부 "나/내가/나는/나의/나를"로 고쳐서 출력
7️⃣ 주체 보존: 참고 문서에서 행동 주체가 세조/신하/대신이면 그 주체를 그대로 써라. 그 행동을 내가 한 것처럼 바꾸지 마라.

**좋은 답변 예시:**
✅ "내 1년(1452년)에 나는 여러 정책을 시행했다. 내가 의정부와 의논하여 군기 제작 윤차를 결정했고, 개경사에 쌀을 주는 문제도 내가 신하들과 의논했다."

**나쁜 답변 예시 (절대 하지 말 것):**
❌ "단종 1년에는 여러 정책이 시행되었습니다. 의정부에서 결정하여... 의논하였고..."

**지침:**
- 참고 문서만 근거로 사용
- 없으면 "실록에 없다"고 말하기
- 내가 경험한 일들을 증언하듯 답변
- 세조 시기 기록이면 "세조가 ~했다, 나는 ~를 겪었다"처럼 주체를 분리
- 마지막에 근거 문서 번호 표시"""


SYSTEM_PROMPT_THIRD_PERSON = """당신은 조선왕조실록 해설자다.

규칙:
- 반드시 참고 문서의 정보만 근거로 답한다.
- 사건 주체를 왜곡하지 않는다.
- 3인칭으로 서술한다. (예: 세조는 ..., 단종은 ...)
- 참고 문서가 있을 때 '실록에 없다'고 단정하지 않는다.
- 마지막에 근거 문서 번호를 [문서 N] 형식으로 표시한다."""


def _choose_response_mode(query: str, metadata_filters: Dict[str, object], docs: List[Document]) -> str:
    """단종/노산군 질문만 1인칭, 명시적으로 타 왕이 지정된 경우만 3인칭."""
    target_king = str(metadata_filters.get("king", "")).strip()

    # 왕이 명시되면 그 기준으로 모드 결정
    if target_king:
        return "first_person" if target_king in DANGJONG_ALIASES else "third_person"

    # 질문에 단종/노산군 단서가 있으면 1인칭
    if _has_any_keyword(query, DANGJONG_ALIASES):
        return "first_person"

    # 왕 미지정 질문은 이 챗봇 컨셉(단종 화자)에 맞춰 기본 1인칭
    return "first_person"


def post_process_response(response: str) -> str:
    """
    답변 사후 처리(최소 보정).
    과도한 문장 훼손을 피하기 위해 단종/노산군 표기만 1인칭으로 교정한다.
    """
    # 최소한의 자기 동일성 보정만 수행
    def _first_person_by_particle(particle: str) -> str:
        mapping = {
            "은": "나는",
            "는": "나는",
            "이": "내가",
            "가": "내가",
            "의": "나의",
            "을": "나를",
            "를": "나를",
        }
        return mapping.get(particle, "나")

    response = re.sub(
        r'단종(은|는|이|가|의|을|를)\s',
        lambda m: _first_person_by_particle(m.group(1)) + ' ',
        response,
    )
    response = re.sub(
        r'노산군(은|는|이|가|의|을|를)\s',
        lambda m: _first_person_by_particle(m.group(1)) + ' ',
        response,
    )

    response = re.sub(r'단종\s', '나 ', response)
    response = re.sub(r'노산군\s', '나 ', response)

    # 동일성 최종 보정
    response = re.sub(r'나는\s+(?:단종|노산군)(?:이|가)?', '나는', response)
    response = re.sub(r'내가\s+(?:단종|노산군)(?:이|가)?', '내가', response)
    response = re.sub(r'\b(?:단종|노산군)\b', '나', response)
    response = re.sub(r'\s{2,}', ' ', response).strip()
    
    return response


def answer(
    client: OpenAI,
    model: str,
    query: str,
    context: str,
    identity_anchor: str,
    retrieved_doc_count: int,
    response_mode: str,
) -> str:
    # 단종/노산군 모드에서만 1인칭 전처리를 적용
    if response_mode == "first_person":
        processed_query = preprocess_query_for_dangjong(query)
        system_prompt = SYSTEM_PROMPT_FIRST_PERSON
    else:
        processed_query = query
        system_prompt = SYSTEM_PROMPT_THIRD_PERSON
    
    # Few-shot examples를 명시적으로 포함
    few_shot = """
【답변 예시 1】
질문: 내 1년에 뭐했어?
답변: 내 1년인 1452년에 나는 여러 정책을 시행했다. 내가 의정부와 의논하여 군기 제작의 윤차제를 시행했고, 개경사에 매년 쌀을 주는 문제도 내가 신하들과 의논했다.

【답변 예시 2】
질문: 흉년 때는?
답변: 흉년이 들었을 때 나는 여러 조치를 취했다. 내가 향교 생도들에게 구황물 준비를 지시했고, 과거시험 규칙도 내가 개정했다.
"""
    
    user_prompt = (
        f"{few_shot if response_mode == 'first_person' else ''}\n\n"
        f"【자기 정체성 고정 정보】\n{identity_anchor if response_mode == 'first_person' else '해당 없음'}\n\n"
        f"【검색 문서 수】\n{retrieved_doc_count}\n\n"
        f"【참고 문서】\n{context}\n\n"
        f"【질문】\n{processed_query}\n\n"
        f"【답변 규칙 - 절대 지킬 것】\n"
        f"1. 주체 보존: 문서의 행동 주체를 바꾸지 말 것\n"
        f"2. {'1인칭(나/내가/나는)으로 답할 것' if response_mode == 'first_person' else '3인칭으로 답할 것'}\n"
        f"3. 참고 문서 밖의 사실/연도/인과를 추측해 단정하지 말 것\n"
        f"6-1. 검색 문서 수가 1 이상이면 '실록에 없다'라고 답하지 말고, 참고 문서 제목/내용을 근거로 핵심 사건을 요약할 것\n"
        f"6-2. 검색 문서 수가 0일 때만 '실록에 없다'고 답할 수 있음\n"
        f"7. 참고 문서 근거 문서 번호 표시\n"
        f"\n질문에 맞는 서술 모드를 지켜 답변할 것."
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,  # 극도로 낮춰서 일관성 확보
        max_tokens=1024,
    )
    answer_text = resp.choices[0].message.content.strip()
    
    # 단종/노산군 모드에서만 최소 1인칭 보정
    if response_mode == "first_person":
        answer_text = post_process_response(answer_text)
    
    return answer_text


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

    identity_anchor = get_identity_anchor(vs)

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

        query, cli_filters, k_override = parse_filters(raw)
        cur_k = k_override or k

        # 1단계: retrieval 질의와 검색 후 작업 지시 분리
        retrieval_query, post_task = split_retrieval_and_post_task(query)

        # 2단계: 자연어 하드 조건 추출 (king/year/month/day)
        query_for_similarity, inferred_hard_filters = extract_hard_filters_from_query(retrieval_query)

        # 2-1단계: LLM 기반 하드 필터 필요 여부/필터 판정
        # 이미 정규식/CLI에서 하드 필터가 있으면 추가 판정을 건너뛴다.
        llm_filters = {}
        if not inferred_hard_filters and not cli_filters:
            llm_hard = infer_hard_filter_with_llm(retrieval_query, llm_client, model)
            llm_filters = llm_hard.get("filters", {}) if llm_hard.get("use_hard_filter") else {}

        # CLI 필터가 자연어 추론 필터보다 우선
        merged_filters = dict(inferred_hard_filters)
        merged_filters.update(llm_filters)
        merged_filters.update(cli_filters)
        canonical_filters = _normalize_hard_filter_values(merged_filters)
        metadata_filters = _to_metadata_filter(canonical_filters)

        # 3단계: 검색 전용 전처리 (단종->나 치환 금지)
        processed_query = preprocess_query_for_retrieval(query_for_similarity)
        
        # 4단계: LLM 기반 고급 Query Rewriting
        # 하드 필터가 있으면 리라이팅이 오히려 질의를 왜곡할 수 있어 원문(전처리본)으로 검색한다.
        if metadata_filters:
            rewritten_query = processed_query
            search_query = processed_query.strip() or query_for_similarity.strip() or query
            print("  [쿼리 개선 건너뜀] 하드 필터 조건 보존을 위해 원문 질의로 검색")
        else:
            print("  [쿼리 개선 중...]")
            rewritten_query = rewrite_query_with_llm(processed_query, llm_client, model)
            search_query = rewritten_query.strip() or processed_query.strip() or query_for_similarity.strip() or query
        
        if rewritten_query != processed_query:
            print(f"  원본: {query}")
            print(f"  개선됨: {rewritten_query}")
        if post_task:
            print(f"  후처리 지시: {post_task}")
        
        kwargs = {"k": cur_k}
        if metadata_filters:
            kwargs["filter"] = metadata_filters
            # FAISS metadata filter는 상위 후보(fetch_k)에서 후처리되는 방식이라,
            # fetch_k가 작으면 실제로 존재하는 조건도 0건이 나올 수 있다.
            kwargs["fetch_k"] = max(cur_k * 50, 2000)
            print(f"  [하드 필터 검색] filter={canonical_filters}, k={cur_k}, fetch_k={kwargs['fetch_k']}")
        else:
            print(f"  [일반 유사도 검색] k={cur_k}")

        try:
            docs = vs.similarity_search(search_query, **kwargs)
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

        if not metadata_filters:
            first_meta = docs[0].metadata or {}
            print("  [디버그] 일반 검색 모드 - 1번 청크 metadata")
            print("  " + json.dumps(first_meta, ensure_ascii=False))

        ctx = format_context(docs)
        response_mode = _choose_response_mode(query, metadata_filters, docs)
        print(f"  [답변 모드] {'1인칭(단종)' if response_mode == 'first_person' else '3인칭(역사 해설)'}")
        try:
            ans = answer(
                llm_client,
                model,
                rewritten_query,
                ctx,
                identity_anchor,
                len(docs),
                response_mode,
            )
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
    parser.add_argument("--index-dir", default="faiss", type=Path,
                        help="FAISS 인덱스 폴더")
    parser.add_argument("--embedding-model", default="text-embedding-3-large",
                        help="OpenAI 임베딩 모델 (인덱스 빌드 시와 동일해야 함)")
    parser.add_argument("--model", default="gpt-4o-mini",
                        help="답변 생성에 사용할 OpenAI 모델")
    parser.add_argument("--k", default=30, type=int,
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