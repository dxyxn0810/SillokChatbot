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
TIME_HINT_KEYWORDS = (
    "원년", "년", "월", "일", "당시", "그때", "무렵",
    "단종원년", "단종1년", "단종2년", "단종3년", "단종4년",
)

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
    - 시간 표현을 구체화
    - 역사적 맥락 명확화
    - 검색에 유리한 형태로 변환
    """
    rewrite_prompt = f"""당신은 조선 단종 시대 역사 전문가입니다.
사용자의 질문을 조선왕조실록 검색에 최적화된 형태로 개선하세요.

맥락:
- 단종 재위: 1452-1455년
- 단종원년 = 1452년, 단종2년 = 1453년 등
- 단종과 노산군은 같은 인물이다 (동일 인물 표기)
- 세조: 단종의 숙부

원본 질문: {query}

개선 방향:
1. 시간 표현을 구체화 (예: "그때" → "해당 시기")
2. 역사적 맥락은 원문에 근거가 있을 때만 보강
3. 검색에 유리한 용어 사용
4. 단종/노산군 표기가 있으면 두 표기를 함께 유지

절대 금지:
- 원문에 없는 사건명을 임의로 추가하지 말 것
- 특히 원문 근거 없이 "계유정난"을 넣지 말 것
- 원문 근거 없이 특정 연/월/일(예: 1453년 10월)을 단정해서 넣지 말 것

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

    # 시간 단서가 없는 질문에 특정 연/월/일을 임의 삽입하는 경우 차단
    original_has_time_hint = _has_any_keyword(original, TIME_HINT_KEYWORDS)
    rewritten_has_specific_date = bool(re.search(r"\b\d{4}년(?:\s*\d{1,2}월)?(?:\s*\d{1,2}일)?", rewritten))
    if rewritten_has_specific_date and not original_has_time_hint:
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


# --------------------------------------------------------------------------- #
# 3) LLM 호출
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT = """당신은 조선 단종 임금입니다. 당신은 나이 12-14세 때 재위하던 시절의 증언자입니다.

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


def post_process_response(response: str) -> str:
    """
    답변을 사후 처리하여 3인칭/설명식/수동태를 1인칭/능동태로 변환
    이것이 마지막 보루다!
    """
    # ===== 3인칭 → 1인칭 =====
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
    response = re.sub(
        r'그(은|는|이|가|의|을|를)\s',
        lambda m: _first_person_by_particle(m.group(1)) + ' ',
        response,
    )
    response = re.sub(r'단종\s', '나 ', response)
    response = re.sub(r'노산군\s', '나 ', response)
    response = re.sub(r'그\s', '나 ', response)
    response = re.sub(r'그것(은|는|이|가|을|를)\s', r'그것\1 ', response)

    # 노산군/단종 자기동일성 최종 보정
    response = response.replace('나는 나가', '나는 내가')
    response = response.replace('나 사건은', '그 사건은')
    response = response.replace('나 상황을', '그 상황을')
    
    # ===== 수동태 최소 정리 (주체 보존) =====
    # 주어를 임의로 "내가"로 바꾸지 않고, 서술만 간단히 정리한다.
    response = re.sub(r'취해졌(다|습니다)', r'취했\1', response)
    response = re.sub(r'내려졌(다|습니다)', r'내렸\1', response)
    
    # "결정하였고/했습니다" -> "결정했다"
    response = re.sub(r'(?:하|을)였고', '했고', response)
    response = re.sub(r'(?:하|을)였습니다', '했다', response)
    
    # "～하려 했습니다/했다" -> "～하려 했다"
    response = re.sub(r'했습니다', '했다', response)
    response = re.sub(r'하려\s+했(습니다|었습니다)', lambda m: '하려 했' + ('다' if '습' in m.group(1) else ''),  response)
    
    # ===== 존댓말 제거 =====
    response = re.sub(r'습니다\b', '다', response)
    response = re.sub(r'(?<![ㄴㄷㄹ])습니다\b', '다', response)  # 이중 제거 방지
    response = re.sub(r'(?<![ㄴㄷㄹ])었습니다\b', '었다', response)
    response = re.sub(r'(?<![ㄴㄷㄹ])ㅂ니다\b', '다', response)
    response = re.sub(r'여겼습니다\b', '여겼다', response)
    response = re.sub(r'었습니까', '었는가', response)
    
    # ===== 기타 정제 =====
    # "내 1년/원년" 정규화
    response = re.sub(r'(?:나의|내\s+)([0-9]년|원년)', r'내 \1', response)
    
    # "있었습니다" -> "있었다"
    response = re.sub(r'있었습니다', '있었다', response)
    response = re.sub(r'있습니다', '있다', response)
    
    # 문장 시작 정규화: "있었다. 나는" 형태로
    response = re.sub(r'있었다\.\s+나(?:의|는)', '있었다. 나는', response)

    # ===== 최종 동일성 강제 보정 =====
    # 모델이 남긴 자기-3인칭 표현을 마지막에 한 번 더 강제로 교정한다.
    response = re.sub(r'나는\s+(?:단종|노산군)(?:이|가)?', '나는', response)
    response = re.sub(r'내가\s+(?:단종|노산군)(?:이|가)?', '내가', response)
    response = re.sub(r'\b(?:단종|노산군)\b', '나', response)
    response = re.sub(r'\s{2,}', ' ', response).strip()
    
    return response


def answer(client: OpenAI, model: str, query: str, context: str, identity_anchor: str) -> str:
    # 질문 전처리: 단종 -> 나 로 변환
    processed_query = preprocess_query_for_dangjong(query)
    
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
        f"{few_shot}\n\n"
        f"【자기 정체성 고정 정보】\n{identity_anchor}\n\n"
        f"【참고 문서】\n{context}\n\n"
        f"【질문】\n{processed_query}\n\n"
        f"【답변 규칙 - 절대 지킬 것】\n"
        f"1. '나', '내가', '나는', '내' 1인칭만 사용\n"
        f"2. '단종', '노산군', '그' 같은 3인칭 절대 금지\n"
        f"2-1. 단종=노산군=나 (동일 인물)로 처리\n"
        f"2-2. 출력 직전 자기검증: 답변에 '단종'/'노산군'이 남아 있으면 전부 1인칭으로 수정 후 출력\n"
        f"2-3. 주체 보존: 세조/신하/대신이 한 행동은 그 주체로 유지하고, 내가 한 일처럼 바꾸지 말 것\n"
        f"3. 수동태 금지: '취해졌다' X → '취했다' O\n"
        f"4. 존댓말 금지: '입니다' X → '다' O\n"
        f"5. 예시처럼 내가 주인공이 되어 답변하기\n"
        f"6. 참고 문서 밖의 사실/연도/인과를 추측해 단정하지 말고, 없으면 '실록에 없다'고 답하기\n"
        f"7. 참고 문서 근거 문서 번호 표시\n"
        f"\n당신은 위의 예시처럼 1인칭으로만 답변할 것입니다."
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,  # 극도로 낮춰서 일관성 확보
        max_tokens=1024,
    )
    answer_text = resp.choices[0].message.content.strip()
    
    # 사후 처리: 남은 모든 3인칭 표현을 1인칭으로 변환 (이것이 마지막 보루!)
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

        query, filters, k_override = parse_filters(raw)
        cur_k = k_override or k

        # 1단계: 기본 전처리 (단종->나, 시간 정규화)
        processed_query = preprocess_query_for_dangjong(query)
        
        # 2단계: LLM 기반 고급 Query Rewriting
        print("  [쿼리 개선 중...]")
        rewritten_query = rewrite_query_with_llm(processed_query, llm_client, model)
        
        if rewritten_query != processed_query:
            print(f"  원본: {query}")
            print(f"  개선됨: {rewritten_query}")
        
        kwargs = {"k": cur_k}
        if filters:
            kwargs["filter"] = filters
            print(f"  (filter={filters}, k={cur_k})")
        else:
            print(f"  (k={cur_k})")

        try:
            docs = vs.similarity_search(rewritten_query, **kwargs)
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
            ans = answer(llm_client, model, query, ctx, identity_anchor)
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
    parser.add_argument("--embedding-model", default="text-embedding-3-small",
                        help="OpenAI 임베딩 모델 (인덱스 빌드 시와 동일해야 함)")
    parser.add_argument("--model", default="gpt-4o-mini",
                        help="답변 생성에 사용할 OpenAI 모델")
    parser.add_argument("--k", default=15, type=int,
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