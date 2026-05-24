"""
cot_planner.py
--------------
사용자 질문 -> Chain-of-Thought -> 검색에 쓸 메타데이터 필터(plan) 생성.

흐름:
  1) gpt-4o가 질문을 보고 "어떤 메타데이터로 좁힐 수 있는지" 추론(CoT)하고,
     날짜 확인이 필요하면 위키백과 검색 키워드를 제안한다.
  2) (필요 시) 위키백과를 조회해 근거 텍스트를 모은다.
  3) gpt-4o가 위키 근거 + 자체 지식을 종합해 최종 필터(JSON)를 확정한다.

반환되는 plan(dict) 예:
  {
    "reasoning": "...CoT 설명...",
    "king": "단종",
    "solar_year": 1452,
    "month": "5월",
    "day": "18일",
    "doc_types": ["article", "daily_summary"],
    "search_query": "단종 즉위식 근정문 경복궁",
    "needs_date_filter": true
  }
"""
from __future__ import annotations
import json
from typing import Dict, Any, List

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from wiki_tool import wiki_search_summary, wiki_section_text, extract_lunar_dates
from metadata_utils import (
    norm_year, norm_month, norm_day, solar_from_reign, VALID_TYPES,
    DANJONG_REIGN_TO_SOLAR,
)

CHAT_MODEL = "gpt-4o"

# 데이터에 존재하는 사실을 모델에 알려주는 배경지식
DATA_CONTEXT = """\
[데이터 배경지식]
- 이 데이터는 조선왕조실록 중 문종/단종/세조 시기 기록이다.
- 단종 재위 매핑: 재위 0년=양력 1452년, 1년=1453, 2년=1454, 3년=1455.
- 단종은 1452년 5월(음력) 문종 승하 직후 12세에 즉위했고, 1455년(세조 즉위)에 양위했다.
- 양위 이후의 단종 관련 사건은 'king' 메타데이터가 "세조"로 분류되어 있다.
  특히 단종의 강봉(노산군), 영월 유배, 죽음(1457년, 세조 3년)은 모두 king="세조" 기록이다.
- 따라서 단종의 '말년/최후/죽음/유배/사사' 등을 물으면 king을 "단종"으로 좁히면
  자료를 놓친다. 이때는 king을 "세조"로 두거나, 아예 비워서(=null) 검색해야 한다.
- 단종 생애·시대의 주요 사건 시점 (양력연도 / 실록 음력 월):
    문종 승하·단종 즉위: 1452년 5월
    계유정난(수양대군이 김종서·황보인 등 제거): 1453년 10월
    단종 양위·세조 즉위: 1455년 윤6월
    사육신 단종복위운동: 1456년 6월
    노산군 강봉·영월 유배·죽음: 1457년 (6월 강봉, 10월 죽음)
  -> 특정 사건을 물으면 위 표를 근거로 solar_year와 month(음력)를 채워라.
- month 값은 음력 기준이며 윤달이 있으면 '윤6월'처럼 표기된다.
- 문서 type: article(개별 기사), daily_summary(하루 요약),
  monthly_summary(달 요약), yearly_summary(연 요약), base_information(총서).
"""


def _llm() -> ChatOpenAI:
    return ChatOpenAI(model=CHAT_MODEL, temperature=0)


# ---------------------------------------------------------------------------
# 1단계: CoT로 "무엇을 찾아야 하는가"를 계획
# ---------------------------------------------------------------------------
STEP1_SYSTEM = """\
너는 조선왕조실록 RAG 챗봇의 '검색 계획가'다.
사용자의 질문을 보고, 벡터DB에서 관련 문서를 잘 찾기 위해 어떤 메타데이터로
범위를 좁힐 수 있을지 단계적으로(Chain-of-Thought) 추론한다.

사용 가능한 메타데이터: king, solar_year, year(재위N년), month(음력), day, type.

특정 사건/날짜를 물으면 날짜 메타데이터(solar_year, month, day)가 유용하다.
정확한 날짜를 모르면 위키백과로 확인하는 것이 좋다.

주의:
- '최후/죽음/사사/말년/유배/강봉/즉위/양위'처럼 생애의 특정 사건은 '넓은 일생 질문'이
  아니라 '특정 시점 사건'이다. 이런 질문은 needs_date_filter=true 로 두고,
  위키백과로 연도(가능하면 월)를 확인하라.
- 단종의 양위(1455) 이후 사건(말년·유배·죽음 등)은 실록에서 king="세조"로 분류된다.
  이런 질문이면 tentative.king 을 "세조" 로 두거나, 확실치 않으면 null 로 두어
  king 으로 너무 좁히지 않게 하라.
- 범위가 넓은 질문(성격, 됨됨이, 전반적 평가 등)만 needs_date_filter=false 로 둔다.
- 문서 type 선택 기준:
  * 구체적 사건/대화/명령 확인: type=["article"]
  * 시대적 흐름/사건 요약: type=["daily_summary", "monthly_summary"]
  * 인물의 생애 총괄/평가: type=["base_information", "yearly_summary"]
- wiki_queries: 날짜뿐 아니라, 질문에 포함된 사건의 '핵심 인물'이나 '다른 명칭'을 확인하는 용도로도 적극 활용하라.

반드시 아래 JSON 형식으로만 답하라(설명 텍스트 금지):
{
  "reasoning": "추론 과정을 한국어로 서술",
  "needs_date_filter": true | false,
  "wiki_queries": ["위키백과에서 검색할 문구", ...],   // 날짜/사실 확인이 필요 없으면 []
  "tentative": {                                       // 지금 단계에서 짐작되는 값(없으면 null)
     "king": "단종"/"세조"/"문종" 또는 null,
     "solar_year": 정수 또는 null,
     "month": "5월" 같은 문자열 또는 null,
     "day": "18일" 같은 문자열 또는 null
  }
}"""


def plan_step1(query: str) -> Dict[str, Any]:
    llm = _llm()
    msgs = [
        SystemMessage(content=STEP1_SYSTEM + "\n\n" + DATA_CONTEXT),
        HumanMessage(content=f"사용자 질문: {query}"),
    ]
    resp = llm.invoke(msgs, response_format={"type": "json_object"})
    try:
        return json.loads(resp.content)
    except Exception:
        return {
            "reasoning": "(파싱 실패) 기본 의미검색으로 진행",
            "needs_date_filter": False,
            "wiki_queries": [],
            "tentative": {"king": "단종", "solar_year": None, "month": None, "day": None},
        }


# ---------------------------------------------------------------------------
# 2단계: 위키백과 근거 수집
# ---------------------------------------------------------------------------
def gather_wiki_evidence(wiki_queries: List[str], tentative_king: str = None):
    """
    개선된 근거 수집 함수:
    1. 검색 결과 중 가장 관련성 높은 페이지 제목들을 먼저 식별.
    2. 식별된 페이지들에서 핵심 섹션(생애, 사건 등)을 동적으로 추출.
    3. 중복 정보를 제거하고 음력 날짜를 더 정밀하게 파싱.
    """
    if not wiki_queries:
        return "", []

    collected_context = []
    raw_texts = []

    # 1. 다각도 쿼리로 요약 정보 수집
    for q in wiki_queries[:3]:  # 너무 많으면 토큰 낭비이므로 핵심 3개만
        search_results = wiki_search_summary(q, num_results=2) # 상위 2개씩 확인
        if search_results:
            collected_context.append(f"### 검색어: {q}\n{search_results}")
            raw_texts.append(search_results)
            # 검색 결과에서 실제 페이지 제목 후보군 추출 (내부 로직에 따라 제목 파싱 필요)

    # 2. 동적 섹션 추출 (고정된 '단종'이 아니라 질문의 주인공 위주로)
    # 예: tentative_king이 있으면 해당 왕의 문서를, 없으면 검색 결과 첫 페이지를 타겟팅
    target_page = tentative_king if tentative_king else "단종 (조선)" 
    
    # 주요 섹션 키워드를 확장하여 더 넓은 맥락 확보
    important_sections = ["생애", "즉위", "재위", "사건", "최후", "가족 관계"]
    sec_text = wiki_section_text(target_page, section_keywords=important_sections)
    
    if sec_text:
        collected_context.append(f"### [{target_page}] 상세 섹션 정보\n{sec_text}")
        raw_texts.append(sec_text)

    # 3. 음력 날짜 추출 및 중복 제거
    combined_text = "\n".join(raw_texts)
    lunar_dates = extract_lunar_dates(combined_text)
    
    # 날짜 중복 제거 (set 활용)
    unique_lunar = []
    seen_date = set()
    for d in lunar_dates:
        date_key = f"{d['month']}-{d['day']}"
        if date_key not in seen_date:
            unique_lunar.append(d)
            seen_date.add(date_key)

    evidence_final = "\n\n".join(collected_context)
    return evidence_final, unique_lunar


# ---------------------------------------------------------------------------
# 3단계: 위키 근거 + 배경지식으로 최종 필터 확정
# ---------------------------------------------------------------------------
STEP3_SYSTEM = """\
너는 검색 계획가다. 1단계 추론과 위키백과 근거를 종합하여,
벡터DB 검색에 사용할 '최종 메타데이터 필터'와 '의미검색 쿼리'를 확정한다.

규칙:
- 날짜를 특정할 수 있으면 solar_year/month/day를 채운다. (month는 음력 기준)
- 위키 근거의 날짜가 양력인지 음력인지 애매하면, 너무 좁히지 말고
  day는 비워두고 month까지만 채우는 것이 안전하다.
- 단종 즉위는 1452년이다. 재위년 표기가 필요하면 0년=1452 매핑을 쓴다.
- king 선택에 주의: 단종의 양위(1455) 이후 사건(말년·유배·강봉·죽음 등)은
  실록에서 king="세조"로 분류된다. 이런 질문이면 king을 "세조"로 두거나,
  연도만 확실하고 어느 왕대인지 애매하면 king을 null로 두어 너무 좁히지 마라.
  (예: '단종의 최후' -> king="세조", solar_year=1457)
- doc_types 는 ["article","daily_summary","monthly_summary","yearly_summary","base_information"]
  중 적절한 것들. 특정 사건/날짜를 묻는 질문이면 반드시 "article" 을 포함하라
  (개별 기사에 날짜가 붙어 있어 날짜 필터와 함께 써야 정확하다).
  "자세히/전후 상황/경위" 처럼 구체적 서술을 원하면 ["article","daily_summary"] 를 쓴다.
  성격·평가 등 넓고 추상적인 질문일 때만 ["yearly_summary","monthly_summary"] 또는 [] 를 쓴다.
- 사건의 연도를 알면 month(음력)도 위 배경지식 표를 근거로 가능한 한 채워라.
  연도까지만 확실하고 월이 불확실하면 month는 비워도 된다(이후 단계가 알아서 좁힌다).
- 질문이 '...부터 ...까지', '...하는 동안', '재위 기간', '양위 후 유배 전' 처럼
  여러 해에 걸친 '기간'을 가리키면, 단일 solar_year 대신 solar_year_range 에
  [시작연도, 끝연도] 를 넣어라. (예: 양위(1455)~유배(1457) -> [1455, 1457])
  이 경우 solar_year, month, day 는 null 로 둔다.
- 같은 해 안에서 '며칠/몇 달에 걸친' 기간이면 solar_year 는 채우고:
    * 여러 '달'에 걸치면 month_range=[시작월, 끝월] (숫자만, 예: [9,10]). month/day 는 null.
    * 한 달 안에서 '며칠' 범위면 month 를 채우고 day_range=[시작일, 끝일] (숫자만, 예: [8,12]).
  예: '계유정난 전후'(1453년 10월 8~12일경) -> solar_year=1453, month="10월", day_range=[8,12]
  예: '즉위 직후 몇 달'(1452년 5~7월) -> solar_year=1452, month_range=[5,7]
  (윤달은 시스템이 자동으로 포함하므로 신경 쓰지 않아도 된다.)
- 기간이 양위(1455) 전후에 걸치면 1455년에는 king="단종"과 "세조" 기록이 모두 있으므로,
  king 은 null 로 두는 것이 안전하다(연도 범위로만 좁힌다).
- 위키 근거에 "본문에서 추출한 '음력' 날짜 후보" 가 있으면, 우리 데이터가 음력 기준이므로
  그 값(특히 근거가 '음력명시'인 것)을 month/day 에 우선 사용하라.
  위키 본문에 양력과 음력이 함께 적힌 경우(예: "11월 10일 (음력 10월 10일)")에는
  반드시 음력 쪽(10월 10일)을 택하라.
- search_query 는 벡터검색용 핵심 키워드 위주의 한국어 문구.

반드시 아래 JSON으로만 답하라:
{
  "reasoning": "최종 판단 근거",
  "king": "단종"/"세조"/"문종" 또는 null,
  "solar_year": 정수 또는 null,
  "solar_year_range": [시작연도, 끝연도] 또는 null,
  "month": "5월" 또는 null,
  "month_range": [시작월숫자, 끝월숫자] 또는 null,
  "day": "18일" 또는 null,
  "day_range": [시작일숫자, 끝일숫자] 또는 null,
  "doc_types": [...],
  "search_query": "..."
}"""


def plan_step3(query: str, step1: Dict[str, Any], wiki_evidence: str) -> Dict[str, Any]:
    llm = _llm()
    payload = {
        "user_question": query,
        "step1_reasoning": step1.get("reasoning"),
        "step1_tentative": step1.get("tentative"),
        "wiki_evidence": wiki_evidence or "(없음)",
    }
    msgs = [
        SystemMessage(content=STEP3_SYSTEM + "\n\n" + DATA_CONTEXT),
        HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
    ]
    resp = llm.invoke(msgs, response_format={"type": "json_object"})
    try:
        plan = json.loads(resp.content)
    except Exception:
        plan = {}

    # 정규화 및 보정
    # king 은 모델 판단을 존중한다. null 이면 비워 두어 (왕 필터 없이) 검색하게 한다.
    # (단종 말년·죽음 등은 king="세조" 기록이므로 단종으로 강제하면 자료를 놓침)
    king = plan.get("king")
    plan["king"] = king if king in ("단종", "세조", "문종") else None
    plan["month"] = norm_month(plan.get("month"))
    plan["day"] = norm_day(plan.get("day"))
    sy = plan.get("solar_year")
    plan["solar_year"] = int(sy) if isinstance(sy, (int, float)) or (isinstance(sy, str) and sy.isdigit()) else None
    # 연도 범위 정규화: [시작, 끝] 정수 쌍이면 보존, 아니면 None
    def _int_pair(v):
        if isinstance(v, (list, tuple)) and len(v) == 2:
            try:
                a, b = int(v[0]), int(v[1])
                return [min(a, b), max(a, b)]
            except (TypeError, ValueError):
                return None
        return None

    plan["solar_year_range"] = _int_pair(plan.get("solar_year_range"))
    plan["month_range"] = _int_pair(plan.get("month_range"))
    plan["day_range"] = _int_pair(plan.get("day_range"))
    # 단일 값과 범위가 동시에 오면 단일 값을 우선(범위 해제)하여 충돌 방지
    if plan.get("month") and plan.get("month_range"):
        plan["month_range"] = None
    if plan.get("day") and plan.get("day_range"):
        plan["day_range"] = None
    dts = plan.get("doc_types") or []
    plan["doc_types"] = [t for t in dts if t in VALID_TYPES]
    if not plan.get("search_query"):
        plan["search_query"] = query
    return plan


# ---------------------------------------------------------------------------
# 전체 파이프라인
# ---------------------------------------------------------------------------
def make_plan(query: str, verbose: bool = False) -> Dict[str, Any]:
    step1 = plan_step1(query)
    if verbose:
        print("\n[CoT 1단계 추론]\n", step1.get("reasoning"))

    wiki_evidence = ""
    lunar_dates: List[Dict[str, str]] = []
    if step1.get("needs_date_filter") and step1.get("wiki_queries"):
        if verbose:
            print("[위키 검색]", step1.get("wiki_queries"))
        wiki_evidence, lunar_dates = gather_wiki_evidence(step1.get("wiki_queries", []))
        if verbose and lunar_dates:
            print("[위키 추출 음력 날짜]",
                  ", ".join(f"{d['month']}{d['day']}({d['source']})" for d in lunar_dates))

    plan = plan_step3(query, step1, wiki_evidence)

    # 보정: step3가 day를 비웠는데, 위키에서 '음력 명시' 날짜가 추출됐고
    # 그 month가 plan의 month와 일치하면 day를 채운다. (계유정난 10월 10일 같은 경우)
    # 단, 일부러 day_range 를 쓴 경우엔 보정하지 않는다.
    if plan.get("month") and not plan.get("day") and not plan.get("day_range"):
        for d in lunar_dates:
            if d["source"] == "음력명시" and norm_month(d["month"]) == plan["month"]:
                plan["day"] = norm_day(d["day"])
                if verbose:
                    print(f"[day 보정] 위키 음력 명시 근거로 day={plan['day']} 적용")
                break

    plan["_step1"] = step1
    plan["_wiki_used"] = bool(wiki_evidence)
    plan["_lunar_dates"] = lunar_dates
    if verbose:
        print("[최종 검색 계획]")
        print("  king         :", plan.get("king"))
        print("  solar_year   :", plan.get("solar_year"))
        print("  solar_year_range:", plan.get("solar_year_range"))
        print("  month        :", plan.get("month"))
        print("  month_range  :", plan.get("month_range"))
        print("  day          :", plan.get("day"))
        print("  day_range    :", plan.get("day_range"))
        print("  doc_types    :", plan.get("doc_types"))
        print("  search_query :", plan.get("search_query"))
    return plan