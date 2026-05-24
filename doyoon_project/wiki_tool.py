"""
wiki_tool.py
------------
CoT 단계에서 날짜/사실 확인을 위해 한국어 위키백과를 조회하는 도구.

wikipedia-api 패키지를 사용하며, 네트워크/패키지 문제로 실패하더라도
챗봇 전체가 죽지 않도록 항상 안전하게 빈 결과를 반환한다.
"""
from __future__ import annotations
import re
from typing import Optional, Dict, List

try:
    import wikipediaapi
    _WIKI = wikipediaapi.Wikipedia(
        user_agent="DanjongInterviewBot/1.0 (educational RAG demo)",
        language="ko",
    )
except Exception:  # 패키지 미설치 등
    _WIKI = None


# ---------------------------------------------------------------------------
# 음력 날짜 추출
# ---------------------------------------------------------------------------
# 우리 데이터의 month/day 는 '음력' 기준이다. 위키 본문에는 양력과 음력이 함께
# 적히는 경우가 많으므로(예: "11월 10일 (음력 10월 10일)"), 음력 날짜를 우선
# 정확히 골라내야 한다.

# 패턴 1: 명시적 "음력 N월 N일" / "음력 윤N월 N일"
_RE_LUNAR_EXPLICIT = re.compile(r"음력\s*(윤?\d{1,2})\s*월\s*(\d{1,2})\s*일")
# 패턴 2: 실록식 "단종 1년(1453년) 10월 10일" 또는 "단종 1년 ... 10월 10일" (실록 날짜는 음력)
_RE_SILLOK = re.compile(
    r"(?:단종|세조|문종)\s*\d+\s*년[^\d]{0,12}?(?:\(\s*\d{4}\s*년\s*\))?\s*(윤?\d{1,2})\s*월\s*(\d{1,2})\s*일"
)
# 패턴 3: "1453년 ... 10월 10일" 같이 연도와 함께 나오는 월·일
_RE_YEAR_MONTH_DAY = re.compile(r"(\d{4})\s*년[^\d]{0,15}?(윤?\d{1,2})\s*월\s*(\d{1,2})\s*일")


def extract_lunar_dates(text: str) -> List[Dict[str, str]]:
    """
    텍스트에서 음력으로 판단되는 (month, day) 후보를 신뢰도 순으로 추출.
    반환: [{"month": "10월", "day": "10일", "source": "음력명시"}...]
    중복 제거하며, 명시적 '음력' 표기 > 실록식 > 일반 순으로 정렬.
    """
    if not text:
        return []
    found: List[Dict[str, str]] = []
    seen = set()

    def push(mon: str, day: str, src: str):
        mon = f"{mon}월" if not mon.endswith("월") else mon
        day = f"{day}일" if not day.endswith("일") else day
        key = (mon, day)
        if key not in seen:
            seen.add(key)
            found.append({"month": mon, "day": day, "source": src})

    for m, d in _RE_LUNAR_EXPLICIT.findall(text):
        push(m, d, "음력명시")
    for m, d in _RE_SILLOK.findall(text):
        push(m, d, "실록식")
    # 연도까지 붙은 일반 패턴(양력일 수 있어 신뢰도 낮음, 참고용)
    for _y, m, d in _RE_YEAR_MONTH_DAY.findall(text):
        push(m, d, "일반(양력가능)")
    return found


def wiki_search_summary(query: str, max_chars: int = 2000) -> str:
    if _WIKI is None: return ""
    
    # 1. 먼저 검색어를 통해 가장 연관성 높은 문서 제목 리스트를 가져옵니다.
    search_results = _WIKI.search(query) 
    
    if search_results:
        # 2. 검색 결과 중 첫 번째(가장 유사한) 제목으로 페이지를 가져옵니다.
        # 예: "수양대군 단종복위운동" 검색 -> "단종 복위 운동" 제목 획득
        best_match = search_results[0]
        page = _WIKI.page(best_match)
        if page.exists():
            return page.summary[:max_chars]
    
    # 3. 검색 결과도 없다면 그때 최후의 보루로 단종의 이름을 시도합니다.
    p = _WIKI.page("단종 (조선)")
    return p.summary[:max_chars]


def wiki_section_text(title: str, section_keywords=None, max_chars: int = 2500) -> str:
    """
    특정 문서에서 키워드가 들어간 섹션의 본문을 가져온다.
    예: title='단종 (조선)', section_keywords=['즉위','생애']
    """
    if _WIKI is None:
        return ""
    try:
        page = _WIKI.page(title)
        if not page.exists():
            return ""
        if not section_keywords:
            return page.text[:max_chars]
        chunks = []

        def walk(sections):
            for s in sections:
                if any(kw in s.title for kw in section_keywords):
                    chunks.append(f"[{s.title}]\n{s.text}")
                walk(s.sections)

        walk(page.sections)
        joined = "\n\n".join(chunks)
        return (joined or page.text)[:max_chars]
    except Exception:
        return ""