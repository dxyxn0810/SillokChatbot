"""
metadata_utils.py
-----------------
data.jsonl 메타데이터 형식을 다루기 위한 유틸리티.

데이터에서 관찰된 형식:
    year:        "0년", "1년", "2년", "3년"  (재위 N년, 문자열)
    solar_year:  1452, 1453, ...            (정수, 양력)
    month:       "1월" ~ "12월", "윤6월", "윤9월" (윤달 접두사 가능)
    day:         "1일" ~ "30일"
    type:        article / daily_summary / monthly_summary / yearly_summary / base_information

단종 재위 매핑(데이터 기준):
    0년 -> 1452, 1년 -> 1453, 2년 -> 1454, 3년 -> 1455
"""
from __future__ import annotations
import re
from typing import Optional, Dict, Any

# 단종 재위년(재위 N년) <-> 양력 매핑 (data.jsonl에서 확인된 값)
DANJONG_REIGN_TO_SOLAR = {0: 1452, 1: 1453, 2: 1454, 3: 1455}
DANJONG_SOLAR_TO_REIGN = {v: k for k, v in DANJONG_REIGN_TO_SOLAR.items()}

# 참고: 인접 왕 재위(문맥 검색 시 활용 가능)
FULL_REIGN_TABLE = {
    "문종": {0: 1450, 1: 1451, 2: 1452},
    "단종": {0: 1452, 1: 1453, 2: 1454, 3: 1455},
    "세조": {1: 1455, 2: 1456, 3: 1457},
}

VALID_TYPES = {
    "article",
    "daily_summary",
    "monthly_summary",
    "yearly_summary",
    "base_information",
}


def norm_year(value) -> Optional[str]:
    """'2', 2, '2년' -> '2년' 형태로 정규화. None이면 None."""
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    m = re.search(r"(\d+)", s)
    if not m:
        return None
    return f"{int(m.group(1))}년"


def norm_month(value) -> Optional[str]:
    """'2', 2, '2월', '윤6', '윤6월' -> '2월' / '윤6월' 형태로 정규화."""
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    leap = "윤" in s
    m = re.search(r"(\d+)", s)
    if not m:
        return None
    num = int(m.group(1))
    return f"윤{num}월" if leap else f"{num}월"


def norm_day(value) -> Optional[str]:
    """'18', 18, '18일' -> '18일' 형태로 정규화."""
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    m = re.search(r"(\d+)", s)
    if not m:
        return None
    return f"{int(m.group(1))}일"


def month_num(value) -> Optional[int]:
    """'10월','윤6월' -> 정수 10, 6. 윤달 여부는 무시한 순수 숫자."""
    if value is None:
        return None
    m = re.search(r"(\d+)", str(value))
    return int(m.group(1)) if m else None


def is_leap_month(value) -> bool:
    return value is not None and "윤" in str(value)


def lunar_month_key(value):
    """음력 월 정렬 키. 같은 숫자면 평달 < 윤달 (6월 < 윤6월 < 7월)."""
    n = month_num(value) or 0
    return (n, 1 if is_leap_month(value) else 0)


def day_num(value) -> Optional[int]:
    """'18일' -> 18."""
    if value is None:
        return None
    m = re.search(r"(\d+)", str(value))
    return int(m.group(1)) if m else None


def months_in_range(available_months, start_month, end_month):
    """
    available_months: 그 연도에 실제 존재하는 월 목록(윤달 포함, 예: ['6월','윤6월','7월']).
    start_month~end_month(숫자 기준, 윤달 포함) 범위에 드는 월들을 달력 순서로 반환.
    윤달도 자동 포함된다(예: 6~7월 -> ['6월','윤6월','7월']).
    """
    s = month_num(start_month)
    e = month_num(end_month)
    if s is None or e is None:
        return []
    lo, hi = min(s, e), max(s, e)
    picked = [m for m in available_months if lo <= (month_num(m) or -1) <= hi]
    return sorted(picked, key=lunar_month_key)


def solar_from_reign(king: str, reign_year) -> Optional[int]:
    """왕 + 재위년 -> 양력 연도."""
    if king not in FULL_REIGN_TABLE:
        return None
    n = norm_year(reign_year)
    if n is None:
        return None
    num = int(re.search(r"(\d+)", n).group(1))
    return FULL_REIGN_TABLE[king].get(num)


def reign_from_solar(king: str, solar_year: int) -> Optional[str]:
    """왕 + 양력 연도 -> 재위년('N년')."""
    table = FULL_REIGN_TABLE.get(king, {})
    for reign, solar in table.items():
        if solar == solar_year:
            return f"{reign}년"
    return None


def build_filter(
    king: Optional[str] = None,
    solar_year: Optional[int] = None,
    year: Optional[str] = None,
    month: Optional[str] = None,
    day: Optional[str] = None,
    doc_type: Optional[str] = None,
) -> Dict[str, Any]:
    """주어진 값들로 FAISS 메타데이터 필터 dict를 만든다 (None은 제외)."""
    f: Dict[str, Any] = {}
    if king:
        f["king"] = king
    if solar_year is not None:
        f["solar_year"] = int(solar_year)
    if year:
        f["year"] = norm_year(year)
    if month:
        f["month"] = norm_month(month)
    if day:
        f["day"] = norm_day(day)
    if doc_type:
        f["type"] = doc_type
    return f


def metadata_matches(meta: Dict[str, Any], flt: Dict[str, Any]) -> bool:
    """문서 메타데이터가 필터를 만족하는지 검사 (수동 필터링용)."""
    for k, v in flt.items():
        if v is None:
            continue
        mv = meta.get(k)
        if k == "solar_year":
            try:
                if mv is None or int(mv) != int(v):
                    return False
            except (TypeError, ValueError):
                return False
        else:
            if str(mv) != str(v):
                return False
    return True