"""
persona.py
──────────
인물 이름을 입력받아 Wikipedia API로 기본 정보를 조회하고,
해당 인물의 활동 시기(재위/생몰년)를 파악하는 모듈.

반환 예시:
{
  "name": "세종",
  "display_name": "세종대왕",
  "is_king": True,
  "birth_year": 1397,
  "death_year": 1450,
  "reign_start": 1418,
  "reign_end": 1450,
  "active_years": (1418, 1450),   # 실록 검색 필터에 사용
  "summary": "조선의 제4대 왕 ...",
  "wiki_url": "https://ko.wikipedia.org/wiki/세종"
}
"""

import re
import requests

WIKI_API = "https://ko.wikipedia.org/w/api.php"
HEADERS  = {"User-Agent": "SillokChatbot/1.0 (research project)"}

# 왕 이름 별칭 정규화
NAME_ALIAS: dict[str, str] = {
    "세종대왕": "세종",
    "태조대왕": "태조",
    "정조대왕": "정조",
    "영조대왕": "영조",
    "이도":     "세종",
    "이성계":   "태조",
}

# 왕 목록 (실록에 등장하는 왕)
KING_SET = {
    "태조", "정종", "태종", "세종", "문종", "단종", "세조", "예종",
    "성종", "연산군", "중종", "인종", "명종", "선조", "광해군",
    "인조", "효종", "현종", "숙종", "경종", "영조", "정조", "순조",
    "헌종", "철종", "고종", "순종",
}

# 왕별 재위 기간 (실록 필터용 fallback)
KING_REIGN: dict[str, tuple[int, int]] = {
    "태조":  (1392, 1398), "정종":  (1398, 1400), "태종":  (1400, 1418),
    "세종":  (1418, 1450), "문종":  (1450, 1452), "단종":  (1452, 1455),
    "세조":  (1455, 1468), "예종":  (1468, 1469), "성종":  (1469, 1494),
    "연산군":(1494, 1506), "중종":  (1506, 1544), "인종":  (1544, 1545),
    "명종":  (1545, 1567), "선조":  (1567, 1608), "광해군":(1608, 1623),
    "인조":  (1623, 1649), "효종":  (1649, 1659), "현종":  (1659, 1674),
    "숙종":  (1674, 1720), "경종":  (1720, 1724), "영조":  (1724, 1776),
    "정조":  (1776, 1800), "순조":  (1800, 1834), "헌종":  (1834, 1849),
    "철종":  (1849, 1863), "고종":  (1863, 1907), "순종":  (1907, 1910),
}


# ── Wikipedia API 헬퍼 ───────────────────────────────────────────────────────

def _wiki_search(query: str) -> str | None:
    """검색어로 가장 관련성 높은 Wikipedia 문서 제목을 반환."""
    params = {
        "action": "query", "list": "search",
        "srsearch": query, "srlimit": 1,
        "format": "json", "utf8": 1,
    }
    try:
        r = requests.get(WIKI_API, params=params, headers=HEADERS, timeout=10)
        r.raise_for_status()
        hits = r.json().get("query", {}).get("search", [])
        return hits[0]["title"] if hits else None
    except Exception:
        return None


def _wiki_summary(title: str) -> dict:
    """문서 제목으로 intro 요약과 URL을 가져온다."""
    params = {
        "action": "query", "prop": "extracts|info",
        "titles": title, "exintro": True, "explaintext": True,
        "inprop": "url", "redirects": True,
        "format": "json", "utf8": 1,
    }
    try:
        r = requests.get(WIKI_API, params=params, headers=HEADERS, timeout=10)
        r.raise_for_status()
        pages = r.json()["query"]["pages"]
        page  = next(iter(pages.values()))
        return {
            "title":   page.get("title", title),
            "extract": page.get("extract", ""),
            "url":     page.get("fullurl", f"https://ko.wikipedia.org/wiki/{title}"),
        }
    except Exception:
        return {"title": title, "extract": "", "url": ""}


# ── 연도 추출 ────────────────────────────────────────────────────────────────

def _extract_years(text: str) -> dict:
    """
    Wikipedia 요약 텍스트에서 생몰년 / 재위 기간을 추출한다.
    패턴 예:
      "1397년 ~ 1450년"  / "재위: 1418년 ~ 1450년"
      "(1418년 - 1450년)"
    """
    result = {"birth": None, "death": None, "reign_start": None, "reign_end": None}

    # 재위 기간 패턴
    reign_m = re.search(
        r"재위[:\s]*(\d{3,4})년?\s*[~\-–]\s*(\d{3,4})년?", text
    )
    if reign_m:
        result["reign_start"] = int(reign_m.group(1))
        result["reign_end"]   = int(reign_m.group(2))

    # 생몰년 패턴 (괄호 안의 경우도 포함)
    birth_m = re.search(r"(\d{3,4})년\s*[~\-–]\s*(\d{3,4})년", text)
    if birth_m:
        y1, y2 = int(birth_m.group(1)), int(birth_m.group(2))
        # 재위년과 겹치지 않는 경우만 생몰년으로 간주
        if result["reign_start"] is None:
            result["birth"] = y1
            result["death"] = y2
        else:
            if y1 < result["reign_start"]:
                result["birth"] = y1
            if y2 > result["reign_end"]:
                result["death"] = y2

    return result


# ── 공개 인터페이스 ──────────────────────────────────────────────────────────

def get_persona(raw_name: str) -> dict:
    """
    인물 이름 → Wikipedia 조회 → 페르소나 dict 반환.
    Wikipedia 조회 실패 시 fallback 데이터로 대체.
    """
    name = NAME_ALIAS.get(raw_name.strip(), raw_name.strip())
    is_king = name in KING_SET

    # Wikipedia 검색
    search_q  = f"{name} 조선" if is_king else f"{name} 조선 신하"
    wiki_title = _wiki_search(search_q) or name
    wiki_data  = _wiki_summary(wiki_title)

    summary = wiki_data["extract"]
    years   = _extract_years(summary)

    # fallback: 왕인 경우 KING_REIGN에서 보충
    if is_king and name in KING_REIGN:
        rs, re_ = KING_REIGN[name]
        if years["reign_start"] is None:
            years["reign_start"] = rs
        if years["reign_end"] is None:
            years["reign_end"] = re_

    # 실록 검색에 사용할 활동 기간 결정
    if years["reign_start"] and years["reign_end"]:
        active = (years["reign_start"], years["reign_end"])
    elif years["birth"] and years["death"]:
        active = (years["birth"], years["death"])
    else:
        active = (1300, 1910)   # 전체 범위 (fallback)

    # 요약문 앞 500자만 사용
    short_summary = summary[:500].strip() if summary else f"{name}에 대한 정보를 찾지 못했습니다."

    return {
        "name":         name,
        "display_name": wiki_data["title"],
        "is_king":      is_king,
        "birth_year":   years["birth"],
        "death_year":   years["death"],
        "reign_start":  years["reign_start"],
        "reign_end":    years["reign_end"],
        "active_years": active,
        "summary":      short_summary,
        "wiki_url":     wiki_data["url"],
    }


if __name__ == "__main__":
    import json
    p = get_persona("세종대왕")
    print(json.dumps(p, ensure_ascii=False, indent=2))
