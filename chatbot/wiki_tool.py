"""
wiki_tool.py
------------
CoT 단계에서 날짜/사실 확인을 위해 한국어 위키백과를 조회하는 도구.

wikipedia-api 패키지를 사용하며, 네트워크/패키지 문제로 실패하더라도
챗봇 전체가 죽지 않도록 항상 안전하게 빈 결과를 반환한다.
"""
from __future__ import annotations
from typing import Optional, Dict, List

try:
    import wikipediaapi
    _WIKI = wikipediaapi.Wikipedia(
        user_agent="DanjongInterviewBot/1.0 (educational RAG demo)",
        language="ko",
    )
except Exception:  # 패키지 미설치 등
    _WIKI = None


def wiki_search_summary(query: str, max_chars: int = 3000) -> str:
    if _WIKI is None:
        return ""

    # 1. 검색어로 가장 관련성 높은 문서 제목을 찾는다.
    #    wikipedia-api 버전에 따라 .search() 반환형이 다르고(list / SearchResults 객체),
    #    실패할 수도 있으므로 방어적으로 첫 제목만 안전하게 뽑아낸다.
    best_match = None
    try:
        results = _WIKI.search(query)
        titles = None
        if isinstance(results, (list, tuple)):
            titles = list(results)
        elif hasattr(results, "pages"):          # 일부 버전: .pages 속성
            titles = list(results.pages)
        elif hasattr(results, "__iter__"):       # 순회 가능한 경우
            titles = list(results)
        if titles:
            first = titles[0]
            # 제목 문자열일 수도, 페이지/딕셔너리일 수도 있어 안전 변환
            best_match = getattr(first, "title", None) or (
                first.get("title") if isinstance(first, dict) else None
            ) or (first if isinstance(first, str) else None)
    except Exception:
        best_match = None

    # 2. 검색으로 제목을 얻었으면 그 페이지를, 못 얻었으면 검색어 자체를 제목으로 시도.
    #    .summary 는 도입부(첫 섹션 이전)만 주므로, 본문 섹션까지 담기도록 .text 를 쓰고
    #    max_chars 로 잘라 토큰 사용량을 제한한다.
    for title in (best_match, query):
        if not title:
            continue
        try:
            page = _WIKI.page(title)
            if page.exists():
                return page.text[:max_chars]
        except Exception:
            continue

    # 3. 최후의 보루: 단종 문서.
    try:
        p = _WIKI.page("단종 (조선)")
        if p.exists():
            return p.text[:max_chars]
    except Exception:
        pass
    return ""


def wiki_section_text(title: str, section_keywords=None, max_chars: int = 3000) -> str:
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