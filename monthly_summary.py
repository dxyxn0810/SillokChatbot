"""
daily_summary 폴더의 jsonl 파일들을 읽어,
각 파일(= 한 달분)을 하나의 monthly_summary Document 로 요약하고
모두 모아 kfa.jsonl 한 파일로 저장한다.

사용법:
    export OPENAI_API_KEY=...
    python monthly_summary.py --daily-summary-dir ./daily_summary --output-file ./kfa.jsonl
"""

import argparse
import json
import os
import re
import time
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Tuple

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --------------------------------------------------------------------------- #
# 1) 입력 파일 로드 & 월별 그룹화
# --------------------------------------------------------------------------- #
def load_jsonl(path: Path) -> List[dict]:
    """jsonl 파일을 읽어 dict 리스트로 반환한다."""
    docs = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            docs.append(json.loads(line))
    return docs


def _day_sort_key(day_str: str) -> int:
    """'14일' -> 14. 변환 실패 시 큰 값으로 뒤에 둔다."""
    m = re.search(r"\d+", day_str or "")
    return int(m.group(0)) if m else 10**9


def group_by_month(docs: List[dict]) -> "OrderedDict[Tuple, List[dict]]":
    """
    docs 를 (king, year, month) -> [daily_summary_doc, ...] 로 묶는다.
    각 리스트는 일(day) 오름차순으로 정렬한다.
    한 daily_summary 파일은 통상 한 달치이므로 그룹이 1개이지만,
    혹시 섞여 있어도 안전하게 동작하도록 일반화해 둔다.
    """
    grouped: "OrderedDict[Tuple, List[dict]]" = OrderedDict()
    for d in docs:
        m = d["metadata"]
        if m.get("type") != "daily_summary":
            continue
        key = (m["king"], m["year"], m["month"])
        grouped.setdefault(key, []).append(d)

    for key, lst in grouped.items():
        lst.sort(key=lambda x: _day_sort_key(x["metadata"].get("day", "")))
    return grouped


# --------------------------------------------------------------------------- #
# 2) full_text 생성 (한 달치 daily_summary 모음)
# --------------------------------------------------------------------------- #
_HEADER_RE = re.compile(
    r"^제목:\s*(?P<title>.*?)\n"
    r"날짜:\s*(?P<date>.*?)\n"
    r"본문 내용:\s*(?P<body>.*)$",
    re.DOTALL,
)


def _parse_daily(page_content: str) -> Dict[str, str]:
    """daily_summary 의 page_content 에서 title / body 를 추출한다."""
    m = _HEADER_RE.match(page_content)
    if not m:
        return {"title": "", "body": page_content}
    return {
        "title": m.group("title").strip(),
        "body": m.group("body").strip(),
    }


def build_full_text(month_key: Tuple, dailies: List[dict]) -> str:
    """
    한 달치 daily_summary 들을 합쳐 요약용 full_text 를 만든다.

        기간: 단종 0년(1452년) 5월

        5월 14일
        제목: ...
        본문 내용: ...

        5월 15일
        제목: ...
        ...
    """
    king, year, month = month_key
    solar_year = dailies[0]["metadata"].get("solar_year")
    header = f"기간: {king} {year}({solar_year}년) {month}\n\n"

    blocks = []
    for d in dailies:
        day = d["metadata"].get("day", "")
        parsed = _parse_daily(d["page_content"])
        block = (
            f"{month} {day}\n"
            f"제목: {parsed['title']}\n"
            f"본문 내용: {parsed['body']}"
        )
        blocks.append(block)

    return header + "\n\n".join(blocks)


# --------------------------------------------------------------------------- #
# 3) LLM 요약
# --------------------------------------------------------------------------- #
SUMMARY_SYSTEM_PROMPT = """당신은 조선왕조실록의 한 달치 일별 요약(daily summary)들을 읽고,
그 달 전체를 종합 요약하는 역사 전문가입니다.
주어진 한 달치 일별 요약을 종합하여 다음 두 가지를 JSON 으로만 출력하세요.

- "title": 그 달의 핵심을 한 줄로 압축한 제목.
  * 한국어 단형 어구로, 짧고 동사형으로 끝나는 형식(예: "단종이 즉위하고 국상을 거행하다").
  * 따옴표나 마침표는 넣지 마세요. 25자 내외.
- "summary": 그 달에 일어난 일들을 본문 형태로 종합 요약한 한국어 텍스트.
  * 일별로 단순 나열하지 말고, 주제(국상·인사·외교·재정·재난 등)별로 흐름을 묶어 정리합니다.
  * 핵심 인명·관직·사건명 등 고유 정보는 유지하되, 중복은 정리합니다.
  * 분량은 원문(일별 요약 합본) 글자수에 대체로 비례하게 (대략 원문의 1/4 ~ 1/6 분량)
    작성하되, 4~10 문장 정도가 적절합니다.

출력은 반드시 다음 JSON 형식만:
{"title": "...", "summary": "..."}
설명, 마크다운 펜스, 다른 텍스트는 절대 넣지 마세요."""


def _extract_json(text: str) -> dict:
    """LLM 응답에서 JSON 객체를 안전하게 추출."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            return json.loads(m.group(0))
        raise


def summarize_with_openai(
    full_text: str,
    model: str = "gpt-4o-mini",
    max_retries: int = 3,
) -> Dict[str, str]:
    """full_text 를 받아 {'title': ..., 'summary': ...} 를 반환한다."""
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
                    {"role": "user", "content": full_text},
                ],
                max_tokens=2048,
                temperature=0.3,
            )
            text = resp.choices[0].message.content
            data = _extract_json(text)
            title = str(data.get("title", "")).strip()
            summary = str(data.get("summary", "")).strip()
            if not title or not summary:
                raise ValueError(f"빈 필드: {data!r}")
            return {"title": title, "summary": summary}
        except Exception as e:
            last_err = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"요약 실패: {last_err}")


# --------------------------------------------------------------------------- #
# 4) 요약 결과를 monthly_summary Document 로 만들기
# --------------------------------------------------------------------------- #
def make_summary_document(
    month_key: Tuple,
    solar_year: int,
    title: str,
    summary: str,
) -> dict:
    """요약 결과를 monthly_summary 형식의 Document(dict) 로 변환한다."""
    king, year, month = month_key
    period_line = f"{king} {year}({solar_year}년) {month}"

    page_content = (
        f"제목: {title}\n"
        f"기간: {period_line}\n"
        f"본문 내용: {summary}"
    )
    return {
        "page_content": page_content,
        "metadata": {
            "type": "monthly_summary",
            "title": title,
            "king": king,
            "year": year,
            "solar_year": solar_year,
            "month": month,
            "day": None,
            "idx": None,
            "chunk_id": None,
        },
    }


# --------------------------------------------------------------------------- #
# 5) 파일 단위 처리
# --------------------------------------------------------------------------- #
def process_file(in_path: Path, model: str) -> List[dict]:
    """
    daily_summary jsonl 파일 한 개를 읽어,
    포함된 모든 (king, year, month) 그룹에 대해 monthly_summary Document 를 생성하여 반환한다.
    (보통 한 파일에는 한 달 그룹만 존재)
    """
    docs = load_jsonl(in_path)
    grouped = group_by_month(docs)
    if not grouped:
        print("  - daily_summary 항목 없음, 스킵")
        return []

    results: List[dict] = []
    for month_key, dailies in grouped.items():
        full_text = build_full_text(month_key, dailies)
        solar_year = dailies[0]["metadata"].get("solar_year")
        try:
            result = summarize_with_openai(full_text, model=model)
        except Exception as e:
            print(f"    ! 요약 실패 {month_key}: {e}")
            continue

        doc = make_summary_document(
            month_key=month_key,
            solar_year=solar_year,
            title=result["title"],
            summary=result["summary"],
        )
        results.append(doc)
        print(f"    · {month_key} -> {result['title']}")
    return results


# --------------------------------------------------------------------------- #
# 6) 엔트리 포인트
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--daily-summary-dir", default="daily_summary", type=Path,
                        help="daily_summary jsonl 파일 폴더")
    parser.add_argument("--output-file", default="kfa_month.jsonl", type=Path,
                        help="합쳐서 저장할 monthly_summary jsonl 파일")
    parser.add_argument("--model", default="gpt-4o-mini",
                        help="요약에 사용할 OpenAI 모델")
    parser.add_argument("--pattern", default="kfa_*_day.jsonl",
                        help="입력 파일 글롭 패턴")
    args = parser.parse_args()

    files = sorted(args.daily_summary_dir.glob(args.pattern))
    if not files:
        print(f"입력 파일 없음: {args.daily_summary_dir}/{args.pattern}")
        return

    print(f"총 {len(files)}개 파일 처리 시작")
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with args.output_file.open("w", encoding="utf-8") as fout:
        for in_path in files:
            print(f"[{in_path.name}]")
            try:
                docs = process_file(in_path, args.model)
            except Exception as e:
                print(f"  ! 파일 처리 실패: {e}")
                continue
            for d in docs:
                fout.write(json.dumps(d, ensure_ascii=False) + "\n")

    print(f"\n완료 -> {args.output_file}")


if __name__ == "__main__":
    main()