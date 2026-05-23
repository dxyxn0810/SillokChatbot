"""
kfa.jsonl (monthly_summary 모음) 을 읽어,
각 (king, year) 별로 한 해를 종합 요약한 yearly_summary Document 들을
kfa_year.jsonl 한 파일로 저장한다.

사용법:
    export OPENAI_API_KEY=...
    cd codes
    python yearly_summary.py
    # 또는 명시적으로 절대 경로 지정:
    # python yearly_summary.py --input-file D:\\path\\to\\input.jsonl --output-file D:\\path\\to\\output.jsonl
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

# 스크립트 위치 기반 프로젝트 루트 경로 계산
SCRIPT_DIR = Path(__file__).parent.resolve()      # codes 폴더의 절대 경로
PROJECT_ROOT = SCRIPT_DIR.parent                   # SillokChatbot 폴더
DEFAULT_INPUT_FILE = PROJECT_ROOT / "data" / "monthly_summary" / "kga_month.jsonl"
DEFAULT_OUTPUT_FILE = PROJECT_ROOT / "data" / "yearly_summary" / "kga_year.jsonl"

# --------------------------------------------------------------------------- #
# 1) 입력 파일 로드 & 연도별 그룹화
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


def _month_sort_key(month_str: str) -> Tuple[int, int]:
    """
    '5월' -> (5, 0), '윤5월' -> (5, 1) 형태로 정렬 키를 만든다.
    같은 달 안에서는 평달이 먼저, 윤달이 그 다음에 오도록 한다.
    """
    s = month_str or ""
    is_leap = 1 if s.startswith("윤") else 0
    m = re.search(r"\d+", s)
    n = int(m.group(0)) if m else 10**9
    return (n, is_leap)


def group_by_year(docs: List[dict]) -> "OrderedDict[Tuple, List[dict]]":
    """
    docs 를 (king, year) -> [monthly_summary_doc, ...] 로 묶는다.
    각 리스트는 월(month) 오름차순(윤달은 같은 달 뒤)으로 정렬한다.
    """
    grouped: "OrderedDict[Tuple, List[dict]]" = OrderedDict()
    for d in docs:
        m = d["metadata"]
        if m.get("type") != "monthly_summary":
            continue
        key = (m["king"], m["year"])
        grouped.setdefault(key, []).append(d)

    for key, lst in grouped.items():
        lst.sort(key=lambda x: _month_sort_key(x["metadata"].get("month", "")))
    return grouped


# --------------------------------------------------------------------------- #
# 2) full_text 생성 (한 해치 monthly_summary 모음)
# --------------------------------------------------------------------------- #
_HEADER_RE = re.compile(
    r"^제목:\s*(?P<title>.*?)\n"
    r"기간:\s*(?P<period>.*?)\n"
    r"본문 내용:\s*(?P<body>.*)$",
    re.DOTALL,
)


def _parse_monthly(page_content: str) -> Dict[str, str]:
    """monthly_summary 의 page_content 에서 title / body 를 추출한다."""
    m = _HEADER_RE.match(page_content)
    if not m:
        return {"title": "", "body": page_content}
    return {
        "title": m.group("title").strip(),
        "body": m.group("body").strip(),
    }


def build_full_text(year_key: Tuple, monthlies: List[dict]) -> str:
    """
    한 해치 monthly_summary 들을 합쳐 요약용 full_text 를 만든다.

        기간: 단종 0년(1452년)

        5월
        제목: ...
        본문 내용: ...

        6월
        제목: ...
        ...
    """
    king, year = year_key
    solar_year = monthlies[0]["metadata"].get("solar_year")
    header = f"기간: {king} {year}({solar_year}년)\n\n"

    blocks = []
    for d in monthlies:
        month = d["metadata"].get("month", "")
        parsed = _parse_monthly(d["page_content"])
        block = (
            f"{month}\n"
            f"제목: {parsed['title']}\n"
            f"본문 내용: {parsed['body']}"
        )
        blocks.append(block)

    return header + "\n\n".join(blocks)


# --------------------------------------------------------------------------- #
# 3) LLM 요약
# --------------------------------------------------------------------------- #
SUMMARY_SYSTEM_PROMPT = """당신은 조선왕조실록의 한 해치 월별 요약(monthly summary)들을 읽고,
그 한 해 전체를 종합 요약하는 역사 전문가입니다.
주어진 한 해치 월별 요약을 종합하여 다음 두 가지를 JSON 으로만 출력하세요.

- "title": 그 해의 핵심을 한 줄로 압축한 제목.
  * 한국어 단형 어구로, 짧고 동사형으로 끝나는 형식
    (예: "단종이 즉위하고 국상을 마치다", "계유정난이 일어나다").
  * 따옴표나 마침표는 넣지 마세요. 30자 내외.
- "summary": 그 해에 일어난 일들을 본문 형태로 종합 요약한 한국어 텍스트.
  * 월별로 단순 나열하지 말고, 주제(왕실·국상·인사·외교·재정·재난·반란 등)별로
    한 해의 흐름을 묶어 정리합니다.
  * 핵심 인명·관직·사건명 등 고유 정보는 유지하되, 중복은 정리합니다.
  * 분량은 원문(월별 요약 합본) 글자수에 대체로 비례하게 (대략 원문의 1/4 ~ 1/6 분량)
    작성하되, 6~15 문장 정도가 적절합니다.

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
# 4) 요약 결과를 yearly_summary Document 로 만들기
# --------------------------------------------------------------------------- #
def make_summary_document(
    year_key: Tuple,
    solar_year: int,
    title: str,
    summary: str,
) -> dict:
    """요약 결과를 yearly_summary 형식의 Document(dict) 로 변환한다."""
    king, year = year_key
    period_line = f"{king} {year}({solar_year}년)"

    page_content = (
        f"제목: {title}\n"
        f"기간: {period_line}\n"
        f"본문 내용: {summary}"
    )
    return {
        "page_content": page_content,
        "metadata": {
            "type": "yearly_summary",
            "title": title,
            "king": king,
            "year": year,
            "solar_year": solar_year,
            "month": None,
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
    monthly_summary jsonl 파일을 읽어,
    포함된 모든 (king, year) 그룹에 대해 yearly_summary Document 리스트를 반환한다.
    """
    docs = load_jsonl(in_path)
    grouped = group_by_year(docs)
    if not grouped:
        print("  - monthly_summary 항목 없음, 스킵")
        return []

    print(f"  - 연도 그룹 수: {len(grouped)}")
    results: List[dict] = []
    for year_key, monthlies in grouped.items():
        full_text = build_full_text(year_key, monthlies)
        solar_year = monthlies[0]["metadata"].get("solar_year")
        try:
            result = summarize_with_openai(full_text, model=model)
        except Exception as e:
            print(f"    ! 요약 실패 {year_key}: {e}")
            continue

        doc = make_summary_document(
            year_key=year_key,
            solar_year=solar_year,
            title=result["title"],
            summary=result["summary"],
        )
        results.append(doc)
        print(f"    · {year_key} ({solar_year}년, 월수={len(monthlies)}) -> {result['title']}")
    return results


# --------------------------------------------------------------------------- #
# 6) 엔트리 포인트
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", default=str(DEFAULT_INPUT_FILE), type=Path,
                        help="monthly_summary jsonl 입력 파일")
    parser.add_argument("--output-file", default=str(DEFAULT_OUTPUT_FILE), type=Path,
                        help="yearly_summary jsonl 저장 파일")
    parser.add_argument("--model", default="gpt-4o-mini",
                        help="요약에 사용할 OpenAI 모델")
    args = parser.parse_args()

    if not args.input_file.exists():
        print(f"입력 파일 없음: {args.input_file}")
        return

    print(f"[{args.input_file.name}]")
    try:
        docs = process_file(args.input_file, args.model)
    except Exception as e:
        print(f"  ! 파일 처리 실패: {e}")
        return

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with args.output_file.open("w", encoding="utf-8") as fout:
        for d in docs:
            fout.write(json.dumps(d, ensure_ascii=False) + "\n")

    print(f"\n완료 ({len(docs)}개 Document) -> {args.output_file}")


if __name__ == "__main__":
    main()