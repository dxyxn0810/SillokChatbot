"""
조선왕조실록 article 폴더의 jsonl 파일들을 읽어,
각 날짜별로 그날의 기사들을 모아 요약한 daily_summary jsonl 파일을 생성한다.

사용법:
    export OPENAI_API_KEY=...
    python daily_summary.py --article-dir ./article --output-dir ./daily_summary
"""

import argparse
import json
import os
import re
import time
from collections import defaultdict, OrderedDict
from pathlib import Path
from typing import Dict, List, Tuple

from openai import OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

client = OpenAI(api_key="sk-proj-...")

# --------------------------------------------------------------------------- #
# 1) 입력 파일 로드 & 날짜·기사별 그룹화
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


def group_by_date_and_article(docs: List[dict]) -> "OrderedDict[Tuple, OrderedDict[int, List[dict]]]":
    """
    docs 를 (king, year, month, day) -> idx -> [chunk_doc, ...] 형태로 묶는다.
    날짜 등장 순서와 idx 순서를 보존하기 위해 OrderedDict 를 사용한다.
    chunk_doc 리스트는 chunk_id 오름차순으로 정렬한다.
    """
    grouped: "OrderedDict[Tuple, OrderedDict[int, List[dict]]]" = OrderedDict()
    for d in docs:
        m = d["metadata"]
        if m.get("type") != "article":
            # daily_summary 등 다른 타입은 스킵
            continue
        date_key = (m["king"], m["year"], m["month"], m["day"])
        idx = m["idx"]
        grouped.setdefault(date_key, OrderedDict()).setdefault(idx, []).append(d)

    # chunk_id 순으로 정렬
    for date_key, by_idx in grouped.items():
        for idx in by_idx:
            by_idx[idx].sort(key=lambda x: x["metadata"].get("chunk_id", 0))
    return grouped


# --------------------------------------------------------------------------- #
# 2) full_text 생성
# --------------------------------------------------------------------------- #
_HEADER_RE = re.compile(
    r"^기사 제목:\s*(?P<title>.*?)\n"
    r"날짜:\s*(?P<date>.*?)\n"
    r"카테고리:\s*(?P<cat>.*?)\n"
    r"본문 내용:\s*(?P<body>.*)$",
    re.DOTALL,
)


def _join_overlapping(bodies: List[str], max_overlap: int = 200) -> str:
    """
    같은 idx 의 본문 chunk 들을 이어 붙일 때, splitter 의 chunk_overlap 으로
    생긴 머리·꼬리 중복을 가능한 한 제거하면서 합친다.
    """
    if not bodies:
        return ""
    out = bodies[0]
    for nxt in bodies[1:]:
        # out 의 끝부분과 nxt 의 앞부분이 최대 max_overlap 자까지 겹치는지 검사
        overlap_len = 0
        upper = min(len(out), len(nxt), max_overlap)
        for k in range(upper, 0, -1):
            if out.endswith(nxt[:k]):
                overlap_len = k
                break
        out += nxt[overlap_len:]
    return out


def _parse_chunk(page_content: str) -> Dict[str, str]:
    """page_content 에서 title / categories / body 를 추출한다."""
    m = _HEADER_RE.match(page_content)
    if not m:
        # 형식이 다를 경우 안전하게 본문 전체를 body 로
        return {"title": "", "categories": "", "body": page_content}
    return {
        "title": m.group("title").strip(),
        "categories": m.group("cat").strip(),
        "body": m.group("body").strip(),
    }


def build_full_text(date_key: Tuple, by_idx: "OrderedDict[int, List[dict]]") -> str:
    """
    요청한 형식의 full_text 를 만든다.

        날짜: 단종 0년(1452년) 5월 17일

        기사 1
        기사 제목: ...
        카테고리: ...
        본문 내용: ...

        기사 2
        ...
    """
    king, year, month, day = date_key
    # solar_year 는 chunk 메타데이터에서 가져온다
    first_chunk_meta = next(iter(by_idx.values()))[0]["metadata"]
    solar_year = first_chunk_meta.get("solar_year")

    header = f"날짜: {king} {year}({solar_year}년) {month} {day}\n\n"

    blocks = []
    # idx 오름차순으로 기사 순서 결정 (원본 등장순도 보존됨)
    for n, idx in enumerate(sorted(by_idx.keys()), start=1):
        chunks = by_idx[idx]
        first_parsed = _parse_chunk(chunks[0]["page_content"])
        title = first_parsed["title"]
        categories = first_parsed["categories"]
        # 같은 idx 의 chunk 본문을 chunk_id 순으로 이어붙이되 overlap 중복을 제거
        bodies = [_parse_chunk(c["page_content"])["body"] for c in chunks]
        body_joined = _join_overlapping(bodies)

        block = (
            f"기사 {n}\n"
            f"기사 제목: {title}\n"
            f"카테고리: {categories}\n"
            f"본문 내용: {body_joined}"
        )
        blocks.append(block)

    return header + "\n\n".join(blocks)


# --------------------------------------------------------------------------- #
# 3) LLM 요약
# --------------------------------------------------------------------------- #
SUMMARY_SYSTEM_PROMPT = """당신은 조선왕조실록의 하루치 기사들을 읽고 요약하는 역사 전문가입니다.
주어진 하루의 기사들을 종합하여 다음 두 가지를 JSON 으로만 출력하세요.

- "title": 그 날의 핵심을 한 줄로 압축한 제목.
  * 입력 기사들의 제목과 유사한 한국어 스타일(짧고 동사형으로 끝나는 단형 어구)을 따릅니다.
  * 예: "빈전 도감을 설치하다", "사헌부에서 ...을 청하다".
  * 따옴표나 마침표는 넣지 마세요. 20자 내외.
- "summary": 그 날 일어난 일을 본문 형태로 요약한 한국어 텍스트.
  * 원문의 인명·관직·사건명 등 고유 정보를 유지하되, 중복은 정리합니다.
  * 분량은 원문 글자수에 대체로 비례하게 (대략 원문의 1/4 ~ 1/6 분량) 작성하세요.
  * 단, 너무 짧으면 의미가 사라지므로 최소 2~3 문장 이상 작성합니다.

출력은 반드시 다음 JSON 형식만:
{"title": "...", "summary": "..."}
설명, 마크다운 펜스, 다른 텍스트는 절대 넣지 마세요."""


def _extract_json(text: str) -> dict:
    """LLM 응답에서 JSON 객체를 안전하게 추출."""
    text = text.strip()
    # 코드펜스 제거
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    # 가장 바깥의 {...} 만 시도
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
            # v1.0.0부터는 딕셔너리가 아닌 객체 속성(Attribute) 접근법을 사용하므로 코드는 그대로 유지해도 작동합니다.
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
# 4) 요약 결과를 chunk 단위 Document 로 만들기
# --------------------------------------------------------------------------- #
def make_summary_documents(
    date_key: Tuple,
    solar_year: int,
    title: str,
    summary: str,
    splitter: RecursiveCharacterTextSplitter,
) -> List[dict]:
    """
    요약 결과를 splitter 로 chunk 분할해 Document(dict) 리스트로 변환한다.
    """
    king, year, month, day = date_key
    date_line = f"{king} {year}({solar_year}년) {month} {day}"

    # 분할 대상은 본문 내용 (요약문) 자체. 제목/날짜 헤더는 매 chunk 에 동일하게 붙인다.
    body_chunks = splitter.split_text(summary)
    if not body_chunks:
        body_chunks = [summary]

    docs = []
    for chunk_id, body in enumerate(body_chunks):
        page_content = (
            f"제목: {title}\n"
            f"날짜: {date_line}\n"
            f"본문 내용: {body}"
        )
        docs.append({
            "page_content": page_content,
            "metadata": {
                "type": "daily_summary",
                "title": title,
                "king": king,
                "year": year,
                "solar_year": solar_year,
                "month": month,
                "day": day,
                "idx": None,
                "chunk_id": None
            },
        })
    return docs


# --------------------------------------------------------------------------- #
# 5) 파일 단위 처리
# --------------------------------------------------------------------------- #
def process_file(
    in_path: Path,
    out_path: Path,
    splitter: RecursiveCharacterTextSplitter,
    model: str,
    overwrite: bool = False,
) -> None:
    if out_path.exists() and not overwrite:
        print(f"  - skip (이미 존재): {out_path.name}")
        return

    docs = load_jsonl(in_path)
    grouped = group_by_date_and_article(docs)
    print(f"  - 날짜 수: {len(grouped)}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fout:
        for date_key, by_idx in grouped.items():
            full_text = build_full_text(date_key, by_idx)
            solar_year = next(iter(by_idx.values()))[0]["metadata"].get("solar_year")

            try:
                result = summarize_with_openai(full_text, model=model)
            except Exception as e:
                print(f"    ! 요약 실패 {date_key}: {e}")
                continue

            summary_docs = make_summary_documents(
                date_key=date_key,
                solar_year=solar_year,
                title=result["title"],
                summary=result["summary"],
                splitter=splitter,
            )
            for d in summary_docs:
                fout.write(json.dumps(d, ensure_ascii=False) + "\n")

            print(f"    · {date_key} -> {result['title']}")


# --------------------------------------------------------------------------- #
# 6) 엔트리 포인트
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--article-dir", default="article", type=Path,
                        help="원본 기사 jsonl 파일 폴더")
    parser.add_argument("--output-dir", default="daily_summary", type=Path,
                        help="요약 jsonl 저장 폴더")
    parser.add_argument("--model", default="gpt-4o-mini",
                        help="요약에 사용할 OpenAI 모델")
    parser.add_argument("--overwrite", action="store_true",
                        help="이미 존재하는 출력 파일을 덮어쓴다")
    parser.add_argument("--pattern", default="kfa_*.jsonl",
                        help="입력 파일 글롭 패턴")
    args = parser.parse_args()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""],
    )

    files = sorted(args.article_dir.glob(args.pattern))
    if not files:
        print(f"입력 파일 없음: {args.article_dir}/{args.pattern}")
        return

    print(f"총 {len(files)}개 파일 처리 시작")
    for in_path in files:
        # kfa_100050.jsonl -> kfa_100050_day.jsonl
        out_name = in_path.stem + "_day.jsonl"
        out_path = args.output_dir / out_name
        print(f"[{in_path.name}] -> {out_path}")
        try:
            process_file(in_path, out_path, splitter, args.model, args.overwrite)
        except Exception as e:
            print(f"  ! 파일 처리 실패: {e}")


if __name__ == "__main__":
    main()