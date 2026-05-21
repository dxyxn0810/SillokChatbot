"""
article/kfa_*.jsonl, daily_summary/kfa_*_day.jsonl,
kfa_month.jsonl, kfa_year.jsonl 을 모두 합쳐 kfa.jsonl 로 저장한다.

사용법:
    python merge.py
    # 또는
    python merge.py --article-dir ./article --daily-summary-dir ./daily_summary \\
                    --monthly-file ./kfa_month.jsonl --yearly-file ./kfa_year.jsonl \\
                    --output-file ./kfa.jsonl
"""

import argparse
import json
from pathlib import Path
from typing import List


def iter_jsonl_lines(path: Path):
    """jsonl 파일의 유효한 라인을 그대로 yield (한 줄 = 하나의 Document JSON)."""
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n").rstrip("\r")
            if not line.strip():
                continue
            # 유효한 JSON 인지 가볍게 검증 (실패 시 스킵)
            try:
                json.loads(line)
            except json.JSONDecodeError as e:
                print(f"    ! JSON 파싱 실패, 스킵: {path.name} -> {e}")
                continue
            yield line


def collect_files(
    article_dir: Path,
    daily_dir: Path,
    monthly_file: Path,
    yearly_file: Path,
) -> List[Path]:
    """합칠 파일들을 article -> daily -> monthly -> yearly 순서로 모은다."""
    files: List[Path] = []

    if article_dir.is_dir():
        files += sorted(article_dir.glob("kfa_*.jsonl"))
    else:
        print(f"  - article 폴더 없음: {article_dir}")

    if daily_dir.is_dir():
        files += sorted(daily_dir.glob("kfa_*_day.jsonl"))
    else:
        print(f"  - daily_summary 폴더 없음: {daily_dir}")

    for f in (monthly_file, yearly_file):
        if f.exists():
            files.append(f)
        else:
            print(f"  - 파일 없음: {f}")

    return files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--article-dir", default="article", type=Path)
    parser.add_argument("--daily-summary-dir", default="daily_summary", type=Path)
    parser.add_argument("--monthly-file", default="kfa_month.jsonl", type=Path)
    parser.add_argument("--yearly-file", default="kfa_year.jsonl", type=Path)
    parser.add_argument("--output-file", default="kfa.jsonl", type=Path)
    args = parser.parse_args()

    files = collect_files(
        args.article_dir, args.daily_summary_dir, args.monthly_file, args.yearly_file
    )
    if not files:
        print("합칠 파일이 없습니다.")
        return

    print(f"총 {len(files)}개 파일 합치기 시작 -> {args.output_file}")
    args.output_file.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    with args.output_file.open("w", encoding="utf-8") as fout:
        for path in files:
            count = 0
            for line in iter_jsonl_lines(path):
                fout.write(line + "\n")
                count += 1
            total += count
            print(f"  · {path}  ({count} docs)")

    print(f"\n완료: 총 {total}개 Document -> {args.output_file}")


if __name__ == "__main__":
    main()