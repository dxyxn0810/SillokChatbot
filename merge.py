"""
`daily_summary`, `monthly_summary`, `yearly_summary` 폴더의 모든 jsonl 데이터를 모으고,
마지막에 `kfa.jsonl` 파일 내용을 추가하여 최종 `data.jsonl` 파일로 저장합니다.

사용법:
    python merge.py
    # 또는
    python merge.py --daily-summary-dir ./daily_summary \\
                    --monthly-summary-dir ./monthly_summary \\
                    --yearly-summary-dir ./yearly_summary \\
                    --kfa-file ./kfa.jsonl --output-file ./data.jsonl
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
    monthly_dir: Path,
    yearly_dir: Path,
    kfa_file: Path,
) -> List[Path]:
    """합칠 파일들을 article -> daily_summary -> monthly_summary -> yearly_summary -> kfa.jsonl 순서로 모은다."""
    files: List[Path] = []

    if article_dir.is_dir():
        files += sorted(article_dir.glob("*.jsonl"))
    else:
        print(f"  - article 폴더 없음: {article_dir}")

    if daily_dir.is_dir():
        files += sorted(daily_dir.glob("*.jsonl"))
    else:
        print(f"  - daily_summary 폴더 없음: {daily_dir}")

    if monthly_dir.is_dir():
        files += sorted(monthly_dir.glob("*.jsonl"))
    else:
        print(f"  - monthly_summary 폴더 없음: {monthly_dir}")

    if yearly_dir.is_dir():
        files += sorted(yearly_dir.glob("*.jsonl"))
    else:
        print(f"  - yearly_summary 폴더 없음: {yearly_dir}")

    if kfa_file.exists():
        files.append(kfa_file)
    else:
        print(f"  - 파일 없음: {kfa_file}")

    return files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--article-dir", default="article", type=Path)
    parser.add_argument("--daily-summary-dir", default="daily_summary", type=Path)
    parser.add_argument("--monthly-summary-dir", default="monthly_summary", type=Path)
    parser.add_argument("--yearly-summary-dir", default="yearly_summary", type=Path)
    parser.add_argument("--kfa-file", default="kfa.jsonl", type=Path)
    parser.add_argument("--output-file", default="data.jsonl", type=Path)
    args = parser.parse_args()

    files = collect_files(
        args.article_dir,
        args.daily_summary_dir,
        args.monthly_summary_dir,
        args.yearly_summary_dir,
        args.kfa_file,
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