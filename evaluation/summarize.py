"""평가 결과 요약 모듈.

이 모듈은 여러 score JSONL 파일을 읽어 baseline별 metric 평균, 평가 개수,
N/A 개수를 집계한다. 결과는 사람이 보기 쉬운 JSON 리포트로
`evaluation/reports/` 아래에 저장한다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .core.schemas import read_jsonl


EVAL_DIR = Path(__file__).resolve().parent
DEFAULT_REPORT = EVAL_DIR / "reports" / "summary.json"


def summarize_scores(score_paths: list[Path]) -> dict[str, Any]:
    """score JSONL 경로들을 baseline별 요약 dict로 집계한다."""
    summary: dict[str, Any] = {}
    for path in score_paths:
        for row in read_jsonl(path):
            baseline = row.get("baseline", "unknown")
            bucket = summary.setdefault(baseline, {"items": 0, "metrics": {}})
            bucket["items"] += 1
            for metric_name, metric in (row.get("metrics") or {}).items():
                metric_bucket = bucket["metrics"].setdefault(
                    metric_name,
                    {"count": 0, "na": 0, "sum": 0.0, "mean": None},
                )
                if metric.get("na"):
                    metric_bucket["na"] += 1
                    continue
                value = metric.get("value")
                if value is None:
                    metric_bucket["na"] += 1
                    continue
                metric_bucket["count"] += 1
                metric_bucket["sum"] += float(value)

    for bucket in summary.values():
        for metric_bucket in bucket["metrics"].values():
            count = metric_bucket["count"]
            metric_bucket["mean"] = metric_bucket["sum"] / count if count else None
    return summary


def write_summary(summary: dict[str, Any], output_path: Path) -> None:
    """요약 리포트를 JSON 파일로 저장한다."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    """CLI 인자를 받아 score 파일들을 요약한다."""
    parser = argparse.ArgumentParser(description="SillokChatbot 평가 요약 리포트 생성기")
    parser.add_argument("--scores", nargs="+", type=Path, required=True, help="score JSONL 파일들")
    parser.add_argument("--out", type=Path, default=DEFAULT_REPORT, help="저장할 summary JSON")
    args = parser.parse_args()

    summary = summarize_scores(args.scores)
    write_summary(summary, args.out)
    print(f"완료: 요약 리포트를 {args.out}에 저장했습니다.")


if __name__ == "__main__":
    main()
