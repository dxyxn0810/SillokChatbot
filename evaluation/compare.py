"""baseline 비교 실행 모듈.

이 모듈은 추천 baseline 5개를 같은 평가셋으로 실행하고, 채점하고, 요약한 뒤
사람이 바로 읽을 수 있는 Markdown 표와 3축 분리 그래프를 만든다. 기존
runner/score/summarize 모듈을 조합하며, 원본 챗봇 코드나 데이터는 수정하지 않는다.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from .core.progress import progress_iter
from .runner import DEFAULT_RUN_DIR, ROOT_DIR, run_baseline
from .score import DEFAULT_SCORE_DIR, score_run
from .summarize import DEFAULT_REPORT, summarize_scores, write_summary


EVAL_DIR = Path(__file__).resolve().parent
DEFAULT_COMPARE_LIMIT = 5
DEFAULT_COMPARE_BASELINES = (
    "api_only",
    "rag_article",
    "rag_rewrite",
    "rag_full",
    "rag_full_rewrite",
)
DEFAULT_TABLE_METRICS = (
    "recall@10",
    "ndcg@10",
    "mrr@10",
    "faithfulness",
    "answer_relevance",
    "answer_correctness",
    "negative_rejection",
    "persona_adherence",
    "multi_turn_context_utilization",
)
RETRIEVAL_METRICS = ("recall@10", "ndcg@10", "mrr@10")
GENERATION_METRICS = (
    "faithfulness",
    "answer_relevance",
    "answer_correctness",
    "negative_rejection",
)
CONVERSATIONAL_METRICS = (
    "persona_adherence",
    "multi_turn_context_utilization",
)
DEFAULT_COMPARE_RETRIEVAL_PLOT = EVAL_DIR / "reports" / "compare5_retrieval_metrics.png"
DEFAULT_COMPARE_GENERATION_PLOT = EVAL_DIR / "reports" / "compare5_generation_metrics.png"
DEFAULT_COMPARE_CONVERSATION_PLOT = EVAL_DIR / "reports" / "compare5_conversation_metrics.png"


def compare_baselines(
    eval_set_path: Path,
    *,
    baselines: list[str],
    limit: int | None,
    judge_model: str,
    run_dir: Path,
    score_dir: Path,
    out_json: Path,
    out_md: Path,
    out_plot: Path | None,
    out_retrieval_plot: Path | None,
    out_generation_plot: Path | None,
    out_conversation_plot: Path | None,
    faiss_dir: Path | None,
    skip_llm_judge: bool,
    verbose: bool,
) -> tuple[dict[str, Any], str]:
    """여러 baseline을 실행·채점·요약하고 Markdown 표를 반환한다."""
    tag = f"compare{limit}" if limit is not None else "compare_all"
    score_paths: list[Path] = []

    for baseline in progress_iter(baselines, total=len(baselines), desc="baseline 비교", unit="baseline"):
        run_path = run_dir / f"{baseline}_{tag}.jsonl"
        score_path = score_dir / f"{baseline}_{tag}.jsonl"
        run_baseline(
            eval_set_path,
            run_path,
            baseline=baseline,
            faiss_dir=faiss_dir,
            limit=limit,
            verbose=verbose,
        )
        score_run(
            eval_set_path,
            run_path,
            score_path,
            judge_model=judge_model,
            skip_llm_judge=skip_llm_judge,
        )
        score_paths.append(score_path)

    summary = summarize_scores(score_paths)
    table = summary_to_markdown_table(summary, baseline_order=baselines)
    write_compare_outputs(
        summary,
        table,
        out_json=out_json,
        out_md=out_md,
        out_plot=out_plot,
        out_retrieval_plot=out_retrieval_plot,
        out_generation_plot=out_generation_plot,
        out_conversation_plot=out_conversation_plot,
        baseline_order=baselines,
    )
    return summary, table


def write_compare_outputs(
    summary: dict[str, Any],
    table: str,
    *,
    out_json: Path,
    out_md: Path,
    out_plot: Path | None,
    out_retrieval_plot: Path | None = DEFAULT_COMPARE_RETRIEVAL_PLOT,
    out_generation_plot: Path | None = DEFAULT_COMPARE_GENERATION_PLOT,
    out_conversation_plot: Path | None = DEFAULT_COMPARE_CONVERSATION_PLOT,
    baseline_order: list[str] | tuple[str, ...],
) -> None:
    """비교 결과 JSON, Markdown 표, 그래프를 저장한다."""
    write_summary(summary, out_json)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(table + "\n", encoding="utf-8")
    if out_retrieval_plot is not None:
        summary_to_plot(
            summary,
            out_retrieval_plot,
            baseline_order=baseline_order,
            metrics=RETRIEVAL_METRICS,
            title="A. Retrieval Metrics",
            ylabel="Mean score (0-1)",
            ylim=(0, 1),
        )
    if out_generation_plot is not None:
        summary_to_plot(
            summary,
            out_generation_plot,
            baseline_order=baseline_order,
            metrics=GENERATION_METRICS,
            title="B. Generation Metrics",
            ylabel="Mean rubric score (1-5)",
            ylim=(0, 5),
        )
    if out_conversation_plot is not None:
        summary_to_plot(
            summary,
            out_conversation_plot,
            baseline_order=baseline_order,
            metrics=CONVERSATIONAL_METRICS,
            title="C. Conversational Metrics",
            ylabel="Mean rubric score (1-5)",
            ylim=(0, 5),
        )
    if out_plot is not None:
        summary_to_plot(
            summary,
            out_plot,
            baseline_order=baseline_order,
            title="SillokChatbot Baseline Comparison",
            ylabel="Mean score (retrieval: 0-1, judge: 1-5)",
            ylim=(0, 5),
        )


def summary_to_markdown_table(
    summary: dict[str, Any],
    *,
    baseline_order: list[str] | tuple[str, ...] = DEFAULT_COMPARE_BASELINES,
    metrics: list[str] | tuple[str, ...] = DEFAULT_TABLE_METRICS,
) -> str:
    """summary dict를 baseline 비교용 Markdown 표로 변환한다."""
    headers = ["baseline", "items", *metrics]
    rows = []
    for baseline in baseline_order:
        bucket = summary.get(baseline)
        if not bucket:
            continue
        metric_buckets = bucket.get("metrics") or {}
        rows.append(
            [
                baseline,
                str(bucket.get("items", 0)),
                *[_format_metric_mean(metric_buckets.get(metric)) for metric in metrics],
            ]
        )

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def summary_to_plot_values(
    summary: dict[str, Any],
    *,
    baseline_order: list[str] | tuple[str, ...] = DEFAULT_COMPARE_BASELINES,
    metrics: list[str] | tuple[str, ...] = DEFAULT_TABLE_METRICS,
) -> tuple[list[str], list[str], list[list[float | None]]]:
    """summary dict에서 그래프에 사용할 baseline, metric, 값 행렬을 만든다."""
    baselines = [baseline for baseline in baseline_order if baseline in summary]
    values: list[list[float | None]] = []
    for baseline in baselines:
        metric_buckets = (summary.get(baseline) or {}).get("metrics") or {}
        row = []
        for metric in metrics:
            bucket = metric_buckets.get(metric)
            mean = bucket.get("mean") if bucket else None
            row.append(float(mean) if mean is not None else None)
        values.append(row)
    return baselines, list(metrics), values


def summary_to_plot(
    summary: dict[str, Any],
    out_path: Path,
    *,
    baseline_order: list[str] | tuple[str, ...] = DEFAULT_COMPARE_BASELINES,
    metrics: list[str] | tuple[str, ...] = DEFAULT_TABLE_METRICS,
    title: str = "SillokChatbot Baseline Comparison",
    ylabel: str = "Mean score",
    ylim: tuple[float, float] | None = None,
) -> None:
    """summary dict를 baseline 비교용 grouped bar chart PNG로 저장한다."""
    baselines, metric_names, values = summary_to_plot_values(
        summary,
        baseline_order=baseline_order,
        metrics=metrics,
    )
    if not baselines:
        raise ValueError("그래프를 만들 baseline 결과가 없습니다.")

    plt = _load_pyplot()
    import math

    x_positions = list(range(len(metric_names)))
    width = min(0.16, 0.8 / max(len(baselines), 1))
    fig_width = max(12.0, len(metric_names) * 1.35)
    fig, ax = plt.subplots(figsize=(fig_width, 6.5))

    for baseline_index, baseline in enumerate(baselines):
        offset = (baseline_index - (len(baselines) - 1) / 2) * width
        y_values = [math.nan if value is None else value for value in values[baseline_index]]
        ax.bar([x + offset for x in x_positions], y_values, width=width, label=baseline)

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(metric_names, rotation=35, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=min(len(baselines), 3))
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _load_pyplot() -> Any:
    """headless 환경에서 matplotlib pyplot을 안전하게 불러온다."""
    os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="sillok-mpl-"))
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("그래프 생성을 위해 matplotlib가 필요합니다. `pip install matplotlib`를 실행하세요.") from exc
    return plt


def _format_metric_mean(metric_bucket: Any) -> str:
    """metric 평균을 표에 넣을 문자열로 바꾼다."""
    if not metric_bucket:
        return "N/A"
    mean = metric_bucket.get("mean")
    if mean is None:
        return "N/A"
    return f"{float(mean):.3f}"


def _parse_baselines(raw: str) -> list[str]:
    """CLI 문자열에서 baseline 목록을 파싱한다."""
    baselines = [part.strip() for part in raw.split(",") if part.strip()]
    if not baselines:
        raise ValueError("baseline 목록이 비어 있습니다.")
    return baselines


def read_summary_json(path: Path) -> dict[str, Any]:
    """summary JSON 파일을 읽는다."""
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    """CLI 인자를 받아 추천 baseline 비교표를 생성한다."""
    parser = argparse.ArgumentParser(description="SillokChatbot baseline 5종 비교 실행기")
    parser.add_argument("--eval-set", type=Path, default=None, help="평가셋 JSONL")
    parser.add_argument("--from-summary", type=Path, default=None, help="기존 summary JSON에서 표와 그래프만 다시 생성")
    parser.add_argument("--limit", type=int, default=DEFAULT_COMPARE_LIMIT, help="앞에서부터 사용할 문항 수")
    parser.add_argument("--judge-model", default="gpt-4o", help="LLM judge 모델")
    parser.add_argument(
        "--baselines",
        default=",".join(DEFAULT_COMPARE_BASELINES),
        help="쉼표로 구분한 baseline 목록",
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR, help="run log 저장 폴더")
    parser.add_argument("--score-dir", type=Path, default=DEFAULT_SCORE_DIR, help="score 저장 폴더")
    parser.add_argument(
        "--out-json",
        type=Path,
        default=DEFAULT_REPORT.parent / "compare5_summary.json",
        help="summary JSON 저장 경로",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=EVAL_DIR / "reports" / "compare5_table.md",
        help="Markdown 표 저장 경로",
    )
    parser.add_argument(
        "--out-plot",
        type=Path,
        default=None,
        help="legacy 통합 그래프 PNG 저장 경로. 지정한 경우에만 생성",
    )
    parser.add_argument(
        "--out-retrieval-plot",
        type=Path,
        default=DEFAULT_COMPARE_RETRIEVAL_PLOT,
        help="Retrieval 축 그래프 PNG 저장 경로",
    )
    parser.add_argument(
        "--out-generation-plot",
        type=Path,
        default=DEFAULT_COMPARE_GENERATION_PLOT,
        help="Generation 축 그래프 PNG 저장 경로",
    )
    parser.add_argument(
        "--out-conversation-plot",
        type=Path,
        default=DEFAULT_COMPARE_CONVERSATION_PLOT,
        help="Conversational 축 그래프 PNG 저장 경로",
    )
    parser.add_argument("--faiss", type=Path, default=ROOT_DIR / "faiss", help="FAISS 인덱스 폴더")
    parser.add_argument("--skip-llm-judge", action="store_true", help="LLM judge 없이 retrieval만 채점")
    parser.add_argument("--verbose", action="store_true", help="기존 챗봇의 verbose 출력을 켠다")
    args = parser.parse_args()
    baselines = _parse_baselines(args.baselines)

    if args.from_summary is not None:
        summary = read_summary_json(args.from_summary)
        table = summary_to_markdown_table(summary, baseline_order=baselines)
        write_compare_outputs(
            summary,
            table,
            out_json=args.out_json,
            out_md=args.out_md,
            out_plot=args.out_plot,
            out_retrieval_plot=args.out_retrieval_plot,
            out_generation_plot=args.out_generation_plot,
            out_conversation_plot=args.out_conversation_plot,
            baseline_order=baselines,
        )
    else:
        if args.eval_set is None:
            parser.error("--eval-set 또는 --from-summary 중 하나가 필요합니다.")
        _, table = compare_baselines(
            args.eval_set,
            baselines=baselines,
            limit=args.limit,
            judge_model=args.judge_model,
            run_dir=args.run_dir,
            score_dir=args.score_dir,
            out_json=args.out_json,
            out_md=args.out_md,
            out_plot=args.out_plot,
            out_retrieval_plot=args.out_retrieval_plot,
            out_generation_plot=args.out_generation_plot,
            out_conversation_plot=args.out_conversation_plot,
            faiss_dir=args.faiss,
            skip_llm_judge=args.skip_llm_judge,
            verbose=args.verbose,
        )
    print(table)
    print(f"\n완료: 비교표를 {args.out_md}에, 요약 JSON을 {args.out_json}에 저장했습니다.")
    print(f"Retrieval 그래프: {args.out_retrieval_plot}")
    print(f"Generation 그래프: {args.out_generation_plot}")
    print(f"Conversational 그래프: {args.out_conversation_plot}")
    if args.out_plot is not None:
        print(f"통합 그래프: {args.out_plot}")


if __name__ == "__main__":
    main()
