"""평가 점수 산출 모듈.

이 모듈은 평가셋의 ideal 값과 baseline 실행 로그의 actual 값을 비교한다.
검색 metric은 결정론적으로 계산하고, 생성/대화 metric은 필요한 경우 LLM judge를
호출한다. 검색이 없는 baseline이나 해당되지 않는 metric은 명시적으로 N/A 처리한다.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .core.baselines import get_baseline
from .core.llm_judge import judge_metric
from .core.progress import progress_iter
from .core.retrieval_metrics import mrr_at_k, ndcg_at_k, recall_at_k
from .core.schemas import (
    MetricResult,
    ScoreRecord,
    dataclass_to_dict,
    eval_item_from_dict,
    metric_to_dict,
    read_jsonl,
    run_log_from_dict,
    write_jsonl,
)


EVAL_DIR = Path(__file__).resolve().parent
DEFAULT_EVAL_SET = EVAL_DIR / "data" / "eval_set.jsonl"
DEFAULT_SCORE_DIR = EVAL_DIR / "scores"
DEFAULT_RETRIEVAL_METRIC_K = 10


def score_run(
    eval_set_path: Path,
    run_path: Path,
    output_path: Path,
    *,
    judge_model: str = "gpt-4o",
    skip_llm_judge: bool = False,
) -> list[ScoreRecord]:
    """실행 로그 전체를 채점하고 JSONL로 저장한다."""
    eval_items = {item.eval_id: item for item in (eval_item_from_dict(row) for row in read_jsonl(eval_set_path))}
    run_logs = [run_log_from_dict(row) for row in read_jsonl(run_path)]

    records: list[ScoreRecord] = []
    for log in progress_iter(run_logs, total=len(run_logs), desc=f"{run_path.stem} 채점", unit="문항"):
        item = eval_items.get(log.eval_id)
        if item is None:
            records.append(
                ScoreRecord(
                    run_id=log.run_id,
                    baseline=log.baseline,
                    eval_id=log.eval_id,
                    metrics={},
                    error="평가셋에서 eval_id를 찾을 수 없습니다.",
                )
            )
            continue
        records.append(score_one(item, log, judge_model=judge_model, skip_llm_judge=skip_llm_judge))

    write_jsonl(output_path, [dataclass_to_dict(record) for record in records])
    return records


def score_one(item: Any, log: Any, *, judge_model: str, skip_llm_judge: bool) -> ScoreRecord:
    """평가 문항 하나와 실행 로그 하나를 비교해 metric을 계산한다."""
    metrics: dict[str, dict[str, Any]] = {}
    judge_reasons: dict[str, str] = {}

    for name, result in compute_retrieval_metric_results(item, log).items():
        metrics[name] = metric_to_dict(result)

    llm_metric_names = _llm_metric_names_for_item(item, log.baseline)

    if skip_llm_judge:
        for name in llm_metric_names:
            metrics[name] = metric_to_dict(MetricResult.not_applicable("LLM judge 실행을 건너뜀"))
    elif log.error:
        for name in llm_metric_names:
            metrics[name] = metric_to_dict(MetricResult.not_applicable("baseline 실행 오류"))
    else:
        context = "\n\n".join(doc.get("text", "") for doc in log.retrieved_context)
        answer = log.generated_answer or ""
        for metric_name in llm_metric_names:
            if metric_name == "answer_correctness" and not item.gold_answer:
                metrics[metric_name] = metric_to_dict(MetricResult.not_applicable("gold_answer 없음"))
                continue
            judged = judge_metric(
                metric_name,
                question=item.question,
                generated_answer=answer,
                retrieved_context=context,
                gold_answer=item.gold_answer,
                gold_evidence=getattr(item, "gold_evidence", None),
                dialogue_history=item.dialogue_history,
                model=judge_model,
            )
            metrics[metric_name] = metric_to_dict(MetricResult.of(judged["score"]))
            judge_reasons[metric_name] = judged["reason"]

    return ScoreRecord(
        run_id=log.run_id,
        baseline=log.baseline,
        eval_id=item.eval_id,
        metrics=metrics,
        judge_reasons=judge_reasons,
        error=log.error,
    )


def compute_retrieval_metric_results(item: Any, log: Any, *, k: int | None = None) -> dict[str, MetricResult]:
    """검색 metric들을 계산하거나 N/A로 표시한다."""
    baseline = get_baseline(log.baseline)
    metric_k = _resolve_metric_k(baseline, log, k)
    if not baseline.uses_retrieval:
        return _retrieval_na(metric_k, "검색을 수행하지 않는 baseline")
    if not item.answerable:
        return _retrieval_na(metric_k, "unanswerable 문항")
    if log.error:
        return _retrieval_na(metric_k, "baseline 실행 오류")

    retrieved_keys = [doc.get("doc_key", "") for doc in log.retrieved_context]
    gold_keys = item.gold_doc_keys
    return {
        f"recall@{metric_k}": MetricResult.of(recall_at_k(gold_keys, retrieved_keys, metric_k)),
        f"ndcg@{metric_k}": MetricResult.of(ndcg_at_k(gold_keys, retrieved_keys, metric_k)),
        f"mrr@{metric_k}": MetricResult.of(mrr_at_k(gold_keys, retrieved_keys, metric_k)),
    }


def _resolve_metric_k(baseline: Any, log: Any, k: int | None) -> int:
    """metric 이름에 사용할 k값을 정한다."""
    if k is not None:
        return max(int(k), 1)
    return max(baseline.retrieval_k, len(log.retrieved_context), DEFAULT_RETRIEVAL_METRIC_K)


def _retrieval_na(metric_k: int, reason: str) -> dict[str, MetricResult]:
    """검색 metric 3종을 같은 이유로 N/A 처리한다."""
    return {
        f"recall@{metric_k}": MetricResult.not_applicable(reason),
        f"ndcg@{metric_k}": MetricResult.not_applicable(reason),
        f"mrr@{metric_k}": MetricResult.not_applicable(reason),
    }


def _llm_metric_names_for_item(item: Any, baseline_name: str) -> list[str]:
    """문항 성격에 따라 적용할 LLM judge metric 목록을 고른다."""
    baseline = get_baseline(baseline_name)
    names = ["answer_relevance", "persona_adherence"]
    if item.answerable:
        if baseline.uses_retrieval:
            names.append("faithfulness")
        names.append("answer_correctness")
    else:
        names.append("negative_rejection")
    if item.dialogue_history or item.item_type == "multi_turn_followup":
        names.append("multi_turn_context_utilization")
    return names


def main() -> None:
    """CLI 인자를 받아 실행 로그를 채점한다."""
    parser = argparse.ArgumentParser(description="SillokChatbot 실행 로그 채점기")
    parser.add_argument("--eval-set", type=Path, default=DEFAULT_EVAL_SET, help="평가셋 JSONL")
    parser.add_argument("--run", type=Path, required=True, help="실행 로그 JSONL")
    parser.add_argument("--out", type=Path, default=None, help="저장할 score JSONL")
    parser.add_argument("--judge-model", default="gpt-4o", help="LLM judge 모델")
    parser.add_argument("--skip-llm-judge", action="store_true", help="LLM judge 없이 retrieval만 채점")
    args = parser.parse_args()

    out = args.out or (DEFAULT_SCORE_DIR / args.run.name)
    records = score_run(
        args.eval_set,
        args.run,
        out,
        judge_model=args.judge_model,
        skip_llm_judge=args.skip_llm_judge,
    )
    print(f"완료: {len(records)}개 점수 기록을 {out}에 저장했습니다.")


if __name__ == "__main__":
    main()
