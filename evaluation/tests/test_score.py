"""채점 라우팅 테스트.

이 테스트는 검색이 없는 baseline과 unanswerable 문항에서 retrieval metric이
N/A 처리되는지 확인한다. OpenAI 호출 없이 순수 함수만 사용한다.
"""

import unittest

from evaluation.core.schemas import EvalItem, RunLog
from evaluation.score import compute_retrieval_metric_results, score_one


class ScoreRoutingTest(unittest.TestCase):
    """metric N/A 처리 규칙을 검증한다."""

    def test_api_only_retrieval_is_na(self):
        """검색 없는 baseline은 retrieval metric이 N/A다."""
        item = EvalItem(eval_id="e1", item_type="grounded_single", question="질문", gold_doc_keys=["a"])
        log = RunLog(run_id="r1", baseline="api_only", eval_id="e1", question="질문")
        metrics = compute_retrieval_metric_results(item, log)
        self.assertIn("recall@10", metrics)
        self.assertNotIn("recall@k", metrics)
        self.assertTrue(all(metric.na for metric in metrics.values()))

    def test_answerable_retrieval_scores(self):
        """검색 결과가 gold를 포함하면 retrieval 점수가 계산된다."""
        item = EvalItem(eval_id="e1", item_type="grounded_single", question="질문", gold_doc_keys=["a"])
        log = RunLog(
            run_id="r1",
            baseline="rag_article",
            eval_id="e1",
            question="질문",
            retrieved_context=[{"doc_key": "x"}, {"doc_key": "a"}],
        )
        metrics = compute_retrieval_metric_results(item, log, k=2)
        self.assertFalse(metrics["recall@2"].na)
        self.assertEqual(metrics["recall@2"].value, 1.0)

    def test_api_only_has_no_faithfulness_metric(self):
        """검색 없는 baseline에는 faithfulness judge를 적용하지 않는다."""
        item = EvalItem(
            eval_id="e1",
            item_type="grounded_single",
            question="질문",
            gold_doc_keys=["a"],
            gold_answer="정답",
        )
        log = RunLog(
            run_id="r1",
            baseline="api_only",
            eval_id="e1",
            question="질문",
            generated_answer="답변",
        )
        record = score_one(item, log, judge_model="gpt-4o", skip_llm_judge=True)
        self.assertNotIn("faithfulness", record.metrics)

    def test_multi_turn_metric_is_included(self):
        """multi-turn 문항에는 이전 턴 활용 metric이 포함된다."""
        item = EvalItem(
            eval_id="e1",
            item_type="multi_turn_followup",
            question="그 일은 왜 문제가 되었나요?",
            dialogue_history=[
                {"role": "user", "content": "정발이 무엇을 했나요?"},
                {"role": "assistant", "content": "정발은 국상 중 풍악을 연주했습니다."},
            ],
            gold_doc_keys=["a"],
            gold_answer="국상 중 풍악을 연주했기 때문입니다.",
        )
        log = RunLog(
            run_id="r1",
            baseline="rag_article",
            eval_id="e1",
            question=item.question,
            generated_answer="답변",
            retrieved_context=[{"doc_key": "a", "text": "근거"}],
        )
        record = score_one(item, log, judge_model="gpt-4o", skip_llm_judge=True)
        self.assertIn("multi_turn_context_utilization", record.metrics)

    def test_unanswerable_uses_negative_rejection(self):
        """unanswerable 문항은 retrieval N/A와 negative rejection metric을 가진다."""
        item = EvalItem(
            eval_id="e1",
            item_type="unanswerable",
            question="단종이 임진왜란 때 어떤 명령을 내렸나요?",
            answerable=False,
            gold_doc_keys=[],
            gold_answer=None,
        )
        log = RunLog(
            run_id="r1",
            baseline="rag_article",
            eval_id="e1",
            question=item.question,
            generated_answer="자료에 없어 답하기 어렵습니다.",
            retrieved_context=[],
        )
        retrieval_metrics = compute_retrieval_metric_results(item, log)
        self.assertIn("recall@10", retrieval_metrics)
        self.assertNotIn("recall@k", retrieval_metrics)
        self.assertTrue(all(metric.na for metric in retrieval_metrics.values()))
        record = score_one(item, log, judge_model="gpt-4o", skip_llm_judge=True)
        self.assertIn("negative_rejection", record.metrics)
        self.assertNotIn("answer_correctness", record.metrics)

    def test_api_only_unanswerable_uses_negative_rejection(self):
        """api_only도 unanswerable 문항에서는 거부 능력을 평가한다."""
        item = EvalItem(
            eval_id="e1",
            item_type="unanswerable",
            question="문종 때 전기차가 처음 도입된 해는 언제인가요?",
            answerable=False,
            gold_doc_keys=[],
            gold_answer=None,
            gold_evidence="전기차는 수집된 조선 전기 corpus로 답할 수 없는 현대 개념이다.",
        )
        log = RunLog(
            run_id="r1",
            baseline="api_only",
            eval_id="e1",
            question=item.question,
            generated_answer="사료에 근거가 없어 답하기 어렵습니다.",
            retrieved_context=[],
        )
        retrieval_metrics = compute_retrieval_metric_results(item, log)
        self.assertTrue(all(metric.na for metric in retrieval_metrics.values()))
        record = score_one(item, log, judge_model="gpt-4o", skip_llm_judge=True)
        self.assertIn("negative_rejection", record.metrics)
        self.assertTrue(record.metrics["negative_rejection"]["na"])
        self.assertNotIn("answer_correctness", record.metrics)


if __name__ == "__main__":
    unittest.main()
