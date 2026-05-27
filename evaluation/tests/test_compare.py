"""baseline 비교표 생성 테스트.

이 테스트는 OpenAI나 FAISS를 호출하지 않고, 요약 dict를 Markdown 표로 바꾸는
순수 함수가 baseline 순서와 N/A 표시를 올바르게 처리하는지 확인한다.
"""

from pathlib import Path
import tempfile
import unittest

from evaluation.compare import (
    CONVERSATIONAL_METRICS,
    DEFAULT_COMPARE_BASELINES,
    GENERATION_METRICS,
    RETRIEVAL_METRICS,
    read_summary_json,
    summary_to_markdown_table,
    summary_to_plot,
    summary_to_plot_values,
    write_compare_outputs,
)


class CompareTableTest(unittest.TestCase):
    """비교표 렌더링 규칙을 검증한다."""

    def test_table_uses_recommended_baseline_order(self):
        """summary 입력 순서와 무관하게 추천 baseline 순서로 표를 만든다."""
        summary = {
            "rag_full": {"items": 5, "metrics": {"answer_relevance": {"mean": 4.0}}},
            "api_only": {"items": 5, "metrics": {"answer_relevance": {"mean": 2.0}}},
        }

        table = summary_to_markdown_table(summary)
        lines = table.splitlines()

        self.assertTrue(lines[2].startswith("| api_only |"))
        self.assertTrue(lines[3].startswith("| rag_full |"))
        self.assertLess(DEFAULT_COMPARE_BASELINES.index("api_only"), DEFAULT_COMPARE_BASELINES.index("rag_full"))

    def test_default_baselines_use_rag_rewrite_instead_of_article_cot(self):
        """기본 비교 세트는 article CoT 대신 rag rewrite를 사용한다."""
        self.assertIn("rag_rewrite", DEFAULT_COMPARE_BASELINES)
        self.assertNotIn("rag_article_rewrite", DEFAULT_COMPARE_BASELINES)
        self.assertNotIn("rag_article_cot", DEFAULT_COMPARE_BASELINES)

    def test_metric_groups_follow_evaluation_axes(self):
        """그래프 metric 그룹은 retrieval/generation/conversation 축을 따른다."""
        self.assertEqual(RETRIEVAL_METRICS, ("recall@10", "ndcg@10", "mrr@10"))
        self.assertEqual(
            GENERATION_METRICS,
            ("faithfulness", "answer_relevance", "answer_correctness", "negative_rejection"),
        )
        self.assertEqual(
            CONVERSATIONAL_METRICS,
            ("persona_adherence", "multi_turn_context_utilization"),
        )

    def test_missing_metric_is_na(self):
        """없는 metric은 N/A로 표시한다."""
        summary = {
            "api_only": {
                "items": 5,
                "metrics": {
                    "answer_relevance": {"mean": 2.12345},
                },
            }
        }

        table = summary_to_markdown_table(summary)

        self.assertIn("| api_only | 5 | N/A |", table)
        self.assertIn("2.123", table)

    def test_plot_values_use_none_for_na(self):
        """그래프용 값 행렬에서는 없는 metric을 None으로 둔다."""
        summary = {
            "api_only": {
                "items": 5,
                "metrics": {
                    "answer_relevance": {"mean": 2.0},
                    "recall@10": {"mean": None},
                },
            }
        }

        baselines, metrics, values = summary_to_plot_values(
            summary,
            baseline_order=["api_only"],
            metrics=["recall@10", "answer_relevance"],
        )

        self.assertEqual(baselines, ["api_only"])
        self.assertEqual(metrics, ["recall@10", "answer_relevance"])
        self.assertEqual(values, [[None, 2.0]])

    def test_read_summary_json(self):
        """기존 summary JSON을 다시 읽을 수 있다."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "summary.json"
            path.write_text('{"api_only": {"items": 5, "metrics": {}}}', encoding="utf-8")

            summary = read_summary_json(path)

        self.assertEqual(summary["api_only"]["items"], 5)

    def test_summary_to_plot_smoke(self):
        """matplotlib가 있는 환경에서는 임시 PNG 그래프를 저장한다."""
        summary = {
            "api_only": {"items": 5, "metrics": {"answer_relevance": {"mean": 2.0}}},
            "rag_full": {"items": 5, "metrics": {"answer_relevance": {"mean": 4.0}}},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "plot.png"
            summary_to_plot(summary, out_path, metrics=["answer_relevance"])

            self.assertTrue(out_path.exists())
            self.assertGreater(out_path.stat().st_size, 0)

    def test_write_compare_outputs_creates_three_axis_plots(self):
        """비교 출력은 retrieval/generation/conversation 그래프를 따로 저장한다."""
        summary = {
            "api_only": {
                "items": 5,
                "metrics": {
                    "answer_relevance": {"mean": 2.0},
                    "negative_rejection": {"mean": 4.0},
                    "persona_adherence": {"mean": 3.0},
                },
            },
            "rag_article": {
                "items": 5,
                "metrics": {
                    "recall@10": {"mean": 0.5},
                    "ndcg@10": {"mean": 0.4},
                    "mrr@10": {"mean": 0.3},
                    "faithfulness": {"mean": 4.0},
                    "answer_relevance": {"mean": 4.2},
                    "answer_correctness": {"mean": 3.8},
                    "persona_adherence": {"mean": 4.5},
                    "multi_turn_context_utilization": {"mean": 3.0},
                },
            },
        }
        table = summary_to_markdown_table(summary, baseline_order=["api_only", "rag_article"])

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            out_json = tmp / "summary.json"
            out_md = tmp / "table.md"
            retrieval_plot = tmp / "retrieval.png"
            generation_plot = tmp / "generation.png"
            conversation_plot = tmp / "conversation.png"
            write_compare_outputs(
                summary,
                table,
                out_json=out_json,
                out_md=out_md,
                out_plot=None,
                out_retrieval_plot=retrieval_plot,
                out_generation_plot=generation_plot,
                out_conversation_plot=conversation_plot,
                baseline_order=["api_only", "rag_article"],
            )

            for path in (out_json, out_md, retrieval_plot, generation_plot, conversation_plot):
                self.assertTrue(path.exists())
                self.assertGreater(path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
