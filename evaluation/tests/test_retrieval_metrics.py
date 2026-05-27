"""검색 metric 테스트.

이 테스트는 Recall@k, NDCG@k, MRR@k가 간단한 gold/retrieved 목록에서
예상한 값을 내는지 검증한다.
"""

import unittest

from evaluation.core.retrieval_metrics import mrr_at_k, ndcg_at_k, recall_at_k


class RetrievalMetricsTest(unittest.TestCase):
    """검색 metric 수식을 검증한다."""

    def test_recall_at_k(self):
        """상위 k개 안에 gold가 들어온 비율을 계산한다."""
        self.assertEqual(recall_at_k(["a", "b"], ["x", "a", "y"], 3), 0.5)

    def test_mrr_at_k(self):
        """첫 gold 문서 순위의 reciprocal rank를 계산한다."""
        self.assertEqual(mrr_at_k(["a"], ["x", "a", "y"], 3), 0.5)

    def test_ndcg_at_k_prefers_higher_rank(self):
        """gold가 위에 있을수록 NDCG가 높다."""
        high = ndcg_at_k(["a"], ["a", "x"], 2)
        low = ndcg_at_k(["a"], ["x", "a"], 2)
        self.assertGreater(high, low)

    def test_multi_chunk_recall(self):
        """gold 문서가 여러 개여도 Recall@k를 비율로 계산한다."""
        self.assertEqual(recall_at_k(["a", "b"], ["x", "a", "y"], 3), 0.5)


if __name__ == "__main__":
    unittest.main()
