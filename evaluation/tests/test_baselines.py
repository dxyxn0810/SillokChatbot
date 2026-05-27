"""baseline 설정 테스트.

이 테스트는 query rewriting baseline이 등록되어 있고, 검색 질의 재작성 플래그가
켜져 있는지 확인한다.
"""

import unittest

from evaluation.core.baselines import get_baseline


class BaselinesTest(unittest.TestCase):
    """baseline 설정값을 검증한다."""

    def test_rag_rewrite_is_registered(self):
        """query rewriting baseline은 검색을 사용하고 rewrite 플래그가 켜져 있어야 한다."""
        baseline = get_baseline("rag_rewrite")
        self.assertTrue(baseline.uses_retrieval)
        self.assertTrue(baseline.use_query_rewrite)
        self.assertFalse(baseline.use_cot)

    def test_rag_article_rewrite_alias_is_kept(self):
        """이전 이름도 과거 실험 호환을 위해 유지한다."""
        baseline = get_baseline("rag_article_rewrite")
        self.assertTrue(baseline.uses_retrieval)
        self.assertTrue(baseline.use_query_rewrite)
        self.assertFalse(baseline.use_cot)

    def test_rag_full_rewrite_is_registered(self):
        """full rewrite baseline은 full 설정과 query rewrite를 함께 사용한다."""
        baseline = get_baseline("rag_full_rewrite")
        self.assertTrue(baseline.uses_retrieval)
        self.assertTrue(baseline.use_query_rewrite)
        self.assertTrue(baseline.use_cot)
        self.assertEqual(baseline.data_level, "wikipedia")


if __name__ == "__main__":
    unittest.main()
