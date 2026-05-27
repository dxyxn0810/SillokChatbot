"""schema 검증 테스트.

이 테스트는 평가셋 record와 실행 로그 record가 필수 필드를 검사하는지 확인한다.
"""

import unittest

from evaluation.core.schemas import eval_item_from_dict, run_log_from_dict


class SchemasTest(unittest.TestCase):
    """평가 데이터 구조 검증을 확인한다."""

    def test_eval_item_requires_gold_for_answerable(self):
        """answerable 문항에는 gold_doc_keys가 있어야 한다."""
        with self.assertRaises(ValueError):
            eval_item_from_dict({"eval_id": "e1", "question": "질문", "answerable": True})

    def test_run_log_validates_history_role(self):
        """대화 기록 role은 정해진 값만 허용한다."""
        with self.assertRaises(ValueError):
            run_log_from_dict(
                {
                    "run_id": "r1",
                    "baseline": "api_only",
                    "eval_id": "e1",
                    "question": "질문",
                    "dialogue_history": [{"role": "bad", "content": "x"}],
                }
            )


if __name__ == "__main__":
    unittest.main()
