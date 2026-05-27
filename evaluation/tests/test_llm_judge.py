"""LLM judge 프롬프트 테스트.

이 테스트는 OpenAI 호출 없이 judge 프롬프트에 평가 근거 필드가 안전하게
포함되는지 확인한다.
"""

import unittest

from evaluation.core.llm_judge import _build_judge_prompt


class LLMJudgePromptTest(unittest.TestCase):
    """judge 프롬프트 구성 규칙을 검증한다."""

    def test_prompt_includes_gold_evidence(self):
        """unanswerable 판단 근거가 judge prompt에 포함된다."""
        prompt = _build_judge_prompt(
            "negative_rejection",
            question="문종 때 전기차가 도입되었나요?",
            generated_answer="사료에 없어 답하기 어렵습니다.",
            retrieved_context="",
            gold_answer=None,
            gold_evidence="전기차는 수집된 조선 전기 corpus로 답할 수 없는 현대 개념이다.",
            dialogue_history=[],
        )
        self.assertIn("[gold_evidence]", prompt)
        self.assertIn("전기차는 수집된 조선 전기 corpus로 답할 수 없는 현대 개념이다.", prompt)

    def test_prompt_keeps_gold_answer(self):
        """기존 gold answer 필드도 그대로 유지된다."""
        prompt = _build_judge_prompt(
            "answer_correctness",
            question="질문",
            generated_answer="답변",
            retrieved_context="근거",
            gold_answer="모범답안",
            gold_evidence=None,
            dialogue_history=[],
        )
        self.assertIn("[gold_answer]", prompt)
        self.assertIn("모범답안", prompt)


if __name__ == "__main__":
    unittest.main()
