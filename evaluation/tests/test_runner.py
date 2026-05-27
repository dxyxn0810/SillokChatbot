"""runner query rewriting 테스트.

이 테스트는 실제 OpenAI나 FAISS를 호출하지 않고, `rag_rewrite` baseline이
검색 질의만 재작성하고 run log에 재작성 결과를 남기는지 확인한다.
"""

import unittest

from evaluation.core.baselines import get_baseline
from evaluation.runner import DanjongEvalAdapter


class RunnerRewriteTest(unittest.TestCase):
    """query rewriting 실행 경로를 검증한다."""

    def test_rewrite_plan_is_logged(self):
        """재작성된 검색 질의가 plan에 기록되고 retrieval에 사용된다."""
        captured = {}
        adapter = object.__new__(DanjongEvalAdapter)
        adapter.config = get_baseline("rag_rewrite")
        adapter.verbose = False
        adapter.vs = object()
        adapter.rewrite_llm = _FakeLLM("정발 국상 풍악 파직")
        adapter.llm = _FakeLLM("단종 답변")
        adapter.format_context = lambda docs: "context"
        adapter.build_messages = lambda history, question, context, wiki_evidence, history_turns: ["message"]

        def fake_retrieve(vs, plan, k):
            captured["plan"] = dict(plan)
            captured["k"] = k
            return [_FakeDoc()]

        adapter.retrieve = fake_retrieve
        log = adapter.run_item(
            {
                "eval_id": "e1",
                "question": "그 일로 어떤 처분을 받았나요?",
                "dialogue_history": [
                    {"role": "user", "content": "정발이 국상 중에 무엇을 했나요?"},
                    {"role": "assistant", "content": "정발은 국상 중 풍악을 연주했습니다."},
                ],
            },
            "run1",
        )

        self.assertIsNone(log.error)
        self.assertEqual(captured["plan"]["search_query"], "정발 국상 풍악 파직")
        self.assertEqual(log.plan["rewritten_query"], "정발 국상 풍악 파직")
        self.assertEqual(log.question, "그 일로 어떤 처분을 받았나요?")

    def test_full_rewrite_uses_rewritten_query_but_keeps_original_question(self):
        """full rewrite는 검색만 재작성하고 답변 생성에는 원 질문을 유지한다."""
        captured = {}
        adapter = object.__new__(DanjongEvalAdapter)
        adapter.config = get_baseline("rag_full_rewrite")
        adapter.verbose = False
        adapter.vs = object()
        adapter.rewrite_llm = _FakeLLM("정발 국상 풍악 파직")
        adapter.llm = _FakeLLM("단종 답변")
        adapter.format_context = lambda docs: "context"

        def fake_build_messages(history, question, context, wiki_evidence, history_turns):
            captured["answer_question"] = question
            captured["wiki_evidence"] = wiki_evidence
            return ["message"]

        def fake_make_plan(query, verbose):
            captured["planner_query"] = query
            return {
                "search_query": query,
                "doc_types": ["article", "daily_summary"],
                "wiki_evidence": "위키 근거",
            }

        def fake_retrieve(vs, plan, k):
            captured["plan"] = dict(plan)
            captured["k"] = k
            return [_FakeDoc("daily_summary"), _FakeDoc("article")]

        adapter.build_messages = fake_build_messages
        adapter.make_plan = fake_make_plan
        adapter.retrieve = fake_retrieve

        log = adapter.run_item(
            {
                "eval_id": "e1",
                "question": "그 일로 어떤 처분을 받았나요?",
                "dialogue_history": [
                    {"role": "user", "content": "정발이 국상 중에 무엇을 했나요?"},
                    {"role": "assistant", "content": "정발은 국상 중 풍악을 연주했습니다."},
                ],
            },
            "run1",
        )

        self.assertIsNone(log.error)
        self.assertEqual(captured["planner_query"], "정발 국상 풍악 파직")
        self.assertEqual(captured["plan"]["search_query"], "정발 국상 풍악 파직")
        self.assertEqual(captured["answer_question"], "그 일로 어떤 처분을 받았나요?")
        self.assertEqual(captured["wiki_evidence"], "위키 근거")
        self.assertEqual(log.plan["rewritten_query"], "정발 국상 풍악 파직")
        self.assertEqual(captured["k"], adapter.config.retrieval_k * 3)

    def test_article_baseline_filters_summary_docs(self):
        """article baseline은 검색 결과에서 article type만 로그에 남긴다."""
        adapter = object.__new__(DanjongEvalAdapter)
        adapter.config = get_baseline("rag_article")
        adapter.verbose = False
        adapter.vs = object()
        adapter.llm = _FakeLLM("단종 답변")
        adapter.format_context = lambda docs: "context"
        adapter.build_messages = lambda history, question, context, wiki_evidence, history_turns: ["message"]
        adapter.retrieve = lambda vs, plan, k: [
            _FakeDoc("daily_summary"),
            _FakeDoc("article"),
            _FakeDoc("monthly_summary"),
        ]

        log = adapter.run_item({"eval_id": "e1", "question": "질문", "dialogue_history": []}, "run1")

        self.assertIsNone(log.error)
        self.assertEqual([doc["metadata"]["type"] for doc in log.retrieved_context], ["article"])

    def test_full_baseline_keeps_summary_docs(self):
        """full baseline은 summary type을 제거하지 않는다."""
        adapter = object.__new__(DanjongEvalAdapter)
        adapter.config = get_baseline("rag_full")
        adapter.verbose = False
        adapter.vs = object()
        adapter.llm = _FakeLLM("단종 답변")
        adapter.format_context = lambda docs: "context"
        adapter.build_messages = lambda history, question, context, wiki_evidence, history_turns: ["message"]
        adapter.make_plan = lambda query, verbose: {
            "search_query": query,
            "doc_types": ["article", "daily_summary"],
            "wiki_evidence": "위키 근거",
        }
        adapter.retrieve = lambda vs, plan, k: [_FakeDoc("daily_summary"), _FakeDoc("article")]

        log = adapter.run_item({"eval_id": "e1", "question": "질문", "dialogue_history": []}, "run1")

        self.assertIsNone(log.error)
        self.assertEqual(
            [doc["metadata"]["type"] for doc in log.retrieved_context],
            ["daily_summary", "article"],
        )


class _FakeResponse:
    """LangChain 응답처럼 content를 가진 테스트 객체."""

    def __init__(self, content: str):
        self.content = content


class _FakeLLM:
    """고정 응답을 반환하는 테스트용 LLM."""

    def __init__(self, content: str):
        self.content = content

    def invoke(self, _prompt):
        return _FakeResponse(self.content)


class _FakeDoc:
    """검색 결과 Document 대역."""

    def __init__(self, doc_type: str = "article"):
        self.metadata = {
            "type": doc_type,
            "king": "문종",
            "solar_year": 1450,
            "month": "12월",
            "day": "29일",
            "idx": 6,
            "chunk_id": 0,
        }
        self.page_content = "정발은 국상 중 풍악을 연주한 일로 파직되었다."


if __name__ == "__main__":
    unittest.main()
