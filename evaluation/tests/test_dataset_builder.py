"""평가셋 생성 helper 테스트.

이 테스트는 OpenAI 호출 없이 item type 분배와 multi-chunk 그룹 생성이 의도대로
동작하는지 확인한다.
"""

import unittest
from pathlib import Path
import tempfile

from evaluation.dataset_builder import (
    allocate_item_counts,
    build_multi_chunk_groups,
    load_source_records,
    make_unanswerable_eval_item,
    make_multi_turn_eval_item,
    parse_item_types,
    parse_source_types,
)


class DatasetBuilderTest(unittest.TestCase):
    """평가셋 생성 보조 함수를 검증한다."""

    def test_parse_item_types(self):
        """쉼표로 구분한 item type 문자열을 목록으로 바꾼다."""
        self.assertEqual(
            parse_item_types("grounded_single,multi_turn_followup,unanswerable"),
            ["grounded_single", "multi_turn_followup", "unanswerable"],
        )

    def test_parse_source_types(self):
        """쉼표로 구분한 source type 문자열을 목록으로 바꾼다."""
        self.assertEqual(
            parse_source_types("article,daily_summary"),
            ["article", "daily_summary"],
        )
        self.assertIn("yearly_summary", parse_source_types("all"))

    def test_load_source_records_includes_summary(self):
        """source type 목록에 summary가 있으면 summary 문서도 읽는다."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "data.jsonl"
            path.write_text(
                "\n".join(
                    [
                        _jsonl_line("article", "기사 본문"),
                        _jsonl_line("daily_summary", "하루 요약"),
                        _jsonl_line("base_information", "총서 정보"),
                    ]
                ),
                encoding="utf-8",
            )

            records = load_source_records(path, source_types=["article", "daily_summary"])

        self.assertEqual([record["metadata"]["type"] for record in records], ["article", "daily_summary"])

    def test_allocate_item_counts(self):
        """전체 문항 수를 item type별로 균등 배분한다."""
        counts = allocate_item_counts(5, ["grounded_single", "multi_turn_followup", "multi_chunk"])
        self.assertEqual(counts, {"grounded_single": 2, "multi_turn_followup": 2, "multi_chunk": 1})

    def test_build_multi_chunk_groups_same_article(self):
        """같은 기사에 속한 chunk들은 multi-chunk 후보 그룹이 된다."""
        records = [_record(chunk_id=0), _record(chunk_id=1)]
        groups = build_multi_chunk_groups(records)
        self.assertTrue(any(group["group_type"] == "same_article" for group in groups))

    def test_make_multi_turn_eval_item(self):
        """multi-turn 문항은 이전 Q/A와 follow-up 질문을 가진다."""
        record = _record(chunk_id=0)
        generated = {
            "previous_question": "정발은 무엇을 했나요?",
            "previous_answer": "정발은 국상 중 풍악을 연주한 일로 문제가 되었습니다.",
            "followup_question": "그 일로 어떤 처분을 받았나요?",
            "gold_answer": "정발은 파직되었습니다.",
            "gold_evidence": "파직하도록 명하였다.",
        }
        item = make_multi_turn_eval_item(
            record,
            generated,
            index=1,
            model="test-model",
            acceptable_doc_keys=[record["_doc_key"]],
        )
        self.assertEqual(item.item_type, "multi_turn_followup")
        self.assertEqual(len(item.dialogue_history), 2)
        self.assertEqual(item.question, "그 일로 어떤 처분을 받았나요?")

    def test_make_unanswerable_eval_item(self):
        """unanswerable 문항은 gold 문서 없이 answerable=false로 생성된다."""
        record = _record(chunk_id=0)
        generated = {
            "question": "단종이 임진왜란 때 어떤 명령을 내렸나요?",
            "gold_evidence": "임진왜란은 수집된 문종·단종·세조 시기 corpus 밖의 사건이다.",
        }
        item = make_unanswerable_eval_item(record, generated, index=2, model="test-model")
        self.assertEqual(item.item_type, "unanswerable")
        self.assertFalse(item.answerable)
        self.assertEqual(item.gold_doc_keys, [])
        self.assertIsNone(item.gold_answer)
        self.assertIn("negative_rejection", item.tags)


def _record(chunk_id: int) -> dict:
    """테스트용 source record를 만든다."""
    metadata = {
        "type": "article",
        "title": "국상 중 풍악 사건",
        "king": "문종",
        "solar_year": 1450,
        "month": "12월",
        "day": "29일",
        "idx": 6,
        "chunk_id": chunk_id,
    }
    return {
        "metadata": metadata,
        "page_content": f"본문 {chunk_id}",
        "_line_no": chunk_id + 1,
        "_doc_key": f"doc-{chunk_id}",
        "_group_key": "same-article",
    }


def _jsonl_line(doc_type: str, page_content: str) -> str:
    """테스트용 JSONL 한 줄을 만든다."""
    import json

    return json.dumps(
        {
            "page_content": page_content,
            "metadata": {
                "type": doc_type,
                "title": "테스트 문서",
                "king": "문종",
                "solar_year": 1450,
                "month": "1월",
                "day": "1일",
                "idx": 1,
                "chunk_id": 0,
            },
        },
        ensure_ascii=False,
    )


if __name__ == "__main__":
    unittest.main()
