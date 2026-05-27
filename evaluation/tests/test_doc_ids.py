"""doc_key 생성 테스트.

이 테스트는 `chunk_id`가 같아도 본문과 메타데이터를 포함한 doc_key가 안정적으로
문서를 구분하는지 확인한다.
"""

import unittest

from evaluation.core.doc_ids import make_doc_key


class DocIdsTest(unittest.TestCase):
    """문서 식별자 생성 규칙을 검증한다."""

    def test_doc_key_is_stable(self):
        """같은 입력은 항상 같은 doc_key를 만든다."""
        metadata = {"type": "article", "king": "문종", "solar_year": 1450, "chunk_id": 0}
        key1 = make_doc_key(metadata, "본문")
        key2 = make_doc_key(metadata, "본문")
        self.assertEqual(key1, key2)

    def test_doc_key_changes_with_content(self):
        """본문이 다르면 같은 chunk_id라도 doc_key가 달라진다."""
        metadata = {"type": "article", "king": "문종", "solar_year": 1450, "chunk_id": 0}
        key1 = make_doc_key(metadata, "본문 A")
        key2 = make_doc_key(metadata, "본문 B")
        self.assertNotEqual(key1, key2)


if __name__ == "__main__":
    unittest.main()
