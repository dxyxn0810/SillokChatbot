"""환경 변수 로딩 테스트.

이 테스트는 `python-dotenv` 없이도 evaluation의 `.env` 파서가 기본 KEY=VALUE
형식을 읽고 환경 변수에 반영하는지 확인한다.
"""

import os
from pathlib import Path
import tempfile
import unittest

from evaluation.core.env_utils import _load_env_file, _parse_env_line


class EnvUtilsTest(unittest.TestCase):
    """`.env` 파서의 핵심 동작을 검증한다."""

    def test_parse_quoted_value(self):
        """따옴표로 감싼 값을 따옴표 없이 읽는다."""
        self.assertEqual(_parse_env_line('OPENAI_API_KEY="abc123"\n'), ("OPENAI_API_KEY", "abc123"))

    def test_load_env_file_without_override(self):
        """이미 있는 환경 변수는 기본적으로 덮어쓰지 않는다."""
        key = "SILLOK_EVAL_TEST_KEY"
        os.environ[key] = "old"
        with tempfile.TemporaryDirectory(dir="/private/tmp") as tmp:
            path = Path(tmp) / ".env"
            path.write_text(f"{key}=new\n", encoding="utf-8")
            _load_env_file(path, override=False)
        self.assertEqual(os.environ[key], "old")
        os.environ.pop(key, None)


if __name__ == "__main__":
    unittest.main()
