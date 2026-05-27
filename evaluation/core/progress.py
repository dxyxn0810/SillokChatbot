"""진행률 표시 공통 모듈.

이 모듈은 evaluation CLI가 긴 작업을 수행할 때 `tqdm` 진행률 막대를 일관되게
보여주도록 돕는다. `tqdm`이 설치되지 않은 환경에서는 작업 자체가 멈추지 않도록
조용히 일반 반복자로 대체한다.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterable, Iterator, TypeVar


T = TypeVar("T")


def progress_iter(
    iterable: Iterable[T],
    *,
    total: int | None = None,
    desc: str,
    unit: str = "it",
) -> Iterable[T]:
    """반복 작업에 tqdm 진행률 막대를 붙인다."""
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return iterable
    return tqdm(iterable, total=total, desc=desc, unit=unit)


@contextmanager
def progress_bar(*, total: int, desc: str, unit: str = "it") -> Iterator[Any]:
    """수동으로 update할 수 있는 tqdm 진행률 막대를 만든다."""
    try:
        from tqdm.auto import tqdm
    except ImportError:
        yield _NoOpProgressBar()
        return

    with tqdm(total=total, desc=desc, unit=unit) as bar:
        yield bar


class _NoOpProgressBar:
    """tqdm이 없을 때 사용하는 빈 진행률 막대."""

    def update(self, n: int = 1) -> None:
        """진행률 업데이트 요청을 무시한다."""
        return None
