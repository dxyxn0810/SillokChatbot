"""검색 metric 계산 모듈.

이 모듈은 gold 문서 키와 실제 검색된 문서 키를 비교해 Recall@k, NDCG@k,
MRR@k를 계산한다. 검색을 수행하지 않는 baseline이나 unanswerable 문항에서는
상위 모듈이 N/A로 처리한다.
"""

from __future__ import annotations

import math
from typing import Iterable


def _top_k(items: Iterable[str], k: int) -> list[str]:
    """중복을 보존한 상위 k개 목록을 만든다."""
    if k <= 0:
        return []
    return list(items)[:k]


def recall_at_k(gold_keys: Iterable[str], retrieved_keys: Iterable[str], k: int) -> float:
    """상위 k개 검색 결과가 gold 문서를 얼마나 포함하는지 계산한다."""
    gold = set(gold_keys)
    if not gold:
        return 0.0
    retrieved = set(_top_k(retrieved_keys, k))
    return len(gold & retrieved) / len(gold)


def dcg_at_k(gold_keys: Iterable[str], retrieved_keys: Iterable[str], k: int) -> float:
    """binary relevance 기준 DCG@k를 계산한다."""
    gold = set(gold_keys)
    score = 0.0
    for rank, key in enumerate(_top_k(retrieved_keys, k), start=1):
        rel = 1.0 if key in gold else 0.0
        if rel:
            score += (2**rel - 1) / math.log2(rank + 1)
    return score


def ndcg_at_k(gold_keys: Iterable[str], retrieved_keys: Iterable[str], k: int) -> float:
    """상위 k개 검색 결과의 순위 품질을 NDCG@k로 계산한다."""
    gold = list(dict.fromkeys(gold_keys))
    if not gold:
        return 0.0
    ideal_hits = min(len(gold), max(k, 0))
    if ideal_hits == 0:
        return 0.0
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg_at_k(gold, retrieved_keys, k) / ideal if ideal else 0.0


def mrr_at_k(gold_keys: Iterable[str], retrieved_keys: Iterable[str], k: int) -> float:
    """첫 gold 문서가 등장한 순위의 reciprocal rank를 계산한다."""
    gold = set(gold_keys)
    if not gold:
        return 0.0
    for rank, key in enumerate(_top_k(retrieved_keys, k), start=1):
        if key in gold:
            return 1.0 / rank
    return 0.0

