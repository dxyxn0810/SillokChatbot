"""LLM judge 평가 모듈.

이 모듈은 생성 답변을 rubric 기반으로 1~5점 평가한다. 평가 기준은 검색 근거
충실도, 질문 관련성, 정답 일치도, 부정 질문 거부, 페르소나 유지, 멀티턴 맥락
활용으로 나뉘며, 각 결과는 JSON 객체로 반환된다.
"""

from __future__ import annotations

import json
from typing import Any

from .env_utils import require_env


JUDGE_RUBRICS: dict[str, str] = {
    "faithfulness": """검색된 실록 자료에 근거하는지 평가한다.
1: 대부분 근거 없음 또는 모순
2: 일부만 근거하고 검증 불가능한 주장 다수
3: 핵심은 근거하나 부수 세부에 근거 없는 내용 존재
4: 거의 모든 주장이 자료에서 추론 가능
5: 모든 주장이 자료에서 직접 추론 가능하고 모순 없음""",
    "answer_relevance": """질문에 실제로 답하는지 평가한다.
1: 질문과 무관
2: 일부만 건드리고 대부분 빗나감
3: 답하나 장황하거나 일부 누락
4: 직접 답하고 불필요한 내용이 적음
5: 정확하고 완전하며 간결하게 답함""",
    "answer_correctness": """gold answer와 사실적으로 일치하는지 평가한다.
1: 핵심 사실 전부 틀림
2: 일부만 맞고 중대한 오류 다수
3: 핵심은 맞으나 세부 오류 존재
4: 거의 정확하고 사소한 오류만 있음
5: gold answer와 사실적으로 완전히 일치""",
    "negative_rejection": """근거가 부족한 질문에서 지어내지 않고 거부하는지 평가한다.
1: 근거 없는데 단정적으로 지어냄
2: 대부분 지어내나 불확실성 표현 일부
3: 부분 답변하며 근거 부족을 일부만 인정
4: 거의 거부하나 약간의 추측 포함
5: 근거 불충분을 명확히 밝히고 지어내지 않음""",
    "persona_adherence": """단종 1인칭 페르소나와 시대 맥락을 유지하는지 평가한다.
1: 현대 챗봇 말투 또는 시대착오가 심함
2: 페르소나가 자주 깨짐
3: 대체로 유지하나 간헐적 이탈
4: 일관된 1인칭 임금 어조와 시대 맥락
5: 어조, 지식 범위, 시대 맥락이 매우 자연스럽고 일관됨""",
    "multi_turn_context_utilization": """이전 대화 맥락을 정확히 활용하는지 평가한다.
1: 이전 턴을 전혀 활용하지 못함
2: 일부만 활용하고 대명사/생략 해소 실패
3: 직전 턴은 활용하나 앞선 맥락을 놓침
4: 대부분 맥락을 정확히 연결
5: 이전 턴 전체를 일관되게 활용해 정확히 답함""",
}


def judge_metric(
    metric_name: str,
    *,
    question: str,
    generated_answer: str,
    retrieved_context: str = "",
    gold_answer: str | None = None,
    gold_evidence: str | None = None,
    dialogue_history: list[dict[str, str]] | None = None,
    model: str = "gpt-4o",
) -> dict[str, Any]:
    """단일 LLM judge metric을 평가한다."""
    if metric_name not in JUDGE_RUBRICS:
        raise ValueError(f"알 수 없는 judge metric입니다: {metric_name}")

    from openai import OpenAI

    prompt = _build_judge_prompt(
        metric_name,
        question=question,
        generated_answer=generated_answer,
        retrieved_context=retrieved_context,
        gold_answer=gold_answer,
        gold_evidence=gold_evidence,
        dialogue_history=dialogue_history or [],
    )
    client = OpenAI(api_key=require_env("OPENAI_API_KEY"))
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"},
    )
    result = json.loads(resp.choices[0].message.content or "{}")
    score = int(result.get("score", 0))
    if score < 1 or score > 5:
        raise ValueError(f"{metric_name}: judge score가 1~5 범위를 벗어났습니다: {score}")
    return {"score": score, "reason": str(result.get("reason", "")).strip()}


def _build_judge_prompt(
    metric_name: str,
    *,
    question: str,
    generated_answer: str,
    retrieved_context: str,
    gold_answer: str | None,
    dialogue_history: list[dict[str, str]],
    gold_evidence: str | None = None,
) -> str:
    """judge에게 전달할 평가 프롬프트를 만든다."""
    return f"""\
너는 조선왕조실록 RAG 챗봇의 평가자다.
아래 rubric만 기준으로 metric 하나를 1~5점으로 평가하라.
JSON 객체 하나만 출력한다: {{"score": 1, "reason": "..."}}

[metric]
{metric_name}

[rubric]
{JUDGE_RUBRICS[metric_name]}

[dialogue_history]
{json.dumps(dialogue_history, ensure_ascii=False)}

[question]
{question}

[retrieved_context]
{retrieved_context}

[gold_answer]
{gold_answer or ""}

[gold_evidence]
{gold_evidence or ""}

[generated_answer]
{generated_answer}
"""
