# SillokChatbot Evaluation 사용 가이드

이 폴더는 기존 챗봇 코드를 수정하지 않고 평가셋 생성, baseline 실행,
metric 채점, 요약 리포트 생성을 수행하기 위한 전용 평가 도구입니다.

처음 보면 파일이 많아 보이지만, 직접 실행하는 파일은 5개뿐입니다.

```text
1. dataset_builder.py  평가셋 생성
2. runner.py           baseline 실행 로그 생성
3. score.py            metric 채점
4. summarize.py        평균 리포트 생성
5. compare.py          추천 baseline 5종 비교표 생성
```

나머지 파일은 위 5개가 내부에서 사용하는 helper입니다.

---

## 1. 평가 지표

평가는 크게 검색 품질을 보는 retrieval metric과 답변 품질을 보는 LLM judge
metric으로 나뉩니다. Retrieval metric은 정답 문서 키와 실제 검색 문서 키를
수식으로 비교하고, LLM judge metric은 rubric에 따라 1~5점으로 평가합니다.

`score.py`의 결과 JSONL에는 metric마다 아래 형태로 저장됩니다.

```json
{
  "value": 4.0,
  "na": false,
  "reason": ""
}
```

LLM judge가 왜 그렇게 채점했는지는 같은 줄의 `judge_reasons`에 따로 저장됩니다.

### Retrieval Metric

Retrieval metric은 `gold_doc_keys`와 `retrieved_context[*].doc_key`를 비교합니다.
검색을 사용하지 않는 baseline이거나 `answerable=false` 문항이면 N/A가 됩니다.

#### Recall@k

의미: 답하는 데 필요한 gold chunk가 상위 k개 검색 결과 안에 얼마나 포함됐는지
봅니다. 여러 chunk가 필요한 `multi_chunk` 문항에서는 gold chunk 전체 중 몇 개를
찾았는지 비율로 계산합니다.

입력:

```text
gold_doc_keys
retrieved_doc_keys top-k
```

채점 기준:

```text
Recall@k = |top_k ∩ gold_doc_keys| / |gold_doc_keys|
```

해석:

| 점수 | 의미 |
| --- | --- |
| 1.0 | 필요한 gold chunk를 모두 검색함 |
| 0.0~1.0 | 일부만 검색함 |
| 0.0 | 필요한 gold chunk를 하나도 검색하지 못함 |

N/A 조건:

| 조건 | 이유 |
| --- | --- |
| `api_only`처럼 검색이 없는 baseline | 검색 결과가 없어서 비교 불가 |
| `answerable=false` 문항 | 정답 근거 문서가 없어야 하는 문항 |
| runner error | 실제 검색 결과가 신뢰 불가 |

#### NDCG@k

의미: gold chunk가 검색 결과의 얼마나 높은 순위에 있는지 봅니다. 단순히
포함됐는지만 보는 Recall@k와 달리, 같은 gold chunk라도 1등에 나오면 더 높은
점수를 받고 10등에 나오면 더 낮은 점수를 받습니다.

입력:

```text
gold_doc_keys
retrieved_doc_keys top-k 순서
```

채점 기준:

```text
DCG@k = Σ (2^rel_i - 1) / log2(i + 1)
NDCG@k = DCG@k / IDCG@k
```

여기서 `rel_i`는 i번째 검색 결과가 gold chunk이면 1, 아니면 0입니다.

해석:

| 점수 | 의미 |
| --- | --- |
| 1.0 | gold chunk들이 이상적인 순서로 최상위에 있음 |
| 0.0~1.0 | 일부 gold chunk가 있거나 순위가 낮음 |
| 0.0 | top-k 안에 gold chunk가 없음 |

N/A 조건은 Recall@k와 같습니다.

#### MRR@k

의미: 첫 번째 gold chunk가 몇 번째 순위에 등장했는지 봅니다. 첫 gold chunk가
1등이면 1점, 2등이면 0.5점, 10등이면 0.1점입니다. gold chunk가 하나인 단일
문항에서는 검색 순위 품질을 직관적으로 볼 수 있습니다.

입력:

```text
gold_doc_keys
retrieved_doc_keys top-k 순서
```

채점 기준:

```text
MRR@k = 1 / 첫 gold chunk의 순위
```

top-k 안에 gold chunk가 없으면 0점입니다.

N/A 조건은 Recall@k와 같습니다.

### LLM Judge Metric

LLM judge metric은 `evaluation/core/llm_judge.py`의 rubric을 그대로 사용합니다.
점수는 모두 1~5점이며, 높을수록 좋습니다.

#### Faithfulness

의미: 생성 답변이 실제로 검색된 실록 자료에 근거하는지 평가합니다. 정답
문서가 무엇인지가 아니라, baseline이 실제 검색해 온 context를 기준으로
hallucination 여부를 봅니다.

입력:

```text
retrieved_context
generated_answer
```

Rubric:

| 점수 | 기준 |
| --- | --- |
| 1 | 대부분 근거 없음 또는 모순 |
| 2 | 일부만 근거하고 검증 불가능한 주장 다수 |
| 3 | 핵심은 근거하나 부수 세부에 근거 없는 내용 존재 |
| 4 | 거의 모든 주장이 자료에서 추론 가능 |
| 5 | 모든 주장이 자료에서 직접 추론 가능하고 모순 없음 |

N/A 조건:

| 조건 | 이유 |
| --- | --- |
| 검색이 없는 baseline | 기준 context가 없음 |
| `answerable=false` 문항 | 답을 거부해야 하는 문항이므로 faithfulness보다 negative rejection이 중요 |
| `--skip-llm-judge` | LLM judge를 실행하지 않음 |

#### Answer Relevance

의미: 답변이 사용자의 질문에 실제로 답하는지 평가합니다. 여기서는 사실 여부보다
질문과의 관련성, 누락, 장황함을 봅니다.

입력:

```text
question
generated_answer
```

Rubric:

| 점수 | 기준 |
| --- | --- |
| 1 | 질문과 무관 |
| 2 | 일부만 건드리고 대부분 빗나감 |
| 3 | 답하나 장황하거나 일부 누락 |
| 4 | 직접 답하고 불필요한 내용이 적음 |
| 5 | 정확하고 완전하며 간결하게 답함 |

N/A 조건:

| 조건 | 이유 |
| --- | --- |
| `--skip-llm-judge` | LLM judge를 실행하지 않음 |
| runner error | 생성 답변이 없거나 신뢰 불가 |

#### Answer Correctness

의미: 생성 답변이 평가셋의 `gold_answer`와 사실적으로 일치하는지 평가합니다.
검색 결과에 근거했더라도 context를 잘못 읽으면 이 점수에서 드러납니다.

입력:

```text
gold_answer
generated_answer
```

Rubric:

| 점수 | 기준 |
| --- | --- |
| 1 | 핵심 사실 전부 틀림 |
| 2 | 일부만 맞고 중대한 오류 다수 |
| 3 | 핵심은 맞으나 세부 오류 존재 |
| 4 | 거의 정확하고 사소한 오류만 있음 |
| 5 | gold answer와 사실적으로 완전히 일치 |

N/A 조건:

| 조건 | 이유 |
| --- | --- |
| `answerable=false` 문항 | 모범답안을 두지 않고 거부 여부를 봄 |
| `--skip-llm-judge` | LLM judge를 실행하지 않음 |
| runner error | 생성 답변이 없거나 신뢰 불가 |

#### Negative Rejection

의미: 수집된 corpus로 답할 수 없는 질문에서 근거 없이 지어내지 않고, 사료에
근거가 부족하다고 명확히 거부하는지 평가합니다.

입력:

```text
question
retrieved_context
gold_evidence
generated_answer
```

Rubric:

| 점수 | 기준 |
| --- | --- |
| 1 | 근거 없는데 단정적으로 지어냄 |
| 2 | 대부분 지어내나 불확실성 표현 일부 |
| 3 | 부분 답변하며 근거 부족을 일부만 인정 |
| 4 | 거의 거부하나 약간의 추측 포함 |
| 5 | 근거 불충분을 명확히 밝히고 지어내지 않음 |

N/A 조건:

| 조건 | 이유 |
| --- | --- |
| `answerable=true` 문항 | 답해야 하는 문항이므로 거부 능력 평가 대상이 아님 |
| `--skip-llm-judge` | LLM judge를 실행하지 않음 |

#### Persona Adherence

의미: 답변이 단종 1인칭 페르소나와 조선 전기 시대 맥락을 유지하는지 평가합니다.
현대 챗봇 말투, 시대착오, 1인칭 임금 시점 붕괴를 감점합니다.

입력:

```text
dialogue_history
question
generated_answer
```

Rubric:

| 점수 | 기준 |
| --- | --- |
| 1 | 현대 챗봇 말투 또는 시대착오가 심함 |
| 2 | 페르소나가 자주 깨짐 |
| 3 | 대체로 유지하나 간헐적 이탈 |
| 4 | 일관된 1인칭 임금 어조와 시대 맥락 |
| 5 | 어조, 지식 범위, 시대 맥락이 매우 자연스럽고 일관됨 |

N/A 조건:

| 조건 | 이유 |
| --- | --- |
| `--skip-llm-judge` | LLM judge를 실행하지 않음 |
| runner error | 생성 답변이 없거나 신뢰 불가 |

#### Multi-turn Context Utilization

의미: 이전 대화 맥락을 활용해 현재 질문의 대명사, 생략, 지시어를 정확히
해소하는지 평가합니다.

입력:

```text
dialogue_history
question
generated_answer
```

Rubric:

| 점수 | 기준 |
| --- | --- |
| 1 | 이전 턴을 전혀 활용하지 못함 |
| 2 | 일부만 활용하고 대명사/생략 해소 실패 |
| 3 | 직전 턴은 활용하나 앞선 맥락을 놓침 |
| 4 | 대부분 맥락을 정확히 연결 |
| 5 | 이전 턴 전체를 일관되게 활용해 정확히 답함 |

N/A 조건:

| 조건 | 이유 |
| --- | --- |
| `dialogue_history`가 없고 `item_type`도 `multi_turn_followup`이 아님 | 멀티턴 활용을 볼 문항이 아님 |
| `--skip-llm-judge` | LLM judge를 실행하지 않음 |
| runner error | 생성 답변이 없거나 신뢰 불가 |

### Metric 적용 조건 요약

| metric | 적용 조건 | 주요 비교 대상 |
| --- | --- | --- |
| `recall@k` | 검색 baseline + `answerable=true` | gold doc keys vs retrieved doc keys |
| `ndcg@k` | 검색 baseline + `answerable=true` | gold doc keys vs retrieved doc keys 순위 |
| `mrr@k` | 검색 baseline + `answerable=true` | 첫 gold doc key의 순위 |
| `faithfulness` | 검색 baseline + `answerable=true` | retrieved context vs generated answer |
| `answer_relevance` | 모든 생성 답변 | question vs generated answer |
| `answer_correctness` | `answerable=true` | gold answer vs generated answer |
| `negative_rejection` | `answerable=false` + 모든 baseline | question/context/gold evidence vs generated answer |
| `persona_adherence` | 모든 생성 답변 | persona/dialogue vs generated answer |
| `multi_turn_context_utilization` | dialogue history가 있거나 `multi_turn_followup` | dialogue history vs generated answer |

`api_only`는 검색을 하지 않으므로 retrieval metric과 `faithfulness`는 N/A입니다.
다만 `answerable=false` 문항에서는 검색 없이도 없는 사실을 지어내지 않는지
확인하기 위해 `negative_rejection`을 평가합니다.

---

## 2. 전체 실행 흐름

평가는 아래 순서로 진행됩니다.

```text
data/data.jsonl
  -> evaluation/data/*.jsonl      평가셋
  -> evaluation/runs/*.jsonl      실제 시스템 실행 로그
  -> evaluation/scores/*.jsonl    metric별 채점 결과
  -> evaluation/reports/*.json    baseline별 평균 요약
```

각 단계의 의미는 다음과 같습니다.

| 단계 | 실행 파일 | 입력 | 출력 | 역할 |
| --- | --- | --- | --- | --- |
| 1 | `dataset_builder.py` | `data/data.jsonl` | `evaluation/data/*.jsonl` | chunk에서 평가 질문과 gold answer 생성 |
| 2 | `runner.py` | eval set | `evaluation/runs/*.jsonl` | baseline을 실제 실행하고 검색 결과와 답변 저장 |
| 3 | `score.py` | eval set + run log | `evaluation/scores/*.jsonl` | retrieval metric과 LLM judge 점수 계산 |
| 4 | `summarize.py` | score files | `evaluation/reports/*.json` | 평균 점수와 N/A 개수 요약 |

`dataset_builder.py`, `runner.py`, `score.py`, `compare.py`는 긴 작업의 처리
상태를 `tqdm` 진행률 막대로 보여줍니다. 다른 서버에서 진행률이 보이지 않으면
`pip install -r requirements.txt`로 `tqdm` 의존성을 설치했는지 확인합니다.

---

## 3. 파일 구조

### 실행 엔트리포인트

- `dataset_builder.py`  
  `data/data.jsonl`에서 source chunk를 샘플링하고, OpenAI 모델로
  `question`, `gold_answer`, `gold_doc_keys`를 생성합니다. 단일 chunk,
  대화 follow-up, multi-chunk 문항을 만들 수 있습니다.

- `runner.py`  
  `core/baselines.py`에 정의된 baseline을 실행합니다. 기존 `doyoon_project`
  챗봇 코드를 import해서 검색 계획, 검색 문서, 생성 답변, 실행 시간을
  `runs/`에 저장합니다.

- `score.py`  
  eval set의 ideal 값과 run log의 actual 값을 비교합니다.  
  `Recall@k`, `NDCG@k`, `MRR@k`는 수식으로 계산하고,
  `faithfulness`, `answer_correctness` 등은 LLM judge로 평가합니다.

- `summarize.py`  
  score JSONL 파일들을 baseline별 평균 리포트로 묶습니다.

- `compare.py`  
  추천 baseline 5개를 같은 평가셋으로 실행·채점·요약하고 Markdown 비교표를
  생성합니다.

### 내부 helper

내부 helper는 모두 `core/` 아래에 모여 있습니다. 보통 직접 실행하지 않고,
위의 4개 CLI가 import해서 사용합니다.

- `core/schemas.py`  
  평가셋, 실행 로그, 점수 결과의 공통 데이터 구조와 JSONL 입출력을 정의합니다.

- `core/doc_ids.py`  
  `chunk_id`가 전역 고유값이 아니므로, metadata와 본문 hash를 조합해
  안정적인 `doc_key`를 만듭니다.

- `core/baselines.py`  
  `api_only`, `rag_article`, `rag_rewrite`, `rag_full`, `rag_full_rewrite`
  등의 baseline 설정을 담고 있습니다.

- `core/retrieval_metrics.py`  
  `Recall@k`, `NDCG@k`, `MRR@k` 계산 함수입니다.

- `core/llm_judge.py`  
  LLM judge rubric과 OpenAI 호출 로직입니다.

- `core/env_utils.py`  
  `.env` 파일을 자동으로 찾고 `OPENAI_API_KEY`를 로드합니다.
  `python-dotenv`가 없어도 evaluation 자체는 `.env`를 읽을 수 있습니다.

### 산출물 폴더

- `data/`  
  평가셋 JSONL을 저장합니다.

- `runs/`  
  baseline을 실제 실행한 결과를 저장합니다. 검색 결과와 생성 답변이 들어갑니다.

- `scores/`  
  metric별 채점 결과를 저장합니다.

- `reports/`  
  평균 점수 요약 JSON을 저장합니다.

### 개발 검증용

- `tests/`  
  OpenAI 호출 없이 동작하는 단위 테스트입니다.

- `__pycache__/`  
  Python이 자동으로 만드는 캐시입니다. git에 넣지 않아도 됩니다.

---

## 4. 환경 준비

프로젝트 루트에서 실행합니다.

```bash
cd SillokChatbot
conda activate sillok
```

`.env`에는 다음 값이 있어야 합니다.

```text
OPENAI_API_KEY=...
```

evaluation은 아래 위치들을 자동으로 확인합니다.

```text
현재 실행 위치의 .env
현재 실행 위치의 상위 .env
SillokChatbot/.env
SillokChatbot 상위 폴더의 .env
```

환경 확인:

```bash
python -c "from evaluation.core.env_utils import require_env; require_env('OPENAI_API_KEY'); print('OPENAI_API_KEY set')"
python -c "import openai, faiss, langchain_openai; print('deps ok')"
python -m unittest discover -s evaluation/tests
```

---

## 5. OpenMP 오류 설명

macOS/conda 환경에서 아래 오류가 날 수 있습니다.

```text
OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib already initialized.
```

### 이게 무엇인가?

`libomp.dylib`는 OpenMP 런타임입니다. 쉽게 말해, CPU 병렬 계산을 관리하는
라이브러리입니다.

`faiss`, `numpy`, `torch`, `scikit-learn` 같은 고성능 계산 라이브러리는 내부에서
OpenMP를 사용합니다. 그런데 conda/pip로 설치된 라이브러리들이 서로 다른 OpenMP
런타임을 각각 들고 있으면, Python 프로세스 안에서 OpenMP가 두 번 초기화됩니다.
그때 macOS에서는 안전하지 않다고 보고 프로세스를 중단할 수 있습니다.

### smoke/local 실행용 우회

로컬 smoke test에서는 아래 환경변수로 우회할 수 있습니다.

```bash
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1
```

의미는 다음과 같습니다.

- `KMP_DUPLICATE_LIB_OK=TRUE`  
  OpenMP 런타임이 중복 로드되어도 일단 실행을 허용합니다.

- `OMP_NUM_THREADS=1`  
  OpenMP가 쓰는 CPU thread 수를 1개로 제한해서 충돌 가능성과 과도한 병렬 실행을 줄입니다.

이 설정은 “평가 코드가 틀려서 필요한 옵션”이 아니라, Python 과학계산 패키지들의
런타임 충돌을 피하기 위한 실행 환경 우회입니다.

최종 대량 실험에서는 임시 우회보다 conda 환경을 정리하는 편이 더 좋습니다. 예를 들어
`faiss-cpu`, `numpy`, `torch`, `scikit-learn`, `libomp`를 같은 채널, 가능하면
`conda-forge` 중심으로 맞추는 방식이 안정적입니다.

---

## 6. Mixed Eval 실행 순서

기본 평가는 4종 문항을 섞은 40문항 mixed set으로 실행합니다.

### 1단계: 환경 확인

```bash
cd SillokChatbot
conda activate sillok

export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1

python -m unittest discover -s evaluation/tests
```

### 2단계: 평가셋 생성

이미 `evaluation/data/eval_set_mixed.jsonl`이 있으면 이 단계는 건너뛰어도 됩니다.

```bash
python -m evaluation.dataset_builder \
  --input data/data.jsonl \
  --output evaluation/data/eval_set_mixed.jsonl \
  --n 40 \
  --model gpt-4o \
  --source-types article,daily_summary,monthly_summary,yearly_summary \
  --item-types grounded_single,multi_turn_followup,multi_chunk,unanswerable
```

출력:

```text
evaluation/data/eval_set_mixed.jsonl
```

내용:

```text
eval_id, question, gold_answer, gold_doc_keys, gold_evidence, source_metadata
```

`--source-types`는 평가 질문을 만들 source 문서 종류입니다. 기본 예시는 article
chunk뿐 아니라 daily/monthly/yearly summary도 함께 사용합니다. `--n`은 전체
문항 수이며, 지정한 item type들에 최대한 균등하게 배분됩니다.

지원하는 item type:

| item type | 의미 |
| --- | --- |
| `grounded_single` | chunk 하나로 답할 수 있는 단일 질문 |
| `multi_turn_followup` | 이전 Q/A 1쌍을 봐야 현재 질문의 생략/대명사를 해소할 수 있는 대화형 질문 |
| `multi_chunk` | 관련 chunk 2~4개를 함께 봐야 답할 수 있는 종합 질문 |
| `unanswerable` | 수집된 실록 corpus로는 답할 수 없어야 하는 negative rejection 질문 |

### 3단계: baseline 실행

```bash
python -m evaluation.runner \
  --eval-set evaluation/data/eval_set_mixed.jsonl \
  --baseline rag_article \
  --out evaluation/runs/rag_article_mixed.jsonl
```

출력:

```text
evaluation/runs/rag_article_mixed.jsonl
```

내용:

```text
baseline, eval_id, plan, retrieved_context, generated_answer, error, timing_ms
```

### 4단계: 채점

```bash
python -m evaluation.score \
  --eval-set evaluation/data/eval_set_mixed.jsonl \
  --run evaluation/runs/rag_article_mixed.jsonl \
  --out evaluation/scores/rag_article_mixed.jsonl \
  --judge-model gpt-4o
```

출력:

```text
evaluation/scores/rag_article_mixed.jsonl
```

내용:

```text
retrieval metric
faithfulness
answer_relevance
answer_correctness
persona_adherence
negative_rejection 또는 multi_turn_context_utilization, 해당 시
```

LLM judge 비용 없이 검색 metric만 빠르게 확인하고 싶으면 `--skip-llm-judge`를
붙여 retrieval-only 결과를 따로 저장할 수 있습니다. 이 경우 LLM judge metric은
N/A로 기록됩니다.

```bash
python -m evaluation.score \
  --eval-set evaluation/data/eval_set_mixed.jsonl \
  --run evaluation/runs/rag_article_mixed.jsonl \
  --out evaluation/scores/rag_article_mixed_retrieval.jsonl \
  --skip-llm-judge
```

### 5단계: 요약 리포트 생성

```bash
python -m evaluation.summarize \
  --scores evaluation/scores/rag_article_mixed.jsonl \
  --out evaluation/reports/mixed_summary.json
```

출력:

```text
evaluation/reports/mixed_summary.json
```

내용:

```text
baseline별 items 수
metric별 count, N/A 수, 평균 점수
```

여기서 `items`는 baseline으로 실행한 전체 문항 수이고, `count`는 해당 metric이
실제 점수로 계산된 건수입니다. `na`는 같은 metric이 N/A 처리된 건수입니다.

---

## 7. Baseline 종류

추천 비교 baseline은 아래 5개입니다.

| 이름 | 검색 사용 | query rewriting | CoT 사용 | 설명 |
| --- | --- | --- | --- | --- |
| `api_only` | 아니오 | 아니오 | 아니오 | 검색 없이 페르소나 프롬프트만 사용 |
| `rag_article` | 예 | 아니오 | 아니오 | article 문서만 원 질문으로 순수 의미검색 |
| `rag_rewrite` | 예 | 예 | 아니오 | 검색 질의 재작성 + article 문서 |
| `rag_full` | 예 | 아니오 | 예 | article, summary, 위키 보조근거 포함 |
| `rag_full_rewrite` | 예 | 예 | 예 | 검색 질의 재작성 + full 검색 |

summary 문서는 FAISS에 들어 있습니다. `daily_summary`, `monthly_summary`,
`yearly_summary`는 `rag_full` 계열에서만 허용하고, `rag_article` 계열은
evaluation runner가 검색 결과를 article-only로 한 번 더 보정합니다.

query rewriting은 답변 생성 질문을 바꾸는 기능이 아니라 검색 질의만 바꾸는
기능입니다. 최종 답변 생성에는 원래 사용자 질문과 `dialogue_history`가 그대로
사용됩니다.

참고로 `rag_article_cot`와 `rag_article_rewrite`도 코드에는 남아 있습니다.
이전 실험 호환용 baseline이며, 기본 비교 세트에서는 사용하지 않습니다.

`rag_rewrite` 실행 예:

```bash
python -m evaluation.runner \
  --eval-set evaluation/data/eval_set_mixed.jsonl \
  --baseline rag_rewrite \
  --out evaluation/runs/rag_rewrite_mixed.jsonl
```

`rag_full_rewrite` 실행 예:

```bash
python -m evaluation.runner \
  --eval-set evaluation/data/eval_set_mixed.jsonl \
  --baseline rag_full_rewrite \
  --out evaluation/runs/rag_full_rewrite_mixed.jsonl
```

여러 baseline을 채점한 뒤 한 번에 요약할 수 있습니다.

```bash
python -m evaluation.summarize \
  --scores evaluation/scores/*_mixed.jsonl \
  --out evaluation/reports/all_mixed_summary.json
```

---

## 8. 5문항 빠른 비교표 생성

추천 baseline 5개를 같은 5문항으로 빠르게 비교하려면 `compare.py`를 사용합니다.
기본 baseline은 `api_only`, `rag_article`, `rag_rewrite`, `rag_full`,
`rag_full_rewrite`입니다.

```bash
python -m evaluation.compare \
  --eval-set evaluation/data/eval_set_mixed.jsonl \
  --limit 5 \
  --judge-model gpt-4o \
  --out-md evaluation/reports/compare5_table.md \
  --out-json evaluation/reports/compare5_summary.json \
  --out-retrieval-plot evaluation/reports/compare5_retrieval_metrics.png \
  --out-generation-plot evaluation/reports/compare5_generation_metrics.png \
  --out-conversation-plot evaluation/reports/compare5_conversation_metrics.png
```

출력:

```text
evaluation/runs/api_only_compare5.jsonl
evaluation/runs/rag_article_compare5.jsonl
evaluation/runs/rag_rewrite_compare5.jsonl
evaluation/runs/rag_full_compare5.jsonl
evaluation/runs/rag_full_rewrite_compare5.jsonl

evaluation/scores/*_compare5.jsonl
evaluation/reports/compare5_summary.json
evaluation/reports/compare5_table.md
evaluation/reports/compare5_retrieval_metrics.png
evaluation/reports/compare5_generation_metrics.png
evaluation/reports/compare5_conversation_metrics.png
```

`compare5_table.md`는 아래 형태의 Markdown 표입니다.

```text
baseline | items | recall@10 | ndcg@10 | mrr@10 | faithfulness | ...
```

그래프는 평가 설계 축에 맞춰 3개로 나누어 저장됩니다.

| 파일 | 축 | metric | y축 |
| --- | --- | --- | --- |
| `compare5_retrieval_metrics.png` | A. Retrieval | `Recall@10`, `NDCG@10`, `MRR@10` | 0~1 |
| `compare5_generation_metrics.png` | B. Generation | `Faithfulness`, `Answer Relevance`, `Answer Correctness`, `Negative Rejection` | rubric 1~5 |
| `compare5_conversation_metrics.png` | C. Conversational | `Persona Adherence`, `Multi-turn Context Utilization` | rubric 1~5 |

N/A metric은 그래프에서 막대를 그리지 않습니다. 예전 방식처럼 모든 metric을 한
그래프에 섞은 legacy 통합 그래프가 필요하면 `--out-plot`을 추가로 지정할 수
있습니다.

이미 `compare5_summary.json`이 있다면 LLM/FAISS를 다시 실행하지 않고 표와 그래프만
다시 만들 수 있습니다.

```bash
python -m evaluation.compare \
  --from-summary evaluation/reports/compare5_summary.json \
  --out-md evaluation/reports/compare5_table.md \
  --out-retrieval-plot evaluation/reports/compare5_retrieval_metrics.png \
  --out-generation-plot evaluation/reports/compare5_generation_metrics.png \
  --out-conversation-plot evaluation/reports/compare5_conversation_metrics.png
```

5문항짜리 평가셋을 별도로 만들고 싶으면 아래처럼 생성할 수도 있습니다.

```bash
python -m evaluation.dataset_builder \
  --input data/data.jsonl \
  --output evaluation/data/eval_set_compare5.jsonl \
  --n 5 \
  --model gpt-4o \
  --source-types article,daily_summary,monthly_summary,yearly_summary \
  --item-types grounded_single,multi_turn_followup,multi_chunk,unanswerable
```

---

## 9. 다른 서버에서 자주 나는 에러

### `.env` 인식 실패

확인:

```bash
python -c "from evaluation.core.env_utils import require_env; require_env('OPENAI_API_KEY'); print('OPENAI_API_KEY set')"
```

해결:

- `.env`에 `OPENAI_API_KEY=...`가 있는지 확인합니다.
- `cd SillokChatbot` 위치에서 실행합니다.
- 서버에서 환경변수로 직접 넣어도 됩니다.

```bash
export OPENAI_API_KEY="..."
```

### `ModuleNotFoundError: No module named 'dotenv'`

기존 `doyoon_project/chatbot.py`가 `python-dotenv`를 import합니다.

해결:

```bash
pip install python-dotenv
```

### 잘못된 Python/conda 환경 사용

증상:

```text
ModuleNotFoundError: No module named 'openai'
ModuleNotFoundError: No module named 'faiss'
```

확인:

```bash
which python
python -c "import sys; print(sys.executable)"
python -c "import openai, faiss, langchain_openai; print('deps ok')"
```

해결:

```bash
conda activate sillok
```

또는 해당 서버의 실제 conda Python을 명시해서 실행합니다.

```bash
/path/to/conda/envs/sillok/bin/python -m evaluation.runner ...
```

### OpenMP `libomp.dylib already initialized`

해결:

```bash
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1
```

대량 실험용 서버에서는 conda 환경을 정리하는 것을 권장합니다.

### OpenAI `APIConnectionError`

원인:

- 네트워크 차단
- 서버 방화벽
- proxy 설정 필요
- API key 문제

확인:

```bash
python -c "from evaluation.core.env_utils import require_env; print(require_env('OPENAI_API_KEY')[:7])"
```

해결:

- 인터넷 연결 확인
- OpenAI API 접근 가능한 네트워크인지 확인
- 필요한 경우 proxy 환경변수 설정

### JSONL을 `json.tool`로 읽을 때 `Extra data`

`eval_set_mixed.jsonl`은 JSON 하나가 아니라 줄마다 JSON 객체가 있는 JSONL입니다.

잘못된 확인:

```bash
python -m json.tool evaluation/data/eval_set_mixed.jsonl
```

올바른 확인:

```bash
wc -l evaluation/data/eval_set_mixed.jsonl
python -c "from pathlib import Path; from evaluation.core.schemas import read_jsonl; print(len(read_jsonl(Path('evaluation/data/eval_set_mixed.jsonl'))))"
```

---

## 10. Mixed 실행 후 확인할 파일

`rag_article` mixed run을 끝까지 실행하면 아래 파일들을 확인합니다.

```text
evaluation/data/eval_set_mixed.jsonl
evaluation/runs/rag_article_mixed.jsonl
evaluation/scores/rag_article_mixed.jsonl
evaluation/scores/rag_article_mixed_retrieval.jsonl
evaluation/reports/mixed_summary.json
```

`evaluation/reports/mixed_summary.json`에는 baseline별 문항 수, metric별 실제
계산 건수, N/A 수, 평균 점수가 저장됩니다.

```text
baseline별 items 수
metric별 count
metric별 N/A 수
metric별 평균 점수
```
