# chatbot/

단종 가상 인터뷰 챗봇의 핵심 모듈 모음.  
조선왕조실록 FAISS 인덱스 + CoT 플래너 + RAG 파이프라인으로 단종의 1인칭 답변을 생성한다.

---

## 파일 구성

```
chatbot/
├── chatbot.py          # 챗봇 핵심 로직 + CLI 실행 진입점
├── chatbot_web.py      # HTTP 웹 서버 버전 (브라우저 UI 포함)
├── cot_planner.py      # Chain-of-Thought 기반 검색 계획 생성
├── retriever.py        # FAISS 벡터스토어 로드 + 점진적 완화 검색
├── metadata_utils.py   # 메타데이터 정규화 및 필터 생성 유틸
└── wiki_tool.py        # 한국어 위키백과 조회 도구
```

---

## 전체 파이프라인

```
사용자 질문
    │
    ▼
cot_planner.make_plan()          # CoT로 검색 메타데이터 결정 + 위키백과 근거 수집
    │   ├─ STEP 1: LLM이 "어떤 날짜/필터가 필요한가" 추론
    │   ├─ STEP 2: 위키백과 검색으로 날짜·인물 관계 확인
    │   └─ STEP 3: 위키 근거 + 배경지식 → 최종 메타데이터 필터(JSON) 확정
    │
    ▼
retriever.retrieve()             # 점진적 완화 검색
    │   ├─ 연-월-일 + doc_type (가장 좁음)
    │   ├─ 연-월, 연, 왕 순으로 단계적 완화
    │   └─ 순수 의미검색(무필터) 폴백
    │
    ▼
GPT-4o (단종 페르소나)           # 실록 자료 + 위키 근거 → 1인칭 답변 생성
```

---

## 파일별 설명

### `chatbot.py` — 챗봇 핵심 로직 + CLI

| 항목 | 내용 |
|------|------|
| **주요 클래스** | `DanjongBot` |
| **LLM** | `gpt-4o` (temperature=0.7) |
| **역할** | CoT 플래너 → 검색 → 답변 생성의 전체 흐름 조율 |

**`DanjongBot` 동작 흐름**

1. `make_plan(user_q)` 으로 메타데이터 필터 및 위키 근거 획득
2. `retrieve(vs, plan, k)` 로 실록 문서 검색
3. 실록 자료 + 위키 근거 + 대화 기록 → GPT-4o에 전달
4. 단종 1인칭 답변 반환 및 대화 기록(`history`) 업데이트

**`--data` 옵션 (정보 소스 범위, 누적적)**

| 값 | 허용 문서 type | 위키 사용 |
|----|---------------|-----------|
| `prompt` | *(검색 없음)* | ✗ |
| `article` | `article` | ✗ |
| `summary` | `article`, `*_summary` | ✗ |
| `wikipedia` | `article`, `*_summary` | ✓ |

**CLI 실행**

```bash
# 기본 대화형 모드
python chatbot.py

# 주요 옵션
python chatbot.py --data wikipedia   # 정보 소스 범위 (기본값)
python chatbot.py --CoT on           # CoT 필터링 사용 여부 (기본값: on)
python chatbot.py --history 6        # 참고할 직전 대화 턴 수 (기본값: 6)
python chatbot.py --retrieval 10     # 검색 문서 수 k (기본값: 10)
python chatbot.py --faiss ../faiss   # FAISS 인덱스 폴더
python chatbot.py --verbose          # CoT·검색 과정 상세 출력
python chatbot.py --once "단종 즉위는 언제인가"  # 한 번만 질문 후 종료
```

---

### `chatbot_web.py` — HTTP 웹 서버 버전

| 항목 | 내용 |
|------|------|
| **서버** | `ThreadingHTTPServer` (Python 표준 라이브러리만 사용) |
| **UI** | 인라인 HTML/CSS/JS (한지 느낌 디자인, 고운바탕 폰트) |
| **세션** | 쿠키 `sid` 로 다중 사용자 세션 관리 |

**API 엔드포인트**

| 메서드 | 경로 | 설명 |
|--------|------|------|
| `GET` | `/` | 채팅 UI 페이지 반환 |
| `POST` | `/api/chat` | `{"message": "..."}` → `{"answer": "...", "elapsed_ms": N}` |
| `POST` | `/api/reset` | 현재 세션의 대화 기록 초기화 |

**실행**

```bash
python chatbot_web.py                        # 기본: http://127.0.0.1:8000
python chatbot_web.py --host 0.0.0.0 --port 8080
python chatbot_web.py --data wikipedia --CoT on --verbose
```

`chatbot.py`와 동일한 `--data`, `--CoT`, `--history`, `--retrieval`, `--faiss` 옵션 지원.

---

### `cot_planner.py` — CoT 검색 계획 생성

사용자 질문 → **3단계 CoT 파이프라인** → 메타데이터 필터(`plan` dict) 반환

| 단계 | 함수 | 역할 |
|------|------|------|
| STEP 1 | `plan_step1(query)` | LLM이 필요한 날짜·인물 필터를 Chain-of-Thought로 추론 |
| STEP 2 | `gather_wiki_evidence(queries, king)` | 위키백과에서 날짜·인물 관계 근거 텍스트 수집 |
| STEP 3 | `plan_step3(query, step1, wiki)` | 위키 근거 + 배경지식 → 최종 필터(JSON) 확정 |

**반환되는 `plan` dict 예시**

```python
{
    "king": "단종",          # 왕 필터 ("단종"/"세조"/"문종"/null)
    "solar_year": 1452,     # 양력 연도
    "solar_year_range": None, # 연도 범위 [시작, 끝] (기간 질문 시)
    "month": "5월",          # 음력 월
    "month_range": None,     # 월 범위 [시작, 끝] (같은 해 여러 달)
    "day": "18일",           # 일
    "day_range": None,       # 일 범위 [시작, 끝] (같은 달 며칠)
    "doc_types": ["article"],
    "search_query": "단종 즉위식 근정문",
    "wiki_evidence": "...",  # 위키 근거 텍스트 (답변 생성 시 보조 사용)
}
```

> **주의사항**: 단종 양위(1455) 이후 사건(말년·유배·죽음 등)은 실록에서 `king="세조"`로
> 분류되므로, 이런 질문에는 `king="세조"` 또는 `null`로 설정한다.

---

### `retriever.py` — 점진적 완화 검색

**핵심 전략**: 날짜 추정이 조금 어긋나도 빈 결과가 나오지 않도록, 좁은 필터에서 넓은 필터로 단계적으로 완화.

```
연-월-일 + doc_type
    → 연-월-일
    → 연-월 + doc_type → 연-월
    → 연 + doc_type    → 연
    → 왕(king)만
    → 무필터 (순수 의미검색)
```

**범위 검색 지원** (`FAISS`가 직접 지원하지 않으므로 개별 값으로 분해):

- `solar_year_range` → 연도별로 나눠 검색 후 라운드로빈 병합
- `month_range` → 범위 내 월(윤달 포함)별 검색 후 병합
- `day_range` → 범위 내 일별 검색 후 병합

**주요 함수**

```python
load_vectorstore(faiss_dir)       # FAISS 인덱스 로드 (text-embedding-3-large)
retrieve(vs, plan, k=6)           # plan 기반 점진적 완화 검색 → Document 리스트
format_context(docs)              # Document 리스트 → LLM 프롬프트용 문자열
```

---

### `metadata_utils.py` — 메타데이터 유틸

실록 메타데이터의 형식 정규화 및 FAISS 필터 생성을 담당.

**재위년 ↔ 양력 연도 매핑**

| 왕 | 재위 0년 | 재위 1년 | 재위 2년 | 재위 3년 |
|----|---------|---------|---------|---------|
| 문종 | 1450 | 1451 | 1452 | — |
| 단종 | 1452 | 1453 | 1454 | 1455 |
| 세조 | — | 1455 | 1456 | 1457 |

**주요 함수**

```python
norm_year("2")       # → "2년"
norm_month("윤6")    # → "윤6월"
norm_day(18)         # → "18일"
build_filter(king="단종", solar_year=1452, month="5월")  # FAISS 필터 dict 생성
months_in_range(avail, 6, 8)  # 윤달 포함 월 범위 반환
lunar_month_key("윤6월")      # 음력 월 정렬 키 (6월 < 윤6월 < 7월)
```

---

### `wiki_tool.py` — 위키백과 조회 도구

`cot_planner`의 STEP 2에서 날짜·인물 관계 확인에 사용.  
`wikipedia-api` 패키지가 없거나 네트워크 오류 시 빈 문자열을 반환해 챗봇 전체가 중단되지 않는다.

**주요 함수**

```python
wiki_search_summary(query, max_chars=3000)
# 검색어로 위키 문서를 찾아 본문 텍스트를 반환 (없으면 단종 문서로 폴백)

wiki_section_text(title, section_keywords=["생애","즉위"], max_chars=3000)
# 특정 문서에서 키워드가 포함된 섹션의 본문만 추출
```

---

## 환경 설정

```bash
# 필수 환경변수
export OPENAI_API_KEY="sk-..."

# 필수 패키지
pip install langchain langchain-openai langchain-community faiss-cpu wikipedia-api python-dotenv
```

`.env` 파일을 `chatbot/` 또는 프로젝트 루트(`SillokChatbot/`)에 두어도 자동으로 로드된다.

---

## FAISS 인덱스

기본 경로: `SillokChatbot/faiss/` (스크립트 위치 기준 `../faiss`)  
`--faiss` 옵션으로 경로를 변경할 수 있다.

임베딩 모델: `text-embedding-3-large` (OpenAI)

---

## 의존 관계

```
chatbot.py
├── cot_planner.py
│   ├── wiki_tool.py
│   └── metadata_utils.py
├── retriever.py
│   └── metadata_utils.py
└── (FAISS 인덱스)

chatbot_web.py
└── chatbot.py (DanjongBot, DATA_LEVELS, DEFAULT_INDEX_DIR)
```
