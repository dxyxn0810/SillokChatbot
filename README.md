# SillokChatbot — 단종 가상 인터뷰 RAG 챗봇

조선왕조실록(문종·단종·세조 시기) 데이터를 기반으로, **조선 제6대 임금 단종(이홍위)** 페르소나가 1인칭으로 답하는 RAG 챗봇입니다.

전체 시스템은 두 부분으로 나뉩니다.

- **`preprocessing/`** — 실록 데이터를 수집·가공·요약하고 FAISS 벡터 인덱스를 구축하는 데이터 파이프라인
- **`doyoon_project/`** — 그 인덱스를 사용해 CoT 검색 계획 + RAG 로 답변을 생성하는 챗봇

```
실록 사이트
   │  (preprocessing)
   ▼
url_crawler → crawl → daily_summary → monthly_summary → yearly_summary → merge → build_faiss
                                                                                      │
                                                                                      ▼
                                                                                 faiss/ 인덱스
                                                                                      │  (doyoon_project)
                                                                                      ▼
                                                              chatbot ← cot_planner ← wiki_tool
                                                                  ↑         ↓
                                                              retriever ← metadata_utils
```

데이터는 같은 사건을 **개별 기사 → 일별 요약 → 월별 요약 → 연별 요약** 4개 층위로 함께 인덱싱합니다. 질문의 범위(구체적 사건인지, 전반적 평가인지)에 맞는 입자도의 문서를 검색하기 위한 설계입니다.

---

## 1. `preprocessing/` — 데이터 구축 파이프라인

아래 순서대로 실행하면 원본 수집부터 벡터 인덱스 구축까지 완료됩니다. 요약·임베딩 단계는 OpenAI API 키(`OPENAI_API_KEY`)가 필요합니다.

### `url_crawler.py`
실록 사이트(`sillok.history.go.kr`)에서 **기사 URL 목록**을 수집합니다.

- 표준 라이브러리(`urllib`)만 사용하며 3단계로 진행: 왕 코드 목록(kaa, kba…) → 왕별 연·월 ID → 각 월의 개별 기사 ID
- 정규식으로 HTML을 파싱하고, 요청 간 딜레이(`--delay`)와 재시도(최대 3회)로 서버 부하를 관리
- 결과를 왕별로 `data/url/{왕이름}_url.txt`에 저장

```bash
python url_crawler.py --delay 1.0
```

### `crawl.py`
수집한 URL의 **본문을 크롤링**해 LangChain `Document`로 변환합니다.

- `requests` + `BeautifulSoup`으로 본문을 받아 제목·국역 내용·분류(카테고리)를 추출
- 기사 ID를 정규식으로 파싱해 왕·재위년·양력연도·월(윤달 포함)·일 메타데이터 생성 (`KING_START_YEAR`로 음력 재위년을 양력으로 변환)
- `RecursiveCharacterTextSplitter`로 800자(overlap 100) 단위로 청크 분할
- URL을 월 단위로 그룹화해 `{왕코드}_{YYMMl}.jsonl`로 저장 (`skip_existing`으로 재실행 시 기존 파일 건너뜀)

주요 함수: `collect_sillok_data()`, `create_sillok_documents()`, `crawl_urls_from_file()`

### `daily_summary.py`
article jsonl을 읽어 **같은 날짜의 기사들을 하루 단위로 요약**합니다.

- 같은 `idx`의 청크를 overlap 중복을 제거하며 이어붙임(`_join_overlapping`)
- `gpt-4o-mini`로 `{"title", "summary"}` JSON을 생성 (시스템 프롬프트로 형식 강제)
- 결과를 `type="daily_summary"`로 표시해 `data/daily_summary/`에 저장

```bash
python daily_summary.py --pattern "kga_*.jsonl"
```

### `monthly_summary.py`
daily_summary를 **한 달 단위로 종합 요약**합니다(`type="monthly_summary"`). 일별 나열이 아니라 주제(국상·인사·외교·재정 등)별 흐름으로 묶도록 프롬프트가 설계되어 있습니다.

### `yearly_summary.py`
monthly_summary를 **한 해 단위로 종합 요약**합니다(`type="yearly_summary"`). 윤달 정렬(`_month_sort_key`)을 처리합니다.

### `merge.py`
`article` + 일/월/연 요약 + `kfa` 파일을 순서대로 합쳐 단일 **`data/data.jsonl`**을 만듭니다. 유효하지 않은 JSON 라인은 건너뜁니다.

### `build_faiss.py`
`data.jsonl`을 LangChain `Document`로 로드한 뒤 OpenAI 임베딩(`text-embedding-3-large`)으로 **배치 임베딩하여 FAISS 인덱스를 구축**하고 `faiss/`에 저장합니다. type별 문서 분포를 출력해 줍니다.

```bash
python build_faiss.py --batch-size 200
```

---

## 2. `doyoon_project/` — RAG 챗봇

사용자 질문을 받아 **CoT로 검색 계획을 세우고 → 메타데이터 필터로 검색 → 단종 1인칭으로 답변**하는 흐름입니다.

```
질문 → cot_planner.make_plan()  (CoT + 위키백과로 검색 메타데이터 결정)
     → retriever.retrieve()      (메타데이터 필터 + 점진적 완화 검색)
     → gpt-4o 가 단종 페르소나로 답변
```

### `chatbot.py` — 메인 실행 스크립트
전체 파이프라인을 묶어 대화형 인터페이스를 제공합니다. `DanjongBot` 클래스가 검색·답변·대화기록을 관리하고, `PERSONA_SYSTEM` 프롬프트에 단종의 인물 관계(숙부/사촌 항렬), 어조, "자료에 없으면 지어내지 말 것" 같은 환각 방지 규칙이 담겨 있습니다.

**실행 옵션 (argparse)**

| 옵션 | 값 / 기본값 | 설명 |
|------|------------|------|
| `--data` | `prompt` \| `article` \| `summary` \| `wikipedia` (기본 `wikipedia`) | 사용할 정보 소스 범위(누적적) |
| `--CoT` | `on` \| `off` (기본 `on`) | CoT 기반 메타데이터 필터링 사용 여부 |
| `--history` | 정수 (기본 6) | 답변 시 참고할 직전 대화 맥락 턴 수 (0이면 미사용) |
| `--retrieval` | 정수 (기본 10) | 검색해 올 문서 수(k) |
| `--faiss` | 경로 | FAISS 인덱스 폴더 |
| `--verbose` | flag | CoT·검색 과정 출력 |
| `--once` | 문자열 | 한 번만 질문하고 종료 |

`--data` 레벨은 누적적입니다.

- `prompt` : 검색 없이 페르소나 프롬프트만으로 답변
- `article` : 실록 기사(`article`)만 검색
- `summary` : 기사 + 일/월/연 요약까지 검색
- `wikipedia` : 위 + 위키백과 근거까지 사용

```bash
python chatbot.py --data wikipedia --CoT on --history 6 --retrieval 10 --verbose
python chatbot.py --data article --CoT off --retrieval 5
python chatbot.py --once "단종은 어떻게 즉위했나요?"
```

> **참고:** 위키 근거는 CoT 플래너(`cot_planner`)가 수집하므로, `--data wikipedia`는 `--CoT on`과 함께 써야 위키 정보가 반영됩니다. `--CoT off`이면 위키 없이 진행되며 경고가 출력됩니다.

### `cot_planner.py` — CoT 검색 계획가
질문을 받아 **검색에 쓸 메타데이터 필터(JSON)와 의미검색 쿼리를 추론**하는 3단계 파이프라인입니다.

1. **STEP1** : `gpt-4o`가 어떤 메타데이터(왕/연도/월/일/type)로 좁힐지 추론(CoT)하고, 날짜·인물관계 확인이 필요하면 위키 검색어를 제안
2. **STEP2** : 위키백과에서 근거 텍스트 수집
3. **STEP3** : 위키 근거 + 배경지식을 종합해 최종 필터를 확정·정규화

도메인 지식이 프롬프트에 녹아 있습니다. 예를 들어 **단종의 양위(1455) 이후 사건(유배·죽음 등)은 실록에서 `king="세조"`로 분류**되므로 "단종의 최후"를 물어도 king을 단종으로 좁히지 않도록 처리하고, "승하"와 "장례(발인/안장)"의 시점 차이, 음력/양력 구분(데이터는 음력 기준)을 다룹니다. 기간 질문용 `solar_year_range`, `month_range`, `day_range`도 지원합니다.

반환되는 plan 예시:
```json
{
  "king": "세조",
  "solar_year": 1457,
  "month": "10월",
  "doc_types": ["article"],
  "search_query": "노산군 영월 유배 사사"
}
```

### `retriever.py` — 점진적 완화 검색
FAISS 벡터스토어를 로드하고 plan에 따라 검색합니다. 핵심은 **점진적 완화(progressive relaxation)**입니다.

- 연-월-일+type처럼 좁은 필터로 먼저 검색하고, 결과가 부족하면 단계적으로 조건을 풀어 최종적으로 순수 의미검색까지 폴백 → 날짜 추정이 빗나가도 빈 결과가 나오지 않음
- LangChain-FAISS가 "벡터 상위 `fetch_k`개를 먼저 뽑은 뒤 필터링"하는 특성 때문에 `fetch_k=5000`으로 크게 잡아 필터 누락 방지
- 연/월 범위 질문은 연도·월별로 나눠 검색한 뒤 라운드로빈으로 균형 병합

주요 함수: `load_vectorstore()`, `retrieve()`, `format_context()`

### `metadata_utils.py` — 메타데이터 유틸리티
`data.jsonl` 메타데이터 형식을 다루는 헬퍼 모음입니다.

- 정규화: `norm_year` / `norm_month` / `norm_day` (예: `2` → `"2월"`, `"윤6"` → `"윤6월"`)
- 윤달 정렬 키: `lunar_month_key` (같은 숫자면 평달 < 윤달)
- 재위년 ↔ 양력 변환 테이블 (`FULL_REIGN_TABLE`: 문종/단종/세조)
- FAISS 필터 빌더: `build_filter`, 수동 매칭: `metadata_matches`

### `wiki_tool.py` — 위키백과 조회 도구
`wikipedia-api`로 한국어 위키를 조회해 날짜·인물관계 등 외부 사실을 확인합니다.

- `wiki_search_summary()` : 검색어로 문서를 찾아 본문 텍스트 반환
- `wiki_section_text()` : 특정 문서에서 키워드가 든 섹션 본문 추출
- 패키지 미설치·네트워크 실패 시에도 빈 문자열을 반환해 **챗봇 전체가 죽지 않도록** 방어적으로 작성됨

---

## 실행 전 준비

```bash
pip install -r requirements.txt
export OPENAI_API_KEY="sk-..."   # 또는 .env 파일에 작성
```

위키백과 기능을 쓰려면 `wikipedia-api` 패키지가 추가로 필요합니다.

## 문서 type 요약

| type | 설명 | 생성 단계 |
|------|------|-----------|
| `article` | 개별 실록 기사(청크) | `crawl.py` |
| `daily_summary` | 하루치 요약 | `daily_summary.py` |
| `monthly_summary` | 한 달치 종합 요약 | `monthly_summary.py` |
| `yearly_summary` | 한 해치 종합 요약 | `yearly_summary.py` |