# Preprocessing

조선왕조실록 챗봇을 위한 데이터 수집 및 전처리 파이프라인입니다.  
URL 수집 → 기사 크롤링 → 계층적 요약 생성 → 데이터 병합 → FAISS 벡터 인덱스 구축의 순서로 실행합니다.

---

## 전체 파이프라인

```
url_crawler.py
      │  data/url/{왕이름}_url.txt
      ▼
  crawl.py
      │  article/{왕코드}_{YYMMM}.jsonl
      ▼
daily_summary.py
      │  data/daily_summary/{왕코드}_{YYMMM}_day.jsonl
      ▼
monthly_summary.py
      │  data/monthly_summary/{왕코드}_month.jsonl
      ▼
yearly_summary.py
      │  data/yearly_summary/{왕코드}_year.jsonl
      ▼
   merge.py
      │  data/data.jsonl
      ▼
build_faiss.py
         faiss/  (벡터 인덱스)
```

---

## 파일별 설명

### 1. `url_crawler.py` — 기사 URL 수집기

조선왕조실록 공식 사이트(`sillok.history.go.kr`)에서 전체 기사 URL을 왕별로 수집합니다.

**동작 방식**

| 단계 | 엔드포인트 | 수집 내용 |
|------|-----------|-----------|
| 1 | `/search/inspectionList.do` | 왕 코드 목록 (`kaa`, `kba`, …) |
| 2 | `/search/inspectionMonthList.do?id={king_code}` | 해당 왕의 연월 ID 목록 |
| 3 | `/search/inspectionView.do` (POST) | 해당 월의 기사 URL 전체 |

- 수집한 URL은 `data/url/{왕이름}_url.txt` 파일로 왕별로 저장됩니다.
- 요청 간 딜레이, 재시도 로직(최대 3회)으로 서버 부하를 최소화합니다.

**실행 방법**

```bash
cd preprocessing
python url_crawler.py              # 기본 실행 (딜레이 0.5초)
python url_crawler.py --delay 1.0  # 딜레이 1초
```

---

### 2. `crawl.py` — 기사 본문 크롤러

`url_crawler.py`가 생성한 URL 파일을 읽어 각 기사의 본문을 크롤링하고, LangChain `Document` 형식의 JSONL 파일로 저장합니다.

**주요 기능**

- `requests` + `BeautifulSoup`으로 HTML 파싱
- URL의 기사 ID(`kga_10203014_001` 등)에서 왕, 재위년, 양력 연도, 월, 일 자동 추출
- 국역 본문, 기사 제목, 분류 카테고리 추출
- `RecursiveCharacterTextSplitter`로 본문을 청크 분할 (chunk_size=800, overlap=100)
- URL을 월별로 그룹화하여 `article/{왕코드}_{YYMMM}.jsonl` 파일로 저장
- `skip_existing=True` 옵션으로 이미 완료된 월은 건너뜀 (재실행 안전)

**Document 메타데이터 구조**

```json
{
  "type": "article",
  "title": "기사 제목",
  "king": "세조",
  "year": "1년",
  "solar_year": 1455,
  "month": "윤6월",
  "day": "11일",
  "idx": 1,
  "chunk_id": 0
}
```

**실행 방법**

```bash
cd preprocessing
python crawl.py  # url_file 경로는 스크립트 내 DEFAULT_URL_DIR 참조
```

---

### 3. `daily_summary.py` — 일별 요약 생성기

`article/` 폴더의 JSONL 파일을 읽어, 같은 날짜에 기록된 기사들을 하나로 모아 OpenAI LLM으로 **하루치 요약**을 생성합니다.

**동작 방식**

1. 기사 청크를 `(왕, 재위년, 월, 일)` 기준으로 그룹화
2. 청크 overlap을 제거하며 기사별 본문 복원
3. GPT 모델에 하루치 기사 전체를 입력하여 `{"title": ..., "summary": ...}` JSON 생성
4. 요약 결과를 `daily_summary` 타입 Document로 변환하여 저장

**출력 파일**: `data/daily_summary/{왕코드}_{YYMMM}_day.jsonl`

**실행 방법**

```bash
export OPENAI_API_KEY=...
cd preprocessing
python daily_summary.py
python daily_summary.py --pattern "kga_*.jsonl" --model gpt-4o
```

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--article-dir` | `data/article` | 원본 기사 폴더 |
| `--output-dir` | `data/daily_summary` | 출력 폴더 |
| `--model` | `gpt-4o-mini` | 요약에 사용할 OpenAI 모델 |
| `--pattern` | `kga_*.jsonl` | 처리할 파일 글롭 패턴 |
| `--overwrite` | False | 기존 파일 덮어쓰기 여부 |

---

### 4. `monthly_summary.py` — 월별 요약 생성기

`daily_summary/` 폴더의 JSONL 파일을 읽어, 한 달치 일별 요약들을 종합하여 OpenAI LLM으로 **월별 요약**을 생성합니다.

**동작 방식**

1. daily_summary 문서를 `(왕, 재위년, 월)` 기준으로 그룹화 (일 오름차순 정렬)
2. 한 달치 일별 요약을 연결한 전체 텍스트 구성
3. GPT 모델로 월 단위 종합 요약 생성 (주제별 흐름으로 정리)
4. `monthly_summary` 타입 Document로 저장

**출력 파일**: `data/monthly_summary/{왕코드}_month.jsonl`

**실행 방법**

```bash
export OPENAI_API_KEY=...
cd preprocessing
python monthly_summary.py
python monthly_summary.py --pattern "kga_*_day.jsonl"
```

---

### 5. `yearly_summary.py` — 연별 요약 생성기

`monthly_summary/` 폴더의 JSONL 파일을 읽어, 한 해치 월별 요약들을 종합하여 OpenAI LLM으로 **연별 요약**을 생성합니다.

**동작 방식**

1. monthly_summary 문서를 `(왕, 재위년)` 기준으로 그룹화 (월 오름차순, 윤달은 해당 달 뒤)
2. 한 해치 월별 요약을 연결한 전체 텍스트 구성
3. GPT 모델로 연 단위 종합 요약 생성 (왕실·외교·재정·반란 등 주제별로 묶어 정리)
4. `yearly_summary` 타입 Document로 저장

**출력 파일**: `data/yearly_summary/{왕코드}_year.jsonl`

**실행 방법**

```bash
export OPENAI_API_KEY=...
cd preprocessing
python yearly_summary.py
python yearly_summary.py --input-file data/monthly_summary/kga_month.jsonl
```

---

### 6. `merge.py` — 데이터 병합기

`article`, `daily_summary`, `monthly_summary`, `yearly_summary` 폴더의 모든 JSONL 파일을 **하나의 파일**로 합칩니다.

**병합 순서**

```
article/*.jsonl
  → data/daily_summary/*.jsonl
    → data/monthly_summary/*.jsonl
      → data/yearly_summary/*.jsonl
        → (data/monthly_summary/kga_month.jsonl 추가)
```

- 각 줄마다 JSON 유효성 검증을 수행하며, 파싱 실패 줄은 스킵합니다.

**출력 파일**: `data/data.jsonl`

**실행 방법**

```bash
cd preprocessing
python merge.py
```

---

### 7. `build_faiss.py` — FAISS 벡터 인덱스 구축기

`data/data.jsonl`을 LangChain `Document` 리스트로 변환하고, OpenAI 임베딩 모델로 임베딩하여 **FAISS 벡터스토어**를 구축합니다.

**동작 방식**

1. JSONL 파일의 각 줄을 `Document(page_content, metadata)` 로 변환
2. Document type별 분포 출력 (`article`, `daily_summary`, `monthly_summary`, `yearly_summary`)
3. 배치 단위(`--batch-size`, 기본 200)로 임베딩 요청 후 FAISS 인덱스에 누적
4. 완성된 인덱스를 `faiss/` 폴더에 저장

**출력 디렉토리**: `faiss/` (LangChain FAISS 포맷)

**실행 방법**

```bash
export OPENAI_API_KEY=...
cd preprocessing
python build_faiss.py
python build_faiss.py --embedding-model text-embedding-3-large --batch-size 500
```

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--input-file` | `data/data.jsonl` | 입력 파일 |
| `--index-dir` | `faiss/` | FAISS 인덱스 저장 폴더 |
| `--embedding-model` | `text-embedding-3-large` | OpenAI 임베딩 모델 |
| `--batch-size` | `200` | 배치 크기 |

---

## 환경 설정

```bash
pip install requests beautifulsoup4 selenium langchain langchain-openai \
            langchain-community faiss-cpu sentence-transformers openai python-dotenv
```

`.env` 파일 또는 환경 변수로 API 키를 설정합니다.

```env
OPENAI_API_KEY=sk-...
```

---

## 데이터 디렉토리 구조

```
SillokChatbot/
├── preprocessing/          # 이 폴더 (전처리 스크립트)
├── data/
│   ├── url/                # 왕별 기사 URL 목록 (url_crawler.py 출력)
│   ├── article/            # 기사 원문 청크 (crawl.py 출력) → article/ 에도 저장
│   ├── daily_summary/      # 일별 요약 (daily_summary.py 출력)
│   ├── monthly_summary/    # 월별 요약 (monthly_summary.py 출력)
│   ├── yearly_summary/     # 연별 요약 (yearly_summary.py 출력)
│   └── data.jsonl          # 전체 병합 파일 (merge.py 출력)
└── faiss/                  # FAISS 벡터 인덱스 (build_faiss.py 출력)
```
