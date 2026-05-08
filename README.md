# SillokChatbot - 조선왕조실록 챗봇

조선왕조실록(史庫)의 데이터를 수집하고 처리하여 RAG(Retrieval-Augmented Generation) 기반의 챗봇을 구축하는 프로젝트입니다.

## 📋 프로젝트 개요

이 프로젝트는 다음 두 단계로 구성됩니다:

1. **URL 수집** (`sillok_crawler.py`): 조선왕조실록 웹사이트에서 모든 기사의 URL을 수집
2. **데이터 처리** (`crawl.py`): 수집된 URL에서 HTML을 크롤링하고 구조화된 문서로 변환

## 🏗️ 프로젝트 구조

```
SillokChatbot/
├── sillok_crawler.py      # 전체 기사 URL 수집 크롤러
├── crawl.py               # 개별 기사 데이터 처리 및 문서화
├── requirements.txt       # 필요한 Python 패키지
└── README.md             # 본 파일
```

## 🔧 설치 및 환경 설정

### 1. 필수 패키지 설치

```bash
pip install -r requirements.txt
```

### 필요한 라이브러리

- **requests**: HTTP 요청 및 웹 크롤링
- **beautifulsoup4**: HTML 파싱
- **selenium**: 자동화된 브라우저 제어 (동적 콘텐츠 처리)
- **langchain** 계열: 문서 처리 및 RAG 파이프라인
- **sentence-transformers**: 텍스트 임베딩
- **faiss-cpu**: 벡터 검색 인덱싱

## 📚 주요 기능

### 1. `sillok_crawler.py` - URL 수집 크롤러

조선왕조실록 웹사이트의 모든 기사 URL을 체계적으로 수집합니다.

**동작 방식:**

1. `/search/inspectionList.do`에서 왕 코드 목록 추출 (kaa, kba, kca, ...)
2. `/search/inspectionMonthList.do`에서 각 왕의 연/월 ID 목록 추출
3. `/search/inspectionView.do`에서 해당 월의 모든 기사 URL 추출

**사용 방법:**

```bash
# 기본 실행 (지정된 왕부터 시작)
python sillok_crawler.py

# 요청 간 딜레이 조정 (기본값: 0.5초)
python sillok_crawler.py --delay 1.0

# 중단된 작업 재개
python sillok_crawler.py --resume
```

**주요 기능:**

- ✅ 자동 재시도 로직 (HTTP 에러 발생 시)
- ✅ 진행 상황 저장 (`sillok_progress.json`)
- ✅ 중단 후 재개 가능
- ✅ 서버 부하 최소화를 위한 딜레이 적용
- ✅ 왕별로 URL 파일 분리 저장 (`url/` 폴더)

### 2. `crawl.py` - 데이터 처리 및 문서화

개별 기사 URL에서 HTML을 크롤링하고 구조화된 LangChain Document로 변환합니다.

**주요 함수:**

#### HTML 크롤링
```python
crawl_with_requests(url)
```
- 브라우저처럼 보이기 위한 User-Agent 헤더 설정
- 자동 재시도 로직 포함
- 일반적인 인식 회피를 위한 랜덤 딜레이 (1.5~3.5초)

#### 텍스트 추출
```python
extract_raw_text_from_html(html_content)
```
- HTML에서 불필요한 태그(script, style, nav, footer) 제거
- 본문 콘텐츠 영역 (`#content`, `<article>`, `<main>`) 우선 추출
- 유의미한 길이(100자 이상)의 텍스트만 추출

#### 메타데이터 추출

**제목 추출:**
```python
extract_title_from_text(raw_text)
```
- "신역 바로가기" 키워드 기반 제목 탐색
- 연도 표시 다음 줄 추출
- 최후의 수단: 특정 키워드 기반 유추

**본문 추출:**
```python
extract_content_from_text(raw_text)
```
- "국역" 섹션에서 본문 추출
- 저작권 마크("ⓒ 세종대왕기념사업회")를 경계선으로 사용
- "원문" 섹션 이전까지의 콘텐츠만 추출

**카테고리 추출:**
```python
extract_category_from_text(raw_text)
```
- "【분류】" 섹션에서 카테고리 추출
- "왕실-의식" 형태의 대분류-중분류 구조만 필터링
- 버튼 텍스트(첫 페이지, 이전 페이지 등) 제외

#### 문서 변환 및 청크 분할
```python
create_sillok_documents(url, raw_text)
```
- URL에서 왕 이름, 날짜, 인덱스 등의 메타데이터 파싱
- 본문을 청크 크기 800, 중복 100으로 분할 (RAG 최적화)
- LangChain Document 객체로 변환
- 각 문서에는 제목, 카테고리, 본문, 메타데이터 포함

#### 데이터 저장
```python
save_docs_to_jsonl(docs, filename)
```
- Document 객체들을 JSONL 형식으로 저장
- 각 줄은 하나의 JSON 객체 (page_content + metadata)

#### 대량 수집
```python
collect_sillok_custom(target_king_name, target_year, target_month, target_lunar, target_day)
```
- 특정 왕의 특정 시점 데이터 수집
- 연/월/윤달/일 단위로 세밀한 필터링 가능
- 자동 재시도 및 404 처리

**사용 예시:**

```python
# 세종의 7년 7월 윤달 2일 데이터 수집
docs = collect_sillok_custom("세종", 7, 7, 1, 2)

# JSONL 파일로 저장
save_docs_to_jsonl(docs, "data.jsonl")
```

## 🔑 핵심 데이터 구조

### URL 형식
```
https://sillok.history.go.kr/id/{king_code}_{date_index}_{article_id}
```
- `king_code`: 왕 코드 (kaa=태조, kda=세종, kua=영조 등)
- `date_index`: YYMMLD 형식 (Y=재위년, M=월, L=윤달여부, D=일)
- `article_id`: 해당 날짜의 기사 인덱스

### Document 메타데이터
```python
{
    "king": "세종",          # 왕 이름
    "year": "7년",           # 재위년
    "month": "7월",          # 월 (윤달이면 "윤7월")
    "day": "2일",            # 일
    "idx": 1,                # 기사 인덱스
    "title": "기사 제목",
    "article_id": "kda_...",
    "category": ["왕실-의식", ...],  # 분류 목록
    "chunk_id": 0            # 청크 인덱스
}
```

## 📊 사용 워크플로우

### 단계 1: URL 수집
```bash
cd SillokChatbot
python sillok_crawler.py --delay 0.5
# 생성: url/ 폴더에 왕별 URL 파일 (예: 세종_url.txt)
```

### 단계 2: 데이터 처리
```bash
python crawl.py
# 생성: data.jsonl (구조화된 문서)
```

### 단계 3: 임베딩 및 벡터 인덱싱 (예상)
```python
from sentence_transformers import SentenceTransformer
import faiss

# Document들을 로드하여 임베딩 생성
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embeddings = model.encode([doc.page_content for doc in docs])

# FAISS 인덱스 구축
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)
```

## ⚙️ 주요 설정값

### 요청 설정
- **User-Agent**: Chrome 기반 (봇 차단 회피)
- **요청 타임아웃**: 30초
- **재시도 횟수**: 최대 5회 (지수 백오프)
- **딜레이**: 기본 0.5초~3.5초

### 텍스트 청킹
- **청크 크기**: 800 토큰
- **중복 범위**: 100 토큰
- **구분자 우선순위**: "\n\n" > "\n" > " " > ""

## 🚀 실행 중 주의사항

1. **서버 부하**: 전체 데이터 수집 시 상당한 시간이 걸립니다. 필요시 특정 왕으로 제한하세요.
2. **네트워크**: 안정적인 인터넷 연결이 필수입니다.
3. **저장 공간**: 전체 실록 데이터는 수GB의 저장 공간이 필요할 수 있습니다.
4. **차단 회피**: 과도한 요청으로 IP 차단될 수 있으니 `--delay`를 충분히 설정하세요.

## 📝 로그 및 디버깅

- **진행 상황**: `sillok_progress.json`에 저장
- **에러 로그**: 표준 에러 출력 (stderr)
- **데이터 검증**: 저장된 JSONL 파일의 각 줄이 유효한 JSON인지 확인

## 🔗 참고자료

- 조선왕조실록: https://sillok.history.go.kr/
- LangChain 공식 문서: https://python.langchain.com/
- FAISS: https://github.com/facebookresearch/faiss

## 📄 라이선스

이 프로젝트는 교육 및 연구 목적으로 작성되었습니다.
조선왕조실록의 저작권은 세종대왕기념사업회에 있습니다.
