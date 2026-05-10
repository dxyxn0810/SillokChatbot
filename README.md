# 📜 조선왕조실록 가상 인터뷰 챗봇

조선왕조실록 데이터를 기반으로 역사적 인물과 대화할 수 있는 **GPT-4o 기반 RAG 챗봇**입니다.

Wikipedia API로 인물 정보를 조회하고, FAISS 벡터 검색과 실록 사이트 실시간 크롤링을 결합하여 거시적·미시적 질문 모두에 답변합니다.

```
SillokChatbot: 안녕하세요, 조선 시대 인물 중 인터뷰를 하고 싶은 사람이 있다면 입력해주세요.
User: 세종대왕
SillokChatbot: "세종대왕" 페르소나를 불러오는 중입니다…
SillokChatbot: 인터뷰 준비가 완료되었습니다!

User: 훈민정음은 어떻게 만드시게 되었나요?
세종: 과인이 글자를 만든 이유는 오직 하나, 백성을 사랑하는 마음 때문이었노라...
```

---

## 🗂️ 프로젝트 구조

```
.
├── xml/                      # 조선왕조실록 원본 XML 파일
│   └── 2nd_{king_code}_{idx}.xml
├── data/                     # XML → JSON 변환 결과
│   └── {king_code}_{idx}.json
├── index/                    # FAISS 벡터 인덱스 (자동 생성)
│   ├── {king_name}.faiss
│   └── {king_name}_meta.pkl
│
├── crawl.py                  # 조선왕조실록 사이트 크롤러
├── xml_to_jsonl.py           # XML → JSONL 단일 파일 변환
├── batch_convert.py          # XML 폴더 일괄 변환 (xml/ → data/)
├── persona.py                # Wikipedia API 인물 정보 조회
├── retriever.py              # RAG 검색 엔진 (FAISS + 크롤링)
├── sillok_chatbot.py         # 메인 챗봇 실행 파일
└── README.md
```

---

## ⚙️ 시스템 아키텍처

```
[인물 입력]
     │
     ▼
[persona.py] ── Wikipedia API ──▶ 생몰년 / 재위기간 / 인물 요약
     │
     ▼
[retriever.py] ── data/*.json ──▶ 왕별 FAISS 인덱스 빌드
     │
     ▼
[사용자 질문]
     │
     ├─ 거시적 질문 ("생애", "업적", "전반")
     │       └──▶ 연도별 대표 기사 + 벡터 유사도 top-k
     │
     └─ 미시적 질문 ("28년 9월에", "몇 월 며칠")
             └──▶ ① 날짜 필터 (year / month / day)
                  ② 필터 결과 내 벡터 유사도 재정렬
                  ③ 결과 부족 시 실록 사이트 실시간 크롤링
                  ④ 전체 벡터 검색 (최후 수단)
     │
     ▼
[GPT-4o]  system: 페르소나 + 인물 배경
          user:   [실록 참고 기사] + 질문
     │
     ▼
[조선 시대 말투 답변]
```

---

## 🚀 설치 및 실행

### 1. 의존 패키지 설치

```bash
pip install openai faiss-cpu sentence-transformers requests \
            beautifulsoup4 langchain langchain-text-splitters selenium
```

### 2. API 키 설정

```bash
export OPENAI_API_KEY="sk-..."
```

### 3. XML 데이터 준비 및 변환

`xml/` 폴더에 원본 XML 파일을 넣고 일괄 변환합니다.
파일명은 반드시 `2nd_{king_code}_{idx}.xml` 형식이어야 하며, `idx` 앞자리가 `1`인 파일(즉위 후 재위 연도 기록)만 처리됩니다.

```bash
python batch_convert.py --xml-dir ./xml --data-dir ./data
```

### 4. 챗봇 실행

```bash
python sillok_chatbot.py
```

최초 실행 시 FAISS 인덱스가 자동으로 빌드됩니다. 이후 실행부터는 저장된 인덱스를 재사용합니다.

---

## 💬 사용법

| 입력 | 동작 |
|------|------|
| 인물 이름 | 페르소나 설정 (예: `세종대왕`, `태조`, `정도전`) |
| 자유 질문 | 해당 인물로서 실록 기반 답변 |
| `/인물 태종` | 대화 중 인물 교체 |
| `q` / `quit` | 종료 |

### 질문 예시

```
# 거시적 질문
세종대왕의 주요 업적을 알려주세요.
재위 기간 동안 어떤 일들을 하셨나요?

# 미시적 질문
세종 28년 9월에 어떤 일이 있었나요?
1446년에 훈민정음 반포와 관련된 기록이 있나요?
세종 10년 3월 5일에는 무슨 일을 하셨나요?
```

---

## 🔧 개별 모듈 사용법

### XML → JSON 단일 파일 변환

```bash
python xml_to_jsonl.py input.xml [output.jsonl]
```

### 실록 사이트 크롤링 (특정 날짜)

```python
from crawl import collect_sillok_custom, save_docs_to_jsonl

docs = collect_sillok_custom("세종", target_year=28, target_month=9)
save_docs_to_jsonl(docs, "sejong_28_09.jsonl")
```

### Wikipedia 인물 정보 조회

```python
from persona import get_persona

info = get_persona("세종대왕")
print(info["summary"])        # 인물 요약
print(info["active_years"])   # (1418, 1450)
```

---

## 📁 king_code 표

XML 파일명 및 실록 URL에 사용되는 왕 코드입니다.

| king_code | 왕 | king_code | 왕 |
|-----------|-----|-----------|-----|
| waa / kaa | 태조 | wna / kna | 선조 |
| wba / kba | 정종 | woa / koa | 광해군 |
| wca / kca | 태종 | wpa / kpa | 인조 |
| wda / kda | 세종 | wqa / kqa | 효종 |
| wea / kea | 문종 | wra / kra | 현종 |
| wfa / kfa | 단종 | wsa / ksa | 숙종 |
| wga / kga | 세조 | wta / kta | 경종 |
| wha / kha | 예종 | wua / kua | 영조 |
| wia / kia | 성종 | wva / kva | 정조 |
| wja / kja | 연산군 | wwa / kwa | 순조 |
| wka / kka | 중종 | wxa / kxa | 헌종 |
| wla / kla | 인종 | wya / kya | 철종 |
| wma / kma | 명종 | wza / kza | 고종 |

> `w`로 시작하는 코드는 XML 파일명용, `k`로 시작하는 코드는 실록 사이트 URL용입니다.

---

## 🛠️ 기술 스택

| 구성 요소 | 기술 |
|-----------|------|
| LLM | GPT-4o (`gpt-4o`) |
| 임베딩 | `paraphrase-multilingual-MiniLM-L12-v2` |
| 벡터 DB | FAISS (`faiss-cpu`) |
| 인물 정보 | Wikipedia API (한국어) |
| 크롤링 | `requests` + `BeautifulSoup` + `selenium` |
| 텍스트 분할 | LangChain `RecursiveCharacterTextSplitter` |
| 데이터 출처 | [조선왕조실록](https://sillok.history.go.kr) |

---

## 📝 데이터 출처 및 저작권

본 프로젝트는 [국사편찬위원회 조선왕조실록](https://sillok.history.go.kr)의 데이터를 연구·교육 목적으로 활용합니다. 국역 본문의 저작권은 세종대왕기념사업회에 있습니다.
