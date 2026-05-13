# SillokChatbot

`SillokChatbot`는 조선왕조실록 데이터를 기반으로 가상 인터뷰를 진행하는 Python CLI 챗봇입니다. 사용자 입력에 따라 특정 왕과 연도, 월, 일자에 해당하는 실록 기사를 검색하고, 해당 기사 내용을 바탕으로 임금 페르소나가 답변하도록 구성되어 있습니다.

## 주요 기능

- 왕 이름, 즉위 연도, 월, 일자 입력을 받음
- `url/` 폴더에 있는 URL 목록 파일에서 해당 기사 URL만 필터링
- `crawl.py`의 `collect_sillok_data()`를 통해 기사 내용을 크롤링하고 LangChain `Document`로 변환
- OpenAI Embeddings로 문서를 벡터화하고 FAISS 인덱스로 저장
- RAG(검색-지원 생성) 방식으로 `gpt-4o-mini` 기반 인터뷰 답변 생성
- 동일 데이터에 대한 재사용 가능한 캐시(`jsonl`, `faiss`) 지원

## 파일 설명

### `sillok_chatbot.py`

이 프로젝트의 메인 실행 스크립트입니다. 주요 역할은 다음과 같습니다.

1. 입력 파싱
   - `parse_king_name()`
   - `parse_year_number()`
   - `parse_month()`
   - `parse_days()`
2. URL 필터링
   - `filter_urls()`를 통해 `url/{king}_url.txt`에서 조건에 맞는 기사 URL만 선택
3. 크롤링과 벡터스토어 생성
   - `build_vectorstore()`는 FAISS 인덱스가 이미 존재하는지 확인하고 없으면 크롤링 및 임베딩 수행
   - `find_reusable_jsonl()`으로 기존 JSONL 캐시 재사용 가능 여부 확인
4. 페르소나 챗봇 생성
   - `SillokInterviewBot` 클래스는 RAG 기반 질의 응답과 대화 히스토리를 관리
5. CLI 인터페이스
   - `interactive_setup()`으로 인터뷰 조건을 입력받고 준비
   - `chat_loop()`으로 사용자 질문을 받아 모델 답변 출력

### `crawl.py`

- `collect_sillok_data()`
- `save_docs_to_jsonl()`
- `KING_MAP`, `KING_START_YEAR`

이 파일은 실록 데이터 수집과 문서 생성 로직을 담당하며, `sillok_chatbot.py`에서 직접 재사용합니다.

## 실행 전 준비

1. Python 가상환경 활성화
2. 필요한 패키지 설치

```powershell
pip install -r requirements.txt
```

3. OpenAI API 키 설정

`OPENAI_API_KEY`를 코드 상단 또는 환경 변수에 설정해야 합니다.

> 현재 `sillok_chatbot.py`는 코드 상단에 `os.environ['OPENAI_API_KEY'] = "sk-proj-..."`가 포함되어 있습니다. 실제 키로 교체하거나 환경 변수로 설정하세요.

## 사용 방법

1. `url/` 폴더에 왕별 URL 목록 파일(`세종_url.txt` 등)을 준비
2. `python sillok_chatbot.py` 실행
3. 다음 순서대로 입력
   - 왕 이름
   - 즉위 몇 년차인지
   - 몇 월인지
   - 몇 일인지 또는 `전체`
4. 질문을 입력하면 해당 왕 페르소나가 답변

종료하려면 `exit`, `quit`, `종료`를 입력합니다.

## 저장 폴더

- `url/` : 기사 URL 목록 파일 저장
- `jsonl/` : 크롤링된 문서 캐시 저장
- `faiss/` : 생성된 FAISS 벡터 인덱스 저장

## 주의 사항

- Windows 환경에서 FAISS는 한글 경로를 제대로 처리하지 못할 수 있습니다. `faiss/`와 `jsonl/` 경로는 ASCII 문자만 포함되도록 관리하세요.
- `sillok_chatbot.py`와 `crawl.py`는 같은 디렉터리에 있어야 합니다.

## 확장 포인트

- 더 많은 왕과 연도 지원
- RAG 문맥 구성 강화
- 추가적인 입력 오류 처리 및 사용자 경험 개선
- 다른 LLM 모델이나 임베딩 모델로 대체
