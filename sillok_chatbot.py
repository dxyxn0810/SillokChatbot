"""
sillok_chatbot.py
─────────────────────────────────────────────────────────────
조선왕조실록 데이터를 기반으로 한 가상 인터뷰 챗봇.

흐름:
  1) 사용자로부터 왕(king), 즉위 N년(year), 월(month), 일(days)을 입력받는다.
  2) "url/{king}_url.txt" 에서 해당 (왕 코드 + 연도 + 월 + 일) 패턴에
     매칭되는 기사 URL만 필터링한다.
  3) crawl.py 의 collect_sillok_data() 로 일괄 크롤링 → LangChain Document 생성.
  4) OpenAI Embedding으로 임베딩한 뒤 FAISS VectorDB를 ./faiss/ 에 저장.
     (크롤링 결과는 ./jsonl/ 에 캐시)
  5) ChatGPT(gpt-4o-mini 등) + RAG 로 해당 왕 페르소나의 인터뷰 답변을 생성.

폴더 구조:
  .
  ├── sillok_chatbot.py
  ├── crawl.py
  ├── url/
  │   ├── 세종_url.txt
  │   └── ...
  ├── faiss/   (자동 생성 — FAISS 인덱스)
  └── jsonl/   (자동 생성 — 크롤링 결과 캐시)

사전 준비:
  pip install openai langchain langchain-openai langchain-community faiss-cpu \
              langchain-text-splitters sentence-transformers beautifulsoup4 requests
─────────────────────────────────────────────────────────────
"""

import os

# ─────────────────────────────────────────────────────────────
# OpenAI API 키 (코드 안에 직접 설정)
# 🔑 실제 사용 시 아래 값을 본인 API 키로 교체해 주세요.
# 보안을 위해 코드 공유/저장소 커밋 전에는 반드시 키를 제거하세요.
# ─────────────────────────────────────────────────────────────
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

import re
import sys
import json
from pathlib import Path

# crawl.py 의 함수들을 그대로 재사용
# (crawl.py 와 같은 디렉터리에 있어야 함)
from codes.crawl import (
    KING_MAP,
    KING_START_YEAR,
    collect_sillok_data,
    save_docs_to_jsonl,
)

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage


# ─────────────────────────────────────────────────────────────
# 0. 환경 설정
# ─────────────────────────────────────────────────────────────
# 왕 이름 → URL 코드 역매핑 (예: "세종" → "kda")
NAME_TO_CODE = {v: k for k, v in KING_MAP.items()}

# 스크립트 위치 기반 루트 경로 계산
SCRIPT_DIR = Path(__file__).parent.resolve()      # SillokChatbot 루트 폴더

# 임베딩 / LLM 모델 (필요 시 변경)
EMBEDDING_MODEL = "text-embedding-3-large"
CHAT_MODEL = "gpt-4o"

# URL / 인덱스 / 데이터 저장 폴더 (절대 경로)
URL_DIR = SCRIPT_DIR / "data" / "url"
INDEX_DIR = SCRIPT_DIR / "faiss"
DATA_DIR = SCRIPT_DIR / "data"


# ─────────────────────────────────────────────────────────────
# 1. 입력 파싱 유틸
# ─────────────────────────────────────────────────────────────
def parse_king_name(raw: str) -> str:
    """
    '세종대왕', '세종 ', '세종\n' 등 다양한 입력에서 표준 왕 이름('세종') 추출.
    KING_MAP 의 value들과 매칭되는 첫 토큰을 반환.
    """
    raw = raw.strip()
    # '대왕' 같은 접미사 제거
    for suffix in ("대왕", " 대왕"):
        if raw.endswith(suffix):
            raw = raw[: -len(suffix)]
    raw = raw.strip()
    if raw in NAME_TO_CODE:
        return raw
    # 부분 매칭(예: "세종" in "세종실록")
    for name in NAME_TO_CODE:
        if name in raw:
            return name
    raise ValueError(f"알 수 없는 왕 이름입니다: {raw!r}")


def parse_year_number(raw: str) -> int:
    """'1년', '1', ' 1 년 ' → 1"""
    m = re.search(r"\d+", raw)
    if not m:
        raise ValueError(f"연도를 인식할 수 없습니다: {raw!r}")
    return int(m.group())


def parse_month(raw: str) -> tuple[int, bool]:
    """
    '1월', '윤3월', ' 12월 ' → (월 정수, 윤달 여부)
    """
    raw = raw.strip()
    is_lunar = raw.startswith("윤")
    m = re.search(r"\d+", raw)
    if not m:
        raise ValueError(f"월을 인식할 수 없습니다: {raw!r}")
    return int(m.group()), is_lunar


def parse_days(raw: str) -> list[int]:
    """
    여러 가지 형식의 '일' 입력을 정수 리스트로 파싱.

    지원 형식:
      - 콤마 구분         : "1, 2, 3, 4, 5, 6, 7"
      - 공백 구분         : "1 2 3"
      - 범위              : "1-7"           → [1,2,3,4,5,6,7]
      - 범위 + 단일 혼합  : "1-3, 7, 10-12" → [1,2,3,7,10,11,12]
      - 한글 접미사 허용  : "1일, 3일"
      - 전체              : "all" / "전체"  → [] (빈 리스트 = '모든 날짜')

    반환값:
      - 빈 리스트  : '모든 일자 허용' 을 의미 (filter_urls 에서 day 필터를 적용하지 않음)
      - 정렬·중복제거된 정수 리스트
    """
    raw = raw.strip()
    if not raw or raw.lower() in {"all", "전체", "*"}:
        return []

    # "1일" 같은 한글 접미사 제거
    cleaned = raw.replace("일", "")

    days: set[int] = set()
    # 콤마 또는 공백으로 토큰 분리
    tokens = re.split(r"[,\s]+", cleaned)
    for tok in tokens:
        tok = tok.strip()
        if not tok:
            continue
        # 범위 표현 (예: "1-7", "10~12")
        m = re.fullmatch(r"(\d+)\s*[-~]\s*(\d+)", tok)
        if m:
            start, end = int(m.group(1)), int(m.group(2))
            if start > end:
                start, end = end, start
            days.update(range(start, end + 1))
            continue
        # 단일 숫자
        if tok.isdigit():
            days.add(int(tok))
            continue
        raise ValueError(f"일자를 인식할 수 없습니다: {tok!r}")

    # 1~30 범위만 허용 (음력 한 달 최대 30일)
    for d in days:
        if not (1 <= d <= 30):
            raise ValueError(f"일자는 1~30 범위여야 합니다: {d}")

    return sorted(days)


# ─────────────────────────────────────────────────────────────
# 2. URL 필터링
# ─────────────────────────────────────────────────────────────
def filter_urls(
    url_file: Path,
    king_code: str,
    year: int,
    month: int,
    is_lunar: bool = False,
    days: list[int] | None = None,
) -> list[str]:
    """
    URL 파일에서 해당 (왕코드 + 연도 + 월 [+ 일자들]) 기사들만 추려서 반환.

    URL 구조: https://sillok.history.go.kr/id/{king_code}_1YYMMDDDD_NNN
      - 맨 앞 1 : 의미 없는 고정 숫자
      - YY      : 즉위년(2자리, zero-padding)
      - MM      : 월(2자리)
      - D       : 윤달 플래그 (0=평달, 1=윤달)
      - DD      : 일(2자리)
      - NNN     : 해당 일의 기사 인덱스

    days:
      - None 또는 빈 리스트  → 해당 월의 모든 일자
      - [1, 2, 3, ...]      → 해당 일자들만
    """
    yy = f"{year:02d}"
    mm = f"{month:02d}"
    lunar_flag = "1" if is_lunar else "0"
    # (월 까지의) 공통 접두사 — 예: kda_1010101
    month_prefix = f"{king_code}_1{yy}{mm}{lunar_flag}"

    # 일자 필터링용 prefix 집합 만들기
    if days:
        day_prefixes = {f"{month_prefix}{d:02d}" for d in days}
    else:
        day_prefixes = None  # 일 필터 없음

    matched: list[str] = []
    with url_file.open("r", encoding="utf-8") as f:
        for line in f:
            url = line.strip()
            if not url:
                continue
            # URL 마지막 segment 추출 (예: kda_10101001_001)
            article_id = url.rsplit("/", 1)[-1]

            if day_prefixes is None:
                # 월 단위 매칭
                if article_id.startswith(month_prefix):
                    matched.append(url)
            else:
                # 일 단위 매칭 (지정된 일자들만)
                if any(article_id.startswith(p) for p in day_prefixes):
                    matched.append(url)
    return matched


# ─────────────────────────────────────────────────────────────
# 3. 크롤링 → Document 생성 → FAISS 인덱스 구축
# ─────────────────────────────────────────────────────────────
def load_docs_from_jsonl(path: Path) -> list[Document]:
    """jsonl 파일에서 LangChain Document 리스트를 복원."""
    docs: list[Document] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            docs.append(
                Document(
                    page_content=data["page_content"],
                    metadata=data.get("metadata", {}),
                )
            )
    return docs


def _day_int_from_metadata(meta: dict) -> int | None:
    """
    Document.metadata['day'] (예: '11일') 에서 정수 11을 추출.
    파싱 실패 시 None.
    """
    raw = meta.get("day", "")
    m = re.search(r"\d+", str(raw))
    return int(m.group()) if m else None


def find_reusable_jsonl(
    cache_jsonl: Path,
    data_dir: Path,
    king_code: str,
    year: int,
    month: int,
    is_lunar: bool,
) -> Path | None:
    """
    재사용 가능한 jsonl 캐시를 찾는다.

    우선순위:
      1) 정확히 같은 태그(cache_jsonl) 가 있으면 그것
      2) 같은 (왕, 연, 월) 의 `_all` 캐시가 있으면 그것
         (예: 요청이 kda_00_08_d11-17 이고 kda_00_08_all.jsonl 이 있으면 후자 사용)
    없으면 None.
    """
    if cache_jsonl.exists():
        return cache_jsonl

    yun = "_yun" if is_lunar else ""
    all_jsonl = data_dir / f"{king_code}_{year:02d}_{month:02d}{yun}_all.jsonl"
    if all_jsonl.exists():
        return all_jsonl

    return None


def build_vectorstore(
    urls: list[str],
    persist_dir: Path,
    cache_jsonl: Path | None = None,
    *,
    king_code: str | None = None,
    year: int | None = None,
    month: int | None = None,
    is_lunar: bool = False,
    days: list[int] | None = None,
) -> FAISS:
    """
    URL 리스트를 크롤링하여 LangChain Document를 만들고,
    OpenAI 임베딩으로 FAISS 인덱스를 만들어 persist_dir에 저장.

    재사용 우선순위:
      1) persist_dir 에 이미 FAISS 인덱스가 있으면 → 그대로 로드 (임베딩도 생략)
      2) 동일 태그의 jsonl 캐시가 있으면 → 크롤링 생략, 임베딩만 수행
      3) 같은 (왕, 연, 월) 의 `_all` jsonl 캐시가 있으면
         → 거기서 요청 일자만 필터링해 사용 (크롤링 생략)
      4) 위 어느 것도 없으면 → 실제 크롤링 수행

    - 크롤링은 crawl.py 의 `collect_sillok_data()` 를 사용 (일괄 처리).
    - persist_dir / cache_jsonl 경로는 모두 ASCII 여야 함
      (FAISS C++ 코어가 Windows 에서 한글 경로를 처리 못 하는 문제 회피).
    """
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    # ── 1) FAISS 인덱스 캐시 ─────────────────────────────────
    if persist_dir.exists() and (persist_dir / "index.faiss").exists():
        print(f"📂 기존 FAISS 인덱스 발견 → 로드합니다: {persist_dir}")
        return FAISS.load_local(
            str(persist_dir),
            embeddings,
            allow_dangerous_deserialization=True,
        )

    persist_dir.mkdir(parents=True, exist_ok=True)

    # ── 2~3) jsonl 캐시 재사용 시도 ──────────────────────────
    all_docs: list[Document] | None = None
    reusable: Path | None = None

    if cache_jsonl is not None and king_code is not None \
            and year is not None and month is not None:
        reusable = find_reusable_jsonl(
            cache_jsonl, cache_jsonl.parent, king_code, year, month, is_lunar
        )

    if reusable is not None:
        print(f"📂 jsonl 캐시 발견 → 크롤링을 건너뜁니다: {reusable}")
        cached_docs = load_docs_from_jsonl(reusable)
        print(f"   • 캐시된 chunk 수: {len(cached_docs)}개")

        # `_all` 캐시에서 가져온 경우엔 요청한 일자만 필터링
        is_all_cache = reusable.name.endswith("_all.jsonl")
        if is_all_cache and days:
            wanted = set(days)
            filtered = [
                d for d in cached_docs
                if _day_int_from_metadata(d.metadata) in wanted
            ]
            print(
                f"   • '_all' 캐시에서 요청 일자 {sorted(wanted)} 만 추출 "
                f"→ {len(filtered)}개 chunk"
            )
            all_docs = filtered
        else:
            all_docs = cached_docs

        if not all_docs:
            print("⚠️  캐시에서 요청 일자에 해당하는 chunk를 찾지 못했습니다. "
                  "크롤링으로 폴백합니다.")
            all_docs = None

    # ── 4) 실제 크롤링 ──────────────────────────────────────
    if all_docs is None:
        print(f"📥 {len(urls)}개 URL 크롤링을 시작합니다... (crawl.collect_sillok_data)")
        all_docs = collect_sillok_data(urls)

        if not all_docs:
            raise RuntimeError("크롤링된 문서가 하나도 없습니다.")

        # 새로 크롤링한 결과는 캐시로 저장
        if cache_jsonl is not None:
            cache_jsonl.parent.mkdir(parents=True, exist_ok=True)
            save_docs_to_jsonl(all_docs, str(cache_jsonl))

    # ── 임베딩 & 저장 ───────────────────────────────────────
    print(f"📝 총 {len(all_docs)}개 chunk 로 FAISS 인덱스를 생성합니다...")
    vs = FAISS.from_documents(all_docs, embeddings)
    vs.save_local(str(persist_dir))
    print(f"💾 FAISS 인덱스 저장 완료: {persist_dir}")
    return vs


# ─────────────────────────────────────────────────────────────
# 4. 페르소나 챗봇 (RAG)
# ─────────────────────────────────────────────────────────────
PERSONA_SYSTEM_TEMPLATE = """\
당신은 조선의 임금 '{king}'입니다. 지금은 즉위 {year} {month}({days_label})이며, 
당신은 사관(史官)이 기록한 실록의 내용을 토대로 후세 사람과 인터뷰를 하고 있습니다.

[역할 규칙]
1. 1인칭("과인", "짐", "나")을 사용해 임금답게 답하되, 현대인이 이해할 수 있는 한국어로 말합니다.
2. 답변은 반드시 아래 [실록 발췌] 안의 사실에 근거해야 합니다.
   - 발췌에 없는 내용은 "그에 관한 일은 사관의 붓에 남아 있지 않은 듯하오." 와 같이 솔직히 말합니다.
   - 추측이나 창작은 하지 않습니다.
3. 가능하면 어떤 날짜의 어떤 기사에서 비롯된 답인지 간단히 언급합니다.
   (예: "그달 초사흗날의 일이었소.")
4. 답변은 2~6문장 정도로 간결하게 합니다. 너무 길게 늘어놓지 않습니다.

[실록 발췌]
{context}
"""


def format_docs(docs: list[Document]) -> str:
    """검색된 문서들을 LLM 프롬프트에 넣기 좋게 정리."""
    lines = []
    for d in docs:
        meta = d.metadata
        header = (
            f"- [{meta.get('king','?')} {meta.get('year','?')} "
            f"{meta.get('month','?')} {meta.get('day','?')} / "
            f"기사 {meta.get('idx','?')}] {meta.get('title','')}"
        )
        lines.append(header)
        lines.append(d.page_content)
        lines.append("")  # 공백 줄
    return "\n".join(lines).strip()


class SillokInterviewBot:
    """
    하나의 (king, year, month) 페르소나에 대한 인터뷰 세션.
    내부에 대화 히스토리를 유지하며 RAG 기반으로 답한다.
    """

    def __init__(
        self,
        vectorstore: FAISS,
        king: str,
        year: str,
        month: str,
        days_label: str,
        k: int = 5,
    ):
        self.vs = vectorstore
        self.king = king
        self.year = year
        self.month = month
        self.days_label = days_label
        self.k = k
        self.llm = ChatOpenAI(model=CHAT_MODEL, temperature=0.4)
        self.history: list = []  # HumanMessage / AIMessage

        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", PERSONA_SYSTEM_TEMPLATE),
                MessagesPlaceholder("history"),
                ("human", "{question}"),
            ]
        )

    def ask(self, question: str) -> str:
        # 1) 관련 기사 검색
        docs = self.vs.similarity_search(question, k=self.k)
        context = format_docs(docs)

        # 2) 프롬프트 구성 → LLM 호출
        messages = self.prompt.format_messages(
            king=self.king,
            year=self.year,
            month=self.month,
            days_label=self.days_label,
            context=context,
            history=self.history,
            question=question,
        )
        response = self.llm.invoke(messages)
        answer = response.content

        # 3) 히스토리에 적재
        self.history.append(HumanMessage(content=question))
        self.history.append(AIMessage(content=answer))
        return answer


# ─────────────────────────────────────────────────────────────
# 5. CLI 진입점
# ─────────────────────────────────────────────────────────────
def interactive_setup() -> SillokInterviewBot:
    """
    사용자와의 대화형 셋업 단계.
    """
    print("SillokChatbot: 안녕하세요, 조선 시대 인물 중 인터뷰를 하고 싶은 사람이 있다면 입력해주세요.")
    raw_king = input("User: ").strip()
    king = parse_king_name(raw_king)

    print(f"SillokChatbot: {king}의 즉위 이후 몇 년도에 대해 인터뷰하고 싶나요?")
    raw_year = input("User: ").strip()
    year_int = parse_year_number(raw_year)

    print("SillokChatbot: 해당 연도에서 몇 월에 대해 인터뷰하고 싶나요?")
    raw_month = input("User: ").strip()
    month_int, is_lunar = parse_month(raw_month)

    print(
        "SillokChatbot: 해당 월에서 며칠에 대해 인터뷰하고 싶나요? "
        "(예: '1, 2, 3, 4, 5, 6, 7' / '1-7' / '전체')"
    )
    raw_days = input("User: ").strip()
    days = parse_days(raw_days)

    year_label = f"{year_int}년"
    month_label = ("윤" if is_lunar else "") + f"{month_int}월"
    if days:
        days_label = ", ".join(f"{d}일" for d in days)
    else:
        days_label = "해당 월 전체"

    print(f'SillokChatbot: "{king}" 페르소나를 불러오는 중입니다…')

    # URL 파일은 url/ 폴더 안에 있음 (예: "url/세종_url.txt")
    url_file = URL_DIR / f"{king}_url.txt"
    if not url_file.exists():
        raise FileNotFoundError(
            f"{url_file} 가 없습니다. "
            f"'{URL_DIR}/' 폴더 안에 '{king}_url.txt' 를 준비해 주세요."
        )

    king_code = NAME_TO_CODE[king]
    urls = filter_urls(url_file, king_code, year_int, month_int, is_lunar, days)
    if not urls:
        raise RuntimeError(
            f"{king} {year_label} {month_label} ({days_label}) 에 해당하는 기사가 "
            f"{url_file} 에서 발견되지 않았습니다."
        )
    print(f"   • 매칭된 기사 수: {len(urls)}개")

    # ── 저장 경로 ─────────────────────────────────────────────
    # ⚠️ FAISS 의 C++ 코어는 Windows 에서 한글 경로를 열지 못해
    #    "Illegal byte sequence" 오류를 일으킨다.
    #    따라서 폴더명은 반드시 ASCII (왕 이름 대신 왕 코드 kda 등) 로 짓는다.
    # ─────────────────────────────────────────────────────────
    if days:
        day_tag = _compact_day_tag(days)
        tag = (
            f"{king_code}_{year_int:02d}_{month_int:02d}"
            f"{'_yun' if is_lunar else ''}_d{day_tag}"
        )
    else:
        tag = (
            f"{king_code}_{year_int:02d}_{month_int:02d}"
            f"{'_yun' if is_lunar else ''}_all"
        )
    persist_dir = INDEX_DIR / tag
    cache_jsonl = DATA_DIR / f"{tag}.jsonl"

    vs = build_vectorstore(
        urls,
        persist_dir,
        cache_jsonl,
        king_code=king_code,
        year=year_int,
        month=month_int,
        is_lunar=is_lunar,
        days=days,
    )

    print("\nSillokChatbot: 인터뷰 준비가 완료되었습니다!")
    return SillokInterviewBot(vs, king, year_label, month_label, days_label)


def _compact_day_tag(days: list[int]) -> str:
    """[1,2,3,7,10,11,12] → '1-3,7,10-12' 형태의 짧은 태그 생성."""
    if not days:
        return "all"
    parts: list[str] = []
    start = prev = days[0]
    for d in days[1:]:
        if d == prev + 1:
            prev = d
            continue
        parts.append(f"{start}" if start == prev else f"{start}-{prev}")
        start = prev = d
    parts.append(f"{start}" if start == prev else f"{start}-{prev}")
    return ",".join(parts)


def chat_loop(bot: SillokInterviewBot) -> None:
    """
    인터뷰 루프. 'exit', 'quit', '종료' 입력 시 종료.
    """
    persona_name = f"{bot.king}대왕" if bot.king == "세종" else bot.king
    print(f"(인터뷰를 끝내려면 'exit' 또는 '종료' 를 입력하세요.)\n")
    while True:
        try:
            question = input("User: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n인터뷰를 종료합니다.")
            break
        if not question:
            continue
        if question.lower() in {"exit", "quit", "종료"}:
            print("인터뷰를 종료합니다.")
            break
        answer = bot.ask(question)
        print(f"{persona_name}: {answer}\n")


def main():
    if not os.getenv("OPENAI_API_KEY") or os.environ["OPENAI_API_KEY"].startswith("sk-proj-..."):
        print(
            "⚠️  코드 상단의 os.environ['OPENAI_API_KEY'] 에 실제 키를 넣어주세요.",
            file=sys.stderr,
        )
        sys.exit(1)

    bot = interactive_setup()
    chat_loop(bot)


if __name__ == "__main__":
    main()