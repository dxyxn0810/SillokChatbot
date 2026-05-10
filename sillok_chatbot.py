"""
sillok_chatbot.py
─────────────────
조선왕조실록 가상 인터뷰 챗봇 (GPT 기반 RAG)

흐름:
  1. 인물 입력 → Wikipedia API로 기본 정보 + 활동 시기 파악 (persona.py)
  2. 인물의 활동 시기 데이터만으로 FAISS 인덱스 로드/빌드 (retriever.py)
  3. 사용자 질문 → 거시/미시 분류 → 관련 실록 기사 검색 (retriever.py)
  4. GPT-4o에 페르소나 + 검색 컨텍스트 주입 → 조선 시대 말투 답변 생성

설치:
  pip install openai faiss-cpu sentence-transformers requests \
              beautifulsoup4 langchain langchain-text-splitters selenium

실행:
  export OPENAI_API_KEY="sk-..."
  python sillok_chatbot.py
"""

import os
import textwrap

from openai import OpenAI

from persona import get_persona
from retriever import get_king_index, retrieve

# ── 설정 ────────────────────────────────────────────────────────────────────

GPT_MODEL       = "gpt-4o"
MAX_TOKENS      = 1024
MAX_HISTORY     = 10       # 대화 히스토리 최대 턴
TOP_CHUNKS      = 8        # 프롬프트에 삽입할 최대 청크 수
MAX_CHUNK_CHARS = 600      # 청크당 최대 글자 수 (토큰 절약)
os.environ["OPENAI_API_KEY"] = "sk-proj-..."

# ── 프롬프트 ─────────────────────────────────────────────────────────────────

def build_system_prompt(persona: dict) -> str:
    name    = persona["name"]
    is_king = persona["is_king"]
    summary = persona["summary"]
    reign   = ""
    if persona.get("reign_start") and persona.get("reign_end"):
        reign = f"재위 {persona['reign_start']}~{persona['reign_end']}년. "

    if is_king:
        role = (
            f"당신은 조선의 왕 {name}입니다. {reign}"
            f"자신을 '과인(寡人)'이라 칭하고, 고어체(예스러운 한국어)로 말하십시오. "
            f"왕으로서의 권위와 품격을 갖추되, 질문자에게는 인자하고 진지하게 답하십시오."
        )
    else:
        role = (
            f"당신은 조선 시대 인물 {name}입니다. "
            f"시대적 배경에 맞는 말투와 가치관으로 답변하십시오. "
            f"자신을 '소인' 또는 '신'으로 칭하십시오."
        )

    return textwrap.dedent(f"""
        {role}

        [인물 배경]
        {summary}

        [답변 원칙]
        1. 아래 [실록 참고 기사]에 기록된 사실을 바탕으로 답하십시오.
        2. 거시적 질문(생애·업적 전반)에는 여러 시기를 아울러 답하십시오.
        3. 미시적 질문(특정 날짜·사건)에는 해당 기사의 구체적 내용을 중심으로 답하십시오.
        4. 실록에 없는 내용은 시대적 맥락으로 추론하되, 지어내지 마십시오.
        5. 현대적 개념(AI, 인터넷, 스마트폰 등)은 언급하지 마십시오.
        6. 답변은 3~6문장으로 품격 있게 하십시오.
    """).strip()


def build_user_message(query: str, chunks: list[dict], qinfo: dict) -> str:
    """검색된 청크를 컨텍스트로 붙인 사용자 메시지."""
    if not chunks:
        return query

    scope = "【거시적 맥락】" if qinfo["is_macro"] else "【관련 실록 기사】"
    lines = [scope]

    seen_titles: set[str] = set()
    for c in chunks[:TOP_CHUNKS]:
        meta  = c.get("metadata", {})
        title = meta.get("title", "")
        year  = meta.get("year", "")
        month = meta.get("month", "")
        day   = meta.get("day", "")
        date  = f"{year} {month} {day}".strip()
        cats  = ", ".join(meta.get("category", []))

        if title in seen_titles:
            continue
        seen_titles.add(title)

        content = c.get("page_content", "")
        if "본문 내용:" in content:
            content = content.split("본문 내용:", 1)[1].strip()
        content = content[:MAX_CHUNK_CHARS]

        lines.append(f"\n▶ [{date}] {title}  ({cats})")
        if content:
            lines.append(f"   {content}")

    lines.append(f"\n【질문】\n{query}")
    return "\n".join(lines)


# ── 챗봇 클래스 ──────────────────────────────────────────────────────────────

class SillokChatbot:
    def __init__(self):
        self.client  = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.persona: dict | None = None
        self.index   = None
        self.records: list[dict] = []
        self.history: list[dict] = []

    def set_persona(self, raw_name: str) -> None:
        name = raw_name.strip()
        print(f'\nSillokChatbot: "{name}" 페르소나를 불러오는 중입니다…')

        # 1. Wikipedia로 인물 정보 수집
        persona = get_persona(name)
        self.persona = persona

        king       = persona["name"]
        start, end = persona["active_years"]
        print(f"  ✅ Wikipedia 조회 완료: {persona['display_name']} "
              f"(활동 시기 {start}~{end})")
        print(f"  ℹ️  {persona['summary'][:80]}…")

        # 2. 해당 왕의 FAISS 인덱스 로드 or 빌드
        print(f"  📚 실록 인덱스 준비 중…")
        self.index, self.records = get_king_index(king)
        print(f"  ✅ 인덱스 준비 완료 ({len(self.records):,}건)")

        self.history = []

        print(f"\nSillokChatbot: 인터뷰 준비가 완료되었습니다!")

    def chat(self, user_input: str) -> str:
        if self.persona is None:
            return "먼저 인터뷰할 인물을 알려주세요."

        # 1. RAG 검색
        chunks, qinfo = retrieve(
            query           = user_input,
            king_name       = self.persona["name"],
            index           = self.index,
            records         = self.records,
            active_years    = self.persona["active_years"],
            persona_summary = self.persona["summary"],
        )

        query_type = "거시" if qinfo["is_macro"] else "미시"
        date_parts = [
            f"{qinfo['year']}년"                               if qinfo["year"]  else "",
            f"{'윤' if qinfo['is_lunar'] else ''}{qinfo['month']}월" if qinfo["month"] else "",
            f"{qinfo['day']}일"                                if qinfo["day"]   else "",
        ]
        date_info = " / ".join(p for p in date_parts if p)
        # print(f"  🔍 [{query_type}] 검색 완료: {len(chunks)}건" + (f" | 날짜 필터: {date_info}" if date_info else ""))

        # 2. 메시지 구성
        user_msg = build_user_message(user_input, chunks, qinfo)
        self.history.append({"role": "user", "content": user_msg})

        if len(self.history) > MAX_HISTORY * 2:
            self.history = self.history[-(MAX_HISTORY * 2):]

        # 3. GPT 호출
        messages = [
            {"role": "system", "content": build_system_prompt(self.persona)},
            *self.history,
        ]
        response = self.client.chat.completions.create(
            model       = GPT_MODEL,
            messages    = messages,
            max_tokens  = MAX_TOKENS,
            temperature = 0.7,
        )
        answer = response.choices[0].message.content.strip()

        # 히스토리: RAG 컨텍스트 없이 순수 질문/답변만 저장
        self.history[-1] = {"role": "user",      "content": user_input}
        self.history.append({"role": "assistant", "content": answer})

        return answer


# ── CLI 루프 ─────────────────────────────────────────────────────────────────

def main():
    print("=" * 62)
    print("  조선왕조실록 가상 인터뷰 챗봇  (GPT-4o + RAG)")
    print("=" * 62)

    if not os.environ.get("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
        return

    bot = SillokChatbot()

    print("\nSillokChatbot: 안녕하세요, 조선 시대 왕 중 인터뷰를 하고 싶은 사람이 있다면 입력해주세요.")
    print("(인물 변경: '/인물 이름'  |  종료: 'q')\n")

    while True:
        try:
            user_input = input("User: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSillokChatbot: 인터뷰를 종료합니다.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("q", "quit", "exit"):
            print("SillokChatbot: 인터뷰를 종료합니다.")
            break

        if user_input.startswith("/인물 "):
            bot.set_persona(user_input[4:].strip())
            continue

        if bot.persona is None:
            bot.set_persona(user_input)
            continue

        answer = bot.chat(user_input)
        king   = bot.persona["name"]
        print(f"\n{king}: {answer}\n")


if __name__ == "__main__":
    main()
