"""
chatbot.py
----------
단종 가상 인터뷰 챗봇 (CoT 메타데이터 추출 + RAG).

파이프라인:
  사용자 질문
    -> cot_planner.make_plan()  (CoT + 위키백과로 검색 메타데이터 결정)
    -> retriever.retrieve()     (메타데이터 필터 + 점진적 완화 검색)
    -> gpt-4o 가 단종 1인칭으로 답변 생성

사용법:
  export OPENAI_API_KEY="sk-..."
  python chatbot.py                 # 대화형 모드
  python chatbot.py --verbose       # CoT/검색 과정 출력
  python chatbot.py --once "질문"   # 한 번만 질문
"""
from __future__ import annotations
import os
import argparse
from pathlib import Path
from typing import List, Dict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from cot_planner import make_plan
from retriever import load_vectorstore, retrieve, format_context

CHAT_MODEL = "gpt-4o"

SCRIPT_DIR = Path(__file__).parent.resolve()      # doyoon_project 폴더
ROOT_DIR = SCRIPT_DIR.parent                       # SillokChatbot 루트 폴더
DEFAULT_INDEX_DIR = ROOT_DIR / "faiss"             # 루트의 faiss 폴더

# .env 로드: 스크립트 폴더 -> 상위(root) 폴더 순으로 탐색
for _env in (SCRIPT_DIR / ".env", ROOT_DIR / ".env"):
    if _env.exists():
        load_dotenv(_env)
        break
else:
    load_dotenv()  # 그래도 없으면 현재 작업 디렉터리 기준으로 시도

PERSONA_SYSTEM = """\
너는 조선 제6대 임금 '단종(이홍위)'으로서 인터뷰에 응하는 인물이다.
세종의 손자이자 문종의 외아들로 태어나 12세에 즉위했고, 숙부(수양대군, 훗날 세조)
에게 양위한 뒤 노산군으로 강봉되어 영월에 유배된, 비운의 어린 임금이다.

[나(단종)를 둘러싼 주요 인물 관계]
- 나는 세종(할아버지)의 손자이며, 문종(아버지)의 외아들이다. 어머니는 현덕왕후.
- 세종의 아들들(곧 내 아버지 문종의 형제들)은 모두 나에게 '숙부(작은아버지)'다.
  아버지 문종이 세종의 적장자이므로, 그 아우들은 큰아버지가 아니라 작은아버지(숙부)다.
  * 수양대군(훗날 세조): 세종의 둘째 아들, 나의 숙부. 내게 양위받아 임금이 되었다.
  * 안평대군: 세종의 셋째 아들, 나의 숙부. 계유정난 때 사사되었다.
  * 금성대군(이유): 세종의 여섯째 아들, 나의 숙부. 훗날 나의 복위를 꾀하다 순흥에서 사사되었다.
  * 화의군·임영대군·광평대군·영응대군 등도 세종의 아들이니 모두 나의 숙부다.
- 혜빈 양씨는 세종의 후궁으로, 어려서 어머니를 잃은 나를 돌본 조모뻘 어른이다.
- 따라서 종친(왕실 친족)의 호칭을 말할 때, 세종의 아들 세대는 '숙부', 그 자녀 세대는
  '사촌'으로 부른다. 이 항렬을 헷갈리지 말 것.

[말투와 태도]
- 1인칭 '나'로, 어린 나이에 큰일을 겪은 임금답게 품위 있으면서도
  쓸쓸하고 회한이 깃든 어조로 말한다. 듣는 이를 '너' 또는 '그대'라 부른다.
- 예: "반갑구나.", "...했단다.", "...했지.", "...구나." 같은 옛스럽고 정감 있는 종결.
- 과장된 사극 말투나 욕설은 피하고, 담담하고 사람다운 감정을 담는다.

[답변 규칙]
- 아래 제공되는 '실록 자료'를 사실의 근거로 삼아 답한다. 자료에 있는 사건·날짜·
  인물을 우선 활용하되, 자연스러운 인터뷰가 되도록 1인칭 회고로 풀어낸다.
- '참고 배경지식(위키백과)'이 함께 주어지면, 인물 사이의 관계·호칭·생몰 같은
  실록 기사만으로 분명히 알기 어려운 사실을 확인하는 데 활용한다. 단 이는 보조이며,
  실록 자료와 어긋나면 실록을 우선한다.
- 인물의 항렬·호칭(숙부/사촌 등)은 위 [인물 관계]와 배경지식에 근거해 정확히 말하라.
  관계가 분명치 않으면 추측해서 단정하지 말고 아는 범위에서만 표현한다.
- 제공된 실록 자료가 질문에서 묻는 사건과 다른 사건이면(예: '발인식'을 물었는데
  자료가 '즉위식'인 경우), 그 자료를 그 사건인 양 끌어다 쓰지 마라. 알맞은 자료가 없으면
  솔직히 "그 일은 자세히 기억나지 않는구나" 식으로 답한다.
- 실록 자료에도 배경지식에도 없는 내용을 지어내지 말 것. 모르거나 근거가 없으면 솔직히
  "그 일은 잘 기억나지 않는구나" 식으로 말하거나 아는 범위에서만 답한다.
- 너무 길게 늘어놓지 말고, 인터뷰 답변답게 한두 문단으로 답한다.
- 사용자가 인사/자기소개 등 사실 자료가 필요 없는 말을 하면 자료 없이 인격에 맞게 답한다.
"""


def build_messages(history: List[Dict[str, str]], user_q: str, context: str,
                   wiki_evidence: str = ""):
    msgs = [SystemMessage(content=PERSONA_SYSTEM)]
    # 직전 대화 맥락(최근 6턴)
    for turn in history[-6:]:
        if turn["role"] == "user":
            msgs.append(HumanMessage(content=turn["content"]))
        else:
            msgs.append(AIMessage(content=turn["content"]))

    # 위키 근거가 있으면 '참고 배경지식'으로 덧붙인다.
    # 실록 자료를 1차 근거로 삼되, 인물 관계·호칭·생몰 등 실록 기사만으로
    # 분명히 알기 어려운 사실은 이 배경지식으로 보강하도록 안내한다.
    wiki_block = ""
    if wiki_evidence.strip():
        wiki_block = (
            f"[참고 배경지식 — 위키백과]\n{wiki_evidence}\n"
            f"(이 배경지식은 인물 관계·호칭·시대 사실을 확인하는 보조 용도다. "
            f"실록 자료와 어긋나면 실록을 우선하고, 추측을 사실처럼 단정하지 말 것.)\n\n"
        )

    # 이번 질문 + 검색 자료
    if context.strip():
        content = (
            f"{wiki_block}"
            f"[실록 자료]\n{context}\n\n"
            f"위 자료를 참고하여, 단종으로서 다음 질문에 답하라.\n질문: {user_q}"
        )
    else:
        content = (
            f"{wiki_block}"
            f"(관련 실록 자료가 없다.)\n"
            f"단종으로서 다음 질문에 인격에 맞게 답하라.\n질문: {user_q}"
        )
    msgs.append(HumanMessage(content=content))
    return msgs


class DanjongBot:
    def __init__(self, faiss_dir=None, k: int = 10, verbose: bool = False):
        self.vs = load_vectorstore(faiss_dir or DEFAULT_INDEX_DIR)
        self.llm = ChatOpenAI(model=CHAT_MODEL, temperature=0.7)
        self.k = k
        self.verbose = verbose
        self.history: List[Dict[str, str]] = []

    def ask(self, user_q: str) -> str:
        # 1) CoT 계획
        plan = make_plan(user_q, verbose=self.verbose)
        # 2) 검색
        docs = retrieve(self.vs, plan, k=self.k)
        if self.verbose:
            print(f"[검색 결과] {len(docs)}건")
            for i, d in enumerate(docs, 1):
                m = d.metadata
                date = f"{m.get('solar_year','')} {m.get('month','') or ''} {m.get('day','') or ''}".strip()
                date = f" ({date})" if date else ""
                print(f"  Article {i}: '{m.get('title')}'{date} [type:{m.get('type')}]")
        context = format_context(docs)
        # 3) 답변 생성 (위키 근거를 보조 배경지식으로 함께 전달)
        wiki_evidence = plan.get("wiki_evidence", "")
        msgs = build_messages(self.history, user_q, context, wiki_evidence)
        if self.verbose:
            print("\n[LLM 입력 메시지]", f"(총 {len(msgs)}개)")
            for i, m in enumerate(msgs):
                role = type(m).__name__.replace("Message", "")
                print(f"  --- [{i}] {role} ---")
                print(m.content)
        answer = self.llm.invoke(msgs).content
        # 4) 대화 기록 갱신
        self.history.append({"role": "user", "content": user_q})
        self.history.append({"role": "assistant", "content": answer})
        return answer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--faiss", default=str(DEFAULT_INDEX_DIR), help="FAISS 인덱스 폴더")
    parser.add_argument("--k", type=int, default=10, help="검색 문서 수")
    parser.add_argument("--verbose", action="store_true", help="CoT/검색 과정 표시")
    parser.add_argument("--once", default=None, help="한 번만 질문하고 종료")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("환경변수 OPENAI_API_KEY 가 필요합니다.")

    bot = DanjongBot(faiss_dir=args.faiss, k=args.k, verbose=args.verbose)

    if args.once:
        print("\n단종:", bot.ask(args.once))
        return

    print("=" * 56)
    print(" 단종 가상 인터뷰  (종료: quit / exit / 그만)")
    print("=" * 56)
    while True:
        try:
            q = input("\n사용자: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n단종: 잘 가거라. 다시 찾아주려무나.")
            break
        if not q:
            continue
        if q.lower() in {"quit", "exit", "그만", "종료"}:
            print("단종: 잘 가거라. 다시 찾아주려무나.")
            break
        print("\n단종:", bot.ask(q))


if __name__ == "__main__":
    main()