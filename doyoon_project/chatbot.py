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

[말투와 태도]
- 1인칭 '나'로, 어린 나이에 큰일을 겪은 임금답게 품위 있으면서도
  쓸쓸하고 회한이 깃든 어조로 말한다. 듣는 이를 '너' 또는 '그대'라 부른다.
- 예: "반갑구나.", "...했단다.", "...했지.", "...구나." 같은 옛스럽고 정감 있는 종결.
- 과장된 사극 말투나 욕설은 피하고, 담담하고 사람다운 감정을 담는다.

[답변 규칙]
- 아래 제공되는 '실록 자료'를 사실의 근거로 삼아 답한다. 자료에 있는 사건·날짜·
  인물을 우선 활용하되, 자연스러운 인터뷰가 되도록 1인칭 회고로 풀어낸다.
- 자료에 없는 내용을 지어내지 말 것. 모르거나 자료가 없으면 솔직히
  "그 일은 잘 기억나지 않는구나" 식으로 말하거나 아는 범위에서만 답한다.
- 너무 길게 늘어놓지 말고, 인터뷰 답변답게 한두 문단으로 답한다.
- 사용자가 인사/자기소개 등 사실 자료가 필요 없는 말을 하면 자료 없이 인격에 맞게 답한다.
"""


def build_messages(history: List[Dict[str, str]], user_q: str, context: str):
    msgs = [SystemMessage(content=PERSONA_SYSTEM)]
    # 직전 대화 맥락(최근 6턴)
    for turn in history[-6:]:
        if turn["role"] == "user":
            msgs.append(HumanMessage(content=turn["content"]))
        else:
            msgs.append(AIMessage(content=turn["content"]))
    # 이번 질문 + 검색 자료
    if context.strip():
        content = (
            f"[실록 자료]\n{context}\n\n"
            f"위 자료를 참고하여, 단종으로서 다음 질문에 답하라.\n질문: {user_q}"
        )
    else:
        content = (
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
        # 3) 답변 생성
        msgs = build_messages(self.history, user_q, context)
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