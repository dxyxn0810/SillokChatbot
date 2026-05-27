from __future__ import annotations

import argparse
import json
import os
import threading
import time
import uuid
from http import HTTPStatus
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from chatbot import DanjongBot, DATA_LEVELS, DEFAULT_INDEX_DIR


HTML_PAGE = """<!doctype html>
<html lang="ko">
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>단종 가상 인터뷰</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Gowun+Batang:wght@400;700&family=Nanum+Myeongjo:wght@700&display=swap" rel="stylesheet">
    <style>
        :root {
            --ink: #231a14;
            --paper: #f6efe2;
            --paper-deep: #ecdfc8;
            --accent: #85603b;
            --accent-strong: #5f3f25;
            --mist: #d2be9f;
            --user: #fff6e8;
            --bot: #f0e4cf;
            --error: #7e1d1d;
            --ok: #294f2a;
        }

        * { box-sizing: border-box; }

        body {
            margin: 0;
            min-height: 100vh;
            color: var(--ink);
            font-family: "Gowun Batang", serif;
            background:
                radial-gradient(1200px 600px at 8% 2%, rgba(255,255,255,0.6), transparent 45%),
                radial-gradient(700px 500px at 92% 12%, rgba(133,96,59,0.2), transparent 55%),
                linear-gradient(150deg, #fdf8ee 0%, #f7eddb 45%, #f2e4cb 100%);
            display: flex;
            justify-content: center;
            padding: 20px;
        }

        .frame {
            width: min(980px, 100%);
            min-height: calc(100vh - 40px);
            display: grid;
            grid-template-rows: auto 1fr auto;
            border: 1px solid rgba(63, 43, 27, 0.22);
            border-radius: 18px;
            overflow: hidden;
            background: linear-gradient(180deg, rgba(255,255,255,0.58), rgba(255,255,255,0.35));
            box-shadow: 0 24px 60px rgba(48, 29, 13, 0.18);
            backdrop-filter: blur(2px);
        }

        .topbar {
            padding: 18px 20px 14px;
            border-bottom: 1px solid rgba(63, 43, 27, 0.12);
            background:
                linear-gradient(180deg, rgba(255,255,255,0.76), rgba(255,255,255,0.34));
        }

        .title {
            margin: 0;
            font-family: "Nanum Myeongjo", serif;
            letter-spacing: 0.06em;
            font-size: clamp(1.1rem, 2.6vw, 1.7rem);
        }

        .subtitle {
            margin-top: 7px;
            opacity: 0.82;
            font-size: 0.93rem;
            line-height: 1.45;
        }

        .chat {
            overflow-y: auto;
            padding: 20px;
            display: flex;
            flex-direction: column;
            gap: 12px;
            scroll-behavior: smooth;
            background:
                repeating-linear-gradient(
                    to bottom,
                    rgba(133,96,59,0.08) 0,
                    rgba(133,96,59,0.08) 1px,
                    transparent 1px,
                    transparent 32px
                );
        }

        .msg {
            max-width: min(86%, 760px);
            padding: 12px 14px;
            border-radius: 13px;
            line-height: 1.55;
            white-space: pre-wrap;
            word-break: keep-all;
            animation: rise 0.28s ease-out;
            border: 1px solid rgba(63, 43, 27, 0.16);
            box-shadow: 0 6px 20px rgba(44, 28, 12, 0.08);
        }

        .msg.user {
            align-self: flex-end;
            background: var(--user);
            border-top-right-radius: 4px;
        }

        .msg.bot {
            align-self: flex-start;
            background: var(--bot);
            border-top-left-radius: 4px;
        }

        .meta {
            display: block;
            margin-bottom: 5px;
            font-size: 0.76rem;
            opacity: 0.72;
            letter-spacing: 0.02em;
        }

        .composer {
            border-top: 1px solid rgba(63, 43, 27, 0.12);
            padding: 14px;
            display: grid;
            grid-template-columns: 1fr auto auto;
            gap: 8px;
            background:
                linear-gradient(180deg, rgba(247,237,219,0.94), rgba(240,226,202,0.94));
        }

        .input {
            width: 100%;
            border: 1px solid rgba(63,43,27,0.32);
            border-radius: 12px;
            padding: 12px 13px;
            font-size: 1rem;
            font-family: "Gowun Batang", serif;
            outline: none;
            background: rgba(255,255,255,0.85);
        }

        .input:focus {
            border-color: var(--accent);
            box-shadow: 0 0 0 3px rgba(133,96,59,0.18);
        }

        .btn {
            border: 0;
            border-radius: 12px;
            padding: 0 16px;
            font-size: 0.95rem;
            font-family: "Gowun Batang", serif;
            cursor: pointer;
            transition: transform 0.08s ease, filter 0.2s ease;
        }

        .btn:active { transform: translateY(1px); }

        .btn.send {
            background: linear-gradient(180deg, var(--accent), var(--accent-strong));
            color: #fff;
        }

        .btn.reset {
            background: linear-gradient(180deg, #4f433b, #352b25);
            color: #fff;
        }

        .btn:hover { filter: brightness(1.06); }

        .status {
            margin: 2px 0 0 2px;
            font-size: 0.83rem;
            min-height: 1.2em;
            opacity: 0.9;
        }

        .status.ok { color: var(--ok); }
        .status.err { color: var(--error); }

        .thinking {
            display: inline-flex;
            gap: 4px;
            align-items: center;
            font-size: 0.85rem;
            opacity: 0.72;
        }

        .dot {
            width: 6px;
            height: 6px;
            border-radius: 50%;
            background: #8d6d4f;
            animation: pulse 1.2s infinite;
        }

        .dot:nth-child(2) { animation-delay: 0.18s; }
        .dot:nth-child(3) { animation-delay: 0.36s; }

        @keyframes pulse {
            0%, 80%, 100% { transform: scale(0.85); opacity: 0.4; }
            40% { transform: scale(1.2); opacity: 1; }
        }

        @keyframes rise {
            from { opacity: 0; transform: translateY(6px); }
            to { opacity: 1; transform: translateY(0); }
        }

        @media (max-width: 700px) {
            body { padding: 8px; }
            .frame {
                min-height: calc(100vh - 16px);
                border-radius: 14px;
            }
            .chat { padding: 12px; }
            .msg { max-width: 92%; }
            .composer {
                grid-template-columns: 1fr;
            }
            .btn {
                height: 40px;
            }
        }
    </style>
</head>
<body>
    <main class="frame">
        <header class="topbar">
            <h1 class="title">단종 가상 인터뷰</h1>
            <div class="subtitle">실록 기반으로 답하되, 말투는 어린 임금 단종의 1인칭 회고임.</div>
        </header>

        <section id="chat" class="chat" aria-live="polite"></section>

        <footer class="composer">
            <input id="input" class="input" type="text" maxlength="2000" placeholder="질문을 입력해 보거라. (Enter 전송)" />
            <button id="sendBtn" class="btn send" type="button">질문하기</button>
            <button id="resetBtn" class="btn reset" type="button">새 대화</button>
            <div id="status" class="status"></div>
        </footer>
    </main>

    <script>
        const chat = document.getElementById("chat");
        const input = document.getElementById("input");
        const sendBtn = document.getElementById("sendBtn");
        const resetBtn = document.getElementById("resetBtn");
        const statusEl = document.getElementById("status");

        function addMessage(role, text) {
            const wrap = document.createElement("article");
            wrap.className = "msg " + role;

            const meta = document.createElement("span");
            meta.className = "meta";
            meta.textContent = role === "user" ? "사용자" : "단종";

            const body = document.createElement("div");
            body.textContent = text;

            wrap.appendChild(meta);
            wrap.appendChild(body);
            chat.appendChild(wrap);
            chat.scrollTop = chat.scrollHeight;
            return wrap;
        }

        function setStatus(text, kind = "ok") {
            statusEl.textContent = text;
            statusEl.className = "status " + kind;
        }

        function setBusy(busy) {
            input.disabled = busy;
            sendBtn.disabled = busy;
            resetBtn.disabled = busy;
            if (!busy) {
                input.focus();
            }
        }

        async function sendMessage() {
            const message = input.value.trim();
            if (!message) return;

            addMessage("user", message);
            input.value = "";
            setBusy(true);
            setStatus("단종이 답을 정리하고 있다...", "ok");

            const pending = document.createElement("article");
            pending.className = "msg bot";
            pending.innerHTML =
                '<span class="meta">단종</span>' +
                '<span class="thinking"><span class="dot"></span><span class="dot"></span><span class="dot"></span></span>';
            chat.appendChild(pending);
            chat.scrollTop = chat.scrollHeight;

            try {
                const res = await fetch("/api/chat", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ message })
                });

                const data = await res.json().catch(() => ({}));
                pending.remove();

                if (!res.ok) {
                    const errText = data.error || "요청 처리 중 오류가 발생했다.";
                    addMessage("bot", "미안하구나. " + errText);
                    setStatus("오류가 발생했다.", "err");
                    return;
                }

                addMessage("bot", data.answer || "그 일은 잘 기억나지 않는구나.");
                setStatus("답변 완료", "ok");
            } catch (err) {
                pending.remove();
                addMessage("bot", "바깥 세상과의 연결이 잠시 흐려졌구나. 잠시 뒤 다시 물어보거라.");
                setStatus("네트워크 오류", "err");
            } finally {
                setBusy(false);
            }
        }

        async function resetChat() {
            setBusy(true);
            try {
                const res = await fetch("/api/reset", { method: "POST" });
                if (!res.ok) {
                    throw new Error("reset failed");
                }
                chat.innerHTML = "";
                addMessage("bot", "다시 시작해 보자꾸나. 무엇이 궁금하냐?");
                setStatus("새 대화가 시작되었다.", "ok");
            } catch (err) {
                setStatus("초기화에 실패했다.", "err");
            } finally {
                setBusy(false);
            }
        }

        input.addEventListener("keydown", (e) => {
            if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });

        sendBtn.addEventListener("click", sendMessage);
        resetBtn.addEventListener("click", resetChat);

        addMessage("bot", "반갑구나. 나는 단종이니라. 무엇을 묻고 싶으냐?");
        input.focus();
    </script>
</body>
</html>
"""


class BotManager:
        """세션별 DanjongBot 인스턴스를 보관한다."""

        def __init__(self, *, faiss_dir: str, k: int, verbose: bool,
                                 data_level: str, use_cot: bool, history_turns: int):
                self._params = {
                        "faiss_dir": faiss_dir,
                        "k": k,
                        "verbose": verbose,
                        "data_level": data_level,
                        "use_cot": use_cot,
                        "history_turns": history_turns,
                }
                self._bots = {}
                self._lock = threading.Lock()

        def _new_bot(self) -> DanjongBot:
                return DanjongBot(**self._params)

        def get_or_create(self, session_id: str) -> DanjongBot:
                with self._lock:
                        bot = self._bots.get(session_id)
                        if bot is None:
                                bot = self._new_bot()
                                self._bots[session_id] = bot
                        return bot

        def reset(self, session_id: str) -> None:
                with self._lock:
                        self._bots[session_id] = self._new_bot()


def make_handler(bot_manager: BotManager, verbose: bool):
        class ChatHandler(BaseHTTPRequestHandler):
                server_version = "DanjongWeb/1.0"

                def log_message(self, fmt: str, *args):
                        if verbose:
                                super().log_message(fmt, *args)

                def _read_json(self):
                        raw_len = self.headers.get("Content-Length")
                        if not raw_len:
                                return {}
                        length = int(raw_len)
                        payload = self.rfile.read(length)
                        if not payload:
                                return {}
                        return json.loads(payload.decode("utf-8"))

                def _session_id(self) -> tuple[str, bool]:
                        cookie_raw = self.headers.get("Cookie") or ""
                        cookie = SimpleCookie()
                        cookie.load(cookie_raw)
                        morsel = cookie.get("sid")
                        if morsel and morsel.value:
                                return morsel.value, False
                        return uuid.uuid4().hex, True

                def _send_json(self, status: int, payload: dict, *, set_sid: str | None = None):
                        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                        self.send_response(status)
                        self.send_header("Content-Type", "application/json; charset=utf-8")
                        self.send_header("Content-Length", str(len(body)))
                        if set_sid:
                                self.send_header("Set-Cookie", f"sid={set_sid}; Path=/; HttpOnly; SameSite=Lax")
                        self.end_headers()
                        self.wfile.write(body)

                def do_GET(self):
                        if self.path not in ("/", "/index.html"):
                                self.send_error(HTTPStatus.NOT_FOUND)
                                return
                        sid, is_new = self._session_id()
                        bot_manager.get_or_create(sid)
                        body = HTML_PAGE.encode("utf-8")

                        self.send_response(HTTPStatus.OK)
                        self.send_header("Content-Type", "text/html; charset=utf-8")
                        self.send_header("Content-Length", str(len(body)))
                        if is_new:
                                self.send_header("Set-Cookie", f"sid={sid}; Path=/; HttpOnly; SameSite=Lax")
                        self.end_headers()
                        self.wfile.write(body)

                def do_POST(self):
                        if self.path not in ("/api/chat", "/api/reset"):
                                self.send_error(HTTPStatus.NOT_FOUND)
                                return

                        sid, is_new = self._session_id()

                        try:
                                if self.path == "/api/reset":
                                        bot_manager.reset(sid)
                                        self._send_json(HTTPStatus.OK, {"ok": True}, set_sid=sid if is_new else None)
                                        return

                                data = self._read_json()
                                message = (data.get("message") or "").strip()
                                if not message:
                                        self._send_json(
                                                HTTPStatus.BAD_REQUEST,
                                                {"error": "message 필드가 비어 있습니다."},
                                                set_sid=sid if is_new else None,
                                        )
                                        return

                                bot = bot_manager.get_or_create(sid)
                                start = time.perf_counter()
                                answer = bot.ask(message)
                                elapsed_ms = int((time.perf_counter() - start) * 1000)

                                self._send_json(
                                        HTTPStatus.OK,
                                        {"answer": answer, "elapsed_ms": elapsed_ms},
                                        set_sid=sid if is_new else None,
                                )
                        except json.JSONDecodeError:
                                self._send_json(
                                        HTTPStatus.BAD_REQUEST,
                                        {"error": "요청 본문이 JSON 형식이 아닙니다."},
                                        set_sid=sid if is_new else None,
                                )
                        except Exception as exc:
                                self._send_json(
                                        HTTPStatus.INTERNAL_SERVER_ERROR,
                                        {"error": str(exc)},
                                        set_sid=sid if is_new else None,
                                )

        return ChatHandler


def main():
        parser = argparse.ArgumentParser(
                description="단종 가상 인터뷰 웹 챗봇",
                formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        parser.add_argument("--host", default="127.0.0.1", help="서버 바인딩 주소")
        parser.add_argument("--port", type=int, default=8000, help="서버 포트")
        parser.add_argument(
                "--data",
                choices=DATA_LEVELS,
                default="wikipedia",
                help=(
                        "사용할 정보 소스 범위(누적적). "
                        "prompt=페르소나 프롬프트만, "
                        "article=prompt+실록 기사, "
                        "summary=prompt+기사+요약, "
                        "wikipedia=prompt+기사+요약+위키백과"
                ),
        )
        parser.add_argument(
                "--CoT",
                dest="cot",
                choices=["on", "off"],
                default="on",
                help="CoT 기반 메타데이터 필터링 사용 여부. on=사용, off=순수 의미검색.",
        )
        parser.add_argument(
                "--history",
                type=int,
                default=6,
                help="답변 생성 시 참고할 직전 대화 맥락 턴 수(0이면 맥락 미사용).",
        )
        parser.add_argument(
                "--retrieval",
                type=int,
                default=10,
                help="검색해 올 문서 수(k).",
        )
        parser.add_argument("--faiss", default=str(DEFAULT_INDEX_DIR), help="FAISS 인덱스 폴더")
        parser.add_argument("--verbose", action="store_true", help="요청/검색 로그 출력")
        args = parser.parse_args()

        if not os.environ.get("OPENAI_API_KEY"):
                raise SystemExit("환경변수 OPENAI_API_KEY 가 필요합니다.")

        use_cot = args.cot == "on"
        manager = BotManager(
                faiss_dir=args.faiss,
                k=args.retrieval,
                verbose=args.verbose,
                data_level=args.data,
                use_cot=use_cot,
                history_turns=args.history,
        )

        handler_cls = make_handler(manager, verbose=args.verbose)
        server = ThreadingHTTPServer((args.host, args.port), handler_cls)

        print("=" * 62)
        print(" 단종 가상 인터뷰 웹 서버")
        print(f" 주소: http://{args.host}:{args.port}")
        print(f" 설정: data={args.data}, CoT={args.cot}, history={args.history}, retrieval={args.retrieval}")
        print(" 종료: Ctrl+C")
        print("=" * 62)

        try:
                server.serve_forever()
        except KeyboardInterrupt:
                print("\n서버를 종료합니다.")
        finally:
                server.server_close()


if __name__ == "__main__":
        main()