from __future__ import annotations

import argparse
import atexit
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import streamlit as st

from web_nav_agent.browser import ThreadedBrowserSession
from web_nav_agent.chatbot import finalize_chat_turn, run_chat_turn
from web_nav_agent.llm_adapter import create_adapter
from web_nav_agent.schemas import ChatState


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--headless", action="store_true", help="Run Playwright in headless mode")
    parser.add_argument("--step-delay-ms", type=int, default=1200, help="Delay after each nav step")
    parser.add_argument("--nav-max-steps", type=int, default=12, help="Maximum steps for the nav agent")
    parser.add_argument("--chat-max-turns", type=int, default=10, help="Maximum turns for chat mode")
    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args(sys.argv[1:])


def safe_stop_session(session: ThreadedBrowserSession | None) -> None:
    if session is None:
        return
    try:
        session.stop()
    except Exception:
        pass


def register_cleanup(session: ThreadedBrowserSession) -> None:
    atexit.register(safe_stop_session, session)


def reset_runtime_state() -> None:
    safe_stop_session(st.session_state.get("browser_session"))
    for key in [
        "browser_session",
        "browser_headless",
        "chat_state",
        "chat_llm",
        "nav_llm",
        "transcript",
        "chat_closed",
    ]:
        st.session_state.pop(key, None)


def ensure_runtime(headless: bool, chat_max_turns: int) -> None:
    if "browser_session" not in st.session_state:
        session = ThreadedBrowserSession(headless=headless)
        session.start()
        register_cleanup(session)
        st.session_state.browser_session = session
        st.session_state.browser_headless = headless
        st.session_state.chat_state = ChatState(max_turns=chat_max_turns)
        st.session_state.chat_llm = create_adapter(model_name_key="CHAT_MODEL_NAME")
        st.session_state.nav_llm = create_adapter(model_name_key="NAV_MODEL_NAME")
        st.session_state.transcript = []
        st.session_state.chat_closed = False
        return

    st.session_state.chat_state.max_turns = chat_max_turns


def main() -> None:
    args = parse_args()

    st.set_page_config(page_title="Web Navigation Agent", page_icon=":globe_with_meridians:", layout="centered")
    st.title("Web Navigation Agent")
    st.write("Chat with the router directly. It can answer normally or trigger the browser agent when needed.")

    with st.sidebar:
        st.header("Settings")
        headless = st.checkbox("Headless browser", value=args.headless)
        step_delay_ms = st.number_input("Step delay (ms)", min_value=0, value=args.step_delay_ms, step=100)
        nav_max_steps = st.number_input("Nav max steps", min_value=1, value=args.nav_max_steps, step=1)
        chat_max_turns = st.number_input("Chat max turns", min_value=1, value=args.chat_max_turns, step=1)
        if st.button("Reset session", use_container_width=True):
            reset_runtime_state()

    try:
        ensure_runtime(headless=headless, chat_max_turns=int(chat_max_turns))
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    if st.session_state.browser_headless != headless:
        st.sidebar.info("Changing headless mode takes effect after resetting the session.")

    for message in st.session_state.transcript:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if st.session_state.chat_closed:
        st.info("The chat session has ended. Use Reset session to start a new one.")
        return

    user_message = st.chat_input("Ask the assistant to browse or answer")
    if not user_message:
        return

    st.session_state.transcript.append({"role": "user", "content": user_message})
    with st.chat_message("user"):
        st.markdown(user_message)

    session = st.session_state.browser_session
    try:
        turn = run_chat_turn(
            state=st.session_state.chat_state,
            user_message=user_message,
            session=session,
            llm=st.session_state.chat_llm,
        )
    except Exception as exc:
        st.session_state.transcript.append({"role": "assistant", "content": str(exc)})
        with st.chat_message("assistant"):
            st.markdown(str(exc))
        st.rerun()
        return

    st.session_state.transcript.append({"role": "assistant", "content": turn.assistant_message})
    with st.chat_message("assistant"):
        st.markdown(turn.assistant_message)

    if turn.decision.type == "RUN_NAV":
        with st.spinner("Running browser agent..."):
            result = finalize_chat_turn(
                state=st.session_state.chat_state,
                turn=turn,
                session=session,
                nav_llm=st.session_state.nav_llm,
                nav_max_steps=int(nav_max_steps),
                step_delay_ms=int(step_delay_ms),
                include_assistant_message=False,
            )
    else:
        result = finalize_chat_turn(
            state=st.session_state.chat_state,
            turn=turn,
            session=session,
            nav_llm=st.session_state.nav_llm,
            nav_max_steps=int(nav_max_steps),
            step_delay_ms=int(step_delay_ms),
            include_assistant_message=False,
        )

    for message in result.messages:
        st.session_state.transcript.append({"role": "assistant", "content": message})
        with st.chat_message("assistant"):
            st.markdown(message)
    if result.should_exit:
        st.session_state.chat_closed = True
    st.rerun()


if __name__ == "__main__":
    main()
