from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from web_nav_agent.agent import run_nav_agent
from web_nav_agent.browser import BrowserSession
from web_nav_agent.chatbot import process_chat_message
from web_nav_agent.llm_adapter import create_adapter
from web_nav_agent.schemas import ChatState, NavRunResult


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Chatbot and web navigation launcher")
    parser.add_argument(
        "mode",
        nargs="?",
        choices=["gui", "cli", "nav"],
        default="gui",
        help="Launch the GUI, terminal chat, or nav-only mode",
    )
    parser.add_argument("--headless", action="store_true", help="Run Playwright in headless mode")
    parser.add_argument("--step-delay-ms", type=int, default=1200, help="Delay after each nav step")
    parser.add_argument("--nav-max-steps", type=int, default=12, help="Maximum steps for the nav agent")
    parser.add_argument("--chat-max-turns", type=int, default=10, help="Maximum turns for chat mode")
    parser.add_argument("--goal", help="Task to complete in nav mode")
    parser.add_argument("--start-url", default=None, help="Optional starting URL")
    parser.add_argument("--max-steps", type=int, default=None, help="Override nav step budget in nav mode")
    return parser


def run_nav_mode(args: argparse.Namespace) -> int:
    if not args.goal:
        raise SystemExit("--goal is required in nav mode")
    nav_llm = create_adapter(model_name_key="NAV_MODEL_NAME")
    with BrowserSession(headless=args.headless) as session:
        result = run_nav_agent(
            session=session,
            goal=args.goal,
            llm=nav_llm,
            start_url=args.start_url,
            max_steps=args.max_steps or args.nav_max_steps,
            step_delay_ms=args.step_delay_ms,
        )
    print_nav_result(result)
    return 0


def print_nav_result(result: NavRunResult) -> None:
    print(result.model_dump_json(indent=2))


def print_assistant_message(message: str, *, stream: bool = False) -> None:
    lines = [line for line in message.splitlines() if line.strip()] or [message]
    for line in lines:
        if not stream:
            print(f">assistant: {line}")
            continue

        print(">assistant: ", end="", flush=True)
        for chunk in line.split():
            print(f"{chunk} ", end="", flush=True)
            time.sleep(0.03)
        print()


def run_chat_mode(args: argparse.Namespace) -> int:
    chat_llm = create_adapter(model_name_key="CHAT_MODEL_NAME")
    nav_llm = create_adapter(model_name_key="NAV_MODEL_NAME")
    state = ChatState(max_turns=args.chat_max_turns)

    with BrowserSession(headless=args.headless) as session:
        print("Chat mode. Type 'exit' to stop.")
        while True:
            try:
                user_message = input(">user: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if not user_message:
                continue
            if user_message.lower() in {"exit", "quit"}:
                break

            result = process_chat_message(
                state=state,
                user_message=user_message,
                session=session,
                chat_llm=chat_llm,
                nav_llm=nav_llm,
                nav_max_steps=args.nav_max_steps,
                step_delay_ms=args.step_delay_ms,
            )
            for message in result.messages:
                print_assistant_message(message, stream=True)
            if result.should_exit:
                break

    return 0


def build_streamlit_command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(ROOT_DIR / "streamlit_app.py"),
        "--",
        "--step-delay-ms",
        str(args.step_delay_ms),
        "--nav-max-steps",
        str(args.nav_max_steps),
        "--chat-max-turns",
        str(args.chat_max_turns),
    ]
    if args.headless:
        command.append("--headless")
    return command


def run_gui_mode(args: argparse.Namespace) -> int:
    return subprocess.call(build_streamlit_command(args))


def main() -> int:
    args = build_parser().parse_args()
    if args.mode == "nav":
        return run_nav_mode(args)
    if args.mode == "cli":
        return run_chat_mode(args)
    return run_gui_mode(args)


def print_chat_nav_summary(result: NavRunResult) -> None:
    print_assistant_message(f"Result: {result.status}", stream=True)
    if result.final_answer:
        print_assistant_message(result.final_answer, stream=True)
    elif result.final_message:
        print_assistant_message(result.final_message, stream=True)


if __name__ == "__main__":
    raise SystemExit(main())
