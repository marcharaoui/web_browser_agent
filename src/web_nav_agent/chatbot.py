from __future__ import annotations

from dataclasses import dataclass
import re

from .agent import run_nav_agent
from .browser import BrowserSession
from .llm_adapter import BaseLLMAdapter
from .schemas import ChatMessage, ChatState, ChatTurnResult, ExitDecision, NavRunResult


CHATBOT_SYSTEM_PROMPT = """You are the chat agent for a browser automation project built by Marc Haraoui.

Decide whether to:
- ANSWER directly
- CLARIFY when the request is ambiguous
- RUN_NAV when the browser agent should handle the task
- EXIT when the user is ending the conversation

Use the current page when it helps with follow-up questions.
Only choose RUN_NAV for tasks that need a website or browser interaction.
Return JSON only.
"""

SITE_URLS = {
    "amazon": "https://www.amazon.com",
    "google": "https://www.google.com",
    "youtube": "https://www.youtube.com",
    "wikipedia": "https://www.wikipedia.org",
    "github": "https://www.github.com",
}


@dataclass
class ChatProcessingResult:
    messages: list[str]
    turn: ChatTurnResult | None = None
    nav_result: NavRunResult | None = None
    should_exit: bool = False
    error: str | None = None


def guess_start_url(message: str) -> str | None:
    match = re.search(r"https?://[^\s]+", message)
    if match:
        return match.group(0)

    lowered = message.lower()
    for name, url in SITE_URLS.items():
        if name in lowered:
            return url
    return None


def run_chat_turn(
    *,
    state: ChatState,
    user_message: str,
    session: BrowserSession,
    llm: BaseLLMAdapter,
) -> ChatTurnResult:
    page_summary = session.current_page_summary()
    if state.turns_used >= state.max_turns:
        result = ChatTurnResult(
            thought="chat budget reached",
            decision=ExitDecision(type="EXIT", message="Chat budget reached."),
            assistant_message="Chat budget reached.",
            page_summary=page_summary,
        )
    else:
        result = llm.generate_json(
            system_prompt=CHATBOT_SYSTEM_PROMPT,
            user_payload={
                "user_message": user_message,
                "history": [message.model_dump(mode="json") for message in state.history[-8:]],
                "current_page": page_summary.model_dump(mode="json"),
                "default_start_url": "https://www.google.com",
            },
            response_model=ChatTurnResult,
        )
        result.page_summary = page_summary
        if result.decision.type == "RUN_NAV" and not result.decision.reuse_current_page and not result.decision.start_url:
            result.decision.start_url = guess_start_url(user_message) or "https://www.google.com"

    state.turns_used += 1
    state.history.append(ChatMessage(role="user", content=user_message))
    state.history.append(ChatMessage(role="assistant", content=result.assistant_message))
    state.last_page_url = page_summary.url
    state.last_page_title = page_summary.title
    state.has_live_page = page_summary.has_live_page
    return result


def record_nav_result(state: ChatState, result: NavRunResult, session: BrowserSession) -> None:
    summary = session.current_page_summary()
    state.last_nav_goal = result.goal
    state.last_nav_result = result
    state.last_page_url = summary.url
    state.last_page_title = summary.title
    state.has_live_page = summary.has_live_page


def process_chat_message(
    *,
    state: ChatState,
    user_message: str,
    session: BrowserSession,
    chat_llm: BaseLLMAdapter,
    nav_llm: BaseLLMAdapter,
    nav_max_steps: int,
    step_delay_ms: int,
) -> ChatProcessingResult:
    try:
        turn = run_chat_turn(
            state=state,
            user_message=user_message,
            session=session,
            llm=chat_llm,
        )
    except Exception as exc:
        return ChatProcessingResult(messages=[str(exc)], error=str(exc))

    return finalize_chat_turn(
        state=state,
        turn=turn,
        session=session,
        nav_llm=nav_llm,
        nav_max_steps=nav_max_steps,
        step_delay_ms=step_delay_ms,
    )


def finalize_chat_turn(
    *,
    state: ChatState,
    turn: ChatTurnResult,
    session: BrowserSession,
    nav_llm: BaseLLMAdapter,
    nav_max_steps: int,
    step_delay_ms: int,
    include_assistant_message: bool = True,
) -> ChatProcessingResult:
    messages = [turn.assistant_message] if include_assistant_message else []
    should_exit = turn.decision.type == "EXIT"
    nav_result: NavRunResult | None = None

    if turn.decision.type == "RUN_NAV":
        decision = turn.decision
        start_url = None if decision.reuse_current_page else decision.start_url
        try:
            nav_result = run_nav_agent(
                session=session,
                goal=decision.goal,
                llm=nav_llm,
                start_url=start_url,
                max_steps=nav_max_steps,
                step_delay_ms=step_delay_ms,
            )
        except Exception as exc:
            return ChatProcessingResult(
                messages=[*messages, str(exc)],
                turn=turn,
                error=str(exc),
            )

        record_nav_result(state, nav_result, session)
        messages.append(f"Result: {nav_result.status}")
        if nav_result.final_answer:
            messages.append(nav_result.final_answer)
        elif nav_result.final_message:
            messages.append(nav_result.final_message)

    return ChatProcessingResult(
        messages=messages,
        turn=turn,
        nav_result=nav_result,
        should_exit=should_exit,
    )
