from __future__ import annotations

from web_nav_agent.chatbot import guess_start_url, process_chat_message, record_nav_result, run_chat_turn
from web_nav_agent.schemas import ChatState, ChatTurnResult, NavRunResult, PageSummary


class DummySession:
    def __init__(self, summary: PageSummary | None = None) -> None:
        self.summary = summary or PageSummary(has_live_page=False, url=None, title=None, content_excerpt="")

    def current_page_summary(self) -> PageSummary:
        return self.summary


class FakeLLM:
    def __init__(self, response: object) -> None:
        self.response = response

    def generate_json(self, **_: object) -> object:
        return ChatTurnResult.model_validate(self.response)


class RaisingLLM:
    def generate_json(self, **_: object) -> object:
        raise RuntimeError("model failure")


def test_guess_start_url_prefers_explicit_url() -> None:
    assert guess_start_url("go to https://example.com and search") == "https://example.com"


def test_guess_start_url_can_map_common_sites() -> None:
    assert guess_start_url("find this book on amazon") == "https://www.amazon.com"


def test_chat_turn_fills_default_start_url() -> None:
    state = ChatState()
    llm = FakeLLM(
        {
            "thought": "This needs the browser.",
            "decision": {
                "type": "RUN_NAV",
                "goal": "Find the price of Dune",
                "start_url": None,
                "reuse_current_page": False,
            },
            "assistant_message": "I will check that.",
        }
    )

    result = run_chat_turn(
        state=state,
        user_message="Find the price of Dune",
        session=DummySession(),
        llm=llm,
    )

    assert result.decision.type == "RUN_NAV"
    assert result.decision.start_url == "https://www.google.com"


def test_chat_turn_stops_at_budget_limit() -> None:
    state = ChatState(max_turns=1, turns_used=1)
    llm = FakeLLM(
        {
            "thought": "unused",
            "decision": {"type": "ANSWER", "message": "unused"},
            "assistant_message": "unused",
        }
    )

    result = run_chat_turn(
        state=state,
        user_message="hello",
        session=DummySession(),
        llm=llm,
    )

    assert result.decision.type == "EXIT"


def test_record_nav_result_updates_chat_state_from_live_page() -> None:
    state = ChatState()
    result = NavRunResult(
        run_id="run-1",
        goal="Find the price",
        status="success",
        start_url="https://www.google.com",
        final_url="https://example.com/book",
        final_title="Book page",
        final_answer="$12.99",
        final_message="done",
        steps_taken=2,
        max_steps=12,
        run_dir="artifacts/runs/run-1",
        summary_path="artifacts/runs/run-1/summary.json",
        last_screenshot_path="artifacts/runs/run-1/screenshots/step_00.png",
    )
    session = DummySession(
        PageSummary(
            has_live_page=True,
            url="https://example.com/book",
            title="Book page",
            content_excerpt="Price and reviews",
        )
    )

    record_nav_result(state, result, session)

    assert state.last_nav_result == result
    assert state.last_page_url == "https://example.com/book"
    assert state.has_live_page is True


def test_process_chat_message_returns_direct_answer() -> None:
    state = ChatState()
    result = process_chat_message(
        state=state,
        user_message="hello",
        session=DummySession(),
        chat_llm=FakeLLM(
            {
                "thought": "A direct answer is enough.",
                "decision": {"type": "ANSWER", "message": "Hello there."},
                "assistant_message": "Hello there.",
            }
        ),
        nav_llm=FakeLLM(
            {
                "thought": "unused",
                "decision": {"type": "ANSWER", "message": "unused"},
                "assistant_message": "unused",
            }
        ),
        nav_max_steps=5,
        step_delay_ms=0,
    )

    assert result.messages == ["Hello there."]
    assert result.nav_result is None
    assert result.should_exit is False


def test_process_chat_message_runs_navigation(monkeypatch: pytest.MonkeyPatch) -> None:
    nav_result = NavRunResult(
        run_id="run-1",
        goal="Find the price of Dune",
        status="success",
        start_url="https://www.google.com",
        final_url="https://example.com/book",
        final_title="Book page",
        final_answer="$12.99",
        final_message="done",
        steps_taken=2,
        max_steps=12,
        run_dir="artifacts/runs/run-1",
        summary_path="artifacts/runs/run-1/summary.json",
        last_screenshot_path="artifacts/runs/run-1/screenshots/step_00.png",
    )

    monkeypatch.setattr("web_nav_agent.chatbot.run_nav_agent", lambda **_: nav_result)

    state = ChatState()
    result = process_chat_message(
        state=state,
        user_message="Find the price of Dune",
        session=DummySession(
            PageSummary(
                has_live_page=True,
                url="https://example.com/book",
                title="Book page",
                content_excerpt="Price and reviews",
            )
        ),
        chat_llm=FakeLLM(
            {
                "thought": "This needs the browser.",
                "decision": {
                    "type": "RUN_NAV",
                    "goal": "Find the price of Dune",
                    "start_url": "https://www.google.com",
                    "reuse_current_page": False,
                },
                "assistant_message": "I will check that.",
            }
        ),
        nav_llm=FakeLLM(
            {
                "thought": "unused",
                "decision": {"type": "ANSWER", "message": "unused"},
                "assistant_message": "unused",
            }
        ),
        nav_max_steps=5,
        step_delay_ms=0,
    )

    assert result.messages == ["I will check that.", "Result: success", "$12.99"]
    assert result.nav_result == nav_result
    assert state.last_nav_result == nav_result


def test_process_chat_message_returns_error_message_on_failure() -> None:
    result = process_chat_message(
        state=ChatState(),
        user_message="hello",
        session=DummySession(),
        chat_llm=RaisingLLM(),
        nav_llm=RaisingLLM(),
        nav_max_steps=5,
        step_delay_ms=0,
    )

    assert result.messages == ["model failure"]
    assert result.error == "model failure"
