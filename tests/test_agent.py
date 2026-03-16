from __future__ import annotations

from pathlib import Path

from web_nav_agent.agent import run_nav_agent
from web_nav_agent.schemas import NavDecision, Observation, PageSummary


class FakeSession:
    def __init__(self, *, has_live_page: bool) -> None:
        self.live = has_live_page
        self.goto_calls: list[str] = []
        self.wait_calls: list[int] = []
        self.summary = PageSummary(
            has_live_page=has_live_page,
            url="https://example.com/current" if has_live_page else None,
            title="Current page" if has_live_page else None,
            content_excerpt="Current content",
        )

    def has_live_page(self) -> bool:
        return self.live

    def goto(self, url: str) -> None:
        self.goto_calls.append(url)
        self.live = True
        self.summary = PageSummary(
            has_live_page=True,
            url=url,
            title="Loaded page",
            content_excerpt="Loaded content",
        )

    def capture_observation(
        self,
        *,
        step_index: int,
        action_history: list[str],
        screenshot_path: Path | None = None,
    ) -> Observation:
        return Observation(
            has_live_page=self.summary.has_live_page,
            url=self.summary.url,
            title=self.summary.title,
            content_excerpt=self.summary.content_excerpt,
            step_index=step_index,
            viewport_width=1440,
            viewport_height=900,
            screenshot_path=str(screenshot_path) if screenshot_path is not None else None,
            action_history=action_history,
            visible_elements=[],
        )

    def wait(self, milliseconds: int) -> None:
        self.wait_calls.append(milliseconds)

    def current_page_summary(self) -> PageSummary:
        return self.summary


class FakeLLM:
    def __init__(self, decision: dict) -> None:
        self.decision = decision
        self.calls = 0

    def generate_json(self, **_: object) -> NavDecision:
        self.calls += 1
        return NavDecision.model_validate(self.decision)


def test_nav_agent_defaults_to_google_without_live_page() -> None:
    session = FakeSession(has_live_page=False)
    llm = FakeLLM(
        {
            "thought": "The answer is already known for the test.",
            "action": {"type": "DONE", "answer": "$9.99"},
            "expected_outcome": "Return the answer.",
        }
    )

    result = run_nav_agent(
        session=session,
        goal="Find the price",
        llm=llm,
        max_steps=1,
        step_delay_ms=0,
        artifacts_dir=Path("artifacts/test_agent"),
    )

    assert session.goto_calls == ["https://www.google.com"]
    assert result.status == "success"
    assert result.final_answer == "$9.99"


def test_nav_agent_reuses_live_page_when_no_start_url_is_given() -> None:
    session = FakeSession(has_live_page=True)
    llm = FakeLLM(
        {
            "thought": "This task cannot be completed.",
            "action": {"type": "FAIL", "reason": "No matching product found."},
            "expected_outcome": "Stop the run.",
        }
    )

    result = run_nav_agent(
        session=session,
        goal="Find a missing product",
        llm=llm,
        max_steps=1,
        step_delay_ms=0,
        artifacts_dir=Path("artifacts/test_agent"),
    )

    assert session.goto_calls == []
    assert result.status == "failed"
    assert result.final_message == "No matching product found."
