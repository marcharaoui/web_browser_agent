import pytest
from pydantic import ValidationError

from web_nav_agent.schemas import ChatTurnResult, NavDecision


def test_nav_decision_accepts_select_action() -> None:
    decision = NavDecision.model_validate(
        {
            "thought": "The site uses a select dropdown for the format.",
            "action": {"type": "SELECT", "element_id": "el-4", "label": "Hardcover"},
            "expected_outcome": "The hardcover option should be selected.",
        }
    )

    assert decision.action.type == "SELECT"
    assert decision.action.element_id == "el-4"


def test_nav_decision_rejects_click_without_target() -> None:
    with pytest.raises(ValidationError, match="CLICK requires element_id or coordinates"):
        NavDecision.model_validate(
            {
                "thought": "Click the next item.",
                "action": {"type": "CLICK"},
                "expected_outcome": "The next page should open.",
            }
        )


def test_nav_decision_rejects_out_of_bounds_coordinates() -> None:
    with pytest.raises(ValidationError):
        NavDecision.model_validate(
            {
                "thought": "Fallback to coordinates.",
                "action": {"type": "CLICK", "coordinates": {"x": 1200, "y": 100}},
                "expected_outcome": "The target should be clicked.",
            }
        )


def test_nav_decision_accepts_done_action_shorthand_string() -> None:
    decision = NavDecision.model_validate(
        {
            "thought_summary": "The answer is already visible.",
            "action": "DONE",
            "answer": "Python 3.14.3",
            "expected_outcome": "Return the answer.",
        }
    )

    assert decision.thought == "The answer is already visible."
    assert decision.action.type == "DONE"
    assert decision.action.answer == "Python 3.14.3"


def test_chat_turn_accepts_run_nav_followup_payload() -> None:
    result = ChatTurnResult.model_validate(
        {
            "thought": "This is a follow-up on the current page.",
            "decision": {
                "type": "RUN_NAV",
                "goal": "Find the reviews for the same book",
                "start_url": None,
                "reuse_current_page": True,
            },
            "assistant_message": "I will continue from the current page and look for reviews.",
            "page_summary": {
                "has_live_page": True,
                "url": "https://www.amazon.com/example",
                "title": "Example Book",
                "content_excerpt": "Example Book page",
            },
        }
    )

    assert result.decision.type == "RUN_NAV"
    assert result.decision.reuse_current_page is True


def test_chat_turn_rejects_unknown_decision_type() -> None:
    with pytest.raises(ValidationError):
        ChatTurnResult.model_validate(
            {
                "thought": "Unknown routing.",
                "decision": {"type": "SOMETHING_ELSE", "message": "nope"},
                "assistant_message": "Unsupported.",
            }
        )
