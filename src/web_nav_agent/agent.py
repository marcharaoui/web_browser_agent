from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel

from .browser import BrowserSession
from .llm_adapter import BaseLLMAdapter
from .schemas import (
    BackAction,
    ClickAction,
    DoneAction,
    FailAction,
    GotoAction,
    NavDecision,
    NavRunResult,
    NavStepRecord,
    PressAction,
    ScrollAction,
    SelectAction,
    TypeAction,
    WaitAction,
)
from .utils import DEFAULT_ARTIFACTS_DIR, dump_json, ensure_dir, make_run_id

if TYPE_CHECKING:
    from .schemas import NavAction


NAV_SYSTEM_PROMPT = """You are a web navigation agent.
Return JSON only.
Think briefly, then choose exactly one next action.
Use the screenshot and visible elements to decide.
Prefer element_id over coordinates whenever possible.
The action field must be an object, for example {"type":"CLICK","element_id":"el-4"} or {"type":"DONE","answer":"..."}.
Available actions:
- GOTO(url)
- CLICK(element_id or coordinates)
- TYPE(element_id or coordinates, text, clear_first)
- PRESS(key)
- SCROLL(direction, amount)
- WAIT(milliseconds)
- SELECT(element_id, value or label)
- BACK()
- DONE(answer)
- FAIL(reason)
Choose DONE only when the task is fully complete.
Choose FAIL when the task cannot be completed from the current site or page state.
After typing into a search field, prefer PRESS(Enter) over clicking a submit button when both would work.
"""


class StepExecution(BaseModel):
    status: str
    message: str


def _default_start_url(session: BrowserSession) -> str | None:
    if session.has_live_page():
        return None
    return "https://www.google.com"


def _execute_action(session: BrowserSession, action: NavAction) -> StepExecution:
    if isinstance(action, GotoAction):
        session.goto(action.url)
        return StepExecution(status="executed", message=f"navigated to {action.url}")
    if isinstance(action, ClickAction):
        if action.element_id is not None:
            session.click_element_id(action.element_id)
            return StepExecution(status="executed", message=f"clicked {action.element_id}")
        pixels = session.click_coordinates(action.coordinates)
        return StepExecution(status="executed", message=f"clicked at {pixels}")
    if isinstance(action, TypeAction):
        if action.element_id is not None:
            session.type_element_id(action.element_id, action.text, clear_first=action.clear_first)
            return StepExecution(status="executed", message=f"typed into {action.element_id}")
        pixels = session.type_coordinates(action.coordinates, action.text, clear_first=action.clear_first)
        return StepExecution(status="executed", message=f"typed at {pixels}")
    if isinstance(action, PressAction):
        session.press(action.key)
        return StepExecution(status="executed", message=f"pressed {action.key}")
    if isinstance(action, ScrollAction):
        session.scroll_page(action.direction, action.amount)
        return StepExecution(status="executed", message=f"scrolled {action.direction}")
    if isinstance(action, WaitAction):
        session.wait(action.milliseconds)
        return StepExecution(status="executed", message=f"waited {action.milliseconds}ms")
    if isinstance(action, SelectAction):
        session.select_element(action.element_id, value=action.value, label=action.label)
        return StepExecution(status="executed", message=f"selected option in {action.element_id}")
    if isinstance(action, BackAction):
        session.back()
        return StepExecution(status="executed", message="went back")
    if isinstance(action, DoneAction):
        return StepExecution(status="skipped", message=action.answer)
    if isinstance(action, FailAction):
        return StepExecution(status="skipped", message=action.reason)
    raise RuntimeError(f"Unsupported action: {action}")


def run_nav_agent(
    *,
    session: BrowserSession,
    goal: str,
    llm: BaseLLMAdapter,
    start_url: str | None = None,
    max_steps: int = 12,
    step_delay_ms: int = 1200,
    artifacts_dir: Path | None = None,
) -> NavRunResult:
    run_id = make_run_id(goal)
    base_dir = artifacts_dir or DEFAULT_ARTIFACTS_DIR
    run_dir = ensure_dir(base_dir / run_id)
    steps_dir = ensure_dir(run_dir / "steps")
    screenshots_dir = ensure_dir(run_dir / "screenshots")

    effective_start_url = start_url if start_url is not None else _default_start_url(session)
    if effective_start_url:
        session.goto(effective_start_url)

    history: list[str] = []
    step_records: list[NavStepRecord] = []
    final_status = "max_steps"
    final_answer: str | None = None
    final_message = "maximum steps reached"
    last_screenshot_path: str | None = None

    for step_index in range(max_steps):
        screenshot_path = screenshots_dir / f"step_{step_index:02d}.png"
        observation = session.capture_observation(
            step_index=step_index,
            action_history=history,
            screenshot_path=screenshot_path,
        )
        last_screenshot_path = observation.screenshot_path

        try:
            decision = llm.generate_json(
                system_prompt=NAV_SYSTEM_PROMPT,
                user_payload={
                    "goal": goal,
                    "observation": observation.model_dump(mode="json"),
                    "recent_actions": history[-5:],
                },
                response_model=NavDecision,
                image_path=observation.screenshot_path,
            )
            execution = _execute_action(session, decision.action)
        except Exception as exc:
            final_status = "error"
            final_message = str(exc)
            step_records.append(
                NavStepRecord(
                    step_index=step_index,
                    url=observation.url,
                    title=observation.title,
                    screenshot_path=observation.screenshot_path,
                    thought="runtime error",
                    action_type="ERROR",
                    action_payload={},
                    execution_status="failed",
                    message=str(exc),
                    expected_outcome="The step could not be completed.",
                )
            )
            dump_json(
                steps_dir / f"step_{step_index:02d}.json",
                {"observation": observation, "error": str(exc)},
            )
            break

        step_records.append(
            NavStepRecord(
                step_index=step_index,
                url=observation.url,
                title=observation.title,
                screenshot_path=observation.screenshot_path,
                thought=decision.thought,
                action_type=decision.action.type,
                action_payload=decision.action.model_dump(mode="json"),
                execution_status=execution.status,
                message=execution.message,
                expected_outcome=decision.expected_outcome,
            )
        )
        dump_json(
            steps_dir / f"step_{step_index:02d}.json",
            {
                "observation": observation,
                "decision": decision,
                "execution": execution,
            },
        )

        history.append(f"{decision.action.type}: {execution.message}")

        if isinstance(decision.action, DoneAction):
            final_status = "success"
            final_answer = decision.action.answer
            final_message = decision.expected_outcome
            break
        if isinstance(decision.action, FailAction):
            final_status = "failed"
            final_message = decision.action.reason
            break
        if execution.status == "failed":
            final_status = "failed"
            final_message = execution.message
            break
        if step_delay_ms > 0:
            session.wait(step_delay_ms)

    summary = session.current_page_summary()
    result = NavRunResult(
        run_id=run_id,
        goal=goal,
        status=final_status,
        start_url=effective_start_url,
        final_url=summary.url,
        final_title=summary.title,
        final_answer=final_answer,
        final_message=final_message,
        steps_taken=len(step_records),
        max_steps=max_steps,
        run_dir=str(run_dir),
        summary_path=str(run_dir / "summary.json"),
        last_screenshot_path=last_screenshot_path,
    )
    dump_json(Path(result.summary_path), result)
    return result
