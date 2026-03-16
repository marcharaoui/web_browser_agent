from __future__ import annotations

from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, Field, model_validator


class BoundingBox(BaseModel):
    x: int = Field(ge=0, le=1000)
    y: int = Field(ge=0, le=1000)
    width: int = Field(ge=0, le=1000)
    height: int = Field(ge=0, le=1000)


class CoordinatePoint(BaseModel):
    x: int = Field(ge=0, le=1000)
    y: int = Field(ge=0, le=1000)


class VisibleElement(BaseModel):
    element_id: str
    tag_name: str
    role: str | None = None
    accessible_name: str | None = None
    text: str | None = None
    clickable: bool
    editable: bool
    bbox: BoundingBox | None = None


class PageSummary(BaseModel):
    has_live_page: bool
    url: str | None = None
    title: str | None = None
    content_excerpt: str = ""


class Observation(PageSummary):
    step_index: int
    viewport_width: int
    viewport_height: int
    screenshot_path: str | None = None
    action_history: list[str] = Field(default_factory=list)
    visible_elements: list[VisibleElement] = Field(default_factory=list)


class GotoAction(BaseModel):
    type: Literal["GOTO"]
    url: str


class ClickAction(BaseModel):
    type: Literal["CLICK"]
    element_id: str | None = None
    coordinates: CoordinatePoint | None = None

    @model_validator(mode="after")
    def validate_target(self) -> "ClickAction":
        if self.element_id is None and self.coordinates is None:
            raise ValueError("CLICK requires element_id or coordinates")
        return self


class TypeAction(BaseModel):
    type: Literal["TYPE"]
    element_id: str | None = None
    coordinates: CoordinatePoint | None = None
    text: str
    clear_first: bool = True

    @model_validator(mode="after")
    def validate_target(self) -> "TypeAction":
        if self.element_id is None and self.coordinates is None:
            raise ValueError("TYPE requires element_id or coordinates")
        return self


class PressAction(BaseModel):
    type: Literal["PRESS"]
    key: str


class ScrollAction(BaseModel):
    type: Literal["SCROLL"]
    direction: Literal["up", "down"] = "down"
    amount: int = Field(default=700, ge=50, le=5000)


class WaitAction(BaseModel):
    type: Literal["WAIT"]
    milliseconds: int = Field(default=1000, ge=50, le=10000)


class SelectAction(BaseModel):
    type: Literal["SELECT"]
    element_id: str
    value: str | None = None
    label: str | None = None

    @model_validator(mode="after")
    def validate_selection(self) -> "SelectAction":
        if not self.value and not self.label:
            raise ValueError("SELECT requires value or label")
        return self


class BackAction(BaseModel):
    type: Literal["BACK"]


class DoneAction(BaseModel):
    type: Literal["DONE"]
    answer: str


class FailAction(BaseModel):
    type: Literal["FAIL"]
    reason: str


NavAction = Annotated[
    Union[
        GotoAction,
        ClickAction,
        TypeAction,
        PressAction,
        ScrollAction,
        WaitAction,
        SelectAction,
        BackAction,
        DoneAction,
        FailAction,
    ],
    Field(discriminator="type"),
]


class NavDecision(BaseModel):
    thought: str
    action: NavAction
    expected_outcome: str

    @model_validator(mode="before")
    @classmethod
    def normalize_shorthand_fields(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value

        data = dict(value)
        if not data.get("thought") and data.get("thought_summary"):
            data["thought"] = data["thought_summary"]

        action = data.get("action")
        if isinstance(action, str):
            action_type = action.strip().upper()
            normalized_action: dict[str, Any] = {"type": action_type}
            if action_type == "DONE":
                normalized_action["answer"] = (
                    data.get("answer")
                    or data.get("final_answer")
                    or data.get("message")
                    or data.get("response")
                    or ""
                )
            elif action_type == "FAIL":
                normalized_action["reason"] = data.get("reason") or data.get("message") or ""
            elif action_type == "GOTO" and data.get("url"):
                normalized_action["url"] = data["url"]
            elif action_type == "PRESS" and data.get("key"):
                normalized_action["key"] = data["key"]
            elif action_type == "WAIT" and data.get("milliseconds") is not None:
                normalized_action["milliseconds"] = data["milliseconds"]
            elif action_type == "SCROLL":
                if data.get("direction"):
                    normalized_action["direction"] = data["direction"]
                if data.get("amount") is not None:
                    normalized_action["amount"] = data["amount"]
            elif action_type in {"CLICK", "TYPE"}:
                if data.get("element_id"):
                    normalized_action["element_id"] = data["element_id"]
                if data.get("coordinates") is not None:
                    normalized_action["coordinates"] = data["coordinates"]
                if action_type == "TYPE":
                    normalized_action["text"] = data.get("text", "")
                    if "clear_first" in data:
                        normalized_action["clear_first"] = data["clear_first"]
            elif action_type == "SELECT":
                if data.get("element_id"):
                    normalized_action["element_id"] = data["element_id"]
                if data.get("value"):
                    normalized_action["value"] = data["value"]
                if data.get("label"):
                    normalized_action["label"] = data["label"]
            data["action"] = normalized_action

        return data


class NavStepRecord(BaseModel):
    step_index: int
    url: str | None = None
    title: str | None = None
    screenshot_path: str | None = None
    thought: str
    action_type: str
    action_payload: dict[str, Any] = Field(default_factory=dict)
    execution_status: Literal["executed", "failed", "skipped"]
    message: str
    expected_outcome: str


class NavRunResult(BaseModel):
    run_id: str
    goal: str
    status: Literal["success", "failed", "max_steps", "error"]
    start_url: str | None = None
    final_url: str | None = None
    final_title: str | None = None
    final_answer: str | None = None
    final_message: str | None = None
    steps_taken: int
    max_steps: int
    run_dir: str
    summary_path: str
    last_screenshot_path: str | None = None


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatState(BaseModel):
    history: list[ChatMessage] = Field(default_factory=list)
    turns_used: int = 0
    max_turns: int = Field(default=10, ge=1)
    last_nav_goal: str | None = None
    last_nav_result: NavRunResult | None = None
    last_page_url: str | None = None
    last_page_title: str | None = None
    has_live_page: bool = False


class AnswerDecision(BaseModel):
    type: Literal["ANSWER"]
    message: str


class ClarifyDecision(BaseModel):
    type: Literal["CLARIFY"]
    question: str


class RunNavDecision(BaseModel):
    type: Literal["RUN_NAV"]
    goal: str
    start_url: str | None = None
    reuse_current_page: bool = False


class ExitDecision(BaseModel):
    type: Literal["EXIT"]
    message: str


ChatDecision = Annotated[
    Union[AnswerDecision, ClarifyDecision, RunNavDecision, ExitDecision],
    Field(discriminator="type"),
]


class ChatTurnResult(BaseModel):
    thought: str
    decision: ChatDecision
    assistant_message: str
    page_summary: PageSummary | None = None
