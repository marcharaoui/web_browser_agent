from __future__ import annotations

from pathlib import Path

import pytest

from web_nav_agent import browser as browser_module
from web_nav_agent.browser import BrowserSession, ThreadedBrowserSession


class FakeLocator:
    def __init__(self) -> None:
        self.calls: list[bool] = []

    @property
    def first(self) -> "FakeLocator":
        return self

    def click(self, *, force: bool = False) -> None:
        self.calls.append(force)
        if not force:
            raise RuntimeError("subtree intercepts pointer events")


class FakePage:
    def __init__(self, locator: FakeLocator) -> None:
        self._locator = locator

    def locator(self, _: str) -> FakeLocator:
        return self._locator


class FakeScreenshotPage:
    def screenshot(self, **_: object) -> None:
        raise RuntimeError("Page.screenshot: Timeout 3000ms exceeded.")




class FakeNavigatingPage:
    def __init__(self) -> None:
        self.url = "https://example.com"
        self.wait_calls: list[int] = []
        self.evaluate_calls = 0

    def evaluate(self, _: str):
        self.evaluate_calls += 1
        if self.evaluate_calls == 1:
            raise RuntimeError("Page.evaluate: Execution context was destroyed, most likely because of a navigation")
        return [
            {
                "element_id": "el-1",
                "tag_name": "a",
                "role": "a",
                "accessible_name": "Example",
                "text": "Example",
                "clickable": True,
                "editable": False,
                "bbox": {"x": 0, "y": 0, "width": 10, "height": 10},
            }
        ]

    def wait_for_timeout(self, milliseconds: int) -> None:
        self.wait_calls.append(milliseconds)


class FakeWorkerSession:
    def __init__(self, **_: object) -> None:
        self.started = False
        self.urls: list[str] = []

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.started = False

    def goto(self, url: str) -> None:
        self.urls.append(url)

    def current_page_summary(self) -> dict[str, object]:
        return {"started": self.started, "urls": list(self.urls)}


def test_click_element_id_falls_back_to_force_click() -> None:
    locator = FakeLocator()
    session = BrowserSession()
    session.page = FakePage(locator)

    session.click_element_id("el-1")

    assert locator.calls == [False, True]


def test_ensure_windows_subprocess_event_loop_policy_switches_to_proactor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeSelectorPolicy:
        pass

    class FakeProactorPolicy:
        pass

    state = {"policy": FakeSelectorPolicy()}

    monkeypatch.setattr(browser_module.sys, "platform", "win32")
    monkeypatch.setattr(browser_module.asyncio, "WindowsProactorEventLoopPolicy", FakeProactorPolicy, raising=False)
    monkeypatch.setattr(browser_module.asyncio, "get_event_loop_policy", lambda: state["policy"])
    monkeypatch.setattr(browser_module.asyncio, "set_event_loop_policy", lambda policy: state.update(policy=policy))

    browser_module._ensure_windows_subprocess_event_loop_policy()

    assert isinstance(state["policy"], FakeProactorPolicy)


def test_screenshot_returns_none_when_capture_times_out() -> None:
    session = BrowserSession()
    session.page = FakeScreenshotPage()

    result = session.screenshot(Path('artifacts/test_browser/shot.png'))

    assert result is None


def test_threaded_browser_session_reuses_worker_session_across_calls() -> None:
    session = ThreadedBrowserSession(session_factory=FakeWorkerSession)
    session.start()

    session.goto("https://example.com")
    session.goto("https://example.com/next")
    summary = session.current_page_summary()
    session.stop()

    assert summary["started"] is True
    assert summary["urls"] == ["https://example.com", "https://example.com/next"]


def test_get_visible_elements_retries_during_navigation() -> None:
    page = FakeNavigatingPage()
    session = BrowserSession()
    session.page = page

    elements = session.get_visible_elements()

    assert len(elements) == 1
    assert elements[0].element_id == "el-1"
    assert page.wait_calls == [250]
