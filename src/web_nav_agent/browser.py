from __future__ import annotations

import asyncio
from concurrent.futures import Future
from queue import Queue
import sys
from threading import Thread
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from .schemas import CoordinatePoint, Observation, PageSummary, VisibleElement
from .utils import ensure_dir, normalized_to_pixels, truncate_text

if TYPE_CHECKING:
    from playwright.sync_api import Browser, BrowserContext, Page, Playwright
else:
    Browser = BrowserContext = Page = Playwright = Any

try:
    from playwright.sync_api import sync_playwright
except ModuleNotFoundError:
    sync_playwright = None


INTERACTIVE_JS = r"""
() => {
  const clamp = (value) => Math.max(0, Math.min(1000, Math.round(value)));
  const inputType = (node) => (node.getAttribute('type') || 'text').toLowerCase();
  const isEditableInput = (node) => {
    if (node.tagName.toLowerCase() !== 'input') return false;
    return !['button', 'submit', 'reset', 'checkbox', 'radio', 'file', 'image', 'color', 'range'].includes(inputType(node));
  };
  const viewportWidth = window.innerWidth || 1;
  const viewportHeight = window.innerHeight || 1;
  const nodes = Array.from(
    document.querySelectorAll(
      'a, button, input, textarea, select, [role="button"], [role="link"], [contenteditable="true"]'
    )
  );
  const items = [];

  for (const [index, node] of nodes.entries()) {
    const rect = node.getBoundingClientRect();
    const style = window.getComputedStyle(node);
    if (rect.width <= 0 || rect.height <= 0) continue;
    if (style.visibility === 'hidden' || style.display === 'none') continue;
    const inViewport = rect.bottom > 0 && rect.top < viewportHeight && rect.right > 0 && rect.left < viewportWidth;
    if (!inViewport) continue;

    const text = (node.innerText || node.value || '').trim().replace(/\s+/g, ' ').slice(0, 160);
    const accessibleName = (node.getAttribute('aria-label') || node.getAttribute('title') || text).trim().slice(0, 160);
    const role = node.getAttribute('role') || node.tagName.toLowerCase();
    const tagName = node.tagName.toLowerCase();
    const elementId = `el-${index}`;
    node.setAttribute('data-web-nav-id', elementId);

    items.push({
      element_id: elementId,
      tag_name: tagName,
      role,
      accessible_name: accessibleName || null,
      text: text || null,
      clickable: ['a', 'button'].includes(tagName) || ['button', 'link'].includes(role),
      editable: isEditableInput(node) || ['textarea', 'select'].includes(tagName) || node.isContentEditable,
      bbox: {
        x: clamp((rect.x / viewportWidth) * 1000),
        y: clamp((rect.y / viewportHeight) * 1000),
        width: clamp((rect.width / viewportWidth) * 1000),
        height: clamp((rect.height / viewportHeight) * 1000)
      }
    });
  }

  return items;
}
"""


def _ensure_windows_subprocess_event_loop_policy() -> None:
    if sys.platform != "win32":
        return

    proactor_policy_cls = getattr(asyncio, "WindowsProactorEventLoopPolicy", None)
    if proactor_policy_cls is None:
        return

    current_policy = asyncio.get_event_loop_policy()
    if isinstance(current_policy, proactor_policy_cls):
        return

    asyncio.set_event_loop_policy(proactor_policy_cls())


def _is_transient_navigation_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "execution context was destroyed" in message or "most likely because of a navigation" in message


class ThreadedBrowserSession:
    def __init__(
        self,
        *,
        headless: bool = False,
        viewport_width: int = 1440,
        viewport_height: int = 900,
        action_timeout_ms: int = 5000,
        navigation_timeout_ms: int = 15000,
        content_excerpt_chars: int = 1000,
        max_visible_elements: int = 40,
        session_factory: Callable[..., BrowserSession] | None = None,
    ) -> None:
        self.headless = headless
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height
        self.action_timeout_ms = action_timeout_ms
        self.navigation_timeout_ms = navigation_timeout_ms
        self.content_excerpt_chars = content_excerpt_chars
        self.max_visible_elements = max_visible_elements
        self._session_factory = session_factory or BrowserSession
        self._commands: Queue[tuple[str, tuple[Any, ...], dict[str, Any], Future[Any]]] = Queue()
        self._thread: Thread | None = None

    def _make_session(self) -> "BrowserSession":
        return self._session_factory(
            headless=self.headless,
            viewport_width=self.viewport_width,
            viewport_height=self.viewport_height,
            action_timeout_ms=self.action_timeout_ms,
            navigation_timeout_ms=self.navigation_timeout_ms,
            content_excerpt_chars=self.content_excerpt_chars,
            max_visible_elements=self.max_visible_elements,
        )

    def _worker_main(self, started: Future[None]) -> None:
        session: BrowserSession | None = None
        try:
            session = self._make_session()
            session.start()
            started.set_result(None)

            while True:
                method_name, args, kwargs, future = self._commands.get()
                try:
                    if method_name == "__stop__":
                        future.set_result(None)
                        return
                    method = getattr(session, method_name)
                    future.set_result(method(*args, **kwargs))
                except Exception as exc:
                    future.set_exception(exc)
        except Exception as exc:
            if not started.done():
                started.set_exception(exc)
        finally:
            if session is not None:
                try:
                    session.stop()
                except Exception:
                    pass

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return

        started: Future[None] = Future()
        self._thread = Thread(
            target=self._worker_main,
            args=(started,),
            daemon=True,
            name="playwright-browser-worker",
        )
        self._thread.start()
        started.result()

    def stop(self) -> None:
        if self._thread is None:
            return
        if self._thread.is_alive():
            self._call("__stop__")
            self._thread.join(timeout=5)
        self._thread = None

    def _call(self, method_name: str, *args: Any, **kwargs: Any) -> Any:
        if self._thread is None or not self._thread.is_alive():
            raise RuntimeError("ThreadedBrowserSession has not been started")
        future: Future[Any] = Future()
        self._commands.put((method_name, args, kwargs, future))
        return future.result()

    def has_live_page(self) -> bool:
        return self._call("has_live_page")

    def goto(self, url: str) -> None:
        self._call("goto", url)

    def back(self) -> None:
        self._call("back")

    def wait(self, milliseconds: int) -> None:
        self._call("wait", milliseconds)

    def press(self, key: str) -> None:
        self._call("press", key)

    def scroll_page(self, direction: str, amount: int) -> None:
        self._call("scroll_page", direction, amount)

    def click_element_id(self, element_id: str) -> None:
        self._call("click_element_id", element_id)

    def click_coordinates(self, point: CoordinatePoint) -> tuple[int, int]:
        return self._call("click_coordinates", point)

    def type_element_id(self, element_id: str, text: str, clear_first: bool = True) -> None:
        self._call("type_element_id", element_id, text, clear_first=clear_first)

    def type_coordinates(self, point: CoordinatePoint, text: str, clear_first: bool = True) -> tuple[int, int]:
        return self._call("type_coordinates", point, text, clear_first=clear_first)

    def select_element(self, element_id: str, *, value: str | None = None, label: str | None = None) -> None:
        self._call("select_element", element_id, value=value, label=label)

    def screenshot(self, path: Path) -> str | None:
        return self._call("screenshot", path)

    def get_visible_elements(self) -> list[VisibleElement]:
        return self._call("get_visible_elements")

    def current_page_summary(self) -> PageSummary:
        return self._call("current_page_summary")

    def capture_observation(
        self,
        *,
        step_index: int,
        action_history: list[str],
        screenshot_path: Path | None = None,
    ) -> Observation:
        return self._call(
            "capture_observation",
            step_index=step_index,
            action_history=action_history,
            screenshot_path=screenshot_path,
        )


class BrowserSession:
    def __init__(
        self,
        *,
        headless: bool = False,
        viewport_width: int = 1440,
        viewport_height: int = 900,
        action_timeout_ms: int = 5000,
        navigation_timeout_ms: int = 15000,
        content_excerpt_chars: int = 1000,
        max_visible_elements: int = 40,
    ) -> None:
        self.headless = headless
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height
        self.action_timeout_ms = action_timeout_ms
        self.navigation_timeout_ms = navigation_timeout_ms
        self.content_excerpt_chars = content_excerpt_chars
        self.max_visible_elements = max_visible_elements
        self.playwright: Playwright | None = None
        self.browser: Browser | None = None
        self.context: BrowserContext | None = None
        self.page: Page | None = None

    def start(self) -> None:
        if sync_playwright is None:
            raise RuntimeError("Playwright is not installed. Run `pip install -e .` and `playwright install chromium`.")
        # Streamlit on Windows can install a selector-based asyncio policy
        # that breaks Playwright's subprocess startup path.
        _ensure_windows_subprocess_event_loop_policy()
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(headless=self.headless)
        self.context = self.browser.new_context(
            viewport={"width": self.viewport_width, "height": self.viewport_height}
        )
        self.page = self.context.new_page()
        self.page.set_default_timeout(self.action_timeout_ms)
        self.page.set_default_navigation_timeout(self.navigation_timeout_ms)

    def stop(self) -> None:
        if self.context is not None:
            self.context.close()
        if self.browser is not None:
            self.browser.close()
        if self.playwright is not None:
            self.playwright.stop()

    def __enter__(self) -> "BrowserSession":
        self.start()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.stop()

    @property
    def active_page(self) -> Page:
        if self.page is None:
            raise RuntimeError("BrowserSession has not been started")
        return self.page

    def has_live_page(self) -> bool:
        if self.page is None:
            return False
        url = self.page.url or ""
        return bool(url and not url.startswith("about:blank"))

    def goto(self, url: str) -> None:
        self.active_page.goto(url, wait_until="domcontentloaded")

    def back(self) -> None:
        self.active_page.go_back(wait_until="domcontentloaded")

    def wait(self, milliseconds: int) -> None:
        self.active_page.wait_for_timeout(milliseconds)

    def press(self, key: str) -> None:
        self.active_page.keyboard.press(key)

    def scroll_page(self, direction: str, amount: int) -> None:
        delta = amount if direction == "down" else -amount
        self.active_page.mouse.wheel(0, delta)

    def click_element_id(self, element_id: str) -> None:
        locator = self.active_page.locator(f'[data-web-nav-id="{element_id}"]').first
        try:
            locator.click()
        except Exception as exc:
            if "intercepts pointer events" not in str(exc):
                raise
            locator.click(force=True)

    def click_coordinates(self, point: CoordinatePoint) -> tuple[int, int]:
        viewport = self.active_page.viewport_size or {
            "width": self.viewport_width,
            "height": self.viewport_height,
        }
        pixel_x, pixel_y = normalized_to_pixels(point.x, point.y, viewport["width"], viewport["height"])
        self.active_page.mouse.click(pixel_x, pixel_y)
        return pixel_x, pixel_y

    def type_element_id(self, element_id: str, text: str, clear_first: bool = True) -> None:
        locator = self.active_page.locator(f'[data-web-nav-id="{element_id}"]').first
        if clear_first:
            locator.fill(text)
        else:
            locator.press_sequentially(text)

    def type_coordinates(self, point: CoordinatePoint, text: str, clear_first: bool = True) -> tuple[int, int]:
        pixel_x, pixel_y = self.click_coordinates(point)
        if clear_first:
            self.press("Control+A")
        self.active_page.keyboard.type(text)
        return pixel_x, pixel_y

    def select_element(self, element_id: str, *, value: str | None = None, label: str | None = None) -> None:
        locator = self.active_page.locator(f'[data-web-nav-id="{element_id}"]').first
        if value:
            locator.select_option(value=value)
            return
        locator.select_option(label=label)

    def screenshot(self, path: Path) -> str | None:
        ensure_dir(path.parent)
        try:
            self.active_page.screenshot(
                path=str(path),
                full_page=False,
                animations="disabled",
                timeout=3000,
            )
        except Exception:
            return None
        return str(path)

    def _safe_body_text(self) -> str:
        try:
            return self.active_page.locator("body").inner_text(timeout=1500)
        except Exception:
            return ""

    def get_visible_elements(self) -> list[VisibleElement]:
        if not self.has_live_page():
            return []
        for attempt in range(3):
            try:
                raw_items = self.active_page.evaluate(INTERACTIVE_JS)
                items = [VisibleElement.model_validate(item) for item in raw_items]
                return items[: self.max_visible_elements]
            except Exception as exc:
                if not _is_transient_navigation_error(exc):
                    raise
                if attempt == 2:
                    return []
                self.wait(250)
        return []

    def current_page_summary(self) -> PageSummary:
        if not self.has_live_page():
            return PageSummary(has_live_page=False, url=None, title=None, content_excerpt="")
        for attempt in range(3):
            try:
                return PageSummary(
                    has_live_page=True,
                    url=self.active_page.url,
                    title=self.active_page.title(),
                    content_excerpt=truncate_text(self._safe_body_text(), self.content_excerpt_chars),
                )
            except Exception as exc:
                if not _is_transient_navigation_error(exc):
                    raise
                if attempt == 2:
                    return PageSummary(
                        has_live_page=True,
                        url=self.active_page.url if self.page is not None else None,
                        title=None,
                        content_excerpt="",
                    )
                self.wait(250)
        return PageSummary(has_live_page=False, url=None, title=None, content_excerpt="")

    def capture_observation(
        self,
        *,
        step_index: int,
        action_history: list[str],
        screenshot_path: Path | None = None,
    ) -> Observation:
        summary = self.current_page_summary()
        screenshot_str = None
        if screenshot_path is not None and summary.has_live_page:
            screenshot_str = self.screenshot(screenshot_path)

        return Observation(
            has_live_page=summary.has_live_page,
            url=summary.url,
            title=summary.title,
            content_excerpt=summary.content_excerpt,
            step_index=step_index,
            viewport_width=self.viewport_width,
            viewport_height=self.viewport_height,
            screenshot_path=screenshot_str,
            action_history=action_history[-5:],
            visible_elements=self.get_visible_elements(),
        )
