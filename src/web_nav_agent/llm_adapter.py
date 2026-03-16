from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
from pathlib import Path
from typing import Any, TypeVar

from dotenv import load_dotenv
from pydantic import BaseModel

from .utils import ROOT_DIR

try:
    from openai import OpenAI
except ModuleNotFoundError:
    OpenAI = None


load_dotenv(ROOT_DIR / ".env")

ResponseModelT = TypeVar("ResponseModelT", bound=BaseModel)
DEFAULT_MODELS = {
    "NAV_MODEL_NAME": "openai/gpt-4.1-mini",
    "CHAT_MODEL_NAME": "openai/gpt-4.1-mini",
}


class BaseLLMAdapter:
    def generate_json(
        self,
        *,
        system_prompt: str,
        user_payload: dict[str, Any],
        response_model: type[ResponseModelT],
        image_path: str | None = None,
    ) -> ResponseModelT:
        raise NotImplementedError


class OpenRouterAdapter(BaseLLMAdapter):
    def __init__(self, *, model_name: str, api_key: str | None = None) -> None:
        if OpenAI is None:
            raise RuntimeError("openai is not installed.")

        self.model_name = model_name
        self.api_key = api_key or os.getenv("API_KEY")
        if not self.api_key:
            raise RuntimeError("Missing API key. Set API_KEY.")

        self.client = OpenAI(
            base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            api_key=self.api_key,
        )
        self.site_url = os.getenv("OPENROUTER_SITE_URL")
        self.site_name = os.getenv("OPENROUTER_SITE_NAME")

    def _build_messages(
        self,
        system_prompt: str,
        user_payload: dict[str, Any],
        response_model: type[ResponseModelT],
        image_path: str | None,
    ) -> list[dict[str, Any]]:
        prompt = (
            f"{system_prompt}\n\n"
            "Return exactly one valid JSON object and nothing else.\n"
            f"Schema:\n{json.dumps(response_model.model_json_schema(), indent=2)}\n\n"
            f"Input:\n{json.dumps(user_payload, indent=2)}"
        )
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        if image_path:
            content.append({"type": "image_url", "image_url": {"url": self._data_url(image_path)}})
        return [{"role": "user", "content": content}]

    def _data_url(self, image_path: str) -> str:
        path = Path(image_path)
        mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"

    def _request_kwargs(
        self,
        system_prompt: str,
        user_payload: dict[str, Any],
        response_model: type[ResponseModelT],
        image_path: str | None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.model_name,
            "messages": self._build_messages(system_prompt, user_payload, response_model, image_path),
            "temperature": 0.2,
        }
        headers: dict[str, str] = {}
        if self.site_url:
            headers["HTTP-Referer"] = self.site_url
        if self.site_name:
            headers["X-OpenRouter-Title"] = self.site_name
        if headers:
            kwargs["extra_headers"] = headers
        return kwargs

    def _response_text(self, response: Any) -> str:
        message = response.choices[0].message
        content = message.content
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("text"):
                    parts.append(item["text"])
                elif getattr(item, "text", None):
                    parts.append(item.text)
            return "\n".join(parts).strip()
        return ""

    def _clean_json(self, text: str) -> str:
        text = text.strip()
        match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, flags=re.DOTALL | re.IGNORECASE)
        if match:
            text = match.group(1).strip()

        try:
            json.loads(text)
            return text
        except json.JSONDecodeError:
            decoder = json.JSONDecoder()
            for index, char in enumerate(text):
                if char not in "{[":
                    continue
                try:
                    _, end = decoder.raw_decode(text[index:])
                    return text[index : index + end]
                except json.JSONDecodeError:
                    continue
        return text

    def generate_json(
        self,
        *,
        system_prompt: str,
        user_payload: dict[str, Any],
        response_model: type[ResponseModelT],
        image_path: str | None = None,
    ) -> ResponseModelT:
        response = self.client.chat.completions.create(
            **self._request_kwargs(system_prompt, user_payload, response_model, image_path)
        )
        text = self._response_text(response)
        if not text:
            raise RuntimeError(f"Empty response from model '{self.model_name}'.")
        return response_model.model_validate_json(self._clean_json(text))


def create_adapter(*, model_name_key: str, api_key: str | None = None) -> BaseLLMAdapter:
    model_name = os.getenv(model_name_key, DEFAULT_MODELS.get(model_name_key, "openai/gpt-4.1-mini"))
    return OpenRouterAdapter(model_name=model_name, api_key=api_key)
