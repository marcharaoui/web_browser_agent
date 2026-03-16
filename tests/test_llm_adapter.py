from web_nav_agent.llm_adapter import OpenRouterAdapter


def test_request_kwargs_include_openrouter_headers() -> None:
    adapter = object.__new__(OpenRouterAdapter)
    adapter.model_name = "openai/gpt-4.1-mini"
    adapter.site_url = "https://example.com"
    adapter.site_name = "web-nav-agent"
    adapter._data_url = lambda _: "data:image/png;base64,ZmFrZQ=="

    request = adapter._request_kwargs(
        "system rules",
        {"task": "say hi"},
        type("DummyModel", (), {"model_json_schema": staticmethod(lambda: {"type": "object"})}),
        "shot.png",
    )

    assert request["model"] == "openai/gpt-4.1-mini"
    assert request["extra_headers"]["HTTP-Referer"] == "https://example.com"
    assert request["extra_headers"]["X-OpenRouter-Title"] == "web-nav-agent"
    assert request["messages"][0]["content"][1]["type"] == "image_url"


def test_clean_json_removes_markdown_fences() -> None:
    adapter = object.__new__(OpenRouterAdapter)

    cleaned = adapter._clean_json('```json\n{"answer":"hello"}\n```')

    assert cleaned == '{"answer":"hello"}'


def test_clean_json_extracts_embedded_object() -> None:
    adapter = object.__new__(OpenRouterAdapter)

    cleaned = adapter._clean_json('Result:\n{"answer":"hello"}\nDone.')

    assert cleaned == '{"answer":"hello"}'
