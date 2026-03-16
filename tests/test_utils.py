from pathlib import Path

from web_nav_agent.utils import dump_json, load_json, normalized_to_pixels, slugify, truncate_text


def test_slugify_normalizes_and_truncates() -> None:
    slug = slugify(" Find the Price of The Requiem Red!!! ", max_length=12)
    assert slug == "find-the-pri"


def test_dump_and_load_json_round_trip() -> None:
    payload = {"goal": "find book price", "steps": [1, 2, 3]}
    target = Path("artifacts/test_temp/sample.json")

    dump_json(target, payload)
    loaded = load_json(target)

    assert loaded == payload


def test_normalized_to_pixels_converts_coordinates() -> None:
    assert normalized_to_pixels(500, 250, 1200, 800) == (600, 200)


def test_truncate_text_compacts_whitespace() -> None:
    text = "hello   world\n\nfrom   codex"
    assert truncate_text(text, 14) == "hello world fr"
