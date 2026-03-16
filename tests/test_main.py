import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from main import build_parser


def test_parser_defaults_to_gui_mode() -> None:
    args = build_parser().parse_args([])

    assert args.mode == "gui"


def test_parser_accepts_cli_mode() -> None:
    args = build_parser().parse_args(["cli"])

    assert args.mode == "cli"


def test_parser_accepts_nav_mode() -> None:
    args = build_parser().parse_args(["nav", "--goal", "Find the price"])

    assert args.mode == "nav"
    assert args.goal == "Find the price"
