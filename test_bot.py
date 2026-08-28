"""Тесты логики, не требующие сети и токенов: pytest test_bot.py"""

import os

os.environ.setdefault("BOT_TOKEN", "1:test")
os.environ.setdefault("LLM_API_KEY", "test-key")

from bot import split_text  # noqa: E402
from history import InMemoryHistory  # noqa: E402


def test_short_text_not_split():
    assert split_text("привет") == ["привет"]


def test_long_text_split_within_limit():
    text = "\n\n".join(["абзац " * 50 for _ in range(40)])
    parts = split_text(text, limit=1000)
    assert len(parts) > 1
    assert all(len(p) <= 1000 for p in parts)


def test_unbroken_block_is_hard_cut():
    parts = split_text("x" * 5000, limit=1000)
    assert all(len(p) <= 1000 for p in parts)
    assert "".join(parts) == "x" * 5000


def test_history_respects_limit():
    h = InMemoryHistory(limit=4)
    for i in range(10):
        h.add(1, "user", str(i))
    msgs = h.get(1)
    assert len(msgs) == 4
    assert msgs[-1]["content"] == "9"


def test_history_is_per_chat():
    h = InMemoryHistory(limit=5)
    h.add(1, "user", "a")
    h.add(2, "user", "b")
    assert len(h.get(1)) == 1
    assert h.get(2)[0]["content"] == "b"
    h.clear(1)
    assert h.get(1) == []
    assert len(h.get(2)) == 1
