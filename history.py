"""Хранение истории диалога по чатам.

По умолчанию — в памяти процесса (перезапуск = чистая история).
Для персистентности подключите SQLite-реализацию: интерфейс тот же.
"""

from __future__ import annotations

import sqlite3
import time
from collections import defaultdict, deque
from pathlib import Path

Message = dict[str, str]


class InMemoryHistory:
    """Кольцевой буфер сообщений на каждый чат."""

    def __init__(self, limit: int = 20) -> None:
        self._limit = limit
        self._data: dict[int, deque[Message]] = defaultdict(
            lambda: deque(maxlen=limit)
        )

    def add(self, chat_id: int, role: str, content: str) -> None:
        self._data[chat_id].append({"role": role, "content": content})

    def get(self, chat_id: int) -> list[Message]:
        return list(self._data.get(chat_id, ()))

    def clear(self, chat_id: int) -> None:
        self._data.pop(chat_id, None)

    def chats(self) -> int:
        return len(self._data)


class SqliteHistory:
    """Персистентная история. Держит последние `limit` сообщений на чат."""

    def __init__(self, path: str | Path = "history.db", limit: int = 20) -> None:
        self._limit = limit
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id      INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id INTEGER NOT NULL,
                role    TEXT    NOT NULL,
                content TEXT    NOT NULL,
                ts      REAL    NOT NULL
            )
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_chat_id ON messages(chat_id, id)"
        )
        self._conn.commit()

    def add(self, chat_id: int, role: str, content: str) -> None:
        self._conn.execute(
            "INSERT INTO messages (chat_id, role, content, ts) VALUES (?, ?, ?, ?)",
            (chat_id, role, content, time.time()),
        )
        # обрезаем хвост, чтобы таблица не росла бесконечно
        self._conn.execute(
            """
            DELETE FROM messages
             WHERE chat_id = ?
               AND id NOT IN (
                   SELECT id FROM messages
                    WHERE chat_id = ?
                    ORDER BY id DESC
                    LIMIT ?
               )
            """,
            (chat_id, chat_id, self._limit),
        )
        self._conn.commit()

    def get(self, chat_id: int) -> list[Message]:
        rows = self._conn.execute(
            "SELECT role, content FROM messages WHERE chat_id = ? ORDER BY id",
            (chat_id,),
        ).fetchall()
        return [{"role": r, "content": c} for r, c in rows]

    def clear(self, chat_id: int) -> None:
        self._conn.execute("DELETE FROM messages WHERE chat_id = ?", (chat_id,))
        self._conn.commit()

    def chats(self) -> int:
        (n,) = self._conn.execute(
            "SELECT COUNT(DISTINCT chat_id) FROM messages"
        ).fetchone()
        return int(n)
