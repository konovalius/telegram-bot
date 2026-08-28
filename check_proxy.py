"""Быстрая диагностика сети: проверяет доступность Telegram и LLM API.

Запуск:  python check_proxy.py
Читает TG_PROXY / LLM_PROXY из .env, если они заданы.
"""

from __future__ import annotations

import asyncio
import os

import httpx
from dotenv import load_dotenv

load_dotenv()

TARGETS = {
    "Telegram API": ("https://api.telegram.org", os.getenv("TG_PROXY", "").strip()),
    "LLM API": (
        os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1").strip(),
        os.getenv("LLM_PROXY", "").strip(),
    ),
}


async def probe(name: str, url: str, proxy: str) -> None:
    label = f"через прокси" if proxy else "напрямую"
    try:
        async with httpx.AsyncClient(
            proxy=proxy or None, timeout=15, follow_redirects=True
        ) as client:
            resp = await client.get(url)
        # любой HTTP-ответ означает, что сеть до хоста работает
        print(f"  [OK]   {name} ({label}) — соединение есть, код {resp.status_code}")
    except httpx.ProxyError as exc:
        print(f"  [FAIL] {name} — прокси не работает: {exc}")
    except (httpx.ConnectTimeout, httpx.ReadTimeout):
        print(f"  [FAIL] {name} ({label}) — таймаут, хост недоступен")
    except httpx.ConnectError as exc:
        print(f"  [FAIL] {name} ({label}) — не удалось подключиться: {exc}")
    except Exception as exc:  # noqa: BLE001
        print(f"  [FAIL] {name} ({label}) — {type(exc).__name__}: {exc}")


async def main() -> None:
    print("\nПроверка доступности сервисов:\n")
    await asyncio.gather(*(probe(n, u, p) for n, (u, p) in TARGETS.items()))
    print(
        "\nЕсли Telegram API [FAIL], а прокси не задан — укажите TG_PROXY в .env\n"
        "или включите VPN. Подробности в README, раздел «Если бот не подключается».\n"
    )


if __name__ == "__main__":
    asyncio.run(main())
