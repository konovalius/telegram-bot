"""Клиент к LLM. Любой OpenAI-совместимый провайдер: OpenRouter, Groq, Gemini,
Together, DeepSeek, Ollama, локальный vLLM — меняется только base_url и model.
"""

from __future__ import annotations

import logging

import httpx
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    AuthenticationError,
    RateLimitError,
)

log = logging.getLogger(__name__)


class LLMError(Exception):
    """Ошибка, текст которой безопасно показать пользователю."""


class LLMClient:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        *,
        max_tokens: int = 1200,
        temperature: float = 0.7,
        timeout: int = 90,
        proxy: str = "",
    ) -> None:
        # httpx-клиент с прокси нужен, если API провайдера недоступен напрямую
        http_client = httpx.AsyncClient(proxy=proxy, timeout=timeout) if proxy else None
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=2,
            http_client=http_client,
        )
        self.model = model
        self._max_tokens = max_tokens
        self._temperature = temperature

    async def complete(self, messages: list[dict[str, str]]) -> str:
        """Один запрос к модели. Возвращает текст ответа."""
        try:
            resp = await self._client.chat.completions.create(
                model=self.model,
                messages=messages,  # type: ignore[arg-type]
                max_tokens=self._max_tokens,
                temperature=self._temperature,
            )
        except AuthenticationError:
            raise LLMError(
                "Провайдер отклонил API-ключ. Проверьте LLM_API_KEY в .env."
            ) from None
        except RateLimitError:
            raise LLMError(
                "Лимит запросов у провайдера исчерпан. Подождите минуту "
                "или переключите модель через /model."
            ) from None
        except APITimeoutError:
            raise LLMError("Модель не ответила за отведённое время. Повторите запрос.") from None
        except APIConnectionError:
            raise LLMError("Не удалось связаться с API провайдера. Проверьте сеть.") from None
        except APIStatusError as exc:
            log.warning("LLM HTTP %s: %s", exc.status_code, exc.message)
            if exc.status_code == 404:
                raise LLMError(
                    f"Модель «{self.model}» недоступна у провайдера. "
                    "Проверьте LLM_MODEL или выберите другую через /model."
                ) from None
            raise LLMError(f"Провайдер вернул ошибку {exc.status_code}.") from None

        if not resp.choices:
            raise LLMError("Провайдер вернул пустой ответ.")

        text = (resp.choices[0].message.content or "").strip()
        if not text:
            raise LLMError(
                "Модель вернула пустой ответ. Возможно, сработал её фильтр — "
                "переформулируйте запрос."
            )
        return text


# Проверенные бесплатные варианты. Ключ — короткий алиас для команды /model.
FREE_PRESETS: dict[str, tuple[str, str, str]] = {
    # alias: (человекочитаемое имя, base_url, model id)
    "llama-8b": (
        "Llama 3.1 8B · Groq (быстрая)",
        "https://api.groq.com/openai/v1",
        "llama-3.1-8b-instant",
    ),
    "llama-70b": (
        "Llama 3.3 70B · Groq (умная)",
        "https://api.groq.com/openai/v1",
        "llama-3.3-70b-versatile",
    ),
    "gpt-oss": (
        "GPT-OSS 120B · Groq",
        "https://api.groq.com/openai/v1",
        "openai/gpt-oss-120b",
    ),
}
