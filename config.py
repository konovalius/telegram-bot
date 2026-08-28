"""Конфигурация бота. Все значения читаются из переменных окружения / .env."""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Файл с системным промптом рядом с config.py. Удобнее .env для многострочных
# инструкций: правится в любом редакторе, не ломает разбор переменных.
PROMPT_FILE = Path(__file__).with_name("prompt.txt")


def _read_prompt_file() -> str:
    try:
        return PROMPT_FILE.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError):
        return ""


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    try:
        return int(raw) if raw else default
    except ValueError:
        return default


def _admin_ids() -> set[int]:
    raw = os.getenv("ADMIN_IDS", "")
    out: set[int] = set()
    for chunk in raw.replace(";", ",").split(","):
        chunk = chunk.strip()
        if chunk.lstrip("-").isdigit():
            out.add(int(chunk))
    return out


DEFAULT_SYSTEM_PROMPT = (
    "Ты полезный ассистент в Telegram. Отвечай кратко, по делу и на языке "
    "пользователя. Если не знаешь ответа — скажи прямо, не выдумывай факты."
)


@dataclass(frozen=True)
class Settings:
    bot_token: str
    api_key: str
    base_url: str
    model: str
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    # сколько сообщений (user+assistant) держим в контексте на чат
    history_limit: int = 20
    # максимум токенов в ответе модели
    max_tokens: int = 1200
    temperature: float = 0.7
    # таймаут запроса к LLM, секунды
    request_timeout: int = 90
    # антифлуд: минимальный интервал между запросами одного юзера, секунды
    rate_limit_seconds: float = 2.0
    # пустой set = бот открыт для всех
    allowed_user_ids: set[int] = field(default_factory=set)
    admin_ids: set[int] = field(default_factory=set)
    # прокси для api.telegram.org (в РФ обычно заблокирован)
    tg_proxy: str = ""
    # отдельный прокси для LLM API; пусто = прямое соединение
    llm_proxy: str = ""

    @property
    def provider_hint(self) -> str:
        host = self.base_url.split("//")[-1].split("/")[0]
        return host


def load_settings() -> Settings:
    token = os.getenv("BOT_TOKEN", "").strip()
    if not token:
        raise RuntimeError(
            "BOT_TOKEN не задан. Скопируйте .env.example в .env и заполните его."
        )

    api_key = os.getenv("LLM_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "LLM_API_KEY не задан. Получите бесплатный ключ у провайдера "
            "(OpenRouter / Groq / Google AI Studio) и укажите его в .env."
        )

    allowed: set[int] = set()
    for chunk in os.getenv("ALLOWED_USER_IDS", "").replace(";", ",").split(","):
        chunk = chunk.strip()
        if chunk.lstrip("-").isdigit():
            allowed.add(int(chunk))

    return Settings(
        bot_token=token,
        api_key=api_key,
        base_url=os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1").strip(),
        model=os.getenv("LLM_MODEL", "meta-llama/llama-3.3-70b-instruct:free").strip(),
        # приоритет: SYSTEM_PROMPT из .env → prompt.txt → встроенный дефолт
        system_prompt=(
            os.getenv("SYSTEM_PROMPT", "").strip()
            or _read_prompt_file()
            or DEFAULT_SYSTEM_PROMPT
        ),
        history_limit=_int_env("HISTORY_LIMIT", 20),
        max_tokens=_int_env("MAX_TOKENS", 1200),
        temperature=float(os.getenv("TEMPERATURE", "0.7") or 0.7),
        request_timeout=_int_env("REQUEST_TIMEOUT", 90),
        rate_limit_seconds=float(os.getenv("RATE_LIMIT_SECONDS", "2") or 2),
        allowed_user_ids=allowed,
        admin_ids=_admin_ids(),
        tg_proxy=os.getenv("TG_PROXY", "").strip(),
        llm_proxy=os.getenv("LLM_PROXY", "").strip(),
    )
