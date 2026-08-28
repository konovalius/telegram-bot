"""Telegram-бот на aiogram 3 + любой OpenAI-совместимый LLM API.

Запуск:  python bot.py
"""

from __future__ import annotations

import asyncio
import html
import logging
import time
from collections import defaultdict

from aiogram import Bot, Dispatcher, F
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ChatAction, ParseMode
from aiogram.filters import Command, CommandStart
from aiogram.types import BotCommand, Message
from aiogram.utils.keyboard import InlineKeyboardBuilder

from config import Settings, load_settings
from history import InMemoryHistory
from llm import FREE_PRESETS, LLMClient, LLMError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
log = logging.getLogger("bot")

# Telegram режет сообщения на 4096 символах
TG_LIMIT = 4000

settings: Settings = load_settings()
store = InMemoryHistory(limit=settings.history_limit)
llm = LLMClient(
    api_key=settings.api_key,
    base_url=settings.base_url,
    model=settings.model,
    max_tokens=settings.max_tokens,
    temperature=settings.temperature,
    timeout=settings.request_timeout,
    proxy=settings.llm_proxy,
)

dp = Dispatcher()

_last_request: dict[int, float] = defaultdict(float)
_busy: set[int] = set()
_counters = {"requests": 0, "errors": 0}


def allowed(user_id: int) -> bool:
    return not settings.allowed_user_ids or user_id in settings.allowed_user_ids


def split_text(text: str, limit: int = TG_LIMIT) -> list[str]:
    """Режет длинный текст по абзацам/строкам, не ломая код-блоки посередине."""
    if len(text) <= limit:
        return [text]

    parts: list[str] = []
    current = ""
    for block in text.split("\n\n"):
        candidate = f"{current}\n\n{block}" if current else block
        if len(candidate) <= limit:
            current = candidate
            continue
        if current:
            parts.append(current)
        # сам блок больше лимита — рубим по строкам
        while len(block) > limit:
            cut = block.rfind("\n", 0, limit)
            if cut <= 0:
                cut = limit
            parts.append(block[:cut])
            block = block[cut:].lstrip("\n")
        current = block
    if current:
        parts.append(current)
    return parts


async def typing_loop(bot: Bot, chat_id: int) -> None:
    """Держит индикатор «печатает…», пока модель думает."""
    try:
        while True:
            await bot.send_chat_action(chat_id, ChatAction.TYPING)
            await asyncio.sleep(4.5)
    except asyncio.CancelledError:
        pass


@dp.message(CommandStart())
async def cmd_start(message: Message) -> None:
    if not allowed(message.from_user.id):
        await message.answer("Извините, у вас нет доступа к этому боту.")
        return
    store.clear(message.chat.id)
    await message.answer(
        "Привет. Я бот с подключённой языковой моделью — просто напишите "
        "вопрос обычным сообщением.\n\n"
        "<b>Команды</b>\n"
        "/new — очистить контекст диалога\n"
        "/model — сменить модель\n"
        "/help — что я умею\n\n"
        f"Текущая модель: <code>{html.escape(llm.model)}</code>"
    )


@dp.message(Command("help"))
async def cmd_help(message: Message) -> None:
    await message.answer(
        "Я передаю ваши сообщения языковой модели и возвращаю ответ. "
        f"Помню последние {settings.history_limit} сообщений диалога, поэтому "
        "можно задавать уточняющие вопросы.\n\n"
        "/new — сбросить историю, если модель «зацикливается» на старой теме\n"
        "/model — переключить модель на лету\n"
        "/stats — статистика (для админа)\n\n"
        "Длинные ответы приходят несколькими сообщениями."
    )


@dp.message(Command("new", "reset", "clear"))
async def cmd_new(message: Message) -> None:
    store.clear(message.chat.id)
    await message.answer("Контекст очищен. Начинаем с чистого листа.")


@dp.message(Command("model"))
async def cmd_model(message: Message) -> None:
    builder = InlineKeyboardBuilder()
    for alias, (title, _url, _mid) in FREE_PRESETS.items():
        builder.button(text=title, callback_data=f"m:{alias}")
    builder.adjust(1)
    await message.answer(
        f"Активная модель: <code>{html.escape(llm.model)}</code>\n"
        f"Провайдер: <code>{html.escape(settings.provider_hint)}</code>\n\n"
        "Выберите другую. Учтите: у каждого провайдера свой ключ, "
        "и переключение сработает, только если ключ в .env подходит "
        "выбранному сервису.",
        reply_markup=builder.as_markup(),
    )


@dp.callback_query(F.data.startswith("m:"))
async def switch_model(callback) -> None:
    alias = callback.data.split(":", 1)[1]
    preset = FREE_PRESETS.get(alias)
    if not preset:
        await callback.answer("Неизвестная модель")
        return
    title, base_url, model_id = preset
    global llm
    llm = LLMClient(
        api_key=settings.api_key,
        base_url=base_url,
        model=model_id,
        max_tokens=settings.max_tokens,
        temperature=settings.temperature,
        timeout=settings.request_timeout,
        proxy=settings.llm_proxy,
    )
    store.clear(callback.message.chat.id)
    await callback.message.edit_text(
        f"Переключено на <b>{html.escape(title)}</b>\n"
        f"<code>{html.escape(model_id)}</code>\n\nКонтекст сброшен."
    )
    await callback.answer("Готово")


@dp.message(Command("stats"))
async def cmd_stats(message: Message) -> None:
    if message.from_user.id not in settings.admin_ids:
        return
    await message.answer(
        f"Запросов: {_counters['requests']}\n"
        f"Ошибок: {_counters['errors']}\n"
        f"Активных чатов в памяти: {store.chats()}\n"
        f"Модель: <code>{html.escape(llm.model)}</code>"
    )


@dp.message(F.text & ~F.text.startswith("/"))
async def on_text(message: Message, bot: Bot) -> None:
    user_id = message.from_user.id
    chat_id = message.chat.id

    if not allowed(user_id):
        await message.answer("Извините, у вас нет доступа к этому боту.")
        return

    # не даём одному пользователю запускать несколько генераций параллельно
    if user_id in _busy:
        await message.answer("Дождитесь предыдущего ответа.")
        return

    elapsed = time.monotonic() - _last_request[user_id]
    if elapsed < settings.rate_limit_seconds:
        await message.answer(
            f"Слишком часто. Подождите "
            f"{settings.rate_limit_seconds - elapsed:.0f} с."
        )
        return
    _last_request[user_id] = time.monotonic()

    payload = [{"role": "system", "content": settings.system_prompt}]
    payload += store.get(chat_id)
    payload.append({"role": "user", "content": message.text})

    _busy.add(user_id)
    typing = asyncio.create_task(typing_loop(bot, chat_id))
    try:
        _counters["requests"] += 1
        answer = await llm.complete(payload)
    except LLMError as exc:
        _counters["errors"] += 1
        await message.answer(f"⚠️ {html.escape(str(exc))}")
        return
    except Exception:
        _counters["errors"] += 1
        log.exception("Необработанная ошибка при генерации")
        await message.answer("⚠️ Внутренняя ошибка. Попробуйте ещё раз.")
        return
    finally:
        typing.cancel()
        _busy.discard(user_id)

    store.add(chat_id, "user", message.text)
    store.add(chat_id, "assistant", answer)

    for chunk in split_text(answer):
        # Markdown от модели бывает невалидным для Telegram — не падаем из-за него
        try:
            await message.answer(chunk, parse_mode=ParseMode.MARKDOWN)
        except Exception:
            await message.answer(chunk, parse_mode=None)


@dp.message()
async def on_other(message: Message) -> None:
    await message.answer("Я работаю только с текстом. Напишите сообщение словами.")


async def main() -> None:
    bot = Bot(
        token=settings.bot_token,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )
    await bot.set_my_commands(
        [
            BotCommand(command="new", description="Очистить контекст"),
            BotCommand(command="model", description="Сменить модель"),
            BotCommand(command="help", description="Справка"),
        ]
    )
    me = await bot.get_me()
    log.info("Бот @%s запущен. Модель: %s (%s)", me.username, llm.model, settings.provider_hint)
    # на случай зависшего вебхука от прошлых запусков
    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot, allowed_updates=dp.resolve_used_update_types())


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        log.info("Остановлено пользователем")
