import json
import logging
import os
import re
import threading
from logging.handlers import RotatingFileHandler
from pathlib import Path

import telebot
from telebot import types

from main import RAGAgent, extract_urls, extract_user_facts, format_sources, hit_source

_TELEGRAM_LIMIT = 4096
_agent_lock = threading.Lock()
_pending: dict[int, str] = {}
_pending_lock = threading.Lock()
_PENDING_PROMPTS = {
    "search": "Пришлите поисковый запрос следующим сообщением.\nОтмена — /cancel",
    "add_url": "Пришлите ссылку на страницу, которую нужно добавить в базу знаний.\nОтмена — /cancel",
    "add_text": "Пришлите текст, который нужно сохранить в базу знаний.\nОтмена — /cancel",
}
_LOG_DIR = Path(os.getenv("LOG_DIR", Path(__file__).resolve().parent / "logs"))
_QUESTION = re.compile(
    r"(?i)(?:\?|^(?:что|как|где|когда|почему|зачем|кто|какой|какая|какие|сколько|можно|нужно)\b)"
)


def _setup_logging() -> tuple[logging.Logger, logging.Logger]:
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    logger = logging.getLogger("vpg04.bot")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        bot_file = RotatingFileHandler(
            _LOG_DIR / "bot.log",
            maxBytes=2_000_000,
            backupCount=5,
            encoding="utf-8",
        )
        bot_file.setFormatter(formatter)
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        logger.addHandler(bot_file)
        logger.addHandler(console)

    chat_logger = logging.getLogger("vpg04.chat")
    chat_logger.setLevel(logging.INFO)
    chat_logger.propagate = False
    if not chat_logger.handlers:
        chat_file = RotatingFileHandler(
            _LOG_DIR / "chat.log",
            maxBytes=2_000_000,
            backupCount=5,
            encoding="utf-8",
        )
        chat_file.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        chat_logger.addHandler(chat_file)
    return logger, chat_logger


def _log_turn(
    chat_logger: logging.Logger,
    message: types.Message,
    incoming: str,
    outgoing: str,
) -> None:
    user = message.from_user
    user_id = user.id if user else "-"
    username = user.username if user and user.username else "-"
    chat_logger.info(
        "chat=%s user=%s name=%s\nUSER: %s\nBOT: %s",
        message.chat.id,
        user_id,
        username,
        incoming,
        outgoing,
    )


def _chunks(text: str) -> list[str]:
    if len(text) <= _TELEGRAM_LIMIT:
        return [text]
    parts: list[str] = []
    rest = text
    while rest:
        parts.append(rest[:_TELEGRAM_LIMIT])
        rest = rest[_TELEGRAM_LIMIT:]
    return parts


def _send(bot: telebot.TeleBot, chat_id: int, text: str, reply_to: int | None = None) -> None:
    parts = _chunks(text or "Пустой ответ.")
    for index, part in enumerate(parts):
        if index == 0 and reply_to is not None:
            bot.send_message(chat_id, part, reply_to_message_id=reply_to)
        else:
            bot.send_message(chat_id, part)


class ActivityStore:
    """Локальная статистика и факты профиля. Не заменяет векторную базу."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()

    def _load(self) -> dict:
        if not self.path.exists():
            return {}
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def _save(self, data: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def _user(self, data: dict, user_id: str) -> dict:
        record = data.setdefault(
            user_id,
            {"facts": [], "messages": 0, "notes": 0, "urls": 0, "searches": 0},
        )
        record.setdefault("facts", [])
        return record

    def bump(self, user_id: str, field: str, amount: int = 1) -> None:
        with self._lock:
            data = self._load()
            record = self._user(data, user_id)
            record[field] = int(record.get(field, 0)) + amount
            self._save(data)

    def add_facts(self, user_id: str, facts: list[str]) -> None:
        with self._lock:
            data = self._load()
            record = self._user(data, user_id)
            known = {item.casefold() for item in record["facts"]}
            for fact in facts:
                if fact.casefold() not in known:
                    record["facts"].append(fact)
                    known.add(fact.casefold())
            self._save(data)

    def snapshot(self, user_id: str) -> dict:
        with self._lock:
            data = self._load()
            record = self._user(data, user_id)
            return {
                "facts": list(record["facts"]),
                "messages": int(record.get("messages", 0)),
                "notes": int(record.get("notes", 0)),
                "urls": int(record.get("urls", 0)),
                "searches": int(record.get("searches", 0)),
            }

    def forget(self, user_id: str) -> None:
        with self._lock:
            data = self._load()
            data.pop(user_id, None)
            self._save(data)


def _set_pending(user_id: int, action: str) -> None:
    with _pending_lock:
        _pending[user_id] = action


def _take_pending(user_id: int) -> str | None:
    with _pending_lock:
        return _pending.pop(user_id, None)


def _clear_pending(user_id: int | None) -> None:
    if user_id is None:
        return
    with _pending_lock:
        _pending.pop(user_id, None)


def _command_arg(message: types.Message) -> str:
    parts = (message.text or "").split(maxsplit=1)
    return parts[1].strip() if len(parts) > 1 else ""


def _engagement_level(messages: int) -> str:
    if messages < 5:
        return "низкий"
    if messages < 20:
        return "средний"
    return "высокий"


def _help_text() -> str:
    return (
        "🔍 Поиск и вопросы:\n"
        "• Просто напиши вопрос — я найду ответ в базе знаний\n"
        "• /search <запрос> — поиск по базе данных\n\n"
        "📚 Управление знаниями:\n"
        "• /add_url <URL> — добавить страницу в базу знаний\n"
        "• /add_text <текст> — добавить текст в базу знаний\n\n"
        "👤 Профиль:\n"
        "• /profile — посмотреть свой профиль\n"
        "• /summary — краткое резюме о вас\n"
        "• /suggestions — предложения вопросов для изучения\n"
        "• /engagement — уровень вашей активности\n"
        "• /forget_me — удалить всю информацию о себе\n\n"
        "ℹ️ Другое:\n"
        "• /help — это сообщение\n\n"
        "💡 Я автоматически запоминаю важную информацию о тебе из наших разговоров!"
    )


def _memory_only(text: str, facts: list[str]) -> bool:
    return bool(facts) and not extract_urls(text) and not _QUESTION.search(text)


def _reply_for(
    agent: RAGAgent,
    activity: ActivityStore,
    user_id: int,
    text: str,
) -> str:
    user_key = str(user_id)
    facts = extract_user_facts(text)
    if facts:
        agent.remember(user_key, facts)
        activity.add_facts(user_key, facts)
    if _memory_only(text, facts):
        lines = "\n".join(f"— {fact}" for fact in facts)
        return f"Запомнил:\n{lines}"

    notes: list[str] = []
    urls = extract_urls(text)
    page_sources: list[str] = []
    if urls:
        activity.bump(user_key, "urls", len(urls))
    for url in urls:
        try:
            notes.append(agent.ingest_url(url))
            page_sources.append(f"интернет: {url}")
        except Exception as exc:
            notes.append(f"Не удалось прочитать {url}: {exc}")

    if page_sources:
        blocks: list[str] = []
        for url in urls:
            if f"интернет: {url}" not in page_sources:
                continue
            for doc in agent.chunks_from_url(url):
                blocks.append(" ".join(doc.page_content.split())[:1200])
        context = "\n\n".join(blocks)
        agent.prefetched_sources = []
        if not context:
            context = agent.retrieve_context(str(user_id), text)
    else:
        context = agent.retrieve_context(str(user_id), text)
    answer = agent.answer(
        text,
        context=context,
        user_id=str(user_id),
        extra_sources=page_sources,
    )
    prefix = ""
    if facts:
        prefix += "Запомнил: " + "; ".join(facts) + "\n\n"
    if notes:
        prefix += "\n".join(notes) + "\n\n"
    return (prefix + (answer or "Не удалось собрать ответ.")).strip()


def _finish(
    bot: telebot.TeleBot,
    chat_logger: logging.Logger,
    message: types.Message,
    reply: str,
    status: types.Message | None = None,
) -> None:
    _log_turn(chat_logger, message, message.text or "", reply)
    parts = _chunks(reply)
    if status is not None:
        try:
            bot.edit_message_text(parts[0], message.chat.id, status.message_id)
            for part in parts[1:]:
                bot.send_message(message.chat.id, part)
            return
        except Exception:
            pass
    _send(bot, message.chat.id, reply, reply_to=message.message_id)


def _profile_facts(agent: RAGAgent, activity: ActivityStore, user_id: str) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for fact in activity.snapshot(user_id)["facts"] + agent.user_memories(user_id):
        key = fact.casefold()
        if key not in seen:
            seen.add(key)
            merged.append(fact)
    return merged


def main() -> None:
    logger, chat_logger = _setup_logging()
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        logger.error("TELEGRAM_BOT_TOKEN is not set")
        raise RuntimeError("Задайте переменную окружения TELEGRAM_BOT_TOKEN")
    agent = RAGAgent()
    activity = ActivityStore(_LOG_DIR / "activity.json")
    bot = telebot.TeleBot(token, threaded=True)
    bot.set_my_commands(
        [
            types.BotCommand("search", "Поиск по базе данных"),
            types.BotCommand("add_url", "Добавить страницу в базу знаний"),
            types.BotCommand("add_text", "Добавить текст в базу знаний"),
            types.BotCommand("profile", "Посмотреть свой профиль"),
            types.BotCommand("summary", "Краткое резюме о вас"),
            types.BotCommand("suggestions", "Вопросы для изучения"),
            types.BotCommand("engagement", "Уровень вашей активности"),
            types.BotCommand("forget_me", "Удалить информацию о себе"),
            types.BotCommand("help", "Меню команд"),
        ]
    )

    def _require_user(message: types.Message) -> int | None:
        if message.from_user is None:
            return None
        return message.from_user.id

    def _begin_command(message: types.Message) -> int | None:
        user_id = _require_user(message)
        _clear_pending(user_id)
        return user_id

    def _ask_input(message: types.Message, user_id: int, action: str) -> None:
        _set_pending(user_id, action)
        _finish(bot, chat_logger, message, _PENDING_PROMPTS[action])

    def _run_search(message: types.Message, user_id: int, query: str) -> None:
        activity.bump(str(user_id), "messages")
        activity.bump(str(user_id), "searches")
        bot.send_chat_action(message.chat.id, "typing")
        status = bot.send_message(message.chat.id, "Ищу в базе…")
        try:
            with _agent_lock:
                hits = agent.search(
                    query,
                    k=5,
                    metadata_filter={
                        "$or": [
                            {"user_id": str(user_id)},
                            {"kind": "page"},
                            {"kind": {"$exists": False}},
                        ]
                    },
                )
        except Exception:
            logger.exception("filtered search failed chat=%s", message.chat.id)
            try:
                with _agent_lock:
                    hits = agent.search(query, k=5)
            except Exception as exc:
                logger.exception("search failed chat=%s", message.chat.id)
                _finish(bot, chat_logger, message, f"Поиск не выполнен: {exc}", status)
                return
        if not hits:
            reply = f"По запросу «{query}» ничего не найдено." + format_sources(["Pinecone"])
        else:
            lines = [f"Нашёл фрагментов: {len(hits)}. Запрос: {query}"]
            sources: list[str] = []
            for index, doc in enumerate(hits, start=1):
                source = doc.metadata.get("source", "unknown")
                label = hit_source(doc.metadata)
                if label not in sources:
                    sources.append(label)
                snippet = " ".join(doc.page_content.split())[:300]
                lines.append(f"{index}. {source}\n{snippet}")
            reply = "\n\n".join(lines) + format_sources(sources)
        logger.info("search chat=%s hits=%s", message.chat.id, len(hits))
        _finish(bot, chat_logger, message, reply, status)

    def _run_add_url(message: types.Message, user_id: int, raw: str) -> None:
        urls = extract_urls(raw)
        if not urls:
            _set_pending(user_id, "add_url")
            _finish(
                bot,
                chat_logger,
                message,
                "Не вижу ссылки. Пришлите адрес, например https://example.com\nОтмена — /cancel",
            )
            return
        activity.bump(str(user_id), "messages")
        activity.bump(str(user_id), "urls", len(urls))
        bot.send_chat_action(message.chat.id, "typing")
        status = bot.send_message(message.chat.id, "Читаю страницу и записываю в базу…")
        notes: list[str] = []
        try:
            with _agent_lock:
                for url in urls:
                    notes.append(agent.ingest_url(url))
            reply = "\n".join(notes)
            logger.info("add_url chat=%s urls=%s", message.chat.id, len(urls))
        except Exception as exc:
            logger.exception("add_url failed chat=%s", message.chat.id)
            reply = f"Не удалось добавить страницу: {exc}"
        _finish(bot, chat_logger, message, reply, status)

    def _run_add_text(message: types.Message, user_id: int, payload: str) -> None:
        activity.bump(str(user_id), "messages")
        bot.send_chat_action(message.chat.id, "typing")
        try:
            with _agent_lock:
                count = agent.add_note(str(user_id), payload)
            activity.bump(str(user_id), "notes")
            reply = f"Записал текст в базу, чанков: {count}."
            logger.info("add_text chat=%s chunks=%s", message.chat.id, count)
        except Exception as exc:
            logger.exception("add_text failed chat=%s", message.chat.id)
            reply = f"Не удалось сохранить: {exc}"
        _finish(bot, chat_logger, message, reply)

    @bot.message_handler(commands=["start", "help"])
    def help_command(message: types.Message) -> None:
        user_id = _begin_command(message)
        if user_id is not None:
            activity.bump(str(user_id), "messages")
        logger.info("help chat=%s", message.chat.id)
        _finish(bot, chat_logger, message, _help_text())

    @bot.message_handler(commands=["cancel"])
    def cancel_command(message: types.Message) -> None:
        user_id = _require_user(message)
        if user_id is None:
            return
        waiting = _take_pending(user_id)
        if waiting:
            _finish(bot, chat_logger, message, "Отменил. Можно выбрать другую команду или просто задать вопрос.")
        else:
            _finish(bot, chat_logger, message, "Сейчас нечего отменять.")

    @bot.message_handler(commands=["search"])
    def search_command(message: types.Message) -> None:
        user_id = _begin_command(message)
        query = _command_arg(message)
        if user_id is None:
            return
        if not query:
            _ask_input(message, user_id, "search")
            return
        _run_search(message, user_id, query)

    @bot.message_handler(commands=["add_url"])
    def add_url_command(message: types.Message) -> None:
        user_id = _begin_command(message)
        if user_id is None:
            return
        raw = _command_arg(message)
        if not raw:
            _ask_input(message, user_id, "add_url")
            return
        _run_add_url(message, user_id, raw)

    @bot.message_handler(commands=["add_text", "add"])
    def add_text_command(message: types.Message) -> None:
        user_id = _begin_command(message)
        if user_id is None:
            return
        payload = _command_arg(message)
        if not payload:
            _ask_input(message, user_id, "add_text")
            return
        _run_add_text(message, user_id, payload)

    @bot.message_handler(commands=["profile"])
    def profile_command(message: types.Message) -> None:
        user_id = _begin_command(message)
        if user_id is None:
            return
        activity.bump(str(user_id), "messages")
        bot.send_chat_action(message.chat.id, "typing")
        try:
            with _agent_lock:
                facts = _profile_facts(agent, activity, str(user_id))
            stats = activity.snapshot(str(user_id))
        except Exception as exc:
            logger.exception("profile failed chat=%s", message.chat.id)
            _finish(bot, chat_logger, message, f"Не удалось открыть профиль: {exc}")
            return
        if facts:
            fact_lines = "\n".join(f"— {fact}" for fact in facts)
        else:
            fact_lines = "Пока нет сохранённых фактов."
        reply = (
            "👤 Профиль\n"
            f"{fact_lines}\n\n"
            f"Сообщений: {stats['messages']}\n"
            f"Заметок: {stats['notes']}\n"
            f"Страниц: {stats['urls']}\n"
            f"Поисков: {stats['searches']}"
            + format_sources(["профиль", "Pinecone: факты о пользователе"])
        )
        _finish(bot, chat_logger, message, reply)

    @bot.message_handler(commands=["summary"])
    def summary_command(message: types.Message) -> None:
        user_id = _begin_command(message)
        if user_id is None:
            return
        activity.bump(str(user_id), "messages")
        bot.send_chat_action(message.chat.id, "typing")
        status = bot.send_message(message.chat.id, "Собираю резюме…")
        try:
            with _agent_lock:
                facts = _profile_facts(agent, activity, str(user_id))
                if not facts:
                    reply = "Пока я ничего о вас не запомнил."
                else:
                    reply = agent.brief_answer(
                        "Сделай краткое резюме о человеке по этим фактам. "
                        "3–5 предложений, по-русски, без выдуманных деталей. "
                        "Не добавляй строку «Источник»:\n"
                        + "\n".join(f"— {fact}" for fact in facts)
                    )
                    reply = (
                        (reply or "Пустое резюме.")
                        + format_sources(["профиль", "Pinecone: факты о пользователе"])
                    ).strip()
        except Exception as exc:
            logger.exception("summary failed chat=%s", message.chat.id)
            reply = f"Не удалось собрать резюме: {exc}"
        _finish(bot, chat_logger, message, reply or "Пустое резюме.", status)

    @bot.message_handler(commands=["suggestions"])
    def suggestions_command(message: types.Message) -> None:
        user_id = _begin_command(message)
        if user_id is None:
            return
        activity.bump(str(user_id), "messages")
        bot.send_chat_action(message.chat.id, "typing")
        status = bot.send_message(message.chat.id, "Подбираю вопросы…")
        try:
            with _agent_lock:
                context = agent.retrieve_context(str(user_id), "основные темы базы знаний")
                material_sources = agent.take_prefetched_sources()
                facts = _profile_facts(agent, activity, str(user_id))
                reply = agent.brief_answer(
                    "Предложи 5 коротких вопросов для изучения. "
                    "Опирайся только на материалы и факты ниже. "
                    "Если их мало, предложи вопросы, которые помогут пополнить базу. "
                    "Ответ по-русски, нумерованным списком. Не добавляй строку «Источник».\n\n"
                    f"Материалы:\n{context or 'пока пусто'}\n\n"
                    f"Факты о пользователе:\n{chr(10).join(facts) or 'пока нет'}"
                )
                sources = list(material_sources)
                if facts and "профиль" not in sources:
                    sources.append("профиль")
                if not sources:
                    sources.append("модель: без базы и внешних источников")
                reply = ((reply or "Не получилось составить вопросы.") + format_sources(sources)).strip()
        except Exception as exc:
            logger.exception("suggestions failed chat=%s", message.chat.id)
            reply = f"Не удалось предложить вопросы: {exc}"
        _finish(bot, chat_logger, message, reply or "Не получилось составить вопросы.", status)

    @bot.message_handler(commands=["engagement"])
    def engagement_command(message: types.Message) -> None:
        user_id = _begin_command(message)
        if user_id is None:
            return
        activity.bump(str(user_id), "messages")
        stats = activity.snapshot(str(user_id))
        level = _engagement_level(stats["messages"])
        reply = (
            "📈 Активность\n"
            f"Уровень: {level}\n"
            f"Сообщений: {stats['messages']}\n"
            f"Поисков: {stats['searches']}\n"
            f"Добавлено страниц: {stats['urls']}\n"
            f"Добавлено текстов: {stats['notes']}"
        )
        _finish(bot, chat_logger, message, reply)

    @bot.message_handler(commands=["forget_me"])
    def forget_command(message: types.Message) -> None:
        user_id = _begin_command(message)
        if user_id is None:
            return
        bot.send_chat_action(message.chat.id, "typing")
        try:
            with _agent_lock:
                agent.forget_user(str(user_id))
            activity.forget(str(user_id))
            reply = (
                "Удалил факты, заметки и статистику профиля. "
                "Общие страницы базы, добавленные по ссылкам, остались."
            )
            logger.info("forget_me chat=%s user=%s", message.chat.id, user_id)
        except Exception as exc:
            logger.exception("forget_me failed chat=%s", message.chat.id)
            activity.forget(str(user_id))
            reply = (
                "Локальный профиль очищен. "
                f"Векторную базу удалить не удалось: {exc}"
            )
        _finish(bot, chat_logger, message, reply)

    @bot.message_handler(content_types=["text"])
    def on_text(message: types.Message) -> None:
        text = (message.text or "").strip()
        user_id = _require_user(message)
        if not text or user_id is None or text.startswith("/"):
            return
        pending = _take_pending(user_id)
        if pending == "search":
            _run_search(message, user_id, text)
            return
        if pending == "add_url":
            if not extract_urls(text):
                _set_pending(user_id, "add_url")
                _finish(
                    bot,
                    chat_logger,
                    message,
                    "Не вижу ссылки. Пришлите адрес, например https://example.com\nОтмена — /cancel",
                )
                return
            _run_add_url(message, user_id, text)
            return
        if pending == "add_text":
            _run_add_text(message, user_id, text)
            return
        activity.bump(str(user_id), "messages")
        bot.send_chat_action(message.chat.id, "typing")
        status = bot.send_message(message.chat.id, "Ищу в базе и готовлю ответ…")
        try:
            with _agent_lock:
                reply = _reply_for(agent, activity, user_id, text)
            logger.info("reply chat=%s chars=%s", message.chat.id, len(reply))
        except Exception as exc:
            logger.exception("reply failed chat=%s", message.chat.id)
            reply = f"Не получилось ответить: {exc}"
        _finish(bot, chat_logger, message, reply, status)

    logger.info("Telegram bot is running. Logs: %s", _LOG_DIR)
    bot.infinity_polling(skip_pending=True, timeout=30)


if __name__ == "__main__":
    main()
