"""
Telegram-бот: Генератор ответов на типовые вопросы преподавателя.
Поддерживает 3 движка (выбор через переменную ENGINE_TYPE):
  - tfidf           (по умолчанию)
  - tfidf_synonyms
  - embeddings

Inline-навигация:
  - При ненайденном ответе — предлагает похожие вопросы
  - Кнопка «Категории» — каталог всех категорий
  - Из категории — список вопросов с ответами
  - Из ответа — кнопки «Назад» и «Похожие вопросы»
"""

import os
import logging
import numpy as np
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    filters,
    ContextTypes,
)
from nlp_utils import load_knowledge_base, DATA_PATH
from engines import TfidfEngine, TfidfSynonymEngine, AdvancedTfidfEngine

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

ENGINE_MAP = {
    "tfidf": TfidfEngine,
    "tfidf_synonyms": TfidfSynonymEngine,
    "advanced_tfidf": AdvancedTfidfEngine,
    "embeddings": AdvancedTfidfEngine,
}

qa_data = load_knowledge_base(DATA_PATH)
engine_type = os.environ.get("ENGINE_TYPE", "tfidf_synonyms")
EngineClass = ENGINE_MAP.get(engine_type)
if not EngineClass:
    raise ValueError(f"Неизвестный движок: {engine_type}. Допустимые: {list(ENGINE_MAP.keys())}")

logger.info("Инициализация движка: %s", engine_type)
engine = EngineClass(qa_data)
logger.info("Движок %s готов.", engine.name)

# Количество похожих вопросов при неудачном поиске
TOP_SUGGESTIONS = 5
# Количество вопросов в категории на одной странице
PAGE_SIZE = 8


def get_categories() -> list[str]:
    return sorted(set(e.get("category", "—") for e in engine.entries))


def get_questions_by_category(category: str) -> list[dict]:
    return [e for e in engine.entries if e.get("category") == category]


def get_entry_by_id(entry_id: int) -> dict | None:
    for e in engine.entries:
        if e["id"] == entry_id:
            return e
    return None


def get_top_suggestions(user_text: str, exclude_id: int | None = None) -> list[dict]:
    """Возвращает TOP_SUGGESTIONS наиболее похожих вопросов из базы."""
    from sklearn.metrics.pairwise import cosine_similarity
    from nlp_utils import lemmatize

    use_syn = hasattr(engine, "lemmatized")
    lem = lemmatize(user_text, use_synonyms=use_syn)
    if not lem.strip():
        return engine.entries[:TOP_SUGGESTIONS]

    # Используем основной векторизатор движка
    if hasattr(engine, "word_vec"):
        vec = engine.word_vec.transform([lem])
        sims = cosine_similarity(vec, engine.word_matrix)[0]
    else:
        vec = engine.vectorizer.transform([lem])
        sims = cosine_similarity(vec, engine.matrix)[0]

    indexed = sorted(enumerate(sims), key=lambda x: x[1], reverse=True)
    result = []
    for idx, score in indexed:
        entry = engine.entries[idx]
        if exclude_id and entry["id"] == exclude_id:
            continue
        result.append(entry)
        if len(result) >= TOP_SUGGESTIONS:
            break
    return result


def get_similar_in_category(category: str, exclude_id: int) -> list[dict]:
    """Возвращает другие вопросы той же категории (без текущего)."""
    entries = [e for e in engine.entries if e.get("category") == category and e["id"] != exclude_id]
    return entries[:TOP_SUGGESTIONS]


# ── Клавиатуры ──────────────────────────────────────────────────

def kb_categories(back_label: str | None = None) -> InlineKeyboardMarkup:
    cats = get_categories()
    buttons = []
    for cat in cats:
        count = len(get_questions_by_category(cat))
        buttons.append([InlineKeyboardButton(f"📂 {cat} ({count})", callback_data=f"cat:{cat}")])
    if back_label:
        buttons.append([InlineKeyboardButton("⬅️ Назад", callback_data=back_label)])
    return InlineKeyboardMarkup(buttons)


def kb_questions_in_category(category: str, page: int = 0) -> InlineKeyboardMarkup:
    questions = get_questions_by_category(category)
    total = len(questions)
    start = page * PAGE_SIZE
    end = min(start + PAGE_SIZE, total)
    page_items = questions[start:end]

    buttons = []
    for e in page_items:
        label = e["question"][:60] + ("…" if len(e["question"]) > 60 else "")
        buttons.append([InlineKeyboardButton(label, callback_data=f"q:{e['id']}:cat:{category}")])

    nav = []
    if page > 0:
        nav.append(InlineKeyboardButton("◀️", callback_data=f"cat_page:{category}:{page - 1}"))
    if end < total:
        nav.append(InlineKeyboardButton("▶️", callback_data=f"cat_page:{category}:{page + 1}"))
    if nav:
        buttons.append(nav)

    buttons.append([InlineKeyboardButton("⬅️ К категориям", callback_data="show_categories")])
    return InlineKeyboardMarkup(buttons)


def kb_after_answer(entry_id: int, category: str, back_data: str) -> InlineKeyboardMarkup:
    buttons = [
        [InlineKeyboardButton("🔁 Похожие вопросы", callback_data=f"similar:{entry_id}:{category}")],
        [
            InlineKeyboardButton("⬅️ Назад", callback_data=back_data),
            InlineKeyboardButton("📋 Категории", callback_data="show_categories"),
        ],
    ]
    return InlineKeyboardMarkup(buttons)


def kb_suggestions(suggestions: list[dict]) -> InlineKeyboardMarkup:
    buttons = []
    for e in suggestions:
        label = e["question"][:60] + ("…" if len(e["question"]) > 60 else "")
        buttons.append([InlineKeyboardButton(label, callback_data=f"q:{e['id']}:suggest")])
    buttons.append([InlineKeyboardButton("📋 Все категории", callback_data="show_categories")])
    return InlineKeyboardMarkup(buttons)


def kb_similar(similar: list[dict], category: str) -> InlineKeyboardMarkup:
    buttons = []
    for e in similar:
        label = e["question"][:60] + ("…" if len(e["question"]) > 60 else "")
        buttons.append([InlineKeyboardButton(label, callback_data=f"q:{e['id']}:cat:{category}")])
    buttons.append([
        InlineKeyboardButton("⬅️ К категории", callback_data=f"cat:{category}"),
        InlineKeyboardButton("📋 Категории", callback_data="show_categories"),
    ])
    return InlineKeyboardMarkup(buttons)


# ── Команды ─────────────────────────────────────────────────────

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Привет! Я бот-помощник РГСУ.\n\n"
        "Задайте вопрос текстом или выберите категорию ниже.",
        reply_markup=kb_categories(),
    )


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "ℹ️ Напишите вопрос обычным текстом.\n"
        "Бот найдёт ответ или предложит похожие вопросы.\n\n"
        "Примеры:\n"
        "• Когда дедлайн курсовой?\n"
        "• Где взять методичку?\n"
        "• Сколько баллов на автомат?\n"
        "• Что будет на экзамене?"
    )


async def cmd_stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    cats = {}
    for e in engine.entries:
        c = e.get("category", "Без категории")
        cats[c] = cats.get(c, 0) + 1
    lines = [
        "📊 Статистика базы знаний:\n",
        f"Движок: {engine.name}",
        f"Всего вопросов: {len(engine.entries)}",
        "\nКатегории:",
    ]
    for cat, count in sorted(cats.items()):
        lines.append(f"  • {cat}: {count}")
    await update.message.reply_text("\n".join(lines))


async def cmd_categories(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "📂 Выберите категорию:",
        reply_markup=kb_categories(),
    )


# ── Обработка текстовых сообщений ───────────────────────────────

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text.strip()
    if not user_text:
        return

    logger.info("Вопрос от %s: %s", update.effective_user.first_name, user_text)
    try:
        result = engine.find_answer(user_text)
    except Exception:
        logger.exception("Ошибка движка при обработке вопроса: %s", user_text)
        await update.message.reply_text(
            "⚠️ Произошла ошибка при поиске ответа. Попробуйте переформулировать вопрос.")
        return

    score = result["score"]

    if result["matched_question"]:
        emoji = "🟢" if score > 0.5 else "🟡" if score > 0.3 else "🟠"
        entry_id = result["matched_id"]
        category = result["category"]
        text = (
            f"{result['answer']}\n\n"
            f"───────────────\n"
            f"{emoji} Уверенность: {score:.0%}  |  📂 {category}"
        )
        await update.message.reply_text(
            text,
            reply_markup=kb_after_answer(entry_id, category, back_data="show_categories"),
        )
    else:
        suggestions = get_top_suggestions(user_text)
        text = (
            "🤷 Не нашёл точного ответа.\n\n"
            "Возможно, вас интересует одно из этих:"
        )
        await update.message.reply_text(
            text,
            reply_markup=kb_suggestions(suggestions),
        )


# ── Обработка inline-кнопок ─────────────────────────────────────

async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data = query.data

    # Показать список категорий
    if data == "show_categories":
        await query.edit_message_text(
            "📂 Выберите категорию:",
            reply_markup=kb_categories(),
        )

    # Открыть категорию (страница 0)
    elif data.startswith("cat:"):
        category = data[4:]
        questions = get_questions_by_category(category)
        await query.edit_message_text(
            f"📂 {category} — {len(questions)} вопр.:",
            reply_markup=kb_questions_in_category(category, page=0),
        )

    # Пагинация внутри категории
    elif data.startswith("cat_page:"):
        _, category, page_str = data.split(":", 2)
        page = int(page_str)
        questions = get_questions_by_category(category)
        await query.edit_message_text(
            f"📂 {category} — {len(questions)} вопр.:",
            reply_markup=kb_questions_in_category(category, page=page),
        )

    # Показать ответ на конкретный вопрос
    elif data.startswith("q:"):
        # Формат: q:{id}:cat:{category} или q:{id}:suggest
        parts = data.split(":", 3)
        entry_id = int(parts[1])
        back_data = ":".join(parts[2:])  # cat:{category} или suggest

        entry = get_entry_by_id(entry_id)
        if not entry:
            await query.edit_message_text("⚠️ Вопрос не найден.")
            return

        category = entry.get("category", "—")

        # back_data для кнопки «Назад»
        if back_data.startswith("cat:"):
            back_nav = back_data  # вернуться в категорию
        else:
            back_nav = "show_categories"

        text = (
            f"❓ {entry['question']}\n\n"
            f"{entry['answer']}\n\n"
            f"───────────────\n"
            f"📂 {category}"
        )
        await query.edit_message_text(
            text,
            reply_markup=kb_after_answer(entry_id, category, back_data=back_nav),
        )

    # Показать похожие вопросы той же категории
    elif data.startswith("similar:"):
        _, entry_id_str, category = data.split(":", 2)
        entry_id = int(entry_id_str)
        similar = get_similar_in_category(category, exclude_id=entry_id)
        if not similar:
            await query.answer("В этой категории больше нет вопросов.", show_alert=True)
            return
        await query.edit_message_text(
            f"🔁 Похожие вопросы в категории «{category}»:",
            reply_markup=kb_similar(similar, category),
        )


# ── Запуск ──────────────────────────────────────────────────────

def main():
    token = os.environ.get("BOT_TOKEN")
    if not token:
        raise RuntimeError("BOT_TOKEN не задан. export BOT_TOKEN='ваш_токен'")

    app = ApplicationBuilder().token(token).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("stats", cmd_stats))
    app.add_handler(CommandHandler("categories", cmd_categories))
    app.add_handler(CallbackQueryHandler(handle_callback))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Бот запущен (движок: %s)", engine.name)
    app.run_polling()


if __name__ == "__main__":
    main()
