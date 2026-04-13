"""
Telegram-бот: Генератор ответов на типовые вопросы преподавателя.
Поддерживает 3 движка (выбор через переменную ENGINE_TYPE):
  - tfidf           (по умолчанию)
  - tfidf_synonyms
  - embeddings
"""

import os
import logging
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
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


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Привет! Я бот-помощник преподавателя.\n\n"
        "Задайте мне вопрос — я постараюсь найти ответ из базы знаний.\n\n"
        f"🔧 Движок: {engine.name}\n\n"
        "Команды:\n"
        "/help — справка\n"
        "/stats — статистика базы\n"
        "/categories — список категорий"
    )


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "ℹ️ Напишите вопрос обычным текстом.\n"
        "Бот найдёт ответ в базе или предложит обратиться к преподавателю.\n\n"
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
        f"📊 Статистика базы знаний:\n",
        f"Движок: {engine.name}",
        f"Всего вопросов: {len(engine.entries)}",
        f"\nКатегории:",
    ]
    for cat, count in sorted(cats.items()):
        lines.append(f"  • {cat}: {count}")
    await update.message.reply_text("\n".join(lines))


async def cmd_categories(update: Update, context: ContextTypes.DEFAULT_TYPE):
    cats = sorted(set(e.get("category", "—") for e in engine.entries))
    text = "📂 Категории вопросов:\n\n" + "\n".join(f"• {c}" for c in cats)
    await update.message.reply_text(text)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text.strip()
    if not user_text:
        return

    logger.info("Вопрос от %s: %s", update.effective_user.first_name, user_text)
    result = engine.find_answer(user_text)
    score = result["score"]

    if result["matched_question"]:
        emoji = "🟢" if score > 0.5 else "🟡" if score > 0.3 else "🟠"
        reply = (
            f"{result['answer']}\n\n"
            f"───────────────\n"
            f"{emoji} Уверенность: {score:.0%}\n"
            f"📂 Категория: {result['category']}"
        )
    else:
        reply = result["answer"]

    await update.message.reply_text(reply)


def main():
    token = os.environ.get("BOT_TOKEN")
    if not token:
        raise RuntimeError("BOT_TOKEN не задан. export BOT_TOKEN='ваш_токен'")

    app = ApplicationBuilder().token(token).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("stats", cmd_stats))
    app.add_handler(CommandHandler("categories", cmd_categories))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Бот запущен (движок: %s)", engine.name)
    app.run_polling()


if __name__ == "__main__":
    main()
