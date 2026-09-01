import asyncio
import logging
import os
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, filters, ContextTypes

from app.config import settings
from app.services.collector import NewsCollector
from app.services.history import NewsHistory
from app.services.publisher import NewsPublisher
from app.services.summarizer import NewsSummarizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Ваш числовий Telegram ID
ADMIN_TELEGRAM_ID = 6217500239


def get_slot_header_text(news_count: int) -> str:
    kyiv_tz = ZoneInfo("Europe/Kyiv")
    now = datetime.now(kyiv_tz)
    rounded_time = (now + timedelta(minutes=30)).replace(minute=0, second=0, microsecond=0)
    display_hour = rounded_time.strftime("%H:%M")
    date_str = rounded_time.strftime("%d.%m.%Y")

    count_word = "НАЙСВІЖІШИХ НОВИН"
    if news_count == 1:
        count_word = "ГОЛОВНА НОВИНА"
    elif 2 <= news_count <= 4:
        count_word = "НАЙСВІЖІШІ НОВИНИ"

    return (
        f"🔥 <b>{news_count} {count_word}</b>\n"
        f"🕒 <i>Станом на {display_hour}, {date_str}</i>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"<i>Головні події за останні 4 години:</i>"
    )


def build_instagram_carousel_caption(top_news: list) -> str:
    kyiv_tz = ZoneInfo("Europe/Kyiv")
    now = datetime.now(kyiv_tz)
    rounded_time = (now + timedelta(minutes=30)).replace(minute=0, second=0, microsecond=0)
    display_hour = rounded_time.strftime("%H:%M")
    date_str = rounded_time.strftime("%d.%m.%Y")

    lines = [
        "🔥 ТОП ГОЛОВНИХ НОВИН",
        f"🕒 Станом на {display_hour}, {date_str}",
        "━━━━━━━━━━━━━━━━━━━━",
    ]
    for i, item in enumerate(top_news, 1):
        first_line = item["text"].strip().split("\n")[0]
        lines.append(f"{i}. {first_line}")

    channel_name = settings.TARGET_CHANNEL_ID.replace("@", "")
    lines.extend([
        "━━━━━━━━━━━━━━━━━━━━",
        "📲 Більше деталей та всі новини — у нашому Telegram-каналі «Новини UA 6/24»:",
        f"👉 [https://t.me/](https://t.me/){channel_name}",
        "",
        "#новини #україна #новиниукраїни #дайджест #ua #news",
    ])
    return "\n".join(lines)


def cleanup_old_downloads(max_age_minutes: int = 120):
    downloads_dir = "downloads"
    if not os.path.exists(downloads_dir):
        return

    now = time.time()
    for filename in os.listdir(downloads_dir):
        file_path = os.path.join(downloads_dir, filename)
        try:
            if os.path.isfile(file_path):
                if now - os.path.getmtime(file_path) > (max_age_minutes * 60):
                    os.unlink(file_path)
        except Exception as e:
            logger.warning(f"Не вдалося видалити старий файл {file_path}: {e}")


async def handle_admin_forwarded_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id if update.effective_user else None
    if user_id != ADMIN_TELEGRAM_ID:
        return

    message = update.message
    if not message:
        return

    raw_text = message.text or message.caption or ""
    channel_title = "Пріоритет (Адмін)"
    channel_username = ""

    if message.forward_origin:
        origin = message.forward_origin
        if hasattr(origin, 'chat') and origin.chat:
            channel_title = origin.chat.title or channel_title
            channel_username = origin.chat.username or ""

    media_path = None
    media_type = None
    os.makedirs("downloads", exist_ok=True)

    try:
        if message.photo:
            photo = message.photo[-1]
            file = await photo.get_file()
            media_path = f"downloads/manual_{message.message_id}.jpg"
            await file.download_to_drive(media_path)
            media_type = "photo"
        elif message.video:
            video = message.video
            file = await video.get_file()
            media_path = f"downloads/manual_{message.message_id}.mp4"
            await file.download_to_drive(media_path)
            media_type = "video"
    except Exception as e:
        logger.error(f"Не вдалося зберегти прикріплене медіа від адміна: {e}")

    history = NewsHistory()
    history.add_manual_post(
        raw_text=raw_text,
        channel_title=channel_title,
        channel_username=channel_username,
        media_path=media_path,
        media_type=media_type,
        has_media=bool(media_path),
        has_video=(media_type == "video")
    )

    await message.reply_text("✅ Новину збережено до черги! Вона матиме найвищий пріоритет у найближчому слоті.")


async def process_and_publish_news_cycle():
    logger.info("🚀 Початок новинного циклу 4/24...")
    cleanup_old_downloads(max_age_minutes=120)

    collector = None
    publisher = None

    try:
        collector = NewsCollector()
        summarizer = NewsSummarizer()
        publisher = NewsPublisher()
        history = NewsHistory()

        # 1. Підтягуємо переслані новини від адміна
        pending_manual = history.get_pending_manual_posts()
        manual_posts_formatted = []
        for m in pending_manual:
            manual_posts_formatted.append({
                "text": m["raw_text"],
                "channel_title": m["channel_title"],
                "channel_username": m["channel_username"],
                "views": m["views"],
                "has_media": bool(m["has_media"]),
                "has_video": bool(m["has_video"]),
                "manual_media_path": m["media_path"],
                "manual_media_type": m["media_type"],
                "is_priority": True,
                "message_obj": None,
                "message_id": 0
            })
        logger.info(f"Знайдено {len(manual_posts_formatted)} ручних пріоритетних новин від адміна.")

        # 2. Збираємо авто-пости з каналів
        fetched_posts = await collector.fetch_recent_posts(hours=4, limit_per_channel=15)
        logger.info(f"Зібрано {len(fetched_posts)} сирих новин з каналів.")

        # Об'єднуємо список
        posts = manual_posts_formatted + fetched_posts

        if not posts:
            logger.warning("Новин не знайдено, цикл завершено.")
            return

        # 3. Аналіз та формування топу
        past_events = history.get_recent_events(hours=48)
        top_news = summarizer.select_top_distinct_news(posts, past_events=past_events, count=10)
        logger.info(f"Фінальний список містить {len(top_news)} новин.")

        if not top_news:
            logger.warning("Дайджест порожній, публікацію скасовано.")
            return

        # 4. Header Telegram
        header_text = get_slot_header_text(len(top_news))
        await publisher.publish_telegram_post(text=header_text)
        await asyncio.sleep(2)

        # 5. Telegram пости та медіа
        ig_media_items = []
        for index, item in enumerate(top_news, start=1):
            source_idx = item.get("source_id", 0)
            target_post = posts[source_idx] if (isinstance(source_idx, int) and 0 <= source_idx < len(posts)) else None

            media_path, media_type = None, None
            if target_post:
                if target_post.get("manual_media_path"):
                    media_path = target_post["manual_media_path"]
                    media_type = target_post["manual_media_type"]
                elif target_post.get("message_obj"):
                    try:
                        media_path, media_type = await collector.download_post_media(target_post["message_obj"])
                    except Exception as dl_err:
                        logger.warning(f"Помилка завантаження медіа для новини #{index}: {dl_err}")

            if media_path and media_type in {"photo", "video"}:
                ig_media_items.append({"path": media_path, "type": media_type})
                logger.info(f"Instagram media #{index}: {media_type} → {media_path}")

            # Публікація в Telegram
            published = await publisher.publish_telegram_post(
                text=item["text"],
                media_path=media_path,
                media_type=media_type,
            )

            if published and target_post:
                first_line = item["text"].strip().split("\n")[0]
                history.mark_as_published(
                    channel_name=target_post.get("channel_username") or target_post.get("channel_title", "unknown"),
                    message_id=target_post.get("message_id", 0),
                    title=first_line,
                    summary=item.get("summary", ""),
                    category=item.get("category", "")
                )

            await asyncio.sleep(3)

        # 6. Позначаємо ручні новини як оброблені
        if pending_manual:
            history.mark_manual_posts_processed([m["id"] for m in pending_manual])

        # 7. Публікація Instagram
        if ig_media_items:
            logger.info(f"Instagram: підготовлено {len(ig_media_items)} медіа. Публікуємо...")
            caption = build_instagram_carousel_caption(top_news)
            await publisher.publish_instagram_carousel(caption=caption, media_items=ig_media_items)
        else:
            logger.warning("Instagram: валідні медіа відсутні, публікацію пропущено.")

        history.cleanup_old_records(days=5)

    except Exception as e:
        logger.error(f"Помилка новинного циклу: {e}", exc_info=True)
    finally:
        if collector:
            try:
                await collector.close()
            except Exception:
                pass
        if publisher:
            try:
                await publisher.close()
            except Exception:
                pass


async def main():
    # Запуск бота для прийому пересланих повідомлень
    application = ApplicationBuilder().token(settings.BOT_TOKEN).build()
    application.add_handler(MessageHandler(filters.ALL & (~filters.COMMAND), handle_admin_forwarded_message))

    await application.initialize()
    await application.start()
    await application.updater.start_polling()
    logger.info("🤖 Telegram-бот для прийому новин від адміна успішно запущено!")

    # Планувальник чергових випусків
    scheduler = AsyncIOScheduler(timezone="Europe/Kyiv")
    scheduler.add_job(
        process_and_publish_news_cycle,
        trigger=CronTrigger(hour="3,7,11,15,19,23", minute="58", timezone="Europe/Kyiv"),
    )
    scheduler.start()
    logger.info("⏳ Планувальник 4/24 запущено. Очікування наступного слоту...")

    try:
        while True:
            await asyncio.sleep(3600)
    finally:
        await application.updater.stop()
        await application.stop()
        await application.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
