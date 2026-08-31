import asyncio
import logging
import os
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
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
        f"👉 https://t.me/{channel_name}",
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

        posts = await collector.fetch_recent_posts(hours=4, limit_per_channel=15)
        logger.info(f"Зібрано {len(posts)} сирих новин за останні 4 год.")

        if not posts:
            logger.warning("Новин не знайдено, цикл завершено.")
            return

        past_titles = history.get_recent_titles(hours=48)
        top_news = summarizer.select_top_distinct_news(posts, past_titles=past_titles, count=10)
        logger.info(f"Фінальний список містить {len(top_news)} новин.")

        if not top_news:
            logger.warning("Дайджест порожній, публікацію скасовано.")
            return

        # 1. Header Telegram
        header_text = get_slot_header_text(len(top_news))
        await publisher.publish_telegram_post(text=header_text)
        await asyncio.sleep(2)

        # 2. Telegram пости та збір медіа для Instagram
        ig_media_items = []
        for index, item in enumerate(top_news, start=1):
            source_idx = item.get("source_id", 0)
            target_post = posts[source_idx] if (isinstance(source_idx, int) and 0 <= source_idx < len(posts)) else None

            media_path, media_type = None, None
            if target_post:
                try:
                    media_path, media_type = await collector.download_post_media(target_post["message_obj"])
                except Exception as dl_err:
                    logger.warning(f"Помилка завантаження медіа для новини #{index}: {dl_err}")

            if media_path and media_type in {"photo", "video"}:
                ig_media_items.append({"path": media_path, "type": media_type})
                logger.info(f"Instagram media #{index}: {media_type} → {media_path}")

            # Публікуємо та фіксуємо в історію тільки при успішній публікації
            published = await publisher.publish_telegram_post(
                text=item["text"],
                media_path=media_path,
                media_type=media_type,
            )

            if published and target_post:
                first_line = item["text"].strip().split("\n")[0]
                history.mark_as_published(
                    channel_name=target_post["channel_name"],
                    message_id=target_post["message_id"],
                    title=first_line,
                )
            elif not published:
                logger.warning(f"Новина #{index} НЕ була опублікована, пропуск збереження в історію.")

            await asyncio.sleep(3)

        # 3. Публікація Instagram
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
                logger.exception("Помилка закриття Collector")
        if publisher:
            try:
                await publisher.close()
            except Exception:
                logger.exception("Помилка закриття Publisher")


async def main():
    scheduler = AsyncIOScheduler(timezone="Europe/Kyiv")
    scheduler.add_job(
        process_and_publish_news_cycle,
        trigger=CronTrigger(hour="3,7,11,15,19,23", minute="58", timezone="Europe/Kyiv"),
    )
    scheduler.start()
    logger.info("⏳ Планувальник 4/24 запущено. Очікування наступного слоту...")
    while True:
        await asyncio.sleep(3600)


if __name__ == "__main__":
    asyncio.run(main())
