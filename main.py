import asyncio
import logging
import os
import shutil
from datetime import datetime
from zoneinfo import ZoneInfo
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from app.services.collector import NewsCollector
from app.services.summarizer import NewsSummarizer
from app.services.publisher import NewsPublisher
from app.services.history import NewsHistory

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def get_slot_header_text() -> str:
    """Формує заголовок випуску з прив'язкою до слоту та дати."""
    kyiv_tz = ZoneInfo("Europe/Kyiv")
    now = datetime.now(kyiv_tz)

    hour = now.hour
    if hour in [23, 0]:
        display_hour = "00:00"
    elif 5 <= hour <= 6:
        display_hour = "06:00"
    elif 11 <= hour <= 12:
        display_hour = "12:00"
    elif 17 <= hour <= 18:
        display_hour = "18:00"
    else:
        display_hour = now.strftime("%H:%M")

    date_str = now.strftime("%d.%m.%Y")

    return (
        f"🔥 <b>10 НАЙСВІЖІШИХ НОВИН</b>\n"
        f"🕒 <i>Станом на {display_hour}, {date_str}</i>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"<i>Головні події за останні 6 годин:</i>"
    )


def cleanup_downloads_folder():
    """Повне очищення папки завантажень від залишків."""
    downloads_dir = "downloads"
    if os.path.exists(downloads_dir):
        for filename in os.listdir(downloads_dir):
            file_path = os.path.join(downloads_dir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                logger.warning(f"Не вдалося видалити залишковий файл {file_path}: {e}")


async def process_and_publish_news_cycle():
    logger.info("🚀 Початок новинного циклу (ТОП-10)...")
    collector = NewsCollector()
    summarizer = NewsSummarizer()
    publisher = NewsPublisher()
    history = NewsHistory()

    try:
        # 1. Збір постів за останні 6 годин (вже відфільтровані від раніше опублікованих)
        posts = await collector.fetch_recent_posts(hours=6, limit_per_channel=10)
        logger.info(f"Зібрано {len(posts)} нових текстів для аналізу.")

        if not posts:
            logger.info("Нових новин за останні 6 годин не виявлено.")
            return

        # 2. Gemini відбирає ТОП-10 унікальних новин
        top_news = summarizer.select_top_distinct_news(posts, count=10)
        logger.info(f"AI відібрав {len(top_news)} унікальних тем.")

        if not top_news:
            return

        # 3. Публікуємо заголовок випуску
        header_text = get_slot_header_text()
        await publisher.publish_news(text=header_text)
        await asyncio.sleep(2)

        # 4. Публікуємо 10 новин
        for item in top_news:
            source_idx = item.get("source_id", 0)
            target_post = posts[source_idx] if 0 <= source_idx < len(posts) else posts[0]

            # Завантажуємо оригінальне медіа тільки для цього обраного поста
            media_path, media_type = await collector.download_post_media(target_post["message_obj"])

            # Публікуємо новину
            await publisher.publish_news(
                text=item["text"],
                media_path=media_path,
                media_type=media_type
            )

            # Позначаємо в базі, щоб ніколи більше не брати цей пост
            history.mark_as_published(
                channel_name=target_post["channel_name"],
                message_id=target_post["message_id"]
            )

            # Очищуємо файл відразу після відправки
            if media_path and os.path.exists(media_path):
                try:
                    os.remove(media_path)
                except Exception as e:
                    logger.warning(f"Не вдалося видалити {media_path}: {e}")

            await asyncio.sleep(3)

        # 5. Підчищаємо стару історію (> 2 днів) та залишки папки
        history.cleanup_old_records(days=2)
        cleanup_downloads_folder()

        logger.info("✅ Випуск ТОП-10 успішно опубліковано, пам'ять очищено!")

    except Exception as e:
        logger.error(f"Помилка новинного циклу: {e}", exc_info=True)
    finally:
        await collector.close()


async def main():
    scheduler = AsyncIOScheduler(timezone="Europe/Kyiv")

    # Розклад: за 5 хв до 00:00, 06:00, 12:00, 18:00
    scheduler.add_job(
        process_and_publish_news_cycle,
        trigger=CronTrigger(hour="5,11,17,23", minute="55", timezone="Europe/Kyiv")
    )

    scheduler.start()
    logger.info("⏳ Планувальник запущено. Очікування наступного слоту...")

    while True:
        await asyncio.sleep(3600)


if __name__ == "__main__":
    asyncio.run(main())
