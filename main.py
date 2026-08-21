import asyncio
from datetime import datetime, timedelta
import logging
import os
import shutil
from zoneinfo import ZoneInfo
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from app.services.collector import NewsCollector
from app.services.history import NewsHistory
from app.services.publisher import NewsPublisher
from app.services.summarizer import NewsSummarizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def get_slot_header_text() -> str:
    """
    Формує заголовок випуску з автоматичним округленням до найближчої цілої години.
    (03:57 -> 04:00, 07:57 -> 08:00, 11:57 -> 12:00, 15:57 -> 16:00, 19:57 -> 20:00, 23:57 -> 00:00)
    """
    kyiv_tz = ZoneInfo("Europe/Kyiv")
    now = datetime.now(kyiv_tz)

    # Округлюємо до найближчої повної години
    rounded_time = (now + timedelta(minutes=30)).replace(minute=0, second=0, microsecond=0)
    display_hour = rounded_time.strftime("%H:%M")
    date_str = rounded_time.strftime("%d.%m.%Y")

    return (
        f"🔥 <b>10 НАЙСВІЖІШИХ НОВИН</b>\n"
        f"🕒 <i>Станом на {display_hour}, {date_str}</i>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"<i>Головні події за останні 4 години:</i>"
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
    logger.info("🚀 Початок новинного циклу 4/24 (ТОП-10)...")
    collector = NewsCollector()
    summarizer = NewsSummarizer()
    publisher = NewsPublisher()
    history = NewsHistory()

    try:
        # 1. Збір постів за останні 4 години (вже відфільтровані від раніше опублікованих)
        posts = await collector.fetch_recent_posts(hours=4, limit_per_channel=10)
        logger.info(f"Зібрано {len(posts)} нових текстів для аналізу.")

        if not posts:
            logger.info("Нових новин за останні 4 години не виявлено.")
            return

        # 2. Gemini відбирає ТОП-10 унікальних новин
        top_news = summarizer.select_top_distinct_news(posts, count=10)
        logger.info(f"AI відібрав {len(top_news)} унікальних тем.")

        if not top_news:
            return

        # 3. Публікуємо заголовок випуску з округленим часом
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

    # Розклад 4/24: запуск о 03:57, 07:57, 11:57, 15:57, 19:57, 23:57
    scheduler.add_job(
        process_and_publish_news_cycle,
        trigger=CronTrigger(hour="3,7,11,15,19,23", minute="57", timezone="Europe/Kyiv")
    )

    scheduler.start()
    logger.info("⏳ Планувальник 4/24 запущено. Очікування наступного слоту (57 хв)...")

    while True:
        await asyncio.sleep(3600)


if __name__ == "__main__":
    asyncio.run(main())
