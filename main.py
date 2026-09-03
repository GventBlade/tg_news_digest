import asyncio
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    ContextTypes,
    MessageHandler,
    filters,
)

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

# Приховуємо мережевий шум httpx.
logging.getLogger("httpx").setLevel(logging.WARNING)


def get_slot_header_text(
    news_count: int,
) -> str:
    kyiv_tz = ZoneInfo("Europe/Kyiv")
    now = datetime.now(kyiv_tz)

    rounded_time = (
        now
        + timedelta(minutes=30)
    ).replace(
        minute=0,
        second=0,
        microsecond=0,
    )

    display_hour = rounded_time.strftime(
        "%H:%M"
    )
    date_str = rounded_time.strftime(
        "%d.%m.%Y"
    )

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


def build_instagram_carousel_caption(
    top_news: list,
) -> str:
    kyiv_tz = ZoneInfo("Europe/Kyiv")
    now = datetime.now(kyiv_tz)

    rounded_time = (
        now
        + timedelta(minutes=30)
    ).replace(
        minute=0,
        second=0,
        microsecond=0,
    )

    display_hour = rounded_time.strftime(
        "%H:%M"
    )
    date_str = rounded_time.strftime(
        "%d.%m.%Y"
    )

    lines = [
        "🔥 ТОП ГОЛОВНИХ НОВИН",
        f"🕒 Станом на {display_hour}, {date_str}",
        "━━━━━━━━━━━━━━━━━━━━",
    ]

    for i, item in enumerate(
        top_news,
        1,
    ):
        first_line = (
            item["text"]
            .strip()
            .split("\n")[0]
        )
        lines.append(
            f"{i}. {first_line}"
        )

    channel_name = (
        settings.TARGET_CHANNEL_ID
        .replace("@", "")
    )

    lines.extend([
        "━━━━━━━━━━━━━━━━━━━━",
        (
            "📲 Більше деталей та всі новини — "
            "у нашому Telegram-каналі «Новини UA 6/24»:"
        ),
        f"👉 https://t.me/{channel_name}",
        "",
        "#новини #україна #новиниукраїни #дайджест #ua #news",
    ])

    return "\n".join(lines)


def cleanup_old_downloads(
    max_age_minutes: int = 360,
):
    downloads_dir = "downloads"

    if not os.path.exists(
        downloads_dir
    ):
        return

    now = time.time()

    for filename in os.listdir(
        downloads_dir
    ):
        file_path = os.path.join(
            downloads_dir,
            filename,
        )

        try:
            if os.path.isfile(
                file_path
            ):
                age_seconds = (
                    now
                    - os.path.getmtime(
                        file_path
                    )
                )

                if (
                    age_seconds
                    > max_age_minutes * 60
                ):
                    os.unlink(
                        file_path
                    )

        except Exception as e:
            logger.warning(
                "Не вдалося видалити старий "
                f"файл {file_path}: {e}"
            )


async def handle_admin_forwarded_message(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    user_id = (
        update.effective_user.id
        if update.effective_user
        else None
    )

    logger.info(
        "📩 Отримано повідомлення "
        f"від Telegram user_id: {user_id}"
    )

    if settings.ADMIN_TELEGRAM_ID is None:
        logger.error(
            "ADMIN_TELEGRAM_ID не налаштований у .env."
        )
        return

    if user_id != settings.ADMIN_TELEGRAM_ID:
        logger.warning(
            "⛔ Відхилено повідомлення "
            f"від user_id {user_id}."
        )
        return

    message = update.message

    if not message:
        return

    raw_text = (
        message.text
        or message.caption
        or ""
    )

    if raw_text.strip().startswith(
        "/start"
    ):
        await message.reply_text(
            "👋 Бот активний і готовий приймати новини від адміна!"
        )
        return

    # Порожній медіапост Analyzer не зможе нормально оцінити.
    if not raw_text.strip():
        await message.reply_text(
            "⚠️ Додай короткий текст або підпис до новини. "
            "Без тексту Analyzer не зможе коректно оцінити подію."
        )
        return

    channel_title = (
        "Пріоритет (Адмін)"
    )
    channel_username = ""

    if message.forward_origin:
        origin = message.forward_origin

        if (
            hasattr(origin, "chat")
            and origin.chat
        ):
            channel_title = (
                origin.chat.title
                or channel_title
            )
            channel_username = (
                origin.chat.username
                or ""
            )

    media_path = None
    media_type = None

    os.makedirs(
        "downloads",
        exist_ok=True,
    )

    try:
        if message.photo:
            photo = message.photo[-1]
            file = await photo.get_file()

            media_path = (
                f"downloads/manual_"
                f"{message.message_id}.jpg"
            )

            await file.download_to_drive(
                media_path
            )

            media_type = "photo"

        elif message.video:
            video = message.video
            file = await video.get_file()

            media_path = (
                f"downloads/manual_"
                f"{message.message_id}.mp4"
            )

            await file.download_to_drive(
                media_path
            )

            media_type = "video"

    except Exception as e:
        logger.error(
            "Не вдалося зберегти прикріплене "
            f"медіа від адміна: {e}"
        )

    history = NewsHistory()

    queue_id = history.add_manual_post(
        raw_text=raw_text,
        channel_title=channel_title,
        channel_username=channel_username,
        media_path=media_path,
        media_type=media_type,
        has_media=bool(media_path),
        has_video=(
            media_type == "video"
        ),
    )

    await message.reply_text(
        "✅ Новину збережено до черги. "
        "Вона матиме найвищий пріоритет "
        "у найближчому слоті."
    )

    logger.info(
        "✅ Ручну новину додано до черги "
        f"(queue_id={queue_id}, "
        f"telegram_message_id={message.message_id})."
    )


async def process_and_publish_news_cycle():
    logger.info(
        "🚀 Початок новинного циклу 4/24..."
    )

    cleanup_old_downloads(
        max_age_minutes=360
    )

    collector = None
    publisher = None

    try:
        collector = NewsCollector()
        summarizer = NewsSummarizer()
        publisher = NewsPublisher()
        history = NewsHistory()

        # 1. Ручні пріоритетні новини.
        pending_manual = (
            history.get_pending_manual_posts()
        )

        manual_posts_formatted = []

        now_utc = datetime.now(
            timezone.utc
        )

        for manual in pending_manual:
            queue_id = int(
                manual["id"]
            )

            manual_posts_formatted.append({
                "text": manual["raw_text"],
                "channel_name": (
                    manual["channel_username"]
                    or f"manual_{queue_id}"
                ),
                "channel_title": manual[
                    "channel_title"
                ],
                "channel_username": manual[
                    "channel_username"
                ],
                "views": int(
                    manual.get(
                        "views",
                        50000,
                    )
                    or 50000
                ),
                "forwards": 0,
                "replies": 0,
                "has_media": bool(
                    manual["has_media"]
                ),
                "has_video": bool(
                    manual["has_video"]
                ),
                "has_photo": (
                    manual["media_type"]
                    == "photo"
                ),
                "manual_media_path": manual[
                    "media_path"
                ],
                "manual_media_type": manual[
                    "media_type"
                ],
                "manual_queue_id": queue_id,
                "is_priority": True,
                "message_obj": None,

                # Унікальний negative ID замість 0 для всіх
                # ручних новин. Інакше історія могла перезаписуватися.
                "message_id": -queue_id,
                "date": now_utc,
            })

        logger.info(
            "Знайдено "
            f"{len(manual_posts_formatted)} "
            "ручних пріоритетних новин від адміна."
        )

        # 2. Автоматичні пости з каналів.
        fetched_posts = (
            await collector.fetch_recent_posts(
                hours=4,
                limit_per_channel=30,
            )
        )

        logger.info(
            "Зібрано "
            f"{len(fetched_posts)} "
            "сирих новин з каналів."
        )

        posts = (
            manual_posts_formatted
            + fetched_posts
        )

        if not posts:
            logger.warning(
                "Новин не знайдено, "
                "цикл завершено."
            )
            return

        # 3. Аналіз та формування TOP 5-10.
        past_events = (
            history.get_recent_events(
                hours=48
            )
        )

        top_news = (
            summarizer.select_top_distinct_news(
                posts,
                past_events=past_events,
                count=10,
            )
        )

        logger.info(
            "Фінальний список містить "
            f"{len(top_news)} новин."
        )

        if not top_news:
            logger.warning(
                "Дайджест порожній, "
                "публікацію скасовано."
            )
            return

        # 4. Header Telegram.
        header_text = (
            get_slot_header_text(
                len(top_news)
            )
        )

        header_published = (
            await publisher.publish_telegram_post(
                text=header_text
            )
        )

        if not header_published:
            logger.error(
                "Не вдалося опублікувати header. "
                "Цикл зупинено."
            )
            return

        await asyncio.sleep(2)

        # 5. Telegram-пости та медіа.
        ig_media_items = []
        published_news = []
        published_manual_ids = set()

        for index, item in enumerate(
            top_news,
            start=1,
        ):
            source_idx = item.get(
                "source_id"
            )

            target_post = (
                posts[source_idx]
                if (
                    isinstance(
                        source_idx,
                        int,
                    )
                    and 0
                    <= source_idx
                    < len(posts)
                )
                else None
            )

            if not target_post:
                logger.warning(
                    "Новина #%s пропущена: "
                    "некоректний source_id=%r",
                    index,
                    source_idx,
                )
                continue

            media_path = None
            media_type = None

            if target_post.get(
                "manual_media_path"
            ):
                media_path = target_post[
                    "manual_media_path"
                ]
                media_type = target_post[
                    "manual_media_type"
                ]

            elif target_post.get(
                "message_obj"
            ):
                try:
                    (
                        media_path,
                        media_type,
                    ) = (
                        await collector.download_post_media(
                            target_post[
                                "message_obj"
                            ]
                        )
                    )

                except Exception as dl_err:
                    logger.warning(
                        "Помилка завантаження медіа "
                        f"для новини #{index}: {dl_err}"
                    )

            published = (
                await publisher.publish_telegram_post(
                    text=item["text"],
                    media_path=media_path,
                    media_type=media_type,
                )
            )

            if not published:
                logger.warning(
                    f"Новина #{index} не опублікована."
                )
                await asyncio.sleep(3)
                continue

            published_news.append(
                item
            )

            if (
                media_path
                and media_type
                in {"photo", "video"}
            ):
                ig_media_items.append({
                    "path": media_path,
                    "type": media_type,
                })

                logger.info(
                    "Instagram media "
                    f"#{index}: {media_type} "
                    f"→ {media_path}"
                )

            first_line = (
                item["text"]
                .strip()
                .split("\n")[0]
            )

            # Ключова зміна проти дублів:
            # зберігаємо в історію ВСІ source_ids події,
            # а не тільки той пост, з якого взяли медіа.
            source_ids = item.get(
                "source_ids"
            )

            if not isinstance(
                source_ids,
                list,
            ):
                source_ids = [
                    source_idx
                ]

            if source_idx not in source_ids:
                source_ids.append(
                    source_idx
                )

            seen_source_ids = set()

            for event_source_idx in source_ids:
                if (
                    not isinstance(
                        event_source_idx,
                        int,
                    )
                    or event_source_idx
                    in seen_source_ids
                    or not (
                        0
                        <= event_source_idx
                        < len(posts)
                    )
                ):
                    continue

                seen_source_ids.add(
                    event_source_idx
                )

                event_post = posts[
                    event_source_idx
                ]

                history_channel = (
                    event_post.get(
                        "channel_username"
                    )
                    or event_post.get(
                        "channel_name"
                    )
                    or event_post.get(
                        "channel_title"
                    )
                    or "unknown"
                )

                history_message_id = (
                    event_post.get(
                        "message_id"
                    )
                )

                if not isinstance(
                    history_message_id,
                    int,
                ):
                    continue

                history.mark_as_published(
                    channel_name=history_channel,
                    message_id=history_message_id,
                    title=first_line,
                    summary=item.get(
                        "summary",
                        "",
                    ),
                    category=item.get(
                        "category",
                        "",
                    ),
                )

                manual_queue_id = (
                    event_post.get(
                        "manual_queue_id"
                    )
                )

                if isinstance(
                    manual_queue_id,
                    int,
                ):
                    published_manual_ids.add(
                        manual_queue_id
                    )

            await asyncio.sleep(3)

        # 6. Ручні новини позначаємо processed тільки якщо
        # відповідна подія справді була опублікована.
        if published_manual_ids:
            history.mark_manual_posts_processed(
                sorted(
                    published_manual_ids
                )
            )

            logger.info(
                "Позначено обробленими "
                f"{len(published_manual_ids)} "
                "ручних новин."
            )

        # 7. Instagram.
        if (
            published_news
            and ig_media_items
        ):
            logger.info(
                "Instagram: підготовлено "
                f"{len(ig_media_items)} медіа. "
                "Публікуємо..."
            )

            caption = (
                build_instagram_carousel_caption(
                    published_news
                )
            )

            await publisher.publish_instagram_carousel(
                caption=caption,
                media_items=ig_media_items,
            )

        else:
            logger.warning(
                "Instagram: валідні медіа "
                "відсутні або новини не були "
                "успішно опубліковані."
            )

        history.cleanup_old_records(
            days=5
        )

    except Exception as e:
        logger.error(
            f"Помилка новинного циклу: {e}",
            exc_info=True,
        )

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


async def on_startup(
    application,
):
    """
    Запускається всередині активного event loop.
    Scheduler зберігаємо в bot_data, щоб він гарантовано
    жив разом із Application.
    """
    scheduler = AsyncIOScheduler(
        timezone="Europe/Kyiv"
    )

    scheduler.add_job(
        process_and_publish_news_cycle,
        trigger=CronTrigger(
            hour="3,7,11,15,19,23",
            minute="59",
            timezone="Europe/Kyiv",
        ),
        id="news_cycle_4h",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
        misfire_grace_time=600,
    )

    scheduler.start()

    application.bot_data[
        "news_scheduler"
    ] = scheduler

    logger.info(
        "⏳ Планувальник 4/24 успішно запущено. "
        "Очікування наступного слоту..."
    )


def main():
    if settings.ADMIN_TELEGRAM_ID is None:
        logger.warning(
            "ADMIN_TELEGRAM_ID не заданий у .env. "
            "Ручне додавання новин не працюватиме."
        )

    application = (
        ApplicationBuilder()
        .token(settings.BOT_TOKEN)
        .post_init(on_startup)
        .build()
    )

    application.add_handler(
        MessageHandler(
            filters.ALL,
            handle_admin_forwarded_message,
        )
    )

    logger.info(
        "🤖 Запуск Telegram-бота..."
    )

    application.run_polling(
        allowed_updates=Update.ALL_TYPES,
        drop_pending_updates=False,
        close_loop=False,
    )


if __name__ == "__main__":
    main()
