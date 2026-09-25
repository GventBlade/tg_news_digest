import asyncio
import html
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

# Quality audit навмисно імпортуємо fail-safe:
# якщо новий модуль випадково відсутній або має помилку імпорту,
# основний Telegram/Instagram pipeline все одно запускається.
try:
    from app.services.quality_audit import QualityAuditor
    _QUALITY_AUDIT_IMPORT_ERROR = None
except Exception as exc:
    QualityAuditor = None
    _QUALITY_AUDIT_IMPORT_ERROR = exc

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


def append_reference_link(
    text: str,
    reference_url: str | None,
    reference_label: str | None = None,
) -> str:
    """
    Додає коротке клікабельне першоджерело лише коли Summarizer знайшов
    реальний зовнішній URL у початкових Telegram-постах.

    Для читача назва завжди проста й універсальна — "Посилання".
    Внутрішня класифікація Summarizer
    (документ / дослідження / звіт / стаття)
    використовується тільки для вибору правильного URL.
    """
    base = str(text or "").strip()
    url = str(reference_url or "").strip()

    if not url:
        return base

    label = "Посилання"

    safe_url = html.escape(
        url,
        quote=True,
    )
    safe_label = html.escape(
        label
    )

    return (
        f"{base}\n\n"
        f"🔗 <a href=\"{safe_url}\">{safe_label}</a>"
    )


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
        "Вона гарантовано піде в найближчий слот; "
        "короткий опис із медіа теж допускається."
    )

    logger.info(
        "✅ Ручну новину додано до черги "
        f"(queue_id={queue_id}, "
        f"telegram_message_id={message.message_id})."
    )


def _manual_queue_ids_for_item(
    item: dict,
    posts: list,
) -> tuple[list[int], list[int]]:
    """
    Повертає (all_manual_ids, safe_manual_ids) для конкретного final item.

    Якщо один item раптом містить кілька manual IDs без підтвердженого merge,
    вважаємо покритим лише primary manual. Це не дає помилково позначити всі
    ручні новини processed через один змерджений пост.
    """
    source_ids = item.get("source_ids")
    if not isinstance(source_ids, list):
        source_ids = [item.get("source_id")]

    all_ids = []
    source_to_queue = {}
    for source_idx in source_ids:
        if not isinstance(source_idx, int) or not (0 <= source_idx < len(posts)):
            continue
        queue_id = posts[source_idx].get("manual_queue_id")
        if isinstance(queue_id, int):
            all_ids.append(queue_id)
            source_to_queue[source_idx] = queue_id

    all_ids = list(dict.fromkeys(all_ids))
    if len(all_ids) <= 1 or bool(item.get("manual_merge_verified", True)):
        return all_ids, all_ids

    primary_source = item.get("source_id")
    primary_queue = source_to_queue.get(primary_source)
    safe = [primary_queue] if isinstance(primary_queue, int) else all_ids[:1]
    return all_ids, safe


def _build_manual_publication_fallback(
    post: dict,
    source_idx: int,
) -> dict | None:
    """Остання main-level страховка, якщо Summarizer все ж загубив manual."""
    raw = str(post.get("text") or "").strip()
    if not raw:
        return None

    normalized = " ".join(raw.split())
    first_line = normalized.split("\n", 1)[0].strip()
    headline = first_line
    if len(headline) > 150:
        headline = headline[:147].rstrip() + "…"

    attack_markers = (
        "удар", "влуч", "приліт", "прильот", "обстріл", "атака",
        "ракета", "дрон", "бпла", "шахед", "вибух", "пожеж",
    )
    is_attack = any(marker in normalized.lower() for marker in attack_markers)
    emoji = "💥" if is_attack else "📰"

    safe_headline = html.escape(headline)
    safe_body = html.escape(normalized)
    if normalized == headline:
        text = f"{emoji} <b>{safe_headline}</b>"
    else:
        text = f"{emoji} <b>{safe_headline}</b>\n\n{safe_body}"

    return {
        "event_id": f"MAIN_MANUAL_FALLBACK_{post.get('manual_queue_id', source_idx)}",
        "source_id": source_idx,
        "source_ids": [source_idx],
        "text": text,
        "summary": normalized[:900],
        "category": "war" if is_attack else "other",
        "digest_role": "core",
        "is_discovery_candidate": False,
        "is_priority": True,
        "priority_source_ids": [source_idx],
        "manual_merge_verified": True,
        "reference_url": None,
        "reference_label": None,
        "_main_manual_fallback": True,
    }


def _ensure_manual_items_before_publish(
    top_news: list,
    posts: list,
    expected_manual_ids: list[int],
) -> list:
    """
    Незалежна end-to-end страховка перед header/publish.

    Нормально вона нічого не змінює: Summarizer уже гарантує manual до
    FINAL_FACT_CHECK. Якщо ж конкретний queue_id відсутній або захований у
    непідтвердженому multi-manual merge, додаємо raw-safe fallback і НЕ
    видаляємо manual через ліміт 10.
    """
    if not expected_manual_ids:
        return top_news

    result = [dict(item) for item in top_news if isinstance(item, dict)]
    covered = set()
    for item in result:
        _, safe_ids = _manual_queue_ids_for_item(item, posts)
        covered.update(safe_ids)

    missing = [queue_id for queue_id in expected_manual_ids if queue_id not in covered]
    if not missing:
        logger.info(
            "PRE-PUBLISH MANUAL GUARANTEE: усі %s queue_id присутні у фіналі.",
            len(expected_manual_ids),
        )
        return result

    logger.error(
        "CRITICAL PRE-PUBLISH MANUAL GUARANTEE: відсутні queue_id=%s. "
        "Додаємо main-level safe fallback.",
        missing,
    )

    queue_to_source = {}
    for source_idx, post in enumerate(posts):
        queue_id = post.get("manual_queue_id")
        if isinstance(queue_id, int):
            queue_to_source[queue_id] = source_idx

    for queue_id in missing:
        source_idx = queue_to_source.get(queue_id)
        if source_idx is None:
            logger.error("Manual queue_id=%s не має source_idx у posts.", queue_id)
            continue
        fallback = _build_manual_publication_fallback(posts[source_idx], source_idx)
        if not fallback:
            logger.error("Не вдалося побудувати main fallback для queue_id=%s.", queue_id)
            continue

        insert_at = next(
            (
                idx
                for idx, item in enumerate(result)
                if item.get("digest_role") == "discovery"
            ),
            len(result),
        )
        result.insert(insert_at, fallback)
        covered.add(queue_id)

    return result


async def process_and_publish_news_cycle():
    cycle_started_at = datetime.now(
        timezone.utc
    ).strftime(
        "%Y-%m-%d %H:%M:%S"
    )

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

        expected_manual_ids = [
            int(manual["id"])
            for manual in pending_manual
            if str(
                manual.get("id", "")
            ).isdigit()
        ]

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

        # Незалежна end-to-end manual страховка. У нормі нічого не додає;
        # спрацьовує лише якщо Summarizer все ж загубив конкретний queue_id.
        top_news = _ensure_manual_items_before_publish(
            top_news,
            posts,
            expected_manual_ids,
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

            # Зберігаємо початковий media-state для read-only audit.
            # Audit не запускає Vision вдруге; він лише перевірить,
            # що відхилене медіа не залишилось у фінальному pipeline.
            original_media_path = media_path
            original_media_type = media_type
            media_rejected = False
            media_reject_reason = ""

            # ЄДИНИЙ media-gate для ОБОХ платформ.
            # Verdict отримуємо ДО публікації.
            if (
                media_path
                and media_type in {"photo", "video"}
            ):
                media_verdict = (
                    await publisher.validate_media_for_news(
                        text=item["text"],
                        media_path=media_path,
                        media_type=media_type,
                    )
                )

                if not media_verdict.get(
                    "is_relevant",
                    False,
                ):
                    media_rejected = True
                    media_reject_reason = str(
                        media_verdict.get(
                            "reason",
                            "",
                        )
                        or ""
                    )

                    logger.warning(
                        "MEDIA DROPPED FOR ALL PLATFORMS: "
                        "news_index=%s path=%s type=%s reason=%s",
                        index,
                        media_path,
                        media_type,
                        media_verdict.get("reason", ""),
                    )

                    media_path = None
                    media_type = None

            # Посилання додаємо ПІСЛЯ Vision-перевірки,
            # щоб службовий рядок не впливав на media-gate.
            publication_text = append_reference_link(
                item["text"],
                item.get("reference_url"),
                item.get("reference_label"),
            )

            published = (
                await publisher.publish_telegram_post(
                    text=publication_text,
                    media_path=media_path,
                    media_type=media_type,

                    # Уже перевірили вище один раз.
                    validate_media=False,
                )
            )

            if not published:
                logger.warning(
                    f"Новина #{index} не опублікована."
                )
                await asyncio.sleep(3)
                continue

            published_item = dict(item)
            published_item["text"] = (
                publication_text
            )

            all_manual_ids, safe_manual_ids = _manual_queue_ids_for_item(
                item,
                posts,
            )
            published_item["_audit_manual_all_ids"] = all_manual_ids
            published_item["_audit_manual_ids"] = safe_manual_ids
            published_item["_audit_manual_merge_verified"] = bool(
                item.get("manual_merge_verified", True)
            )
            published_item["_audit_main_manual_fallback"] = bool(
                item.get("_main_manual_fallback", False)
            )

            if len(all_manual_ids) > 1 and all_manual_ids != safe_manual_ids:
                logger.error(
                    "UNVERIFIED MANUAL MERGE published: all=%s safe=%s event_id=%s",
                    all_manual_ids,
                    safe_manual_ids,
                    item.get("event_id"),
                )

            # Manual queue_id стає processed ОДРАЗУ після успішної Telegram-
            # публікації конкретного item. Instagram/history помилка пізніше в
            # циклі вже не повинна змусити цю ручну новину вийти вдруге.
            if safe_manual_ids:
                try:
                    history.mark_manual_posts_processed(
                        sorted(set(safe_manual_ids))
                    )
                    published_manual_ids.update(safe_manual_ids)
                    logger.info(
                        "Manual post(s) confirmed published+processed: %s",
                        sorted(set(safe_manual_ids)),
                    )
                except Exception as manual_state_exc:
                    logger.error(
                        "Не вдалося позначити manual processed після успішної "
                        "Telegram-публікації: ids=%s error=%s",
                        safe_manual_ids,
                        manual_state_exc,
                        exc_info=True,
                    )

            # Runtime telemetry лише для post-publication audit.
            # Ці службові поля не публікуються в Telegram і не
            # записуються у semantic history.
            published_item["_audit_media"] = {
                "original_path": original_media_path,
                "original_type": original_media_type,
                "rejected": media_rejected,
                "reject_reason": media_reject_reason,
                "final_path": media_path,
                "final_type": media_type,
            }

            published_news.append(
                published_item
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
                publication_text
                .strip()
                .split("\n")[0]
            )

            # Зберігаємо в історію ВСІ source_ids події,
            # а не тільки пост, з якого взяли медіа.
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

                    # Зберігаємо повний редакторський текст
                    # без URL, щоб наступні цикли мали
                    # сильніший semantic history context.
                    summary=item.get(
                        "text",
                        item.get(
                            "summary",
                            "",
                        ),
                    ),

                    category=item.get(
                        "category",
                        "",
                    ),
                )

                # processed-state оновлюємо нижче з safe_manual_ids конкретного
                # УСПІШНО опублікованого item. Не позначаємо всі source_ids
                # автоматично: це й було причиною хибного "3 processed" після
                # одного невдалого multi-manual merge.

            await asyncio.sleep(3)

        # 6. Manual уже позначаються processed поштучно одразу після
        # успішної Telegram-публікації. Тут лише підсумкова telemetry.
        if published_manual_ids:
            logger.info(
                "Успішно опубліковано й позначено processed "
                f"{len(published_manual_ids)} ручних новин."
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

        # 8. POST-PUBLICATION QUALITY AUDIT.
        #
        # ВАЖЛИВО:
        # - запускається лише ПІСЛЯ основної публікації;
        # - нічого не видаляє, не редагує і не перепубліковує;
        # - працює в окремому thread, щоб Gemini audit не блокував
        #   Telegram polling / event loop;
        # - будь-яка помилка audit НЕ ламає новинний цикл.
        fact_check_stats = getattr(
            summarizer,
            "last_fact_check_stats",
            None,
        )

        if QualityAuditor is None:
            logger.warning(
                "QUALITY AUDIT skipped: модуль quality_audit "
                "не завантажився: %s",
                _QUALITY_AUDIT_IMPORT_ERROR,
            )

        elif (
            not isinstance(
                fact_check_stats,
                dict,
            )
            or not fact_check_stats
        ):
            # Не підробляємо checked=N. Поки Summarizer не віддав
            # реальну telemetry FINAL_FACT_CHECK, audit краще пропустити,
            # ніж записати неправдиве QUALITY AUDIT: OK.
            logger.warning(
                "QUALITY AUDIT skipped: Summarizer ще не віддав "
                "last_fact_check_stats. Потрібен telemetry-патч "
                "summarizer.py."
            )

        elif published_news:
            try:
                auditor = QualityAuditor()

                audit_result = await asyncio.to_thread(
                    auditor.run,
                    published_news=published_news,
                    prior_events=past_events,
                    fact_check_stats=fact_check_stats,
                    expected_manual_ids=expected_manual_ids,
                    published_manual_ids=sorted(
                        published_manual_ids
                    ),
                    cycle_started_at=cycle_started_at,
                )

                audit_id = (
                    history.save_quality_audit(
                        audit_result
                    )
                )

                logger.info(
                    "QUALITY AUDIT saved: id=%s status=%s.",
                    audit_id,
                    audit_result.get(
                        "status",
                        "UNKNOWN",
                    ),
                )

            except Exception as audit_exc:
                logger.warning(
                    "QUALITY AUDIT failed safely: %s",
                    audit_exc,
                    exc_info=True,
                )

        else:
            logger.warning(
                "QUALITY AUDIT skipped: у цьому циклі "
                "немає успішно опублікованих новин."
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

    Scheduler зберігаємо в bot_data,
    щоб він гарантовано жив разом із Application.
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
