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
    telegram_file_id = ""
    telegram_file_unique_id = ""
    telegram_file_size = 0

    os.makedirs(
        "downloads",
        exist_ok=True,
    )

    # Для manual media Telegram file_id є головною страховкою. Bot API може
    # відмовити у download великого відео ("File is too big"), але той самий
    # бот все одно може повторно відправити оригінал за file_id.
    if message.photo:
        photo = message.photo[-1]
        media_type = "photo"
        telegram_file_id = str(
            getattr(photo, "file_id", "") or ""
        )
        telegram_file_unique_id = str(
            getattr(photo, "file_unique_id", "") or ""
        )
        telegram_file_size = int(
            getattr(photo, "file_size", 0) or 0
        )
        target_path = (
            f"downloads/manual_"
            f"{message.message_id}.jpg"
        )
        try:
            file = await photo.get_file()
            await file.download_to_drive(
                target_path
            )
            if os.path.exists(target_path):
                media_path = target_path
        except Exception as e:
            logger.warning(
                "MANUAL MEDIA local download failed; "
                "залишаємо Telegram file_id fallback: %s",
                e,
            )

    elif message.video:
        video = message.video
        media_type = "video"
        telegram_file_id = str(
            getattr(video, "file_id", "") or ""
        )
        telegram_file_unique_id = str(
            getattr(video, "file_unique_id", "") or ""
        )
        telegram_file_size = int(
            getattr(video, "file_size", 0) or 0
        )
        target_path = (
            f"downloads/manual_"
            f"{message.message_id}.mp4"
        )
        try:
            file = await video.get_file()
            await file.download_to_drive(
                target_path
            )
            if os.path.exists(target_path):
                media_path = target_path
        except Exception as e:
            logger.warning(
                "MANUAL MEDIA local download failed; "
                "залишаємо Telegram file_id fallback: %s",
                e,
            )

    has_manual_media = bool(
        media_type in {"photo", "video"}
        and (media_path or telegram_file_id)
    )

    # Якщо користувач реально надіслав media, але ми не маємо ні локального
    # файла, ні Telegram file_id, не створюємо оманливий text-only manual.
    if media_type in {"photo", "video"} and not has_manual_media:
        await message.reply_text(
            "⚠️ Не вдалося зафіксувати медіа. "
            "Новину не додано до черги — надішли її ще раз."
        )
        logger.error(
            "MANUAL MEDIA capture failed completely: telegram_message_id=%s",
            message.message_id,
        )
        return

    history = NewsHistory()

    queue_id = history.add_manual_post(
        raw_text=raw_text,
        channel_title=channel_title,
        channel_username=channel_username,
        media_path=media_path,
        media_type=media_type,
        has_media=has_manual_media,
        has_video=(
            media_type == "video"
        ),
        telegram_file_id=telegram_file_id,
        telegram_file_unique_id=telegram_file_unique_id,
        telegram_file_size=telegram_file_size,
    )

    media_note = (
        " Медіа зафіксовано й не буде замінюватися."
        if has_manual_media
        else ""
    )
    await message.reply_text(
        "✅ Новину збережено до черги. "
        "Вона гарантовано піде в найближчий слот; "
        "короткий опис із медіа теж допускається."
        + media_note
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
        "manual_fact_locked": True,
        "manual_fact_source_ids": [source_idx],
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


def _manual_source_indexes_for_item(
    item: dict,
    posts: list,
) -> list[int]:
    """Return concrete admin/manual source indexes referenced by one final item."""
    ordered = []
    for value in (
        item.get("manual_fact_source_ids") or []
    ):
        if isinstance(value, int):
            ordered.append(value)
    for value in (item.get("priority_source_ids") or []):
        if isinstance(value, int):
            ordered.append(value)
    if isinstance(item.get("source_id"), int):
        ordered.append(item.get("source_id"))
    raw_sources = item.get("source_ids")
    if isinstance(raw_sources, list):
        ordered.extend(value for value in raw_sources if isinstance(value, int))

    result = []
    seen = set()
    for source_idx in ordered:
        if source_idx in seen or not (0 <= source_idx < len(posts)):
            continue
        seen.add(source_idx)
        if bool(posts[source_idx].get("is_priority")):
            result.append(source_idx)
    return result


def _enforce_manual_fact_lock_before_publish(
    top_news: list,
    posts: list,
) -> list:
    """
    Publication-boundary invariant for manual news.

    New Summarizer versions mark manual items as manual_fact_locked and rebuild
    them from admin-only sources. If an older/regressed code path ever reaches
    main without that marker, do NOT trust the mixed Editor text: replace it
    with a raw-safe manual-only fallback before any media preparation starts.
    """
    result = []
    for item in top_news or []:
        if not isinstance(item, dict):
            continue
        item_copy = dict(item)
        manual_sources = _manual_source_indexes_for_item(item_copy, posts)
        if not manual_sources:
            result.append(item_copy)
            continue

        source_ids = item_copy.get("source_ids")
        source_ids = source_ids if isinstance(source_ids, list) else []
        foreign_sources = [
            sid for sid in source_ids
            if isinstance(sid, int)
            and 0 <= sid < len(posts)
            and not bool(posts[sid].get("is_priority"))
        ]

        if bool(item_copy.get("manual_fact_locked")) and not foreign_sources:
            item_copy["source_ids"] = list(manual_sources)
            item_copy["priority_source_ids"] = list(manual_sources)
            item_copy["manual_fact_source_ids"] = list(manual_sources)
            logger.info(
                "PRE-PUBLISH MANUAL FACT LOCK OK: event_id=%s source_ids=%s",
                item_copy.get("event_id"),
                manual_sources,
            )
            result.append(item_copy)
            continue

        # Last-resort fail-safe: use exactly one concrete manual source. If an
        # unverified multi-manual merge slipped through, publishing one clean
        # admin item is safer than one post containing several unrelated stories.
        preferred = item_copy.get("source_id")
        source_idx = (
            preferred
            if isinstance(preferred, int) and preferred in manual_sources
            else manual_sources[0]
        )
        fallback = _build_manual_publication_fallback(posts[source_idx], source_idx)
        if not fallback:
            logger.error(
                "PRE-PUBLISH MANUAL FACT LOCK FAILED: event_id=%s source_id=%s",
                item_copy.get("event_id"),
                source_idx,
            )
            result.append(item_copy)
            continue

        # Keep the existing event_id so history/order/audit still refer to the
        # same selected event, but factual text/source are manual-only.
        fallback["event_id"] = str(
            item_copy.get("event_id") or fallback.get("event_id") or ""
        )
        logger.error(
            "PRE-PUBLISH MANUAL FACT LOCK REBUILT: event_id=%s manual_source=%s "
            "foreign_sources=%s. Mixed text was discarded.",
            fallback.get("event_id"),
            source_idx,
            foreign_sources,
        )
        result.append(fallback)

    return result


def _locked_manual_media_source_for_item(
    item: dict,
    posts: list,
) -> int | None:
    """Independent publication-level manual media lock."""
    candidates = []

    preferred = item.get("manual_media_source_id")
    if isinstance(preferred, int):
        candidates.append(preferred)

    source_id = item.get("source_id")
    if isinstance(source_id, int):
        candidates.append(source_id)

    for value in (item.get("priority_source_ids") or []):
        if isinstance(value, int):
            candidates.append(value)

    source_ids = item.get("source_ids")
    if isinstance(source_ids, list):
        candidates.extend(value for value in source_ids if isinstance(value, int))

    seen = set()
    for source_idx in candidates:
        if source_idx in seen or not (0 <= source_idx < len(posts)):
            continue
        seen.add(source_idx)
        post = posts[source_idx]
        media_path = str(post.get("manual_media_path") or "").strip()
        media_file_id = str(
            post.get("manual_telegram_file_id")
            or post.get("telegram_file_id")
            or ""
        ).strip()
        media_type = str(post.get("manual_media_type") or "").strip().lower()
        if (
            bool(post.get("is_priority"))
            and (media_path or media_file_id)
            and media_type in {"photo", "video"}
        ):
            return source_idx

    return None


AUTO_MEDIA_FALLBACK_MAX_CANDIDATES = 8


def _auto_media_candidate_source_ids(
    item: dict,
    posts: list,
) -> list[int]:
    """
    Ordered AUTO-media candidates for one final event.

    Summarizer now supplies `media_candidate_source_ids`, already filtered by
    its deterministic event-consistency gate. For backward compatibility with
    an older summarizer we append the selected source and the event source_ids.
    We never search unrelated posts here: fallback stays inside the same event.
    """
    ordered = []

    raw_ranked = item.get("media_candidate_source_ids")
    if isinstance(raw_ranked, list):
        ordered.extend(raw_ranked)

    preferred = item.get("source_id")
    if isinstance(preferred, int):
        ordered.append(preferred)

    raw_sources = item.get("source_ids")
    if isinstance(raw_sources, list):
        ordered.extend(raw_sources)

    result = []
    seen = set()
    for source_idx in ordered:
        if (
            not isinstance(source_idx, int)
            or source_idx in seen
            or not (0 <= source_idx < len(posts))
        ):
            continue
        seen.add(source_idx)

        post = posts[source_idx]
        if post.get("is_priority"):
            # Manual media has a separate immutable lock path.
            continue
        if not (post.get("has_video") or post.get("has_photo") or post.get("has_media")):
            continue
        if not post.get("message_obj"):
            continue

        # Keep in sync with collector.MAX_VIDEO_SIZE. Oversized video cannot be
        # downloaded by the AUTO collector, so do not waste a fallback attempt.
        if post.get("has_video"):
            try:
                media_size = int(post.get("media_size") or 0)
            except (TypeError, ValueError):
                media_size = 0
            if media_size > 35 * 1024 * 1024:
                continue

        result.append(source_idx)
        if len(result) >= AUTO_MEDIA_FALLBACK_MAX_CANDIDATES:
            break

    return result


async def _resolve_auto_media_for_item(
    *,
    item: dict,
    posts: list,
    collector,
    publisher,
    history,
    used_media_fingerprints: set,
    news_index: int,
) -> dict:
    """
    Try event media one-by-one until one really survives publication gates.

    Sequence for every candidate:
    download -> Vision/video check -> 24h reuse-lock.
    If a candidate fails, try the next media source from THE SAME event. Text-only
    is used only after the safe candidate list is exhausted.
    """
    candidate_ids = _auto_media_candidate_source_ids(item, posts)
    preferred_source = item.get("source_id")

    result = {
        "source_idx": preferred_source if isinstance(preferred_source, int) else None,
        "media_path": None,
        "media_type": None,
        "media_file_id": None,
        "media_verdict": {},
        "media_fingerprint": None,
        "media_rejected": False,
        "media_reject_reason": "",
        "media_reuse_suppressed": False,
        "original_media_path": None,
        "original_media_type": None,
        "attempted_source_ids": [],
    }

    if not candidate_ids:
        logger.info(
            "MEDIA FALLBACK: news_index=%s event_id=%s has no AUTO candidates.",
            news_index,
            item.get("event_id"),
        )
        return result

    reject_reasons = []

    for attempt_no, source_idx in enumerate(candidate_ids, start=1):
        post = posts[source_idx]
        result["attempted_source_ids"].append(source_idx)

        try:
            media_path, media_type = await collector.download_post_media(
                post["message_obj"]
            )
        except Exception as dl_err:
            logger.warning(
                "MEDIA FALLBACK download failed: news_index=%s event_id=%s "
                "attempt=%s source_id=%s error=%s",
                news_index,
                item.get("event_id"),
                attempt_no,
                source_idx,
                dl_err,
            )
            continue

        if not media_path or media_type not in {"photo", "video"}:
            logger.info(
                "MEDIA FALLBACK candidate unavailable: news_index=%s event_id=%s "
                "attempt=%s source_id=%s",
                news_index,
                item.get("event_id"),
                attempt_no,
                source_idx,
            )
            continue

        if result["original_media_path"] is None:
            result["original_media_path"] = media_path
            result["original_media_type"] = media_type

        # Backup videos are always Vision-checked. This prevents a second random
        # clip from slipping through merely because the first one was rejected.
        force_video_validation = bool(
            item.get("video_validation_needed", False)
            or (
                media_type == "video"
                and source_idx != preferred_source
            )
        )

        verdict = await publisher.validate_media_for_news(
            text=item["text"],
            media_path=media_path,
            media_type=media_type,
            video_validation_needed=force_video_validation,
        )
        result["media_verdict"] = verdict

        if not verdict.get("is_relevant", False):
            result["media_rejected"] = True
            reason = str(verdict.get("reason") or "")
            if reason:
                reject_reasons.append(
                    f"source_id={source_idx}: {reason}"
                )
            logger.warning(
                "MEDIA CANDIDATE REJECTED: news_index=%s event_id=%s "
                "attempt=%s/%s source_id=%s type=%s reason=%s",
                news_index,
                item.get("event_id"),
                attempt_no,
                len(candidate_ids),
                source_idx,
                media_type,
                reason,
            )
            continue

        media_fingerprint = publisher.build_media_fingerprint(
            media_path,
            media_type,
        )
        fingerprint_key = (
            media_type,
            media_fingerprint,
        )

        if (
            media_fingerprint
            and (
                fingerprint_key in used_media_fingerprints
                or history.was_media_recently_used(
                    media_fingerprint,
                    media_type,
                    hours=24,
                )
            )
        ):
            result["media_reuse_suppressed"] = True
            logger.info(
                "MEDIA CANDIDATE REUSE-SKIP: news_index=%s event_id=%s "
                "attempt=%s/%s source_id=%s type=%s fingerprint=%s",
                news_index,
                item.get("event_id"),
                attempt_no,
                len(candidate_ids),
                source_idx,
                media_type,
                media_fingerprint[:24],
            )
            continue

        result.update({
            "source_idx": source_idx,
            "media_path": media_path,
            "media_type": media_type,
            "media_fingerprint": media_fingerprint,
        })
        result["media_reject_reason"] = " | ".join(reject_reasons)[:1500]

        if attempt_no > 1 or source_idx != preferred_source:
            logger.info(
                "MEDIA FALLBACK SELECTED: news_index=%s event_id=%s "
                "attempt=%s/%s source_id=%s type=%s",
                news_index,
                item.get("event_id"),
                attempt_no,
                len(candidate_ids),
                source_idx,
                media_type,
            )
        return result

    result["media_reject_reason"] = " | ".join(reject_reasons)[:1500]
    logger.warning(
        "MEDIA FALLBACK EXHAUSTED: news_index=%s event_id=%s candidates=%s. "
        "Публікуємо text-only, бо жодне медіа цієї події не пройшло gates.",
        news_index,
        item.get("event_id"),
        candidate_ids,
    )
    return result



async def _prepare_publication_item(
    *,
    item: dict,
    posts: list,
    collector,
    publisher,
    history,
    reserved_media_fingerprints: set,
    news_index: int,
) -> dict | None:
    """
    PREPARE-фаза для одного елемента дайджесту.

    Тут виконується все повільне й потенційно нестабільне ДО першої публікації:
    - manual media lock;
    - AUTO download;
    - photo/video validation;
    - fallback між media-кандидатами;
    - same-cycle media reservation.

    Після повернення цього dict Telegram-публікація вже не повинна чекати Gemini
    або перебирати інші медіа. Вона лише відправляє зафіксований результат.
    """
    source_idx = item.get("source_id")

    manual_media_source_idx = _locked_manual_media_source_for_item(
        item,
        posts,
    )
    manual_media_locked = manual_media_source_idx is not None
    if manual_media_locked:
        source_idx = manual_media_source_idx
        logger.info(
            "MANUAL MEDIA LOCK (prepare): news_index=%s event_id=%s source_id=%s",
            news_index,
            item.get("event_id"),
            source_idx,
        )

    target_post = (
        posts[source_idx]
        if (
            isinstance(source_idx, int)
            and 0 <= source_idx < len(posts)
        )
        else None
    )

    if not target_post:
        logger.warning(
            "PUBLICATION PREP SKIP: news_index=%s event_id=%s "
            "некоректний source_id=%r",
            news_index,
            item.get("event_id"),
            source_idx,
        )
        return None

    media_path = None
    media_type = None
    media_file_id = None
    media_verdict = {}
    media_rejected = False
    media_reject_reason = ""
    media_reuse_suppressed = False
    media_fingerprint = None
    original_media_path = None
    original_media_file_id = None
    original_media_type = None
    attempted_media_source_ids = []

    if manual_media_locked:
        media_type = str(
            target_post.get("manual_media_type") or ""
        ).strip().lower()
        candidate_path = str(
            target_post.get("manual_media_path") or ""
        ).strip()
        if candidate_path and os.path.exists(candidate_path):
            media_path = candidate_path

        media_file_id = str(
            target_post.get("manual_telegram_file_id")
            or target_post.get("telegram_file_id")
            or ""
        ).strip() or None

        # Manual media не має права тихо деградувати до text-only. Якщо файл
        # справді втрачено, item не входить до prepared batch, а queue лишається
        # pending для наступного циклу/повторної відправки.
        if (
            media_type not in {"photo", "video"}
            or not (media_path or media_file_id)
        ):
            logger.error(
                "MANUAL MEDIA LOCK FAILED DURING PREP: news_index=%s event_id=%s "
                "source_id=%s path=%s file_id=%s type=%s. Queue лишається pending.",
                news_index,
                item.get("event_id"),
                source_idx,
                media_path,
                bool(media_file_id),
                media_type,
            )
            return None

        original_media_path = media_path
        original_media_file_id = media_file_id
        original_media_type = media_type
        media_verdict = {
            "is_relevant": True,
            "confidence": 100,
            "reason": "manual_media_locked_no_validation",
            "media_type": media_type,
        }
    else:
        auto_media = await _resolve_auto_media_for_item(
            item=item,
            posts=posts,
            collector=collector,
            publisher=publisher,
            history=history,
            used_media_fingerprints=reserved_media_fingerprints,
            news_index=news_index,
        )

        selected_source_idx = auto_media.get("source_idx")
        if isinstance(selected_source_idx, int):
            source_idx = selected_source_idx
            target_post = posts[source_idx]

        media_path = auto_media.get("media_path")
        media_type = auto_media.get("media_type")
        media_file_id = auto_media.get("media_file_id")
        media_verdict = auto_media.get("media_verdict") or {}
        media_rejected = bool(auto_media.get("media_rejected"))
        media_reject_reason = str(
            auto_media.get("media_reject_reason") or ""
        )
        media_reuse_suppressed = bool(
            auto_media.get("media_reuse_suppressed")
        )
        media_fingerprint = auto_media.get("media_fingerprint")
        original_media_path = auto_media.get("original_media_path")
        original_media_type = auto_media.get("original_media_type")
        attempted_media_source_ids = list(
            auto_media.get("attempted_source_ids") or []
        )

        # Резервуємо медіа вже під час PREPARE, а не після publish. Інакше дві
        # новини, підготовлені до старту випуску, могли б вибрати той самий файл.
        # У persistent history записуємо лише ПІСЛЯ фактичної Telegram-публікації.
        if (
            media_path
            and media_type in {"photo", "video"}
            and media_fingerprint
        ):
            reserved_media_fingerprints.add(
                (media_type, media_fingerprint)
            )

    # Посилання додаємо ПІСЛЯ media validation, щоб службовий рядок не впливав
    # на Vision/video gate.
    publication_text = append_reference_link(
        item["text"],
        item.get("reference_url"),
        item.get("reference_label"),
    )

    logger.info(
        "PUBLICATION PREP READY: news_index=%s event_id=%s media=%s "
        "manual_locked=%s attempts=%s",
        news_index,
        item.get("event_id"),
        media_type or "text-only",
        manual_media_locked,
        len(attempted_media_source_ids),
    )

    return {
        "news_index": news_index,
        "item": item,
        "source_idx": source_idx,
        "target_post": target_post,
        "manual_media_locked": manual_media_locked,
        "publication_text": publication_text,
        "media_path": media_path,
        "media_type": media_type,
        "media_file_id": media_file_id,
        "media_verdict": media_verdict,
        "media_rejected": media_rejected,
        "media_reject_reason": media_reject_reason,
        "media_reuse_suppressed": media_reuse_suppressed,
        "media_fingerprint": media_fingerprint,
        "original_media_path": original_media_path,
        "original_media_file_id": original_media_file_id,
        "original_media_type": original_media_type,
        "attempted_media_source_ids": attempted_media_source_ids,
    }

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
                "manual_telegram_file_id": str(
                    manual.get("telegram_file_id") or ""
                ),
                "manual_telegram_file_unique_id": str(
                    manual.get("telegram_file_unique_id") or ""
                ),
                "manual_telegram_file_size": int(
                    manual.get("telegram_file_size") or 0
                ),
                "manual_queue_id": queue_id,
                "is_priority": True,
                # Media attached by admin is immutable. Summarizer may enrich
                # text from other sources, but publication must use the admin
                # media either from local path OR Telegram file_id.
                "media_locked": bool(
                    manual.get("media_path")
                    or manual.get("telegram_file_id")
                ),
                "media_document_id": None,
                "video_duration": None,
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

        # Hard publication boundary: a manual story must contain ONLY facts from
        # its admin source(s). This is independent of Analyzer/Editor correctness.
        top_news = _enforce_manual_fact_lock_before_publish(
            top_news,
            posts,
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

        # 4. PREPARE ALL: спочатку повністю готуємо ВЕСЬ Telegram batch.
        #
        # Раніше header + news #1 виходили одразу, а news #2 могла потім на 5 хв
        # зависнути на download/Gemini media validation. Тепер читач не бачить
        # внутрішню підготовку: Telegram стартує лише коли для кожної новини вже
        # зафіксовано video/photo/text-only.
        prepare_started = time.monotonic()
        prepared_news = []
        reserved_media_fingerprints = set()

        logger.info(
            "PUBLICATION PREP START: готуємо %s новин ДО першого Telegram-поста.",
            len(top_news),
        )

        for index, item in enumerate(top_news, start=1):
            try:
                prepared = await _prepare_publication_item(
                    item=item,
                    posts=posts,
                    collector=collector,
                    publisher=publisher,
                    history=history,
                    reserved_media_fingerprints=reserved_media_fingerprints,
                    news_index=index,
                )
            except Exception as prep_exc:
                # AUTO item краще пропустити/залишити manual pending, ніж почати
                # напівготовий випуск і зависнути посередині.
                logger.error(
                    "PUBLICATION PREP FAILED: news_index=%s event_id=%s error=%s",
                    index,
                    item.get("event_id"),
                    prep_exc,
                    exc_info=True,
                )
                prepared = None

            if prepared is not None:
                prepared_news.append(prepared)

        prepare_elapsed = time.monotonic() - prepare_started
        logger.info(
            "PUBLICATION PREP COMPLETE: ready=%s/%s elapsed=%.1fs. "
            "Тепер запускаємо безперервну Telegram-публікацію.",
            len(prepared_news),
            len(top_news),
            prepare_elapsed,
        )

        if not prepared_news:
            logger.error(
                "Після PREPARE-фази немає жодної готової новини. "
                "Header не публікуємо."
            )
            return

        # 5. PUBLISH ALL: header + уже повністю підготовлені новини по черзі.
        # Ніяких Gemini/media-selection між Telegram-постами тут більше немає.
        header_text = get_slot_header_text(len(prepared_news))
        header_published = await publisher.publish_telegram_post(
            text=header_text
        )

        if not header_published:
            logger.error(
                "Не вдалося опублікувати header. Цикл зупинено."
            )
            return

        await asyncio.sleep(2)

        ig_media_items = []
        published_news = []
        published_manual_ids = set()

        for publish_position, prepared in enumerate(prepared_news, start=1):
            item = prepared["item"]
            original_index = prepared["news_index"]
            source_idx = prepared["source_idx"]
            target_post = prepared["target_post"]
            manual_media_locked = prepared["manual_media_locked"]
            publication_text = prepared["publication_text"]
            media_path = prepared["media_path"]
            media_type = prepared["media_type"]
            media_file_id = prepared["media_file_id"]
            media_verdict = prepared["media_verdict"]
            media_rejected = prepared["media_rejected"]
            media_reject_reason = prepared["media_reject_reason"]
            media_reuse_suppressed = prepared["media_reuse_suppressed"]
            media_fingerprint = prepared["media_fingerprint"]
            original_media_path = prepared["original_media_path"]
            original_media_file_id = prepared["original_media_file_id"]
            original_media_type = prepared["original_media_type"]
            attempted_media_source_ids = prepared["attempted_media_source_ids"]

            logger.info(
                "PUBLISH READY ITEM: position=%s/%s original_index=%s event_id=%s "
                "media=%s",
                publish_position,
                len(prepared_news),
                original_index,
                item.get("event_id"),
                media_type or "text-only",
            )

            published = await publisher.publish_telegram_post(
                text=publication_text,
                media_path=media_path,
                media_type=media_type,
                media_file_id=media_file_id,
                # Усе вже перевірено під час PREPARE; manual взагалі bypass.
                validate_media=False,
                # Manual не має права тихо впасти до text-only.
                require_media=manual_media_locked,
            )

            if not published:
                logger.warning(
                    "Новина position=%s original_index=%s не опублікована.",
                    publish_position,
                    original_index,
                )
                await asyncio.sleep(3)
                continue

            # Same-cycle fingerprint уже зарезервовано у PREPARE. У persistent
            # history записуємо лише реально опубліковане AUTO media.
            if (
                not manual_media_locked
                and media_path
                and media_type in {"photo", "video"}
                and media_fingerprint
            ):
                history.record_media_used(
                    media_fingerprint,
                    media_type,
                    source_kind="auto",
                )

            published_item = dict(item)
            # source_id in audit/runtime must reflect the media source actually
            # published after the independent manual-media lock.
            published_item["source_id"] = source_idx
            published_item["text"] = publication_text

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
            # публікації конкретного item. PREPARE сам по собі state не змінює.
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
            published_item["_audit_media"] = {
                "original_path": original_media_path,
                "original_file_id": bool(original_media_file_id),
                "original_type": original_media_type,
                "rejected": media_rejected,
                "reject_reason": media_reject_reason,
                "validation_reason": str(media_verdict.get("reason") or ""),
                "validation_confidence": media_verdict.get("confidence"),
                "reuse_suppressed": media_reuse_suppressed,
                "final_path": media_path,
                "final_file_id": bool(media_file_id),
                "final_type": media_type,
                "manual_locked": manual_media_locked,
                "manual_expected_path": (
                    str(target_post.get("manual_media_path") or "")
                    if manual_media_locked
                    else ""
                ),
                "manual_expected_file_id": (
                    bool(
                        target_post.get("manual_telegram_file_id")
                        or target_post.get("telegram_file_id")
                    )
                    if manual_media_locked
                    else False
                ),
                "selected_source_id": source_idx,
                "attempted_source_ids": attempted_media_source_ids,
                "fallback_used": bool(
                    attempted_media_source_ids
                    and isinstance(item.get("source_id"), int)
                    and source_idx != item.get("source_id")
                    and media_path
                ),
                # Нове поле для audit/debug: це медіа було підготовлено до старту
                # Telegram batch, а не вибиралося вже між постами.
                "prepared_before_batch": True,
            }

            published_news.append(published_item)

            if (
                media_path
                and media_type in {"photo", "video"}
            ):
                ig_media_items.append({
                    "path": media_path,
                    "type": media_type,
                })

                logger.info(
                    "Instagram media #%s: %s → %s",
                    publish_position,
                    media_type,
                    media_path,
                )

            first_line = publication_text.strip().split("\n")[0]

            # Зберігаємо в історію ВСІ source_ids події,
            # а не тільки пост, з якого взяли медіа.
            source_ids = item.get("source_ids")
            if not isinstance(source_ids, list):
                source_ids = [source_idx]
            else:
                # Не мутуємо item із top_news під час history bookkeeping.
                source_ids = list(source_ids)

            if source_idx not in source_ids:
                source_ids.append(source_idx)

            seen_source_ids = set()
            for event_source_idx in source_ids:
                if (
                    not isinstance(event_source_idx, int)
                    or event_source_idx in seen_source_ids
                    or not (0 <= event_source_idx < len(posts))
                ):
                    continue

                seen_source_ids.add(event_source_idx)
                event_post = posts[event_source_idx]

                history_channel = (
                    event_post.get("channel_username")
                    or event_post.get("channel_name")
                    or event_post.get("channel_title")
                    or "unknown"
                )
                history_message_id = event_post.get("message_id")
                if not isinstance(history_message_id, int):
                    continue

                history.mark_as_published(
                    channel_name=history_channel,
                    message_id=history_message_id,
                    title=first_line,
                    # Зберігаємо повний редакторський текст без URL, щоб наступні
                    # цикли мали сильніший semantic history context.
                    summary=item.get(
                        "text",
                        item.get("summary", ""),
                    ),
                    category=item.get("category", ""),
                )

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


