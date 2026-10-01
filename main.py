import asyncio
import html
import json
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

                # Manual media/bundle files may need to survive one or more
                # failed publication cycles. Keep them for at least 24h; AUTO
                # downloads still use the normal cleanup window.
                effective_max_age_minutes = max_age_minutes
                if filename.startswith("manual_"):
                    effective_max_age_minutes = max(max_age_minutes, 24 * 60)

                if (
                    age_seconds
                    > effective_max_age_minutes * 60
                ):
                    os.unlink(
                        file_path
                    )

        except Exception as e:
            logger.warning(
                "Не вдалося видалити старий "
                f"файл {file_path}: {e}"
            )


async def _capture_manual_media_part(message) -> dict | None:
    """Capture one photo/video part and preserve Telegram file_id as fallback."""
    media_type = None
    telegram_file_id = ""
    telegram_file_unique_id = ""
    telegram_file_size = 0
    media_path = None

    os.makedirs("downloads", exist_ok=True)

    if message.photo:
        media = message.photo[-1]
        media_type = "photo"
        telegram_file_id = str(getattr(media, "file_id", "") or "")
        telegram_file_unique_id = str(getattr(media, "file_unique_id", "") or "")
        telegram_file_size = int(getattr(media, "file_size", 0) or 0)
        target_path = f"downloads/manual_{message.message_id}.jpg"
        try:
            file = await media.get_file()
            await file.download_to_drive(target_path)
            if os.path.exists(target_path):
                media_path = target_path
        except Exception as exc:
            logger.warning(
                "MANUAL MEDIA local download failed; залишаємо Telegram "
                "file_id fallback: %s",
                exc,
            )

    elif message.video:
        media = message.video
        media_type = "video"
        telegram_file_id = str(getattr(media, "file_id", "") or "")
        telegram_file_unique_id = str(getattr(media, "file_unique_id", "") or "")
        telegram_file_size = int(getattr(media, "file_size", 0) or 0)
        target_path = f"downloads/manual_{message.message_id}.mp4"
        try:
            file = await media.get_file()
            await file.download_to_drive(target_path)
            if os.path.exists(target_path):
                media_path = target_path
        except Exception as exc:
            logger.warning(
                "MANUAL MEDIA local download failed; залишаємо Telegram "
                "file_id fallback: %s",
                exc,
            )

    if media_type not in {"photo", "video"}:
        return None

    if not (media_path or telegram_file_id):
        return None

    return {
        "type": media_type,
        "path": media_path,
        "file_id": telegram_file_id,
        "file_unique_id": telegram_file_unique_id,
        "file_size": telegram_file_size,
        "message_id": int(message.message_id),
    }


MANUAL_ALBUM_QUIET_SECONDS = 5.0
MANUAL_ALBUM_DOWNLOAD_TIMEOUT_SECONDS = 90.0


def _manual_media_part_stub(message) -> dict | None:
    """
    Register an album part immediately, without waiting for file download.

    Telegram file_id is enough to re-send the media later, so the album can be
    counted correctly even when several large videos are still downloading in
    the background. This is the key invariant that prevents a fast pair of
    videos from closing a 9-10 item album prematurely.
    """
    media = None
    media_type = None
    if message.photo:
        media = message.photo[-1]
        media_type = "photo"
    elif message.video:
        media = message.video
        media_type = "video"

    if media is None or media_type is None:
        return None

    file_id = str(getattr(media, "file_id", "") or "").strip()
    file_unique_id = str(getattr(media, "file_unique_id", "") or "").strip()
    file_size = int(getattr(media, "file_size", 0) or 0)

    # For album buffering we require at least the Telegram file_id. Local path
    # is only an optional optimization for Instagram / re-upload fallback.
    if not file_id:
        return None

    return {
        "type": media_type,
        "path": None,
        "file_id": file_id,
        "file_unique_id": file_unique_id,
        "file_size": file_size,
        "message_id": int(message.message_id),
    }


def _manual_album_item_key(item: dict) -> str:
    return str(
        item.get("file_unique_id")
        or item.get("file_id")
        or item.get("message_id")
        or item.get("path")
        or ""
    )


def _persist_manual_album_manifest(path: str, media_group_id: str, items: list[dict]) -> None:
    """Rewrite one album manifest atomically enough for the single event loop."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "version": 2,
        "media_group_id": str(media_group_id),
        "items": list(items)[:10],
    }
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)
    os.chmod(path, 0o644)


async def _download_manual_album_part(
    application,
    media_group_id: str,
    part_key: str,
    message,
) -> None:
    """
    Download one album part in background.

    The part is already present in the buffer via file_id before this coroutine
    starts. Therefore a slow/failed download can never reduce the album count.
    """
    captured = None
    error = None
    try:
        captured = await asyncio.wait_for(
            _capture_manual_media_part(message),
            timeout=MANUAL_ALBUM_DOWNLOAD_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        error = "download_timeout"
    except Exception as exc:
        error = str(exc)
    finally:
        buffers = application.bot_data.setdefault("manual_album_buffers", {})
        bundle = buffers.get(str(media_group_id))
        if not bundle:
            # Queue may already be finalized and the process may have cleaned the
            # in-memory bundle. The Telegram file_id stored in the manifest is
            # still sufficient for publication.
            return

        if isinstance(captured, dict):
            for item in bundle.get("items", []):
                if _manual_album_item_key(item) == part_key:
                    # Preserve the immediately captured file_id, enrich only with
                    # the finished local download / metadata.
                    if captured.get("path"):
                        item["path"] = captured.get("path")
                    if captured.get("file_id"):
                        item["file_id"] = captured.get("file_id")
                    if captured.get("file_unique_id"):
                        item["file_unique_id"] = captured.get("file_unique_id")
                    if captured.get("file_size"):
                        item["file_size"] = int(captured.get("file_size") or 0)
                    break

        bundle["inflight_downloads"] = max(
            0,
            int(bundle.get("inflight_downloads") or 0) - 1,
        )

        manifest_path = str(bundle.get("manifest_path") or "").strip()
        if manifest_path:
            try:
                _persist_manual_album_manifest(
                    manifest_path,
                    str(media_group_id),
                    list(bundle.get("items") or [])[:10],
                )
            except Exception as manifest_exc:
                logger.warning(
                    "MANUAL ALBUM manifest refresh failed group=%s part=%s: %s",
                    media_group_id,
                    part_key,
                    manifest_exc,
                )

        logger.info(
            "MANUAL ALBUM download finished: media_group_id=%s part=%s "
            "path=%s inflight=%s%s",
            media_group_id,
            part_key,
            bool(captured and captured.get("path")),
            bundle.get("inflight_downloads"),
            f" error={error}" if error else "",
        )

        # Once a finalized album has no background downloads left, it no longer
        # needs to occupy bot_data. The manifest already contains all file_ids.
        if bundle.get("finalized") and int(bundle.get("inflight_downloads") or 0) == 0:
            buffers.pop(str(media_group_id), None)


def _manual_forward_source_info(message) -> tuple[str, str]:
    channel_title = "Пріоритет (Адмін)"
    channel_username = ""
    if message.forward_origin:
        origin = message.forward_origin
        if hasattr(origin, "chat") and origin.chat:
            channel_title = origin.chat.title or channel_title
            channel_username = origin.chat.username or ""
    return channel_title, channel_username


def _write_manual_album_manifest(media_group_id: str, items: list[dict]) -> str:
    safe_group = "".join(ch for ch in str(media_group_id) if ch.isalnum() or ch in "-_")
    safe_group = (safe_group or "group")[:80]
    path = f"downloads/manual_album_{safe_group}_{int(time.time() * 1000)}.json"
    _persist_manual_album_manifest(path, str(media_group_id), list(items)[:10])
    return path


def _read_manual_album_manifest(path: str | None) -> list[dict]:
    manifest_path = str(path or "").strip()
    if not manifest_path or not os.path.exists(manifest_path):
        return []
    try:
        with open(manifest_path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except Exception as exc:
        logger.error("MANUAL ALBUM manifest read failed %s: %s", manifest_path, exc)
        return []

    raw_items = payload.get("items", []) if isinstance(payload, dict) else []
    result = []
    seen = set()
    for raw in raw_items:
        if not isinstance(raw, dict):
            continue
        media_type = str(raw.get("type") or "").strip().lower()
        media_path = str(raw.get("path") or "").strip() or None
        file_id = str(raw.get("file_id") or "").strip() or None
        unique_id = str(raw.get("file_unique_id") or "").strip()
        if media_type not in {"photo", "video"} or not (media_path or file_id):
            continue
        key = unique_id or file_id or media_path
        if key in seen:
            continue
        seen.add(key)
        result.append({
            "type": media_type,
            "path": media_path,
            "file_id": file_id,
            "file_unique_id": unique_id,
            "file_size": int(raw.get("file_size") or 0),
            "message_id": raw.get("message_id"),
        })
        if len(result) >= 10:
            break
    return result


async def _finalize_manual_album(application, media_group_id: str):
    """
    Finalize exactly one manual queue row after the Telegram media group is quiet.

    Important: all album parts are registered synchronously via file_id BEFORE
    any local download starts. The quiet timer therefore measures Telegram
    delivery, not download speed. Slow videos can keep downloading after the
    queue row is created; they only enrich the manifest with local paths.
    """
    key = str(media_group_id)
    try:
        while True:
            buffers = application.bot_data.setdefault("manual_album_buffers", {})
            bundle = buffers.get(key)
            if not bundle:
                return

            last_update = float(bundle.get("last_update_monotonic") or time.monotonic())
            idle_for = time.monotonic() - last_update
            remaining = MANUAL_ALBUM_QUIET_SECONDS - idle_for
            if remaining <= 0:
                break
            await asyncio.sleep(min(max(remaining, 0.05), MANUAL_ALBUM_QUIET_SECONDS))
    except asyncio.CancelledError:
        return

    buffers = application.bot_data.setdefault("manual_album_buffers", {})
    bundle = buffers.get(key)
    if not bundle or bundle.get("finalized"):
        return

    # Mark finalized BEFORE any await below so no second finalizer can create a
    # duplicate queue row for the same Telegram media_group_id.
    bundle["finalized"] = True

    raw_text = str(bundle.get("raw_text") or "").strip()
    chat_id = bundle.get("chat_id")
    items = list(bundle.get("items") or [])[:10]

    if not raw_text:
        logger.warning(
            "MANUAL ALBUM rejected: media_group_id=%s has no caption/text; items=%s.",
            media_group_id,
            len(items),
        )
        if chat_id is not None:
            await application.bot.send_message(
                chat_id=chat_id,
                text=(
                    "⚠️ Альбом отримано, але в ньому немає підпису. "
                    "Додай короткий текст до альбому й надішли ще раз."
                ),
            )
        if int(bundle.get("inflight_downloads") or 0) == 0:
            buffers.pop(key, None)
        return

    if not items:
        logger.error(
            "MANUAL ALBUM capture failed completely: media_group_id=%s",
            media_group_id,
        )
        if chat_id is not None:
            await application.bot.send_message(
                chat_id=chat_id,
                text=(
                    "⚠️ Не вдалося зафіксувати фото/відео альбому. "
                    "Новину не додано — надішли її ще раз."
                ),
            )
        if int(bundle.get("inflight_downloads") or 0) == 0:
            buffers.pop(key, None)
        return

    manifest_path = _write_manual_album_manifest(key, items)
    bundle["manifest_path"] = manifest_path

    history = NewsHistory()
    queue_id = history.add_manual_post(
        raw_text=raw_text,
        channel_title=str(bundle.get("channel_title") or "Пріоритет (Адмін)"),
        channel_username=str(bundle.get("channel_username") or ""),
        media_path=manifest_path,
        media_type="album",
        has_media=True,
        has_video=any(item.get("type") == "video" for item in items),
        telegram_file_id="",
        telegram_file_unique_id=f"album:{media_group_id}",
        telegram_file_size=sum(int(item.get("file_size") or 0) for item in items),
    )
    bundle["queue_id"] = queue_id
    finalized_groups = application.bot_data.setdefault(
        "manual_album_finalized_groups", {}
    )
    finalized_groups[key] = time.monotonic() + 60.0

    photos = sum(1 for item in items if item.get("type") == "photo")
    videos = sum(1 for item in items if item.get("type") == "video")
    local_ready = sum(1 for item in items if item.get("path") and os.path.exists(str(item.get("path"))))
    logger.info(
        "✅ Ручний альбом додано як ОДНУ новину "
        "(queue_id=%s, media_group_id=%s, photos=%s, videos=%s, "
        "items=%s, local_ready=%s, inflight=%s).",
        queue_id,
        media_group_id,
        photos,
        videos,
        len(items),
        local_ready,
        int(bundle.get("inflight_downloads") or 0),
    )
    if chat_id is not None:
        await application.bot.send_message(
            chat_id=chat_id,
            text=(
                "✅ Альбом збережено як одну ручну новину. "
                f"Зафіксовано медіа: {photos} фото, {videos} відео "
                f"(усього {len(items)}). "
                "У найближчому слоті вони підуть разом; факти беруться "
                "лише з твого підпису."
            ),
        )

    # Keep bundle only while background downloads enrich local paths for IG.
    if int(bundle.get("inflight_downloads") or 0) == 0:
        buffers.pop(key, None)


async def handle_admin_forwarded_message(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    user_id = update.effective_user.id if update.effective_user else None
    logger.info("📩 Отримано повідомлення від Telegram user_id: %s", user_id)

    if settings.ADMIN_TELEGRAM_ID is None:
        logger.error("ADMIN_TELEGRAM_ID не налаштований у .env.")
        return

    if user_id != settings.ADMIN_TELEGRAM_ID:
        logger.warning("⛔ Відхилено повідомлення від user_id %s.", user_id)
        return

    message = update.message
    if not message:
        return

    raw_text = message.text or message.caption or ""
    if raw_text.strip().startswith("/start"):
        await message.reply_text("👋 Бот активний і готовий приймати новини від адміна!")
        return

    channel_title, channel_username = _manual_forward_source_info(message)
    media_group_id = getattr(message, "media_group_id", None)

    # Telegram sends one album as several updates. IMPORTANT: register every
    # part immediately from Telegram file_id, then download local files in the
    # background. The handler must return quickly so all 2-10 updates can enter
    # the same buffer before the quiet timer closes the group.
    if media_group_id:
        buffers = context.application.bot_data.setdefault("manual_album_buffers", {})
        key = str(media_group_id)
        now_mono = time.monotonic()

        # Short tombstone prevents an extremely late Telegram update from
        # creating a second queue row after the album was already finalized.
        finalized_groups = context.application.bot_data.setdefault(
            "manual_album_finalized_groups", {}
        )
        for old_key, expires_at in list(finalized_groups.items()):
            if float(expires_at or 0) <= now_mono:
                finalized_groups.pop(old_key, None)
        if key in finalized_groups:
            logger.warning(
                "MANUAL ALBUM late Telegram part ignored (already finalized): "
                "media_group_id=%s message_id=%s",
                key,
                message.message_id,
            )
            return

        bundle = buffers.setdefault(key, {
            "raw_text": "",
            "channel_title": channel_title,
            "channel_username": channel_username,
            "chat_id": message.chat_id,
            "items": [],
            "task": None,
            "inflight_downloads": 0,
            "first_update_monotonic": now_mono,
            "last_update_monotonic": now_mono,
            "finalized": False,
            "manifest_path": None,
        })

        # A same media_group_id should never be reused after finalization, but if
        # Telegram delivers an extremely late update while local downloads are
        # still running, do not create a second queue row silently.
        if bundle.get("finalized"):
            logger.error(
                "MANUAL ALBUM late part ignored after finalization: "
                "media_group_id=%s message_id=%s",
                key,
                message.message_id,
            )
            return

        bundle["last_update_monotonic"] = now_mono
        if raw_text.strip():
            bundle["raw_text"] = raw_text.strip()
        if channel_title and channel_title != "Пріоритет (Адмін)":
            bundle["channel_title"] = channel_title
        if channel_username:
            bundle["channel_username"] = channel_username

        media_stub = _manual_media_part_stub(message)
        if media_stub:
            part_key = _manual_album_item_key(media_stub)
            known = {
                _manual_album_item_key(item)
                for item in bundle["items"]
            }
            if part_key and part_key not in known and len(bundle["items"]) < 10:
                # Register FIRST. Count is now correct regardless of download speed.
                bundle["items"].append(media_stub)
                bundle["inflight_downloads"] = int(
                    bundle.get("inflight_downloads") or 0
                ) + 1
                asyncio.create_task(
                    _download_manual_album_part(
                        context.application,
                        key,
                        part_key,
                        message,
                    )
                )
        else:
            logger.warning(
                "MANUAL ALBUM unsupported/empty media part: "
                "media_group_id=%s message_id=%s",
                key,
                message.message_id,
            )

        previous_task = bundle.get("task")
        if previous_task and not previous_task.done():
            previous_task.cancel()
        bundle["task"] = asyncio.create_task(
            _finalize_manual_album(context.application, key)
        )
        logger.info(
            "MANUAL ALBUM part registered: media_group_id=%s message_id=%s "
            "items=%s inflight=%s caption=%s",
            key,
            message.message_id,
            len(bundle["items"]),
            int(bundle.get("inflight_downloads") or 0),
            bool(bundle.get("raw_text")),
        )
        return

    # Non-album manual item: one message = one queue row.
    if not raw_text.strip():
        await message.reply_text(
            "⚠️ Додай короткий текст або підпис до новини. "
            "Без тексту Analyzer не зможе коректно оцінити подію."
        )
        return

    media_part = await _capture_manual_media_part(message)
    media_path = media_part.get("path") if media_part else None
    media_type = media_part.get("type") if media_part else None
    telegram_file_id = str(media_part.get("file_id") or "") if media_part else ""
    telegram_file_unique_id = (
        str(media_part.get("file_unique_id") or "") if media_part else ""
    )
    telegram_file_size = int(media_part.get("file_size") or 0) if media_part else 0

    has_manual_media = bool(
        media_type in {"photo", "video"}
        and (media_path or telegram_file_id)
    )

    if (message.photo or message.video) and not has_manual_media:
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
        has_video=(media_type == "video"),
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
        "(queue_id=%s, telegram_message_id=%s).",
        queue_id,
        message.message_id,
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

        manual_text_safe = bool(
            item_copy.get("manual_editor_verified")
            or item_copy.get("manual_editor_fallback")
            or item_copy.get("_main_manual_fallback")
        )

        if (
            bool(item_copy.get("manual_fact_locked"))
            and not foreign_sources
            and manual_text_safe
        ):
            item_copy["source_ids"] = list(manual_sources)
            item_copy["priority_source_ids"] = list(manual_sources)
            item_copy["manual_fact_source_ids"] = list(manual_sources)
            logger.info(
                "PRE-PUBLISH MANUAL FACT LOCK OK: event_id=%s source_ids=%s "
                "editor_verified=%s fallback=%s",
                item_copy.get("event_id"),
                manual_sources,
                bool(item_copy.get("manual_editor_verified")),
                bool(item_copy.get("manual_editor_fallback")),
            )
            result.append(item_copy)
            continue

        if bool(item_copy.get("manual_fact_locked")) and not foreign_sources:
            logger.error(
                "PRE-PUBLISH MANUAL TEXT UNVERIFIED: event_id=%s source_ids=%s. "
                "Discarding edited text and rebuilding from the admin source.",
                item_copy.get("event_id"),
                manual_sources,
            )

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
        album_items = post.get("manual_media_items") or []
        has_locked_media = bool(
            (media_type in {"photo", "video"} and (media_path or media_file_id))
            or (media_type == "album" and isinstance(album_items, list) and album_items)
        )
        if bool(post.get("is_priority")) and has_locked_media:
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
        media_type = str(
            target_post.get("manual_media_type") or ""
        ).strip().lower()

        if media_type == "album":
            raw_album_items = target_post.get("manual_media_items") or []
            for raw in raw_album_items:
                if not isinstance(raw, dict):
                    continue
                item_type = str(raw.get("type") or "").strip().lower()
                item_path = str(raw.get("path") or "").strip() or None
                item_file_id = str(raw.get("file_id") or "").strip() or None
                if item_type not in {"photo", "video"}:
                    continue
                if item_path and not os.path.exists(item_path):
                    item_path = None
                if not (item_path or item_file_id):
                    continue
                media_items.append({
                    "type": item_type,
                    "path": item_path,
                    "file_id": item_file_id,
                })
                if len(media_items) >= 10:
                    break

            if not media_items:
                logger.error(
                    "MANUAL ALBUM LOCK FAILED DURING PREP: news_index=%s "
                    "event_id=%s source_id=%s. Queue лишається pending.",
                    news_index,
                    item.get("event_id"),
                    source_idx,
                )
                return None

            original_media_path = str(target_post.get("manual_media_path") or "") or None
            original_media_type = "album"
            media_verdict = {
                "is_relevant": True,
                "confidence": 100,
                "reason": "manual_album_locked_no_validation",
                "media_type": "album",
            }
        else:
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
        "media_items": media_items,
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

            manual_media_type = str(manual.get("media_type") or "").strip().lower()
            manual_media_items = (
                _read_manual_album_manifest(manual.get("media_path"))
                if manual_media_type == "album"
                else []
            )
            manual_has_media = bool(
                manual_media_items
                if manual_media_type == "album"
                else manual.get("has_media")
            )
            manual_has_video = bool(
                any(item.get("type") == "video" for item in manual_media_items)
                if manual_media_type == "album"
                else manual.get("has_video")
            )
            manual_has_photo = bool(
                any(item.get("type") == "photo" for item in manual_media_items)
                if manual_media_type == "album"
                else manual_media_type == "photo"
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
                "has_media": manual_has_media,
                "has_video": manual_has_video,
                "has_photo": manual_has_photo,
                "manual_media_path": manual[
                    "media_path"
                ],
                "manual_media_type": manual_media_type,
                "manual_media_items": manual_media_items,
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
                    manual_media_items
                    or manual.get("media_path")
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
            media_items = prepared.get("media_items") or []
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
                (f"album[{len(media_items)}]" if media_items else (media_type or "text-only")),
            )

            published = await publisher.publish_telegram_post(
                text=publication_text,
                media_path=media_path,
                media_type=media_type,
                media_file_id=media_file_id,
                media_items=media_items,
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
                "final_type": ("album" if media_items else media_type),
                "final_media_items": len(media_items),
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

            if media_items:
                for album_position, album_item in enumerate(media_items, start=1):
                    album_path = str(album_item.get("path") or "").strip()
                    album_type = str(album_item.get("type") or "").strip().lower()
                    if (
                        album_path
                        and os.path.exists(album_path)
                        and album_type in {"photo", "video"}
                    ):
                        ig_media_items.append({
                            "path": album_path,
                            "type": album_type,
                        })
                        logger.info(
                            "Instagram media #%s.%s: %s → %s",
                            publish_position,
                            album_position,
                            album_type,
                            album_path,
                        )
            elif (
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


