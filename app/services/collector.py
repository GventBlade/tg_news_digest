import logging
import os
import re
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from telethon import TelegramClient

from app.config import settings
from app.services.history import NewsHistory

logger = logging.getLogger(__name__)

DOWNLOAD_DIR = "downloads"
MAX_VIDEO_SIZE = 35 * 1024 * 1024

os.makedirs(DOWNLOAD_DIR, exist_ok=True)


class NewsCollector:
    def __init__(self, session_name: str = "news_session"):
        self.client = TelegramClient(
            session_name,
            settings.TG_API_ID,
            settings.TG_API_HASH,
        )
        self.history = NewsHistory()

    async def fetch_recent_posts(
        self,
        hours: int = 4,
        limit_per_channel: int = 30,
    ) -> List[Dict[str, Any]]:
        """
        Збирає пости за останні `hours` годин.

        Важливо:
        - не відсіює короткі новини за довжиною;
        - відсіює лише повністю порожні текстові повідомлення;
        - пропускає повідомлення, які вже були опубліковані раніше;
        - зберігає інформацію про фото/відео, engagement і час публікації;
        - зберігає зовнішні URL із тексту/inline-link/web-preview, щоб
          фінальний редактор міг додати реальне посилання на закон,
          дослідження, звіт або статтю без вигадування URL;
        - НЕ намагається оцінювати зміст картинки на етапі збору: фактична
          vision-перевірка фото виконується у Publisher лише для фіналу.
        """
        if not self.client.is_connected():
            await self.client.start()

        time_threshold = datetime.now(timezone.utc) - timedelta(hours=hours)
        collected: List[Dict[str, Any]] = []

        for channel in settings.source_channels_list:
            try:
                entity = await self.client.get_entity(channel)

                channel_username = (
                    getattr(entity, "username", None)
                    or str(channel)
                )
                channel_username = (
                    str(channel_username)
                    .replace("@", "")
                    .strip()
                )

                channel_title = (
                    getattr(entity, "title", None)
                    or channel_username
                )

                async for message in self.client.iter_messages(
                    entity,
                    limit=limit_per_channel,
                ):
                    if not message.date:
                        continue

                    if message.date < time_threshold:
                        break

                    if self.history.is_published(
                        channel_username,
                        message.id,
                    ):
                        continue

                    text = (
                        message.text
                        or message.message
                        or ""
                    ).strip()

                    has_video = self._is_video(message)
                    has_photo = bool(message.photo)
                    has_media = has_photo or has_video

                    if not text:
                        continue

                    views = int(
                        getattr(message, "views", 0) or 0
                    )
                    forwards = int(
                        getattr(message, "forwards", 0) or 0
                    )
                    replies = int(
                        message.replies.replies
                        if message.replies
                        else 0
                    )

                    external_links = self._extract_external_links(
                        message,
                        text,
                    )

                    source_post_url = None
                    if channel_username:
                        source_post_url = (
                            f"https://t.me/{channel_username}/{message.id}"
                        )

                    collected.append({
                        "channel_name": channel_username,
                        "channel_username": channel_username,
                        "channel_title": channel_title,
                        "message_id": message.id,
                        "text": text,
                        "message_obj": message,
                        "has_media": has_media,
                        "has_video": has_video,
                        "has_photo": has_photo,
                        "views": views,
                        "forwards": forwards,
                        "replies": replies,
                        "media_size": self._get_media_size(message),
                        "date": message.date,
                        "is_priority": False,
                        # Зовнішні посилання НЕ є Telegram source URL.
                        # Вони використовуються лише як можливе першоджерело
                        # документа/дослідження/статті у фінальному пості.
                        "external_links": external_links,
                        "source_post_url": source_post_url,
                        # Корисно для логів/майбутньої роботи з альбомами.
                        # Поточний pipeline лишається повністю сумісним.
                        "media_group_id": getattr(message, "grouped_id", None),
                    })

            except Exception as e:
                logger.error(
                    f"Помилка зчитування з @{channel}: {e}"
                )

        logger.info(
            f"Зібрано {len(collected)} сирих новин "
            f"за останні {hours} год."
        )

        return collected

    @classmethod
    def _extract_external_links(
        cls,
        message,
        text: str,
    ) -> List[Dict[str, str]]:
        """
        Витягує реальні зовнішні URL із Telegram-поста.

        Підтримує:
        - inline TextUrl;
        - звичайні http(s) URL у тексті;
        - web-preview;
        - URL-кнопки.

        Telegram-посилання t.me сюди не додаємо: потрібні лінки саме на
        статті/закони/звіти, а не на ще один Telegram-пост.
        """
        result: List[Dict[str, str]] = []
        seen = set()

        def add(url: Any, label: Any = ""):
            normalized = cls._normalize_external_url(url)
            if not normalized or normalized in seen:
                return
            seen.add(normalized)
            result.append({
                "url": normalized,
                "label": str(label or "").strip()[:180],
            })

        # Найнадійніший спосіб отримати TextUrl без ручного UTF-16 slicing.
        try:
            getter = getattr(message, "get_entities_text", None)
            if callable(getter):
                for pair in getter() or []:
                    if not isinstance(pair, (tuple, list)) or len(pair) < 2:
                        continue
                    entity, entity_text = pair[0], pair[1]
                    hidden_url = getattr(entity, "url", None)
                    if hidden_url:
                        add(hidden_url, entity_text)
                    elif isinstance(entity_text, str):
                        if re.match(r"^https?://", entity_text.strip(), re.I):
                            add(entity_text, entity_text)
        except Exception:
            pass

        # Видимі URL у plain text — fallback і доповнення.
        for match in re.findall(r"https?://[^\s<>\]\[(){}]+", text or ""):
            add(match, match)

        # Web-preview інколи містить URL навіть коли у тексті видно лише anchor.
        preview_candidates = [
            getattr(message, "web_preview", None),
            getattr(message, "webpage", None),
        ]
        media = getattr(message, "media", None)
        if media is not None:
            preview_candidates.append(getattr(media, "webpage", None))

        for preview in preview_candidates:
            if preview is None:
                continue
            try:
                add(
                    getattr(preview, "url", None),
                    getattr(preview, "title", None) or "",
                )
            except Exception:
                continue

        # URL-кнопки під постом.
        try:
            buttons = getattr(message, "buttons", None) or []
            for row in buttons:
                row_items = row if isinstance(row, (list, tuple)) else [row]
                for button in row_items:
                    add(
                        getattr(button, "url", None),
                        getattr(button, "text", None) or "",
                    )
        except Exception:
            pass

        # Не передаємо в summarizer безмежну кількість рекламних/службових URL.
        return result[:8]

    @staticmethod
    def _normalize_external_url(url: Any) -> Optional[str]:
        value = str(url or "").strip()
        if not value:
            return None

        value = value.rstrip(".,;:!?)]}>'\"")
        try:
            parsed = urlparse(value)
        except Exception:
            return None

        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            return None

        host = (parsed.hostname or "").lower()
        telegram_hosts = {
            "t.me",
            "telegram.me",
            "telegram.org",
        }
        if host in telegram_hosts or any(
            host.endswith("." + blocked)
            for blocked in telegram_hosts
        ):
            return None

        return value

    async def download_post_media(
        self,
        message_obj,
    ) -> Tuple[Optional[str], Optional[str]]:
        if not message_obj:
            return None, None

        try:
            message_id = getattr(
                message_obj,
                "id",
                "media",
            )
            unique_id = uuid.uuid4().hex[:10]

            if message_obj.photo:
                target_path = os.path.join(
                    DOWNLOAD_DIR,
                    f"photo_{message_id}_{unique_id}.jpg",
                )

                path = await message_obj.download_media(
                    file=target_path
                )

                if path and os.path.exists(path):
                    os.chmod(path, 0o644)
                    return path, "photo"

            if self._is_video(message_obj):
                file_size = self._get_media_size(message_obj)

                if file_size and file_size > MAX_VIDEO_SIZE:
                    logger.warning(
                        f"Відео {message_id} пропущено: "
                        f"{file_size / 1024 / 1024:.1f} MB"
                    )
                    return None, None

                target_path = os.path.join(
                    DOWNLOAD_DIR,
                    f"video_{message_id}_{unique_id}.mp4",
                )

                path = await message_obj.download_media(
                    file=target_path
                )

                if path and os.path.exists(path):
                    os.chmod(path, 0o644)
                    return path, "video"

        except Exception as e:
            logger.error(
                f"Не вдалося завантажити медіа: {e}"
            )

        return None, None

    @staticmethod
    def _is_video(message) -> bool:
        if getattr(message, "video", None):
            return True

        document = getattr(
            message,
            "document",
            None,
        )
        mime_type = getattr(
            document,
            "mime_type",
            None,
        )

        if mime_type and mime_type.startswith("video/"):
            return True

        file_obj = getattr(
            message,
            "file",
            None,
        )
        file_name = getattr(
            file_obj,
            "name",
            None,
        )

        if file_name:
            return file_name.lower().endswith(
                (
                    ".mp4",
                    ".mov",
                    ".avi",
                    ".mkv",
                    ".webm",
                    ".m4v",
                )
            )

        return False

    @staticmethod
    def _get_media_size(message) -> int:
        file_obj = getattr(
            message,
            "file",
            None,
        )
        return int(
            getattr(file_obj, "size", 0) or 0
        )

    async def close(self):
        if self.client.is_connected():
            await self.client.disconnect()
