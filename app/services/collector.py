import logging
import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

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
        - зберігає інформацію про фото/відео, engagement і час публікації.
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

                    # Використовуємо фактичний username каналу, а не рядок
                    # із .env. Це прибирає розбіжності після перейменувань
                    # або різного регістру username.
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

                    # Довжина тексту не є критерієм якості новини.
                    # Короткий пост може бути важливою гарячою новиною.
                    #
                    # Порожні медіапости без підпису поки не передаємо
                    # Analyzer, оскільки він аналізує текст і метадані,
                    # а не вміст самого зображення/відео.
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
