import os
import uuid
import logging
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional, Tuple
from telethon import TelegramClient
from app.config import settings
from app.services.history import NewsHistory

logger = logging.getLogger(__name__)
DOWNLOAD_DIR = "downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)


class NewsCollector:
    def __init__(self, session_name: str = "news_session"):
        self.client = TelegramClient(session_name, settings.TG_API_ID, settings.TG_API_HASH)
        self.history = NewsHistory()

    async def fetch_recent_posts(self, hours: int = 4, limit_per_channel: int = 15) -> List[Dict[str, Any]]:
        """Збирає сирі Telegram-пости з каналів-донорів за вказаний проміжок часу."""
        if not self.client.is_connected():
            await self.client.start()

        time_threshold = datetime.now(timezone.utc) - timedelta(hours=hours)
        collected: List[Dict[str, Any]] = []

        for channel in settings.source_channels_list:
            try:
                entity = await self.client.get_entity(channel)
                channel_username = getattr(entity, "username", None) or str(channel)
                channel_username = str(channel_username).replace("@", "").strip()
                channel_title = getattr(entity, "title", channel_username)

                async for message in self.client.iter_messages(entity, limit=limit_per_channel):
                    if message.date < time_threshold:
                        break

                    if self.history.is_published(str(channel), message.id):
                        continue

                    text = (message.text or message.message or "").strip()
                    if len(text) < 35:
                        continue

                    is_video = self._is_video(message)
                    has_media = bool(message.photo or is_video)
                    views = int(getattr(message, "views", 0) or 0)
                    forwards = int(getattr(message, "forwards", 0) or 0)
                    replies = int(message.replies.replies if message.replies else 0)

                    collected.append({
                        "channel_name": str(channel),
                        "channel_username": channel_username,
                        "channel_title": channel_title,
                        "message_id": message.id,
                        "text": text,
                        "message_obj": message,
                        "has_media": has_media,
                        "has_video": is_video,
                        "views": views,
                        "forwards": forwards,
                        "replies": replies,
                        "date": message.date,
                    })

            except Exception as e:
                logger.error(f"Помилка зчитування з @{channel}: {e}")

        logger.info(f"Зібрано {len(collected)} сирих новин за останні {hours} год.")
        return collected

    @staticmethod
    def _is_video(message) -> bool:
        if message.video:
            return True
        if message.document and message.document.mime_type and message.document.mime_type.startswith("video/"):
            return True
        if message.file and message.file.name:
            if message.file.name.lower().endswith((".mp4", ".mov", ".avi", ".mkv", ".webm")):
                return True
        return False

    async def download_post_media(self, message_obj) -> Tuple[Optional[str], Optional[str]]:
        """Завантажує медіа лише для обраного поста з безпечним латинським іменем файлу."""
        try:
            # Генерація безпечного латинського імені файлу для Nginx URL
            unique_id = uuid.uuid4().hex[:10]
            msg_id = getattr(message_obj, "id", "media")

            if message_obj.photo:
                target_path = os.path.join(DOWNLOAD_DIR, f"photo_{msg_id}_{unique_id}.jpg")
                path = await message_obj.download_media(file=target_path)
                if path and os.path.exists(path):
                    # Права на читання для Nginx
                    os.chmod(path, 0o644)
                    return path, "photo"

            if self._is_video(message_obj):
                file_size = getattr(message_obj.file, "size", 0) or 0
                if 0 < file_size < 35 * 1024 * 1024:
                    target_path = os.path.join(DOWNLOAD_DIR, f"video_{msg_id}_{unique_id}.mp4")
                    path = await message_obj.download_media(file=target_path)
                    if path and os.path.exists(path):
                        os.chmod(path, 0o644)
                        return path, "video"
        except Exception as e:
            logger.error(f"Не вдалося завантажити медіа: {e}")

        return None, None

    async def close(self):
        if self.client.is_connected():
            await self.client.disconnect()
