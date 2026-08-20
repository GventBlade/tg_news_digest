import os
import logging
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional
from telethon import TelegramClient
from app.config import settings
from app.services.history import NewsHistory

logger = logging.getLogger(__name__)
DOWNLOAD_DIR = "downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)


class NewsCollector:
    def __init__(self, session_name: str = "news_session"):
        self.client = TelegramClient(
            session_name,
            settings.TG_API_ID,
            settings.TG_API_HASH
        )
        self.history = NewsHistory()

    async def fetch_recent_posts(self, hours: int = 6, limit_per_channel: int = 10) -> List[Dict[str, Any]]:
        await self.client.start()
        time_threshold = datetime.now(timezone.utc) - timedelta(hours=hours)
        collected = []

        for channel in settings.source_channels_list:
            try:
                entity = await self.client.get_entity(channel)
                async for message in self.client.iter_messages(entity, limit=limit_per_channel):
                    if message.date < time_threshold:
                        break

                    # 1. Пропускаємо, якщо новину з цього каналу вже публікували
                    if self.history.is_published(str(channel), message.id):
                        continue

                    text = message.text or message.message
                    if not text or len(text.strip()) < 35:
                        continue

                    # Фіксуємо медіа
                    is_video = bool(
                        message.video or
                        (message.document and message.document.mime_type and message.document.mime_type.startswith(
                            "video/")) or
                        (message.file and message.file.name and message.file.name.lower().endswith((".mp4", ".mov")))
                    )
                    has_media = bool(message.photo or is_video)

                    collected.append({
                        "channel_name": str(channel),
                        "message_id": message.id,
                        "channel_title": getattr(entity, 'title', channel),
                        "text": text.strip(),
                        "message_obj": message,
                        "has_media": has_media,
                        "has_video": is_video,
                        "date": message.date
                    })

            except Exception as e:
                logger.error(f"Помилка зчитування з @{channel}: {e}")

        return collected

    async def download_post_media(self, message_obj) -> tuple[Optional[str], Optional[str]]:
        """Завантажує файл тільки тоді, коли Gemini точно обрала цей пост"""
        try:
            if message_obj.photo:
                path = await message_obj.download_media(file=DOWNLOAD_DIR)
                return path, "photo"

            # Перевірка на відео (і як video, і як document mp4)
            is_video = (
                    message_obj.video or
                    (
                                message_obj.document and message_obj.document.mime_type and message_obj.document.mime_type.startswith(
                            "video/")) or
                    (message_obj.file and message_obj.file.name and message_obj.file.name.lower().endswith(
                        (".mp4", ".mov")))
            )

            if is_video:
                file_size = getattr(message_obj.file, "size", 0) if message_obj.file else 0
                if 0 < file_size < 35 * 1024 * 1024:  # обмеження до 35 МБ
                    path = await message_obj.download_media(file=DOWNLOAD_DIR)
                    return path, "video"
        except Exception as e:
            logger.error(f"Не вдалося завантажити медіа: {e}")
        return None, None

    async def close(self):
        await self.client.disconnect()
