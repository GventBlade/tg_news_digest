import logging
from aiogram import Bot
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.types import FSInputFile
from app.config import settings

logger = logging.getLogger(__name__)


class NewsPublisher:
    def __init__(self):
        # Сучасний спосіб налаштування HTML-форматування для aiogram 3.7+
        self.bot = Bot(
            token=settings.BOT_TOKEN,
            default=DefaultBotProperties(parse_mode=ParseMode.HTML)
        )

    async def publish_news(self, text: str, media_path: str = None, media_type: str = None):
        """
        Публікує новину в цільовий канал: текст, фото або відео з коректним форматуванням.
        """
        try:
            if media_path and media_type == "photo":
                photo = FSInputFile(media_path)
                await self.bot.send_photo(
                    chat_id=settings.TARGET_CHANNEL_ID,
                    photo=photo,
                    caption=text
                )
            elif media_path and media_type == "video":
                video = FSInputFile(media_path)
                await self.bot.send_video(
                    chat_id=settings.TARGET_CHANNEL_ID,
                    video=video,
                    caption=text,
                    supports_streaming=True
                )
            else:
                await self.bot.send_message(
                    chat_id=settings.TARGET_CHANNEL_ID,
                    text=text
                )
            logger.info("Новину успішно опубліковано в канал!")
        except Exception as e:
            logger.error(f"Помилка при публікації в канал: {e}")
        finally:
            await self.bot.session.close()
