import asyncio
import logging
import re
import aiohttp
from aiogram import Bot
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.types import FSInputFile
from app.config import settings

logger = logging.getLogger(__name__)


async def upload_temp_media(file_path: str) -> str:
    """Вивантажує локальне медіа на тимчасовий хостинг catbox для Instagram API."""
    url = "https://catbox.moe/user/api.php"
    async with aiohttp.ClientSession() as session:
        try:
            with open(file_path, "rb") as f:
                data = aiohttp.FormData()
                data.add_field("reqtype", "fileupload")
                filename = "video.mp4" if file_path.lower().endswith((".mp4", ".mov")) else "image.jpg"
                data.add_field("fileToUpload", f, filename=filename)
                async with session.post(url, data=data) as resp:
                    if resp.status == 200:
                        public_url = await resp.text()
                        return public_url.strip()
                    logger.warning(f"Не вдалося вивантажити {file_path} на catbox: статус {resp.status}")
        except Exception as e:
            logger.error(f"Виняток при вивантаженні {file_path}: {e}")
    return ""


class NewsPublisher:
    def __init__(self):
        self.bot = Bot(
            token=settings.BOT_TOKEN,
            default=DefaultBotProperties(parse_mode=ParseMode.HTML)
        )
        self.ig_account_id = settings.INSTAGRAM_ACCOUNT_ID
        self.ig_access_token = settings.INSTAGRAM_ACCESS_TOKEN
        self.graph_url = "https://graph.facebook.com/v26.0"

    @staticmethod
    def _strip_html(text: str) -> str:
        clean = re.compile("<.*?>")
        return re.sub(clean, "", text)

    async def publish_telegram_post(self, text: str, media_path: str = None, media_type: str = None):
        """Публікація окремого поста в Telegram."""
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
            logger.info("Пост опубліковано в Telegram.")
        except Exception as e:
            logger.error(f"Помилка публікації в Telegram: {e}")

    async def publish_instagram_carousel(self, caption: str, media_items: list):
        """
        Публікує пост-карусель в Instagram (від 1 до 10 медіа-елементів).
        media_items = [{"url": "https://...", "type": "photo"|"video"}, ...]
        """
        if not self.ig_account_id or not self.ig_access_token:
            logger.warning("Instagram credentials не налаштовані.")
            return

        if not media_items:
            logger.warning("Немає медіа для публікації в Instagram.")
            return

        clean_caption = self._strip_html(caption)

        async with aiohttp.ClientSession() as session:
            try:
                # Якщо лише 1 медіа — публікуємо одиночним постом
                if len(media_items) == 1:
                    m = media_items[0]
                    container_url = f"{self.graph_url}/{self.ig_account_id}/media"
                    params = {
                        "caption": clean_caption,
                        "access_token": self.ig_access_token
                    }
                    if m["type"] == "video":
                        params.update({"media_type": "VIDEO", "video_url": m["url"]})
                    else:
                        params.update({"image_url": m["url"]})

                    async with session.post(container_url, data=params) as resp:
                        c_data = await resp.json()
                        creation_id = c_data.get("id")
                else:
                    # Створення дочірніх контейнерів слайдів
                    child_ids = []
                    for item in media_items[:10]:
                        item_url = f"{self.graph_url}/{self.ig_account_id}/media"
                        params = {
                            "is_carousel_item": "true",
                            "access_token": self.ig_access_token
                        }
                        if item["type"] == "video":
                            params.update({"media_type": "VIDEO", "video_url": item["url"]})
                        else:
                            params.update({"image_url": item["url"]})

                        async with session.post(item_url, data=params) as resp:
                            res = await resp.json()
                            if "id" in res:
                                child_ids.append(res["id"])
                            else:
                                logger.error(f"Помилка створення слайда каруселі: {res}")

                    if not child_ids:
                        logger.error("Не вдалося створити елементи каруселі.")
                        return

                    await asyncio.sleep(4)

                    # Створення батьківського контейнера каруселі
                    parent_url = f"{self.graph_url}/{self.ig_account_id}/media"
                    parent_params = {
                        "media_type": "CAROUSEL",
                        "children": ",".join(child_ids),
                        "caption": clean_caption,
                        "access_token": self.ig_access_token
                    }
                    async with session.post(parent_url, data=parent_params) as resp:
                        parent_data = await resp.json()
                        creation_id = parent_data.get("id")

                if not creation_id:
                    logger.error("Не вдалося отримати ID контейнера Instagram.")
                    return

                await asyncio.sleep(5)

                # Фінальна публікація
                publish_url = f"{self.graph_url}/{self.ig_account_id}/media_publish"
                pub_params = {
                    "creation_id": creation_id,
                    "access_token": self.ig_access_token
                }
                async with session.post(publish_url, data=pub_params) as resp:
                    pub_res = await resp.json()
                    if "id" in pub_res:
                        logger.info(f"✅ Карусель із {len(media_items)} слайдів опубліковано в Instagram! ID: {pub_res['id']}")
                    else:
                        logger.error(f"Помилка публікації в Instagram: {pub_res}")

            except Exception as e:
                logger.error(f"Виняток при публікації в Instagram: {e}")

    async def close(self):
        await self.bot.session.close()
