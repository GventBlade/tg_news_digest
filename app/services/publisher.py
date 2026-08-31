import asyncio
import logging
import re
from pathlib import Path
from urllib.parse import quote
import aiohttp
from aiogram import Bot
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.types import FSInputFile
from app.config import settings

logger = logging.getLogger(__name__)


class NewsPublisher:
    def __init__(self):
        self.bot = Bot(
            token=settings.BOT_TOKEN,
            default=DefaultBotProperties(parse_mode=ParseMode.HTML),
        )
        self.ig_account_id = settings.INSTAGRAM_ACCOUNT_ID
        self.ig_access_token = settings.INSTAGRAM_ACCESS_TOKEN
        self.graph_url = "https://graph.facebook.com/v26.0"
        self.media_base_url = (settings.MEDIA_BASE_URL or "").rstrip("/")

    async def publish_telegram_post(
        self,
        text: str,
        media_path: str | None = None,
        media_type: str | None = None,
    ) -> bool:
        """Публікує пост у Telegram. Повертає True у разі успіху, інакше False."""
        try:
            if media_path and Path(media_path).exists():
                try:
                    if media_type == "photo":
                        await self.bot.send_photo(
                            chat_id=settings.TARGET_CHANNEL_ID,
                            photo=FSInputFile(media_path),
                            caption=text,
                        )
                        logger.info("Фото-пост опубліковано в Telegram.")
                        return True
                    elif media_type == "video":
                        await self.bot.send_video(
                            chat_id=settings.TARGET_CHANNEL_ID,
                            video=FSInputFile(media_path),
                            caption=text,
                            supports_streaming=True,
                        )
                        logger.info("Відео-пост опубліковано в Telegram.")
                        return True
                except Exception as media_err:
                    logger.warning(f"Не вдалося відправити з медіа ({media_err}), пробуємо відправити текстом...")

            await self.bot.send_message(
                chat_id=settings.TARGET_CHANNEL_ID,
                text=text,
            )
            logger.info("Текстовий пост опубліковано в Telegram.")
            return True

        except Exception as e:
            logger.error(f"Помилка публікації в Telegram: {e}", exc_info=True)
            return False

    def create_public_media_url(self, file_path: str) -> str:
        if not self.media_base_url:
            raise RuntimeError("MEDIA_BASE_URL не налаштований у .env")

        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Файл не знайдено: {file_path}")

        filename = quote(path.name)
        return f"{self.media_base_url}/media/{filename}"

    @staticmethod
    def _strip_html(text: str) -> str:
        clean = re.sub(r"<.*?>", "", text).strip()
        if len(clean) > 2150:
            clean = clean[:2140] + "...\n(продовження в Telegram)"
        return clean

    async def _create_container(
        self,
        session: aiohttp.ClientSession,
        media_url: str,
        media_type: str,
        caption: str | None = None,
        is_carousel_item: bool = False,
    ) -> str | None:
        url = f"{self.graph_url}/{self.ig_account_id}/media"
        params = {"access_token": self.ig_access_token}

        if is_carousel_item:
            params["is_carousel_item"] = "true"

        if media_type == "video":
            params["media_type"] = "VIDEO"
            params["video_url"] = media_url
        else:
            params["image_url"] = media_url

        if caption:
            params["caption"] = caption

        async with session.post(url, data=params) as resp:
            data = await resp.json()
            if resp.status >= 400:
                logger.error(f"Instagram Container Error: {data}")
                return None
            return data.get("id")

    async def _wait_for_container(
        self,
        session: aiohttp.ClientSession,
        creation_id: str,
        timeout: int = 180,
    ) -> bool:
        url = f"{self.graph_url}/{creation_id}"
        deadline = asyncio.get_running_loop().time() + timeout

        while asyncio.get_running_loop().time() < deadline:
            params = {
                "fields": "status_code,status",
                "access_token": self.ig_access_token,
            }
            try:
                async with session.get(url, params=params) as resp:
                    data = await resp.json()
                    status_code = data.get("status_code")
                    if status_code == "FINISHED":
                        return True
                    if status_code in {"ERROR", "EXPIRED"}:
                        logger.error(f"Instagram Container Error: {data}")
                        return False
            except Exception as e:
                logger.warning(f"Очікування контейнера {creation_id}: {e}")

            await asyncio.sleep(4)
        return False

    async def _publish_container(
        self, session: aiohttp.ClientSession, creation_id: str
    ) -> str | None:
        url = f"{self.graph_url}/{self.ig_account_id}/media_publish"
        params = {
            "creation_id": creation_id,
            "access_token": self.ig_access_token,
        }
        async with session.post(url, data=params) as resp:
            data = await resp.json()
            if resp.status >= 400:
                logger.error(f"Instagram Publish Error: {data}")
                return None
            return data.get("id")

    async def publish_instagram_carousel(self, caption: str, media_items: list):
        if not self.ig_account_id or not self.ig_access_token or not self.media_base_url:
            logger.warning("Instagram параметри не заповнені або відсутній MEDIA_BASE_URL.")
            return

        if not media_items:
            logger.warning("Немає медіа для створення каруселі в Instagram.")
            return

        clean_caption = self._strip_html(caption)
        items = media_items[:10]
        is_carousel = len(items) > 1

        async with aiohttp.ClientSession() as session:
            try:
                child_ids = []
                for item in items:
                    public_url = self.create_public_media_url(item["path"])
                    slide_caption = None if is_carousel else clean_caption

                    child_id = await self._create_container(
                        session=session,
                        media_url=public_url,
                        media_type=item["type"],
                        caption=slide_caption,
                        is_carousel_item=is_carousel,
                    )
                    if not child_id:
                        continue

                    if await self._wait_for_container(session, child_id):
                        child_ids.append(child_id)

                if not child_ids:
                    logger.error("Жоден слайд не пройшов валідацію в Meta.")
                    return

                if not is_carousel:
                    media_id = await self._publish_container(session, child_ids[0])
                else:
                    parent_url = f"{self.graph_url}/{self.ig_account_id}/media"
                    parent_params = {
                        "media_type": "CAROUSEL",
                        "children": ",".join(child_ids),
                        "caption": clean_caption,
                        "access_token": self.ig_access_token,
                    }
                    async with session.post(parent_url, data=parent_params) as resp:
                        parent_data = await resp.json()
                        carousel_id = parent_data.get("id")

                    if not carousel_id:
                        logger.error(f"Не вдалося створити CAROUSEL контейнер: {parent_data}")
                        return

                    if not await self._wait_for_container(session, carousel_id):
                        logger.error(f"CAROUSEL контейнер {carousel_id} не готовий до публікації.")
                        return

                    media_id = await self._publish_container(session, carousel_id)

                if media_id:
                    logger.info(f"✅ Instagram публікацію успішно виконано! ID: {media_id}")

            except Exception as e:
                logger.error(f"Помилка при постінгу в Instagram: {e}", exc_info=True)

    async def close(self):
        await self.bot.session.close()
