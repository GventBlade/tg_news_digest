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

                    if media_type == "video":
                        await self.bot.send_video(
                            chat_id=settings.TARGET_CHANNEL_ID,
                            video=FSInputFile(media_path),
                            caption=text,
                            supports_streaming=True,
                        )
                        logger.info("Відео-пост опубліковано в Telegram.")
                        return True

                except Exception as media_error:
                    logger.warning(
                        f"Не вдалося відправити медіа ({media_error}), "
                        f"відправляємо текстом."
                    )

            await self.bot.send_message(
                chat_id=settings.TARGET_CHANNEL_ID,
                text=text,
            )

            logger.info("Текстовий пост опубліковано в Telegram.")
            return True

        except Exception as e:
            logger.error(
                f"Помилка публікації в Telegram: {e}",
                exc_info=True
            )
            return False

    def create_public_media_url(self, file_path: str) -> str:
        if not self.media_base_url:
            raise RuntimeError("MEDIA_BASE_URL не налаштований у .env")

        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"Файл не знайдено: {file_path}")

        filename = quote(path.name)
        return f"{self.media_base_url}/media/{filename}"

    async def publish_instagram_carousel(self, caption: str, media_items: list):
        if not self.ig_account_id or not self.ig_access_token or not self.media_base_url:
            logger.warning(
                "Instagram параметри не заповнені або "
                "відсутній MEDIA_BASE_URL."
            )
            return

        filtered_items = self._filter_instagram_media(media_items)

        if not filtered_items:
            logger.warning("Немає валідних медіа для Instagram.")
            return

        clean_caption = self._strip_html(caption)

        async with aiohttp.ClientSession() as session:
            try:
                valid_child_ids = []

                for item in filtered_items:
                    child_id = await self._prepare_carousel_item(
                        session,
                        item
                    )

                    if child_id:
                        valid_child_ids.append(child_id)

                if not valid_child_ids:
                    logger.error(
                        "Жоден слайд не пройшов "
                        "обробку Instagram."
                    )
                    return

                if len(valid_child_ids) == 1:
                    await self._publish_single_item(
                        session,
                        filtered_items[0],
                        clean_caption
                    )
                    return

                carousel_id = await self._create_carousel(
                    session,
                    valid_child_ids,
                    clean_caption
                )

                if not carousel_id:
                    return

                if not await self._wait_for_container(session, carousel_id):
                    return

                media_id = await self._publish_container(
                    session,
                    carousel_id
                )

                if media_id:
                    logger.info(
                        f"Instagram carousel опубліковано. "
                        f"ID: {media_id}"
                    )

            except Exception as e:
                logger.error(
                    f"Помилка Instagram-публікації: {e}",
                    exc_info=True
                )

    async def _prepare_carousel_item(
        self,
        session: aiohttp.ClientSession,
        item: dict
    ) -> str | None:
        try:
            public_url = self.create_public_media_url(item["path"])

            child_id = await self._create_container(
                session=session,
                media_url=public_url,
                media_type=item["type"],
                caption=None,
                is_carousel_item=True,
            )

            if not child_id:
                return None

            if not await self._wait_for_container(session, child_id):
                return None

            return child_id

        except Exception as e:
            logger.warning(
                f"Помилка підготовки Instagram media "
                f"{item.get('path')}: {e}"
            )
            return None

    async def _publish_single_item(
        self,
        session: aiohttp.ClientSession,
        item: dict,
        caption: str
    ):
        public_url = self.create_public_media_url(item["path"])

        creation_id = await self._create_container(
            session=session,
            media_url=public_url,
            media_type=item["type"],
            caption=caption,
            is_carousel_item=False,
        )

        if not creation_id:
            return

        if not await self._wait_for_container(session, creation_id):
            return

        media_id = await self._publish_container(
            session,
            creation_id
        )

        if media_id:
            logger.info(
                f"Instagram post опубліковано. "
                f"ID: {media_id}"
            )

    async def _create_carousel(
        self,
        session: aiohttp.ClientSession,
        child_ids: list,
        caption: str
    ) -> str | None:
        url = f"{self.graph_url}/{self.ig_account_id}/media"

        params = {
            "media_type": "CAROUSEL",
            "children": ",".join(child_ids),
            "caption": caption,
            "access_token": self.ig_access_token,
        }

        async with session.post(url, data=params) as response:
            data = await response.json()

            if response.status >= 400:
                logger.error(
                    f"Помилка створення Instagram carousel: {data}"
                )
                return None

            return data.get("id")

    async def _create_container(
        self,
        session: aiohttp.ClientSession,
        media_url: str,
        media_type: str,
        caption: str | None = None,
        is_carousel_item: bool = False,
    ) -> str | None:
        url = f"{self.graph_url}/{self.ig_account_id}/media"

        params = {
            "access_token": self.ig_access_token
        }

        if is_carousel_item:
            params["is_carousel_item"] = "true"

        if media_type == "video":
            params["media_type"] = "VIDEO"
            params["video_url"] = media_url
        else:
            params["image_url"] = media_url

        if caption:
            params["caption"] = caption

        async with session.post(url, data=params) as response:
            data = await response.json()

            if response.status >= 400:
                logger.warning(
                    f"Instagram container error "
                    f"({media_type}): {data}"
                )
                return None

            return data.get("id")

    async def _wait_for_container(
        self,
        session: aiohttp.ClientSession,
        creation_id: str,
        timeout: int = 120,
    ) -> bool:
        url = f"{self.graph_url}/{creation_id}"
        deadline = asyncio.get_running_loop().time() + timeout

        while asyncio.get_running_loop().time() < deadline:
            params = {
                "fields": "status_code,status",
                "access_token": self.ig_access_token,
            }

            try:
                async with session.get(url, params=params) as response:
                    data = await response.json()
                    status_code = data.get("status_code")

                    if status_code == "FINISHED":
                        return True

                    if status_code in {"ERROR", "EXPIRED"}:
                        logger.warning(
                            f"Instagram container "
                            f"{creation_id} відхилено: {data}"
                        )
                        return False

            except Exception as e:
                logger.warning(
                    f"Помилка перевірки Instagram "
                    f"container {creation_id}: {e}"
                )

            await asyncio.sleep(4)

        return False

    async def _publish_container(
        self,
        session: aiohttp.ClientSession,
        creation_id: str
    ) -> str | None:
        url = f"{self.graph_url}/{self.ig_account_id}/media_publish"

        params = {
            "creation_id": creation_id,
            "access_token": self.ig_access_token,
        }

        async with session.post(url, data=params) as response:
            data = await response.json()

            if response.status >= 400:
                logger.error(
                    f"Instagram publish error: {data}"
                )
                return None

            return data.get("id")

    @staticmethod
    def _filter_instagram_media(media_items: list) -> list:
        filtered = []

        for item in media_items[:10]:
            path = Path(item.get("path", ""))
            media_type = item.get("type")

            if not path.exists():
                continue

            if media_type not in {"photo", "video"}:
                continue

            file_size_mb = path.stat().st_size / (1024 * 1024)

            if media_type == "video" and file_size_mb > 45:
                logger.warning(
                    f"Instagram video {path.name} пропущено: "
                    f"{file_size_mb:.1f} MB"
                )
                continue

            filtered.append({
                "path": str(path),
                "type": media_type
            })

        return filtered

    @staticmethod
    def _strip_html(text: str) -> str:
        clean = re.sub(r"<.*?>", "", text).strip()

        if len(clean) > 2150:
            clean = clean[:2140] + "...\n(продовження в Telegram)"

        return clean

    async def close(self):
        await self.bot.session.close()
