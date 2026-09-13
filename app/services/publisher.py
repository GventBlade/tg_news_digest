import asyncio
import io
import json
import logging
import mimetypes
import re
from pathlib import Path
from urllib.parse import quote

import aiohttp
from aiogram import Bot
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.types import FSInputFile
from PIL import Image, ImageOps
from google import genai
from google.genai import types

from app.config import settings

logger = logging.getLogger(__name__)


class NewsPublisher:
    def __init__(self):
        self.bot = Bot(
            token=settings.BOT_TOKEN,
            default=DefaultBotProperties(
                parse_mode=ParseMode.HTML
            ),
        )

        self.ig_account_id = settings.INSTAGRAM_ACCOUNT_ID
        self.ig_access_token = settings.INSTAGRAM_ACCESS_TOKEN
        self.graph_url = "https://graph.facebook.com/v26.0"
        self.media_base_url = (
            settings.MEDIA_BASE_URL or ""
        ).rstrip("/")

        # Окрема vision-перевірка фактичного зображення перед публікацією.
        # Це фінальна страховка від випадку, коли Telegram-пост має правильний
        # текст, але прикріплену картинку від іншої новини.
        self.media_validation_client = genai.Client(
            api_key=settings.GEMINI_API_KEY
        )
        self.media_validation_models = [
            "gemini-3.5-flash-lite",
            "gemini-3.1-flash-lite",
        ]
        self.media_validation_fail_closed = True

    async def publish_telegram_post(
        self,
        text: str,
        media_path: str | None = None,
        media_type: str | None = None,
    ) -> bool:
        try:
            # Фінальний media gate. Для фото аналізуємо САМЕ зображення,
            # а не лише текст source-поста. Якщо на картинці є заголовок
            # про іншу подію/місто/компанію — фото відкидаємо і публікуємо
            # новину текстом. Краще без фото, ніж з оманливим фото.
            if (
                media_path
                and media_type == "photo"
                and Path(media_path).exists()
            ):
                verdict = await self._validate_photo_relevance(
                    text=text,
                    media_path=media_path,
                )

                if not verdict.get("is_relevant", False):
                    logger.warning(
                        "MEDIA REJECTED: path=%s confidence=%s prominent_text=%s "
                        "conflicting_text=%s reason=%s",
                        media_path,
                        verdict.get("confidence"),
                        verdict.get("has_prominent_text"),
                        verdict.get("conflicting_text"),
                        verdict.get("reason"),
                    )
                    media_path = None
                    media_type = None
                else:
                    logger.info(
                        "MEDIA OK: path=%s confidence=%s prominent_text=%s",
                        media_path,
                        verdict.get("confidence"),
                        verdict.get("has_prominent_text"),
                    )

            if media_path and Path(media_path).exists():
                try:
                    if media_type == "photo":
                        await self.bot.send_photo(
                            chat_id=settings.TARGET_CHANNEL_ID,
                            photo=FSInputFile(media_path),
                            caption=text,
                        )
                        logger.info(
                            "Фото-пост опубліковано в Telegram."
                        )
                        return True

                    if media_type == "video":
                        await self.bot.send_video(
                            chat_id=settings.TARGET_CHANNEL_ID,
                            video=FSInputFile(media_path),
                            caption=text,
                            supports_streaming=True,
                        )
                        logger.info(
                            "Відео-пост опубліковано в Telegram."
                        )
                        return True

                except Exception as media_error:
                    logger.warning(
                        f"Не вдалося відправити медіа "
                        f"({media_error}), відправляємо текстом."
                    )

            await self.bot.send_message(
                chat_id=settings.TARGET_CHANNEL_ID,
                text=text,
            )

            logger.info(
                "Текстовий пост опубліковано в Telegram."
            )
            return True

        except Exception as e:
            logger.error(
                f"Помилка публікації в Telegram: {e}",
                exc_info=True,
            )
            return False

    async def _validate_photo_relevance(
        self,
        text: str,
        media_path: str,
    ) -> dict:
        """
        Порівнює фінальний текст новини з фактичним фото через Gemini Vision.

        Перевірка навмисно сувора до зображень із великим текстом:
        якщо напис на картинці описує іншу подію, місце, компанію, людину
        або інший сюжет — таке фото не публікується.
        """
        try:
            return await asyncio.to_thread(
                self._validate_photo_relevance_sync,
                text,
                media_path,
            )
        except Exception as e:
            logger.error(
                "Помилка media validation для %s: %s",
                media_path,
                e,
                exc_info=True,
            )
            return {
                "is_relevant": not self.media_validation_fail_closed,
                "confidence": 0,
                "has_prominent_text": False,
                "conflicting_text": False,
                "reason": f"validation_error: {e}",
            }

    def _validate_photo_relevance_sync(
        self,
        text: str,
        media_path: str,
    ) -> dict:
        image_bytes, mime_type = self._prepare_image_for_validation(
            media_path
        )
        clean_text = self._strip_html(text)

        prompt = f"""
Ти — суворий фоторедактор українського новинного Telegram-каналу.

Порівняй ФАКТИЧНЕ ЗОБРАЖЕННЯ з текстом новини нижче.
Треба вирішити, чи можна показувати це фото прямо над цією новиною.

КРИТИЧНЕ ПРАВИЛО:
якщо на зображенні є великий/помітний напис, заголовок, плашка або скриншот
іншої новини, і цей текст стосується ІНШОЇ події, міста, країни, компанії,
людини, об'єкта чи наслідку — is_relevant=false і conflicting_text=true.

Приклад логіки: якщо новина про завод Cersanit на Житомирщині, а на фото
великим текстом написано про Петербург, бензин і атаки на НПЗ — це НЕПРАВИЛЬНЕ
фото, навіть якщо обидві теми побічно пов'язані з війною.

МОЖНА дозволити:
- реальне фото саме цієї події/людини/об'єкта;
- архівне або ілюстративне фото, якщо воно очевидно про той самий об'єкт/тему
  і НЕ вводить читача в оману;
- логотип/будівлю/портрет, якщо вони прямо стосуються героя новини.

ТРЕБА відхилити:
- фото іншої новини;
- картку/скриншот із заголовком про іншу подію;
- інше місто/компанію/особу, якщо це не пояснюється текстом новини;
- стару ілюстрацію, яка створює хибне враження про конкретний новий інцидент;
- зображення, де помітний текст суперечить новині.

Якщо не впевнений, але бачиш сильний текстовий конфлікт — ВІДХИЛЯЙ.
Якщо фото просто нейтральне й релевантне без конфлікту — можна дозволити.

НОВИНА:
{clean_text[:1800]}

Відповідь ТІЛЬКИ JSON:
{{
  "is_relevant": true,
  "confidence": 0,
  "has_prominent_text": false,
  "conflicting_text": false,
  "image_text_summary": "коротко, який текст видно на зображенні, якщо є",
  "reason": "коротке пояснення рішення"
}}
"""

        last_error = None
        for model in self.media_validation_models:
            try:
                response = self.media_validation_client.models.generate_content(
                    model=model,
                    contents=[
                        prompt,
                        types.Part.from_bytes(
                            data=image_bytes,
                            mime_type=mime_type,
                        ),
                    ],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0.05,
                    ),
                )

                raw = self._clean_json_response(
                    (response.text or "").strip()
                )
                data = json.loads(raw)
                if not isinstance(data, dict):
                    raise ValueError("media validator returned non-object JSON")

                is_relevant = bool(data.get("is_relevant", False))
                conflicting_text = bool(data.get("conflicting_text", False))

                # Жорстка локальна страховка: модель не може одночасно
                # сказати "relevant" і "conflicting_text=true".
                if conflicting_text:
                    is_relevant = False

                try:
                    confidence = int(float(data.get("confidence", 0) or 0))
                except (TypeError, ValueError):
                    confidence = 0
                confidence = max(0, min(100, confidence))

                # Дуже невпевнене "так" не приймаємо для оманливих фото.
                if is_relevant and confidence < 55:
                    is_relevant = False
                    data["reason"] = (
                        str(data.get("reason") or "")
                        + " | rejected: low confidence"
                    ).strip()

                return {
                    "is_relevant": is_relevant,
                    "confidence": confidence,
                    "has_prominent_text": bool(
                        data.get("has_prominent_text", False)
                    ),
                    "conflicting_text": conflicting_text,
                    "image_text_summary": str(
                        data.get("image_text_summary") or ""
                    )[:300],
                    "reason": str(data.get("reason") or "")[:500],
                }

            except Exception as e:
                last_error = e
                logger.warning(
                    "Media validation failed via %s for %s: %s",
                    model,
                    media_path,
                    e,
                )

        if self.media_validation_fail_closed:
            return {
                "is_relevant": False,
                "confidence": 0,
                "has_prominent_text": False,
                "conflicting_text": False,
                "reason": f"all_validation_models_failed: {last_error}",
            }

        return {
            "is_relevant": True,
            "confidence": 0,
            "has_prominent_text": False,
            "conflicting_text": False,
            "reason": f"validation_skipped_after_error: {last_error}",
        }

    @staticmethod
    def _prepare_image_for_validation(
        media_path: str,
    ) -> tuple[bytes, str]:
        """
        Нормалізує фото перед Vision: EXIF rotation, RGB, max 1600 px.
        Це зменшує payload, але лишає достатню якість для читання написів.
        """
        path = Path(media_path)
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            image.thumbnail((1600, 1600), Image.Resampling.LANCZOS)

            buffer = io.BytesIO()
            image.save(
                buffer,
                format="JPEG",
                quality=88,
                optimize=True,
            )
            return buffer.getvalue(), "image/jpeg"

    @staticmethod
    def _clean_json_response(text: str) -> str:
        text = str(text or "").strip()
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        first = text.find("{")
        last = text.rfind("}")
        if first >= 0 and last > first:
            text = text[first:last + 1]

        return re.sub(r",\s*([}\]])", r"\1", text).strip()

    def create_public_media_url(
        self,
        file_path: str,
    ) -> str:
        if not self.media_base_url:
            raise RuntimeError(
                "MEDIA_BASE_URL не налаштований у .env"
            )

        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Файл не знайдено: {file_path}"
            )

        filename = quote(path.name)
        return (
            f"{self.media_base_url}/media/{filename}"
        )

    async def publish_instagram_carousel(
        self,
        caption: str,
        media_items: list,
    ):
        if (
            not self.ig_account_id
            or not self.ig_access_token
            or not self.media_base_url
        ):
            logger.warning(
                "Instagram параметри не заповнені або "
                "відсутній MEDIA_BASE_URL."
            )
            return

        filtered_items = self._filter_instagram_media(
            media_items
        )

        if not filtered_items:
            logger.warning(
                "Немає валідних медіа для Instagram."
            )
            return

        clean_caption = self._strip_html(caption)

        timeout = aiohttp.ClientTimeout(total=180)

        async with aiohttp.ClientSession(
            timeout=timeout
        ) as session:
            try:
                valid_children = []

                for item in filtered_items:
                    child_id = await self._prepare_carousel_item(
                        session,
                        item,
                    )

                    if child_id:
                        valid_children.append({
                            "id": child_id,
                            "item": item,
                        })

                if not valid_children:
                    logger.error(
                        "Жоден слайд не пройшов "
                        "обробку Instagram."
                    )
                    return

                # Якщо після фільтрації лишився один валідний елемент,
                # публікуємо його як звичайний пост, а не карусель.
                if len(valid_children) == 1:
                    await self._publish_single_item(
                        session,
                        valid_children[0]["item"],
                        clean_caption,
                    )
                    return

                child_ids = [
                    child["id"]
                    for child in valid_children
                ]

                carousel_id = await self._create_carousel(
                    session,
                    child_ids,
                    clean_caption,
                )

                if not carousel_id:
                    return

                if not await self._wait_for_container(
                    session,
                    carousel_id,
                ):
                    return

                media_id = await self._publish_container(
                    session,
                    carousel_id,
                )

                if media_id:
                    logger.info(
                        "Instagram carousel опубліковано. "
                        f"ID: {media_id}"
                    )

            except Exception as e:
                logger.error(
                    f"Помилка Instagram-публікації: {e}",
                    exc_info=True,
                )

    async def _prepare_carousel_item(
        self,
        session: aiohttp.ClientSession,
        item: dict,
    ) -> str | None:
        try:
            public_url = self.create_public_media_url(
                item["path"]
            )

            child_id = await self._create_container(
                session=session,
                media_url=public_url,
                media_type=item["type"],
                caption=None,
                is_carousel_item=True,
            )

            if not child_id:
                return None

            if not await self._wait_for_container(
                session,
                child_id,
            ):
                return None

            return child_id

        except Exception as e:
            logger.warning(
                "Помилка підготовки Instagram media "
                f"{item.get('path')}: {e}"
            )
            return None

    async def _publish_single_item(
        self,
        session: aiohttp.ClientSession,
        item: dict,
        caption: str,
    ):
        public_url = self.create_public_media_url(
            item["path"]
        )

        creation_id = await self._create_container(
            session=session,
            media_url=public_url,
            media_type=item["type"],
            caption=caption,
            is_carousel_item=False,
        )

        if not creation_id:
            return

        if not await self._wait_for_container(
            session,
            creation_id,
        ):
            return

        media_id = await self._publish_container(
            session,
            creation_id,
        )

        if media_id:
            logger.info(
                "Instagram post опубліковано. "
                f"ID: {media_id}"
            )

    async def _create_carousel(
        self,
        session: aiohttp.ClientSession,
        child_ids: list,
        caption: str,
    ) -> str | None:
        url = (
            f"{self.graph_url}/"
            f"{self.ig_account_id}/media"
        )

        params = {
            "media_type": "CAROUSEL",
            "children": ",".join(child_ids),
            "caption": caption,
            "access_token": self.ig_access_token,
        }

        async with session.post(
            url,
            data=params,
        ) as response:
            data = await response.json(
                content_type=None
            )

            if response.status >= 400:
                logger.error(
                    "Помилка створення Instagram "
                    f"carousel: {data}"
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
        url = (
            f"{self.graph_url}/"
            f"{self.ig_account_id}/media"
        )

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

        async with session.post(
            url,
            data=params,
        ) as response:
            data = await response.json(
                content_type=None
            )

            if response.status >= 400:
                logger.warning(
                    "Instagram container error "
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
        deadline = (
            asyncio.get_running_loop().time()
            + timeout
        )

        while (
            asyncio.get_running_loop().time()
            < deadline
        ):
            params = {
                "fields": "status_code,status",
                "access_token": self.ig_access_token,
            }

            try:
                async with session.get(
                    url,
                    params=params,
                ) as response:
                    data = await response.json(
                        content_type=None
                    )

                    status_code = data.get(
                        "status_code"
                    )

                    if status_code == "FINISHED":
                        return True

                    if status_code in {
                        "ERROR",
                        "EXPIRED",
                    }:
                        logger.warning(
                            "Instagram container "
                            f"{creation_id} відхилено: {data}"
                        )
                        return False

            except Exception as e:
                logger.warning(
                    "Помилка перевірки Instagram "
                    f"container {creation_id}: {e}"
                )

            await asyncio.sleep(4)

        logger.warning(
            "Instagram container "
            f"{creation_id} не завершив обробку "
            f"за {timeout} с."
        )
        return False

    async def _publish_container(
        self,
        session: aiohttp.ClientSession,
        creation_id: str,
    ) -> str | None:
        url = (
            f"{self.graph_url}/"
            f"{self.ig_account_id}/media_publish"
        )

        params = {
            "creation_id": creation_id,
            "access_token": self.ig_access_token,
        }

        async with session.post(
            url,
            data=params,
        ) as response:
            data = await response.json(
                content_type=None
            )

            if response.status >= 400:
                logger.error(
                    f"Instagram publish error: {data}"
                )
                return None

            return data.get("id")

    def _filter_instagram_media(
        self,
        media_items: list,
    ) -> list:
        """
        Instagram іноді відхиляє фото через aspect ratio або EXIF.
        Тому кожне фото перед завантаженням:
        - повертаємо відповідно до EXIF;
        - переводимо в RGB;
        - вписуємо без обрізання в 1080x1350 (4:5);
        - перевидаємо як звичайний JPEG.

        Це усуває помилку OAuthException 36003 для нестандартних фото.
        """
        filtered = []

        for item in media_items[:10]:
            path = Path(
                item.get("path", "")
            )
            media_type = item.get("type")

            if not path.exists():
                continue

            if media_type not in {
                "photo",
                "video",
            }:
                continue

            if media_type == "photo":
                try:
                    normalized_path = (
                        self._prepare_instagram_photo(
                            path
                        )
                    )
                    path = Path(normalized_path)

                except Exception as e:
                    logger.warning(
                        "Instagram photo "
                        f"{path.name} пропущено: "
                        f"не вдалося нормалізувати ({e})"
                    )
                    continue

            file_size_mb = (
                path.stat().st_size
                / (1024 * 1024)
            )

            if (
                media_type == "video"
                and file_size_mb > 45
            ):
                logger.warning(
                    f"Instagram video {path.name} "
                    f"пропущено: {file_size_mb:.1f} MB"
                )
                continue

            filtered.append({
                "path": str(path),
                "type": media_type,
            })

        return filtered

    @staticmethod
    def _prepare_instagram_photo(
        path: Path,
    ) -> str:
        target_size = (1080, 1350)

        output_path = path.with_name(
            f"ig_{path.stem}.jpg"
        )

        with Image.open(path) as image:
            image = ImageOps.exif_transpose(
                image
            ).convert("RGB")

            # Колір фону беремо з усередненого пікселя самого фото,
            # щоб поля не виглядали різко білими або чорними.
            sample = image.resize((1, 1))
            background_color = sample.getpixel(
                (0, 0)
            )

            image.thumbnail(
                target_size,
                Image.Resampling.LANCZOS,
            )

            canvas = Image.new(
                "RGB",
                target_size,
                background_color,
            )

            x = (
                target_size[0] - image.width
            ) // 2
            y = (
                target_size[1] - image.height
            ) // 2

            canvas.paste(
                image,
                (x, y),
            )

            canvas.save(
                output_path,
                format="JPEG",
                quality=92,
                optimize=True,
            )

        return str(output_path)

    @staticmethod
    def _strip_html(
        text: str,
    ) -> str:
        clean = re.sub(
            r"<.*?>",
            "",
            text,
        ).strip()

        if len(clean) > 2150:
            clean = (
                clean[:2140]
                + "...\n(продовження в Telegram)"
            )

        return clean

    async def close(self):
        await self.bot.session.close()
