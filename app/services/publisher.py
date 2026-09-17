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
        validate_media: bool = True,
    ) -> bool:
        try:
            # За замовчуванням метод сам захищений media-gate.
            # main.py може передати validate_media=False, якщо вже отримав
            # verdict через validate_media_for_news() і використовує його
            # одночасно для Telegram та Instagram.
            if media_path and validate_media:
                verdict = await self.validate_media_for_news(
                    text=text,
                    media_path=media_path,
                    media_type=media_type,
                )

                if not verdict.get("is_relevant", False):
                    media_path = None
                    media_type = None

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

    async def validate_media_for_news(
        self,
        text: str,
        media_path: str | None,
        media_type: str | None,
    ) -> dict:
        """
        Єдина перевірка медіа перед публікацією на будь-якій платформі.

        Фото проходять Gemini Vision-перевірку. Відео поки не аналізуємо
        покадрово, тому зберігаємо попередню поведінку й дозволяємо їх,
        якщо файл існує. Результат цього методу треба використовувати
        одночасно для Telegram та Instagram.
        """
        if not media_path or not Path(media_path).exists():
            return {
                "is_relevant": False,
                "confidence": 0,
                "has_prominent_text": False,
                "conflicting_text": False,
                "reason": "media_file_missing",
                "media_type": media_type,
            }

        if media_type == "photo":
            verdict = await self._validate_photo_relevance(
                text=text,
                media_path=media_path,
            )
            verdict["media_type"] = "photo"

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
            else:
                logger.info(
                    "MEDIA OK: path=%s confidence=%s prominent_text=%s",
                    media_path,
                    verdict.get("confidence"),
                    verdict.get("has_prominent_text"),
                )

            return verdict

        if media_type == "video":
            # Поточний Vision-gate створений для фото. Не змінюємо
            # поведінку відео цим невеликим патчем.
            return {
                "is_relevant": True,
                "confidence": 100,
                "has_prominent_text": False,
                "conflicting_text": False,
                "reason": "video_validation_not_enabled",
                "media_type": "video",
            }

        return {
            "is_relevant": False,
            "confidence": 0,
            "has_prominent_text": False,
            "conflicting_text": False,
            "reason": f"unsupported_media_type: {media_type}",
            "media_type": media_type,
        }

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

confidence — ЦІЛЕ ЧИСЛО ВІД 0 ДО 100, де 100 = повна впевненість.
Не використовуй шкалу 0-1.

Відповідь ТІЛЬКИ JSON:
{{
  "is_relevant": true,
  "confidence": 95,
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

                # Gemini іноді повертає confidence у шкалі 0-1, навіть
                # коли ми просимо 0-100. Раніше int(0.95) перетворювався
                # на 0, через що правильні фото помилково відкидалися.
                try:
                    raw_confidence = float(
                        data.get("confidence", 0) or 0
                    )
                except (TypeError, ValueError):
                    raw_confidence = 0.0

                if 0.0 <= raw_confidence <= 1.0:
                    raw_confidence *= 100.0

                confidence = int(round(raw_confidence))
                confidence = max(0, min(100, confidence))

                # Дуже невпевнене "так" все ще не приймаємо, але тепер
                # confidence спочатку нормалізований до єдиної шкали 0-100.
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
                # Важливо: якщо медіа лише одне, НЕ створюємо для нього
                # carousel-child. Раніше один відеофайл спочатку йшов як
                # is_carousel_item=true + media_type=VIDEO, хоча фактично
                # потім мав публікуватися як одиночний пост. У Graph API
                # одиночне feed-відео тепер треба створювати як REELS.
                if len(filtered_items) == 1:
                    await self._publish_single_item(
                        session,
                        filtered_items[0],
                        clean_caption,
                    )
                    return

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

                # Якщо з початкової каруселі після помилок лишився один
                # валідний елемент, публікуємо його окремо. Контейнер
                # створюємо заново вже у правильному standalone-режимі.
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
            # Для одиночного відео Graph API v26+ вимагає REELS.
            # Для відео всередині каруселі залишаємо VIDEO, оскільки
            # саме такий режим уже успішно працює в поточному пайплайні.
            if is_carousel_item:
                params["media_type"] = "VIDEO"
            else:
                params["media_type"] = "REELS"
                params["share_to_feed"] = "true"

            params["video_url"] = media_url
        else:
            params["image_url"] = media_url

        if caption:
            params["caption"] = caption

        # Meta іноді повертає 9004/2207052 одразу після появи нового
        # публічного файла на MEDIA_BASE_URL. Робимо короткий retry лише
        # для помилок завантаження медіа; інші API-помилки не маскуємо.
        max_attempts = 3

        for attempt in range(1, max_attempts + 1):
            async with session.post(
                url,
                data=params,
            ) as response:
                data = await response.json(
                    content_type=None
                )

                if response.status < 400:
                    return data.get("id")

                retryable = self._is_retryable_instagram_media_error(
                    data
                )

                logger.warning(
                    "Instagram container error (%s), attempt %s/%s: %s",
                    media_type,
                    attempt,
                    max_attempts,
                    data,
                )

                if not retryable or attempt >= max_attempts:
                    return None

            await asyncio.sleep(
                4 * attempt
            )

        return None

    @staticmethod
    def _is_retryable_instagram_media_error(
        data: dict,
    ) -> bool:
        """
        Retry лише для ситуацій, коли Meta тимчасово не може
        завантажити файл за нашим public URL.
        """
        if not isinstance(data, dict):
            return False

        error = data.get("error")
        if not isinstance(error, dict):
            return False

        try:
            code = int(error.get("code") or 0)
        except (TypeError, ValueError):
            code = 0

        try:
            subcode = int(error.get("error_subcode") or 0)
        except (TypeError, ValueError):
            subcode = 0

        text = " ".join(
            str(error.get(key) or "")
            for key in (
                "message",
                "error_user_title",
                "error_user_msg",
            )
        ).lower()

        if code == 9004 or subcode in {2207052, 2207082}:
            return True

        retry_markers = (
            "couldn't download",
            "could not download",
            "unable to fetch",
            "failed to fetch",
            "media upload has failed",
            "не удалось извлечь медиафайл",
            "не удалось скачать медиафайл",
        )
        return any(marker in text for marker in retry_markers)

    async def _wait_for_container(
        self,
        session: aiohttp.ClientSession,
        creation_id: str,
        timeout: int = 180,
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
