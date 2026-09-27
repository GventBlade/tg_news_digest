import asyncio
import io
import hashlib
import json
import logging
import mimetypes
import re
import time
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
        # Video validation: small clips go inline; larger clips use Gemini Files
        # API. On validator/API failure we fail open for AUTO video because the
        # Summarizer already applies a stricter event-consistency gate. Clear
        # high-confidence mismatches are still rejected.
        self.video_validation_inline_max_bytes = 12 * 1024 * 1024
        self.video_validation_processing_timeout = 45
        # Для відео, які ОБОВ'ЯЗКОВО пішли на Gemini-перевірку (насамперед
        # AUTO-відео фізичних атак), безпечніше втратити медіа і лишити текст,
        # ніж опублікувати ролик з іншого влучання/місця.
        self.video_validation_fail_closed = True

    async def publish_telegram_post(
        self,
        text: str,
        media_path: str | None = None,
        media_type: str | None = None,
        validate_media: bool = True,
        media_file_id: str | None = None,
        require_media: bool = False,
        video_validation_needed: bool = False,
    ) -> bool:
        """
        Publish one Telegram item.

        `media_file_id` is primarily for manual admin media that Telegram already
        stores. It lets us re-send a large original video even when Bot API
        refuses to download it locally. If `require_media=True`, failure to send
        that media returns False instead of silently degrading to text-only.
        """
        try:
            # Local AUTO media can still be validated here when caller did not
            # already validate it. Telegram file_id media is used only for the
            # manual locked path and intentionally bypasses Vision.
            if media_path and validate_media:
                verdict = await self.validate_media_for_news(
                    text=text,
                    media_path=media_path,
                    media_type=media_type,
                    video_validation_needed=video_validation_needed,
                )

                if not verdict.get("is_relevant", False):
                    media_path = None
                    media_type = None

            has_local_media = bool(
                media_path
                and Path(media_path).exists()
                and media_type in {"photo", "video"}
            )
            has_file_id_media = bool(
                media_file_id
                and media_type in {"photo", "video"}
            )

            if has_file_id_media or has_local_media:
                media_candidates = []
                if has_file_id_media:
                    media_candidates.append(
                        ("file_id", str(media_file_id))
                    )
                if has_local_media:
                    media_candidates.append(
                        ("local", FSInputFile(str(media_path)))
                    )

                last_media_error = None
                for source_kind, payload in media_candidates:
                    try:
                        if media_type == "photo":
                            await self.bot.send_photo(
                                chat_id=settings.TARGET_CHANNEL_ID,
                                photo=payload,
                                caption=text,
                            )
                            logger.info(
                                "Фото-пост опубліковано в Telegram%s.",
                                " через file_id"
                                if source_kind == "file_id"
                                else "",
                            )
                            return True

                        if media_type == "video":
                            await self.bot.send_video(
                                chat_id=settings.TARGET_CHANNEL_ID,
                                video=payload,
                                caption=text,
                                supports_streaming=True,
                            )
                            logger.info(
                                "Відео-пост опубліковано в Telegram%s.",
                                " через file_id"
                                if source_kind == "file_id"
                                else "",
                            )
                            return True

                    except Exception as media_error:
                        last_media_error = media_error
                        logger.warning(
                            "Telegram media send failed via %s: %s",
                            source_kind,
                            media_error,
                        )

                if require_media:
                    logger.error(
                        "REQUIRED MEDIA send failed via all available paths; "
                        "text-only fallback заборонено: %s",
                        last_media_error,
                        exc_info=last_media_error is not None,
                    )
                    return False

                logger.warning(
                    "Не вдалося відправити медіа, відправляємо текстом."
                )

            elif require_media:
                logger.error(
                    "REQUIRED MEDIA missing: type=%s path=%s file_id=%s",
                    media_type,
                    media_path,
                    bool(media_file_id),
                )
                return False

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
        video_validation_needed: bool = False,
    ) -> dict:
        """
        Єдина перевірка медіа перед публікацією на будь-якій платформі.

        Фото проходять Gemini Vision-перевірку. AUTO-відео фізичних атак
        теж обов'язково перевіряємо Gemini, навіть якщо caller забув передати
        video_validation_needed=True. Для решти відео зберігаємо дешевий fast-path,
        якщо Summarizer не позначив ролик як підозрілий. Результат цього методу
        треба використовувати одночасно для Telegram та Instagram.
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
            # Подвійна страховка. Summarizer передає video_validation_needed=True
            # для фізичних атак, але Publisher сам повторно впізнає такі новини.
            # Тому навіть якщо десь у caller загубиться цей прапорець, AUTO-відео
            # удару/влучання не пройде VIDEO MEDIA FAST-PASS без Vision-check.
            force_video_validation = bool(
                video_validation_needed
                or self._looks_like_physical_attack_news(text)
            )

            if not force_video_validation:
                verdict = {
                    "is_relevant": True,
                    "confidence": 0,
                    "has_prominent_text": False,
                    "conflicting_text": False,
                    "conflicting_context": False,
                    "reason": "auto_video_fast_path_no_gemini",
                    "media_type": "video",
                }
                logger.info(
                    "VIDEO MEDIA FAST-PASS: path=%s reason=%s",
                    media_path,
                    verdict["reason"],
                )
                return verdict

            logger.info(
                "VIDEO MEDIA REQUIRED CHECK: path=%s attack=%s requested=%s",
                media_path,
                self._looks_like_physical_attack_news(text),
                bool(video_validation_needed),
            )
            verdict = await self._validate_video_relevance(
                text=text,
                media_path=media_path,
            )
            verdict["media_type"] = "video"

            if not verdict.get("is_relevant", False):
                logger.warning(
                    "VIDEO MEDIA REJECTED: path=%s confidence=%s "
                    "conflicting_context=%s reason=%s",
                    media_path,
                    verdict.get("confidence"),
                    verdict.get("conflicting_context"),
                    verdict.get("reason"),
                )
            else:
                logger.info(
                    "VIDEO MEDIA OK: path=%s confidence=%s reason=%s",
                    media_path,
                    verdict.get("confidence"),
                    verdict.get("reason"),
                )
            return verdict

        return {
            "is_relevant": False,
            "confidence": 0,
            "has_prominent_text": False,
            "conflicting_text": False,
            "reason": f"unsupported_media_type: {media_type}",
            "media_type": media_type,
        }

    @staticmethod
    def _looks_like_physical_attack_news(text: str) -> bool:
        """
        Publisher-side fail-safe for AUTO video.

        Не покладаємось лише на прапорець із Summarizer: якщо фінальний текст
        явно описує фізичний удар/влучання з ракетою, БпЛА тощо, відео має
        пройти Gemini-перевірку. Метафоричні/кібер "атаки" не підходять.
        """
        clean = NewsPublisher._strip_html(str(text or "")).lower()
        clean = re.sub(r"\s+", " ", clean)
        if not clean:
            return False

        cyber_markers = (
            "кібератак", "хакер", "cyberattack", "кібершпиг",
        )
        weapon_markers = (
            "бпла", "дрон", "шахед", "ракет", "обстріл", "обстрілу",
            "артилер", "авіабомб", "каб ", "кабами",
        )
        impact_markers = (
            "влуч", "приліт", "прильот", "удар", "вибух", "пожеж",
            "зруйн", "руйнув", "пошкод",
        )

        has_weapon = any(marker in clean for marker in weapon_markers)
        has_impact = any(marker in clean for marker in impact_markers)

        if has_weapon and has_impact:
            return True

        # "обстріл" сам по собі вже достатньо фізичний сигнал.
        if any(marker in clean for marker in ("обстріл", "обстрілу")):
            return True

        # Одне слово "атака" не використовуємо: воно може бути політичним чи
        # кібернетичним. Якщо є лише cyber-сигнали — точно не фізична атака.
        if any(marker in clean for marker in cyber_markers) and not has_weapon:
            return False

        return False

    async def _validate_video_relevance(
        self,
        text: str,
        media_path: str,
    ) -> dict:
        try:
            return await asyncio.to_thread(
                self._validate_video_relevance_sync,
                text,
                media_path,
            )
        except Exception as e:
            logger.error(
                "Помилка video media validation для %s: %s",
                media_path,
                e,
                exc_info=True,
            )
            return {
                "is_relevant": not self.video_validation_fail_closed,
                "confidence": 0,
                "conflicting_context": False,
                "reason": f"video_validation_error: {e}",
            }

    def _validate_video_relevance_sync(
        self,
        text: str,
        media_path: str,
    ) -> dict:
        path = Path(media_path)
        clean_text = self._strip_html(text)
        mime_type = mimetypes.guess_type(path.name)[0] or "video/mp4"
        size = path.stat().st_size

        prompt = f"""
Ти — обережний відеоредактор українського новинного Telegram-каналу.

Порівняй ФАКТИЧНЕ ВІДЕО з текстом новини нижче. Потрібно визначити, чи
цей ролик справді стосується тієї самої конкретної події.

Особливо для ударів/пожеж/аварій:
- інше місто, район, об'єкт або очевидно інший інцидент = reject;
- якщо текст називає КОНКРЕТНУ ціль/місце (житловий будинок, завод, міст,
  порт, водойма тощо), а відео явно показує влучання в несумісне середовище
  або іншу ціль — is_relevant=false і conflicting_context=true;
- приклад: текст про влучання у житловий будинок, а ролик явно показує
  падіння/вибух у воді чи відкритій місцевості без будинку — це reject;
- загальні кадри вибуху/диму, де конкретну ціль неможливо розпізнати і немає
  видимого протиріччя, можна дозволити, але з нижчою confidence;
- водяні знаки каналу самі по собі НЕ є конфліктом;
- якщо в кадрі/аудіо/плашках видно назву іншого міста, об'єкта, компанії чи
  іншої новини — conflicting_context=true;
- не вимагай, щоб відео показувало всі факти тексту: достатньо, щоб воно
  чесно ілюструвало саме цю подію;
- якщо не можеш встановити відповідність або сумніваєшся — НЕ відхиляй;\n  conflicting_context=true став лише при явному, видимому протиріччі.

НОВИНА:
{clean_text[:1800]}

confidence — ціле число 0-100.
Відповідь ТІЛЬКИ JSON:
{{
  "is_relevant": true,
  "confidence": 90,
  "conflicting_context": false,
  "video_summary": "коротко, що реально видно/чути",
  "reason": "коротке пояснення"
}}
"""

        uploaded = None
        last_error = None
        try:
            for model in self.media_validation_models:
                try:
                    if size <= self.video_validation_inline_max_bytes:
                        video_bytes = path.read_bytes()
                        video_part = types.Part.from_bytes(
                            data=video_bytes,
                            mime_type=mime_type,
                        )
                        contents = [video_part, prompt]
                    else:
                        if uploaded is None:
                            uploaded = self.media_validation_client.files.upload(
                                file=str(path)
                            )
                            deadline = (
                                time.monotonic()
                                + self.video_validation_processing_timeout
                            )
                            while (
                                getattr(getattr(uploaded, "state", None), "name", "")
                                not in {"ACTIVE", "FAILED"}
                                and time.monotonic() < deadline
                            ):
                                time.sleep(2)
                                uploaded = self.media_validation_client.files.get(
                                    name=uploaded.name
                                )

                            state_name = getattr(
                                getattr(uploaded, "state", None),
                                "name",
                                "",
                            )
                            if state_name != "ACTIVE":
                                raise TimeoutError(
                                    f"video file processing state={state_name or 'unknown'}"
                                )
                        contents = [uploaded, prompt]

                    response = self.media_validation_client.models.generate_content(
                        model=model,
                        contents=contents,
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
                        raise ValueError("video validator returned non-object JSON")

                    try:
                        raw_confidence = float(data.get("confidence", 0) or 0)
                    except (TypeError, ValueError):
                        raw_confidence = 0.0
                    if 0.0 <= raw_confidence <= 1.0:
                        raw_confidence *= 100.0
                    confidence = max(0, min(100, int(round(raw_confidence))))

                    model_relevant = bool(data.get("is_relevant", False))
                    conflicting = bool(data.get("conflicting_context", False))

                    # Для ролика, який уже пішов на обов'язкову перевірку,
                    # краще відкинути явний mismatch, ніж показати інше влучання.
                    # Водночас не караємо нічні/задимлені кадри лише за невисоку
                    # впевненість, якщо модель не бачить конкретного протиріччя.
                    clear_mismatch = bool(
                        (conflicting and confidence >= 80)
                        or ((not model_relevant) and confidence >= 90)
                    )
                    is_relevant = not clear_mismatch

                    reason = str(data.get("reason") or "")[:500]
                    if clear_mismatch:
                        reason = (
                            reason
                            + " | rejected: clear video/event mismatch"
                        ).strip()
                    else:
                        reason = (
                            reason
                            + " | allowed: no clear video/event mismatch"
                        ).strip()

                    return {
                        "is_relevant": is_relevant,
                        "confidence": confidence,
                        "has_prominent_text": False,
                        "conflicting_text": False,
                        "conflicting_context": conflicting,
                        "video_summary": str(data.get("video_summary") or "")[:300],
                        "reason": reason,
                    }

                except Exception as e:
                    last_error = e
                    logger.warning(
                        "Video validation failed via %s for %s: %s",
                        model,
                        media_path,
                        e,
                    )

            return {
                "is_relevant": not self.video_validation_fail_closed,
                "confidence": 0,
                "has_prominent_text": False,
                "conflicting_text": False,
                "conflicting_context": False,
                "reason": f"all_video_validation_models_failed: {last_error}",
            }
        finally:
            if uploaded is not None and getattr(uploaded, "name", None):
                try:
                    self.media_validation_client.files.delete(name=uploaded.name)
                except Exception as cleanup_error:
                    logger.warning(
                        "Не вдалося видалити тимчасовий Gemini video file %s: %s",
                        getattr(uploaded, "name", ""),
                        cleanup_error,
                    )

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

    @staticmethod
    def build_media_fingerprint(
        media_path: str,
        media_type: str,
    ) -> str | None:
        """
        Stable lightweight fingerprint for 24h AUTO-media reuse lock.

        Photos use a 64-bit dHash, which normally survives Telegram resize/
        recompression. Videos use sampled SHA1 + size; this intentionally catches
        exact/same-file reuse without trying to "understand" video content.
        """
        path = Path(str(media_path or ""))
        kind = str(media_type or "").strip().lower()
        if not path.exists() or kind not in {"photo", "video"}:
            return None

        try:
            if kind == "photo":
                with Image.open(path) as image:
                    image = ImageOps.exif_transpose(image).convert("L")
                    image = image.resize((9, 8), Image.Resampling.LANCZOS)
                    pixels = list(image.getdata())

                bits = 0
                bit_index = 0
                for row in range(8):
                    offset = row * 9
                    for col in range(8):
                        left = pixels[offset + col]
                        right = pixels[offset + col + 1]
                        if left > right:
                            bits |= 1 << bit_index
                        bit_index += 1

                # Додаємо average-hash, щоб не було випадкових колізій
                # на дуже простих/монотонних зображеннях.
                with Image.open(path) as image:
                    avg_img = ImageOps.exif_transpose(image).convert("L")
                    avg_img = avg_img.resize(
                        (8, 8),
                        Image.Resampling.LANCZOS,
                    )
                    avg_pixels = list(avg_img.getdata())
                avg_value = sum(avg_pixels) / max(1, len(avg_pixels))
                avg_bits = 0
                for idx, value in enumerate(avg_pixels):
                    if value >= avg_value:
                        avg_bits |= 1 << idx

                return f"phash128:{bits:016x}{avg_bits:016x}"

            size = path.stat().st_size
            hasher = hashlib.sha1()
            chunk = 1024 * 1024
            with path.open("rb") as fh:
                first = fh.read(chunk)
                hasher.update(first)

                if size > chunk * 2:
                    middle_pos = max(0, size // 2 - chunk // 2)
                    fh.seek(middle_pos)
                    hasher.update(fh.read(chunk))

                if size > chunk:
                    fh.seek(max(0, size - chunk))
                    hasher.update(fh.read(chunk))

            hasher.update(str(size).encode("ascii"))
            return f"vsha1:{hasher.hexdigest()}:{size}"

        except Exception as exc:
            logger.warning(
                "Не вдалося побудувати media fingerprint для %s: %s",
                media_path,
                exc,
            )
            return None

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
