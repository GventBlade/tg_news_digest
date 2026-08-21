import json
import logging
import time
from typing import Any, Dict, List
from google import genai
from google.genai import types
from app.config import settings

logger = logging.getLogger(__name__)


class NewsSummarizer:
    def __init__(self):
        self.client = genai.Client(api_key=settings.GEMINI_API_KEY)

    def select_top_distinct_news(
        self, posts: List[Dict[str, Any]], count: int = 10, max_retries: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Аналізує масив новин за 6 годин, прибирає дублікати
        і формує ТОП-10 унікальних тем без нав'язливих маркерів.
        Включає автоматичний retry при тимчасових збоях 503 / 429 API.
        """
        if not posts:
            return []

        posts_context = []
        for idx, p in enumerate(posts):
            if p.get("has_video"):
                media_tag = "[ВІДЕО]"
            elif p.get("has_media"):
                media_tag = "[ФОТО]"
            else:
                media_tag = "[ТЕКСТ]"

            posts_context.append(
                f"ID {idx} {media_tag} [{p['channel_title']}]: {p['text']}"
            )

        all_text = "\n---\n".join(posts_context)

        prompt = f"""
Ти — головний редактор провідного новинного Telegram-каналу "Новини UA 6/24".
Перед тобою всі повідомлення з українських медіа за останні 6 годин.

ТВОЄ ЗАВДАННЯ:
1. Відібрати РІВНО {count} НАЙВАЖЛИВІШИХ і принципово РІЗНИХ новинних тем.
2. ВАЖЛИВА ВИМОГА ДО КОНТЕНТУ: серед 10 обраних новин ОБОВ'ЯЗКОВО має бути щонайменше 1-2 важливі події з міткою [ВІДЕО] (робота ППО, фронт, наслідки атак, обміни полоненими тощо).
3. БАЛАНС ТЕМ: фронт/ЗСУ, безпека/обстріли, рішення влади/виплати/закони (з цифрами), ключові міста, міжнародна політика.
4. ЖОДНИХ ДУБЛІВ: одна подія = один зведений пост.
5. Відкидай кримінал, пусті чутки/політичні роздуми та рекламу.
6. Для кожного пункту вкажи `source_id` поста з найкращим джерелом/медіа.

ВИМОГИ ДО ОФОРМЛЕННЯ ТА СТИЛЮ ТЕКСТУ:
- Загальний розмір тексту — до 350 символів.
- Перший рядок: один тематичний емодзі за змістом події (наприклад, 📹, 🚀, ⚖️, 🇰🇵, 🛢, 🛡) + жирний чіткий заголовок.
- Основний текст: 2-3 короткі змістовні речення/абзаци з фактами та цифрами.
- СУВОРА ЗАБОРОНА: НЕ використовуй червоні маркери-булавки (📍, 📌) на початку кожного рядка. Текст має читатися чисто і природно (використовуй звичайний відступ рядка або просте тире "—").

Формат відповіді ВИКЛЮЧНО JSON:
{{
  "news": [
    {{
      "source_id": 0,
      "text": "🛢 <b>У Росії через дефіцит бензину стрімко зріс попит на каністри</b>\\n\\nКількість пошукових запитів на Wildberries за тиждень зросла в чотири рази та перевищила 139 тисяч.\\n\\nДефіцит пального в регіонах загострюється через систематичні удари по російських НПЗ."
    }}
  ]
}}

Новини:
{all_text[:40000]}
"""

        for attempt in range(1, max_retries + 1):
            try:
                response = self.client.models.generate_content(
                    model="gemini-3.6-flash",
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0.2,
                    ),
                )

                raw_text = response.text.strip()
                # Захист від можливих markdown-тегів
                if raw_text.startswith("```json"):
                    raw_text = raw_text[7:]
                if raw_text.startswith("```"):
                    raw_text = raw_text[3:]
                if raw_text.endswith("```"):
                    raw_text = raw_text[:-3]

                data = json.loads(raw_text.strip())
                return data.get("news", [])

            except Exception as e:
                error_str = str(e)
                if (
                    "503" in error_str
                    or "429" in error_str
                    or "UNAVAILABLE" in error_str
                    or "ResourceExhausted" in error_str
                ):
                    if attempt < max_retries:
                        sleep_seconds = attempt * 5
                        logger.warning(
                            f"⚠️ Gemini API тимчасово перевантажено ({error_str[:60]}...). "
                            f"Спроба {attempt}/{max_retries}. Очікування {sleep_seconds} сек..."
                        )
                        time.sleep(sleep_seconds)
                        continue

                logger.error(
                    f"❌ Помилка відбору топ-новин через Gemini (спроба {attempt}/{max_retries}): {e}"
                )
                if attempt == max_retries:
                    return []

        return []
