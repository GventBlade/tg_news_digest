import json
import logging
import time
from typing import Any, Dict, List, Optional
from google import genai
from google.genai import types
from app.config import settings

logger = logging.getLogger(__name__)


class NewsSummarizer:
    def __init__(self):
        self.client = genai.Client(api_key=settings.GEMINI_API_KEY)

    def select_top_distinct_news(
        self,
        posts: List[Dict[str, Any]],
        past_titles: Optional[List[str]] = None,
        count: int = 10,
        max_retries: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Аналізує масив новин за 4 години, прибирає дублікати відносно поточної вибірки
        та відносно вже опублікованих тем за минулу добу.
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

        history_block = ""
        if past_titles:
            recent_list = "\n- ".join(past_titles[-30:])
            history_block = f"""
ВЖЕ ОПУБЛІКОВАНІ ТЕМИ ЗА ДОБУ (СУВОРО ЗАБОРОНЕНО ПОВТОРЮВАТИ АБО РОБИТИ СХОЖІ ПОСТИ НА ЦІ ТЕМИ):
- {recent_list}
"""

        prompt = f"""
Ти — головний редактор провідного новинного Telegram-каналу "Новини UA 4/24".
Перед тобою всі повідомлення з українських медіа за останні години.

ТВОЄ ЗАВДАННЯ:
1. Відібрати РІВНО {count} НАЙВАЖЛИВІШИХ і принципово РІЗНИХ новинних тем.
2. ВАЖЛИВА ВИМОГА ДО КОНТЕНТУ: серед обраних новин ОБОВ'ЯЗКОВО має бути щонайменше 1-2 важливі події з міткою [ВІДЕО] (робота ППО, фронт, наслідки атак, обміни полоненими тощо).
3. БАЛАНС ТЕМ: фронт/ЗСУ, важливі рішення влади/закони/виплати (з цифрами), ключові міста, міжнародна політика.
4. Для кожного пункту вкажи `source_id` поста з найкращим джерелом/медіа.

СУВОРІ ЗАБОРОНИ ТА ФІЛЬТРИ (ВІДКИДАТИ ОДРАЗУ):
- СИТУАТИВНИЙ РАДАРНИЙ МОНІТОРИНГ ТА ТРИВОГИ: ЗАБОРОНЕНО публікувати рух БпЛА, "Шахеди курсують на Дніпро/Одесу", загрози балістики, пуски КАБів чи вибухи без підтверджених наслідків. Канал не є радаром тривог.
- ПОВТОРИ: ЖОДНИХ дублів як всередині поточної вибірки, так і з темами, які вже публікувалися раніше (звіряйся зі списком нижче).
- ДРІБНИЙ ПОБУТОВИЙ КРИМІНАЛ (якщо це не резонансна подія національного масштабу), чутки та рекламу.
{history_block}
ВИМОГИ ДО ОФОРМЛЕННЯ ТА СТИЛЮ ТЕКСТУ:
- Загальний розмір тексту — до 350 символів.
- Перший рядок: один тематичний емодзі за змістом події (наприклад, 📹, 🚀, ⚖️, 🏛, 🛢, 🛡) + жирний чіткий заголовок.
- Основний текст: 2-3 короткі змістовні речення/абзаци з фактами та цифрами.
- СУВОРА ЗАБОРОНА: НЕ використовуй червоні маркери-булавки (📍, 📌) на початку кожного рядка. Текст має читатися чисто (використовуй звичайний відступ або тире "—").

Формат відповіді ВИКЛЮЧНО JSON:
{{
  "news": [
    {{
      "source_id": 0,
      "text": "🛢 <b>У Росії через дефіцит бензину стрімко зріс попит на каністри</b>\\n\\nКількість пошукових запитів на маркетплейсах зросла в чотири рази.\\n\\nДефіцит загострюється через систематичні удари по російських НПЗ."
    }}
  ]
}}

Новини для обробки:
{all_text[:40000]}
"""

        for attempt in range(1, max_retries + 1):
            try:
                response = self.client.models.generate_content(
                    model="gemini-3.7-flash",
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0.2,
                    ),
                )

                raw_text = response.text.strip()
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