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
        # Каскад моделей у порядку пріоритету:
        self.models_priority = [
            "gemini-3.5-flash",
            "gemini-3.5-flash-lite",
            "gemini-3.1-flash-lite",
        ]

    def select_top_distinct_news(
        self,
        posts: List[Dict[str, Any]],
        past_titles: Optional[List[str]] = None,
        count: int = 10,
        max_retries_per_model: int = 2
    ) -> List[Dict[str, Any]]:
        """
        Аналізує масив новин за 4 години, прибирає дублікати відносно поточної вибірки
        та відносно вже опублікованих тем за минулу добу.
        Використовує каскадне перемикання на резервні моделі при 503 / 429 збоях.
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

        # Каскадний прохід по моделях
        for model_name in self.models_priority:
            for attempt in range(1, max_retries_per_model + 1):
                try:
                    logger.info(f"Спроба генерації через {model_name} (спроба {attempt}/{max_retries_per_model})...")

                    response = self.client.models.generate_content(
                        model=model_name,
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
                    selected_news = data.get("news", [])
                    if selected_news:
                        logger.info(f"✅ Успішно отримано {len(selected_news)} новин через модель {model_name}")
                        return selected_news

                except Exception as e:
                    error_str = str(e)
                    is_retryable = any(
                        err_code in error_str
                        for err_code in ["503", "429", "UNAVAILABLE", "ResourceExhausted", "high demand", "NOT_FOUND"]
                    )

                    if is_retryable:
                        logger.warning(
                            f"⚠️ Модель {model_name} повернула помилку або недоступна ({error_str[:70]}...)."
                        )
                        if attempt < max_retries_per_model:
                            time.sleep(3 * attempt)
                            continue
                        else:
                            logger.info(f"🔄 Перемикаємося на наступну модель у ланцюжку...")
                            break
                    else:
                        logger.error(f"❌ Помилка запиту до {model_name}: {e}")
                        break

        logger.error("❌ Усі доступні моделі Gemini вичерпали спроби або недоступні.")
        return []
