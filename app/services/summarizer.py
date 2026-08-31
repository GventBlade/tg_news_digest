import html
import json
import logging
import math
import re
import time
from typing import Any, Dict, List, Optional
from google import genai
from google.genai import types
from app.config import settings

logger = logging.getLogger(__name__)


class NewsSummarizer:
    DEFAULT_COUNT = 10
    EDITOR_CANDIDATES = 25
    HISTORY_LIMIT = 100
    MAX_INPUT_CHARS = 55000
    MAX_EVENT_SOURCE_CHARS = 2500
    MAX_NEWS_CHARS = 350
    ALLOWED_CATEGORIES = {
        "war", "politics", "economy", "international", "society",
        "technology", "science", "culture", "other"
    }

    def __init__(self):
        self.client = genai.Client(api_key=settings.GEMINI_API_KEY)
        self.models_priority = ["gemini-2.5-flash", "gemini-2.0-flash"]

    def select_top_distinct_news(
        self,
        posts: List[Dict[str, Any]],
        past_titles: Optional[List[str]] = None,
        count: int = DEFAULT_COUNT,
        max_retries_per_model: int = 2,
    ) -> List[Dict[str, Any]]:
        if not posts:
            return []

        logger.info(f"Формування дайджесту: {len(posts)} постів → пошук найкращих {count} новин")
        posts_context = self._build_posts_context(posts)
        if not posts_context:
            return []

        # 1. AI шукає ВСІ унікальні події без штучного ліміту
        analyzed_events = self._analyze_events(posts_context, past_titles, max_retries_per_model)
        if not analyzed_events:
            logger.warning("Analyzer не повернув подій.")
            return []

        logger.info(f"Analyzer знайшов {len(analyzed_events)} унікальних подій.")

        # 2. Python оцінює всі події та балансує категорії
        ranked_events = self._rank_events(analyzed_events, posts)
        if not ranked_events:
            logger.warning("Після Python scoring не залишилося подій.")
            return []

        logger.info(f"Після ranking залишилося {len(ranked_events)} кандидатів.")

        # 3. Передаємо редактору пул найкращих кандидатів
        editor_events = ranked_events[:self.EDITOR_CANDIDATES]
        logger.info(f"EDITOR отримує TOP-{len(editor_events)} кандидатів для формування {count} новин.")

        # 4. AI-редактор формує фінальні публікації
        final_news = self._generate_final_digest(editor_events, posts, past_titles, count, max_retries_per_model)

        # 5. Валідація тексту та медіа
        validated = self._validate_final_news(final_news, posts, count)

        # 6. Fallback-добір з ranked_events, якщо валідацію пройшло менше ніж count
        if len(validated) < count:
            logger.warning(f"EDITOR сформував {len(validated)}/{count}. Запуск fallback-добору.")
            validated = self._fill_missing_news(validated, ranked_events, posts, count)

        logger.info(f"Фінальний дайджест сформовано: {len(validated)}/{count} новин")
        return validated

    def _build_posts_context(self, posts: List[Dict[str, Any]]) -> str:
        prepared = []
        for idx, post in enumerate(posts):
            text = (post.get("text") or "").strip()
            if not text:
                continue

            media_tag = "[ВІДЕО]" if post.get("has_video") else ("[ФОТО]" if post.get("has_media") else "[ТЕКСТ]")
            channel_title = post.get("channel_title") or post.get("channel_username") or "Джерело"
            channel_username = str(post.get("channel_username") or "").replace("@", "").strip()
            views = int(post.get("views") or 0)

            score = (25 if media_tag == "[ВІДЕО]" else (15 if media_tag == "[ФОТО]" else 0)) + min(
                math.log10(max(views, 1)) * 5, 35
            )

            prepared.append({
                "idx": idx,
                "text": text,
                "media_tag": media_tag,
                "channel_title": channel_title,
                "channel_username": channel_username,
                "views": views,
                "score": score
            })

        if not prepared:
            return ""

        prepared.sort(key=lambda x: x["score"], reverse=True)
        result, current_length = [], 0

        for item in prepared:
            block = f"ID {item['idx']} {item['media_tag']} [{item['channel_title']}] @{item['channel_username']}\nПерегляди: {item['views']}\n{item['text']}"
            if current_length + len(block) > self.MAX_INPUT_CHARS:
                continue
            result.append(block)
            current_length += len(block) + 10

        return "\n\n---\n\n".join(result)

    def _analyze_events(self, posts_context: str, past_titles: Optional[List[str]], max_retries: int) -> List[Dict[str, Any]]:
        history_block = self._build_history_block(past_titles)
        prompt = f"""Ти — старший новинний аналітик новинної редакції.

ТВОЄ ГОЛОВНЕ ЗАВДАННЯ:
Проаналізуй ВСІ надані Telegram-повідомлення та знайди МАКСИМАЛЬНО ПОВНИЙ список унікальних, актуальних і суспільно значущих новинних подій.
НЕ потрібно обмежуватися 10 подіями. Знайди ВСІ якісні події (від 15 до 40+), які варті уваги.

━━━━━━━━━━━━━━━━━━━━
АРХІВ ВЖЕ ОПУБЛІКОВАНИХ ТЕМ (СУВОРО ЗАБОРОНЕНО ПОВТОРЮВАТИ):
{history_block}
━━━━━━━━━━━━━━━━━━━━

СУВОРО ЗАБОРОНЕНО:
1. Повторювати події з АРХІВУ (інше формулювання тієї самої теми НЕ робить її новою).
2. Брати рутинні обстріли прифронтових міст/сіл без значних наслідків (від 5 загиблих чи руйнувань ТЕЦ/НПЗ).
3. Брати радарний шум: рух дронів, загрози балістики без влучань, тривоги.
4. Брати комунальні аварії, дрібні ДТП, локальні перекриття доріг.
5. Брати непідтверджені чутки.

ОЦІНКА ПОКАЗНИКІВ (0-100): importance, scale, reliability, public_interest, novelty, media_value.

Відповідь ТІЛЬКИ у форматі JSON:
{{
  "events": [
    {{
      "event_id": "E1",
      "source_ids": [0, 2, 5],
      "best_source_id": 0,
      "category": "war",
      "importance": 90,
      "scale": 85,
      "reliability": 90,
      "public_interest": 90,
      "novelty": 80,
      "media_value": 85,
      "headline_hint": "Масована атака на енергетику",
      "summary": "Короткий фактологічний опис події"
    }}
  ]
}}

TELEGRAM POSTS:
{posts_context}"""

        data = self._call_json_with_cascade(prompt, max_retries, "ANALYZER")
        return data.get("events", []) if data and isinstance(data.get("events"), list) else []

    def _rank_events(self, events: List[Dict[str, Any]], posts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        ranked = []

        for ev in events:
            try:
                src_ids = [s for s in ev.get("source_ids", []) if isinstance(s, int) and 0 <= s < len(posts)]
                best_id = ev.get("best_source_id")
                if not src_ids and isinstance(best_id, int) and 0 <= best_id < len(posts):
                    src_ids = [best_id]
                if not src_ids:
                    continue

                imp = self._safe_score(ev.get("importance"))
                scale = self._safe_score(ev.get("scale"))
                rel = self._safe_score(ev.get("reliability"))
                pub = self._safe_score(ev.get("public_interest"))
                nov = self._safe_score(ev.get("novelty"))
                med = self._safe_score(ev.get("media_value"))

                has_video = any(posts[s].get("has_video") for s in src_ids)
                has_media = any(posts[s].get("has_media") for s in src_ids)

                score = (imp * 0.35 + scale * 0.15 + rel * 0.20 + pub * 0.12 + nov * 0.08 + med * 0.05)
                score += min(len(src_ids) * 2, 8)

                if has_video:
                    score += 5
                elif has_media:
                    score += 3
                else:
                    score -= 3

                if rel < 45: score -= 15
                elif rel < 60: score -= 7
                if len(src_ids) == 1: score -= 2

                cat = ev.get("category", "other")
                if cat not in self.ALLOWED_CATEGORIES:
                    cat = "other"

                ev_copy = dict(ev)
                ev_copy.update({
                    "source_ids": src_ids,
                    "best_source_id": self._select_best_source(src_ids, posts),
                    "has_video": has_video,
                    "has_media": has_media,
                    "category": cat,
                    "raw_score": round(score, 2)
                })
                ranked.append(ev_copy)
            except Exception as e:
                logger.warning(f"Помилка ранжування події: {e}")

        ranked.sort(key=lambda x: x.get("raw_score", 0), reverse=True)

        # Баланс категорій
        category_counts: Dict[str, int] = {}
        for ev in ranked:
            category = ev["category"]
            current = category_counts.get(category, 0)
            penalty = 8 if current >= 4 else (3 if current >= 3 else 0)
            ev["balanced_score"] = round(ev["raw_score"] - penalty, 2)
            category_counts[category] = current + 1

        ranked.sort(key=lambda x: x.get("balanced_score", 0), reverse=True)
        return ranked

    def _generate_final_digest(
        self, events: List[Dict[str, Any]], posts: List[Dict[str, Any]],
        past_titles: Optional[List[str]], count: int, max_retries: int
    ) -> List[Dict[str, Any]]:
        ev_blocks = []
        for ev in events:
            best_id = ev.get("best_source_id")
            p = posts[best_id]
            media = "[ВІДЕО]" if p.get("has_video") else ("[ФОТО]" if p.get("has_media") else "[ТЕКСТ]")

            ev_blocks.append(
                f"=== EVENT_ID: {ev.get('event_id')} ===\n"
                f"МЕДІА: {media} [{p.get('channel_title', 'Джерело')}]\n"
                f"СУТЬ: {ev.get('summary', '')}\n"
                f"ТЕКСТ ДЖЕРЕЛА: {p.get('text', '')[:self.MAX_EVENT_SOURCE_CHARS]}\n"
            )

        history_block = self._build_history_block(past_titles)
        prompt = f"""Ти — головний редактор новинного Telegram-каналу.
Створи фінальний дайджест із РІВНО {count} найкращих новин із наданого пулу кандидатів.

━━━━━━━━━━━━━━━━━━━━
АРХІВ ВЖЕ ОПУБЛІКОВАНИХ ТЕМ (СУВОРО ЗАБОРОНЕНО ПОВТОРЮВАТИ):
{history_block}
━━━━━━━━━━━━━━━━━━━━

ВИМОГИ:
1. Вибери РІВНО {count} найважливіших та найрізноманітніших EVENT_ID.
2. Текст новини повинен строго відповідати своєму EVENT_ID.
3. Розмір: до 350 символів.
4. Перший рядок: ОДИН тематичний емодзі (🇺🇦, 🇺🇸, 💥, 🚀, ⚖️, 🏛, 🛢, 📹, 🌍, 💰, ⚡) + <b>Короткий жирний заголовок</b>.
5. Основний текст: 2-3 короткі речення з чіткими фактами і цифрами.
6. СУВОРО ЗАБОРОНЕНО вставляти слова: "[ФОТО]", "[ВІДЕО]", "[ТЕКСТ]", "ФОТО:", "ВІДЕО:".
7. ЗАБОРОНЕНО маркери 📍, 📌, клікбейт та непідтверджені чутки.

Формат відповіді ТІЛЬКИ JSON:
{{
  "news": [
    {{
      "event_id": "E1",
      "text": "🛢 <b>Заголовок новини</b>\\n\\nТекст події з фактами."
    }}
  ]
}}

КАНДИДАТИ ДЛЯ ДАЙДЖЕСТУ:
{chr(10).join(ev_blocks)}"""

        data = self._call_json_with_cascade(prompt, max_retries, "EDITOR")
        raw_news = data.get("news", []) if data and isinstance(data.get("news"), list) else []

        event_map = {ev["event_id"]: ev for ev in events}
        final_list = []

        for item in raw_news:
            if not isinstance(item, dict):
                continue
            eid = item.get("event_id")
            text = item.get("text")
            if eid in event_map and isinstance(text, str) and text.strip():
                ev = event_map[eid]
                final_list.append({
                    "source_id": ev["best_source_id"],
                    "text": text.strip()
                })

        return final_list

    def _validate_final_news(self, news: List[Dict[str, Any]], posts: List[Dict[str, Any]], count: int) -> List[Dict[str, Any]]:
        if not isinstance(news, list):
            return []

        validated, used_ids = [], set()
        for item in news:
            if not isinstance(item, dict):
                continue
            s_id, text = item.get("source_id"), item.get("text")
            if not (isinstance(s_id, int) and 0 <= s_id < len(posts)) or s_id in used_ids:
                continue
            if not isinstance(text, str) or not text.strip():
                continue

            text = text.strip()
            text = re.sub(r'\[(?:ФОТО|ВІДЕО|ТЕКСТ|PHOTO|VIDEO|TEXT)\]\s*', '', text, flags=re.IGNORECASE)
            text = re.sub(r'(?:ФОТО|ВІДЕО|ТЕКСТ):\s*', '', text, flags=re.IGNORECASE)
            text = text.replace("📍", "").replace("📌", "")

            # Автовиправлення Markdown жирного шрифту на HTML
            text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)

            if "<b>" not in text or "</b>" not in text:
                lines = text.split("\n", 1)
                first_line = lines[0].strip()
                rest = ("\n" + lines[1]) if len(lines) > 1 else ""
                text = f"<b>{first_line}</b>{rest}"

            if len(text) > self.MAX_NEWS_CHARS:
                text = text[:self.MAX_NEWS_CHARS]
                last_sp = text.rfind(" ")
                text = (text[:last_sp].rstrip() if last_sp > 200 else text) + "…"
                if "<b>" in text and "</b>" not in text:
                    text += "</b>"

            validated.append({"source_id": s_id, "text": text.strip()})
            used_ids.add(s_id)
            if len(validated) >= count:
                break

        return validated

    def _fill_missing_news(
        self,
        validated: List[Dict[str, Any]],
        ranked_events: List[Dict[str, Any]],
        posts: List[Dict[str, Any]],
        count: int
    ) -> List[Dict[str, Any]]:
        result = list(validated)
        used_source_ids = {
            item["source_id"] for item in result
            if isinstance(item, dict) and isinstance(item.get("source_id"), int)
        }
        used_event_ids = set()

        for item in result:
            s_id = item.get("source_id")
            for ev in ranked_events:
                if s_id in ev.get("source_ids", []):
                    used_event_ids.add(ev.get("event_id"))
                    break

        emoji_map = {
            "war": "💥", "politics": "🏛", "economy": "💰",
            "international": "🌍", "society": "🇺🇦", "technology": "⚡",
            "science": "🔬", "culture": "🎭", "other": "📰"
        }

        for ev in ranked_events:
            if len(result) >= count:
                break

            event_id = ev.get("event_id")
            if event_id in used_event_ids:
                continue

            source_id = ev.get("best_source_id")
            if not isinstance(source_id, int) or source_id in used_source_ids:
                continue

            post = posts[source_id]
            raw_text = (post.get("text") or "").strip()
            if not raw_text:
                continue

            headline = (ev.get("headline_hint") or "Важлива новина").strip()
            summary = (ev.get("summary") or raw_text[:280]).strip()
            category = ev.get("category", "other")
            emoji = emoji_map.get(category, "📰")

            fallback_text = f"{emoji} <b>{headline}</b>\n\n{summary}"

            if len(fallback_text) > self.MAX_NEWS_CHARS:
                fallback_text = fallback_text[:self.MAX_NEWS_CHARS]
                last_sp = fallback_text.rfind(" ")
                fallback_text = (fallback_text[:last_sp].rstrip() if last_sp > 200 else fallback_text) + "…"
                if "<b>" in fallback_text and "</b>" not in fallback_text:
                    fallback_text += "</b>"

            result.append({
                "source_id": source_id,
                "text": fallback_text
            })
            used_source_ids.add(source_id)
            used_event_ids.add(event_id)

        return result[:count]

    def _call_json_with_cascade(self, prompt: str, max_retries: int, op_name: str) -> Optional[Dict[str, Any]]:
        for model in self.models_priority:
            for attempt in range(1, max_retries + 1):
                try:
                    logger.info(f"{op_name}: спроба {attempt}/{max_retries} через {model}...")
                    response = self.client.models.generate_content(
                        model=model,
                        contents=prompt,
                        config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.2),
                    )
                    raw_text = self._clean_json_response((response.text or "").strip())
                    data = json.loads(raw_text)
                    if isinstance(data, dict):
                        return data
                except Exception as e:
                    err = str(e)
                    if any(c in err for c in ["503", "429", "UNAVAILABLE", "ResourceExhausted", "NOT_FOUND"]):
                        if attempt < max_retries:
                            time.sleep(3 * attempt)
                            continue
                        break
                    logger.error(f"Помилка {op_name} ({model}): {e}")
                    break
        return None

    @staticmethod
    def _clean_json_response(text: str) -> str:
        text = text.strip()
        if text.startswith("```json"): text = text[7:]
        elif text.startswith("```"): text = text[3:]
        if text.endswith("```"): text = text[:-3]
        return text.strip()

    def _build_history_block(self, past_titles: Optional[List[str]]) -> str:
        if not past_titles:
            return "Історія опублікованих тем відсутня."
        titles = [t.strip() for t in past_titles[-self.HISTORY_LIMIT:] if t and t.strip()]
        return "\n".join(f"- {t}" for t in titles) if titles else "Історія опублікованих тем відсутня."

    @staticmethod
    def _safe_score(val: Any) -> float:
        try:
            return max(0.0, min(100.0, float(val)))
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _select_best_source(source_ids: List[int], posts: List[Dict[str, Any]]) -> int:
        def score(s_id: int) -> float:
            p = posts[s_id]
            bonus = 20 if p.get("has_video") else (10 if p.get("has_media") else 0)
            views = int(p.get("views") or 0)
            return bonus + min(math.log10(max(views, 1)) * 5, 30)

        return max(source_ids, key=score)
