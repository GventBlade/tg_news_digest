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
    ANALYZER_CANDIDATES = 25
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
        self.models_priority = ["gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]

    def select_top_distinct_news(
        self,
        posts: List[Dict[str, Any]],
        past_titles: Optional[List[str]] = None,
        count: int = DEFAULT_COUNT,
        max_retries_per_model: int = 2,
    ) -> List[Dict[str, Any]]:
        if not posts:
            return []

        logger.info(f"Формування дайджесту: {len(posts)} постів → TOP {count}")
        posts_context = self._build_posts_context(posts)
        if not posts_context:
            return []

        # Крок 1: Аналіз, дедуплікація та відсіювання вже опублікованого
        analyzed_events = self._analyze_events(posts_context, past_titles, max_retries_per_model)
        if not analyzed_events:
            return []

        # Крок 2: Python Scoring та балансування
        ranked_events = self._rank_events(analyzed_events, posts)
        ranked_events = ranked_events[:max(count * 2, self.ANALYZER_CANDIDATES)]

        # Крок 3: Фінальна генерація текстів редактором із жорсткою прив'язкою за event_id
        final_news = self._generate_final_digest(ranked_events, posts, past_titles, count, max_retries_per_model)

        # Крок 4: Валідація результату
        validated = self._validate_final_news(final_news, posts, count)
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

            prepared.append({
                "idx": idx,
                "text": text,
                "media_tag": media_tag,
                "channel_title": channel_title,
                "channel_username": channel_username,
                "views": views,
                "score": (25 if media_tag == "[ВІДЕО]" else (15 if media_tag == "[ФОТО]" else 0)) + min(
                    math.log10(max(views, 1)) * 5, 35)
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

    def _analyze_events(self, posts_context: str, past_titles: Optional[List[str]], max_retries: int) -> List[
        Dict[str, Any]]:
        history_block = self._build_history_block(past_titles)
        prompt = f"""Ти — старший новинний аналітик редакції "Новини UA 6/24".
ТВОЄ ЗАВДАННЯ:
1. Знайти нові унікальні новинні події.
2. Об'єднати всі повідомлення різних каналів про одну й ту саму подію в одну EVENT.
3. СУВОРО ВІДСІЯТИ теми, які вже публікувалися (дивись список АРХІВ нижче). Якщо подія про смерть монарха, зсуви в Непалі чи танкери/нафту вже була — ПОВТОРНО НЕ БРАТИ!

━━━━━━━━━━━━━━━━━━━━
АРХІВ ВЖЕ ОПУБЛІКОВАНИХ ТЕМ (СУВОРО ЗАБОРОНЕНО ВИБИРАТИ СХОЖІ ТЕМИ):
{history_block}
━━━━━━━━━━━━━━━━━━━━

СУВОРІ ФІЛЬТРИ:
- ЗАБОРОНЕНО брати теми з АРХІВУ (інше формулювання тієї самої новини НЕ робить її новою).
- ЗАБОРОНЕНО рутинні обстріли прифронтових міст/сіл (Нікополь, прифронтовий Херсон, села Сумщини) без масових жертв (від 5 загиблих) чи руйнувань ТЕЦ/НПЗ.
- ЗАБОРОНЕНО радарний шум (рух дронів, загрози балістики без влучань), комуналку, перекриття доріг, дрібні ДТП.

ОЦІНКА ПОКАЗНИКІВ (0-100): importance, scale, reliability, public_interest, novelty, media_value.

Відповідь ТІЛЬКИ у форматі JSON:
{{
  "events": [
    {{
      "event_id": "E1",
      "source_ids": [0, 2],
      "best_source_id": 0,
      "category": "war",
      "importance": 90,
      "scale": 85,
      "reliability": 90,
      "public_interest": 90,
      "novelty": 80,
      "media_value": 85,
      "headline_hint": "Масована атака на Київ",
      "summary": "Короткий опис"
    }}
  ]
}}

TELEGRAM POSTS:
{posts_context}"""

        data = self._call_json_with_cascade(prompt, max_retries, "ANALYZER")
        return data.get("events", []) if data and isinstance(data.get("events"), list) else []

    def _rank_events(self, events: List[Dict[str, Any]], posts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        ranked, cat_counts = [], {}

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

                score = (imp * 0.35 + scale * 0.15 + rel * 0.20 + pub * 0.12 + nov * 0.08 + med * 0.05 + min(
                    len(src_ids) * 2, 8))
                score += 5 if has_video else (3 if has_media else -5)
                if rel < 45:
                    score -= 15
                elif rel < 60:
                    score -= 7
                if len(src_ids) == 1:
                    score -= 2

                cat = ev.get("category", "other")
                if cat not in self.ALLOWED_CATEGORIES:
                    cat = "other"

                cur_c = cat_counts.get(cat, 0)
                bal_score = score - (8 if cur_c >= 4 else (3 if cur_c >= 3 else 0))
                cat_counts[cat] = cur_c + 1

                ev_copy = dict(ev)
                ev_copy.update({
                    "source_ids": src_ids,
                    "best_source_id": self._select_best_source(src_ids, posts),
                    "has_video": has_video,
                    "has_media": has_media,
                    "category": cat,
                    "balanced_score": round(bal_score, 2)
                })
                ranked.append(ev_copy)
            except Exception as e:
                logger.warning(f"Помилка ранжування події: {e}")

        ranked.sort(key=lambda x: x.get("balanced_score", 0), reverse=True)
        return ranked

    def _generate_final_digest(
        self, events: List[Dict[str, Any]], posts: List[Dict[str, Any]],
        past_titles: Optional[List[str]], count: int, max_retries: int
    ) -> List[Dict[str, Any]]:
        # Відбираємо рівно потрібну кількість подій
        target_events = events[:count]
        if not target_events:
            return []

        ev_blocks = []
        for ev in target_events:
            best_id = ev.get("best_source_id")
            p = posts[best_id]
            media = "[ВІДЕО]" if p.get("has_video") else ("[ФОТО]" if p.get("has_media") else "[ТЕКСТ]")

            # Передаємо лише першоджерело, закріплене за цією подією
            ev_blocks.append(
                f"=== EVENT_ID: {ev.get('event_id')} ===\n"
                f"МЕДІА ДЖЕРЕЛА: {media} [{p.get('channel_title', 'Джерело')}]\n"
                f"СУТЬ: {ev.get('summary', '')}\n"
                f"ТЕКСТ ДЖЕРЕЛА: {p.get('text', '')[:self.MAX_EVENT_SOURCE_CHARS]}\n"
            )

        history_block = self._build_history_block(past_titles)
        prompt = f"""Ти — головний редактор Telegram-каналу "Новини UA 6/24".
Твоє завдання: написати лаконічні фінальні публікації для кожної події зі списку нижче.

━━━━━━━━━━━━━━━━━━━━
АРХІВ ВЖЕ ОПУБЛІКОВАНИХ ТЕМ (СУВОРО ЗАБОРОНЕНО ПОВТОРЮВАТИ):
{history_block}
━━━━━━━━━━━━━━━━━━━━

ВИМОГИ:
1. Напиши новину ДЛЯ КОЖНОГО наданого EVENT_ID. Текст новини повинен строго описувати тільки свій EVENT_ID.
2. Розмір: до 350 символів.
3. Перший рядок: ОДИН тематичний емодзі (🇺🇦, 🇺🇸, 💥, 🚀, ⚖️, 🏛, 🛢, 📹, 🌍, 💰, ⚡) + <b>Короткий жирний заголовок</b>.
4. Основний текст: 2-3 короткі речення з чіткими фактами і цифрами.
5. СУВОРО ЗАБОРОНЕНО вставляти слова: "[ФОТО]", "[ВІДЕО]", "[ТЕКСТ]", "ФОТО:", "ВІДЕО:".
6. ЗАБОРОНЕНО маркери 📍, 📌, клікбейт та непідтверджені чутки.

Формат відповіді ТІЛЬКИ JSON:
{{
  "news": [
    {{
      "event_id": "E1",
      "text": "🛢 <b>Заголовок новини</b>\\n\\nТекст події з фактами."
    }}
  ]
}}

ПОДІЇ ДЛЯ РЕДАГУВАННЯ:
{chr(10).join(ev_blocks)}"""

        data = self._call_json_with_cascade(prompt, max_retries, "EDITOR")
        raw_news = data.get("news", []) if data and isinstance(data.get("news"), list) else []

        # Співставляємо відповідь моделі суворо з закріпленим best_source_id
        event_map = {ev["event_id"]: ev for ev in target_events}
        final_list = []

        for item in raw_news:
            if not isinstance(item, dict):
                continue
            eid = item.get("event_id")
            text = item.get("text")
            if eid in event_map and isinstance(text, str) and text.strip():
                ev = event_map[eid]
                final_list.append({
                    "source_id": ev["best_source_id"],  # Точний індекс поста
                    "text": text.strip()
                })

        return final_list

    def _validate_final_news(self, news: List[Dict[str, Any]], posts: List[Dict[str, Any]], count: int) -> List[
        Dict[str, Any]]:
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

            # Примусово очищаємо залишки службових слів
            text = re.sub(r'\[(ФОТО\vert{}ВІДЕО\vert{}ТЕКСТ\vert{}PHOTO\vert{}VIDEO\vert{}TEXT)\]\s*', '', text,
                          flags=re.IGNORECASE)
            text = re.sub(r'(ФОТО|ВІДЕО|ТЕКСТ):\s*', '', text, flags=re.IGNORECASE)

            if "📍" in text or "📌" in text or "<b>" not in text or "</b>" not in text:
                continue

            if len(text) > self.MAX_NEWS_CHARS:
                text = text[:self.MAX_NEWS_CHARS]
                last_sp = text.rfind(" ")
                text = (text[:last_sp].rstrip() if last_sp > 200 else text) + "…"

            validated.append({"source_id": s_id, "text": text.strip()})
            used_ids.add(s_id)
            if len(validated) >= count:
                break

        return validated

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
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
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
            return bonus + min(math.log10(max(p.get("views", 0) or 0, 1)) * 2, 15)

        return max(source_ids, key=score)
