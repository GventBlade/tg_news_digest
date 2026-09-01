import html
import json
import logging
import math
import re
import time
from typing import Any, Dict, List, Optional, Union
from google import genai
from google.genai import types
from app.config import settings

logger = logging.getLogger(__name__)

# Вагові коефіцієнти джерел за рівнями довіри та якості
SOURCE_TIERS = {
    # Tier A: Еталонні медіа (максимальна довіра) — коефіцієнт 1.2
    "suspilnenews": 1.2,
    "ukrpravda_news": 1.2,
    "babel": 1.2,
    "nvua_official": 1.2,
    "liganet": 1.2,
    "bbcukrainian": 1.2,
    "radiosvoboda": 1.2,
    "forbesukraine": 1.2,

    # Tier B: Спеціалізовані першоджерела та мілітарна аналітика — коефіцієнт 1.1
    "DeepStateUA": 1.1,
    "DIUkraine": 1.1,
    "milinua": 1.1,
    "kpszsu": 1.1,
    "operativnoZSU": 1.1,
    "Tsaplienko": 1.1,

    # Tier C: Швидкі агрегатори (оперативність/медіасигнал) — коефіцієнт 0.9
    "TCH_channel": 0.9,
    "times_ukraina": 0.9,
    "truexanewsua": 0.9,
    "voynareal": 0.9,
    "lachentyt": 0.9,
    "vanek_nikolaev": 0.9,
}


class NewsSummarizer:
    DEFAULT_COUNT = 10
    EDITOR_CANDIDATES = 25
    HISTORY_LIMIT = 150
    MAX_INPUT_CHARS = 55000
    MAX_EVENT_SOURCE_CHARS = 2500
    MAX_NEWS_CHARS = 400
    ALLOWED_CATEGORIES = {
        "war", "politics", "economy", "international", "society",
        "technology", "science", "culture", "other"
    }

    def __init__(self):
        self.client = genai.Client(api_key=settings.GEMINI_API_KEY)
        self.models_priority = [
            "gemini-3.5-flash-lite",
            "gemini-3.1-flash-lite",
        ]

    def select_top_distinct_news(
        self,
        posts: List[Dict[str, Any]],
        past_events: Optional[Union[List[str], List[Dict[str, str]]]] = None,
        count: int = DEFAULT_COUNT,
        max_retries_per_model: int = 2,
    ) -> List[Dict[str, Any]]:
        if not posts:
            return []

        logger.info(f"Формування дайджесту: {len(posts)} постів → пошук найкращих {count} новин")
        posts_context = self._build_posts_context(posts)
        if not posts_context:
            return []

        # 1. AI аналіз усіх унікальних подій
        analyzed_events = self._analyze_events(posts_context, past_events, max_retries_per_model)
        if not analyzed_events:
            logger.warning("Analyzer не повернув подій.")
            return []

        logger.info(f"Analyzer знайшов {len(analyzed_events)} унікальних подій.")

        # 2. Python-ранжування та баланс категорій з урахуванням Tier-ваг
        ranked_events = self._rank_events(analyzed_events, posts)
        if not ranked_events:
            logger.warning("Після Python scoring не залишилося подій.")
            return []

        logger.info(f"Після ranking залишилося {len(ranked_events)} кандидатів.")

        # 3. Вибірка найкращих кандидатів для редактора
        editor_events = ranked_events[:self.EDITOR_CANDIDATES]
        logger.info(f"EDITOR отримує TOP-{len(editor_events)} кандидатів для формування {count} новин.")

        # 4. Формування фінального дайджесту редактором
        final_news = self._generate_final_digest(editor_events, posts, past_events, count, max_retries_per_model)

        # 5. Валідація тексту та виключення дублів
        validated = self._validate_final_news(final_news, ranked_events, posts, count)

        # 6. Fallback-добір при нестачі валідних новин
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
            is_priority = bool(post.get("is_priority"))

            tier_mult = SOURCE_TIERS.get(channel_username, 1.0)

            base_calc = (
                (25 if media_tag == "[ВІДЕО]" else (15 if media_tag == "[ФОТО]" else 0)) +
                min(math.log10(max(views, 1)) * 5, 35)
            )

            score = 1000 if is_priority else (base_calc * tier_mult)
            priority_flag = " ⭐ [ПРІОРИТЕТ АДМІНІСТРАТОРА]" if is_priority else ""

            prepared.append({
                "idx": idx,
                "text": text,
                "media_tag": media_tag,
                "channel_title": channel_title,
                "channel_username": channel_username,
                "views": views,
                "score": score,
                "priority_flag": priority_flag
            })

        if not prepared:
            return ""

        prepared.sort(key=lambda x: x["score"], reverse=True)
        result, current_length = [], 0

        for item in prepared:
            block = f"ID {item['idx']} {item['media_tag']}{item['priority_flag']} [{item['channel_title']}] @{item['channel_username']}\nПерегляди: {item['views']}\n{item['text']}"
            if current_length + len(block) > self.MAX_INPUT_CHARS:
                continue
            result.append(block)
            current_length += len(block) + 10

        return "\n\n---\n\n".join(result)

    def _analyze_events(
        self,
        posts_context: str,
        past_events: Optional[Union[List[str], List[Dict[str, str]]]],
        max_retries: int
    ) -> List[Dict[str, Any]]:
        history_block = self._build_history_block(past_events)
        prompt = f"""Ти — старший новинний аналітик новинної редакції.

ТВОЄ ГОЛОВНЕ ЗАВДАННЯ:
Проаналізуй ВСІ надані Telegram-повідомлення та знайди МАКСИМАЛЬНО ПОВНИЙ список унікальних, актуальних і суспільно значущих новинних подій.
Згрупуй повідомлення, які описують ОДНУ Й ТУ САМУ подію з різних джерел, під один спільний event_id.

Якщо пост містить позначку "⭐ [ПРІОРИТЕТ АДМІНІСТРАТОРА]", обов'язково створи під нього окрему подію з показником importance=100 та novelty=100.

━━━━━━━━━━━━━━━━━━━━
АРХІВ ВЖЕ ОПУБЛІКОВАНИХ ПОДІЙ ЗА 48 ГОДИН (СУВОРО ЗАБОРОНЕНО ПОВТОРЮВАТИ):
{history_block}
━━━━━━━━━━━━━━━━━━━━

СУВОРО ЗАБОРОНЕНО:
1. Повторювати події з АРХІВУ (якщо подія вже висвітлювалася і не має кардинально нових суттєвих фактів — ВІДКИДАЙ її).
2. Брати рутинні обстріли прифронтових міст/сіл без значних наслідків (від 5 загиблих чи руйнувань ТЕЦ/НПЗ).
3. Брати радарний шум: рух дронів, загрози балістики без зафіксованих влучань, тривоги.
4. Брати побутові ДТП, комунальні відключення, локальні перекриття доріг.
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

                is_manual_priority = any(posts[s].get("is_priority") for s in src_ids)

                imp = self._safe_score(ev.get("importance"))
                scale = self._safe_score(ev.get("scale"))
                rel = self._safe_score(ev.get("reliability"))
                pub = self._safe_score(ev.get("public_interest"))
                nov = self._safe_score(ev.get("novelty"))
                med = self._safe_score(ev.get("media_value"))

                has_video = any(posts[s].get("has_video") for s in src_ids)
                has_media = any(posts[s].get("has_media") for s in src_ids)

                # Базовий розрахунок значущості
                base_score = (imp * 0.35 + scale * 0.15 + rel * 0.20 + pub * 0.12 + nov * 0.08 + med * 0.05)

                # Розрахунок середнього вагового коефіцієнта джерел
                tier_multipliers = []
                for s in src_ids:
                    ch_name = str(posts[s].get("channel_username") or "").replace("@", "").strip()
                    tier_multipliers.append(SOURCE_TIERS.get(ch_name, 1.0))
                avg_tier_mult = (sum(tier_multipliers) / len(tier_multipliers)) if tier_multipliers else 1.0

                score = base_score * avg_tier_mult
                score += min(len(src_ids) * 2, 8)

                if is_manual_priority:
                    score += 500  # Гарантований топ слот

                if has_video:
                    score += 5
                elif has_media:
                    score += 3
                else:
                    score -= 3

                if not is_manual_priority:
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
                    "is_priority": is_manual_priority,
                    "category": cat,
                    "raw_score": round(score, 2)
                })
                ranked.append(ev_copy)
            except Exception as e:
                logger.warning(f"Помилка ранжування події: {e}")

        ranked.sort(key=lambda x: x.get("raw_score", 0), reverse=True)

        category_counts: Dict[str, int] = {}
        for ev in ranked:
            if ev.get("is_priority"):
                ev["balanced_score"] = ev["raw_score"]
                continue
            category = ev["category"]
            current = category_counts.get(category, 0)
            penalty = 8 if current >= 4 else (3 if current >= 3 else 0)
            ev["balanced_score"] = round(ev["raw_score"] - penalty, 2)
            category_counts[category] = current + 1

        ranked.sort(key=lambda x: x.get("balanced_score", 0), reverse=True)
        return ranked

    def _generate_final_digest(
        self,
        events: List[Dict[str, Any]],
        posts: List[Dict[str, Any]],
        past_events: Optional[Union[List[str], List[Dict[str, str]]]],
        count: int,
        max_retries: int
    ) -> List[Dict[str, Any]]:
        ev_blocks = []
        for ev in events:
            best_id = ev.get("best_source_id")
            p = posts[best_id]
            media = "[ВІДЕО]" if p.get("has_video") else ("[ФОТО]" if p.get("has_media") else "[ТЕКСТ]")
            p_flag = " ⭐ [ПРІОРИТЕТ АДМІНІСТРАТОРА]" if ev.get("is_priority") else ""

            ev_blocks.append(
                f"=== EVENT_ID: {ev.get('event_id')}{p_flag} ===\n"
                f"МЕДІА: {media} [{p.get('channel_title', 'Джерело')}]\n"
                f"СУТЬ: {ev.get('summary', '')}\n"
                f"ТЕКСТ ДЖЕРЕЛА: {p.get('text', '')[:self.MAX_EVENT_SOURCE_CHARS]}\n"
            )

        history_block = self._build_history_block(past_events)
        prompt = f"""Ти — головний редактор новинного Telegram-каналу.
Створи фінальний дайджест із РІВНО {count} найкращих новин із наданого пулу кандидатів.

ОБОВ'ЯЗКОВО: Якщо серед кандидатів є новини з позначкою "⭐ [ПРІОРИТЕТ АДМІНІСТРАТОРА]", вони ПОВИННІ бути включені до випуску та розміщені на самому початку!

━━━━━━━━━━━━━━━━━━━━
АРХІВ ВЖЕ ОПУБЛІКОВАНИХ ПОДІЙ ЗА 48 ГОДИН (СУВОРО ЗАБОРОНЕНО ПОВТОРЮВАТИ):
{history_block}
━━━━━━━━━━━━━━━━━━━━

ВИМОГИ:
1. Вибери РІВНО {count} найважливіших та найрізноманітніших EVENT_ID.
2. Текст новини повинен строго відповідати своєму EVENT_ID.
3. Розмір: до {self.MAX_NEWS_CHARS} символів.
4. Перший рядок: ОДИН тематичний емодзі (🇺🇦, 🇺🇸, 💥, 🚀, ⚖️, 🏛, 🛢, 📹, 🌍, 💰, ⚡) + <b>Короткий жирний заголовок</b>.
5. Основний текст: 2-4 змістовні речення з фактами, цифрами та контекстом.
6. СУВОРО ЗАБОРОНЕНО вставляти технічні маркери: "[ФОТО]", "[ВІДЕО]", "[ТЕКСТ]", "ФОТО:", "ВІДЕО:".
7. ЗАБОРОНЕНО клікбейт та непідтверджені чутки.

Формат відповіді ТІЛЬКИ JSON:
{{
  "news": [
    {{
      "event_id": "E1",
      "text": "🛢 <b>Заголовок новини</b>\\n\\nРозширений фактологічний текст події."
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
                    "event_id": eid,
                    "source_id": ev["best_source_id"],
                    "summary": ev.get("summary", ""),
                    "category": ev.get("category", "other"),
                    "text": text.strip()
                })

        return final_list

    def _validate_final_news(
        self,
        news: List[Dict[str, Any]],
        ranked_events: List[Dict[str, Any]],
        posts: List[Dict[str, Any]],
        count: int
    ) -> List[Dict[str, Any]]:
        if not isinstance(news, list):
            return []

        validated, used_source_ids, used_event_ids = [], set(), set()
        for item in news:
            if not isinstance(item, dict):
                continue
            s_id, text = item.get("source_id"), item.get("text")
            e_id = item.get("event_id")

            if not (isinstance(s_id, int) and 0 <= s_id < len(posts)):
                continue
            if s_id in used_source_ids or (e_id and e_id in used_event_ids):
                continue
            if not isinstance(text, str) or not text.strip():
                continue

            text = text.strip()
            text = re.sub(r'\[(?:ФОТО|ВІДЕО|ТЕКСТ|PHOTO|VIDEO|TEXT)\]\s*', '', text, flags=re.IGNORECASE)
            text = re.sub(r'(?:ФОТО|ВІДЕО|ТЕКСТ):\s*', '', text, flags=re.IGNORECASE)
            text = text.replace("📍", "").replace("📌", "")
            text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)

            if "<b>" not in text or "</b>" not in text:
                lines = text.split("\n", 1)
                first_line = lines[0].strip()
                rest = ("\n" + lines[1]) if len(lines) > 1 else ""
                text = f"<b>{first_line}</b>{rest}"

            if len(text) > self.MAX_NEWS_CHARS:
                text = text[:self.MAX_NEWS_CHARS]
                last_sp = text.rfind(" ")
                text = (text[:last_sp].rstrip() if last_sp > 250 else text) + "…"
                if "<b>" in text and "</b>" not in text:
                    text += "</b>"

            validated.append({
                "source_id": s_id,
                "text": text.strip(),
                "summary": item.get("summary", ""),
                "category": item.get("category", "other")
            })
            used_source_ids.add(s_id)
            if e_id:
                used_event_ids.add(e_id)

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
            summary = (ev.get("summary") or raw_text[:320]).strip()
            category = ev.get("category", "other")
            emoji = emoji_map.get(category, "📰")

            fallback_text = f"{emoji} <b>{headline}</b>\n\n{summary}"

            if len(fallback_text) > self.MAX_NEWS_CHARS:
                fallback_text = fallback_text[:self.MAX_NEWS_CHARS]
                last_sp = fallback_text.rfind(" ")
                fallback_text = (fallback_text[:last_sp].rstrip() if last_sp > 250 else fallback_text) + "…"
                if "<b>" in fallback_text and "</b>" not in fallback_text:
                    fallback_text += "</b>"

            result.append({
                "source_id": source_id,
                "text": fallback_text,
                "summary": summary,
                "category": category
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

    def _build_history_block(self, past_events: Optional[Union[List[str], List[Dict[str, str]]]]) -> str:
        if not past_events:
            return "Історія опублікованих подій порожня."

        items = past_events[:self.HISTORY_LIMIT]
        formatted_lines = []

        for itm in items:
            if isinstance(itm, dict):
                title = itm.get("title", "").strip()
                summary = itm.get("summary", "").strip()
                pub_at = itm.get("published_at", "").strip()
                if title or summary:
                    time_info = f" [{pub_at}]" if pub_at else ""
                    desc = f" — {summary}" if summary else ""
                    formatted_lines.append(f"- {title}{desc}{time_info}")
            elif isinstance(itm, str) and itm.strip():
                formatted_lines.append(f"- {itm.strip()}")

        return "\n".join(formatted_lines) if formatted_lines else "Історія опублікованих подій порожня."

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
            if p.get("is_priority"):
                return 10000
            bonus = 20 if p.get("has_video") else (10 if p.get("has_media") else 0)
            views = int(p.get("views") or 0)
            ch_name = str(p.get("channel_username") or "").replace("@", "").strip()
            tier_mult = SOURCE_TIERS.get(ch_name, 1.0)
            return (bonus + min(math.log10(max(views, 1)) * 5, 30)) * tier_mult

        return max(source_ids, key=score)
