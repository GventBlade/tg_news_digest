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


SOURCE_TIERS = {
    "suspilnenews": 1.2,
    "ukrpravda_news": 1.2,
    "babel": 1.2,
    "nvua_official": 1.2,
    "liganet": 1.2,
    "bbcukrainian": 1.2,
    "radiosvoboda": 1.2,
    "forbesukraine": 1.2,

    "DeepStateUA": 1.1,
    "DIUkraine": 1.1,
    "milinua": 1.1,
    "kpszsu": 1.1,
    "operativnoZSU": 1.1,
    "Tsaplienko": 1.1,

    "TCH_channel": 0.9,
    "times_ukraina": 0.9,
    "truexanewsua": 0.9,
    "voynareal": 0.9,
    "lachentyt": 0.9,
    "vanek_nikolaev": 0.9,
}


LOW_VALUE_EVENT_TYPES = {
    "routine_attack",
    "routine_statement",
    "minor_local_event",
    "minor_accident",
    "alert_only",
}


class NewsSummarizer:
    DEFAULT_COUNT = 10
    MIN_DIGEST_COUNT = 5
    EDITOR_CANDIDATES = 25
    HISTORY_LIMIT = 150

    MAX_INPUT_CHARS = 55000
    MAX_EVENT_SOURCE_CHARS = 2500
    MAX_NEWS_CHARS = 500

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

        logger.info(f"Формування дайджесту: {len(posts)} постів → максимум {count} новин")

        posts_context = self._build_posts_context(posts)
        if not posts_context:
            return []

        analyzed_events = self._analyze_events(posts_context, past_events, max_retries_per_model)
        if not analyzed_events:
            logger.warning("Analyzer не повернув подій.")
            return []

        logger.info(f"Analyzer знайшов {len(analyzed_events)} потенційних подій.")

        ranked_events = self._rank_events(analyzed_events, posts)
        if not ranked_events:
            logger.warning("Після editorial gate не залишилось подій.")
            return []

        logger.info(f"Після ranking залишилось {len(ranked_events)} подій.")

        editor_events = ranked_events[:self.EDITOR_CANDIDATES]
        final_news = self._generate_final_digest(
            editor_events, posts, past_events, count, max_retries_per_model
        )

        validated = self._validate_final_news(final_news, ranked_events, posts, count)

        if len(validated) < self.MIN_DIGEST_COUNT:
            logger.warning(
                f"EDITOR сформував лише {len(validated)} новин. "
                f"Fallback до мінімуму {self.MIN_DIGEST_COUNT}."
            )
            validated = self._fill_missing_news(
                validated,
                ranked_events,
                posts,
                min(count, self.MIN_DIGEST_COUNT)
            )

        logger.info(f"Фінальний дайджест: {len(validated)} новин.")
        return validated[:count]

    def _build_posts_context(self, posts: List[Dict[str, Any]]) -> str:
        prepared = []

        for idx, post in enumerate(posts):
            text = (post.get("text") or "").strip()
            if not text:
                continue

            media_tag = "[ВІДЕО]" if post.get("has_video") else (
                "[ФОТО]" if post.get("has_media") else "[ТЕКСТ]"
            )

            channel_title = post.get("channel_title") or post.get("channel_username") or "Джерело"
            channel_username = str(post.get("channel_username") or "").replace("@", "").strip()

            views = int(post.get("views") or 0)
            forwards = int(post.get("forwards") or 0)
            replies = int(post.get("replies") or 0)
            is_priority = bool(post.get("is_priority"))

            tier_mult = self._get_source_multiplier(channel_username)

            engagement_score = (
                min(math.log10(max(views, 1)) * 4, 26)
                + min(math.log10(max(forwards, 1)) * 3, 12)
                + min(math.log10(max(replies, 1)) * 2, 8)
            )

            media_bonus = 8 if post.get("has_video") else (4 if post.get("has_media") else 0)
            score = 10000 if is_priority else (engagement_score + media_bonus) * tier_mult

            prepared.append({
                "idx": idx,
                "text": text,
                "media_tag": media_tag,
                "channel_title": channel_title,
                "channel_username": channel_username,
                "views": views,
                "forwards": forwards,
                "replies": replies,
                "score": score,
                "priority_flag": " ⭐ [ПРІОРИТЕТ АДМІНІСТРАТОРА]" if is_priority else ""
            })

        prepared.sort(key=lambda x: x["score"], reverse=True)

        result = []
        current_length = 0

        for item in prepared:
            block = (
                f"ID {item['idx']} {item['media_tag']}{item['priority_flag']} "
                f"[{item['channel_title']}] @{item['channel_username']}\n"
                f"Перегляди: {item['views']}\n"
                f"Пересилання: {item['forwards']}\n"
                f"Відповіді: {item['replies']}\n"
                f"{item['text']}"
            )

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

        prompt = f"""
Ти — старший редактор загальноукраїнського новинного Telegram-дайджесту.

ТВОЯ ЗАДАЧА:
Із потоку Telegram-повідомлень знайти події, які реально заслуговують
на місце серед головних новин останніх 4 годин.

Це НЕ звичайна стрічка новин.

Для кожної події запитай:
"Чи варто знати це людині, яка прочитає лише кілька головних новин?"

━━━━━━━━━━━━━━━━━━━━
АРХІВ ВЖЕ ОПУБЛІКОВАНИХ ПОДІЙ:
{history_block}
━━━━━━━━━━━━━━━━━━━━

ЕТАП 1 — ЗГРУПУЙ ПОСТИ У ПОДІЇ.

Одна реальна подія = один event_id.

Об'єднуй:
- повідомлення про одну атаку;
- перші дані та подальші уточнення;
- фото та відео тієї самої події;
- повідомлення різних каналів про один факт.

Не створюй новий event_id лише через інше формулювання.

━━━━━━━━━━━━━━━━━━━━
ЕТАП 2 — EDITORIAL GATE.

Для кожної події визнач eligible_for_digest=true або false.

eligible_for_digest=true став, якщо подія має хоча б один сильний фактор:

1. Великий масштаб.
2. Значні людські наслідки.
3. Важлива зміна на фронті.
4. Значне військове рішення або операція.
5. Важливе рішення української влади.
6. Важливе рішення США, ЄС, НАТО або великої держави.
7. Значний міжнародний вплив.
8. Значні економічні наслідки.
9. Удар по критичній або стратегічній інфраструктурі.
10. Великий суспільний резонанс із реальним значенням.
11. Унікальна або виняткова подія.
12. Суттєвий новий розвиток великої історії.

ЗАЗВИЧАЙ ВІДКИДАЙ:

- рутинні обстріли окремих районів;
- локальні пошкодження будинків без ширших наслідків;
- 1-2 поранених без інших значних факторів;
- тривоги;
- рух БпЛА;
- загрози ракет без результату;
- дрібні ДТП;
- локальні пожежі;
- дрібний кримінал;
- комунальні аварії;
- заяви політиків без реального рішення;
- повтори старих новин;
- чутки.

Для атак допускай подію, якщо:
- атака масована або комбінована;
- є значна кількість жертв;
- пошкоджена критична інфраструктура;
- є серйозні наслідки для великого міста;
- є військовий або політичний результат;
- подія має винятковий характер.

━━━━━━━━━━━━━━━━━━━━
ЕТАП 3 — ДЖЕРЕЛА.

best_factual_source_id:
найкраще джерело для підтвердження фактів.

best_media_source_id:
джерело з найкращим фото або відео з місця події.

Це можуть бути різні джерела.

Медіа НЕ робить слабку подію важливою.
Але якщо подія важлива, віддавай перевагу реальному фото або відео з місця.

━━━━━━━━━━━━━━━━━━━━
ОЦІНКИ 0-100:

importance
scale
reliability
public_interest
novelty
media_quality
national_relevance

event_type:

major_attack
battlefield_change
military_event
political_decision
international_decision
economic_event
critical_infrastructure
major_accident
major_crime
science_tech
social_event
culture_event
routine_attack
routine_statement
minor_local_event
minor_accident
alert_only
other

ДОДАТКОВО:

headline_hint — короткий заголовок.
key_facts — 3-5 найважливіших фактів.
why_it_matters — коротко, чому це важливо.
summary — стислий фактологічний опис.
rejection_reason — причина відхилення.

Якщо є ⭐ [ПРІОРИТЕТ АДМІНІСТРАТОРА]:
eligible_for_digest=true
importance=100
novelty=100

Не намагайся створити багато подій.
Краще 7 сильних, ніж 20 слабких.

ВІДПОВІДЬ ТІЛЬКИ JSON:

{{
  "events": [
    {{
      "event_id": "E1",
      "source_ids": [0, 2, 5],
      "best_factual_source_id": 0,
      "best_media_source_id": 5,
      "eligible_for_digest": true,
      "rejection_reason": "",
      "event_type": "major_attack",
      "category": "war",
      "importance": 94,
      "scale": 88,
      "reliability": 93,
      "public_interest": 91,
      "novelty": 82,
      "media_quality": 95,
      "national_relevance": 94,
      "headline_hint": "Масована атака на Одесу",
      "key_facts": ["Факт 1", "Факт 2", "Факт 3"],
      "why_it_matters": "Коротке пояснення.",
      "summary": "Фактологічний опис."
    }}
  ]
}}

TELEGRAM POSTS:
{posts_context}
"""

        data = self._call_json_with_cascade(prompt, max_retries, "ANALYZER")
        return data.get("events", []) if data and isinstance(data.get("events"), list) else []

    def _rank_events(self, events: List[Dict[str, Any]], posts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        ranked = []

        for ev in events:
            try:
                src_ids = [
                    s for s in ev.get("source_ids", [])
                    if isinstance(s, int) and 0 <= s < len(posts)
                ]

                if not src_ids:
                    continue

                is_priority = any(posts[s].get("is_priority") for s in src_ids)
                eligible = bool(ev.get("eligible_for_digest", False))
                event_type = str(ev.get("event_type") or "other")

                if not is_priority:
                    if not eligible:
                        continue
                    if event_type in LOW_VALUE_EVENT_TYPES:
                        continue

                imp = self._safe_score(ev.get("importance"))
                scale = self._safe_score(ev.get("scale"))
                rel = self._safe_score(ev.get("reliability"))
                pub = self._safe_score(ev.get("public_interest"))
                nov = self._safe_score(ev.get("novelty"))
                med = self._safe_score(ev.get("media_quality"))
                national = self._safe_score(ev.get("national_relevance"))

                category = ev.get("category", "other")
                if category not in self.ALLOWED_CATEGORIES:
                    category = "other"

                tier_mult = self._average_source_multiplier(src_ids, posts)

                base_score = (
                    imp * 0.30
                    + scale * 0.15
                    + rel * 0.18
                    + pub * 0.10
                    + nov * 0.10
                    + national * 0.15
                    + med * 0.02
                )

                score = base_score * tier_mult
                score += min(len(src_ids) * 1.5, 6)

                if rel < 45:
                    score -= 20
                elif rel < 60:
                    score -= 8

                if national < 40:
                    score -= 12

                if is_priority:
                    score += 500

                factual_source = self._select_factual_source(
                    src_ids, posts, ev.get("best_factual_source_id")
                )
                media_source = self._select_media_source(
                    src_ids, posts, ev.get("best_media_source_id")
                )

                publishing_source = media_source if media_source is not None else factual_source

                ev_copy = dict(ev)
                ev_copy.update({
                    "source_ids": src_ids,
                    "best_factual_source_id": factual_source,
                    "best_media_source_id": media_source,
                    "best_source_id": publishing_source,
                    "is_priority": is_priority,
                    "eligible_for_digest": True,
                    "event_type": event_type,
                    "category": category,
                    "raw_score": round(score, 2)
                })

                ranked.append(ev_copy)

            except Exception as e:
                logger.warning(f"Помилка ranking події: {e}")

        ranked.sort(key=lambda x: x.get("raw_score", 0), reverse=True)

        category_counts = {}

        for ev in ranked:
            if ev.get("is_priority"):
                ev["balanced_score"] = ev["raw_score"]
                continue

            category = ev["category"]
            current = category_counts.get(category, 0)

            penalty = 10 if current >= 4 else (5 if current >= 3 else 0)
            ev["balanced_score"] = round(ev["raw_score"] - penalty, 2)

            category_counts[category] = current + 1

        ranked.sort(key=lambda x: x.get("balanced_score", 0), reverse=True)
        return ranked

    def _generate_final_digest(
        self,
        events: List[Dict[str, Any]],
        posts: List[Dict[str, Any]],
        past_events: Optional[Union[List[str], List[Dict[str, str]]]],
        max_count: int,
        max_retries: int
    ) -> List[Dict[str, Any]]:
        event_blocks = []

        for ev in events:
            factual_id = ev["best_factual_source_id"]
            media_id = ev.get("best_media_source_id")

            factual_post = posts[factual_id]
            media_description = "немає"

            if isinstance(media_id, int):
                if posts[media_id].get("has_video"):
                    media_description = "реальне відео з події"
                elif posts[media_id].get("has_media"):
                    media_description = "реальне фото з події"

            key_facts = ev.get("key_facts", [])
            key_facts_text = "; ".join(str(x) for x in key_facts[:5]) if isinstance(key_facts, list) else str(key_facts)

            priority_flag = " ⭐ ПРІОРИТЕТ АДМІНІСТРАТОРА" if ev.get("is_priority") else ""

            event_blocks.append(
                f"=== EVENT_ID: {ev.get('event_id')}{priority_flag} ===\n"
                f"ТИП: {ev.get('event_type', 'other')}\n"
                f"КАТЕГОРІЯ: {ev.get('category', 'other')}\n"
                f"МЕДІА: {media_description}\n"
                f"СУТЬ: {ev.get('summary', '')}\n"
                f"ЧОМУ ВАЖЛИВО: {ev.get('why_it_matters', '')}\n"
                f"КЛЮЧОВІ ФАКТИ: {key_facts_text}\n"
                f"ТЕКСТ ДЖЕРЕЛА: {str(factual_post.get('text') or '')[:self.MAX_EVENT_SOURCE_CHARS]}\n"
            )

        history_block = self._build_history_block(past_events)

        prompt = f"""
Ти — головний редактор новинного Telegram-каналу.

Сформуй фінальний дайджест із найважливіших подій.
Максимум: {max_count} новин.

Не потрібно обов'язково набирати {max_count}.
Якщо сильних подій менше — поверни менше.

Події ⭐ ПРІОРИТЕТ АДМІНІСТРАТОРА обов'язково включи першими.

━━━━━━━━━━━━━━━━━━━━
АРХІВ:
{history_block}
━━━━━━━━━━━━━━━━━━━━

ВИМОГИ ДО ВИБОРУ:

1. Не повторюй одну реальну подію двічі.
2. Не додавай слабку подію тільки для заповнення кількості.
3. Не вигадуй факти.
4. Не використовуй чутки.
5. Відео або фото не повинно робити слабку подію важливішою.
6. Якщо подія важлива і має реальне відео або фото — це плюс.

━━━━━━━━━━━━━━━━━━━━
ВИМОГИ ДО ТЕКСТУ:

Максимальна довжина однієї новини — {self.MAX_NEWS_CHARS} символів.

Кожна новина повинна містити НЕ МЕНШЕ 4 коротких речень.
Оптимально 4-5 речень.

Речення мають бути короткими, фактологічними та насиченими інформацією.

СТРУКТУРА:

Речення 1:
що сталося і де.

Речення 2:
головний наслідок або результат.

Речення 3:
ключові цифри, масштаби або деталі.

Речення 4:
важливий контекст або реакція сторін.

Речення 5:
чому ця подія важлива або що вона змінює.

Не розтягуй текст водою.
Кожне речення повинно додавати новий факт.

Заголовок:
- 4-9 слів;
- без клікбейту;
- максимально конкретний;
- не повторює перше речення.

Формат:

ОДИН тематичний емодзі + <b>Заголовок</b>

порожній рядок

4-5 коротких речень.

НЕ ВИКОРИСТОВУЙ:
"Стало відомо..."
"Повідомляється, що..."
"Наразі відомо..."
"Як зазначають..."
"За інформацією джерел..."
"Ситуація залишається..."

Не вставляй технічні маркери:
[ФОТО]
[ВІДЕО]
[ТЕКСТ]

ВІДПОВІДЬ ТІЛЬКИ JSON:

{{
  "news": [
    {{
      "event_id": "E1",
      "text": "💥 <b>Короткий заголовок</b>\\n\\nПерше речення. Друге речення. Третє речення. Четверте речення."
    }}
  ]
}}

КАНДИДАТИ:
{chr(10).join(event_blocks)}
"""

        data = self._call_json_with_cascade(prompt, max_retries, "EDITOR")
        raw_news = data.get("news", []) if data and isinstance(data.get("news"), list) else []

        event_map = {ev["event_id"]: ev for ev in events}
        final_list = []

        for item in raw_news:
            if not isinstance(item, dict):
                continue

            event_id = item.get("event_id")
            text = item.get("text")

            if event_id not in event_map or not isinstance(text, str) or not text.strip():
                continue

            ev = event_map[event_id]

            final_list.append({
                "event_id": event_id,
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
        validated = []
        used_event_ids = set()

        valid_event_ids = {
            str(ev.get("event_id"))
            for ev in ranked_events
            if ev.get("event_id")
        }

        for item in news:
            source_id = item.get("source_id")
            event_id = str(item.get("event_id") or "")
            text = item.get("text")

            if event_id not in valid_event_ids:
                continue

            if not isinstance(source_id, int) or not 0 <= source_id < len(posts):
                continue

            if event_id in used_event_ids:
                continue

            if not isinstance(text, str) or not text.strip():
                continue

            text = self._clean_generated_news_text(text)
            if not text:
                continue

            validated.append({
                "source_id": source_id,
                "text": text,
                "summary": item.get("summary", ""),
                "category": item.get("category", "other")
            })

            used_event_ids.add(event_id)

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
        used_source_ids = {item["source_id"] for item in result if isinstance(item.get("source_id"), int)}

        emoji_map = {
            "war": "💥",
            "politics": "🏛",
            "economy": "💰",
            "international": "🌍",
            "society": "🇺🇦",
            "technology": "⚡",
            "science": "🔬",
            "culture": "🎭",
            "other": "📰",
        }

        for ev in ranked_events:
            if len(result) >= count:
                break

            source_id = ev.get("best_source_id")

            if not isinstance(source_id, int) or source_id in used_source_ids:
                continue

            headline = (ev.get("headline_hint") or "Важлива подія").strip()
            category = ev.get("category", "other")
            emoji = emoji_map.get(category, "📰")

            key_facts = ev.get("key_facts", [])
            facts = [str(x).strip() for x in key_facts if str(x).strip()] if isinstance(key_facts, list) else []

            summary = (ev.get("summary") or "").strip()
            why = (ev.get("why_it_matters") or "").strip()

            sentences = []
            if summary:
                sentences.append(summary.rstrip(".") + ".")
            sentences.extend(fact.rstrip(".") + "." for fact in facts[:4])

            if why:
                sentences.append(why.rstrip(".") + ".")

            while len(sentences) < 5 and summary:
                sentences.append(summary.rstrip(".") + ".")

            text = f"{emoji} <b>{headline}</b>\n\n{' '.join(sentences[:6])}"
            text = self._clean_generated_news_text(text)

            result.append({
                "source_id": source_id,
                "text": text,
                "summary": summary,
                "category": category
            })

            used_source_ids.add(source_id)

        return result[:count]

    def _select_factual_source(
        self,
        source_ids: List[int],
        posts: List[Dict[str, Any]],
        preferred_id: Any = None
    ) -> int:
        if isinstance(preferred_id, int) and preferred_id in source_ids:
            return preferred_id

        return max(source_ids, key=lambda s: self._factual_source_score(posts[s]))

    def _select_media_source(
        self,
        source_ids: List[int],
        posts: List[Dict[str, Any]],
        preferred_id: Any = None
    ) -> Optional[int]:
        media_ids = [
            s for s in source_ids
            if posts[s].get("has_video") or posts[s].get("has_media")
        ]

        if not media_ids:
            return None

        if isinstance(preferred_id, int) and preferred_id in media_ids:
            return preferred_id

        return max(media_ids, key=lambda s: self._media_source_score(posts[s]))

    def _factual_source_score(self, post: Dict[str, Any]) -> float:
        if post.get("is_priority"):
            return 10000

        username = str(post.get("channel_username") or "").replace("@", "").strip()
        views = int(post.get("views") or 0)
        forwards = int(post.get("forwards") or 0)
        text_length = len(post.get("text") or "")

        score = (
            min(math.log10(max(views, 1)) * 5, 25)
            + min(math.log10(max(forwards, 1)) * 3, 10)
            + min(text_length / 150, 10)
        )

        return score * self._get_source_multiplier(username)

    @staticmethod
    def _media_source_score(post: Dict[str, Any]) -> float:
        score = 40 if post.get("has_video") else (20 if post.get("has_media") else 0)

        views = int(post.get("views") or 0)
        forwards = int(post.get("forwards") or 0)

        score += min(math.log10(max(views, 1)) * 3, 18)
        score += min(math.log10(max(forwards, 1)) * 2, 8)

        return score

    def _average_source_multiplier(
        self,
        source_ids: List[int],
        posts: List[Dict[str, Any]]
    ) -> float:
        multipliers = []

        for source_id in source_ids:
            username = str(posts[source_id].get("channel_username") or "").replace("@", "").strip()
            multipliers.append(self._get_source_multiplier(username))

        return sum(multipliers) / len(multipliers) if multipliers else 1.0

    @staticmethod
    def _get_source_multiplier(username: str) -> float:
        if username in SOURCE_TIERS:
            return SOURCE_TIERS[username]

        username_lower = username.lower()

        for source, multiplier in SOURCE_TIERS.items():
            if source.lower() == username_lower:
                return multiplier

        return 1.0

    def _call_json_with_cascade(
        self,
        prompt: str,
        max_retries: int,
        op_name: str
    ) -> Optional[Dict[str, Any]]:
        for model in self.models_priority:
            for attempt in range(1, max_retries + 1):
                try:
                    logger.info(f"{op_name}: спроба {attempt}/{max_retries} через {model}")

                    response = self.client.models.generate_content(
                        model=model,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            temperature=0.15
                        ),
                    )

                    raw_text = self._clean_json_response((response.text or "").strip())
                    data = json.loads(raw_text)

                    if isinstance(data, dict):
                        return data

                except Exception as e:
                    err = str(e)

                    if any(x in err for x in ["503", "429", "UNAVAILABLE", "ResourceExhausted", "NOT_FOUND"]):
                        if attempt < max_retries:
                            time.sleep(3 * attempt)
                            continue
                        break

                    logger.error(f"Помилка {op_name} ({model}): {e}")
                    break

        return None

    def _build_history_block(
        self,
        past_events: Optional[Union[List[str], List[Dict[str, str]]]]
    ) -> str:
        if not past_events:
            return "Історія опублікованих подій порожня."

        lines = []

        for item in past_events[:self.HISTORY_LIMIT]:
            if isinstance(item, dict):
                title = (item.get("title") or "").strip()
                summary = (item.get("summary") or "").strip()
                published_at = (item.get("published_at") or "").strip()

                if title or summary:
                    time_info = f" [{published_at}]" if published_at else ""
                    desc = f" — {summary}" if summary else ""
                    lines.append(f"- {title}{desc}{time_info}")

            elif isinstance(item, str) and item.strip():
                lines.append(f"- {item.strip()}")

        return "\n".join(lines) if lines else "Історія опублікованих подій порожня."

    def _clean_generated_news_text(self, text: str) -> str:
        text = text.strip()

        text = re.sub(
            r'\[(?:ФОТО|ВІДЕО|ТЕКСТ|PHOTO|VIDEO|TEXT)\]\s*',
            '',
            text,
            flags=re.IGNORECASE
        )

        text = re.sub(
            r'(?:ФОТО|ВІДЕО|ТЕКСТ):\s*',
            '',
            text,
            flags=re.IGNORECASE
        )

        text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
        text = text.replace("📍", "").replace("📌", "")

        if "<b>" not in text or "</b>" not in text:
            lines = text.split("\n", 1)
            first_line = lines[0].strip()
            rest = "\n" + lines[1] if len(lines) > 1 else ""
            text = f"<b>{first_line}</b>{rest}"

        if len(text) > self.MAX_NEWS_CHARS:
            text = text[:self.MAX_NEWS_CHARS]
            last_space = text.rfind(" ")

            if last_space > 350:
                text = text[:last_space].rstrip()

            text += "…"

        if "<b>" in text and "</b>" not in text:
            text += "</b>"

        return text.strip()

    @staticmethod
    def _safe_score(value: Any) -> float:
        try:
            return max(0.0, min(100.0, float(value)))
        except (TypeError, ValueError):
            return 0.0

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
