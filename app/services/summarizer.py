import json
import logging
import math
import re
import time
from difflib import SequenceMatcher
from datetime import datetime, timezone
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
    "forbesukraines": 1.2,

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
}

HARD_REJECT_EVENT_TYPES = {
    "alert_only",
}


class NewsSummarizer:
    DEFAULT_COUNT = 10

    # Тепер намагаємось давати щільніший випуск: якщо є достатньо
    # придатних подій, бажано мати щонайменше 7 матеріалів.
    MIN_DIGEST_COUNT = 7

    EDITOR_CANDIDATES = 30
    HISTORY_LIMIT = 150

    MAX_INPUT_CHARS = 55000
    PRIORITY_RECOVERY_MAX_CHARS = 30000

    # Окремий контекст для пошуку "цікавинок". Він коротший на один пост,
    # зате навмисно більш різноманітний за джерелами і темами, щоб великі
    # воєнно-політичні пости не витісняли технології, науку, бізнес і
    # практично корисні зміни ще ДО Analyzer.
    DISCOVERY_RECOVERY_MAX_CHARS = 50000
    DISCOVERY_MAX_POST_CHARS = 750
    MAX_DISCOVERY_PER_DIGEST = 3
    DISCOVERY_RECOVERY_CANDIDATES = 8

    MAX_EVENT_SOURCE_CHARS = 3000

    # Дозволяємо трохи більше контексту й 3-6 повних речень.
    MAX_NEWS_CHARS = 900

    ALLOWED_CATEGORIES = {
        "war",
        "politics",
        "economy",
        "international",
        "society",
        "technology",
        "science",
        "culture",
        "other",
    }

    def __init__(self):
        self.client = genai.Client(
            api_key=settings.GEMINI_API_KEY
        )
        self.models_priority = [
            "gemini-3.5-flash-lite",
            "gemini-3.1-flash-lite",
        ]

    def select_top_distinct_news(
        self,
        posts: List[Dict[str, Any]],
        past_events: Optional[
            Union[
                List[str],
                List[Dict[str, str]],
            ]
        ] = None,
        count: int = DEFAULT_COUNT,
        max_retries_per_model: int = 2,
    ) -> List[Dict[str, Any]]:
        if not posts:
            return []

        logger.info(
            "Формування дайджесту: "
            f"{len(posts)} постів → максимум {count} новин"
        )

        priority_post_ids = self._get_priority_post_ids(posts)
        if priority_post_ids:
            logger.info(
                "У поточному пулі %s ручних пріоритетних постів.",
                len(priority_post_ids),
            )

        posts_context = self._build_posts_context(posts)
        if not posts_context:
            return []

        analyzed_events = self._analyze_events(
            posts_context,
            past_events,
            max_retries_per_model,
        )

        if not analyzed_events:
            logger.warning(
                "Analyzer не повернув подій. "
                "Перевіряємо ручні пріоритетні пости окремо."
            )
            analyzed_events = []
        else:
            logger.info(
                "Analyzer знайшов "
                f"{len(analyzed_events)} потенційних подій."
            )

        # ЖОРСТКА ГАРАНТІЯ MANUAL:
        # ручна новина не має права зникнути між Analyzer і ranking.
        analyzed_events = self._ensure_priority_events(
            analyzed_events,
            posts,
            past_events,
            max_retries_per_model,
        )

        if not analyzed_events:
            logger.warning(
                "Після Analyzer і priority-recovery немає подій."
            )
            return []

        ranked_events = self._rank_events(
            analyzed_events,
            posts,
        )

        if not ranked_events:
            logger.warning(
                "Після editorial gate не залишилось подій."
            )
            return []

        # ОКРЕМИЙ DISCOVERY-PASS.
        # Основний Analyzer може чудово знайти важкі новини, але загубити
        # науку/технології/бізнес/корисні зміни в великому контексті.
        # Тому спочатку дивимось, скільки є справжніх core-подій і скільки
        # слотів за редакційним правилом лишається під "цікавинки".
        core_count = len(
            [
                ev for ev in ranked_events
                if self._event_digest_role(ev) == "core"
            ]
        )
        desired_discovery_slots = self._desired_discovery_slots(
            core_count,
            count,
        )
        available_discovery = len(
            [
                ev for ev in ranked_events
                if self._is_publishable_discovery(ev)
            ]
        )

        if (
            desired_discovery_slots > 0
            and available_discovery < desired_discovery_slots
        ):
            logger.info(
                "Discovery-check: core=%s, discovery=%s, "
                "можливих слотів=%s. Запускаємо окремий пошук цікавинок.",
                core_count,
                available_discovery,
                desired_discovery_slots,
            )

            analyzed_events = self._ensure_discovery_events(
                analyzed_events,
                posts,
                past_events,
                max_retries_per_model,
                desired_count=desired_discovery_slots,
            )

            # Після discovery-recovery обов'язково ранжуємо весь пул заново:
            # нові кандидати мають пройти ТОЙ САМИЙ quality gate, history gate
            # і source/media selection, що й звичайні події.
            ranked_events = self._rank_events(
                analyzed_events,
                posts,
            )

            if not ranked_events:
                logger.warning(
                    "Після discovery-recovery ranking не залишив подій."
                )
                return []

        priority_ranked = [
            ev
            for ev in ranked_events
            if ev.get("is_priority")
        ]

        # Manual завжди важливіший за стандартний ліміт. Якщо адміністратор
        # навмисно додасть >10 РІЗНИХ подій, вони не обріжуться мовчки.
        effective_count = max(count, len(priority_ranked))
        if effective_count > count:
            logger.warning(
                "Ручних пріоритетних подій (%s) більше, ніж ліміт "
                "дайджесту (%s). Тимчасово розширюємо випуск до %s.",
                len(priority_ranked),
                count,
                effective_count,
            )

        core_ranked = [
            ev
            for ev in ranked_events
            if self._event_digest_role(ev) == "core"
        ]
        discovery_ranked = [
            ev
            for ev in ranked_events
            if self._is_publishable_discovery(ev)
        ]

        logger.info(
            "Після ranking залишилось %s подій: core=%s, "
            "discovery=%s, priority=%s.",
            len(ranked_events),
            len(core_ranked),
            len(discovery_ranked),
            len(priority_ranked),
        )

        for idx, event in enumerate(
            ranked_events[:20],
            start=1,
        ):
            logger.info(
                "RANK #%s: %.2f | %s | role=%s | priority=%s | "
                "cur=%.0f | practical=%.0f | %s",
                idx,
                float(
                    event.get(
                        "balanced_score",
                        event.get("raw_score", 0),
                    )
                    or 0
                ),
                event.get("event_type", "other"),
                self._event_digest_role(event),
                bool(event.get("is_priority")),
                float(event.get("curiosity", 0) or 0),
                float(event.get("practical_value", 0) or 0),
                event.get(
                    "headline_hint",
                    event.get("summary", ""),
                ),
            )

        # Editor як і раніше бачить широкий TOP-кандидатів і може написати
        # природні тексти. Але додатково гарантовано підсовуємо йому:
        # 1) усі manual; 2) найкращі discovery, навіть якщо їхній загальний
        # score нижчий за TOP-30 через нижчу стратегічну важливість.
        editor_events = ranked_events[
            :self.EDITOR_CANDIDATES
        ]
        editor_event_ids = {
            str(ev.get("event_id") or "")
            for ev in editor_events
        }

        must_show_to_editor = list(priority_ranked)
        must_show_to_editor.extend(
            sorted(
                discovery_ranked,
                key=self._discovery_sort_score,
                reverse=True,
            )[: self.MAX_DISCOVERY_PER_DIGEST * 2]
        )

        for ev in must_show_to_editor:
            event_id = str(ev.get("event_id") or "")
            if event_id and event_id not in editor_event_ids:
                editor_events.append(ev)
                editor_event_ids.add(event_id)

        final_news = self._generate_final_digest(
            editor_events,
            posts,
            past_events,
            effective_count,
            max_retries_per_model,
        )

        validated = self._validate_final_news(
            final_news,
            ranked_events,
            posts,
            effective_count,
        )

        # Manual лишається абсолютною гарантією навіть якщо Editor
        # проігнорував конкретний event_id у своєму JSON.
        validated = self._ensure_priority_news_in_final(
            validated,
            ranked_events,
            posts,
            effective_count,
        )

        target_min = min(
            effective_count,
            self.MIN_DIGEST_COUNT,
            len(ranked_events),
        )

        if len(validated) < target_min:
            logger.warning(
                "EDITOR сформував лише "
                f"{len(validated)} новин. "
                f"Fallback до {target_min}."
            )

            validated = self._fill_missing_news(
                validated,
                ranked_events,
                posts,
                target_min,
            )

        validated = self._ensure_priority_news_in_final(
            validated,
            ranked_events,
            posts,
            effective_count,
        )

        # ФІНАЛЬНА ДЕТЕРМІНОВАНА РЕДАКЦІЙНА СТРУКТУРА:
        # - 10 core => 10 core, 0 discovery;
        # - 9 core  => 9 core + 1 discovery;
        # - 8 core  => 8 core + до 2 discovery;
        # - 7 core  => 7 core + до 3 discovery;
        # - 5-6 core => вони + до 3 discovery.
        # Discovery завжди ставимо В КІНЕЦЬ випуску.
        # Manual при конфлікті має вищий пріоритет за цю квоту.
        validated = self._enforce_digest_mix(
            validated,
            ranked_events,
            posts,
            effective_count,
        )

        logger.info(
            "Фінальний дайджест: "
            f"{len(validated)} новин."
        )

        return validated[:effective_count]

    def _build_posts_context(
        self,
        posts: List[Dict[str, Any]],
        only_ids: Optional[List[int]] = None,
        max_chars: Optional[int] = None,
    ) -> str:
        prepared = []
        now_utc = datetime.now(timezone.utc)

        allowed_ids = (
            set(only_ids)
            if only_ids is not None
            else None
        )
        char_limit = (
            int(max_chars)
            if isinstance(max_chars, int) and max_chars > 0
            else self.MAX_INPUT_CHARS
        )

        for idx, post in enumerate(posts):
            if allowed_ids is not None and idx not in allowed_ids:
                continue

            text = (post.get("text") or "").strip()
            if not text:
                continue

            media_tag = (
                "[ВІДЕО]"
                if post.get("has_video")
                else (
                    "[ФОТО]"
                    if post.get("has_media")
                    else "[ТЕКСТ]"
                )
            )

            channel_title = (
                post.get("channel_title")
                or post.get("channel_username")
                or "Джерело"
            )

            channel_username = (
                str(post.get("channel_username", "") or "")
                .replace("@", "")
                .strip()
            )

            views = int(post.get("views") or 0)
            forwards = int(post.get("forwards") or 0)
            replies = int(post.get("replies") or 0)
            is_priority = bool(post.get("is_priority"))

            post_date = post.get("date")
            age_minutes: Optional[float] = None
            published_at = "невідомо"

            if isinstance(post_date, datetime):
                if post_date.tzinfo is None:
                    post_date = post_date.replace(
                        tzinfo=timezone.utc
                    )

                post_date_utc = post_date.astimezone(
                    timezone.utc
                )

                published_at = post_date_utc.strftime(
                    "%Y-%m-%d %H:%M UTC"
                )

                age_minutes = max(
                    0.0,
                    (
                        now_utc - post_date_utc
                    ).total_seconds()
                    / 60.0,
                )

            tier_mult = self._get_source_multiplier(
                channel_username
            )

            engagement_score = (
                min(
                    math.log10(max(views, 1)) * 4,
                    26,
                )
                + min(
                    math.log10(max(forwards, 1)) * 3,
                    12,
                )
                + min(
                    math.log10(max(replies, 1)) * 2,
                    8,
                )
            )

            media_bonus = (
                9
                if post.get("has_video")
                else (
                    4.5
                    if post.get("has_media")
                    else 0
                )
            )

            # Freshness потрібен лише для доступу нового поста до Analyzer.
            # Він не є автоматичним доказом важливості.
            freshness_bonus = 0.0
            if age_minutes is not None:
                freshness_bonus = max(
                    0.0,
                    12.0
                    * (
                        1.0
                        - min(age_minutes, 240.0)
                        / 240.0
                    ),
                )

            # Manual завжди стоїть на початку контексту і гарантовано
            # поміщається у ліміт символів раніше за звичайні пости.
            score = (
                10000.0
                if is_priority
                else (
                    engagement_score
                    + media_bonus
                    + freshness_bonus
                )
                * tier_mult
            )

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
                "published_at": published_at,
                "age_minutes": age_minutes,
                "priority_flag": (
                    " ⭐ [ПРІОРИТЕТ АДМІНІСТРАТОРА]"
                    if is_priority
                    else ""
                ),
            })

        prepared.sort(
            key=lambda x: x["score"],
            reverse=True,
        )

        result = []
        current_length = 0

        for item in prepared:
            age_text = (
                f"{item['age_minutes']:.0f} хв тому"
                if isinstance(
                    item.get("age_minutes"),
                    (int, float),
                )
                else "невідомо"
            )

            block = (
                f"ID {item['idx']} "
                f"{item['media_tag']}"
                f"{item['priority_flag']} "
                f"[{item['channel_title']}] "
                f"@{item['channel_username']}\n"
                f"Час: {item['published_at']} "
                f"({age_text})\n"
                f"Перегляди: {item['views']}\n"
                f"Пересилання: {item['forwards']}\n"
                f"Відповіді: {item['replies']}\n"
                f"{item['text']}"
            )

            if current_length + len(block) > char_limit:
                continue

            result.append(block)
            current_length += len(block) + 10

        logger.info(
            "У контекст Analyzer потрапило "
            f"{len(result)} з {len(prepared)} постів "
            f"({current_length} символів)."
        )

        return "\n\n---\n\n".join(result)

    def _build_discovery_context(
        self,
        posts: List[Dict[str, Any]],
    ) -> str:
        """
        Будує окремий контекст для discovery-pass.

        На відміну від основного контексту, тут ми:
        - сильніше цінуємо тематичну новизну, а не лише великі перегляди;
        - обмежуємо домінування одного каналу;
        - даємо бонус науці, технологіям, бізнесу, сервісам, правилам,
          виробництву, досягненням і незвичайним суспільним фактам;
        - трохи знижуємо суто рутинний hard-news шум.

        Це НЕ вибір новин у фінал. Контекст лише підвищує recall, а кожен
        знайдений event потім проходить звичайний ranking/history gate.
        """
        now_utc = datetime.now(timezone.utc)
        prepared: List[Dict[str, Any]] = []

        discovery_keywords = [
            "вчен", "дослід", "наук", "відкрит", "винахід",
            "технолог", "штучн", "інтелект", "нейромереж", "gpt",
            "стартап", "робот", "чип", "процесор", "космос",
            "медицин", "лікуван", "біотех", "наномат", "матеріал",
            "виробництв", "завод", "серійне", "запуст", "контракт",
            "компан", "бізнес", "ринок", "авто", "автомоб",
            "рекорд", "досягнен", "перший у світі", "вперше",
            "правил", "тариф", "сервіс", "послуг", "застосунок",
            "транспорт", "метро", "поїзд", "аеропорт", "обмеженн",
            "освіта", "університет", "культур", "фільм", "музей",
            "археолог", "історичн", "еколог", "енергі", "сонячн",
        ]

        hard_news_keywords = [
            "повітряна тривога", "рух бпла", "загроза баліст",
            "обстріл", "масована атака", "фронт", "штурм",
            "загинув", "поранен", "влучання ракети", "бойові дії",
        ]

        for idx, post in enumerate(posts):
            text = (post.get("text") or "").strip()
            if not text:
                continue

            text_lower = text.lower()
            username = (
                str(post.get("channel_username", "") or "")
                .replace("@", "")
                .strip()
            )
            title = (
                post.get("channel_title")
                or post.get("channel_username")
                or "Джерело"
            )

            views = int(post.get("views") or 0)
            forwards = int(post.get("forwards") or 0)

            post_date = post.get("date")
            age_minutes: Optional[float] = None
            published_at = "невідомо"
            if isinstance(post_date, datetime):
                if post_date.tzinfo is None:
                    post_date = post_date.replace(tzinfo=timezone.utc)
                post_date_utc = post_date.astimezone(timezone.utc)
                published_at = post_date_utc.strftime(
                    "%Y-%m-%d %H:%M UTC"
                )
                age_minutes = max(
                    0.0,
                    (now_utc - post_date_utc).total_seconds() / 60.0,
                )

            keyword_hits = sum(
                1 for keyword in discovery_keywords
                if keyword in text_lower
            )
            hard_hits = sum(
                1 for keyword in hard_news_keywords
                if keyword in text_lower
            )

            freshness = 0.0
            if age_minutes is not None:
                freshness = max(
                    0.0,
                    8.0 * (1.0 - min(age_minutes, 240.0) / 240.0),
                )

            engagement = (
                min(math.log10(max(views, 1)) * 2.2, 12)
                + min(math.log10(max(forwards, 1)) * 1.5, 5)
            )

            media_bonus = (
                5.0
                if post.get("has_video")
                else (2.5 if post.get("has_media") else 0.0)
            )

            # Keyword bonus домінує лише в recovery-pass. Сам ranking далі
            # оцінить фактичну інформаційну цінність події.
            score = (
                keyword_hits * 10.0
                + engagement
                + freshness
                + media_bonus
            ) * self._get_source_multiplier(username)

            if hard_hits and keyword_hits == 0:
                score -= min(hard_hits * 6.0, 18.0)

            # Manual не треба "шукати" вдруге, але якщо він тематично
            # discovery — залишаємо шанс моделі правильно класифікувати його.
            if post.get("is_priority"):
                score += 4.0

            media_tag = (
                "[ВІДЕО]"
                if post.get("has_video")
                else ("[ФОТО]" if post.get("has_media") else "[ТЕКСТ]")
            )

            excerpt = re.sub(r"\s+", " ", text).strip()
            excerpt = self._truncate_plain_text(
                excerpt,
                self.DISCOVERY_MAX_POST_CHARS,
            )

            prepared.append({
                "idx": idx,
                "score": score,
                "channel_key": username.lower() or str(title).lower(),
                "title": title,
                "username": username,
                "media_tag": media_tag,
                "published_at": published_at,
                "excerpt": excerpt,
            })

        prepared.sort(key=lambda item: item["score"], reverse=True)

        # Перший прохід: не більше 7 постів одного джерела. Це різко зменшує
        # шанс, що один великий канал заб'є весь discovery-контекст однією темою.
        selected: List[Dict[str, Any]] = []
        per_channel: Dict[str, int] = {}
        for item in prepared:
            key = item["channel_key"]
            if per_channel.get(key, 0) >= 7:
                continue
            selected.append(item)
            per_channel[key] = per_channel.get(key, 0) + 1

        result: List[str] = []
        current_length = 0

        for item in selected:
            block = (
                f"ID {item['idx']} {item['media_tag']} "
                f"[{item['title']}] @{item['username']}\n"
                f"Час: {item['published_at']}\n"
                f"{item['excerpt']}"
            )

            if (
                current_length + len(block)
                > self.DISCOVERY_RECOVERY_MAX_CHARS
            ):
                continue

            result.append(block)
            current_length += len(block) + 10

        logger.info(
            "У discovery-контекст потрапило %s з %s постів (%s символів).",
            len(result),
            len(prepared),
            current_length,
        )

        return "\n\n---\n\n".join(result)

    def _desired_discovery_slots(
        self,
        core_count: int,
        max_count: int,
    ) -> int:
        """
        Редакційне правило користувача:
        10 core -> 0 discovery
         9 core -> 1 discovery
         8 core -> до 2 discovery
         7 core -> до 3 discovery
        <=6 core -> до 3 discovery

        Формула проста: заповнюємо вільні місця, але не більше трьох.
        """
        if max_count <= 0 or core_count >= max_count:
            return 0

        free_slots = max_count - max(0, core_count)
        return min(self.MAX_DISCOVERY_PER_DIGEST, free_slots)

    def _event_digest_role(
        self,
        ev: Dict[str, Any],
    ) -> str:
        role = str(ev.get("digest_role") or "").strip().lower()
        if role in {"core", "discovery"}:
            return role
        return "discovery" if ev.get("is_discovery_candidate") else "core"

    def _is_publishable_discovery(
        self,
        ev: Dict[str, Any],
    ) -> bool:
        if self._event_digest_role(ev) != "discovery":
            return False

        # Manual ніколи не відкидаємо через score-based discovery quality.
        if ev.get("is_priority"):
            return True

        return bool(ev.get("discovery_qualified", False))

    def _resolve_digest_role(
        self,
        ev: Dict[str, Any],
        event_type: str,
        category: str,
        importance: float,
        national_relevance: float,
        urgency: float,
        curiosity: float,
        practical_value: float,
        novelty: float,
        public_interest: float,
    ) -> str:
        """
        Нормалізує core/discovery. LLM має перше слово, але великі hard-news
        події захищаємо від випадкової класифікації як "цікавинка".
        """
        explicit = str(ev.get("digest_role") or "").strip().lower()

        always_core_types = {
            "major_attack",
            "battlefield_change",
            "critical_infrastructure",
            "major_accident",
            "major_crime",
            "political_decision",
        }
        strategic_core_types = {
            "military_event",
            "international_decision",
            "economic_event",
            "social_event",
        }

        if (
            event_type in always_core_types
            and (
                importance >= 72
                or national_relevance >= 70
                or urgency >= 78
            )
        ):
            return "core"

        if (
            event_type in strategic_core_types
            and (
                importance >= 84
                or national_relevance >= 84
                or urgency >= 90
            )
        ):
            return "core"

        if explicit in {"core", "discovery"}:
            return explicit

        if bool(ev.get("is_discovery_candidate")):
            return "discovery"

        if category in {"technology", "science", "culture"}:
            if (
                curiosity >= 65
                or practical_value >= 70
                or novelty >= 70
            ):
                return "discovery"

        if event_type in {"science_tech", "culture_event"}:
            if curiosity >= 65 or novelty >= 70:
                return "discovery"

        # Суспільні/економічні/міжнародні цікаві факти можуть бути
        # discovery, якщо їхня головна сила — новизна/користь, а не кризовість.
        if (
            importance < 82
            and urgency < 86
            and (
                (curiosity >= 80 and novelty >= 62)
                or (practical_value >= 82 and public_interest >= 58)
            )
        ):
            return "discovery"

        return "core"

    @staticmethod
    def _discovery_quality(
        reliability: float,
        novelty: float,
        curiosity: float,
        practical_value: float,
        public_interest: float,
        category: str,
    ) -> bool:
        if reliability < 50 or novelty < 50:
            return False

        if curiosity >= 76 and public_interest >= 50:
            return True

        if practical_value >= 80 and public_interest >= 55:
            return True

        if (
            category in {"technology", "science", "culture"}
            and curiosity >= 66
            and novelty >= 60
        ):
            return True

        if curiosity >= 70 and novelty >= 72:
            return True

        return False

    @staticmethod
    def _calculate_discovery_score(
        curiosity: float,
        novelty: float,
        practical_value: float,
        public_interest: float,
        reliability: float,
        media_quality: float,
    ) -> float:
        return round(
            curiosity * 0.34
            + novelty * 0.23
            + practical_value * 0.16
            + public_interest * 0.12
            + reliability * 0.10
            + media_quality * 0.05,
            2,
        )

    @staticmethod
    def _discovery_sort_score(
        ev: Dict[str, Any],
    ) -> float:
        return float(
            ev.get(
                "discovery_score",
                ev.get("editorial_score", ev.get("balanced_score", 0)),
            )
            or 0
        )

    @staticmethod
    def _core_presentation_score(
        ev: Dict[str, Any],
    ) -> float:
        # presentation score навмисно НЕ містить +500 manual bonus.
        # Manual гарантовано входить, але розташовується органічно за змістом.
        return float(
            ev.get(
                "editorial_score",
                ev.get("balanced_score", ev.get("raw_score", 0)),
            )
            or 0
        )

    def _ensure_discovery_events(
        self,
        events: List[Dict[str, Any]],
        posts: List[Dict[str, Any]],
        past_events: Optional[
            Union[
                List[str],
                List[Dict[str, str]],
            ]
        ],
        max_retries: int,
        desired_count: int,
    ) -> List[Dict[str, Any]]:
        if desired_count <= 0:
            return events

        discovery_context = self._build_discovery_context(posts)
        if not discovery_context:
            logger.warning(
                "Discovery-pass: не вдалося побудувати окремий контекст."
            )
            return events

        recovered = self._analyze_discovery_events(
            discovery_context,
            past_events,
            max_retries,
            desired_count,
        )

        if not recovered:
            logger.warning(
                "Discovery-pass не повернув кандидатів. "
                "Нічого не вигадуємо і не знижуємо quality gate."
            )
            return events

        result: List[Dict[str, Any]] = [
            dict(ev)
            for ev in events
            if isinstance(ev, dict)
        ]

        added = 0
        merged_count = 0

        for recovered_ev in recovered:
            if not isinstance(recovered_ev, dict):
                continue

            candidate = dict(recovered_ev)
            source_ids = self._valid_source_ids(
                candidate.get("source_ids"),
                posts,
            )
            if not source_ids:
                continue

            candidate["source_ids"] = source_ids
            candidate["digest_role"] = "discovery"
            candidate["is_discovery_candidate"] = True

            # Discovery-pass не має права обходити history gate.
            # eligible лишаємо тим, що повернула модель; ranking перевірить.
            candidate["eligible_for_digest"] = bool(
                candidate.get("eligible_for_digest", True)
            )

            match_idx = self._find_matching_event_index(
                result,
                candidate,
                posts,
            )

            if match_idx is not None:
                existing_role = str(
                    result[match_idx].get("digest_role") or ""
                ).strip().lower()
                result[match_idx] = self._merge_events(
                    result[match_idx],
                    candidate,
                    posts,
                )
                # Не перетворюємо вже явну core-подію на цікавинку.
                if existing_role not in {"core", "discovery"}:
                    result[match_idx]["digest_role"] = "discovery"
                    result[match_idx]["is_discovery_candidate"] = True
                merged_count += 1
                continue

            candidate["event_id"] = self._unique_event_id(
                str(candidate.get("event_id") or "D_RECOVER"),
                result,
            )
            result.append(candidate)
            added += 1

        logger.info(
            "Discovery-pass: отримано=%s, додано=%s, змерджено=%s.",
            len(recovered),
            added,
            merged_count,
        )

        return result

    def _analyze_discovery_events(
        self,
        posts_context: str,
        past_events: Optional[
            Union[
                List[str],
                List[Dict[str, str]],
            ]
        ],
        max_retries: int,
        desired_count: int,
    ) -> List[Dict[str, Any]]:
        history_block = self._build_history_block(past_events)
        candidate_count = max(
            4,
            min(
                self.DISCOVERY_RECOVERY_CANDIDATES,
                desired_count * 2 + 2,
            ),
        )

        prompt = f"""
Ти — окремий discovery-редактор українського Telegram-дайджесту.

Основний Analyzer уже займається важкими новинами: війною, великими атаками,
політикою, міжнародними рішеннями та кризами. Твоє завдання — НЕ дублювати
його роботу, а знайти до {candidate_count} справді якісних "цікавинок".

ЩО ТАКЕ DISCOVERY-НОВИНА:
- сильна технологічна або наукова новина;
- цікаве українське виробництво, винахід, стартап або бізнес-подія;
- незвичайний міжнародний факт, який має самостійну інформаційну цінність;
- практично корисна зміна правил, сервісів, транспорту, тарифів чи побуту;
- помітне досягнення, рекорд, новий продукт, дослідження або відкриття;
- якісна суспільна/культурна подія, про яку природно сказати
  "О, цього я не знав".

ЦЕ НЕ DISCOVERY:
- звичайна тривога, рух БпЛА, рутинний обстріл;
- чергова політична заява без рішення;
- дрібний кримінал, ДТП, локальна пожежа;
- шок-контент, плітки, клікбейт;
- просто важка воєнна новина, якщо її єдина цінність — стратегічна важливість.

QUALITY GATE:
Поверни лише події, які реально не соромно поставити в КІНЕЦЬ короткого
дайджесту після 5-9 серйозних новин. Краще 1 сильний кандидат, ніж 5 слабких.
Зазвичай сильна discovery-подія має novelty >= 60 і хоча б один фактор:
curiosity >= 70 або practical_value >= 75. Не підганяй оцінки штучно.

АРХІВ ВЖЕ ОПУБЛІКОВАНИХ ПОДІЙ:
{history_block}

ПОВТОРИ:
Якщо ця сама реальна подія вже була в архіві, став is_history_repeat=true.
Повтор може бути eligible_for_digest=true лише якщо history_update_strength
>= 60 і є справді новий значущий розвиток. Інше відкидай.

Одна реальна подія = один event_id. Об'єднуй дублікати різних каналів, фото
та відео однієї події. Обирай найкраще factual-source і найкраще media-source.

Для КОЖНОЇ повернутої події:
- digest_role="discovery";
- is_discovery_candidate=true;
- eligible_for_digest=true лише якщо вона проходить quality gate;
- усі оцінки 0-100 виставляй чесно;
- не вигадуй жодного факту.

ДОЗВОЛЕНІ category:
war, politics, economy, international, society, technology, science, culture, other.

ДОЗВОЛЕНІ event_type:
major_attack, battlefield_change, military_event, political_decision,
international_decision, economic_event, critical_infrastructure,
major_accident, major_crime, science_tech, social_event, culture_event,
routine_attack, routine_statement, minor_local_event, minor_accident,
alert_only, other.

ВІДПОВІДЬ ТІЛЬКИ JSON:
{{
  "events": [
    {{
      "event_id": "D1",
      "source_ids": [12, 44],
      "best_factual_source_id": 12,
      "best_media_source_id": 44,
      "eligible_for_digest": true,
      "rejection_reason": "",
      "digest_role": "discovery",
      "is_discovery_candidate": true,
      "event_type": "science_tech",
      "category": "technology",
      "importance": 66,
      "scale": 55,
      "reliability": 84,
      "public_interest": 76,
      "novelty": 88,
      "curiosity": 91,
      "practical_value": 52,
      "media_quality": 80,
      "national_relevance": 58,
      "urgency": 64,
      "is_history_repeat": false,
      "history_update_strength": 0,
      "headline_hint": "Короткий конкретний заголовок",
      "key_facts": ["Факт 1", "Факт 2"],
      "why_it_matters": "Чому це справді цікаво або корисно.",
      "summary": "Стислий фактологічний опис."
    }}
  ]
}}

TELEGRAM POSTS FOR DISCOVERY SEARCH:
{posts_context}
"""

        data = self._call_json_with_cascade(
            prompt,
            max_retries,
            "DISCOVERY_ANALYZER",
            temperature=0.18,
        )

        return (
            data.get("events", [])
            if (
                data
                and isinstance(data.get("events"), list)
            )
            else []
        )

    def _analyze_events(
        self,
        posts_context: str,
        past_events: Optional[
            Union[
                List[str],
                List[Dict[str, str]],
            ]
        ],
        max_retries: int,
    ) -> List[Dict[str, Any]]:
        history_block = self._build_history_block(
            past_events
        )

        prompt = f"""
Ти — старший редактор загальноукраїнського новинного Telegram-дайджесту.

ТВОЯ ЗАДАЧА:
Із потоку Telegram-повідомлень знайти події, які реально заслуговують
на місце серед головних, найкорисніших ТА найцікавіших новин останніх 4 годин.

Це НЕ звичайна стрічка новин і НЕ збір усіх повідомлень.

Читач відкриває канал кілька разів на день і хоче за кілька хвилин:
- зрозуміти головне;
- не пропустити важливе;
- побачити 1-3 події, про які природно хочеться сказати
  "О, цього я не знав" або "Цікаво".

Для кожної події запитай:
"Чи варто знати це людині, яка прочитає лише 7-10 новин?"

Не плутай "цікаво" з клікбейтом.
Наша мета — не сенсаційність, а сильна інформаційна цінність,
корисність, новизна та тематичне різноманіття.

━━━━━━━━━━━━━━━━━━━━
АРХІВ ВЖЕ ОПУБЛІКОВАНИХ ПОДІЙ:
{history_block}
━━━━━━━━━━━━━━━━━━━━

ПРАВИЛО ПРО ПОВТОРИ:

Архів — це вже опубліковані новини.
Не повторюй ту саму реальну подію тільки через новий пост,
інше формулювання, інший канал або нове фото.

Якщо подія вже є в архіві:
- is_history_repeat=true;
- history_update_strength показує силу НОВОГО розвитку від 0 до 100.

history_update_strength 0-39:
суттєво нового немає — eligible_for_digest=false.

history_update_strength 40-59:
є невелике уточнення, але його недостатньо для повторної появи
у короткому дайджесті — зазвичай eligible_for_digest=false.

history_update_strength 60-100:
з'явився реально новий значущий розвиток: нові великі наслідки,
важливе рішення, підтвердження масштабу, нові жертви, новий об'єкт,
результат операції або інший факт, який змінює картину події.
Тоді подію можна допустити повторно.

━━━━━━━━━━━━━━━━━━━━
ЕТАП 1 — ЗГРУПУЙ ПОСТИ У ПОДІЇ.

Одна реальна подія = один event_id.

Об'єднуй:
- повідомлення про одну атаку;
- перші дані та подальші уточнення;
- фото та відео тієї самої події;
- повідомлення різних каналів про один факт.

Не створюй новий event_id лише через інше формулювання
або через появу ще одного фото/відео.

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
13. Гаряча подія, яка прямо зараз суттєво змінює інформаційну картину.
14. Підтверджене влучання або наслідки на важливому промисловому,
    енергетичному, логістичному, військовому чи великому комерційному об'єкті,
    якщо це має реальне економічне, суспільне або новинне значення.
15. Висока самостійна цікавість: незвичайна, пізнавальна,
    технологічна, наукова, бізнесова або суспільна подія,
    про яку значна частина читачів захоче дізнатися.
16. Висока практична цінність: зміна правил, тарифів, транспорту,
    сервісів, інфраструктури або повсякденного життя,
    яка прямо стосується великої кількості людей.
17. Помітне українське досягнення: нове виробництво, технологія,
    винахід, інфраструктурний проєкт, великий контракт або інша подія,
    яка показує реальну зміну можливостей країни.
18. Сильна "discovery"-новина: не обов'язково стратегічна,
    але вона має новизну, конкретику і природно запам'ятовується.

ВАЖЛИВО ПРО РІЗНОМАНІТТЯ:

Короткий дайджест повинен показувати не лише те, що було НАЙВАЖЛИВІШИМ,
а й те, що було НАЙЦІКАВІШИМ або НАЙКОРИСНІШИМ.

Якщо за останні 4 години є 1-3 сильні технологічні, наукові,
суспільні, бізнесові, практично корисні чи просто незвичайні події,
не відкидай їх лише тому, що вони менш стратегічні,
ніж війна, політика або міжнародні рішення.

Не занижуй подію тільки тому, що вона локальна,
якщо вона має високу практичну цінність, цікавість або резонанс.

ЗАЗВИЧАЙ ВІДКИДАЙ:

- рутинні обстріли без суттєвих наслідків;
- локальні пошкодження без ширшого значення;
- 1-2 поранених без інших значних факторів;
- тривоги;
- рух БпЛА;
- загрози ракет без підтверджених наслідків;
- дрібні ДТП;
- локальні побутові пожежі;
- дрібний кримінал без широкого резонансу;
- комунальні аварії без значного впливу;
- заяви політиків без реального рішення;
- повтори старих новин;
- чутки;
- клікбейтні курйози без інформаційної цінності;
- плітки про знаменитостей;
- контент, єдина цінність якого — шок або емоція.

Для атак допускай подію, якщо:
- атака масована або комбінована;
- є значна кількість жертв;
- пошкоджена критична або стратегічна інфраструктура;
- є серйозні наслідки для великого міста;
- є військовий, політичний або значний економічний результат;
- пошкоджено важливий промисловий, логістичний або великий комерційний об'єкт
  і це має помітне ширше значення;
- подія має винятковий характер;
- з'явився суттєвий новий розвиток уже відомої великої атаки.

━━━━━━━━━━━━━━━━━━━━
ГАРЯЧІ ТА КОРОТКІ НОВИНИ.

Довжина Telegram-повідомлення НЕ є показником важливості.

Короткий пост із одного або двох речень може бути сильнішою новиною
за довгий текст.

НЕ знижуй importance, novelty, public_interest, curiosity,
practical_value або urgency лише через малу довжину повідомлення.

Коротка новина може бути eligible_for_digest=true, якщо вона містить
самодостатній сильний факт, зокрема:

- підтверджене влучання;
- серйозні наслідки атаки;
- пожежу або пошкодження важливого об'єкта;
- удар по значному промисловому підприємству;
- удар по енергетичному, логістичному або військовому об'єкту;
- незвичну або значущу ціль атаки;
- перші підтверджені наслідки великої події;
- важливий новий розвиток історії, яка відбувається прямо зараз;
- нове правило, яке безпосередньо вплине на людей;
- сильний технологічний або науковий факт;
- помітне українське виробництво чи досягнення;
- незвичайну подію з широким суспільним інтересом.

━━━━━━━━━━━━━━━━━━━━
ЦІКАВІСТЬ ТА ПРАКТИЧНА КОРИСТЬ.

curiosity:
наскільки новина викликає природну реакцію:
"О, цього я не знав", "Оце цікаво", "Це варто запам'ятати".

Високий curiosity може мати:
- незвичайна технологія або відкриття;
- цікаве українське виробництво;
- неочікувана міжнародна подія;
- незвичайний бізнес-кейс;
- рекорд;
- сильне досягнення;
- помітна зміна у звичному житті;
- резонансна подія;
- конкретний факт, який легко переказати іншій людині.

curiosity НЕ означає клікбейт.
Не підвищуй оцінку через плітки, шок-контент або дрібний кримінал.

practical_value:
наскільки інформація реально корисна читачеві.

Високий practical_value мають:
- зміни правил;
- транспорт;
- тарифи;
- державні сервіси;
- соціальні правила;
- зміни роботи міст;
- обмеження;
- нові можливості або сервіси;
- рішення, які прямо впливають на повсякденне життя.

━━━━━━━━━━━━━━━━━━━━
ФОТО ТА ВІДЕО:

Наявність фото або відео сама по собі НЕ робить слабку подію важливою.

Але реальне фото або відео безпосередньо з місця події є
додатковим сильним фактором, якщо воно:
- показує реальні наслідки значущої події;
- є першими кадрами з місця;
- додає нову фактичну інформацію;
- підтверджує масштаб або характер події;
- показує наслідки для важливого об'єкта.

Не плутай це зі звичайним ілюстративним фото.

━━━━━━━━━━━━━━━━━━━━
ЕТАП 3 — ДЖЕРЕЛА.

best_factual_source_id:
найкраще джерело для підтвердження фактів.

best_media_source_id:
джерело з найкращим фото або відео з місця події.

Це можуть бути різні джерела.

Не став best_media_source_id лише тому, що пост має картинку.
Віддавай перевагу медіа, яке за текстом поста схоже саме на кадри
з місця події або наслідків.

━━━━━━━━━━━━━━━━━━━━
ОЦІНКИ 0-100:

importance
scale
reliability
public_interest
novelty
curiosity
practical_value
media_quality
national_relevance
urgency
history_update_strength

importance:
наскільки подія важлива сама по собі.

public_interest:
наскільки багато читачів реально захочуть про це знати.

novelty:
наскільки це новий факт або новий розвиток.

curiosity:
наскільки подія цікава, незвичайна, пізнавальна або запам'ятовується.

practical_value:
наскільки подія корисна у повсякденному житті читача.

urgency:
наскільки подія є гарячою і актуальною саме зараз.

Високий urgency став, якщо:
- подія відбулася щойно або активно розвивається;
- з'явилися перші підтверджені наслідки;
- це перша достовірна інформація про значущу подію;
- з'явилися важливі нові факти або кадри з місця.

Сам по собі високий urgency НЕ робить тривогу,
рух БпЛА або непідтверджену загрозу головною новиною.

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

headline_hint — короткий, конкретний заголовок.
key_facts — 2-6 найважливіших підтверджених фактів.
why_it_matters — коротко, чому це важливо, цікаво або корисно.
summary — стислий фактологічний опис.
rejection_reason — конкретна причина відхилення.
is_history_repeat — чи ця сама реальна подія вже є в архіві.

digest_role — редакційна роль події:
- "core" = важка/головна новина: війна, значуща політика, великі рішення,
  серйозні наслідки, важлива економіка, безпека, великі суспільні події;
- "discovery" = якісна цікавинка для КІНЦЯ випуску: наука, технології,
  бізнес, виробництво, досягнення, практично корисна зміна, незвичайний
  факт або інша подія, чия головна сила — curiosity/practical value.

is_discovery_candidate=true став лише для справді самодостатньої
discovery-події. Не називай discovery звичайну важку новину тільки через
високий інтерес аудиторії.

Якщо є ⭐ [ПРІОРИТЕТ АДМІНІСТРАТОРА]:
- ОБОВ'ЯЗКОВО включи цей пост до однієї з подій у source_ids;
- eligible_for_digest=true;
- не відкидай його через історію, низьку важливість або тип події;
- якщо кілька priority-постів описують одну реальну подію — об'єднай їх;
- importance, novelty, urgency, curiosity та інші оцінки виставляй ЧЕСНО
  за змістом самої події, не завищуй їх автоматично лише через priority.

Не створюй події з очевидного шуму серед звичайних постів.
Але й не будь надто суворим до якісних discovery-новин.
Якщо в потоці є достатньо матеріалу, поверни орієнтовно 10-16 добрих
кандидатів різних типів. НЕ зупиняйся штучно на 5 подіях, якщо є інші
якісні кандидати. Краще 10-16 добрих кандидатів різних типів,
ніж 7 однакових важких новин і пропущені цікаві події.

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
      "digest_role": "core",
      "is_discovery_candidate": false,
      "event_type": "major_attack",
      "category": "war",
      "importance": 94,
      "scale": 88,
      "reliability": 93,
      "public_interest": 91,
      "novelty": 82,
      "curiosity": 68,
      "practical_value": 20,
      "media_quality": 95,
      "national_relevance": 94,
      "urgency": 91,
      "is_history_repeat": false,
      "history_update_strength": 0,
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

        data = self._call_json_with_cascade(
            prompt,
            max_retries,
            "ANALYZER",
            temperature=0.15,
        )

        return (
            data.get("events", [])
            if (
                data
                and isinstance(data.get("events"), list)
            )
            else []
        )

    def _get_priority_post_ids(
        self,
        posts: List[Dict[str, Any]],
    ) -> List[int]:
        return [
            idx
            for idx, post in enumerate(posts)
            if (
                bool(post.get("is_priority"))
                and bool((post.get("text") or "").strip())
            )
        ]

    def _covered_priority_ids(
        self,
        events: List[Dict[str, Any]],
        posts: List[Dict[str, Any]],
    ) -> set:
        priority_ids = set(self._get_priority_post_ids(posts))
        covered = set()

        for ev in events:
            for source_id in ev.get("source_ids", []) or []:
                if (
                    isinstance(source_id, int)
                    and source_id in priority_ids
                ):
                    covered.add(source_id)

        return covered

    def _ensure_priority_events(
        self,
        events: List[Dict[str, Any]],
        posts: List[Dict[str, Any]],
        past_events: Optional[
            Union[
                List[str],
                List[Dict[str, str]],
            ]
        ],
        max_retries: int,
    ) -> List[Dict[str, Any]]:
        priority_ids = self._get_priority_post_ids(posts)
        if not priority_ids:
            return events

        result: List[Dict[str, Any]] = []
        for ev in events or []:
            if not isinstance(ev, dict):
                continue

            ev_copy = dict(ev)
            ev_copy["source_ids"] = self._valid_source_ids(
                ev_copy.get("source_ids"),
                posts,
            )
            result.append(ev_copy)

        covered = self._covered_priority_ids(result, posts)
        missing = [
            source_id
            for source_id in priority_ids
            if source_id not in covered
        ]

        if not missing:
            logger.info(
                "Priority-check: Analyzer покрив усі %s manual-пости.",
                len(priority_ids),
            )
            return result

        logger.warning(
            "Priority-check: Analyzer пропустив %s з %s manual-постів: %s. "
            "Запускаємо окремий priority-pass.",
            len(missing),
            len(priority_ids),
            missing,
        )

        # Аналізуємо ВСІ priority-пости разом, а не лише missing.
        # Це дає моделі шанс правильно склеїти два ручні дублі в одну подію.
        priority_context = self._build_posts_context(
            posts,
            only_ids=priority_ids,
            max_chars=self.PRIORITY_RECOVERY_MAX_CHARS,
        )

        recovered_events: List[Dict[str, Any]] = []
        if priority_context:
            recovered_events = self._analyze_priority_events(
                priority_context,
                past_events,
                max_retries,
            )

        if recovered_events:
            logger.info(
                "Priority-pass повернув %s подій.",
                len(recovered_events),
            )

            for recovered in recovered_events:
                if not isinstance(recovered, dict):
                    continue

                recovered = dict(recovered)
                recovered_ids = [
                    source_id
                    for source_id in self._valid_source_ids(
                        recovered.get("source_ids"),
                        posts,
                    )
                    if source_id in priority_ids
                ]

                if not recovered_ids:
                    continue

                recovered["source_ids"] = recovered_ids
                recovered["eligible_for_digest"] = True
                recovered["rejection_reason"] = ""

                match_idx = self._find_matching_event_index(
                    result,
                    recovered,
                    posts,
                )

                if match_idx is not None:
                    result[match_idx] = self._merge_events(
                        result[match_idx],
                        recovered,
                        posts,
                    )
                else:
                    recovered["event_id"] = self._unique_event_id(
                        str(recovered.get("event_id") or "P_RECOVER"),
                        result,
                    )
                    result.append(recovered)

        # Після LLM recovery пробуємо приклеїти ще не покритий manual-post
        # до вже наявної події суто за текстовою схожістю.
        covered = self._covered_priority_ids(result, posts)
        still_missing = [
            source_id
            for source_id in priority_ids
            if source_id not in covered
        ]

        for source_id in list(still_missing):
            match_idx = self._find_event_for_post(
                result,
                source_id,
                posts,
            )
            if match_idx is None:
                continue

            merged_ids = self._valid_source_ids(
                list(result[match_idx].get("source_ids", []))
                + [source_id],
                posts,
            )
            result[match_idx]["source_ids"] = merged_ids
            result[match_idx]["eligible_for_digest"] = True

        # Абсолютна гарантія: якщо моделі не спрацювали або знову щось
        # не повернули, Python сам створює priority-event із сирого поста.
        covered = self._covered_priority_ids(result, posts)
        still_missing = [
            source_id
            for source_id in priority_ids
            if source_id not in covered
        ]

        if still_missing:
            groups = self._group_priority_ids_by_similarity(
                still_missing,
                posts,
            )

            for group_number, source_ids in enumerate(
                groups,
                start=1,
            ):
                synthetic = self._build_synthetic_priority_event(
                    source_ids,
                    posts,
                    group_number,
                )

                match_idx = self._find_matching_event_index(
                    result,
                    synthetic,
                    posts,
                )

                if match_idx is not None:
                    result[match_idx] = self._merge_events(
                        result[match_idx],
                        synthetic,
                        posts,
                    )
                else:
                    synthetic["event_id"] = self._unique_event_id(
                        str(synthetic.get("event_id") or "P_SYNTH"),
                        result,
                    )
                    result.append(synthetic)

        final_covered = self._covered_priority_ids(result, posts)
        final_missing = [
            source_id
            for source_id in priority_ids
            if source_id not in final_covered
        ]

        if final_missing:
            # Сюди код практично не повинен доходити. Лог залишаємо,
            # щоб будь-яку структурну помилку було видно одразу.
            logger.error(
                "CRITICAL priority guarantee failed for source_ids=%s",
                final_missing,
            )
        else:
            logger.info(
                "Priority guarantee: усі %s manual-пости присутні "
                "у подіях після Analyzer.",
                len(priority_ids),
            )

        return result

    def _analyze_priority_events(
        self,
        posts_context: str,
        past_events: Optional[
            Union[
                List[str],
                List[Dict[str, str]],
            ]
        ],
        max_retries: int,
    ) -> List[Dict[str, Any]]:
        history_block = self._build_history_block(past_events)

        prompt = f"""
Ти — редактор аварійного priority-pass для новинного Telegram-дайджесту.

Цей запит містить ТІЛЬКИ пости, які адміністратор вручну додав у чергу.
Вони вже пройшли людський вибір і НЕ МОЖУТЬ бути відкинуті.

ТВОЯ ЗАДАЧА:
1. Перетвори КОЖЕН наданий ID на подію.
2. Якщо два або більше ID описують ОДНУ Й ТУ САМУ реальну подію,
   об'єднай їх в ОДИН event і помісти ВСІ такі ID у source_ids.
3. Не створюй два events для одного дубля тільки через різний канал,
   фото, відео або інше формулювання.
4. Якщо це розвиток уже опублікованої історії — можеш позначити
   is_history_repeat=true та оцінити history_update_strength,
   але eligible_for_digest ЗАВЖДИ має бути true: це manual override.
5. Не вигадуй фактів. Використовуй лише текст постів.
6. importance, scale, public_interest, curiosity, practical_value та інші
   оцінки виставляй чесно за змістом. Manual priority гарантує включення,
   але не означає автоматично importance=100.
7. Визнач digest_role: "core" для головної важкої новини або "discovery"
   для якісної цікавої/практичної події, яку органічно ставити в кінці випуску.
   Для discovery також став is_discovery_candidate=true.
8. КОЖЕН ID із вхідного блоку повинен зустрітися РІВНО в одному event.source_ids.

АРХІВ:
{history_block}

ДОЗВОЛЕНІ category:
war, politics, economy, international, society, technology,
science, culture, other.

ДОЗВОЛЕНІ event_type:
major_attack, battlefield_change, military_event, political_decision,
international_decision, economic_event, critical_infrastructure,
major_accident, major_crime, science_tech, social_event, culture_event,
routine_attack, routine_statement, minor_local_event, minor_accident,
alert_only, other.

ВІДПОВІДЬ ТІЛЬКИ JSON:
{{
  "events": [
    {{
      "event_id": "P1",
      "source_ids": [7, 8],
      "best_factual_source_id": 7,
      "best_media_source_id": 8,
      "eligible_for_digest": true,
      "rejection_reason": "",
      "digest_role": "core",
      "is_discovery_candidate": false,
      "event_type": "other",
      "category": "other",
      "importance": 70,
      "scale": 60,
      "reliability": 80,
      "public_interest": 75,
      "novelty": 80,
      "curiosity": 75,
      "practical_value": 40,
      "media_quality": 80,
      "national_relevance": 70,
      "urgency": 80,
      "is_history_repeat": false,
      "history_update_strength": 0,
      "headline_hint": "Короткий конкретний заголовок",
      "key_facts": ["Факт 1", "Факт 2"],
      "why_it_matters": "Коротко про значення події.",
      "summary": "Стислий фактологічний опис."
    }}
  ]
}}

MANUAL POSTS:
{posts_context}
"""

        data = self._call_json_with_cascade(
            prompt,
            max_retries,
            "PRIORITY_ANALYZER",
            temperature=0.10,
        )

        return (
            data.get("events", [])
            if (
                data
                and isinstance(data.get("events"), list)
            )
            else []
        )

    def _valid_source_ids(
        self,
        source_ids: Any,
        posts: List[Dict[str, Any]],
    ) -> List[int]:
        if not isinstance(source_ids, list):
            return []

        result = []
        seen = set()

        for source_id in source_ids:
            if (
                isinstance(source_id, int)
                and 0 <= source_id < len(posts)
                and source_id not in seen
            ):
                result.append(source_id)
                seen.add(source_id)

        return result

    def _find_matching_event_index(
        self,
        events: List[Dict[str, Any]],
        candidate: Dict[str, Any],
        posts: List[Dict[str, Any]],
    ) -> Optional[int]:
        candidate_ids = set(
            self._valid_source_ids(
                candidate.get("source_ids"),
                posts,
            )
        )

        for idx, event in enumerate(events):
            event_ids = set(
                self._valid_source_ids(
                    event.get("source_ids"),
                    posts,
                )
            )

            if candidate_ids & event_ids:
                return idx

        for idx, event in enumerate(events):
            if self._events_are_same(event, candidate, posts):
                return idx

        return None

    def _find_event_for_post(
        self,
        events: List[Dict[str, Any]],
        source_id: int,
        posts: List[Dict[str, Any]],
    ) -> Optional[int]:
        if not (0 <= source_id < len(posts)):
            return None

        post_text = (posts[source_id].get("text") or "").strip()
        if not post_text:
            return None

        for idx, event in enumerate(events):
            for ref_text in self._event_reference_texts(event, posts):
                if self._texts_same_event(post_text, ref_text):
                    return idx

        return None

    def _events_are_same(
        self,
        left: Dict[str, Any],
        right: Dict[str, Any],
        posts: List[Dict[str, Any]],
    ) -> bool:
        left_texts = self._event_reference_texts(left, posts)
        right_texts = self._event_reference_texts(right, posts)

        for left_text in left_texts[:8]:
            for right_text in right_texts[:8]:
                if self._texts_same_event(left_text, right_text):
                    return True

        return False

    def _event_reference_texts(
        self,
        event: Dict[str, Any],
        posts: List[Dict[str, Any]],
    ) -> List[str]:
        texts: List[str] = []

        for source_id in self._valid_source_ids(
            event.get("source_ids"),
            posts,
        ):
            text = (posts[source_id].get("text") or "").strip()
            if text:
                texts.append(text)

        for field in [
            "headline_hint",
            "summary",
            "why_it_matters",
        ]:
            value = event.get(field)
            if isinstance(value, str) and value.strip():
                texts.append(value.strip())

        key_facts = event.get("key_facts")
        if isinstance(key_facts, list):
            facts_text = " ".join(
                str(item).strip()
                for item in key_facts[:6]
                if str(item).strip()
            )
            if facts_text:
                texts.append(facts_text)

        return texts

    @staticmethod
    def _normalize_similarity_text(text: str) -> str:
        text = str(text or "").lower()
        text = re.sub(r"https?://\S+|t\.me/\S+", " ", text)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"[^0-9a-zа-яіїєґёъыэ\s-]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _texts_same_event(
        self,
        left: str,
        right: str,
    ) -> bool:
        a = self._normalize_similarity_text(left)
        b = self._normalize_similarity_text(right)

        if not a or not b:
            return False

        if a == b:
            return True

        shorter = min(len(a), len(b))
        longer = max(len(a), len(b))

        if (
            shorter >= 45
            and (a in b or b in a)
            and shorter / max(longer, 1) >= 0.55
        ):
            return True

        seq_ratio = SequenceMatcher(
            None,
            a[:1200],
            b[:1200],
        ).ratio()

        tokens_a = {
            token
            for token in re.findall(
                r"[0-9a-zа-яіїєґёъыэ-]{3,}",
                a,
            )
        }
        tokens_b = {
            token
            for token in re.findall(
                r"[0-9a-zа-яіїєґёъыэ-]{3,}",
                b,
            )
        }

        if not tokens_a or not tokens_b:
            return seq_ratio >= 0.88

        common = tokens_a & tokens_b
        union = tokens_a | tokens_b

        jaccard = len(common) / max(len(union), 1)
        overlap = len(common) / max(
            min(len(tokens_a), len(tokens_b)),
            1,
        )

        if seq_ratio >= 0.82:
            return True

        if (
            len(common) >= 6
            and overlap >= 0.58
            and jaccard >= 0.40
        ):
            return True

        if (
            len(common) >= 9
            and overlap >= 0.54
            and jaccard >= 0.34
        ):
            return True

        return False

    def _merge_events(
        self,
        base: Dict[str, Any],
        incoming: Dict[str, Any],
        posts: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        merged = dict(base)

        source_ids = self._valid_source_ids(
            list(base.get("source_ids", []) or [])
            + list(incoming.get("source_ids", []) or []),
            posts,
        )
        merged["source_ids"] = source_ids

        # Основний Analyzer зазвичай має кращий широкий контекст, тому його
        # поля лишаємо. Recovery заповнює лише порожні місця.
        for field in [
            "headline_hint",
            "summary",
            "why_it_matters",
            "event_type",
            "category",
            "digest_role",
            "is_discovery_candidate",
            "best_factual_source_id",
            "best_media_source_id",
        ]:
            current = merged.get(field)
            incoming_value = incoming.get(field)
            if (
                (current is None or current == "" or current == [])
                and incoming_value not in (None, "", [])
            ):
                merged[field] = incoming_value

        base_facts = merged.get("key_facts")
        incoming_facts = incoming.get("key_facts")
        if isinstance(base_facts, list) or isinstance(incoming_facts, list):
            facts = []
            seen = set()
            for item in (
                (base_facts if isinstance(base_facts, list) else [])
                + (
                    incoming_facts
                    if isinstance(incoming_facts, list)
                    else []
                )
            ):
                value = str(item).strip()
                key = value.lower()
                if value and key not in seen:
                    facts.append(value)
                    seen.add(key)
            merged["key_facts"] = facts[:8]

        merged["eligible_for_digest"] = True
        merged["rejection_reason"] = ""

        return merged

    def _unique_event_id(
        self,
        preferred: str,
        events: List[Dict[str, Any]],
    ) -> str:
        preferred = re.sub(
            r"[^A-Za-z0-9_-]+",
            "_",
            preferred or "P_RECOVER",
        ).strip("_") or "P_RECOVER"

        existing = {
            str(ev.get("event_id") or "")
            for ev in events
        }

        if preferred not in existing:
            return preferred

        counter = 2
        while f"{preferred}_{counter}" in existing:
            counter += 1

        return f"{preferred}_{counter}"

    def _group_priority_ids_by_similarity(
        self,
        source_ids: List[int],
        posts: List[Dict[str, Any]],
    ) -> List[List[int]]:
        groups: List[List[int]] = []

        for source_id in source_ids:
            text = (posts[source_id].get("text") or "").strip()
            placed = False

            for group in groups:
                if any(
                    self._texts_same_event(
                        text,
                        posts[other_id].get("text") or "",
                    )
                    for other_id in group
                ):
                    group.append(source_id)
                    placed = True
                    break

            if not placed:
                groups.append([source_id])

        return groups

    def _build_synthetic_priority_event(
        self,
        source_ids: List[int],
        posts: List[Dict[str, Any]],
        sequence: int,
    ) -> Dict[str, Any]:
        source_ids = self._valid_source_ids(source_ids, posts)
        if not source_ids:
            raise ValueError("Synthetic priority event without source_ids")

        # Для тексту беремо найінформативніший manual-post, а для публікації
        # ranking пізніше окремо вибере найкращий factual/media source.
        text_source_id = max(
            source_ids,
            key=lambda source_id: len(
                posts[source_id].get("text") or ""
            ),
        )
        source_text = (
            posts[text_source_id].get("text") or ""
        ).strip()

        sentences = self._extract_sentences(source_text)
        if not sentences and source_text:
            sentences = [source_text]

        headline = self._priority_headline_from_text(source_text)
        summary_sentences = [
            self._ensure_sentence_end(sentence)
            for sentence in sentences[:2]
            if sentence.strip()
        ]
        summary = " ".join(summary_sentences).strip()
        if not summary:
            summary = self._ensure_sentence_end(
                source_text[:400].strip()
            )

        key_facts = [
            self._ensure_sentence_end(sentence)
            for sentence in sentences[:5]
            if sentence.strip()
        ]

        category, event_type = self._infer_priority_category_and_type(
            source_text
        )

        has_video = any(
            posts[source_id].get("has_video")
            for source_id in source_ids
        )
        has_media = any(
            posts[source_id].get("has_media")
            for source_id in source_ids
        )

        return {
            "event_id": f"P_SYNTH_{sequence}",
            "source_ids": source_ids,
            "best_factual_source_id": text_source_id,
            "best_media_source_id": None,
            "eligible_for_digest": True,
            "rejection_reason": "",
            "digest_role": (
                "discovery"
                if category in {"technology", "science", "culture"}
                else "core"
            ),
            "is_discovery_candidate": (
                category in {"technology", "science", "culture"}
            ),
            "event_type": event_type,
            "category": category,
            "importance": 72,
            "scale": 60,
            "reliability": 78,
            "public_interest": 75,
            "novelty": 85,
            "curiosity": 78,
            "practical_value": 45,
            "media_quality": (
                88
                if has_video
                else (75 if has_media else 35)
            ),
            "national_relevance": 68,
            "urgency": 85,
            "is_history_repeat": False,
            "history_update_strength": 0,
            "headline_hint": headline,
            "key_facts": key_facts,
            "why_it_matters": "",
            "summary": summary,
        }

    def _priority_headline_from_text(
        self,
        text: str,
    ) -> str:
        clean = re.sub(r"https?://\S+|t\.me/\S+", " ", text or "")
        clean = re.sub(r"<[^>]+>", " ", clean)
        clean = re.sub(r"\s+", " ", clean).strip()

        if not clean:
            return "Пріоритетна подія"

        first_sentence = re.split(r"(?<=[.!?])\s+", clean)[0]
        first_sentence = first_sentence.strip(" -–—:;,.!?")

        # Забираємо частину декоративних символів на початку, але не
        # переписуємо сам зміст.
        first_sentence = re.sub(
            r"^[^0-9A-Za-zА-Яа-яІіЇїЄєҐґ]+",
            "",
            first_sentence,
        ).strip()

        if len(first_sentence) > 110:
            first_sentence = self._truncate_plain_text(
                first_sentence,
                110,
            )

        return first_sentence or "Пріоритетна подія"

    @staticmethod
    def _truncate_plain_text(
        text: str,
        max_chars: int,
    ) -> str:
        if len(text) <= max_chars:
            return text.strip()

        candidate = text[:max_chars].rstrip()
        last_space = candidate.rfind(" ")
        if last_space >= int(max_chars * 0.65):
            candidate = candidate[:last_space]

        return candidate.rstrip(" ,;:-") + "…"

    @staticmethod
    def _infer_priority_category_and_type(
        text: str,
    ) -> tuple:
        t = str(text or "").lower()

        if any(
            key in t
            for key in [
                "обстріл", "атака", "ракет", "дрон", "бпла",
                "фронт", "зсу", "окуп", "військ", "ппо",
                "бойов", "удар", "сбу", "гур", "розвід",
            ]
        ):
            event_type = (
                "major_attack"
                if any(
                    key in t
                    for key in [
                        "обстріл", "атака", "ракет", "дрон",
                        "бпла", "влуч", "удар",
                    ]
                )
                else "military_event"
            )
            return "war", event_type

        if any(
            key in t
            for key in [
                "дбр", "прокурат", "підозр", "затрим", "вбив",
                "стрілянин", "злочин", "поліці",
            ]
        ):
            return "society", "major_crime"

        if any(
            key in t
            for key in [
                "верховн", "рада", "кабмін", "уряд", "президент",
                "зеленськ", "міністр", "закон", "постанова",
                "вибор", "депутат",
            ]
        ):
            return "politics", "political_decision"

        if any(
            key in t
            for key in [
                "сша", "євросоюз", "нато", "польщ", "німеч",
                "франц", "британ", "трамп", "європ", "китай",
                "япон", "канада", "румун", "угорщ",
            ]
        ):
            return "international", "international_decision"

        if any(
            key in t
            for key in [
                "грн", "долар", "євро", "банк", "бюджет", "подат",
                "тариф", "економ", "ринок", "компан", "завод",
                "виробництв", "контракт", "зарплат", "пенсі",
            ]
        ):
            return "economy", "economic_event"

        if any(
            key in t
            for key in [
                "штучн", "інтелект", "нейромереж", "ai ", "gpt",
                "технолог", "стартап", "робот", "кібер", "додаток",
                "смартфон", "комп'ют", "чип", "процесор",
            ]
        ):
            return "technology", "science_tech"

        if any(
            key in t
            for key in [
                "вчен", "дослід", "науков", "відкрит", "медицин",
                "лікуван", "біолог", "космос", "фізик", "хімі",
            ]
        ):
            return "science", "science_tech"

        if any(
            key in t
            for key in [
                "фільм", "музик", "культур", "театр", "музей",
                "книг", "премі", "фестив",
            ]
        ):
            return "culture", "culture_event"

        return "society", "social_event"

    def _rank_events(
        self,
        events: List[Dict[str, Any]],
        posts: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        ranked = []

        rejected = {
            "no_sources": 0,
            "ineligible": 0,
            "hard_reject": 0,
            "history_repeat": 0,
            "low_value": 0,
        }

        for ev in events:
            try:
                src_ids = [
                    s
                    for s in ev.get("source_ids", [])
                    if (
                        isinstance(s, int)
                        and 0 <= s < len(posts)
                    )
                ]

                if not src_ids:
                    rejected["no_sources"] += 1
                    continue

                is_priority = any(
                    posts[s].get("is_priority")
                    for s in src_ids
                )

                eligible = bool(
                    ev.get("eligible_for_digest", False)
                )

                event_type = str(
                    ev.get("event_type") or "other"
                )

                is_history_repeat = bool(
                    ev.get("is_history_repeat", False)
                )

                history_update = self._safe_score(
                    ev.get("history_update_strength")
                )

                if not is_priority:
                    if not eligible:
                        rejected["ineligible"] += 1
                        continue

                    if event_type in HARD_REJECT_EVENT_TYPES:
                        rejected["hard_reject"] += 1
                        continue

                    if (
                        is_history_repeat
                        and history_update < 60
                    ):
                        rejected["history_repeat"] += 1
                        continue

                imp = self._safe_score(ev.get("importance"))
                scale = self._safe_score(ev.get("scale"))
                rel = self._safe_score(ev.get("reliability"))
                pub = self._safe_score(ev.get("public_interest"))
                nov = self._safe_score(ev.get("novelty"))
                cur = self._safe_score(ev.get("curiosity"))
                practical = self._safe_score(
                    ev.get("practical_value")
                )
                med = self._safe_score(ev.get("media_quality"))
                national = self._safe_score(
                    ev.get("national_relevance")
                )
                urgency = self._safe_score(ev.get("urgency"))

                category = ev.get("category", "other")
                if category not in self.ALLOWED_CATEGORIES:
                    category = "other"

                digest_role = self._resolve_digest_role(
                    ev,
                    event_type,
                    category,
                    imp,
                    national,
                    urgency,
                    cur,
                    practical,
                    nov,
                    pub,
                )

                discovery_qualified = (
                    digest_role == "discovery"
                    and (
                        is_priority
                        or self._discovery_quality(
                            rel,
                            nov,
                            cur,
                            practical,
                            pub,
                            category,
                        )
                    )
                )

                has_video = any(
                    posts[s].get("has_video")
                    for s in src_ids
                )

                has_media = any(
                    posts[s].get("has_media")
                    for s in src_ids
                )

                # Слабкі типи можуть пройти, якщо вони реально важливі,
                # дуже цікаві, корисні або мають сильний новий розвиток.
                if (
                    not is_priority
                    and event_type in LOW_VALUE_EVENT_TYPES
                ):
                    hot_exception = (
                        imp >= 70
                        or national >= 70
                        or cur >= 82
                        or practical >= 82
                        or (
                            urgency >= 75
                            and imp >= 60
                        )
                        or (
                            pub >= 70
                            and nov >= 65
                        )
                        or (
                            cur >= 75
                            and nov >= 70
                            and pub >= 60
                        )
                        or (
                            practical >= 75
                            and pub >= 60
                        )
                        or (
                            med >= 80
                            and imp >= 60
                            and (has_video or has_media)
                        )
                        or (
                            is_history_repeat
                            and history_update >= 75
                        )
                    )

                    if not hot_exception:
                        rejected["low_value"] += 1
                        continue

                tier_mult = self._event_source_multiplier(
                    src_ids,
                    posts,
                )

                # Важливість лишається головним фактором, але цікавість і
                # практична цінність достатньо сильні, щоб discovery не зникала.
                base_score = (
                    imp * 0.24
                    + scale * 0.10
                    + rel * 0.16
                    + pub * 0.10
                    + nov * 0.08
                    + cur * 0.10
                    + practical * 0.06
                    + national * 0.10
                    + urgency * 0.04
                    + med * 0.02
                )

                score = base_score * tier_mult

                score += min(len(src_ids) * 1.2, 6)

                meaningful_event = (
                    imp >= 60
                    or national >= 60
                    or pub >= 65
                    or urgency >= 75
                    or cur >= 75
                    or practical >= 75
                )

                if meaningful_event:
                    if has_video:
                        score += 5
                    elif has_media:
                        score += 2.5

                if urgency >= 80 and nov >= 65:
                    score += 4

                if urgency >= 85 and imp >= 75:
                    score += 4

                if cur >= 85 and nov >= 70:
                    score += 6
                elif cur >= 78 and nov >= 65:
                    score += 3

                if practical >= 85 and pub >= 65:
                    score += 6
                elif practical >= 75 and pub >= 60:
                    score += 3

                if (
                    event_type == "science_tech"
                    and cur >= 70
                    and nov >= 65
                ):
                    score += 3

                if (
                    is_history_repeat
                    and history_update >= 60
                ):
                    score += min(
                        (history_update - 60) * 0.10,
                        4,
                    )

                if rel < 45:
                    score -= 20
                elif rel < 60:
                    score -= 8

                if (
                    national < 40
                    and imp < 70
                    and pub < 70
                    and cur < 75
                    and practical < 75
                ):
                    score -= 10

                if (
                    not is_priority
                    and event_type in LOW_VALUE_EVENT_TYPES
                ):
                    score -= 4

                # editorial_score — реальна редакційна сила БЕЗ manual boost.
                # Саме її використовуємо для природного порядку у випуску.
                editorial_score = round(score, 2)

                if is_priority:
                    score += 500

                factual_source = self._select_factual_source(
                    src_ids,
                    posts,
                    ev.get("best_factual_source_id"),
                )

                media_source = self._select_media_source(
                    src_ids,
                    posts,
                    ev.get("best_media_source_id"),
                )

                publishing_source = (
                    media_source
                    if media_source is not None
                    else factual_source
                )

                discovery_score = self._calculate_discovery_score(
                    cur,
                    nov,
                    practical,
                    pub,
                    rel,
                    med,
                )

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
                    "digest_role": digest_role,
                    "is_discovery_candidate": (
                        digest_role == "discovery"
                    ),
                    "discovery_qualified": discovery_qualified,
                    "discovery_score": discovery_score,
                    "has_video": has_video,
                    "has_media": has_media,
                    "is_history_repeat": is_history_repeat,
                    "history_update_strength": history_update,
                    "importance": imp,
                    "scale": scale,
                    "reliability": rel,
                    "public_interest": pub,
                    "novelty": nov,
                    "curiosity": cur,
                    "practical_value": practical,
                    "media_quality": med,
                    "national_relevance": national,
                    "urgency": urgency,
                    "editorial_score": editorial_score,
                    "raw_score": round(score, 2),
                })

                ranked.append(ev_copy)

            except Exception as e:
                logger.warning(
                    "Помилка ranking події: "
                    f"{e}"
                )

        logger.info(
            "Ranking gate: "
            f"ineligible={rejected['ineligible']}, "
            f"history_repeat={rejected['history_repeat']}, "
            f"hard_reject={rejected['hard_reject']}, "
            f"low_value={rejected['low_value']}, "
            f"no_sources={rejected['no_sources']}."
        )

        ranked.sort(
            key=lambda x: x.get("raw_score", 0),
            reverse=True,
        )

        # Баланс категорій: після 3-4 матеріалів однієї теми наступному стає
        # трохи важче. Manual не караємо, бо воно вже обране адміністратором.
        category_counts: Dict[str, int] = {}

        for ev in ranked:
            if ev.get("is_priority"):
                ev["balanced_score"] = ev["raw_score"]
                continue

            category = ev["category"]
            current = category_counts.get(category, 0)

            penalty = (
                12
                if current >= 4
                else (
                    6
                    if current >= 3
                    else 0
                )
            )

            diversity_bonus = 0.0

            if ev.get("curiosity", 0) >= 82:
                diversity_bonus += 2.5

            if ev.get("practical_value", 0) >= 82:
                diversity_bonus += 2.5

            # Справжній discovery-кандидат отримує маленький бонус доступу до
            # candidate pool. Квоту у фіналі все одно контролює окремий mix.
            if (
                ev.get("digest_role") == "discovery"
                and ev.get("discovery_qualified")
            ):
                diversity_bonus += 2.0

            ev["balanced_score"] = round(
                ev["raw_score"]
                - penalty
                + diversity_bonus,
                2,
            )

            category_counts[category] = current + 1

        ranked.sort(
            key=lambda x: x.get("balanced_score", 0),
            reverse=True,
        )

        return ranked

    def _generate_final_digest(
        self,
        events: List[Dict[str, Any]],
        posts: List[Dict[str, Any]],
        past_events: Optional[
            Union[
                List[str],
                List[Dict[str, str]],
            ]
        ],
        max_count: int,
        max_retries: int,
    ) -> List[Dict[str, Any]]:
        event_blocks = []

        for ev in events:
            factual_id = ev["best_factual_source_id"]
            media_id = ev.get("best_media_source_id")
            factual_post = posts[factual_id]

            media_description = "немає"
            if isinstance(media_id, int):
                if posts[media_id].get("has_video"):
                    media_description = (
                        "відео з Telegram-поста цієї події"
                    )
                elif posts[media_id].get("has_media"):
                    media_description = (
                        "фото з Telegram-поста цієї події"
                    )

            key_facts = ev.get("key_facts", [])
            key_facts_text = (
                "; ".join(
                    str(x)
                    for x in key_facts[:6]
                )
                if isinstance(key_facts, list)
                else str(key_facts)
            )

            priority_flag = (
                " ⭐ ПРІОРИТЕТ АДМІНІСТРАТОРА"
                if ev.get("is_priority")
                else ""
            )

            repeat_info = (
                "так, але є значущий новий розвиток"
                if ev.get("is_history_repeat")
                else "ні"
            )

            event_blocks.append(
                "=== EVENT_ID: "
                f"{ev.get('event_id')}"
                f"{priority_flag} ===\n"
                "ТИП: "
                f"{ev.get('event_type', 'other')}\n"
                "КАТЕГОРІЯ: "
                f"{ev.get('category', 'other')}\n"
                "РОЛЬ У ДАЙДЖЕСТІ: "
                f"{self._event_digest_role(ev)}\n"
                "IMPORTANCE: "
                f"{ev.get('importance', 0)}\n"
                "PUBLIC_INTEREST: "
                f"{ev.get('public_interest', 0)}\n"
                "NOVELTY: "
                f"{ev.get('novelty', 0)}\n"
                "CURIOSITY: "
                f"{ev.get('curiosity', 0)}\n"
                "PRACTICAL_VALUE: "
                f"{ev.get('practical_value', 0)}\n"
                "URGENCY: "
                f"{ev.get('urgency', 0)}\n"
                "ПОВТОР ІСТОРІЇ: "
                f"{repeat_info}\n"
                "СИЛА НОВОГО РОЗВИТКУ: "
                f"{ev.get('history_update_strength', 0)}\n"
                "МЕДІА: "
                f"{media_description}\n"
                "СУТЬ: "
                f"{ev.get('summary', '')}\n"
                "ЧОМУ ВАЖЛИВО/ЦІКАВО: "
                f"{ev.get('why_it_matters', '')}\n"
                "КЛЮЧОВІ ФАКТИ: "
                f"{key_facts_text}\n"
                "ТЕКСТ ДЖЕРЕЛА: "
                f"{str(factual_post.get('text') or '')[:self.MAX_EVENT_SOURCE_CHARS]}\n"
            )

        history_block = self._build_history_block(
            past_events
        )

        prompt = f"""
Ти — головний редактор українського новинного Telegram-каналу.

Сформуй фінальний дайджест із найважливіших,
найцікавіших і найактуальніших подій.
Максимум: {max_count} новин.

Якщо є достатньо якісних кандидатів, бажано сформувати 7-10 новин.
Не потрібно штучно набирати {max_count}, якщо кандидат справді слабкий.

ВАЖЛИВА РЕДАКЦІЙНА СТРУКТУРА:
Кандидати мають роль CORE або DISCOVERY.

CORE — головні/важкі новини. DISCOVERY — якісні цікаві або практично
корисні події, які органічно завершують випуск.

ПРАВИЛО КІЛЬКОСТІ ДЛЯ ЛІМІТУ 10:
- якщо є 10 сильних CORE — бери 10 CORE і НЕ додавай discovery;
- якщо є 9 CORE — додай 1 discovery;
- якщо є 8 CORE — додай 1-2 discovery;
- якщо є 7 CORE — додай 1-3 discovery;
- якщо є 5-6 CORE — залиш їх ядром і додай до 3 discovery.

Не витісняй справді важливу десяту CORE-новину цікавинкою.
Але коли після важких новин є вільні місця, не заповнюй їх слабкою
однотипною hard-news подією, якщо є якісна DISCOVERY.

УСІ DISCOVERY-НОВИНИ СТАВ У КІНЦІ ДАЙДЖЕСТУ, після CORE.
Всередині CORE і DISCOVERY порядок визначай природно за важливістю/силою.

Події ⭐ ПРІОРИТЕТ АДМІНІСТРАТОРА обов'язково включи у фінальний список.
Manual priority сильніше за звичайну квоту: його не можна відкинути.
При цьому не треба механічно ставити всі manual-події на початок —
розташовуй їх органічно за змістом; discovery-manual теж іде в кінцевий
discovery-блок.

━━━━━━━━━━━━━━━━━━━━
АРХІВ:
{history_block}
━━━━━━━━━━━━━━━━━━━━

ВИМОГИ ДО ВИБОРУ:

1. Не повторюй одну реальну подію двічі.
2. Якщо подія вже була в архіві, включай її знову лише коли кандидат
   містить реально значущий новий розвиток.
3. Не додавай відверто слабку подію тільки для заповнення кількості.
4. Не вигадуй факти.
5. Не використовуй чутки.
6. Не оцінюй важливість за довжиною початкового Telegram-посту.
7. Коротка гаряча новина може бути однією з головних новин дайджесту.
8. Відео або фото саме по собі не робить слабку подію важливою.
9. Якщо значуща подія має реальне фото чи відео з місця — це плюс.
10. При близьких оцінках віддавай перевагу події,
    яка додає нову тему, корисність або цікавість,
    а не четвертій однотипній новині про вже представлену тему.
11. Високий CURIOSITY означає, що подія може зайняти 7-10 місце,
    навіть якщо її стратегічна IMPORTANCE нижча.
12. Високий PRACTICAL_VALUE означає, що подія корисна людям
    і теж може виправдано потрапити у фінальний список.

━━━━━━━━━━━━━━━━━━━━
ВИМОГИ ДО ТЕКСТУ:

Ти пишеш НЕ для сухого інформагентства,
а для сучасного короткого Telegram-дайджесту.

Читач має за 15-25 секунд:
1. зрозуміти, що сталося;
2. побачити найважливішу або найцікавішу деталь;
3. зрозуміти масштаб, наслідок або практичне значення;
4. отримати достатньо контексту, щоб новина не виглядала як обірваний факт.

СТИЛЬ:
- живий;
- природний;
- конкретний;
- компактний;
- інформаційний;
- без канцеляриту;
- без штучної сенсаційності.

Текст має читатися як хороша редакторська розповідь,
а не як список пунктів із пресрелізу.

Кожна новина повинна мати маленький природний "гачок":
сильну цифру, конкретну деталь, наслідок, контраст,
незвичайний факт або просте пояснення, чому це цікаво.

ГАЧОК НЕ ОЗНАЧАЄ КЛІКБЕЙТ.
Не перебільшуй і не домислюй.

━━━━━━━━━━━━━━━━━━━━
ДОВЖИНА:

Максимальна довжина однієї новини — {self.MAX_NEWS_CHARS} символів.

Бажана довжина — приблизно 450-800 символів разом із заголовком,
якщо кандидат містить достатньо підтверджених фактів.

Типово пиши 3-6 ЗАВЕРШЕНИХ речень.

Для простої гарячої події достатньо 2-3 речень.
Для змістовної новини з цифрами, контекстом або наслідками — 4-6 речень.

Не розтягуй матеріал, якщо фактів мало.
Краще 3 сильні речення, ніж 6 речень із водою.

Не роби речення надто довгими.
Частіше використовуй короткі або середні речення,
щоб пост легко читався зі смартфона.

━━━━━━━━━━━━━━━━━━━━
ЯК БУДУВАТИ НОВИНУ:

НЕ використовуй одну жорстку схему для всіх матеріалів.
Обирай найприродніший початок залежно від події.

Можна почати з:
- головного результату;
- найцікавішої деталі;
- сильної цифри;
- незвичайного факту;
- зміни, яка безпосередньо вплине на людей;
- короткого пояснення масштабу.

Якщо серед фактів є одна особливо цікава деталь,
не ховай її в останньому реченні — винеси ближче до початку.

Для війни та атак:
що сталося → головний наслідок → масштаб/місце → важливий контекст.

Для фронту:
що змінилося → де → який результат → чому це важливо.

Для технологій і науки:
що нового → чим це відрізняється → конкретна деталь/цифра →
чому це цікаво або що це може змінити, якщо це випливає з фактів.

Для економіки:
що змінилося → цифри → кого це зачепить → практичний наслідок.

Для суспільних новин:
що змінюється → як працюватиме → кого стосується →
що читачеві важливо запам'ятати.

Для міжнародних:
що сталося → ключова деталь → чому це має значення для України або світу.

Для українських виробництв/досягнень:
що запустили або створили → що саме вміють/виробляють →
масштаб або конкретика → чому це помітна зміна.

━━━━━━━━━━━━━━━━━━━━
ПРИКЛАД ПРИНЦИПУ СТИЛЮ:

СУХО:
"Підприємство налагодило виробництво артилерійських стволів.
Воно виконує замовлення BAE Systems. Калібр становить від 25 до 203 мм."

КРАЩЕ ЗА ЛОГІКОЮ:
"Український завод освоїв серійне виробництво артилерійських стволів —
від 25 до 203 мм. Підприємство вже виконує замовлення BAE Systems
на компоненти для західних артсистем. Це означає, що частину складного
виробництва для таких систем уже локалізують в Україні."

НЕ копіюй цей текст і НЕ додавай висновків,
якщо їх немає у фактах кандидата.
Це лише приклад того, як зробити подачу природнішою.

━━━━━━━━━━━━━━━━━━━━
ЗАГОЛОВОК:

4-10 слів.

Він повинен бути:
- конкретним;
- зрозумілим без читання тексту;
- трохи цікавішим за канцелярський заголовок;
- без клікбейту;
- без порожніх формулювань.

Добре:
"Київ змінює правила роботи під час тривог"
"Україна запускає виробництво стволів для західної артилерії"
"Новий тариф на воду може змінити платіжки киян"

Погано:
"Стало відомо про важливе рішення"
"Нові подробиці ситуації"
"В Україні відбулася важлива подія"

Заголовок не повинен дослівно повторювати перше речення.

━━━━━━━━━━━━━━━━━━━━
ВАЖЛИВО:

- ніколи не обривай останнє речення;
- ніколи не завершуй новину на півслові;
- не додавай фактів, яких немає у кандидатові;
- не роби власних прогнозів;
- не приписуй причин, яких джерело не підтверджує;
- якщо текст виходить задовгим, скороти другорядні деталі;
- кожне речення повинно або додавати факт,
  або пояснювати значення вже наведеного факту;
- не повторюй один і той самий факт різними словами;
- не використовуй сухий стиль протоколу;
- не використовуй надмірно емоційні формулювання.

НЕ ВИКОРИСТОВУЙ шаблони:

"Стало відомо..."
"Повідомляється, що..."
"Наразі відомо..."
"Як зазначають..."
"За інформацією джерел..."
"Ситуація залишається..."
"Варто зазначити..."
"Нагадаємо, що..." — якщо це не справді необхідний контекст.

Не вставляй технічні маркери:
[ФОТО]
[ВІДЕО]
[ТЕКСТ]

ФОРМАТ:

ОДИН тематичний емодзі + <b>Заголовок</b>

порожній рядок

2-6 завершених природних речень.

ПЕРЕД ВІДПОВІДДЮ ПЕРЕВІР КОЖНУ НОВИНУ:

1. Чи не перевищує вона {self.MAX_NEWS_CHARS} символів?
2. Чи має вона достатньо контексту, а не лише сухий факт?
3. Чи завершене останнє речення?
4. Чи немає повторів і води?
5. Чи всі твердження походять із наданих фактів?
6. Чи не дублює вона іншу новину в цьому ж дайджесті?
7. Чи є в ній найцікавіша/найважливіша конкретна деталь кандидата?
8. Чи звучить текст природно українською?
9. Чи не став він клікбейтним?
10. Чи випуск загалом не перевантажений однією категорією,
    якщо є якісні альтернативи?

ВІДПОВІДЬ ТІЛЬКИ JSON:

{{
  "news": [
    {{
      "event_id": "E1",
      "text": "💥 <b>Короткий заголовок</b>\\n\\nПерше завершене речення. Друге завершене речення. Третє завершене речення."
    }}
  ]
}}

КАНДИДАТИ:
{chr(10).join(event_blocks)}
"""

        data = self._call_json_with_cascade(
            prompt,
            max_retries,
            "EDITOR",
            temperature=0.30,
        )

        raw_news = (
            data.get("news", [])
            if (
                data
                and isinstance(data.get("news"), list)
            )
            else []
        )

        event_map = {
            str(ev["event_id"]): ev
            for ev in events
            if ev.get("event_id")
        }

        final_list = []

        for item in raw_news:
            if not isinstance(item, dict):
                continue

            event_id = str(item.get("event_id") or "")
            text = item.get("text")

            if (
                event_id not in event_map
                or not isinstance(text, str)
                or not text.strip()
            ):
                continue

            ev = event_map[event_id]

            final_list.append({
                "event_id": event_id,
                "source_id": ev["best_source_id"],
                "source_ids": list(
                    ev.get("source_ids", [])
                ),
                "summary": ev.get("summary", ""),
                "category": ev.get("category", "other"),
                "digest_role": self._event_digest_role(ev),
                "is_discovery_candidate": bool(
                    ev.get("is_discovery_candidate")
                ),
                "is_priority": bool(ev.get("is_priority")),
                "text": text.strip(),
            })

        return final_list

    def _validate_final_news(
        self,
        news: List[Dict[str, Any]],
        ranked_events: List[Dict[str, Any]],
        posts: List[Dict[str, Any]],
        count: int,
    ) -> List[Dict[str, Any]]:
        validated = []
        used_event_ids = set()

        event_map = {
            str(ev.get("event_id")): ev
            for ev in ranked_events
            if ev.get("event_id")
        }

        for item in news:
            source_id = item.get("source_id")
            event_id = str(item.get("event_id") or "")
            text = item.get("text")

            if event_id not in event_map:
                continue

            if (
                not isinstance(source_id, int)
                or not (0 <= source_id < len(posts))
            ):
                continue

            if event_id in used_event_ids:
                continue

            if not isinstance(text, str) or not text.strip():
                continue

            text = self._clean_generated_news_text(text)
            if not text:
                continue

            ev = event_map[event_id]

            validated.append({
                "event_id": event_id,
                "source_id": source_id,
                "source_ids": list(ev.get("source_ids", [])),
                "text": text,
                "summary": item.get(
                    "summary",
                    ev.get("summary", ""),
                ),
                "category": item.get(
                    "category",
                    ev.get("category", "other"),
                ),
                "digest_role": self._event_digest_role(ev),
                "is_discovery_candidate": bool(
                    ev.get("is_discovery_candidate")
                ),
                "is_priority": bool(ev.get("is_priority")),
            })

            used_event_ids.add(event_id)

            if len(validated) >= count:
                break

        return validated

    def _ensure_priority_news_in_final(
        self,
        validated: List[Dict[str, Any]],
        ranked_events: List[Dict[str, Any]],
        posts: List[Dict[str, Any]],
        count: int,
    ) -> List[Dict[str, Any]]:
        priority_events = [
            ev
            for ev in ranked_events
            if ev.get("is_priority")
        ]
        if not priority_events:
            return validated[:count]

        result = list(validated)
        used_event_ids = {
            str(item.get("event_id") or "")
            for item in result
            if item.get("event_id")
        }

        rank_index = {
            str(ev.get("event_id") or ""): idx
            for idx, ev in enumerate(ranked_events)
            if ev.get("event_id")
        }

        missing_events = [
            ev
            for ev in priority_events
            if str(ev.get("event_id") or "") not in used_event_ids
        ]

        if missing_events:
            logger.warning(
                "EDITOR пропустив %s priority-подій. "
                "Додаємо їх Python-fallback без повторного відбору.",
                len(missing_events),
            )

        for ev in missing_events:
            item = self._build_fallback_news_item(ev, posts)
            if not item:
                logger.error(
                    "Не вдалося побудувати fallback для priority event_id=%s",
                    ev.get("event_id"),
                )
                continue

            event_id = str(ev.get("event_id") or "")
            target_rank = rank_index.get(event_id, len(ranked_events))

            # Вставляємо приблизно відповідно до ranked-позиції,
            # не перебудовуючи весь порядок, який уже створив Editor.
            insert_at = len(result)
            for idx, existing in enumerate(result):
                existing_rank = rank_index.get(
                    str(existing.get("event_id") or ""),
                    len(ranked_events) + 100,
                )
                if existing_rank > target_rank:
                    insert_at = idx
                    break

            result.insert(insert_at, item)
            used_event_ids.add(event_id)

        # Якщо через обов'язкові manual-події перевищили count,
        # прибираємо найслабші NON-priority, а не manual.
        while len(result) > count:
            removable_indexes = [
                idx
                for idx, item in enumerate(result)
                if not item.get("is_priority")
            ]

            if not removable_indexes:
                # count вже має бути >= кількості priority, але не ріжемо
                # manual навіть якщо зовнішній код передав некоректний ліміт.
                break

            worst_idx = max(
                removable_indexes,
                key=lambda idx: rank_index.get(
                    str(result[idx].get("event_id") or ""),
                    len(ranked_events) + 1000,
                ),
            )
            result.pop(worst_idx)

        final_ids = {
            str(item.get("event_id") or "")
            for item in result
        }
        missing_after_guard = [
            str(ev.get("event_id") or "")
            for ev in priority_events
            if str(ev.get("event_id") or "") not in final_ids
        ]

        if missing_after_guard:
            logger.error(
                "CRITICAL final priority guarantee failed for event_ids=%s",
                missing_after_guard,
            )

        return result[:max(count, len(priority_events))]

    def _build_fallback_news_item(
        self,
        ev: Dict[str, Any],
        posts: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        source_id = ev.get("best_source_id")
        if (
            not isinstance(source_id, int)
            or not (0 <= source_id < len(posts))
        ):
            source_ids = self._valid_source_ids(
                ev.get("source_ids"),
                posts,
            )
            if not source_ids:
                return None
            source_id = source_ids[0]

        headline = (
            str(ev.get("headline_hint") or "").strip()
            or self._priority_headline_from_text(
                posts[source_id].get("text") or ""
            )
            or "Важлива подія"
        )

        category = ev.get("category", "other")
        if category not in self.ALLOWED_CATEGORIES:
            category = "other"

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
        emoji = emoji_map.get(category, "📰")

        key_facts = ev.get("key_facts", [])
        facts = (
            [
                str(x).strip()
                for x in key_facts
                if str(x).strip()
            ]
            if isinstance(key_facts, list)
            else []
        )

        summary = str(ev.get("summary") or "").strip()
        why = str(ev.get("why_it_matters") or "").strip()

        sentences: List[str] = []

        if summary:
            sentences.append(
                self._ensure_sentence_end(summary)
            )

        for fact in facts[:5]:
            sentence = self._ensure_sentence_end(fact)
            if sentence and sentence not in sentences:
                sentences.append(sentence)

        if why:
            sentence = self._ensure_sentence_end(why)
            if sentence and sentence not in sentences:
                sentences.append(sentence)

        original_text = posts[source_id].get("text") or ""
        if len(sentences) < 3 and original_text:
            for sentence in self._extract_sentences(original_text):
                clean_sentence = self._ensure_sentence_end(sentence)
                if (
                    clean_sentence
                    and clean_sentence not in sentences
                ):
                    sentences.append(clean_sentence)
                if len(sentences) >= 5:
                    break

        # Для manual гарантія важливіша за ідеальну кількість речень.
        # Якщо текст дуже короткий, дозволяємо один завершений факт.
        if not sentences and original_text.strip():
            sentences.append(
                self._ensure_sentence_end(original_text.strip())
            )

        if not sentences:
            return None

        text = (
            f"{emoji} <b>{headline}</b>\n\n"
            f"{' '.join(sentences[:6])}"
        )
        text = self._clean_generated_news_text(text)
        if not text:
            return None

        return {
            "event_id": str(ev.get("event_id") or ""),
            "source_id": source_id,
            "source_ids": list(ev.get("source_ids", [])),
            "text": text,
            "summary": summary,
            "category": category,
            "digest_role": self._event_digest_role(ev),
            "is_discovery_candidate": bool(
                ev.get("is_discovery_candidate")
            ),
            "is_priority": bool(ev.get("is_priority")),
        }

    def _enforce_digest_mix(
        self,
        validated: List[Dict[str, Any]],
        ranked_events: List[Dict[str, Any]],
        posts: List[Dict[str, Any]],
        count: int,
    ) -> List[Dict[str, Any]]:
        """
        Фінальна Python-гарантія структури випуску.

        Вона НЕ переоцінює факти і НЕ створює нові події. Вона лише бере
        події, які вже пройшли Analyzer + history gate + ranking, і розкладає
        їх у редакційну структуру:

        10 core -> 10 core + 0 discovery
         9 core ->  9 core + 1 discovery
         8 core ->  8 core + до 2 discovery
         7 core ->  7 core + до 3 discovery
        <=6 core -> core + до 3 discovery

        Manual priority є абсолютним override: якщо ручна discovery-подія
        конфліктує з десятою non-priority core, ручна подія лишається.
        """
        if count <= 0 or not ranked_events:
            return []

        event_map = {
            str(ev.get("event_id") or ""): ev
            for ev in ranked_events
            if ev.get("event_id")
        }

        polished_map = {
            str(item.get("event_id") or ""): item
            for item in validated
            if (
                item.get("event_id")
                and str(item.get("event_id") or "") in event_map
            )
        }

        core_events = [
            ev
            for ev in ranked_events
            if self._event_digest_role(ev) == "core"
        ]
        discovery_events = [
            ev
            for ev in ranked_events
            if self._is_publishable_discovery(ev)
        ]

        core_events.sort(
            key=self._core_presentation_score,
            reverse=True,
        )
        discovery_events.sort(
            key=self._discovery_sort_score,
            reverse=True,
        )

        priority_core = [
            ev for ev in core_events
            if ev.get("is_priority")
        ]
        priority_discovery = [
            ev for ev in discovery_events
            if ev.get("is_priority")
        ]

        discovery_slots = self._desired_discovery_slots(
            len(core_events),
            count,
        )

        # Звичайна квота. Якщо місця є і discovery-кандидати пройшли quality
        # gate, намагаємось використати всі дозволені 1-3 місця.
        desired_discovery = min(
            discovery_slots,
            len(discovery_events),
        )

        # Manual discovery не можна відкинути навіть коли є 10 core.
        desired_discovery = max(
            desired_discovery,
            len(priority_discovery),
        )

        desired_core = min(
            len(core_events),
            max(0, count - desired_discovery),
        )
        desired_core = max(
            desired_core,
            len(priority_core),
        )

        # Якщо mandatory manual змінив стандартну квоту, прибираємо спочатку
        # non-priority core/discovery. Самі manual не ріжемо.
        while desired_core + desired_discovery > count:
            if desired_core > len(priority_core):
                desired_core -= 1
                continue
            if desired_discovery > len(priority_discovery):
                desired_discovery -= 1
                continue
            break

        mandatory_total = (
            len(priority_core) + len(priority_discovery)
        )
        if mandatory_total > count:
            # select_top_distinct_news зазвичай уже розширив effective_count,
            # але тут лишаємо останню страховку.
            logger.warning(
                "Digest mix: priority=%s перевищує count=%s. "
                "Manual не обрізаємо.",
                mandatory_total,
                count,
            )
            count = mandatory_total
            desired_core = max(desired_core, len(priority_core))
            desired_discovery = max(
                desired_discovery,
                len(priority_discovery),
            )

        def select_with_priority(
            pool: List[Dict[str, Any]],
            target: int,
        ) -> List[Dict[str, Any]]:
            if target <= 0:
                return []

            selected: List[Dict[str, Any]] = []
            selected_ids = set()

            # Спочатку резервуємо всі manual, але фінальний порядок нижче
            # знову визначатиметься змістовним score, а не priority bonus.
            for ev in pool:
                if not ev.get("is_priority"):
                    continue
                event_id = str(ev.get("event_id") or "")
                if not event_id or event_id in selected_ids:
                    continue
                selected.append(ev)
                selected_ids.add(event_id)

            for ev in pool:
                if len(selected) >= target:
                    break
                event_id = str(ev.get("event_id") or "")
                if not event_id or event_id in selected_ids:
                    continue
                selected.append(ev)
                selected_ids.add(event_id)

            return selected

        selected_core = select_with_priority(
            core_events,
            desired_core,
        )
        selected_discovery = select_with_priority(
            discovery_events,
            desired_discovery,
        )

        # Після priority-reserve кількість може бути > target лише у випадку,
        # коли mandatory manual більше за стандартну квоту. Це очікувано.
        selected_core.sort(
            key=self._core_presentation_score,
            reverse=True,
        )
        selected_discovery.sort(
            key=self._discovery_sort_score,
            reverse=True,
        )

        final_items: List[Dict[str, Any]] = []
        used_ids = set()

        def materialize(ev: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            event_id = str(ev.get("event_id") or "")
            if not event_id or event_id in used_ids:
                return None

            item = polished_map.get(event_id)
            if item is not None:
                result_item = dict(item)
                result_item["digest_role"] = self._event_digest_role(ev)
                result_item["is_discovery_candidate"] = bool(
                    ev.get("is_discovery_candidate")
                )
                result_item["is_priority"] = bool(
                    ev.get("is_priority")
                )
            else:
                result_item = self._build_fallback_news_item(
                    ev,
                    posts,
                )

            if not result_item:
                return None

            used_ids.add(event_id)
            return result_item

        # CORE завжди йде першим.
        for ev in selected_core:
            item = materialize(ev)
            if item:
                final_items.append(item)

        core_materialized = len(final_items)

        # DISCOVERY завжди в кінці.
        for ev in selected_discovery:
            item = materialize(ev)
            if item:
                final_items.append(item)

        discovery_materialized = (
            len(final_items) - core_materialized
        )

        # У дуже рідкісному випадку fallback для обраної події не зібрався,
        # пробуємо наступного кандидата ТІЄЇ Ж ролі, не ламаючи структуру.
        if core_materialized < desired_core:
            for ev in core_events:
                if core_materialized >= desired_core:
                    break
                item = materialize(ev)
                if not item:
                    continue
                # Додаємо до кінця core-блоку, тобто перед discovery.
                final_items.insert(core_materialized, item)
                core_materialized += 1

        current_discovery = len(final_items) - core_materialized
        if current_discovery < desired_discovery:
            for ev in discovery_events:
                if current_discovery >= desired_discovery:
                    break
                item = materialize(ev)
                if not item:
                    continue
                final_items.append(item)
                current_discovery += 1

        # Остання manual-перевірка: жодна priority-подія не має загубитись
        # навіть через неочікувану помилку materialize/квоти.
        priority_events = [
            ev for ev in ranked_events
            if ev.get("is_priority")
        ]
        missing_priority = [
            ev
            for ev in priority_events
            if str(ev.get("event_id") or "") not in used_ids
        ]

        for ev in missing_priority:
            item = materialize(ev)
            if not item:
                logger.error(
                    "CRITICAL: не вдалося матеріалізувати priority event_id=%s",
                    ev.get("event_id"),
                )
                continue

            if self._event_digest_role(ev) == "discovery":
                final_items.append(item)
            else:
                # Core manual вставляємо перед discovery-блоком.
                insert_at = next(
                    (
                        idx
                        for idx, existing in enumerate(final_items)
                        if existing.get("digest_role") == "discovery"
                    ),
                    len(final_items),
                )
                final_items.insert(insert_at, item)
                core_materialized += 1

        # Якщо mandatory manual спричинив перевищення count, видаляємо
        # найслабші NON-priority, починаючи з ролі, де є надлишок.
        while len(final_items) > count:
            removable = [
                (idx, item)
                for idx, item in enumerate(final_items)
                if not item.get("is_priority")
            ]
            if not removable:
                break

            # Віддаємо перевагу видаленню найслабшого core, якщо discovery
            # mandatory; інакше просто найслабшого за content score.
            def removal_score(entry):
                idx, item = entry
                ev = event_map.get(str(item.get("event_id") or ""), {})
                if self._event_digest_role(ev) == "discovery":
                    return self._discovery_sort_score(ev)
                return self._core_presentation_score(ev)

            worst_idx, _ = min(removable, key=removal_score)
            final_items.pop(worst_idx)

        # Після можливого trim ще раз стабілізуємо порядок: core -> discovery.
        core_items = []
        discovery_items = []
        for item in final_items:
            ev = event_map.get(
                str(item.get("event_id") or ""),
                {},
            )
            if self._event_digest_role(ev) == "discovery":
                item["digest_role"] = "discovery"
                discovery_items.append(item)
            else:
                item["digest_role"] = "core"
                core_items.append(item)

        core_items.sort(
            key=lambda item: self._core_presentation_score(
                event_map.get(str(item.get("event_id") or ""), {})
            ),
            reverse=True,
        )
        discovery_items.sort(
            key=lambda item: self._discovery_sort_score(
                event_map.get(str(item.get("event_id") or ""), {})
            ),
            reverse=True,
        )

        final_items = core_items + discovery_items

        final_priority_ids = {
            str(item.get("event_id") or "")
            for item in final_items
            if item.get("is_priority")
        }
        expected_priority_ids = {
            str(ev.get("event_id") or "")
            for ev in priority_events
        }
        lost_priority = sorted(
            expected_priority_ids - final_priority_ids
        )
        if lost_priority:
            logger.error(
                "CRITICAL final digest mix lost priority event_ids=%s",
                lost_priority,
            )

        logger.info(
            "Digest mix: available core=%s, discovery=%s; "
            "selected core=%s, discovery=%s; total=%s.",
            len(core_events),
            len(discovery_events),
            len(core_items),
            len(discovery_items),
            len(final_items),
        )

        for idx, item in enumerate(final_items, start=1):
            logger.info(
                "FINAL #%s role=%s priority=%s event_id=%s",
                idx,
                item.get("digest_role", "core"),
                bool(item.get("is_priority")),
                item.get("event_id"),
            )

        return final_items[:count]

    def _fill_missing_news(
        self,
        validated: List[Dict[str, Any]],
        ranked_events: List[Dict[str, Any]],
        posts: List[Dict[str, Any]],
        count: int,
    ) -> List[Dict[str, Any]]:
        result = list(validated)

        used_event_ids = {
            str(item.get("event_id") or "")
            for item in result
            if item.get("event_id")
        }

        for ev in ranked_events:
            if len(result) >= count:
                break

            event_id = str(ev.get("event_id") or "")
            if not event_id or event_id in used_event_ids:
                continue

            item = self._build_fallback_news_item(ev, posts)
            if not item:
                continue

            result.append(item)
            used_event_ids.add(event_id)

        return result[:count]

    def _select_factual_source(
        self,
        source_ids: List[int],
        posts: List[Dict[str, Any]],
        preferred_id: Any = None,
    ) -> int:
        if (
            isinstance(preferred_id, int)
            and preferred_id in source_ids
        ):
            return preferred_id

        return max(
            source_ids,
            key=lambda s: self._factual_source_score(
                posts[s]
            ),
        )

    def _select_media_source(
        self,
        source_ids: List[int],
        posts: List[Dict[str, Any]],
        preferred_id: Any = None,
    ) -> Optional[int]:
        media_ids = [
            s
            for s in source_ids
            if (
                posts[s].get("has_video")
                or posts[s].get("has_media")
            )
        ]

        if not media_ids:
            return None

        if (
            isinstance(preferred_id, int)
            and preferred_id in media_ids
        ):
            return preferred_id

        return max(
            media_ids,
            key=lambda s: self._media_source_score(
                posts[s]
            ),
        )

    def _factual_source_score(
        self,
        post: Dict[str, Any],
    ) -> float:
        if post.get("is_priority"):
            return 10000.0

        username = (
            str(post.get("channel_username", "") or "")
            .replace("@", "")
            .strip()
        )

        views = int(post.get("views") or 0)
        forwards = int(post.get("forwards") or 0)
        text_length = len(post.get("text") or "")

        # Довжина тут не оцінює важливість події.
        # Вона лише допомагає вибрати інформативніше джерело.
        score = (
            min(
                math.log10(max(views, 1)) * 5,
                25,
            )
            + min(
                math.log10(max(forwards, 1)) * 3,
                10,
            )
            + min(text_length / 180, 7)
        )

        return score * self._get_source_multiplier(
            username
        )

    @staticmethod
    def _media_source_score(
        post: Dict[str, Any],
    ) -> float:
        score = (
            40
            if post.get("has_video")
            else (
                20
                if post.get("has_media")
                else 0
            )
        )

        views = int(post.get("views") or 0)
        forwards = int(post.get("forwards") or 0)

        score += min(
            math.log10(max(views, 1)) * 3,
            18,
        )

        score += min(
            math.log10(max(forwards, 1)) * 2,
            8,
        )

        return score

    def _event_source_multiplier(
        self,
        source_ids: List[int],
        posts: List[Dict[str, Any]],
    ) -> float:
        multipliers = []

        for source_id in source_ids:
            username = (
                str(
                    posts[source_id].get(
                        "channel_username",
                        "",
                    )
                    or ""
                )
                .replace("@", "")
                .strip()
            )

            multipliers.append(
                self._get_source_multiplier(username)
            )

        # Не караємо подію за те, що поряд із сильним джерелом
        # її перепостив слабший агрегатор.
        return max(multipliers) if multipliers else 1.0

    @staticmethod
    def _get_source_multiplier(
        username: str,
    ) -> float:
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
        op_name: str,
        temperature: float = 0.15,
    ) -> Optional[Dict[str, Any]]:
        max_retries = max(1, int(max_retries or 1))

        for model in self.models_priority:
            for attempt in range(1, max_retries + 1):
                retry_hint = ""
                if attempt > 1:
                    retry_hint = (
                        "\n\nКРИТИЧНО: попередня відповідь не пройшла "
                        "машинний JSON-парсер. Поверни ЛИШЕ один валідний "
                        "JSON-об'єкт: подвійні лапки для ключів і рядків, "
                        "без trailing commas, без Markdown і без пояснень."
                    )

                try:
                    logger.info(
                        f"{op_name}: спроба "
                        f"{attempt}/{max_retries} "
                        f"через {model} "
                        f"(temperature={temperature})"
                    )

                    response = self.client.models.generate_content(
                        model=model,
                        contents=prompt + retry_hint,
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            temperature=temperature,
                        ),
                    )

                    raw_text = self._clean_json_response(
                        (response.text or "").strip()
                    )

                    if not raw_text:
                        raise ValueError("Модель повернула порожню відповідь")

                    try:
                        data = json.loads(raw_text)
                    except json.JSONDecodeError as json_error:
                        logger.warning(
                            "%s: невалідний JSON від %s, спроба %s/%s: %s",
                            op_name,
                            model,
                            attempt,
                            max_retries,
                            json_error,
                        )

                        if attempt < max_retries:
                            time.sleep(min(2 * attempt, 4))
                            continue

                        # Після вичерпання спроб цього model переходимо
                        # до наступного model у cascade.
                        break

                    if isinstance(data, dict):
                        return data

                    logger.warning(
                        "%s: %s повернув JSON типу %s замість object.",
                        op_name,
                        model,
                        type(data).__name__,
                    )

                    if attempt < max_retries:
                        time.sleep(min(attempt, 2))
                        continue

                    break

                except Exception as e:
                    err = str(e)

                    transient_error = any(
                        x in err
                        for x in [
                            "503",
                            "429",
                            "UNAVAILABLE",
                            "ResourceExhausted",
                        ]
                    )

                    model_unavailable = any(
                        x in err
                        for x in [
                            "NOT_FOUND",
                            "404",
                        ]
                    )

                    if transient_error:
                        logger.warning(
                            "%s: тимчасова помилка моделі %s, спроба %s/%s: %s",
                            op_name,
                            model,
                            attempt,
                            max_retries,
                            e,
                        )
                        if attempt < max_retries:
                            time.sleep(3 * attempt)
                            continue
                        break

                    if model_unavailable:
                        logger.warning(
                            "%s: модель %s недоступна: %s. "
                            "Переходимо до наступної.",
                            op_name,
                            model,
                            e,
                        )
                        break

                    logger.error(
                        "Помилка "
                        f"{op_name} "
                        f"({model}), спроба {attempt}/{max_retries}: {e}"
                    )

                    # Для неочікуваної локальної/SDK помилки одна повторна
                    # спроба теж корисна. Раніше тут був break уже після 1/2.
                    if attempt < max_retries:
                        time.sleep(min(2 * attempt, 4))
                        continue

                    break

        return None

    def _build_history_block(
        self,
        past_events: Optional[
            Union[
                List[str],
                List[Dict[str, str]],
            ]
        ],
    ) -> str:
        if not past_events:
            return "Історія опублікованих подій порожня."

        lines = []

        for item in past_events[:self.HISTORY_LIMIT]:
            if isinstance(item, dict):
                title = (item.get("title") or "").strip()
                summary = (item.get("summary") or "").strip()
                published_at = (
                    item.get("published_at") or ""
                ).strip()

                if title or summary:
                    time_info = (
                        f" [{published_at}]"
                        if published_at
                        else ""
                    )

                    desc = (
                        f" — {summary}"
                        if summary
                        else ""
                    )

                    lines.append(
                        f"- {title}{desc}{time_info}"
                    )

            elif isinstance(item, str) and item.strip():
                lines.append(f"- {item.strip()}")

        return (
            "\n".join(lines)
            if lines
            else "Історія опублікованих подій порожня."
        )

    def _clean_generated_news_text(
        self,
        text: str,
    ) -> str:
        text = text.strip()

        text = re.sub(
            r"\[(?:ФОТО|ВІДЕО|ТЕКСТ|PHOTO|VIDEO|TEXT)\]\s*",
            "",
            text,
            flags=re.IGNORECASE,
        )

        text = re.sub(
            r"(?:ФОТО|ВІДЕО|ТЕКСТ):\s*",
            "",
            text,
            flags=re.IGNORECASE,
        )

        text = re.sub(
            r"\*\*(.*?)\*\*",
            r"<b>\1</b>",
            text,
        )

        text = (
            text
            .replace("📍", "")
            .replace("📌", "")
        )

        text = re.sub(
            r"\n{3,}",
            "\n\n",
            text,
        )

        if "<b>" not in text or "</b>" not in text:
            lines = text.split("\n", 1)
            first_line = lines[0].strip()
            rest = (
                "\n" + lines[1]
                if len(lines) > 1
                else ""
            )

            text = f"<b>{first_line}</b>{rest}"

        if len(text) > self.MAX_NEWS_CHARS:
            text = self._truncate_to_complete_sentence(
                text,
                self.MAX_NEWS_CHARS,
            )

        if "<b>" in text and "</b>" not in text:
            text += "</b>"

        return text.strip()

    @staticmethod
    def _truncate_to_complete_sentence(
        text: str,
        max_chars: int,
    ) -> str:
        if len(text) <= max_chars:
            return text.strip()

        candidate = text[:max_chars].rstrip()

        sentence_endings = [
            match.end()
            for match in re.finditer(
                r"[.!?](?=\s|$)",
                candidate,
            )
        ]

        if sentence_endings:
            safe_endings = [
                pos
                for pos in sentence_endings
                if pos >= int(max_chars * 0.55)
            ]

            if safe_endings:
                return candidate[:safe_endings[-1]].strip()

        last_space = candidate.rfind(" ")

        if last_space > int(max_chars * 0.70):
            candidate = candidate[:last_space].rstrip()

        return candidate.rstrip(",;:- ") + "…"

    @staticmethod
    def _extract_sentences(
        text: str,
    ) -> List[str]:
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return []

        parts = re.split(
            r"(?<=[.!?])\s+",
            text,
        )

        return [
            part.strip()
            for part in parts
            if len(part.strip()) >= 12
        ]

    @staticmethod
    def _ensure_sentence_end(
        text: str,
    ) -> str:
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return ""

        if text[-1] not in ".!?":
            text += "."

        return text

    @staticmethod
    def _safe_score(
        value: Any,
    ) -> float:
        try:
            return max(
                0.0,
                min(100.0, float(value)),
            )
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _clean_json_response(
        text: str,
    ) -> str:
        text = str(text or "").strip()

        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]

        if text.endswith("```"):
            text = text[:-3]

        text = text.strip()

        # Якщо модель все ж додала короткий префікс/суфікс,
        # беремо тільки зовнішній JSON object.
        first_brace = text.find("{")
        last_brace = text.rfind("}")
        if (
            first_brace >= 0
            and last_brace > first_brace
        ):
            text = text[first_brace:last_brace + 1]

        # Консервативно виправляємо лише найтиповіший дефект —
        # trailing comma перед } або ]. Інші синтаксичні помилки
        # не вгадуємо, а віддаємо на retry тієї ж моделі.
        text = re.sub(
            r",\s*([}\]])",
            r"\1",
            text,
        )

        return text.strip()
