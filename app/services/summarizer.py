import json
import logging
import math
import re
import time
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
    MIN_DIGEST_COUNT = 5
    EDITOR_CANDIDATES = 30
    HISTORY_LIMIT = 150

    MAX_INPUT_CHARS = 55000
    MAX_EVENT_SOURCE_CHARS = 2500
    MAX_NEWS_CHARS = 650

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

        posts_context = self._build_posts_context(
            posts
        )

        if not posts_context:
            return []

        analyzed_events = self._analyze_events(
            posts_context,
            past_events,
            max_retries_per_model,
        )

        if not analyzed_events:
            logger.warning(
                "Analyzer не повернув подій."
            )
            return []

        logger.info(
            "Analyzer знайшов "
            f"{len(analyzed_events)} потенційних подій."
        )

        ranked_events = self._rank_events(
            analyzed_events,
            posts,
        )

        if not ranked_events:
            logger.warning(
                "Після editorial gate "
                "не залишилось подій."
            )
            return []

        logger.info(
            "Після ranking залишилось "
            f"{len(ranked_events)} подій."
        )

        for idx, event in enumerate(
            ranked_events[:10],
            start=1,
        ):
            logger.info(
                "RANK #%s: %.2f | %s | %s",
                idx,
                float(
                    event.get(
                        "balanced_score",
                        event.get("raw_score", 0),
                    )
                    or 0
                ),
                event.get("event_type", "other"),
                event.get(
                    "headline_hint",
                    event.get("summary", ""),
                ),
            )

        editor_events = ranked_events[
            :self.EDITOR_CANDIDATES
        ]

        final_news = self._generate_final_digest(
            editor_events,
            posts,
            past_events,
            count,
            max_retries_per_model,
        )

        validated = self._validate_final_news(
            final_news,
            ranked_events,
            posts,
            count,
        )

        if len(validated) < self.MIN_DIGEST_COUNT:
            logger.warning(
                "EDITOR сформував лише "
                f"{len(validated)} новин. "
                "Fallback до мінімуму "
                f"{self.MIN_DIGEST_COUNT}."
            )

            validated = self._fill_missing_news(
                validated,
                ranked_events,
                posts,
                min(
                    count,
                    self.MIN_DIGEST_COUNT,
                ),
            )

        logger.info(
            "Фінальний дайджест: "
            f"{len(validated)} новин."
        )

        return validated[:count]

    def _build_posts_context(
        self,
        posts: List[Dict[str, Any]],
    ) -> str:
        prepared = []
        now_utc = datetime.now(timezone.utc)

        for idx, post in enumerate(posts):
            text = (
                post.get("text")
                or ""
            ).strip()

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
                str(
                    post.get(
                        "channel_username",
                        "",
                    )
                    or ""
                )
                .replace("@", "")
                .strip()
            )

            views = int(
                post.get("views") or 0
            )
            forwards = int(
                post.get("forwards") or 0
            )
            replies = int(
                post.get("replies") or 0
            )
            is_priority = bool(
                post.get("is_priority")
            )

            post_date = post.get("date")
            age_minutes: Optional[float] = None
            published_at = "невідомо"

            if isinstance(
                post_date,
                datetime,
            ):
                if post_date.tzinfo is None:
                    post_date = post_date.replace(
                        tzinfo=timezone.utc
                    )

                post_date_utc = post_date.astimezone(
                    timezone.utc
                )

                published_at = (
                    post_date_utc.strftime(
                        "%Y-%m-%d %H:%M UTC"
                    )
                )

                age_minutes = max(
                    0.0,
                    (
                        now_utc - post_date_utc
                    ).total_seconds()
                    / 60.0,
                )

            tier_mult = (
                self._get_source_multiplier(
                    channel_username
                )
            )

            engagement_score = (
                min(
                    math.log10(
                        max(views, 1)
                    )
                    * 4,
                    26,
                )
                + min(
                    math.log10(
                        max(forwards, 1)
                    )
                    * 3,
                    12,
                )
                + min(
                    math.log10(
                        max(replies, 1)
                    )
                    * 2,
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

            # Нові пости ще не встигли набрати перегляди.
            # Тому freshness використовується лише для потрапляння
            # в контекст Analyzer, а не як автоматичний доказ важливості.
            freshness_bonus = 0.0

            if age_minutes is not None:
                freshness_bonus = max(
                    0.0,
                    12.0
                    * (
                        1.0
                        - min(
                            age_minutes,
                            240.0,
                        )
                        / 240.0
                    ),
                )

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

            if (
                current_length
                + len(block)
                > self.MAX_INPUT_CHARS
            ):
                continue

            result.append(block)
            current_length += (
                len(block) + 10
            )

        logger.info(
            "У контекст Analyzer потрапило "
            f"{len(result)} з {len(prepared)} постів "
            f"({current_length} символів)."
        )

        return "\n\n---\n\n".join(
            result
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
        history_block = (
            self._build_history_block(
                past_events
            )
        )

        prompt = f"""
Ти — старший редактор загальноукраїнського новинного Telegram-дайджесту.

ТВОЯ ЗАДАЧА:
Із потоку Telegram-повідомлень знайти події, які реально заслуговують
на місце серед головних та найцікавіших новин останніх 4 годин.

Це НЕ звичайна стрічка новин і НЕ збір усіх повідомлень.

Для кожної події запитай:
"Чи варто знати це людині, яка прочитає лише 5-10 головних новин?"

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

ЗАЗВИЧАЙ ВІДКИДАЙ:

- рутинні обстріли без суттєвих наслідків;
- локальні пошкодження без ширшого значення;
- 1-2 поранених без інших значних факторів;
- тривоги;
- рух БпЛА;
- загрози ракет без підтверджених наслідків;
- дрібні ДТП;
- локальні побутові пожежі;
- дрібний кримінал;
- комунальні аварії;
- заяви політиків без реального рішення;
- повтори старих новин;
- чутки.

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

НЕ знижуй importance, novelty, public_interest або urgency
лише через малу довжину повідомлення.

Коротка новина може бути eligible_for_digest=true, якщо вона містить
самодостатній важливий факт, зокрема:

- підтверджене влучання;
- серйозні наслідки атаки;
- пожежу або пошкодження важливого об'єкта;
- удар по значному промисловому підприємству;
- удар по енергетичному, логістичному або військовому об'єкту;
- удар по великому комерційному об'єкту, якщо подія має
  помітний економічний, суспільний або новинний резонанс;
- незвичну або значущу ціль атаки;
- перші підтверджені наслідки великої події;
- важливий новий розвиток історії, яка відбувається прямо зараз.

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

При виборі між:
- довгою, але рутинною новиною;
- короткою, але новою, конкретною і важливою новиною;

віддавай перевагу інформаційній цінності, а не довжині тексту.

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
media_quality
national_relevance
urgency
history_update_strength

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

headline_hint — короткий заголовок.
key_facts — 2-5 найважливіших фактів.
why_it_matters — коротко, чому це важливо.
summary — стислий фактологічний опис.
rejection_reason — конкретна причина відхилення.
is_history_repeat — чи ця сама реальна подія вже є в архіві.

Якщо є ⭐ [ПРІОРИТЕТ АДМІНІСТРАТОРА]:
eligible_for_digest=true
importance=100
novelty=100
urgency=100

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
        )

        return (
            data.get("events", [])
            if (
                data
                and isinstance(
                    data.get("events"),
                    list,
                )
            )
            else []
        )

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
                    for s in ev.get(
                        "source_ids",
                        [],
                    )
                    if (
                        isinstance(s, int)
                        and 0 <= s < len(posts)
                    )
                ]

                if not src_ids:
                    rejected["no_sources"] += 1
                    continue

                is_priority = any(
                    posts[s].get(
                        "is_priority"
                    )
                    for s in src_ids
                )

                eligible = bool(
                    ev.get(
                        "eligible_for_digest",
                        False,
                    )
                )

                event_type = str(
                    ev.get(
                        "event_type"
                    )
                    or "other"
                )

                is_history_repeat = bool(
                    ev.get(
                        "is_history_repeat",
                        False,
                    )
                )

                history_update = self._safe_score(
                    ev.get(
                        "history_update_strength"
                    )
                )

                if not is_priority:
                    if not eligible:
                        rejected["ineligible"] += 1
                        continue

                    if (
                        event_type
                        in HARD_REJECT_EVENT_TYPES
                    ):
                        rejected["hard_reject"] += 1
                        continue

                    if (
                        is_history_repeat
                        and history_update < 60
                    ):
                        rejected["history_repeat"] += 1
                        continue

                imp = self._safe_score(
                    ev.get("importance")
                )
                scale = self._safe_score(
                    ev.get("scale")
                )
                rel = self._safe_score(
                    ev.get("reliability")
                )
                pub = self._safe_score(
                    ev.get("public_interest")
                )
                nov = self._safe_score(
                    ev.get("novelty")
                )
                med = self._safe_score(
                    ev.get("media_quality")
                )
                national = self._safe_score(
                    ev.get(
                        "national_relevance"
                    )
                )
                urgency = self._safe_score(
                    ev.get("urgency")
                )

                category = ev.get(
                    "category",
                    "other",
                )

                if (
                    category
                    not in self.ALLOWED_CATEGORIES
                ):
                    category = "other"

                has_video = any(
                    posts[s].get(
                        "has_video"
                    )
                    for s in src_ids
                )

                has_media = any(
                    posts[s].get(
                        "has_media"
                    )
                    for s in src_ids
                )

                # Типово слабка подія не баниться лише за ярликом.
                # Вона може пройти, якщо Analyzer бачить сильний
                # інформаційний фактор.
                if (
                    not is_priority
                    and event_type
                    in LOW_VALUE_EVENT_TYPES
                ):
                    hot_exception = (
                        imp >= 70
                        or national >= 70
                        or (
                            urgency >= 75
                            and imp >= 60
                        )
                        or (
                            pub >= 70
                            and nov >= 65
                        )
                        or (
                            med >= 80
                            and imp >= 60
                            and (
                                has_video
                                or has_media
                            )
                        )
                        or (
                            is_history_repeat
                            and history_update >= 75
                        )
                    )

                    if not hot_exception:
                        rejected["low_value"] += 1
                        continue

                tier_mult = (
                    self._event_source_multiplier(
                        src_ids,
                        posts,
                    )
                )

                base_score = (
                    imp * 0.28
                    + scale * 0.13
                    + rel * 0.17
                    + pub * 0.10
                    + nov * 0.10
                    + national * 0.14
                    + urgency * 0.06
                    + med * 0.02
                )

                score = (
                    base_score
                    * tier_mult
                )

                # Кілька джерел — плюс, але без надмірного розгону.
                score += min(
                    len(src_ids) * 1.2,
                    6,
                )

                meaningful_event = (
                    imp >= 60
                    or national >= 60
                    or pub >= 65
                    or urgency >= 75
                )

                # Медіа — невеликий плюс тільки для вже змістовної події.
                if meaningful_event:
                    if has_video:
                        score += 5
                    elif has_media:
                        score += 2.5

                # Гарячість + новизна.
                if (
                    urgency >= 80
                    and nov >= 65
                ):
                    score += 4

                if (
                    urgency >= 85
                    and imp >= 75
                ):
                    score += 4

                # Значущий розвиток уже відомої історії.
                if (
                    is_history_repeat
                    and history_update >= 60
                ):
                    score += min(
                        (
                            history_update - 60
                        )
                        * 0.10,
                        4,
                    )

                if rel < 45:
                    score -= 20
                elif rel < 60:
                    score -= 8

                # Локальність не караємо, якщо подія сама по собі
                # дуже важлива або має високий суспільний інтерес.
                if (
                    national < 40
                    and imp < 70
                    and pub < 70
                ):
                    score -= 10

                if (
                    not is_priority
                    and event_type
                    in LOW_VALUE_EVENT_TYPES
                ):
                    score -= 4

                if is_priority:
                    score += 500

                factual_source = (
                    self._select_factual_source(
                        src_ids,
                        posts,
                        ev.get(
                            "best_factual_source_id"
                        ),
                    )
                )

                media_source = (
                    self._select_media_source(
                        src_ids,
                        posts,
                        ev.get(
                            "best_media_source_id"
                        ),
                    )
                )

                publishing_source = (
                    media_source
                    if media_source is not None
                    else factual_source
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
                    "has_video": has_video,
                    "has_media": has_media,
                    "is_history_repeat": is_history_repeat,
                    "history_update_strength": history_update,
                    "raw_score": round(
                        score,
                        2,
                    ),
                })

                ranked.append(
                    ev_copy
                )

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
            key=lambda x: x.get(
                "raw_score",
                0,
            ),
            reverse=True,
        )

        category_counts: Dict[str, int] = {}

        for ev in ranked:
            if ev.get("is_priority"):
                ev["balanced_score"] = (
                    ev["raw_score"]
                )
                continue

            category = ev["category"]
            current = category_counts.get(
                category,
                0,
            )

            penalty = (
                10
                if current >= 4
                else (
                    5
                    if current >= 3
                    else 0
                )
            )

            ev["balanced_score"] = round(
                ev["raw_score"]
                - penalty,
                2,
            )

            category_counts[category] = (
                current + 1
            )

        ranked.sort(
            key=lambda x: x.get(
                "balanced_score",
                0,
            ),
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
            factual_id = ev[
                "best_factual_source_id"
            ]
            media_id = ev.get(
                "best_media_source_id"
            )

            factual_post = posts[
                factual_id
            ]

            media_description = "немає"

            if isinstance(
                media_id,
                int,
            ):
                if posts[
                    media_id
                ].get("has_video"):
                    media_description = (
                        "відео з Telegram-поста "
                        "цієї події"
                    )
                elif posts[
                    media_id
                ].get("has_media"):
                    media_description = (
                        "фото з Telegram-поста "
                        "цієї події"
                    )

            key_facts = ev.get(
                "key_facts",
                [],
            )

            key_facts_text = (
                "; ".join(
                    str(x)
                    for x in key_facts[:5]
                )
                if isinstance(
                    key_facts,
                    list,
                )
                else str(key_facts)
            )

            priority_flag = (
                " ⭐ ПРІОРИТЕТ АДМІНІСТРАТОРА"
                if ev.get("is_priority")
                else ""
            )

            repeat_info = (
                "так, але є значущий новий розвиток"
                if ev.get(
                    "is_history_repeat"
                )
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
                "ЧОМУ ВАЖЛИВО: "
                f"{ev.get('why_it_matters', '')}\n"
                "КЛЮЧОВІ ФАКТИ: "
                f"{key_facts_text}\n"
                "ТЕКСТ ДЖЕРЕЛА: "
                f"{str(factual_post.get('text') or '')[:self.MAX_EVENT_SOURCE_CHARS]}\n"
            )

        history_block = (
            self._build_history_block(
                past_events
            )
        )

        prompt = f"""
Ти — головний редактор новинного Telegram-каналу.

Сформуй фінальний дайджест із найважливіших,
найцікавіших і найактуальніших подій.
Максимум: {max_count} новин.

Цільовий формат — 5-10 сильних новин.
Не потрібно обов'язково набирати {max_count}.
Не додавай слабку новину тільки для кількості.

Події ⭐ ПРІОРИТЕТ АДМІНІСТРАТОРА обов'язково включи першими.

━━━━━━━━━━━━━━━━━━━━
АРХІВ:
{history_block}
━━━━━━━━━━━━━━━━━━━━

ВИМОГИ ДО ВИБОРУ:

1. Не повторюй одну реальну подію двічі.
2. Якщо подія вже була в архіві, включай її знову лише коли кандидат
   містить реально значущий новий розвиток.
3. Не додавай слабку подію тільки для заповнення кількості.
4. Не вигадуй факти.
5. Не використовуй чутки.
6. Не оцінюй важливість за довжиною початкового Telegram-посту.
7. Коротка гаряча новина може бути однією з головних новин дайджесту.
8. Відео або фото саме по собі не робить слабку подію важливою.
9. Якщо значуща гаряча подія має реальне фото чи відео з місця —
   це додатковий плюс.
10. При близьких оцінках віддавай перевагу події,
    яка додає читачеві нову картину того, що відбувається зараз.

━━━━━━━━━━━━━━━━━━━━
ВИМОГИ ДО ТЕКСТУ:

Максимальна довжина однієї новини — {self.MAX_NEWS_CHARS} символів.
Бажана довжина — приблизно 350-620 символів разом із заголовком,
АЛЕ не розтягуй коротку гарячу новину заради довжини.

Кожна новина повинна містити 2-5 ЗАВЕРШЕНИХ речень.

Якщо підтверджених фактів мало:
краще 2 короткі повні речення без вигадок,
ніж 4-5 речень із водою або припущеннями.

ВАЖЛИВО:
- ніколи не обривай останнє речення;
- ніколи не завершуй новину на півслові;
- не додавай фактів, яких немає у кандидатові;
- якщо текст виходить задовгим, скороти формулювання;
- кожне речення повинно додавати нову інформацію.

СТРУКТУРА:

Речення 1:
що сталося і де.

Речення 2:
головний наслідок, результат або найважливіша деталь.

Речення 3:
ключові цифри, масштаб або підтверджені подробиці — якщо вони є.

Речення 4:
контекст, реакція сторін або пояснення значення — тільки якщо це
випливає з наданих фактів.

Речення 5:
лише якщо є ще один справді важливий підтверджений факт.

Не повторюй один і той самий факт різними словами.

Заголовок:
- 4-9 слів;
- без клікбейту;
- максимально конкретний;
- не повторює дослівно перше речення.

Формат:

ОДИН тематичний емодзі + <b>Заголовок</b>

порожній рядок

2-5 завершених коротких речень.

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

ПЕРЕД ВІДПОВІДДЮ ПЕРЕВІР КОЖНУ НОВИНУ:

1. Чи не перевищує вона {self.MAX_NEWS_CHARS} символів?
2. Чи має вона щонайменше 2 повні речення?
3. Чи завершене останнє речення?
4. Чи немає повторів і води?
5. Чи всі твердження походять із наданих фактів?
6. Чи не дублює вона іншу новину в цьому ж дайджесті?

ВІДПОВІДЬ ТІЛЬКИ JSON:

{{
  "news": [
    {{
      "event_id": "E1",
      "text": "💥 <b>Короткий заголовок</b>\\n\\nПерше завершене речення. Друге завершене речення."
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
        )

        raw_news = (
            data.get("news", [])
            if (
                data
                and isinstance(
                    data.get("news"),
                    list,
                )
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
            if not isinstance(
                item,
                dict,
            ):
                continue

            event_id = str(
                item.get("event_id")
                or ""
            )
            text = item.get("text")

            if (
                event_id not in event_map
                or not isinstance(
                    text,
                    str,
                )
                or not text.strip()
            ):
                continue

            ev = event_map[
                event_id
            ]

            final_list.append({
                "event_id": event_id,
                "source_id": ev[
                    "best_source_id"
                ],
                "source_ids": list(
                    ev.get(
                        "source_ids",
                        [],
                    )
                ),
                "summary": ev.get(
                    "summary",
                    "",
                ),
                "category": ev.get(
                    "category",
                    "other",
                ),
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
            source_id = item.get(
                "source_id"
            )
            event_id = str(
                item.get("event_id")
                or ""
            )
            text = item.get("text")

            if event_id not in event_map:
                continue

            if (
                not isinstance(
                    source_id,
                    int,
                )
                or not (
                    0
                    <= source_id
                    < len(posts)
                )
            ):
                continue

            if event_id in used_event_ids:
                continue

            if (
                not isinstance(text, str)
                or not text.strip()
            ):
                continue

            text = (
                self._clean_generated_news_text(
                    text
                )
            )

            if not text:
                continue

            ev = event_map[
                event_id
            ]

            validated.append({
                "event_id": event_id,
                "source_id": source_id,
                "source_ids": list(
                    ev.get(
                        "source_ids",
                        [],
                    )
                ),
                "text": text,
                "summary": item.get(
                    "summary",
                    ev.get(
                        "summary",
                        "",
                    ),
                ),
                "category": item.get(
                    "category",
                    ev.get(
                        "category",
                        "other",
                    ),
                ),
            })

            used_event_ids.add(
                event_id
            )

            if len(validated) >= count:
                break

        return validated

    def _fill_missing_news(
        self,
        validated: List[Dict[str, Any]],
        ranked_events: List[Dict[str, Any]],
        posts: List[Dict[str, Any]],
        count: int,
    ) -> List[Dict[str, Any]]:
        result = list(
            validated
        )

        used_event_ids = {
            str(
                item.get("event_id")
                or ""
            )
            for item in result
            if item.get("event_id")
        }

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

            event_id = str(
                ev.get("event_id")
                or ""
            )

            if (
                not event_id
                or event_id in used_event_ids
            ):
                continue

            source_id = ev.get(
                "best_source_id"
            )

            if (
                not isinstance(
                    source_id,
                    int,
                )
                or not (
                    0
                    <= source_id
                    < len(posts)
                )
            ):
                continue

            headline = (
                ev.get("headline_hint")
                or "Важлива подія"
            ).strip()

            category = ev.get(
                "category",
                "other",
            )
            emoji = emoji_map.get(
                category,
                "📰",
            )

            key_facts = ev.get(
                "key_facts",
                [],
            )

            facts = (
                [
                    str(x).strip()
                    for x in key_facts
                    if str(x).strip()
                ]
                if isinstance(
                    key_facts,
                    list,
                )
                else []
            )

            summary = (
                ev.get("summary")
                or ""
            ).strip()

            why = (
                ev.get("why_it_matters")
                or ""
            ).strip()

            sentences = []

            if summary:
                sentences.append(
                    self._ensure_sentence_end(
                        summary
                    )
                )

            for fact in facts[:4]:
                sentence = (
                    self._ensure_sentence_end(
                        fact
                    )
                )

                if (
                    sentence
                    and sentence
                    not in sentences
                ):
                    sentences.append(
                        sentence
                    )

            if why:
                sentence = (
                    self._ensure_sentence_end(
                        why
                    )
                )

                if (
                    sentence
                    and sentence
                    not in sentences
                ):
                    sentences.append(
                        sentence
                    )

            original_text = (
                posts[source_id].get(
                    "text"
                )
                or ""
            )

            if (
                len(sentences) < 2
                and original_text
            ):
                source_sentences = (
                    self._extract_sentences(
                        original_text
                    )
                )

                for sentence in source_sentences:
                    clean_sentence = (
                        self._ensure_sentence_end(
                            sentence
                        )
                    )

                    if (
                        clean_sentence
                        and clean_sentence
                        not in sentences
                    ):
                        sentences.append(
                            clean_sentence
                        )

                    if len(sentences) >= 2:
                        break

            if len(sentences) < 2:
                continue

            text = (
                f"{emoji} <b>{headline}</b>\n\n"
                f"{' '.join(sentences[:5])}"
            )

            text = (
                self._clean_generated_news_text(
                    text
                )
            )

            if not text:
                continue

            result.append({
                "event_id": event_id,
                "source_id": source_id,
                "source_ids": list(
                    ev.get(
                        "source_ids",
                        [],
                    )
                ),
                "text": text,
                "summary": summary,
                "category": category,
            })

            used_event_ids.add(
                event_id
            )

        return result[:count]

    def _select_factual_source(
        self,
        source_ids: List[int],
        posts: List[Dict[str, Any]],
        preferred_id: Any = None,
    ) -> int:
        if (
            isinstance(
                preferred_id,
                int,
            )
            and preferred_id in source_ids
        ):
            return preferred_id

        return max(
            source_ids,
            key=lambda s: (
                self._factual_source_score(
                    posts[s]
                )
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
                posts[s].get(
                    "has_video"
                )
                or posts[s].get(
                    "has_media"
                )
            )
        ]

        if not media_ids:
            return None

        if (
            isinstance(
                preferred_id,
                int,
            )
            and preferred_id
            in media_ids
        ):
            return preferred_id

        return max(
            media_ids,
            key=lambda s: (
                self._media_source_score(
                    posts[s]
                )
            ),
        )

    def _factual_source_score(
        self,
        post: Dict[str, Any],
    ) -> float:
        if post.get("is_priority"):
            return 10000.0

        username = (
            str(
                post.get(
                    "channel_username",
                    "",
                )
                or ""
            )
            .replace("@", "")
            .strip()
        )

        views = int(
            post.get("views") or 0
        )
        forwards = int(
            post.get("forwards") or 0
        )
        text_length = len(
            post.get("text")
            or ""
        )

        # Довжина тут НЕ оцінює важливість події.
        # Вона лише допомагає вибрати більш інформативне
        # фактичне джерело всередині вже відібраної події.
        score = (
            min(
                math.log10(
                    max(views, 1)
                )
                * 5,
                25,
            )
            + min(
                math.log10(
                    max(forwards, 1)
                )
                * 3,
                10,
            )
            + min(
                text_length / 180,
                7,
            )
        )

        return (
            score
            * self._get_source_multiplier(
                username
            )
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

        views = int(
            post.get("views") or 0
        )
        forwards = int(
            post.get("forwards") or 0
        )

        score += min(
            math.log10(
                max(views, 1)
            )
            * 3,
            18,
        )

        score += min(
            math.log10(
                max(forwards, 1)
            )
            * 2,
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
                    posts[
                        source_id
                    ].get(
                        "channel_username",
                        "",
                    )
                    or ""
                )
                .replace("@", "")
                .strip()
            )

            multipliers.append(
                self._get_source_multiplier(
                    username
                )
            )

        # Подію не караємо за те, що її поряд із сильним джерелом
        # перепостив слабший агрегатор. Беремо найкраще підтвердження.
        return (
            max(multipliers)
            if multipliers
            else 1.0
        )

    @staticmethod
    def _get_source_multiplier(
        username: str,
    ) -> float:
        if username in SOURCE_TIERS:
            return SOURCE_TIERS[
                username
            ]

        username_lower = (
            username.lower()
        )

        for (
            source,
            multiplier,
        ) in SOURCE_TIERS.items():
            if (
                source.lower()
                == username_lower
            ):
                return multiplier

        return 1.0

    def _call_json_with_cascade(
        self,
        prompt: str,
        max_retries: int,
        op_name: str,
    ) -> Optional[Dict[str, Any]]:
        for model in self.models_priority:
            for attempt in range(
                1,
                max_retries + 1,
            ):
                try:
                    logger.info(
                        f"{op_name}: спроба "
                        f"{attempt}/{max_retries} "
                        f"через {model}"
                    )

                    response = (
                        self.client.models.generate_content(
                            model=model,
                            contents=prompt,
                            config=(
                                types.GenerateContentConfig(
                                    response_mime_type=(
                                        "application/json"
                                    ),
                                    temperature=0.15,
                                )
                            ),
                        )
                    )

                    raw_text = (
                        self._clean_json_response(
                            (
                                response.text
                                or ""
                            ).strip()
                        )
                    )

                    data = json.loads(
                        raw_text
                    )

                    if isinstance(
                        data,
                        dict,
                    ):
                        return data

                except Exception as e:
                    err = str(e)

                    if any(
                        x in err
                        for x in [
                            "503",
                            "429",
                            "UNAVAILABLE",
                            "ResourceExhausted",
                            "NOT_FOUND",
                        ]
                    ):
                        if attempt < max_retries:
                            time.sleep(
                                3 * attempt
                            )
                            continue

                        break

                    logger.error(
                        "Помилка "
                        f"{op_name} "
                        f"({model}): {e}"
                    )
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
            return (
                "Історія опублікованих "
                "подій порожня."
            )

        lines = []

        for item in past_events[
            :self.HISTORY_LIMIT
        ]:
            if isinstance(
                item,
                dict,
            ):
                title = (
                    item.get("title")
                    or ""
                ).strip()

                summary = (
                    item.get("summary")
                    or ""
                ).strip()

                published_at = (
                    item.get(
                        "published_at"
                    )
                    or ""
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
                        f"- {title}"
                        f"{desc}"
                        f"{time_info}"
                    )

            elif (
                isinstance(item, str)
                and item.strip()
            ):
                lines.append(
                    f"- {item.strip()}"
                )

        return (
            "\n".join(lines)
            if lines
            else (
                "Історія опублікованих "
                "подій порожня."
            )
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

        if (
            "<b>" not in text
            or "</b>" not in text
        ):
            lines = text.split(
                "\n",
                1,
            )
            first_line = (
                lines[0].strip()
            )
            rest = (
                "\n" + lines[1]
                if len(lines) > 1
                else ""
            )

            text = (
                f"<b>{first_line}</b>"
                f"{rest}"
            )

        if (
            len(text)
            > self.MAX_NEWS_CHARS
        ):
            text = (
                self._truncate_to_complete_sentence(
                    text,
                    self.MAX_NEWS_CHARS,
                )
            )

        if (
            "<b>" in text
            and "</b>" not in text
        ):
            text += "</b>"

        return text.strip()

    @staticmethod
    def _truncate_to_complete_sentence(
        text: str,
        max_chars: int,
    ) -> str:
        if len(text) <= max_chars:
            return text.strip()

        candidate = (
            text[:max_chars]
            .rstrip()
        )

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
                if (
                    pos
                    >= int(
                        max_chars * 0.55
                    )
                )
            ]

            if safe_endings:
                return (
                    candidate[
                        :safe_endings[-1]
                    ].strip()
                )

        last_space = candidate.rfind(
            " "
        )

        if (
            last_space
            > int(
                max_chars * 0.70
            )
        ):
            candidate = (
                candidate[
                    :last_space
                ].rstrip()
            )

        return (
            candidate.rstrip(
                ",;:- "
            )
            + "…"
        )

    @staticmethod
    def _extract_sentences(
        text: str,
    ) -> List[str]:
        text = re.sub(
            r"\s+",
            " ",
            text,
        ).strip()

        if not text:
            return []

        parts = re.split(
            r"(?<=[.!?])\s+",
            text,
        )

        return [
            part.strip()
            for part in parts
            if len(
                part.strip()
            ) >= 12
        ]

    @staticmethod
    def _ensure_sentence_end(
        text: str,
    ) -> str:
        text = re.sub(
            r"\s+",
            " ",
            text,
        ).strip()

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
                min(
                    100.0,
                    float(value),
                ),
            )

        except (
            TypeError,
            ValueError,
        ):
            return 0.0

    @staticmethod
    def _clean_json_response(
        text: str,
    ) -> str:
        text = text.strip()

        if text.startswith(
            "```json"
        ):
            text = text[7:]

        elif text.startswith(
            "```"
        ):
            text = text[3:]

        if text.endswith(
            "```"
        ):
            text = text[:-3]

        return text.strip()
