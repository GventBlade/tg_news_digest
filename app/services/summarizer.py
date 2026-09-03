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

    # Тепер намагаємось давати щільніший випуск: якщо є достатньо
    # придатних подій, бажано мати щонайменше 7 матеріалів.
    MIN_DIGEST_COUNT = 7

    EDITOR_CANDIDATES = 30
    HISTORY_LIMIT = 150

    MAX_INPUT_CHARS = 55000
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

        posts_context = self._build_posts_context(posts)
        if not posts_context:
            return []

        analyzed_events = self._analyze_events(
            posts_context,
            past_events,
            max_retries_per_model,
        )

        if not analyzed_events:
            logger.warning("Analyzer не повернув подій.")
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
                "Після editorial gate не залишилось подій."
            )
            return []

        logger.info(
            "Після ranking залишилось "
            f"{len(ranked_events)} подій."
        )

        for idx, event in enumerate(
            ranked_events[:15],
            start=1,
        ):
            logger.info(
                "RANK #%s: %.2f | %s | cur=%.0f | practical=%.0f | %s",
                idx,
                float(
                    event.get(
                        "balanced_score",
                        event.get("raw_score", 0),
                    )
                    or 0
                ),
                event.get("event_type", "other"),
                float(event.get("curiosity", 0) or 0),
                float(event.get("practical_value", 0) or 0),
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

        target_min = min(
            count,
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
                current_length + len(block)
                > self.MAX_INPUT_CHARS
            ):
                continue

            result.append(block)
            current_length += len(block) + 10

        logger.info(
            "У контекст Analyzer потрапило "
            f"{len(result)} з {len(prepared)} постів "
            f"({current_length} символів)."
        )

        return "\n\n---\n\n".join(result)

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

Якщо є ⭐ [ПРІОРИТЕТ АДМІНІСТРАТОРА]:
eligible_for_digest=true
importance=100
novelty=100
urgency=100
curiosity=max(curiosity, 90)

Не створюй події з очевидного шуму.
Але й не будь надто суворим до якісних discovery-новин.
Краще 10-16 добрих кандидатів різних типів,
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

                # Важливість лишається головним фактором, але тепер
                # цікавість і практична цінність мають реальний вплив.
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

                # Кілька джерел — плюс, але без надмірного розгону.
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

                # Гарячість + новизна.
                if urgency >= 80 and nov >= 65:
                    score += 4

                if urgency >= 85 and imp >= 75:
                    score += 4

                # Discovery-бонус: сильна цікава новина має шанс
                # піднятися в район 7-10 місця, а не загубитися.
                if cur >= 85 and nov >= 70:
                    score += 6
                elif cur >= 78 and nov >= 65:
                    score += 3

                # Практична новина, яка реально зачіпає читача.
                if practical >= 85 and pub >= 65:
                    score += 6
                elif practical >= 75 and pub >= 60:
                    score += 3

                # Наука/технології з хорошою новизною трохи підсилюємо,
                # щоб вони не програвали кожній політичній новині.
                if (
                    event_type == "science_tech"
                    and cur >= 70
                    and nov >= 65
                ):
                    score += 3

                # Значущий розвиток уже відомої історії.
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

                # Локальність не караємо, якщо новина важлива,
                # цікава, практична або має високий суспільний інтерес.
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

        # Баланс категорій: після 3-4 матеріалів однієї теми
        # наступним стає трохи важче зайняти місце у фіналі.
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

            # Сильна цікавинка/корисна подія може трохи компенсувати
            # нижчу стратегічну важливість.
            if ev.get("curiosity", 0) >= 82:
                diversity_bonus += 2.5

            if ev.get("practical_value", 0) >= 82:
                diversity_bonus += 2.5

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

ВАЖЛИВА РЕДАКЦІЙНА МЕТА:
випуск не повинен складатися лише з війни, політики та міжнародки.
Після сильного ядра випуску активно шукай 1-3 якісні discovery-новини:
технології, науку, суспільні зміни, корисні правила, бізнес,
українські виробництва, досягнення або інші події,
які викликають реакцію "О, цього я не знав".

Цікавинки НЕ повинні витісняти справді важливі події.
Але якщо у списку вже є 5-7 сильних головних новин,
1-3 якісні цікаві або практично корисні події — це плюс до випуску.

Події ⭐ ПРІОРИТЕТ АДМІНІСТРАТОРА обов'язково включи першими.

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
        count: int,
    ) -> List[Dict[str, Any]]:
        result = list(validated)

        used_event_ids = {
            str(item.get("event_id") or "")
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

            event_id = str(ev.get("event_id") or "")
            if not event_id or event_id in used_event_ids:
                continue

            source_id = ev.get("best_source_id")
            if (
                not isinstance(source_id, int)
                or not (0 <= source_id < len(posts))
            ):
                continue

            headline = (
                ev.get("headline_hint")
                or "Важлива подія"
            ).strip()

            category = ev.get("category", "other")
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

            summary = (ev.get("summary") or "").strip()
            why = (ev.get("why_it_matters") or "").strip()

            sentences = []

            if summary:
                sentences.append(
                    self._ensure_sentence_end(summary)
                )

            # Більше місця для фактів у fallback.
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
                source_sentences = self._extract_sentences(
                    original_text
                )

                for sentence in source_sentences:
                    clean_sentence = self._ensure_sentence_end(
                        sentence
                    )

                    if (
                        clean_sentence
                        and clean_sentence not in sentences
                    ):
                        sentences.append(clean_sentence)

                    if len(sentences) >= 4:
                        break

            if len(sentences) < 2:
                continue

            text = (
                f"{emoji} <b>{headline}</b>\n\n"
                f"{' '.join(sentences[:6])}"
            )

            text = self._clean_generated_news_text(text)
            if not text:
                continue

            result.append({
                "event_id": event_id,
                "source_id": source_id,
                "source_ids": list(ev.get("source_ids", [])),
                "text": text,
                "summary": summary,
                "category": category,
            })

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
        for model in self.models_priority:
            for attempt in range(1, max_retries + 1):
                try:
                    logger.info(
                        f"{op_name}: спроба "
                        f"{attempt}/{max_retries} "
                        f"через {model} "
                        f"(temperature={temperature})"
                    )

                    response = self.client.models.generate_content(
                        model=model,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            temperature=temperature,
                        ),
                    )

                    raw_text = self._clean_json_response(
                        (response.text or "").strip()
                    )

                    data = json.loads(raw_text)

                    if isinstance(data, dict):
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
                            time.sleep(3 * attempt)
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
        text = text.strip()

        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]

        if text.endswith("```"):
            text = text[:-3]

        return text.strip()
