import json
import logging
import re
import time
from difflib import SequenceMatcher
from hashlib import sha1
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from google import genai
from google.genai import types

from app.config import settings

logger = logging.getLogger(__name__)


class QualityAuditor:
    """
    Read-only post-publication QA.

    ВАЖЛИВО:
    - нічого не видаляє;
    - нічого не редагує;
    - нічого не перепубліковує;
    - не впливає на поточний ranking/dedup.

    Він лише перевіряє вже завершений випуск,
    пише результати у лог і повертає структуру
    для збереження в NewsHistory.
    """

    MAX_HISTORY_ITEMS = 28
    MAX_DUPLICATE_CASES = 24

    # Лише достатньо впевнені semantic matches
    # вважаємо реальною audit-проблемою.
    MIN_LLM_WARNING_CONFIDENCE = 80

    BLOCKED_REFERENCE_HOSTS = {
        "t.me",
        "telegram.me",
        "telegram.org",
        "instagram.com",
        "www.instagram.com",
        "facebook.com",
        "www.facebook.com",
        "x.com",
        "twitter.com",
        "www.twitter.com",
        "youtube.com",
        "www.youtube.com",
        "youtu.be",
        "tiktok.com",
        "www.tiktok.com",
        "vk.com",
        "ok.ru",
    }

    STOPWORDS = {
        "але",
        "або",
        "без",
        "був",
        "була",
        "було",
        "були",
        "від",
        "для",
        "до",
        "з",
        "за",
        "і",
        "й",
        "із",
        "на",
        "не",
        "по",
        "про",
        "та",
        "у",
        "в",
        "що",
        "це",
        "цей",
        "ця",
        "ці",
        "як",
        "який",
        "яка",
        "після",
        "під",
        "при",
        "через",
        "вже",
        "ще",
        "також",
        "свою",
        "свої",
        "свій",
        "the",
        "and",
        "for",
        "with",
        "from",
        "that",
    }

    def __init__(self):
        self.client = genai.Client(
            api_key=settings.GEMINI_API_KEY
        )

        self.models_priority = [
            "gemini-3.5-flash-lite",
            "gemini-3.1-flash-lite",
        ]

    def run(
        self,
        published_news: List[Dict[str, Any]],
        prior_events: List[Dict[str, Any]],
        fact_check_stats: Optional[
            Dict[str, Any]
        ] = None,
        expected_manual_ids: Optional[
            List[int]
        ] = None,
        published_manual_ids: Optional[
            List[int]
        ] = None,
        cycle_started_at: str = "",
    ) -> Dict[str, Any]:
        """
        Основна audit-перевірка одного випуску.
        """

        started = time.monotonic()

        news = [
            item
            for item in (
                published_news
                or []
            )
            if isinstance(
                item,
                dict,
            )
        ]

        history = [
            item
            for item in (
                prior_events
                or []
            )
            if isinstance(
                item,
                dict,
            )
        ]

        history = history[
            :self.MAX_HISTORY_ITEMS
        ]

        issues: List[
            Dict[str, Any]
        ] = []

        # 1. Пошук підозрілих semantic duplicate pairs.
        duplicate_cases = (
            self._build_duplicate_cases(
                news,
                history,
            )
        )

        duplicate_findings = (
            self._review_duplicate_cases(
                duplicate_cases
            )
        )

        for finding in duplicate_findings:
            if finding.get(
                "is_problem"
            ):
                issues.append({
                    "type": (
                        "possible_duplicate"
                    ),
                    **finding,
                })

        # 2. Structural checks.
        reference_issues = (
            self._check_references(
                news
            )
        )

        media_issues = (
            self._check_media_state(
                news
            )
        )

        fact_issues = (
            self._check_fact_check(
                news,
                fact_check_stats or {},
            )
        )

        priority_issues = (
            self._check_priority(
                expected_manual_ids or [],
                published_manual_ids or [],
            )
        )

        issues.extend(
            reference_issues
        )
        issues.extend(
            media_issues
        )
        issues.extend(
            fact_issues
        )
        issues.extend(
            priority_issues
        )

        status = (
            "OK"
            if not issues
            else "WARNING"
        )

        elapsed_ms = int(
            (
                time.monotonic()
                - started
            )
            * 1000
        )

        result = {
            "cycle_started_at": str(
                cycle_started_at
                or ""
            ),
            "status": status,
            "published_count": len(
                news
            ),
            "possible_duplicates": sum(
                1
                for issue in issues
                if issue.get(
                    "type"
                )
                == "possible_duplicate"
            ),
            "reference_issues": len(
                reference_issues
            ),
            "media_issues": len(
                media_issues
            ),
            "fact_check_issues": len(
                fact_issues
            ),
            "priority_issues": len(
                priority_issues
            ),
            "fact_check_stats": dict(
                fact_check_stats
                or {}
            ),
            "issues": issues,

            # Поки лише накопичуємо пам'ять.
            # На dedup вона ЩЕ НЕ впливає.
            "memory_observations": (
                self._memory_observations(
                    duplicate_findings
                )
            ),

            "elapsed_ms": elapsed_ms,
        }

        if status == "OK":
            logger.info(
                "QUALITY AUDIT: OK | "
                "published=%s "
                "duplicates=0 "
                "fact_check_issues=0 "
                "reference_issues=0 "
                "media_issues=0 "
                "priority_issues=0 "
                "elapsed=%sms",
                len(news),
                elapsed_ms,
            )

        else:
            logger.warning(
                "QUALITY AUDIT: WARNING | "
                "published=%s "
                "duplicates=%s "
                "fact_check_issues=%s "
                "reference_issues=%s "
                "media_issues=%s "
                "priority_issues=%s "
                "elapsed=%sms",
                len(news),
                result[
                    "possible_duplicates"
                ],
                result[
                    "fact_check_issues"
                ],
                result[
                    "reference_issues"
                ],
                result[
                    "media_issues"
                ],
                result[
                    "priority_issues"
                ],
                elapsed_ms,
            )

            for issue in issues[
                :12
            ]:
                logger.warning(
                    "QUALITY AUDIT ISSUE: "
                    "type=%s "
                    "event_id=%s "
                    "matched='%s' "
                    "confidence=%s "
                    "reason='%s'",
                    issue.get(
                        "type",
                        "unknown",
                    ),
                    issue.get(
                        "event_id",
                        "",
                    ),
                    str(
                        issue.get(
                            "matched_title"
                        )
                        or ""
                    )[:120],
                    issue.get(
                        "confidence",
                        "",
                    ),
                    str(
                        issue.get(
                            "reason"
                        )
                        or ""
                    )[:260],
                )

        return result

    def _build_duplicate_cases(
        self,
        news: List[
            Dict[str, Any]
        ],
        history: List[
            Dict[str, Any]
        ],
    ) -> List[
        Dict[str, Any]
    ]:
        """
        Python не приймає рішення "дубль / не дубль".

        Він лише вибирає невелику кількість
        підозрілих пар для Gemini audit.
        """

        cases: List[
            Dict[str, Any]
        ] = []

        # ─────────────────────────────
        # A. Дублі всередині випуску.
        # ─────────────────────────────

        for left_index in range(
            len(news)
        ):
            for right_index in range(
                left_index + 1,
                len(news),
            ):
                left = news[
                    left_index
                ]
                right = news[
                    right_index
                ]

                score = (
                    self._candidate_score(
                        self._news_text(
                            left
                        ),
                        self._news_text(
                            right
                        ),
                    )
                )

                if score < 0.34:
                    continue

                cases.append({
                    "case_id": (
                        f"C{left_index}_"
                        f"{right_index}"
                    ),
                    "scope": (
                        "same_cycle"
                    ),
                    "event_id": str(
                        right.get(
                            "event_id"
                        )
                        or ""
                    ),
                    "current_title": (
                        self._title(
                            self._news_text(
                                right
                            )
                        )
                    ),
                    "current_text": (
                        self._news_text(
                            right
                        )[:1500]
                    ),
                    "matched_title": (
                        self._title(
                            self._news_text(
                                left
                            )
                        )
                    ),
                    "matched_text": (
                        self._news_text(
                            left
                        )[:1500]
                    ),
                    "candidate_score": round(
                        score,
                        3,
                    ),
                })

        # ─────────────────────────────
        # B. Поточний випуск ↔ history.
        # ─────────────────────────────

        for index, item in enumerate(
            news
        ):
            ranked = []

            current_text = (
                self._news_text(
                    item
                )
            )

            for (
                history_index,
                old,
            ) in enumerate(
                history
            ):
                old_text = (
                    self._history_text(
                        old
                    )
                )

                score = (
                    self._candidate_score(
                        current_text,
                        old_text,
                    )
                )

                if score >= 0.26:
                    ranked.append(
                        (
                            score,
                            history_index,
                            old,
                            old_text,
                        )
                    )

            ranked.sort(
                key=lambda row: row[0],
                reverse=True,
            )

            # Не посилаємо Gemini сотні пар.
            # Максимум 3 найпідозріліші history pair
            # для кожної фінальної новини.
            for (
                score,
                history_index,
                old,
                old_text,
            ) in ranked[:3]:

                cases.append({
                    "case_id": (
                        f"H{index}_"
                        f"{history_index}"
                    ),
                    "scope": (
                        "history"
                    ),
                    "event_id": str(
                        item.get(
                            "event_id"
                        )
                        or ""
                    ),
                    "current_title": (
                        self._title(
                            current_text
                        )
                    ),
                    "current_text": (
                        current_text[
                            :1500
                        ]
                    ),
                    "matched_title": str(
                        old.get(
                            "title"
                        )
                        or self._title(
                            old_text
                        )
                    ),
                    "matched_text": (
                        old_text[
                            :1500
                        ]
                    ),
                    "matched_published_at": str(
                        old.get(
                            "published_at"
                        )
                        or ""
                    ),
                    "candidate_score": round(
                        score,
                        3,
                    ),
                })

        cases.sort(
            key=lambda case: (
                case.get(
                    "candidate_score",
                    0,
                )
            ),
            reverse=True,
        )

        return cases[
            :self.MAX_DUPLICATE_CASES
        ]

    def _review_duplicate_cases(
        self,
        cases: List[
            Dict[str, Any]
        ],
    ) -> List[
        Dict[str, Any]
    ]:
        if not cases:
            return []

        # Якщо Gemini audit тимчасово
        # недоступний, абсолютно очевидні
        # майже ідентичні пари все одно
        # можна залогувати.
        deterministic = {}

        for case in cases:
            if (
                float(
                    case.get(
                        "candidate_score",
                        0,
                    )
                    or 0
                )
                >= 0.90
            ):
                deterministic[
                    case[
                        "case_id"
                    ]
                ] = {
                    "same_story": True,
                    "material_update": False,
                    "confidence": 96,
                    "reason": (
                        "Майже ідентичний "
                        "текст/набір фактів."
                    ),
                }

        payload = json.dumps(
            cases,
            ensure_ascii=False,
        )

        prompt = f"""
Ти — read-only quality auditor українського новинного дайджесту.

Перед тобою підозрілі пари вже ОПУБЛІКОВАНОЇ новини та іншої новини.

Твоє завдання — лише класифікувати.
Нічого не переписуй і не пропонуй видаляти.

Для кожної пари визнач:

- same_story:
  це та сама базова реальна подія / рішення / інцидент чи ні;

- material_update:
  якщо same_story=true, чи містить CURRENT справді новий
  значущий розвиток, який виправдовує повтор у короткому дайджесті;

- confidence:
  0-100;

- reason:
  одне коротке речення.

КРИТИЧНІ ПРАВИЛА:

- спільна людина, наприклад Трамп або Зеленський,
  НЕ робить дві теми однією історією;

- слово "дрон", "атака", "санкції", "закон"
  саме по собі НЕ достатнє;

- для фізичних атак потрібен збіг конкретної
  локації, цілі, хвилі або самого інциденту;

- різні міста або різні цілі зазвичай означають
  різні події;

- для законів, санкцій та угод інший переказ
  тієї самої процедурної стадії — той самий story;

- "готовий/планує підписати" НЕ дорівнює
  фактично "підписав";

- новий реальний юридичний статус, великі нові
  наслідки або істотно новий результат можуть бути
  material_update=true;

- якщо сумніваєшся — НЕ називай пару дублем.

ВІДПОВІДЬ ТІЛЬКИ JSON:

{{
  "checks": [
    {{
      "case_id": "H0_1",
      "same_story": false,
      "material_update": false,
      "confidence": 92,
      "reason": "Різні теми."
    }}
  ]
}}

CASES:

{payload}
"""

        data = self._call_json(
            prompt
        )

        checks = (
            data.get(
                "checks",
                [],
            )
            if (
                data
                and isinstance(
                    data.get(
                        "checks"
                    ),
                    list,
                )
            )
            else []
        )

        check_map = {
            str(
                check.get(
                    "case_id"
                )
                or ""
            ): check
            for check in checks
            if (
                isinstance(
                    check,
                    dict,
                )
                and check.get(
                    "case_id"
                )
            )
        }

        findings = []

        for case in cases:
            check = (
                check_map.get(
                    case[
                        "case_id"
                    ]
                )
                or deterministic.get(
                    case[
                        "case_id"
                    ]
                )
            )

            if not check:
                continue

            confidence = (
                self._confidence(
                    check.get(
                        "confidence"
                    )
                )
            )

            same_story = bool(
                check.get(
                    "same_story",
                    False,
                )
            )

            material_update = bool(
                check.get(
                    "material_update",
                    False,
                )
            )

            is_problem = (
                same_story
                and not material_update
                and confidence
                >= self.MIN_LLM_WARNING_CONFIDENCE
            )

            findings.append({
                "case_id": (
                    case[
                        "case_id"
                    ]
                ),
                "scope": case.get(
                    "scope",
                    "",
                ),
                "event_id": (
                    case.get(
                        "event_id",
                        "",
                    )
                ),
                "current_title": (
                    case.get(
                        "current_title",
                        "",
                    )
                ),
                "matched_title": (
                    case.get(
                        "matched_title",
                        "",
                    )
                ),
                "same_story": (
                    same_story
                ),
                "material_update": (
                    material_update
                ),
                "confidence": (
                    confidence
                ),
                "reason": str(
                    check.get(
                        "reason"
                    )
                    or ""
                ).strip(),
                "is_problem": (
                    is_problem
                ),
            })

        return findings

    def _call_json(
        self,
        prompt: str,
    ) -> Optional[
        Dict[str, Any]
    ]:
        """
        Один маленький batch Gemini call.
        Якщо він не працює — audit не ламає цикл.
        """

        for model_name in (
            self.models_priority
        ):
            try:
                response = (
                    self.client.models.generate_content(
                        model=model_name,
                        contents=prompt,
                        config=(
                            types.GenerateContentConfig(
                                temperature=0.05,
                                response_mime_type=(
                                    "application/json"
                                ),
                            )
                        ),
                    )
                )

                text = str(
                    getattr(
                        response,
                        "text",
                        "",
                    )
                    or ""
                ).strip()

                if not text:
                    continue

                data = json.loads(
                    text
                )

                if isinstance(
                    data,
                    dict,
                ):
                    return data

            except Exception as exc:
                logger.warning(
                    "QUALITY AUDIT model "
                    "%s unavailable: %s",
                    model_name,
                    exc,
                )

        return None

    def _check_fact_check(
        self,
        news: List[
            Dict[str, Any]
        ],
        stats: Dict[
            str,
            Any,
        ],
    ) -> List[
        Dict[str, Any]
    ]:
        """
        Перевіряє telemetry FINAL_FACT_CHECK.

        Це НЕ другий fact-check.
        Ми лише дивимось, чи всі фінальні items
        реально були передані в попередній factual pass.
        """

        issues = []

        expected = self._int(
            stats.get(
                "expected"
            ),
            default=len(news),
        )

        eligible = self._int(
            stats.get(
                "eligible_cases"
            ),
            default=expected,
        )

        checked = self._int(
            stats.get(
                "checked"
            ),
            default=0,
        )

        if expected != len(news):
            issues.append({
                "type": (
                    "fact_check_count_mismatch"
                ),
                "event_id": "",
                "confidence": 100,
                "reason": (
                    "Final fact-check очікував "
                    f"{expected} новин, а до audit "
                    f"дійшло {len(news)} успішно "
                    "опублікованих."
                ),
            })

        if eligible < expected:
            issues.append({
                "type": (
                    "fact_check_unmapped_items"
                ),
                "event_id": "",
                "confidence": 100,
                "reason": (
                    "Для fact-check вдалося "
                    f"побудувати лише "
                    f"{eligible}/{expected} cases."
                ),
            })

        if checked < eligible:
            issues.append({
                "type": (
                    "fact_check_missing"
                ),
                "event_id": "",
                "confidence": 100,
                "reason": (
                    "Gemini повернув перевірку "
                    f"лише для "
                    f"{checked}/{eligible} cases."
                ),
            })

        return issues

    def _check_references(
        self,
        news: List[
            Dict[str, Any]
        ],
    ) -> List[
        Dict[str, Any]
    ]:
        """
        Без HTTP-запитів.

        Перевіряємо лише structural validity,
        щоб audit не створював зайвий network load.
        """

        issues = []

        for item in news:
            url = str(
                item.get(
                    "reference_url"
                )
                or ""
            ).strip()

            if not url:
                continue

            try:
                parsed = urlparse(
                    url
                )
                host = (
                    parsed.hostname
                    or ""
                ).lower()

            except Exception:
                parsed = None
                host = ""

            reason = ""

            if (
                not parsed
                or parsed.scheme
                not in {
                    "http",
                    "https",
                }
                or not host
            ):
                reason = (
                    "Некоректний формат "
                    "reference_url."
                )

            elif (
                host
                in self.BLOCKED_REFERENCE_HOSTS
                or any(
                    host.endswith(
                        "."
                        + blocked
                    )
                    for blocked
                    in (
                        self.BLOCKED_REFERENCE_HOSTS
                    )
                )
            ):
                reason = (
                    "Reference URL веде на "
                    "social/Telegram host, "
                    "який не має бути "
                    "першоджерелом."
                )

            if reason:
                issues.append({
                    "type": (
                        "reference_issue"
                    ),
                    "event_id": str(
                        item.get(
                            "event_id"
                        )
                        or ""
                    ),
                    "confidence": 100,
                    "reason": reason,
                })

        return issues

    def _check_media_state(
        self,
        news: List[
            Dict[str, Any]
        ],
    ) -> List[
        Dict[str, Any]
    ]:
        """
        Не запускає Vision вдруге.

        Перевіряємо тільки state:
        rejected media не повинно залишитися
        у фінальному Telegram/Instagram pipeline.
        """

        issues = []

        for item in news:
            audit = item.get(
                "_audit_media"
            )

            if not isinstance(
                audit,
                dict,
            ):
                continue

            rejected = bool(
                audit.get(
                    "rejected"
                )
            )

            final_path = str(
                audit.get(
                    "final_path"
                )
                or ""
            ).strip()

            final_type = str(
                audit.get(
                    "final_type"
                )
                or ""
            ).strip()

            if (
                rejected
                and (
                    final_path
                    or final_type
                )
            ):
                issues.append({
                    "type": (
                        "media_state_issue"
                    ),
                    "event_id": str(
                        item.get(
                            "event_id"
                        )
                        or ""
                    ),
                    "confidence": 100,
                    "reason": (
                        "Медіа було відхилене gate, "
                        "але лишилося у final "
                        "media state."
                    ),
                })

            elif (
                bool(final_path)
                != bool(final_type)
            ):
                issues.append({
                    "type": (
                        "media_state_issue"
                    ),
                    "event_id": str(
                        item.get(
                            "event_id"
                        )
                        or ""
                    ),
                    "confidence": 100,
                    "reason": (
                        "media_path і media_type "
                        "неузгоджені."
                    ),
                })

        return issues

    @staticmethod
    def _check_priority(
        expected_manual_ids: List[
            int
        ],
        published_manual_ids: List[
            int
        ],
    ) -> List[
        Dict[str, Any]
    ]:
        expected = {
            int(value)
            for value
            in expected_manual_ids
        }

        published = {
            int(value)
            for value
            in published_manual_ids
        }

        missing = sorted(
            expected
            - published
        )

        if not missing:
            return []

        return [{
            "type": (
                "priority_missing"
            ),
            "event_id": "",
            "confidence": 100,
            "reason": (
                "Manual queue IDs не були "
                "успішно опубліковані: "
                f"{missing}"
            ),
        }]

    def _memory_observations(
        self,
        findings: List[
            Dict[str, Any]
        ],
    ) -> List[
        Dict[str, Any]
    ]:
        """
        Уже зараз накопичуємо duplicate observations
        у БД.

        ВАЖЛИВО:
        поточний dedup їх ще НЕ читає.
        Це навмисно — спочатку перевіримо якість audit.
        """

        observations = []

        for finding in findings:
            if not finding.get(
                "is_problem"
            ):
                continue

            left = self._normalize(
                str(
                    finding.get(
                        "current_title"
                    )
                    or ""
                )
            )

            right = self._normalize(
                str(
                    finding.get(
                        "matched_title"
                    )
                    or ""
                )
            )

            pair_key = "|".join(
                sorted([
                    left,
                    right,
                ])
            )

            pair_hash = sha1(
                pair_key.encode(
                    "utf-8"
                )
            ).hexdigest()

            observations.append({
                "pair_key": (
                    pair_hash
                ),
                "relation": (
                    "confirmed_same_story_no_update"
                ),
                "current_title": (
                    finding.get(
                        "current_title",
                        "",
                    )
                ),
                "matched_title": (
                    finding.get(
                        "matched_title",
                        "",
                    )
                ),
                "confidence": (
                    finding.get(
                        "confidence",
                        0,
                    )
                ),
                "reason": (
                    finding.get(
                        "reason",
                        "",
                    )
                ),
            })

        return observations

    def _candidate_score(
        self,
        left: str,
        right: str,
    ) -> float:
        a = self._normalize(
            left
        )
        b = self._normalize(
            right
        )

        if not a or not b:
            return 0.0

        if a == b:
            return 1.0

        seq = SequenceMatcher(
            None,
            a[:1600],
            b[:1600],
        ).ratio()

        tokens_a = self._tokens(
            a
        )
        tokens_b = self._tokens(
            b
        )

        if (
            not tokens_a
            or not tokens_b
        ):
            return seq

        common = (
            tokens_a
            & tokens_b
        )

        jaccard = (
            len(common)
            / max(
                len(
                    tokens_a
                    | tokens_b
                ),
                1,
            )
        )

        overlap = (
            len(common)
            / max(
                min(
                    len(tokens_a),
                    len(tokens_b),
                ),
                1,
            )
        )

        return max(
            seq,
            jaccard * 1.15,
            overlap * 0.80,
        )

    def _tokens(
        self,
        text: str,
    ) -> set:
        return {
            token
            for token
            in re.findall(
                r"[0-9a-zа-яіїєґёъыэ-]{3,}",
                text,
            )
            if token
            not in self.STOPWORDS
        }

    @staticmethod
    def _news_text(
        item: Dict[str, Any],
    ) -> str:
        return " ".join(
            value
            for value in [
                str(
                    item.get(
                        "text"
                    )
                    or ""
                ).strip(),
                str(
                    item.get(
                        "summary"
                    )
                    or ""
                ).strip(),
            ]
            if value
        )

    @staticmethod
    def _history_text(
        item: Dict[str, Any],
    ) -> str:
        return " ".join(
            value
            for value in [
                str(
                    item.get(
                        "title"
                    )
                    or ""
                ).strip(),
                str(
                    item.get(
                        "summary"
                    )
                    or ""
                ).strip(),
            ]
            if value
        )

    @staticmethod
    def _title(
        text: str,
    ) -> str:
        clean = re.sub(
            r"<[^>]+>",
            " ",
            str(
                text
                or ""
            ),
        )

        clean = re.sub(
            r"\s+",
            " ",
            clean,
        ).strip()

        return clean[:180]

    @staticmethod
    def _normalize(
        text: str,
    ) -> str:
        value = str(
            text
            or ""
        ).lower()

        value = re.sub(
            r"https?://\S+|t\.me/\S+",
            " ",
            value,
        )

        value = re.sub(
            r"<[^>]+>",
            " ",
            value,
        )

        value = re.sub(
            r"[^0-9a-zа-яіїєґёъыэ\s-]",
            " ",
            value,
        )

        return re.sub(
            r"\s+",
            " ",
            value,
        ).strip()

    @staticmethod
    def _confidence(
        value: Any,
    ) -> int:
        try:
            number = float(
                value
            )

        except (
            TypeError,
            ValueError,
        ):
            return 0

        # Gemini іноді повертає 0.95
        # замість 95.
        if (
            0
            <= number
            <= 1
        ):
            number *= 100

        return max(
            0,
            min(
                100,
                int(
                    round(
                        number
                    )
                ),
            ),
        )

    @staticmethod
    def _int(
        value: Any,
        default: int = 0,
    ) -> int:
        try:
            return int(
                value
            )

        except (
            TypeError,
            ValueError,
        ):
            return int(
                default
            )
