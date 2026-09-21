import json
import logging
import os
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

DB_PATH = os.path.join(DATA_DIR, "news_history.db")


class NewsHistory:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._init_db()

    @staticmethod
    def _normalize_channel_name(
        channel_name: str,
    ) -> str:
        """
        Нормалізує username для стабільного порівняння історії:
        @ForbesUkraines -> forbesukraines
        """
        return (
            str(channel_name or "")
            .strip()
            .replace("@", "")
            .lower()
        )

    def _init_db(self):
        """
        Безпечна ініціалізація БД.

        Старі таблиці не змінюємо.
        Нові audit-таблиці створюються автоматично через
        CREATE TABLE IF NOT EXISTS.

        Тобто окрема ручна міграція SQLite не потрібна.
        """

        with sqlite3.connect(
            self.db_path
        ) as conn:

            # ─────────────────────────────────────
            # Основна історія опублікованих новин.
            # ─────────────────────────────────────

            conn.execute("""
                CREATE TABLE IF NOT EXISTS published_news (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    channel_name TEXT,
                    message_id INTEGER,
                    published_title TEXT DEFAULT '',
                    summary TEXT DEFAULT '',
                    category TEXT DEFAULT '',
                    published_at TEXT,
                    UNIQUE(channel_name, message_id)
                )
            """)

            # ─────────────────────────────────────
            # Черга ручних новин адміністратора.
            # ─────────────────────────────────────

            conn.execute("""
                CREATE TABLE IF NOT EXISTS manual_news_queue (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    raw_text TEXT,
                    channel_title TEXT DEFAULT 'Адмін-вибір',
                    channel_username TEXT DEFAULT '',
                    media_path TEXT DEFAULT NULL,
                    media_type TEXT DEFAULT NULL,
                    has_media INTEGER DEFAULT 0,
                    has_video INTEGER DEFAULT 0,
                    views INTEGER DEFAULT 50000,
                    processed INTEGER DEFAULT 0,
                    created_at TEXT
                )
            """)

            # ─────────────────────────────────────
            # Результат кожного post-publication audit.
            #
            # Один запис = один завершений новинний цикл.
            #
            # Audit нічого не видаляє і не змінює.
            # Тут лише зберігаємо QA-результат.
            # ─────────────────────────────────────

            conn.execute("""
                CREATE TABLE IF NOT EXISTS quality_audit_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,

                    cycle_started_at TEXT DEFAULT '',

                    status TEXT DEFAULT 'UNKNOWN',

                    published_count INTEGER DEFAULT 0,

                    possible_duplicates INTEGER DEFAULT 0,

                    reference_issues INTEGER DEFAULT 0,

                    media_issues INTEGER DEFAULT 0,

                    fact_check_issues INTEGER DEFAULT 0,

                    priority_issues INTEGER DEFAULT 0,

                    elapsed_ms INTEGER DEFAULT 0,

                    issues_json TEXT DEFAULT '[]',

                    fact_check_stats_json TEXT DEFAULT '{}',

                    created_at TEXT
                )
            """)

            # ─────────────────────────────────────
            # Audit-memory.
            #
            # Тут накопичуються повторювані помилки,
            # які audit упевнено визначив як одну story
            # без material update.
            #
            # На першому етапі dedup ЦЮ таблицю НЕ читає.
            # Вона лише накопичує статистику.
            #
            # Пізніше, коли перевіримо якість audit,
            # цю пам'ять можна безпечно підключити назад
            # у preventive dedup.
            # ─────────────────────────────────────

            conn.execute("""
                CREATE TABLE IF NOT EXISTS quality_audit_memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,

                    pair_key TEXT NOT NULL UNIQUE,

                    relation TEXT DEFAULT '',

                    current_title TEXT DEFAULT '',

                    matched_title TEXT DEFAULT '',

                    confidence INTEGER DEFAULT 0,

                    reason TEXT DEFAULT '',

                    hit_count INTEGER DEFAULT 1,

                    first_seen_at TEXT,

                    last_seen_at TEXT
                )
            """)

            # ─────────────────────────────────────
            # Індекси старих таблиць.
            # ─────────────────────────────────────

            conn.execute("""
                CREATE INDEX IF NOT EXISTS
                idx_published_news_published_at
                ON published_news(published_at)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS
                idx_manual_news_processed
                ON manual_news_queue(processed, created_at)
            """)

            # ─────────────────────────────────────
            # Індекси audit.
            # ─────────────────────────────────────

            conn.execute("""
                CREATE INDEX IF NOT EXISTS
                idx_quality_audit_runs_created_at
                ON quality_audit_runs(created_at)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS
                idx_quality_audit_runs_status
                ON quality_audit_runs(status, created_at)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS
                idx_quality_audit_memory_hits
                ON quality_audit_memory(
                    hit_count,
                    last_seen_at
                )
            """)

            conn.commit()

    # ═════════════════════════════════════════════
    # MANUAL NEWS
    # ═════════════════════════════════════════════

    def add_manual_post(
        self,
        raw_text: str,
        channel_title: str = "Адмін-вибір",
        channel_username: str = "",
        media_path: str = None,
        media_type: str = None,
        has_media: bool = False,
        has_video: bool = False,
    ) -> int:

        now_str = (
            datetime.now(
                timezone.utc
            )
            .strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        )

        with sqlite3.connect(
            self.db_path
        ) as conn:

            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT INTO manual_news_queue
                (
                    raw_text,
                    channel_title,
                    channel_username,
                    media_path,
                    media_type,
                    has_media,
                    has_video,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    raw_text.strip(),
                    channel_title,
                    channel_username,
                    media_path,
                    media_type,
                    1 if has_media else 0,
                    1 if has_video else 0,
                    now_str,
                ),
            )

            conn.commit()

            return int(
                cursor.lastrowid
            )

    def get_pending_manual_posts(
        self,
    ) -> List[Dict[str, Any]]:

        with sqlite3.connect(
            self.db_path
        ) as conn:

            conn.row_factory = (
                sqlite3.Row
            )

            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT *
                FROM manual_news_queue
                WHERE processed = 0
                ORDER BY id ASC
                """
            )

            rows = (
                cursor.fetchall()
            )

            return [
                dict(row)
                for row in rows
            ]

    def mark_manual_posts_processed(
        self,
        ids: List[int],
    ):
        ids = [
            int(item_id)
            for item_id in ids
            if (
                isinstance(
                    item_id,
                    int,
                )
                or str(
                    item_id
                ).isdigit()
            )
        ]

        if not ids:
            return

        placeholders = ",".join(
            "?"
            for _ in ids
        )

        with sqlite3.connect(
            self.db_path
        ) as conn:

            conn.execute(
                f"""
                UPDATE manual_news_queue
                SET processed = 1
                WHERE id IN ({placeholders})
                """,
                ids,
            )

            conn.commit()

    # ═════════════════════════════════════════════
    # PUBLISHED NEWS HISTORY
    # ═════════════════════════════════════════════

    def is_published(
        self,
        channel_name: str,
        message_id: int,
    ) -> bool:

        normalized = (
            self._normalize_channel_name(
                channel_name
            )
        )

        with sqlite3.connect(
            self.db_path
        ) as conn:

            cursor = (
                conn.cursor()
            )

            # Порівняння нормалізоване прямо в SQL,
            # тому старі записи з @ або іншим
            # регістром теж продовжують працювати.
            cursor.execute(
                """
                SELECT 1
                FROM published_news
                WHERE
                    LOWER(
                        REPLACE(
                            TRIM(channel_name),
                            '@',
                            ''
                        )
                    ) = ?
                    AND message_id = ?
                LIMIT 1
                """,
                (
                    normalized,
                    int(
                        message_id
                    ),
                ),
            )

            return (
                cursor.fetchone()
                is not None
            )

    def mark_as_published(
        self,
        channel_name: str,
        message_id: int,
        title: str = "",
        summary: str = "",
        category: str = "",
    ):
        normalized = (
            self._normalize_channel_name(
                channel_name
            )
        )

        if not normalized:
            normalized = "unknown"

        now_str = (
            datetime.now(
                timezone.utc
            )
            .strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        )

        try:
            with sqlite3.connect(
                self.db_path
            ) as conn:

                conn.execute(
                    """
                    INSERT INTO published_news
                    (
                        channel_name,
                        message_id,
                        published_title,
                        summary,
                        category,
                        published_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?)

                    ON CONFLICT(
                        channel_name,
                        message_id
                    )

                    DO UPDATE SET
                        published_title =
                            excluded.published_title,
                        summary =
                            excluded.summary,
                        category =
                            excluded.category,
                        published_at =
                            excluded.published_at
                    """,
                    (
                        normalized,
                        int(
                            message_id
                        ),
                        title.strip(),
                        summary.strip(),
                        category.strip(),
                        now_str,
                    ),
                )

                conn.commit()

        except Exception as exc:
            logger.warning(
                "Не вдалося записати "
                f"історію: {exc}"
            )

    def get_recent_events(
        self,
        hours: int = 48,
    ) -> List[Dict[str, str]]:
        """
        Повертає семантичну історію подій.

        Одна опублікована подія може бути прив'язана
        до кількох Telegram-повідомлень.

        GROUP BY не дозволяє такій події дублюватися
        в архівному блоці для Analyzer.
        """

        threshold = (
            datetime.now(
                timezone.utc
            )
            - timedelta(
                hours=hours
            )
        ).strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        with sqlite3.connect(
            self.db_path
        ) as conn:

            cursor = (
                conn.cursor()
            )

            cursor.execute(
                """
                SELECT
                    published_title,
                    summary,
                    category,
                    MAX(
                        published_at
                    ) AS latest_published_at

                FROM published_news

                WHERE
                    published_at > ?
                    AND (
                        published_title != ''
                        OR summary != ''
                    )

                GROUP BY
                    published_title,
                    summary,
                    category

                ORDER BY
                    latest_published_at DESC
                """,
                (
                    threshold,
                ),
            )

            rows = (
                cursor.fetchall()
            )

            return [
                {
                    "title": (
                        row[0]
                        or ""
                    ),
                    "summary": (
                        row[1]
                        or ""
                    ),
                    "category": (
                        row[2]
                        or ""
                    ),
                    "published_at": (
                        row[3]
                        or ""
                    ),
                }
                for row in rows
            ]

    # ═════════════════════════════════════════════
    # QUALITY AUDIT
    # ═════════════════════════════════════════════

    def save_quality_audit(
        self,
        audit_result: Dict[
            str,
            Any,
        ],
    ) -> int:
        """
        Зберігає результат одного read-only audit.

        Помилка запису audit НІКОЛИ не повинна
        ламати основний новинний цикл.
        """

        if not isinstance(
            audit_result,
            dict,
        ):
            return 0

        now_str = (
            datetime.now(
                timezone.utc
            )
            .strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        )

        try:
            issues_json = (
                json.dumps(
                    audit_result.get(
                        "issues",
                        [],
                    ),
                    ensure_ascii=False,
                )
            )

            fact_check_stats_json = (
                json.dumps(
                    audit_result.get(
                        "fact_check_stats",
                        {},
                    ),
                    ensure_ascii=False,
                )
            )

        except Exception:
            issues_json = "[]"
            fact_check_stats_json = "{}"

        try:
            with sqlite3.connect(
                self.db_path
            ) as conn:

                cursor = (
                    conn.cursor()
                )

                cursor.execute(
                    """
                    INSERT INTO quality_audit_runs
                    (
                        cycle_started_at,
                        status,
                        published_count,
                        possible_duplicates,
                        reference_issues,
                        media_issues,
                        fact_check_issues,
                        priority_issues,
                        elapsed_ms,
                        issues_json,
                        fact_check_stats_json,
                        created_at
                    )
                    VALUES (
                        ?, ?, ?, ?, ?, ?,
                        ?, ?, ?, ?, ?, ?
                    )
                    """,
                    (
                        str(
                            audit_result.get(
                                "cycle_started_at"
                            )
                            or ""
                        ),
                        str(
                            audit_result.get(
                                "status"
                            )
                            or "UNKNOWN"
                        ),
                        self._safe_int(
                            audit_result.get(
                                "published_count"
                            )
                        ),
                        self._safe_int(
                            audit_result.get(
                                "possible_duplicates"
                            )
                        ),
                        self._safe_int(
                            audit_result.get(
                                "reference_issues"
                            )
                        ),
                        self._safe_int(
                            audit_result.get(
                                "media_issues"
                            )
                        ),
                        self._safe_int(
                            audit_result.get(
                                "fact_check_issues"
                            )
                        ),
                        self._safe_int(
                            audit_result.get(
                                "priority_issues"
                            )
                        ),
                        self._safe_int(
                            audit_result.get(
                                "elapsed_ms"
                            )
                        ),
                        issues_json,
                        fact_check_stats_json,
                        now_str,
                    ),
                )

                audit_id = int(
                    cursor.lastrowid
                )

                conn.commit()

            # Audit-memory пишемо окремо.
            # Якщо одна пара повториться пізніше,
            # hit_count збільшиться.
            memory_items = (
                audit_result.get(
                    "memory_observations"
                )
            )

            if isinstance(
                memory_items,
                list,
            ):
                for observation in (
                    memory_items
                ):
                    if isinstance(
                        observation,
                        dict,
                    ):
                        self.record_audit_memory(
                            observation
                        )

            return audit_id

        except Exception as exc:
            logger.warning(
                "QUALITY AUDIT: "
                "не вдалося записати "
                f"результат у БД: {exc}"
            )

            return 0

    def record_audit_memory(
        self,
        observation: Dict[
            str,
            Any,
        ],
    ):
        """
        Запам'ятовує підтверджений audit-патерн.

        Якщо pair_key уже існує:
        - hit_count += 1
        - last_seen_at оновлюється
        - confidence/reason/title оновлюються

        ВАЖЛИВО:
        ця пам'ять поки НЕ впливає на dedup.
        """

        if not isinstance(
            observation,
            dict,
        ):
            return

        pair_key = str(
            observation.get(
                "pair_key"
            )
            or ""
        ).strip()

        if not pair_key:
            return

        relation = str(
            observation.get(
                "relation"
            )
            or ""
        ).strip()

        current_title = str(
            observation.get(
                "current_title"
            )
            or ""
        ).strip()

        matched_title = str(
            observation.get(
                "matched_title"
            )
            or ""
        ).strip()

        confidence = (
            self._safe_int(
                observation.get(
                    "confidence"
                )
            )
        )

        reason = str(
            observation.get(
                "reason"
            )
            or ""
        ).strip()

        now_str = (
            datetime.now(
                timezone.utc
            )
            .strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        )

        try:
            with sqlite3.connect(
                self.db_path
            ) as conn:

                conn.execute(
                    """
                    INSERT INTO quality_audit_memory
                    (
                        pair_key,
                        relation,
                        current_title,
                        matched_title,
                        confidence,
                        reason,
                        hit_count,
                        first_seen_at,
                        last_seen_at
                    )
                    VALUES (
                        ?, ?, ?, ?, ?, ?,
                        1, ?, ?
                    )

                    ON CONFLICT(pair_key)

                    DO UPDATE SET
                        relation =
                            excluded.relation,

                        current_title =
                            excluded.current_title,

                        matched_title =
                            excluded.matched_title,

                        confidence =
                            MAX(
                                quality_audit_memory.confidence,
                                excluded.confidence
                            ),

                        reason =
                            excluded.reason,

                        hit_count =
                            quality_audit_memory.hit_count
                            + 1,

                        last_seen_at =
                            excluded.last_seen_at
                    """,
                    (
                        pair_key,
                        relation,
                        current_title,
                        matched_title,
                        confidence,
                        reason,
                        now_str,
                        now_str,
                    ),
                )

                conn.commit()

        except Exception as exc:
            logger.warning(
                "QUALITY AUDIT MEMORY: "
                "не вдалося записати "
                f"спостереження: {exc}"
            )

    def get_recent_audit_runs(
        self,
        limit: int = 30,
    ) -> List[Dict[str, Any]]:
        """
        Допоміжний метод.

        Зараз pipeline його не використовує,
        але він дозволить нам через кілька днів
        легко подивитися статистику audit.
        """

        safe_limit = max(
            1,
            min(
                self._safe_int(
                    limit,
                    30,
                ),
                200,
            ),
        )

        with sqlite3.connect(
            self.db_path
        ) as conn:

            conn.row_factory = (
                sqlite3.Row
            )

            cursor = (
                conn.cursor()
            )

            cursor.execute(
                """
                SELECT *
                FROM quality_audit_runs
                ORDER BY id DESC
                LIMIT ?
                """,
                (
                    safe_limit,
                ),
            )

            rows = (
                cursor.fetchall()
            )

        result = []

        for row in rows:
            item = dict(
                row
            )

            try:
                item["issues"] = (
                    json.loads(
                        item.get(
                            "issues_json"
                        )
                        or "[]"
                    )
                )

            except Exception:
                item["issues"] = []

            try:
                item[
                    "fact_check_stats"
                ] = json.loads(
                    item.get(
                        "fact_check_stats_json"
                    )
                    or "{}"
                )

            except Exception:
                item[
                    "fact_check_stats"
                ] = {}

            result.append(
                item
            )

        return result

    def get_audit_memory(
        self,
        min_hits: int = 1,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Повертає накопичену audit-memory.

        Поки це тільки діагностика.
        Summarizer її ЩЕ НЕ використовує.
        """

        safe_min_hits = max(
            1,
            self._safe_int(
                min_hits,
                1,
            ),
        )

        safe_limit = max(
            1,
            min(
                self._safe_int(
                    limit,
                    100,
                ),
                500,
            ),
        )

        with sqlite3.connect(
            self.db_path
        ) as conn:

            conn.row_factory = (
                sqlite3.Row
            )

            cursor = (
                conn.cursor()
            )

            cursor.execute(
                """
                SELECT *
                FROM quality_audit_memory
                WHERE hit_count >= ?
                ORDER BY
                    hit_count DESC,
                    confidence DESC,
                    last_seen_at DESC
                LIMIT ?
                """,
                (
                    safe_min_hits,
                    safe_limit,
                ),
            )

            rows = (
                cursor.fetchall()
            )

            return [
                dict(
                    row
                )
                for row in rows
            ]

    # ═════════════════════════════════════════════
    # CLEANUP
    # ═════════════════════════════════════════════

    def cleanup_old_records(
        self,
        days: int = 5,
    ):
        """
        Стару новинну історію чистимо як і раніше.

        Audit-runs тримаємо довше — 30 днів,
        щоб реально накопичити статистику.

        Audit-memory тримаємо 60 днів.
        """

        threshold = (
            datetime.now(
                timezone.utc
            )
            - timedelta(
                days=days
            )
        ).strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        audit_threshold = (
            datetime.now(
                timezone.utc
            )
            - timedelta(
                days=30
            )
        ).strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        memory_threshold = (
            datetime.now(
                timezone.utc
            )
            - timedelta(
                days=60
            )
        ).strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        with sqlite3.connect(
            self.db_path
        ) as conn:

            conn.execute(
                """
                DELETE FROM published_news
                WHERE published_at < ?
                """,
                (
                    threshold,
                ),
            )

            conn.execute(
                """
                DELETE FROM manual_news_queue
                WHERE
                    processed = 1
                    AND created_at < ?
                """,
                (
                    threshold,
                ),
            )

            conn.execute(
                """
                DELETE FROM quality_audit_runs
                WHERE created_at < ?
                """,
                (
                    audit_threshold,
                ),
            )

            conn.execute(
                """
                DELETE FROM quality_audit_memory
                WHERE last_seen_at < ?
                """,
                (
                    memory_threshold,
                ),
            )

            conn.commit()

    # ═════════════════════════════════════════════
    # HELPERS
    # ═════════════════════════════════════════════

    @staticmethod
    def _safe_int(
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
