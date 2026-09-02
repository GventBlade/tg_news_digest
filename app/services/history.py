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
    def _normalize_channel_name(channel_name: str) -> str:
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
        with sqlite3.connect(self.db_path) as conn:
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

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_published_news_published_at
                ON published_news(published_at)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_manual_news_processed
                ON manual_news_queue(processed, created_at)
            """)

            conn.commit()

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
        now_str = datetime.now(timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        with sqlite3.connect(self.db_path) as conn:
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
            return int(cursor.lastrowid)

    def get_pending_manual_posts(
        self,
    ) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT *
                FROM manual_news_queue
                WHERE processed = 0
                ORDER BY id ASC
                """
            )
            rows = cursor.fetchall()
            return [dict(r) for r in rows]

    def mark_manual_posts_processed(
        self,
        ids: List[int],
    ):
        ids = [
            int(item_id)
            for item_id in ids
            if isinstance(item_id, int)
            or str(item_id).isdigit()
        ]

        if not ids:
            return

        placeholders = ",".join(
            "?"
            for _ in ids
        )

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                f"""
                UPDATE manual_news_queue
                SET processed = 1
                WHERE id IN ({placeholders})
                """,
                ids,
            )
            conn.commit()

    def is_published(
        self,
        channel_name: str,
        message_id: int,
    ) -> bool:
        normalized = self._normalize_channel_name(
            channel_name
        )

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Порівняння нормалізоване прямо в SQL, тому старі записи
            # з @ або іншим регістром теж продовжують працювати.
            cursor.execute(
                """
                SELECT 1
                FROM published_news
                WHERE
                    LOWER(REPLACE(TRIM(channel_name), '@', '')) = ?
                    AND message_id = ?
                LIMIT 1
                """,
                (
                    normalized,
                    int(message_id),
                ),
            )

            return cursor.fetchone() is not None

    def mark_as_published(
        self,
        channel_name: str,
        message_id: int,
        title: str = "",
        summary: str = "",
        category: str = "",
    ):
        normalized = self._normalize_channel_name(
            channel_name
        )

        if not normalized:
            normalized = "unknown"

        now_str = datetime.now(timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        try:
            with sqlite3.connect(self.db_path) as conn:
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
                    ON CONFLICT(channel_name, message_id)
                    DO UPDATE SET
                        published_title = excluded.published_title,
                        summary = excluded.summary,
                        category = excluded.category,
                        published_at = excluded.published_at
                    """,
                    (
                        normalized,
                        int(message_id),
                        title.strip(),
                        summary.strip(),
                        category.strip(),
                        now_str,
                    ),
                )
                conn.commit()

        except Exception as e:
            logger.warning(
                f"Не вдалося записати історію: {e}"
            )

    def get_recent_events(
        self,
        hours: int = 48,
    ) -> List[Dict[str, str]]:
        """
        Повертає семантичну історію подій.

        Одна опублікована подія може бути прив'язана до кількох
        Telegram-повідомлень. GROUP BY не дозволяє такій події
        дублюватися в архівному блоці для Analyzer.
        """
        threshold = (
            datetime.now(timezone.utc)
            - timedelta(hours=hours)
        ).strftime("%Y-%m-%d %H:%M:%S")

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT
                    published_title,
                    summary,
                    category,
                    MAX(published_at) AS latest_published_at
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
                ORDER BY latest_published_at DESC
                """,
                (threshold,),
            )

            rows = cursor.fetchall()

            return [
                {
                    "title": row[0] or "",
                    "summary": row[1] or "",
                    "category": row[2] or "",
                    "published_at": row[3] or "",
                }
                for row in rows
            ]

    def cleanup_old_records(
        self,
        days: int = 5,
    ):
        threshold = (
            datetime.now(timezone.utc)
            - timedelta(days=days)
        ).strftime("%Y-%m-%d %H:%M:%S")

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                DELETE FROM published_news
                WHERE published_at < ?
                """,
                (threshold,),
            )

            conn.execute(
                """
                DELETE FROM manual_news_queue
                WHERE
                    processed = 1
                    AND created_at < ?
                """,
                (threshold,),
            )

            conn.commit()
