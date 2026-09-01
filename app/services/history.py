import sqlite3
import os
import logging
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any

logger = logging.getLogger(__name__)
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
DB_PATH = os.path.join(DATA_DIR, "news_history.db")


class NewsHistory:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            # Таблиця опублікованих новин
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
            # Таблиця ручної черги пересланих новин від адміна
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
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO manual_news_queue 
                (raw_text, channel_title, channel_username, media_path, media_type, has_media, has_video, created_at)
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
                    now_str
                )
            )
            conn.commit()
            return cursor.lastrowid

    def get_pending_manual_posts(self) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM manual_news_queue WHERE processed = 0 ORDER BY id ASC")
            rows = cursor.fetchall()
            return [dict(r) for r in rows]

    def mark_manual_posts_processed(self, ids: List[int]):
        if not ids:
            return
        placeholders = ",".join("?" for _ in ids)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(f"UPDATE manual_news_queue SET processed = 1 WHERE id IN ({placeholders})", ids)
            conn.commit()

    def is_published(self, channel_name: str, message_id: int) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT 1 FROM published_news WHERE channel_name = ? AND message_id = ?",
                (str(channel_name), message_id)
            )
            return cursor.fetchone() is not None

    def mark_as_published(
        self,
        channel_name: str,
        message_id: int,
        title: str = "",
        summary: str = "",
        category: str = ""
    ):
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO published_news 
                    (channel_name, message_id, published_title, summary, category, published_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (str(channel_name), message_id, title.strip(), summary.strip(), category.strip(), now_str)
                )
                conn.commit()
        except Exception as e:
            logger.warning(f"Не вдалося записати історію: {e}")

    def get_recent_events(self, hours: int = 48) -> List[Dict[str, str]]:
        threshold = (datetime.now(timezone.utc) - timedelta(hours=hours)).strftime("%Y-%m-%d %H:%M:%S")
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT published_title, summary, category, published_at
                FROM published_news 
                WHERE published_at > ? AND (published_title != '' OR summary != '')
                ORDER BY published_at DESC
                """,
                (threshold,)
            )
            rows = cursor.fetchall()
            return [
                {
                    "title": r[0] or "",
                    "summary": r[1] or "",
                    "category": r[2] or "",
                    "published_at": r[3] or ""
                }
                for r in rows
            ]

    def cleanup_old_records(self, days: int = 5):
        threshold = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM published_news WHERE published_at < ?", (threshold,))
            conn.execute("DELETE FROM manual_news_queue WHERE processed = 1 AND created_at < ?", (threshold,))
            conn.commit()
