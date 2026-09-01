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
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(published_news)")
            columns = [col[1] for col in cursor.fetchall()]
            if "published_title" not in columns:
                cursor.execute("ALTER TABLE published_news ADD COLUMN published_title TEXT DEFAULT ''")
            if "summary" not in columns:
                cursor.execute("ALTER TABLE published_news ADD COLUMN summary TEXT DEFAULT ''")
            if "category" not in columns:
                cursor.execute("ALTER TABLE published_news ADD COLUMN category TEXT DEFAULT ''")
            conn.commit()

    def is_published(self, channel_name: str, message_id: int) -> bool:
        """Перевіряє, чи публікувався вже цей конкретний пост."""
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
        """Зберігає факт публікації з заголовком та контекстом події."""
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
        """Отримує список опублікованих подій за останні N годин у порядку від найновіших."""
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
        """Видаляє записи старші за 5 днів."""
        threshold = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM published_news WHERE published_at < ?", (threshold,))
            conn.commit()
