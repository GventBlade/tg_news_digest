import sqlite3
import os
from datetime import datetime, timedelta, timezone

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
                    published_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(channel_name, message_id)
                )
            """)
            conn.commit()

    def is_published(self, channel_name: str, message_id: int) -> bool:
        """Перевіряє, чи публікувався вже цей конкретний пост."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT 1 FROM published_news WHERE channel_name = ? AND message_id = ?",
                (channel_name, message_id)
            )
            return cursor.fetchone() is not None

    def mark_as_published(self, channel_name: str, message_id: int):
        """Зберігає факт публікації."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT OR IGNORE INTO published_news (channel_name, message_id) VALUES (?, ?)",
                    (channel_name, message_id)
                )
                conn.commit()
        except Exception:
            pass

    def cleanup_old_records(self, days: int = 2):
        """Видаляє записи старші за 2 дні, щоб база завжди була крихітною."""
        threshold = datetime.now(timezone.utc) - timedelta(days=days)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM published_news WHERE published_at < ?", (threshold,))
            conn.commit()
