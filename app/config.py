from typing import List, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    TG_API_ID: int
    TG_API_HASH: str
    BOT_TOKEN: str
    TARGET_CHANNEL_ID: str
    GEMINI_API_KEY: str
    SOURCE_CHANNELS: str

    # Краще тримати ID адміністратора в .env, а не в main.py.
    ADMIN_TELEGRAM_ID: Optional[int] = None

    # Instagram API
    INSTAGRAM_ACCOUNT_ID: Optional[str] = None
    INSTAGRAM_ACCESS_TOKEN: Optional[str] = None
    MEDIA_BASE_URL: Optional[str] = None

    @property
    def source_channels_list(self) -> List[str]:
        """
        Повертає унікальний список каналів без @.
        Регістр зберігаємо для Telethon, але дублікати
        прибираємо без урахування регістру.
        """
        result: List[str] = []
        seen = set()

        for raw_channel in self.SOURCE_CHANNELS.split(","):
            channel = (
                raw_channel
                .strip()
                .replace("@", "")
            )

            if not channel:
                continue

            key = channel.lower()

            if key in seen:
                continue

            seen.add(key)
            result.append(channel)

        return result

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
