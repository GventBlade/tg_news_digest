from typing import List, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    TG_API_ID: int
    TG_API_HASH: str
    BOT_TOKEN: str
    TARGET_CHANNEL_ID: str
    GEMINI_API_KEY: str
    SOURCE_CHANNELS: str

    # Instagram API
    INSTAGRAM_ACCOUNT_ID: Optional[str] = None
    INSTAGRAM_ACCESS_TOKEN: Optional[str] = None

    @property
    def source_channels_list(self) -> List[str]:
        """Повертає список каналів-донорів у вигляді списку рядків."""
        return [
            ch.strip().replace("@", "")
            for ch in self.SOURCE_CHANNELS.split(",")
            if ch.strip()
        ]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()
