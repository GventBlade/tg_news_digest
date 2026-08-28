import os
import time
import logging
from google import genai
from google.genai import types
from app.config import settings

logger = logging.getLogger(__name__)
DOWNLOAD_DIR = "downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)


class AIImageGenerator:
    def __init__(self):
        self.client = genai.Client(api_key=settings.GEMINI_API_KEY)
        self.image_models = [
            "gemini-3.1-flash-image",
            "gemini-3.1-flash-lite-image"
        ]

    def generate_news_image(self, headline: str, summary: str, news_id: int) -> str | None:
        """Генерує швидку фотоілюстрацію стандартної HD якості для Telegram."""
        prompt = (
            f"Editorial news documentary photo: {headline}. "
            f"Context: {summary}. Clean HD photo, realistic natural lighting, "
            f"no text, no letters, no logos, no watermarks, 16:9 ratio."
        )

        for model in self.image_models:
            try:
                logger.info(f"Генерація HD-зображення для '{headline}' через {model}...")
                response = self.client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="image/jpeg"
                    ),
                )

                for candidate in getattr(response, "candidates", []):
                    content = getattr(candidate, "content", None)
                    if not content:
                        continue
                    for part in getattr(content, "parts", []):
                        inline_data = getattr(part, "inline_data", None)
                        if inline_data and getattr(inline_data, "data", None):
                            file_path = os.path.join(DOWNLOAD_DIR, f"ai_{news_id}_{int(time.time())}.jpg")
                            with open(file_path, "wb") as f:
                                f.write(inline_data.data)
                            logger.info(f"✅ HD-зображення створено: {file_path}")
                            return file_path

            except Exception as e:
                logger.warning(f"Не вдалося згенерувати через {model}: {e}")
                continue

        return None
