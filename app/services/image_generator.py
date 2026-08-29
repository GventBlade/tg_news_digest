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
        """Генерує фотореалістичну репортажну ілюстрацію для Telegram."""
        prompt = (
            f"Photorealistic editorial news photograph illustrating this event: {headline}. "
            f"Context: {summary}. "
            f"Create a realistic documentary-style photograph suitable for a reputable news publication. "
            f"Focus on the main subject, location, event, or situation described in the news. "
            f"Use natural lighting, realistic colors, authentic environments, believable human behavior, "
            f"accurate proportions, natural composition, and realistic camera perspective. "
            f"Do not invent specific people, quotes, brands, locations, or events that are not supported by the context. "
            f"Avoid sensationalism, cinematic effects, exaggerated destruction, fantasy elements, "
            f"staged-looking scenes, or misleading visual details. "
            f"Professional photojournalism, high detail, sharp main subject, subtle photographic imperfections. "
            f"No text, captions, letters, numbers, logos, watermarks, UI elements, or readable writing. "
            f"Non-graphic and respectful photojournalism. "
            f"Landscape 16:9 composition."
        )

        for model in self.image_models:
            try:
                logger.info(f"Генерація AI-зображення для '{headline}' через {model}...")
                response = self.client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_modalities=["IMAGE"]
                    ),
                )

                # Витягуємо згенеровані байти зображення
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
                            logger.info(f"✅ AI-зображення створено: {file_path}")
                            return file_path

            except Exception as e:
                logger.warning(f"Не вдалося згенерувати через {model}: {e}")
                continue

        return None
