import os
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

    def generate_news_image(self, headline: str, summary: str, news_id: int) -> str | None:
        """Генерує тематичне фотореалістичне зображення через Imagen 3."""
        try:
            # Складаємо промт англійською мовою для кращої якості генерації
            prompt = (
                f"Editorial photojournalism style, realistic documentary shot of: {headline}. "
                f"Context: {summary}. High resolution, 8k, cinematic, realistic lighting, "
                f"no text, no watermark, no artifacts, neutral news documentary photography."
            )

            logger.info(f"Генерація AI-зображення для новини: {headline}...")
            result = self.client.models.generate_images(
                model="imagen-3.0-generate-002",
                prompt=prompt,
                config=types.GenerateImagesConfig(
                    number_of_images=1,
                    output_mime_type="image/jpeg",
                    aspect_ratio="16:9",
                )
            )

            if result.generated_images:
                file_path = os.path.join(DOWNLOAD_DIR, f"ai_gen_{news_id}_{int(os.times().system)}.jpg")
                image = result.generated_images[0]
                with open(file_path, "wb") as f:
                    f.write(image.image.image_bytes)
                logger.info(f"✅ AI-зображення успішно створено: {file_path}")
                return file_path

        except Exception as e:
            logger.warning(f"Не вдалося згенерувати AI-зображення: {e}")

        return None
