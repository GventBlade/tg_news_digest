import asyncio
import logging
import os
import aiohttp
from PIL import Image, ImageDraw
from app.config import settings
from app.services.publisher import NewsPublisher

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TEST")

def generate_test_image(path: str):
    """Створює просте тестове зображення 1080x1080 (стандарт Instagram)"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img = Image.new("RGB", (1080, 1080), color=(30, 144, 255))
    draw = ImageDraw.Draw(img)
    draw.text((350, 520), "Test Media from Oracle VPS", fill=(255, 255, 255))
    img.save(path, "JPEG")
    logger.info(f"✅ Тестове зображення збережено: {path}")

async def run_tests():
    logger.info("=== 1. ПЕРЕВІРКА НАЛАШТУВАНЬ (.env / Settings) ===")
    try:
        media_base = settings.MEDIA_BASE_URL
        ig_id = settings.INSTAGRAM_ACCOUNT_ID
        bot_token = settings.BOT_TOKEN
        logger.info(f"MEDIA_BASE_URL: {media_base}")
        logger.info(f"INSTAGRAM_ACCOUNT_ID: {ig_id}")
        logger.info("✅ Settings зчитано без помилок!")
    except Exception as e:
        logger.error(f"❌ Помилка в Settings: {e}")
        return

    test_file_path = "downloads/test_probe.jpg"
    generate_test_image(test_file_path)

    publisher = NewsPublisher()
    public_url = publisher.create_public_media_url(test_file_path)
    logger.info(f"Згенеровано публічний URL: {public_url}")

    logger.info("\n=== 2. ПЕРЕВІРКА ПУБЛІЧНОЇ ДОСТУПНОСТІ ЧЕРЕЗ NGINX (HTTPS) ===")
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(public_url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status == 200:
                    logger.info(f"✅ Nginx віддає файл! HTTP Status: {resp.status}, Content-Type: {resp.headers.get('Content-Type')}")
                else:
                    logger.error(f"❌ Nginx повернув статус {resp.status}. Перевірте конфігурацію /etc/nginx/sites-available/default")
                    return
        except Exception as e:
            logger.error(f"❌ Не вдалося отримати доступ за URL {public_url}: {e}")
            return

        logger.info("\n=== 3. ПЕРЕВІРКА ЗВ'ЯЗКУ З INSTAGRAM GRAPH API ===")
        if not ig_id or not settings.INSTAGRAM_ACCESS_TOKEN:
            logger.warning("Instagram параметри не заповнені, пропускаємо тест Meta API.")
            return

        logger.info("Відправляємо тестовий запит на створення контейнера в Meta...")
        container_id = await publisher._create_container(
            session=session,
            media_url=public_url,
            media_type="photo",
            caption="Test post (dry run)",
        )

        if not container_id:
            logger.error("❌ Meta не змогла створити контейнер (перевірте токен або доступність URL).")
            return

        logger.info(f"✅ Контейнер створено в Meta! ID: {container_id}")
        logger.info("Очікуємо обробки медіа серверами Instagram...")

        ready = await publisher._wait_for_container(session, container_id, timeout=60)
        if ready:
            logger.info("🎉 ВСЕ ПРАЦЮЄ ІДЕАЛЬНО! Meta успішно скачала і обробила зображення з вашого Oracle Nginx.")
        else:
            logger.error("❌ Meta не завершила обробку медіа або виникла помилка завантаження.")

    # Очищення
    if os.path.exists(test_file_path):
        os.remove(test_file_path)
        logger.info(f"Тестовий файл {test_file_path} видалено.")

if __name__ == "__main__":
    asyncio.run(run_tests())
