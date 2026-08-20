import asyncio
import logging
from main import process_and_publish_news_cycle

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

if __name__ == "__main__":
    print("--- Запуск тестового новинного циклу прямо зараз ---")
    asyncio.run(process_and_publish_news_cycle())
