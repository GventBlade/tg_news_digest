import asyncio
import logging
from app.services.publisher import NewsPublisher

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


async def test_publish():
    publisher = NewsPublisher()

    caption = (
        "🔥 ТОП ГОЛОВНИХ НОВИН\n"
        "🕒 Станом на 14:00, 31.08.2026\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        "1. 💥 Перша тестова новина дайджесту\n"
        "2. 🏛 Друга тестова новина дайджесту\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        "📲 Якщо хочете знати більше про ці та інші новини — підписуйтесь на наш Telegram-канал «Новини UA 6/24»:\n"
        "👉 https://t.me/news_ua_624\n\n"
        "#новини #україна #тест"
    )

    test_media = [
        {"url": "https://images.unsplash.com/photo-1585829365295-ab7cd400c167?w=1080&q=80", "type": "photo"},
        {"url": "https://images.unsplash.com/photo-1504711434969-e33886168f5c?w=1080&q=80", "type": "photo"}
    ]

    print("⏳ Відправка тестового поста в Instagram...")
    await publisher.publish_instagram_carousel(caption=caption, media_items=test_media)
    await publisher.close()
    print("🏁 Тест завершено! Перевірте сторінку в Instagram.")


if __name__ == "__main__":
    asyncio.run(test_publish())
