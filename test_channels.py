import asyncio
from telethon import TelegramClient
from app.config import settings

# Список каналів для тестування (існуючі + нові виправлені)
CHANNELS_TO_CHECK = [
    # Поточні канали
    "suspilnenews", "ukrpravda_news", "hromadske_ua", "babel", "nvua_official",
    "liganet", "uniannet", "truexanewsua", "insiderUKR", "voynareal",
    "DeepStateUA", "kievreal1", "suspilnelviv", "suspilneodesa", "dnipro_dnepr2",
    "kharkivlife", "operativnoZSU", "gruntmedia", "berezoview", "interfaxua",
    "milinua", "ukrinform_news", "bbcukrainian", "znua_live",

    # Нові якісні джерела (з виправленими юзернеймами)
    "radiosvoboda", "espresotb", "lbua_official", "textyorgua", "Novynarnia", "slovo_i_dilo"
]


async def check_channels():
    client = TelegramClient("news_session", settings.TG_API_ID, settings.TG_API_HASH)
    await client.start()

    valid_channels = []
    failed_channels = []

    print(f"\n🔍 Перевірка {len(CHANNELS_TO_CHECK)} каналів...\n" + "=" * 50)
    for ch in CHANNELS_TO_CHECK:
        try:
            entity = await client.get_entity(ch)
            title = getattr(entity, "title", "No Title")
            username = getattr(entity, "username", ch)
            print(f"✅ [OK] @{username:<20} | Назва: {title[:30]}")
            valid_channels.append(username)
        except Exception as e:
            print(f"❌ [ПОМИЛКА] @{ch:<20} | Причина: {e}")
            failed_channels.append(ch)

    await client.disconnect()

    print("=" * 50)
    print(f"Успішно перевірено: {len(valid_channels)}/{len(CHANNELS_TO_CHECK)}")
    if failed_channels:
        print(f"Не вдалося знайти: {failed_channels}")

    print("\n📋 Готовий SOURCE_CHANNELS для твого .env файлу:")
    print("SOURCE_CHANNELS=" + ",".join(valid_channels) + "\n")


if __name__ == "__main__":
    asyncio.run(check_channels())
