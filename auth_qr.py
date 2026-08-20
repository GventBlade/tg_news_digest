import asyncio
import getpass
import qrcode
from telethon import TelegramClient
from telethon.errors import SessionPasswordNeededError
from app.config import settings


async def main():
    client = TelegramClient("news_session", settings.TG_API_ID, settings.TG_API_HASH)
    await client.connect()

    if not await client.is_user_authorized():
        print("\n--- Сканування QR-коду ---")
        print("1. Відкрийте Telegram на телефоні.")
        print("2. Перейдіть у Налаштування -> Пристрої -> Підключити пристрій.")
        print("3. Відскануйте QR-код нижче:\n")

        qr_login = await client.qr_login()

        qr = qrcode.QRCode()
        qr.add_data(qr_login.url)
        qr.print_ascii(invert=True)

        print("\nОчікування сканування...")
        try:
            await qr_login.wait()
        except SessionPasswordNeededError:
            # Запитуємо хмарний пароль 2FA
            pwd = input("Введіть ваш пароль двоетапної автентифікації (2FA): ")
            await client.sign_in(password=pwd)

        print("\nАвторизація успішна! Сесію збережено.")
    else:
        print("Клієнт уже успішно авторизований!")

    await client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
