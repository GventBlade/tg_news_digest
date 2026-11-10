import os
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, call, patch

import aiohttp
from aiogram.exceptions import TelegramBadRequest, TelegramNetworkError
from aiogram.methods import SendMessage

with patch.dict(os.environ, {
    "TG_API_ID": "1", "TG_API_HASH": "test", "BOT_TOKEN": "1:test",
    "TARGET_CHANNEL_ID": "@test", "GEMINI_API_KEY": "test",
    "SOURCE_CHANNELS": "test",
}):
    from app.services.publisher import NewsPublisher


class TelegramHeaderRetryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.publisher = NewsPublisher.__new__(NewsPublisher)
        self.send = AsyncMock()
        self.publisher.bot = SimpleNamespace(send_message=self.send)
        self.method = SendMessage(chat_id="@test", text="Digest header")

    def connection_reset(self):
        error = TelegramNetworkError(self.method, "Connection reset by peer")
        error.__cause__ = aiohttp.ClientConnectorError(
            SimpleNamespace(host="api.telegram.org", port=443, ssl=True),
            ConnectionResetError(104, "Connection reset by peer"),
        )
        return error

    async def test_sends_immediately_when_telegram_is_reachable(self):
        with patch("app.services.publisher.asyncio.sleep", new_callable=AsyncMock) as sleep:
            self.assertTrue(await self.publisher.publish_telegram_header("Digest header"))
        self.send.assert_awaited_once()
        sleep.assert_not_awaited()

    async def test_retries_connection_reset_with_same_header_until_success(self):
        self.send.side_effect = [self.connection_reset(), self.connection_reset(), None]
        with patch("app.services.publisher.asyncio.sleep", new_callable=AsyncMock) as sleep:
            self.assertTrue(await self.publisher.publish_telegram_header("Digest header"))
        self.assertEqual(self.send.await_count, 3)
        self.assertEqual(self.send.await_args_list, [self.send.await_args_list[0]] * 3)
        self.assertEqual(sleep.await_args_list, [call(60), call(60)])

    async def test_stops_after_five_failed_connections(self):
        self.send.side_effect = [self.connection_reset() for _ in range(5)]
        with patch("app.services.publisher.asyncio.sleep", new_callable=AsyncMock) as sleep:
            self.assertFalse(await self.publisher.publish_telegram_header("Digest header"))
        self.assertEqual(self.send.await_count, 5)
        self.assertEqual(sleep.await_args_list, [call(60)] * 4)

    async def test_does_not_repeat_a_request_that_may_have_been_accepted(self):
        error = TelegramNetworkError(self.method, "Response lost")
        error.__cause__ = aiohttp.ServerDisconnectedError()
        self.send.side_effect = error
        with patch("app.services.publisher.asyncio.sleep", new_callable=AsyncMock) as sleep:
            self.assertFalse(await self.publisher.publish_telegram_header("Digest header"))
        self.send.assert_awaited_once()
        sleep.assert_not_awaited()

    async def test_does_not_retry_invalid_header(self):
        self.send.side_effect = TelegramBadRequest(self.method, "Invalid HTML")
        with patch("app.services.publisher.asyncio.sleep", new_callable=AsyncMock) as sleep:
            self.assertFalse(await self.publisher.publish_telegram_header("Digest header"))
        self.send.assert_awaited_once()
        sleep.assert_not_awaited()
