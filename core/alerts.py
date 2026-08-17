"""
AlertManager - Telegram / Discord notifications via webhook.

Both channels are simple HTTPS POSTs:
  * Telegram: POST https://api.telegram.org/bot<TOKEN>/sendMessage
              payload {"chat_id": ..., "text": ...}
  * Discord:  POST <WEBHOOK_URL>
              payload {"content": ...}

Sending runs via urllib inside a worker thread so the async loop never blocks.
Alerts are best-effort: failures are logged, never raised - an alert problem
must never take the trading loop down.
"""
import asyncio
import json
import logging
import urllib.request

logger = logging.getLogger(__name__)


class AlertManager:
    def __init__(self, config):
        self.config = config

    async def send(self, message: str) -> None:
        channel = self.config.ALERT_CHANNEL
        if channel == "telegram" and self.config.TELEGRAM_BOT_TOKEN and self.config.TELEGRAM_CHAT_ID:
            url = f"https://api.telegram.org/bot{self.config.TELEGRAM_BOT_TOKEN}/sendMessage"
            payload = {"chat_id": self.config.TELEGRAM_CHAT_ID, "text": message}
            await self._post(url, payload)
        elif channel == "discord" and self.config.DISCORD_WEBHOOK_URL:
            await self._post(self.config.DISCORD_WEBHOOK_URL, {"content": message})

    async def _post(self, url: str, payload: dict) -> None:
        data = json.dumps(payload).encode("utf-8")
        await asyncio.to_thread(self._post_sync, url, data)

    def _post_sync(self, url: str, data: bytes) -> None:
        req = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                if resp.status >= 300:
                    logger.warning("Alert POST returned HTTP %s", resp.status)
        except Exception as exc:
            logger.warning("Alert delivery failed: %s", exc)

    # ------------------------------------------------------------------
    # domain events
    # ------------------------------------------------------------------
    async def startup(self, server: str, login: int, balance: float) -> None:
        await self.send(f"[START] bot connected to {server} login={login} "
                        f"balance={balance:.2f}")

    async def trade_entry(self, symbol: str, direction: int, volume: float,
                          price: float, sl: float, tp: float) -> None:
        await self.send(f"[ENTRY] {symbol} {'BUY' if direction == 1 else 'SELL'} "
                        f"volume={volume} price={price:.5f} sl={sl:.5f} tp={tp:.5f}")

    async def trade_exit(self, symbol: str, ticket: int, profit: float) -> None:
        await self.send(f"[EXIT] {symbol} ticket={ticket} pnl={profit:.2f}")

    async def trailing_update(self, symbol: str, ticket: int, new_sl: float) -> None:
        await self.send(f"[TRAIL] {symbol} ticket={ticket} stop moved to {new_sl:.5f}")

    async def crash(self, error: str) -> None:
        await self.send(f"[CRASH] bot stopped unexpectedly: {error}")

    async def shutdown(self) -> None:
        await self.send("[STOP] bot shut down cleanly")
