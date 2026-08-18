"""
CLI entry point:  python main.py

Same engine as the GUI version (core/bot.py), just without the PyQt5
dashboard. For the GUI / Windows .exe, run:  python gui_main.py
"""
import asyncio
import logging
import logging.handlers
import os

import config
from core.bot import TradingBot

logger = logging.getLogger(__name__)


def setup_logging(cfg) -> None:
    os.makedirs(cfg.LOG_DIR, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)-8s [%(name)s] %(message)s")
    root = logging.getLogger()
    root.setLevel(getattr(logging, cfg.LOG_LEVEL, logging.INFO))

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    root.addHandler(console)

    file_handler = logging.handlers.RotatingFileHandler(
        os.path.join(cfg.LOG_DIR, cfg.LOG_FILE),
        maxBytes=10_000_000, backupCount=5)
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)

    # Keep the MetaTrader5 package's own (very noisy) logs quiet.
    logging.getLogger("MetaTrader5").setLevel(logging.ERROR)


async def main() -> None:
    bot = TradingBot(config)
    try:
        await bot.run_until_stopped()
    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("Shutdown requested (Ctrl+C)")
    except Exception as exc:
        logger.exception("Fatal error in main loop")
        await bot.alerts.crash(repr(exc))
    finally:
        await bot.shutdown()
        logger.info("Bot stopped cleanly")


if __name__ == "__main__":
    setup_logging(config)
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
