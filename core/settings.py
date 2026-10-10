"""
core/settings.py — Pydantic v2 Settings with full environment validation.

All secrets are loaded from the .env file (or real environment variables).
Nothing sensitive is ever hard-coded here.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class TradingMode(str, Enum):
    PAPER = "paper"
    LIVE = "live"
    BACKTEST = "backtest"


class LLMProvider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"
    NONE = "none"


class AlertChannel(str, Enum):
    TELEGRAM = "telegram"
    DISCORD = "discord"
    NONE = "none"


class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class Settings(BaseSettings):
    """
    Central configuration object.  Loaded once at startup.

    Priority (highest first):
        1. Real environment variables
        2. .env file
        3. Defaults defined here

    NEVER add secrets as defaults — they must come from the environment.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ──────────────────────────────────────────────────────────
    # MetaTrader 5
    # ──────────────────────────────────────────────────────────
    mt5_login: int = Field(default=0, description="MT5 account number")
    mt5_password: str = Field(default="", description="MT5 account password")
    mt5_server: str = Field(default="", description="MT5 broker server name")
    mt5_path: str = Field(default="", description="Path to terminal64.exe (empty = auto-detect)")
    server_utc_offset_hours: int = Field(default=2, description="Broker server UTC offset")

    # ──────────────────────────────────────────────────────────
    # Trading mode
    # ──────────────────────────────────────────────────────────
    trading_mode: TradingMode = Field(default=TradingMode.PAPER, description="paper | live | backtest")

    @property
    def is_live(self) -> bool:
        return self.trading_mode == TradingMode.LIVE

    @property
    def is_paper(self) -> bool:
        return self.trading_mode == TradingMode.PAPER

    # ──────────────────────────────────────────────────────────
    # Instruments (comma-separated list in env)
    # ──────────────────────────────────────────────────────────
    raw_symbols: str = Field(
        default="XAUUSDm",
        validation_alias="SYMBOLS",
        description="Comma-separated list of symbols to trade",
    )

    @property
    def symbols(self) -> list[str]:
        return [s.strip() for s in self.raw_symbols.split(",") if s.strip()]

    # ──────────────────────────────────────────────────────────
    # Database
    # ──────────────────────────────────────────────────────────
    database_url: str = Field(
        default="sqlite:///./trading.db",
        description="SQLAlchemy database URL",
    )

    # ──────────────────────────────────────────────────────────
    # MT5 bridge (Windows VM) — see docs/MT5_WINDOWS_BRIDGE.md
    # ──────────────────────────────────────────────────────────
    # The Linux core never imports MetaTrader5. All broker access goes through
    # an HTTP bridge running on the Windows VM that hosts the MT5 terminal.
    # Leave the URL empty when no bridge is configured (single-host/dev mode);
    # broker-dependent operations must fail safely rather than assume a bridge.
    mt5_bridge_url: str = Field(
        default="",
        description="Base URL of the Windows MT5 bridge, e.g. http://192.168.122.212:8100",
    )
    mt5_bridge_token: str = Field(
        default="",
        description="Shared bearer token for bridge authentication (never logged)",
    )
    mt5_bridge_timeout_seconds: float = Field(default=10.0, ge=0.5, le=120.0)
    mt5_bridge_tls_verify: bool = Field(
        default=True,
        description="Verify TLS certificates. Disable ONLY for a trusted lab bridge.",
    )

    @property
    def bridge_configured(self) -> bool:
        """True when a bridge URL is set and the core may contact a broker."""
        return bool(self.mt5_bridge_url)

    @property
    def mt5_bridge_available(self) -> bool:
        """Alias kept for readability at call sites."""
        return self.bridge_configured

    @property
    def mt5_bridge_http_url(self) -> str:
        """Normalised bridge base URL without a trailing slash."""
        return self.mt5_bridge_url.rstrip("/")

    @property
    def mt5_bridge_auth_headers(self) -> dict[str, str]:
        """Auth headers for bridge calls. Empty when no token is configured."""
        if not self.mt5_bridge_token:
            return {}
        return {"Authorization": f"Bearer {self.mt5_bridge_token}"}

    # ──────────────────────────────────────────────────────────
    # LLM
    # ──────────────────────────────────────────────────────────
    llm_provider: LLMProvider = Field(default=LLMProvider.NONE)
    openai_api_key: str = Field(default="")
    openai_model: str = Field(default="gpt-4o")
    anthropic_api_key: str = Field(default="")
    anthropic_model: str = Field(default="claude-3-5-sonnet-20241022")
    ollama_base_url: str = Field(default="http://localhost:11434")
    ollama_model: str = Field(default="")

    @property
    def llm_enabled(self) -> bool:
        return self.llm_provider != LLMProvider.NONE

    # ──────────────────────────────────────────────────────────
    # Notifications
    # ──────────────────────────────────────────────────────────
    telegram_bot_token: str = Field(default="")
    telegram_chat_id: str = Field(default="")
    discord_webhook_url: str = Field(default="")

    @property
    def telegram_enabled(self) -> bool:
        return bool(self.telegram_bot_token and self.telegram_chat_id)

    # ──────────────────────────────────────────────────────────
    # Dashboard / API
    # ──────────────────────────────────────────────────────────
    dashboard_host: str = Field(default="127.0.0.1")
    dashboard_port: int = Field(default=8000)
    dashboard_secret_key: str = Field(
        default="CHANGE_ME",
        description="JWT signing secret — must be changed before deployment",
    )
    dashboard_username: str = Field(default="admin")
    dashboard_password: str = Field(default="CHANGE_ME")

    # ──────────────────────────────────────────────────────────
    # Logging
    # ──────────────────────────────────────────────────────────
    log_level: LogLevel = Field(default=LogLevel.INFO)
    log_dir: Path = Field(default=Path("logs"))

    # ──────────────────────────────────────────────────────────
    # Risk defaults
    # ──────────────────────────────────────────────────────────
    risk_per_trade: float = Field(default=0.01, ge=0.001, le=0.05)
    max_daily_loss_pct: float = Field(default=0.02, ge=0.005, le=0.20)
    max_drawdown_pct: float = Field(default=0.10, ge=0.01, le=0.50)
    max_open_positions: int = Field(default=5, ge=1, le=50)
    min_rr_ratio: float = Field(default=1.5, ge=0.5, le=10.0)
    max_margin_usage: float = Field(default=0.50, ge=0.10, le=0.90)
    magic_number: int = Field(default=888999)

    # ──────────────────────────────────────────────────────────
    # Strategy scoring
    # ──────────────────────────────────────────────────────────
    min_signal_score: float = Field(default=55.0, ge=0.0, le=100.0)
    ai_confidence_threshold: float = Field(default=0.58, ge=0.50, le=1.0)
    ai_train_bars: int = Field(default=500, ge=100)
    ai_retrain_bars: int = Field(default=50, ge=10)

    # ──────────────────────────────────────────────────────────
    # Execution
    # ──────────────────────────────────────────────────────────
    deviation_points: int = Field(default=30, ge=0)
    max_requote_retries: int = Field(default=5, ge=1)
    retry_backoff_seconds: float = Field(default=0.5, ge=0.1)
    poll_interval_seconds: float = Field(default=5.0, ge=0.5)
    max_spread_points: Optional[float] = Field(default=None)
    max_slippage_pct: float = Field(default=0.001)

    # ──────────────────────────────────────────────────────────
    # ATR / SL / TP defaults
    # ──────────────────────────────────────────────────────────
    sl_atr_mult: float = Field(default=2.0, ge=0.5)
    tp_rr_ratio: float = Field(default=2.5, ge=1.0)
    trail_atr_mult: float = Field(default=3.0, ge=0.5)
    trail_activate_atr: float = Field(default=1.0, ge=0.5)

    # Trailing acceleration
    trail_tighten1_atr: float = Field(default=2.5)
    trail_tighten1_mult: float = Field(default=1.5)
    trail_tighten2_atr: float = Field(default=4.0)
    trail_tighten2_mult: float = Field(default=1.0)

    # Breakeven
    enable_breakeven: bool = Field(default=True)
    breakeven_trigger_atr: float = Field(default=1.0)

    # Partial close
    enable_partial_close: bool = Field(default=True)
    partial_close_trigger_atr: float = Field(default=1.5)
    partial_close_fraction: float = Field(default=0.50)
    partial_close2_trigger_atr: float = Field(default=3.0)
    partial_close2_fraction: float = Field(default=0.25)

    # Hard per-trade stop (backstop)
    max_loss_per_trade_pct: float = Field(default=0.01, ge=0.001, le=0.05)

    # Profit lock / giveback
    profit_lock_activate_atr: float = Field(default=1.2)
    profit_giveback_pct: float = Field(default=0.35)

    # Loss timeout
    cut_losing_after_hours: Optional[float] = Field(default=3.0)

    # Exit rules
    close_on_opposite_signal: bool = Field(default=True)
    opposite_signal_min_score: float = Field(default=65.0)
    enable_trend_flip_close: bool = Field(default=True)
    close_old_profitable: bool = Field(default=True)
    max_position_age_hours: float = Field(default=24.0)
    max_trades_per_day: int = Field(default=8, ge=1)
    equity_protect_close_all: bool = Field(default=True)

    # ADX regime gates
    adx_trending_min: float = Field(default=20.0)
    adx_strong_trend: float = Field(default=30.0)
    dynamic_risk: bool = Field(default=True)
    min_risk_scale: float = Field(default=0.5)
    max_risk_scale: float = Field(default=1.5)

    # Session filter
    use_session_filter: bool = Field(default=True)
    london_start_utc: int = Field(default=7)
    london_end_utc: int = Field(default=16)
    ny_start_utc: int = Field(default=12)
    ny_end_utc: int = Field(default=21)

    # ──────────────────────────────────────────────────────────
    # Consecutive-loss protection
    # ──────────────────────────────────────────────────────────
    max_consecutive_losses: int = Field(default=5, ge=1)
    loss_streak_cooldown_minutes: int = Field(default=60, ge=0)

    # ──────────────────────────────────────────────────────────
    # Watchdog
    # ──────────────────────────────────────────────────────────
    watchdog_interval_seconds: float = Field(default=30.0)
    heartbeat_max_age_seconds: float = Field(default=120.0)

    # ──────────────────────────────────────────────────────────
    # Validation
    # ──────────────────────────────────────────────────────────
    @model_validator(mode="after")
    def validate_live_requirements(self) -> "Settings":
        """Extra validation when switching to live mode."""
        if self.trading_mode == TradingMode.LIVE:
            if not self.mt5_password:
                raise ValueError("MT5_PASSWORD is required for live trading")
            if not self.mt5_server:
                raise ValueError("MT5_SERVER is required for live trading")
            if self.dashboard_secret_key == "CHANGE_ME":
                raise ValueError("DASHBOARD_SECRET_KEY must be changed before live trading")
        return self

    def as_legacy_config(self) -> dict:
        """
        Return a dict compatible with the existing StandaloneAIBot CONFIG
        format so the legacy engine can be used while the new core is built.
        """
        try:
            import MetaTrader5 as _mt5
            tf_m15 = _mt5.TIMEFRAME_M15
            tf_h1 = _mt5.TIMEFRAME_H1
            tf_h4 = _mt5.TIMEFRAME_H4
        except ImportError:
            tf_m15 = 15
            tf_h1 = 16385
            tf_h4 = 16388

        return {
            "account": self.mt5_login,
            "password": self.mt5_password,
            "server": self.mt5_server,
            "mt5_path": self.mt5_path,
            "symbol": self.symbols[0] if self.symbols else "XAUUSDm",
            "timeframe": tf_m15,
            "h1_tf": tf_h1,
            "h4_tf": tf_h4,
            "n_candles": 600,
            "ai_confidence_threshold": self.ai_confidence_threshold,
            "ai_train_bars": self.ai_train_bars,
            "ai_retrain_bars": self.ai_retrain_bars,
            "min_signal_score": self.min_signal_score,
            "adx_min": self.adx_trending_min,
            "adx_strong": self.adx_strong_trend,
            "use_session_filter": self.use_session_filter,
            "london_start_utc": self.london_start_utc,
            "london_end_utc": self.london_end_utc,
            "ny_start_utc": self.ny_start_utc,
            "ny_end_utc": self.ny_end_utc,
            "risk_per_trade": self.risk_per_trade,
            "sl_atr_mult": self.sl_atr_mult,
            "tp_rr_ratio": self.tp_rr_ratio,
            "trail_atr_mult": self.trail_atr_mult,
            "trail_activate_atr": self.trail_activate_atr,
            "dynamic_risk": self.dynamic_risk,
            "min_risk_scale": self.min_risk_scale,
            "max_risk_scale": self.max_risk_scale,
            "max_margin_usage": self.max_margin_usage,
            "max_open_pos": self.max_open_positions,
            "max_daily_dd": self.max_daily_loss_pct,
            "enable_breakeven": self.enable_breakeven,
            "breakeven_trigger_atr": self.breakeven_trigger_atr,
            "enable_partial_close": self.enable_partial_close,
            "partial_close_trigger_atr": self.partial_close_trigger_atr,
            "partial_close_fraction": self.partial_close_fraction,
            "partial_close2_trigger_atr": self.partial_close2_trigger_atr,
            "partial_close2_fraction": self.partial_close2_fraction,
            "trail_tighten1_atr": self.trail_tighten1_atr,
            "trail_tighten1_mult": self.trail_tighten1_mult,
            "trail_tighten2_atr": self.trail_tighten2_atr,
            "trail_tighten2_mult": self.trail_tighten2_mult,
            "close_on_opposite_signal": self.close_on_opposite_signal,
            "opposite_signal_min_score": self.opposite_signal_min_score,
            "enable_trend_flip_close": self.enable_trend_flip_close,
            "close_old_profitable": self.close_old_profitable,
            "max_position_age_hours": self.max_position_age_hours,
            "max_trades_per_day": self.max_trades_per_day,
            "equity_protect_close_all": self.equity_protect_close_all,
            "server_utc_offset_hours": self.server_utc_offset_hours,
            "magic_number": self.magic_number,
            "deviation": self.deviation_points,
            "dry_run": not self.is_live,
            "sleep_interval": self.poll_interval_seconds,
            "max_slippage_retries": self.max_requote_retries,
            "telegram_token": self.telegram_bot_token,
            "telegram_chat_id": self.telegram_chat_id,
            "max_loss_per_trade_pct": self.max_loss_per_trade_pct,
            "profit_lock_activate_atr": self.profit_lock_activate_atr,
            "profit_giveback_pct": self.profit_giveback_pct,
            "cut_losing_after_hours": self.cut_losing_after_hours,
        }


# ──────────────────────────────────────────────────────────────
# Singleton accessor
# ──────────────────────────────────────────────────────────────
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Return the cached Settings instance (loaded once from environment)."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reload_settings() -> Settings:
    """Force reload of settings (e.g. after .env file change)."""
    global _settings
    _settings = Settings()
    return _settings
