"""
core/database/models.py — SQLAlchemy ORM models.

Every trade decision, signal, order, position, risk event, and manual action
is persisted here.  Nothing is ever lost.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _uuid() -> str:
    return str(uuid.uuid4())


# ─────────────────────────────────────────────────────────────────────────────
# Base
# ─────────────────────────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    pass


# ─────────────────────────────────────────────────────────────────────────────
# Enums (mirrored as Python enums for type safety)
# ─────────────────────────────────────────────────────────────────────────────

class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(str, Enum):
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    ERROR = "ERROR"


class PositionStatus(str, Enum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    PARTIALLY_CLOSED = "PARTIALLY_CLOSED"


class SignalAction(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    CLOSE = "CLOSE"
    REDUCE = "REDUCE"


class RiskDecision(str, Enum):
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"


class TradingMode(str, Enum):
    PAPER = "paper"
    LIVE = "live"
    BACKTEST = "backtest"


class EventType(str, Enum):
    SIGNAL_GENERATED = "SIGNAL_GENERATED"
    AI_DECISION = "AI_DECISION"
    RISK_APPROVED = "RISK_APPROVED"
    RISK_REJECTED = "RISK_REJECTED"
    ORDER_SUBMITTED = "ORDER_SUBMITTED"
    ORDER_FILLED = "ORDER_FILLED"
    ORDER_REJECTED = "ORDER_REJECTED"
    POSITION_OPENED = "POSITION_OPENED"
    POSITION_CLOSED = "POSITION_CLOSED"
    POSITION_MODIFIED = "POSITION_MODIFIED"
    PARTIAL_CLOSE = "PARTIAL_CLOSE"
    STOP_MOVED = "STOP_MOVED"
    BREAKEVEN_SET = "BREAKEVEN_SET"
    TRAILING_UPDATED = "TRAILING_UPDATED"
    THESIS_INVALIDATED = "THESIS_INVALIDATED"
    DAILY_LIMIT_HIT = "DAILY_LIMIT_HIT"
    DRAWDOWN_WARNING = "DRAWDOWN_WARNING"
    LOSS_STREAK = "LOSS_STREAK"
    MT5_DISCONNECTED = "MT5_DISCONNECTED"
    MT5_RECONNECTED = "MT5_RECONNECTED"
    AI_UNAVAILABLE = "AI_UNAVAILABLE"
    EMERGENCY_STOP = "EMERGENCY_STOP"
    MANUAL_ACTION = "MANUAL_ACTION"
    SYSTEM_START = "SYSTEM_START"
    SYSTEM_STOP = "SYSTEM_STOP"
    HEARTBEAT = "HEARTBEAT"


# ─────────────────────────────────────────────────────────────────────────────
# Accounts
# ─────────────────────────────────────────────────────────────────────────────

class Account(Base):
    __tablename__ = "accounts"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    mt5_login: Mapped[int] = mapped_column(Integer, unique=True, index=True)
    server: Mapped[str] = mapped_column(String(128))
    name: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    currency: Mapped[str] = mapped_column(String(8), default="USD")
    leverage: Mapped[int] = mapped_column(Integer, default=1)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )

    # snapshots, orders, positions attached
    balance_snapshots: Mapped[list["BalanceSnapshot"]] = relationship(
        back_populates="account", lazy="dynamic"
    )
    positions: Mapped[list["Position"]] = relationship(
        back_populates="account", lazy="dynamic"
    )


class BalanceSnapshot(Base):
    """Periodic balance/equity snapshots for equity curve reconstruction."""
    __tablename__ = "balance_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    account_id: Mapped[str] = mapped_column(ForeignKey("accounts.id"), index=True)
    balance: Mapped[float] = mapped_column(Float)
    equity: Mapped[float] = mapped_column(Float)
    margin: Mapped[float] = mapped_column(Float, default=0.0)
    free_margin: Mapped[float] = mapped_column(Float, default=0.0)
    margin_level: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    daily_pnl: Mapped[float] = mapped_column(Float, default=0.0)
    drawdown_pct: Mapped[float] = mapped_column(Float, default=0.0)
    recorded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, index=True
    )

    account: Mapped["Account"] = relationship(back_populates="balance_snapshots")

    __table_args__ = (
        Index("ix_balance_snapshots_account_time", "account_id", "recorded_at"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Symbols
# ─────────────────────────────────────────────────────────────────────────────

class Symbol(Base):
    __tablename__ = "symbols"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(32), unique=True, index=True)
    description: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    digits: Mapped[int] = mapped_column(Integer, default=5)
    point: Mapped[float] = mapped_column(Float, default=0.00001)
    contract_size: Mapped[float] = mapped_column(Float, default=100000.0)
    tick_size: Mapped[float] = mapped_column(Float, default=0.00001)
    tick_value: Mapped[float] = mapped_column(Float, default=1.0)
    volume_min: Mapped[float] = mapped_column(Float, default=0.01)
    volume_max: Mapped[float] = mapped_column(Float, default=100.0)
    volume_step: Mapped[float] = mapped_column(Float, default=0.01)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )


# ─────────────────────────────────────────────────────────────────────────────
# Strategies
# ─────────────────────────────────────────────────────────────────────────────

class Strategy(Base):
    __tablename__ = "strategies"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    description: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    weight: Mapped[float] = mapped_column(Float, default=1.0)
    parameters: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    signals: Mapped[list["StrategySignal"]] = relationship(
        back_populates="strategy", lazy="dynamic"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Signals
# ─────────────────────────────────────────────────────────────────────────────

class StrategySignal(Base):
    """Raw signal from a single strategy for a single symbol + timeframe."""
    __tablename__ = "strategy_signals"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    strategy_id: Mapped[int] = mapped_column(ForeignKey("strategies.id"), index=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    timeframe: Mapped[str] = mapped_column(String(8))
    bar_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    direction: Mapped[int] = mapped_column(Integer)   # +1 long | -1 short | 0 none
    score: Mapped[float] = mapped_column(Float)       # 0-100
    reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    indicators: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON snapshot
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    strategy: Mapped["Strategy"] = relationship(back_populates="signals")

    __table_args__ = (
        Index("ix_strategy_signals_symbol_bar", "symbol", "bar_time"),
    )


class EnsembleSignal(Base):
    """Aggregated confluence score from all active strategies."""
    __tablename__ = "ensemble_signals"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    timeframe: Mapped[str] = mapped_column(String(8))
    bar_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    direction: Mapped[int] = mapped_column(Integer)
    final_score: Mapped[float] = mapped_column(Float)
    trend_score: Mapped[float] = mapped_column(Float, default=0.0)
    momentum_score: Mapped[float] = mapped_column(Float, default=0.0)
    structure_score: Mapped[float] = mapped_column(Float, default=0.0)
    volatility_score: Mapped[float] = mapped_column(Float, default=0.0)
    liquidity_score: Mapped[float] = mapped_column(Float, default=0.0)
    ai_score: Mapped[float] = mapped_column(Float, default=0.0)
    ml_confidence: Mapped[float] = mapped_column(Float, default=0.5)
    regime: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    ai_decisions: Mapped[list["AIDecision"]] = relationship(
        back_populates="ensemble_signal", lazy="dynamic"
    )


# ─────────────────────────────────────────────────────────────────────────────
# AI Decisions
# ─────────────────────────────────────────────────────────────────────────────

class AIDecision(Base):
    """LLM + ML combined decision record."""
    __tablename__ = "ai_decisions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ensemble_signal_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("ensemble_signals.id"), nullable=True, index=True
    )
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    action: Mapped[str] = mapped_column(String(16))   # BUY | SELL | HOLD | CLOSE | REDUCE
    confidence: Mapped[float] = mapped_column(Float)
    entry_reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)      # JSON list
    invalidation_conditions: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON list
    risk_assessment: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    recommended_sl: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    recommended_tp: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    recommended_size: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    expected_rr: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    llm_provider: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    llm_model: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    llm_raw_response: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    ml_direction: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    ml_confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, index=True
    )

    ensemble_signal: Mapped[Optional["EnsembleSignal"]] = relationship(
        back_populates="ai_decisions"
    )
    risk_review: Mapped[Optional["RiskReview"]] = relationship(
        back_populates="ai_decision", uselist=False
    )


# ─────────────────────────────────────────────────────────────────────────────
# Risk Reviews
# ─────────────────────────────────────────────────────────────────────────────

class RiskReview(Base):
    """Risk engine decision for every AI signal."""
    __tablename__ = "risk_reviews"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ai_decision_id: Mapped[int] = mapped_column(
        ForeignKey("ai_decisions.id"), unique=True, index=True
    )
    decision: Mapped[str] = mapped_column(String(16))  # APPROVED | REJECTED
    reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    checks_passed: Mapped[Optional[str]] = mapped_column(Text, nullable=True)   # JSON
    checks_failed: Mapped[Optional[str]] = mapped_column(Text, nullable=True)   # JSON
    calculated_lot_size: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    calculated_sl: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    calculated_tp: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    current_daily_pnl: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    current_drawdown_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    current_open_positions: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    ai_decision: Mapped["AIDecision"] = relationship(back_populates="risk_review")
    order: Mapped[Optional["Order"]] = relationship(
        back_populates="risk_review", uselist=False
    )


# ─────────────────────────────────────────────────────────────────────────────
# Orders
# ─────────────────────────────────────────────────────────────────────────────

class Order(Base):
    """Every order sent to MT5 (or simulated in paper mode)."""
    __tablename__ = "orders"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    risk_review_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("risk_reviews.id"), nullable=True, index=True
    )
    account_id: Mapped[Optional[str]] = mapped_column(
        ForeignKey("accounts.id"), nullable=True, index=True
    )
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    side: Mapped[str] = mapped_column(String(8))         # BUY | SELL
    volume: Mapped[float] = mapped_column(Float)
    order_type: Mapped[str] = mapped_column(String(32))  # MARKET | LIMIT | STOP
    price_requested: Mapped[float] = mapped_column(Float)
    price_filled: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    sl: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    tp: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(String(32), default=OrderStatus.PENDING.value)
    mt5_ticket: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, index=True)
    mt5_retcode: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    mt5_comment: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    magic_number: Mapped[int] = mapped_column(Integer, default=888999)
    trading_mode: Mapped[str] = mapped_column(String(16), default=TradingMode.PAPER.value)
    strategy_name: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    submitted_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, index=True
    )
    filled_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    risk_review: Mapped[Optional["RiskReview"]] = relationship(back_populates="order")
    position: Mapped[Optional["Position"]] = relationship(
        back_populates="open_order", uselist=False, foreign_keys="Position.open_order_id"
    )

    __table_args__ = (
        Index("ix_orders_symbol_time", "symbol", "submitted_at"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Positions
# ─────────────────────────────────────────────────────────────────────────────

class Position(Base):
    """A single open (or closed) position."""
    __tablename__ = "positions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    open_order_id: Mapped[Optional[str]] = mapped_column(
        ForeignKey("orders.id"), nullable=True, index=True
    )
    account_id: Mapped[Optional[str]] = mapped_column(
        ForeignKey("accounts.id"), nullable=True, index=True
    )
    mt5_ticket: Mapped[int] = mapped_column(Integer, unique=True, index=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    side: Mapped[str] = mapped_column(String(8))
    volume: Mapped[float] = mapped_column(Float)
    volume_remaining: Mapped[float] = mapped_column(Float)
    entry_price: Mapped[float] = mapped_column(Float)
    current_price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    sl: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    tp: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(String(32), default=PositionStatus.OPEN.value)
    strategy_name: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    trading_mode: Mapped[str] = mapped_column(String(16), default=TradingMode.PAPER.value)
    magic_number: Mapped[int] = mapped_column(Integer, default=888999)

    # Thesis tracking
    thesis: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON
    thesis_valid: Mapped[bool] = mapped_column(Boolean, default=True)
    invalidation_reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Performance tracking
    unrealized_pnl: Mapped[float] = mapped_column(Float, default=0.0)
    realized_pnl: Mapped[float] = mapped_column(Float, default=0.0)
    peak_profit: Mapped[float] = mapped_column(Float, default=0.0)
    max_adverse_excursion: Mapped[float] = mapped_column(Float, default=0.0)
    swap: Mapped[float] = mapped_column(Float, default=0.0)

    # Timestamps
    opened_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, index=True
    )
    closed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )

    # Flags
    breakeven_set: Mapped[bool] = mapped_column(Boolean, default=False)
    partial_close_stage: Mapped[int] = mapped_column(Integer, default=0)

    account: Mapped[Optional["Account"]] = relationship(back_populates="positions")
    open_order: Mapped[Optional["Order"]] = relationship(
        back_populates="position", foreign_keys=[open_order_id]
    )
    close_orders: Mapped[list["CloseOrder"]] = relationship(
        back_populates="position", lazy="dynamic"
    )

    __table_args__ = (
        Index("ix_positions_symbol_status", "symbol", "status"),
    )


class CloseOrder(Base):
    """A close or partial-close order linked to a position."""
    __tablename__ = "close_orders"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    position_id: Mapped[str] = mapped_column(ForeignKey("positions.id"), index=True)
    volume: Mapped[float] = mapped_column(Float)
    price: Mapped[float] = mapped_column(Float)
    reason: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    pnl: Mapped[float] = mapped_column(Float, default=0.0)
    mt5_ticket: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    mt5_retcode: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    executed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, index=True
    )

    position: Mapped["Position"] = relationship(back_populates="close_orders")


# ─────────────────────────────────────────────────────────────────────────────
# Performance Metrics (daily aggregates)
# ─────────────────────────────────────────────────────────────────────────────

class DailyMetrics(Base):
    __tablename__ = "daily_metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    account_id: Mapped[str] = mapped_column(ForeignKey("accounts.id"), index=True)
    date: Mapped[str] = mapped_column(String(10), index=True)  # YYYY-MM-DD
    trades_taken: Mapped[int] = mapped_column(Integer, default=0)
    trades_won: Mapped[int] = mapped_column(Integer, default=0)
    trades_lost: Mapped[int] = mapped_column(Integer, default=0)
    gross_pnl: Mapped[float] = mapped_column(Float, default=0.0)
    net_pnl: Mapped[float] = mapped_column(Float, default=0.0)
    commission: Mapped[float] = mapped_column(Float, default=0.0)
    swap: Mapped[float] = mapped_column(Float, default=0.0)
    max_drawdown_pct: Mapped[float] = mapped_column(Float, default=0.0)
    win_rate: Mapped[float] = mapped_column(Float, default=0.0)
    profit_factor: Mapped[float] = mapped_column(Float, default=0.0)
    avg_win: Mapped[float] = mapped_column(Float, default=0.0)
    avg_loss: Mapped[float] = mapped_column(Float, default=0.0)
    sharpe_ratio: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    start_balance: Mapped[float] = mapped_column(Float, default=0.0)
    end_balance: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    __table_args__ = (
        UniqueConstraint("account_id", "date", name="uq_daily_metrics_account_date"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Risk Events
# ─────────────────────────────────────────────────────────────────────────────

class RiskEvent(Base):
    """A risk engine alert or limit breach."""
    __tablename__ = "risk_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_type: Mapped[str] = mapped_column(String(64), index=True)
    severity: Mapped[str] = mapped_column(String(16))  # INFO | WARNING | CRITICAL
    symbol: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    description: Mapped[str] = mapped_column(Text)
    context: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON
    resolved: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, index=True
    )


# ─────────────────────────────────────────────────────────────────────────────
# System Events (immutable audit log)
# ─────────────────────────────────────────────────────────────────────────────

class SystemEvent(Base):
    """
    Immutable audit trail.  One row per meaningful system action.
    Every entry in the trade lifecycle produces a row here.
    """
    __tablename__ = "system_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_type: Mapped[str] = mapped_column(String(64), index=True)
    symbol: Mapped[Optional[str]] = mapped_column(String(32), nullable=True, index=True)
    position_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True, index=True)
    order_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True, index=True)
    mt5_ticket: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    message: Mapped[str] = mapped_column(Text)
    details: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, index=True
    )

    __table_args__ = (
        Index("ix_system_events_type_time", "event_type", "created_at"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Manual Actions
# ─────────────────────────────────────────────────────────────────────────────

class ManualAction(Base):
    """Every manual override by the user is logged here."""
    __tablename__ = "manual_actions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    action: Mapped[str] = mapped_column(String(64))
    performed_by: Mapped[str] = mapped_column(String(64), default="admin")
    symbol: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    position_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    parameters: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON
    result: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, index=True
    )


# ─────────────────────────────────────────────────────────────────────────────
# Model Predictions (ML prediction store)
# ─────────────────────────────────────────────────────────────────────────────

class ModelPrediction(Base):
    __tablename__ = "model_predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_name: Mapped[str] = mapped_column(String(64), index=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    timeframe: Mapped[str] = mapped_column(String(8))
    bar_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    direction: Mapped[int] = mapped_column(Integer)    # +1 | -1 | 0
    confidence: Mapped[float] = mapped_column(Float)
    prob_up: Mapped[float] = mapped_column(Float)
    prob_down: Mapped[float] = mapped_column(Float)
    features_snapshot: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON
    train_samples: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    train_up_rate: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    __table_args__ = (
        Index("ix_model_predictions_symbol_bar", "symbol", "bar_time"),
    )

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

class SystemConfig(Base):
    """Dynamic configuration stored in the database."""
    __tablename__ = "system_config"

    key: Mapped[str] = mapped_column(String(128), primary_key=True)
    value: Mapped[str] = mapped_column(Text)  # JSON serialized
    version: Mapped[int] = mapped_column(Integer, default=1)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)

class ConfigAudit(Base):
    """Audit trail for configuration changes."""
    __tablename__ = "config_audit"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    key: Mapped[str] = mapped_column(String(128), index=True)
    old_value: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    new_value: Mapped[str] = mapped_column(Text)
    changed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    changed_by: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)


# ─────────────────────────────────────────────────────────────────────────────
# Users, licences and dashboard access
#
# Who owns a licence, which machines may run the bot on it, who may sign in to
# the dashboard, and every settings change made on the customer's behalf.
#
# Broker credentials are deliberately NOT stored here. The bot keeps its MT5
# login in the local .env on the customer's own machine. Only hashes are stored
# for secrets that this server must check: passwords, licence keys, session
# tokens and machine fingerprints.
# ─────────────────────────────────────────────────────────────────────────────

class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    email: Mapped[str] = mapped_column(String(254), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    role: Mapped[str] = mapped_column(String(16), default="customer")  # owner | admin | customer
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    last_login_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    licences: Mapped[list["Licence"]] = relationship(back_populates="user", lazy="dynamic")


class Licence(Base):
    __tablename__ = "licences"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    key_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)  # sha256 of the key
    plan: Mapped[str] = mapped_column(String(32), default="standard")
    max_installations: Mapped[int] = mapped_column(Integer, default=1)
    status: Mapped[str] = mapped_column(String(16), default="active")  # active | revoked
    expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    user: Mapped["User"] = relationship(back_populates="licences")
    installations: Mapped[list["Installation"]] = relationship(back_populates="licence", lazy="dynamic")


class Installation(Base):
    """One bot installation (one machine) activated against a licence."""
    __tablename__ = "installations"
    __table_args__ = (UniqueConstraint("licence_id", "fingerprint_hash"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    licence_id: Mapped[str] = mapped_column(ForeignKey("licences.id"), index=True)
    fingerprint_hash: Mapped[str] = mapped_column(String(64))  # sha256 of the machine id
    label: Mapped[str] = mapped_column(String(128), default="")
    revoked: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    last_seen_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    licence: Mapped["Licence"] = relationship(back_populates="installations")


class DashboardSession(Base):
    __tablename__ = "dashboard_sessions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    token_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)  # sha256 of the token
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    revoked: Mapped[bool] = mapped_column(Boolean, default=False)


class SettingsRevision(Base):
    """Append-only history of a customer's bot settings. Never updated in place."""
    __tablename__ = "settings_revisions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    installation_id: Mapped[Optional[str]] = mapped_column(
        ForeignKey("installations.id"), nullable=True)
    settings_json: Mapped[str] = mapped_column(Text)  # validated against core/config/schema.py
    changed_by: Mapped[Optional[str]] = mapped_column(ForeignKey("users.id"), nullable=True)
    note: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
