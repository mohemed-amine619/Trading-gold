"""
core/trading — Wiring between analysis and execution.

Exists to hold the orchestrator. Kept thin on purpose: this layer coordinates
and records, it does not compute. Market knowledge belongs in ``core/market``,
sizing and hard limits in ``core/risk``, and order handling in
``core/execution``.
"""

from core.trading.orchestrator import TradingDecision, TradingOrchestrator

__all__ = ["TradingDecision", "TradingOrchestrator"]