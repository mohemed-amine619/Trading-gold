"""
core/execution/engine.py — Execution Engine.

Handles the order lifecycle:
- Sends orders via MT5Adapter
- Retries on requotes or transient errors
- Handles error reporting
"""

import logging
import time
from typing import Optional

from core.mt5.adapter import MT5Adapter
from core.mt5.types import OrderRequest, OrderResult, RetCode, TradeAction, OrderType
from core.risk.engine import TradeProposal

logger = logging.getLogger(__name__)

class ExecutionEngine:
    def __init__(self, adapter: MT5Adapter, max_retries: int = 3, retry_delay_ms: int = 500):
        self.adapter = adapter
        self.max_retries = max_retries
        self.retry_delay_ms = retry_delay_ms

    def execute_proposal(self, proposal: TradeProposal, lot_size: float, magic: int = 888999) -> Optional[OrderResult]:
        """
        Execute an approved TradeProposal as a market order.
        """
        if lot_size <= 0:
            logger.error(f"Cannot execute {proposal.symbol} with lot_size <= 0")
            return None

        # BUY = 0, SELL = 1
        order_type = OrderType.BUY if proposal.direction == 1 else OrderType.SELL
        
        request = OrderRequest(
            symbol=proposal.symbol,
            action=TradeAction.DEAL,
            type=order_type,
            volume=lot_size,
            price=0.0,  # Market order
            sl=proposal.stop_loss,
            tp=proposal.take_profit,
            magic=magic,
            comment=f"{proposal.strategy_name[:20]}"
        )
        
        return self._send_with_retry(request)

    def close_position(self, ticket: int, symbol: str, volume: float, direction: int) -> Optional[OrderResult]:
        """
        Close an existing position by sending an opposite market order.
        """
        order_type = OrderType.SELL if direction == 1 else OrderType.BUY
        
        request = OrderRequest(
            symbol=symbol,
            action=TradeAction.DEAL,
            type=order_type,
            volume=volume,
            price=0.0,
            magic=888999,
            comment="close_pos"
        )
        
        # Set position ticket for MT5 if needed (supported in some brokers)
        # request.position = ticket (would require adding position field to OrderRequest)
        
        return self._send_with_retry(request)
        
    def modify_position(self, ticket: int, symbol: str, new_sl: float, new_tp: float) -> Optional[OrderResult]:
        """
        Modify Stop Loss and Take Profit of an existing position.
        """
        request = OrderRequest(
            symbol=symbol,
            action=TradeAction.SLTP,
            type=OrderType.BUY, # Ignored for SLTP
            volume=0.0,         # Ignored for SLTP
            price=0.0,
            sl=new_sl,
            tp=new_tp,
            magic=888999,
            comment="modify_sltp",
            position=ticket
        )
        
        return self._send_with_retry(request)

    def _send_with_retry(self, request: OrderRequest) -> OrderResult:
        """
        Internal method to send order with retry logic for transient errors.
        """
        retries = 0
        last_result = None
        
        while retries <= self.max_retries:
            result = self.adapter.send_order(request)
            last_result = result
            
            if result.success:
                logger.info(f"Order executed successfully: deal={result.deal} retcode={result.retcode}")
                return result
                
            if result.is_requote:
                retries += 1
                logger.warning(f"Order requote/price changed (retcode={result.retcode}). Retry {retries}/{self.max_retries}")
                time.sleep(self.retry_delay_ms / 1000.0)
                continue
                
            # If we reach here, it's a hard error
            logger.error(f"Order failed (retcode={result.retcode}): {result.comment}")
            break
            
        return last_result
