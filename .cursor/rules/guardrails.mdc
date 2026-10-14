---
description: Mandatory architecture, risk, and execution rules for the multi-strategy trading bot
alwaysApply: true
---

# Trading Bot Core Guardrails

1. **Strict Separation of Concerns:**
   - Strategies ONLY emit standardized Signal objects (direction, confidence, suggested stop-loss, take-profit, metadata).
   - Strategies NEVER call exchange APIs or manage wallet balances directly.
   - Portfolio & Risk Manager validates signals (position sizing, max drawdown, correlation/exposure limits) before passing to Execution.

2. **Financial Math & Precision:**
   - Never use standard floating-point math for order quantities, prices, or PnL. Use exact Decimal types or exchange tick/lot-size rounding helpers.
   - Always validate minimum notional value, step size, and tick size before submitting orders.

3. **Order Idempotency & Safety:**
   - Every order request must carry a deterministic `clientOrderId` to prevent duplicate fills on network retries.
   - Never hardcode API keys or secrets; load strictly from environment variables and verify via `.agents/workflows/code-review.md`.
   - Every live execution path must support a dry-run / paper-trading toggle.

4. **Workflow Discipline:**
   - Do not modify existing working strategies without creating or updating unit tests first (`/tdd`).
