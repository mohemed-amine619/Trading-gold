# ARCHITECTURE.md — AI Trading Platform

## 1. Current State Assessment

### What Exists

| File | Lines | Role |
|---|---|---|
| `run.py` | 1,535 | Monolithic bot: MT5 connection, indicator engine, AI ensemble, risk, execution, position manager |
| `gui.py` | 608 | PyQt5 → actually Tkinter. Config editor, start/stop controls, live log viewer |
| `config.py` | 157 | Shared parameter defaults (partially duplicated by CONFIG dict in run.py) |
| `position_dashboard.py` | 278 | Standalone position monitoring panel |
| `requirements.txt` | 9 | MetaTrader5, pandas, pandas_ta, numpy, PyQt5, matplotlib, xgboost, requests, scikit-learn, scipy |

### What Already Works

- ✅ MT5 connection + login + reconnect with backoff
- ✅ Multi-timeframe data fetch (M15 / H1 / H4) with forming-bar exclusion
- ✅ Full indicator suite (EMA, RSI, MACD, Stoch, ADX, ATR, BB, CCI, Williams%R, OBV, Keltner)
- ✅ AI ensemble (RandomForest + GradientBoosting + ExtraTrees + HistGradientBoosting, soft-vote)
- ✅ Recency-weighted training, background thread retraining
- ✅ Multi-layer confluence scoring (AI + Trend + Momentum + Volume + Structure → 0–100)
- ✅ Session filter (London / New York UTC hours)
- ✅ ADX regime gate
- ✅ Dynamic risk scaling by ADX strength
- ✅ Fixed-fractional position sizing (equity × risk% / SL distance)
- ✅ ATR-based SL/TP with broker min-distance enforcement
- ✅ Margin guard (50% equity cap)
- ✅ Spread guard (optional)
- ✅ Daily drawdown limit with auto-close-all
- ✅ Per-trade hard stop-loss (% equity backstop)
- ✅ Trailing stop with 2-stage tightening
- ✅ Breakeven move
- ✅ 2-stage partial close
- ✅ Profit-lock giveback close
- ✅ Losing-position timeout cut
- ✅ H1 trend-flip close
- ✅ Opposite-signal close
- ✅ Age-timeout close (profitable positions)
- ✅ Telegram notifications (resilient, cooldown, background thread)
- ✅ Paper trading (dry_run flag)
- ✅ Magic number isolation (only touches its own positions)
- ✅ Tkinter GUI with config editor + log viewer
- ✅ Per-session performance tracking (win rate, P&L, best/worst trade)

### Architectural Weaknesses

| # | Weakness | Severity |
|---|---|---|
| W1 | **Monolith** — all logic in one 1,535-line file; impossible to test independently | Critical |
| W2 | **Credentials in source** — password and Telegram token hardcoded in CONFIG dict | Critical |
| W3 | **No database** — no persistent trade history, decisions, or audit trail | Critical |
| W4 | **Single symbol only** — `StandaloneAIBot` handles one symbol; no multi-symbol orchestration | High |
| W5 | **No LLM** — ML only (scikit-learn ensemble), no LLM reasoning layer | High |
| W6 | **No backtesting engine** — strategies cannot be evaluated before live deployment | High |
| W7 | **No web dashboard** — Tkinter GUI is desktop-only, not remotely accessible | High |
| W8 | **No weekly/monthly loss limits** — only daily drawdown guard | Medium |
| W9 | **No consecutive-loss cooldown** — no loss-streak protection | Medium |
| W10 | **No news/event layer** — no awareness of high-impact economic events | Medium |
| W11 | **No trade thesis** — signals are numeric scores, not machine-readable reasoning | Medium |
| W12 | **No structured event log** — only plaintext log file | Medium |
| W13 | **No test coverage** — no unit, integration, or end-to-end tests | Medium |
| W14 | **Config duplication** — `config.py` vs `CONFIG` dict in `run.py` | Low |
| W15 | **Tkinter not cross-platform** — won't render on headless Linux servers | Low |

---

## 2. Proposed Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Web Dashboard (FastAPI + React/Jinja2)       │
│  Overview · Positions · History · AI Decisions · Risk Center     │
│  Strategy Performance · Market Scanner · Settings · Emergency    │
└──────────────────────────────┬──────────────────────────────────┘
                               │ REST + WebSocket
┌──────────────────────────────▼──────────────────────────────────┐
│                      Trading API (FastAPI)                        │
│  Auth · CORS · Rate-limit · Health · WebSocket broadcast         │
└──┬─────────────┬─────────────┬──────────────┬───────────────────┘
   │             │             │              │
   ▼             ▼             ▼              ▼
Market        AI Decision   Risk Engine   Execution Engine
Engine        Engine        (hard limits) (MT5 adapter)
   │             │             │              │
   └─────────────┴─────────────┴──────────────┘
                               │
                    ┌──────────▼──────────┐
                    │   Strategy Engine    │
                    │ Trend | Momentum     │
                    │ Breakout | MeanRev   │
                    │ Volatility | Scanner │
                    │ Multi-Timeframe      │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  Position Monitor    │
                    │  + Auto Management   │
                    │  (thesis validation) │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │     Database         │
                    │  SQLite → PostgreSQL  │
                    │  (all events, trades) │
                    └─────────────────────┘
```

---

## 3. Technology Stack

### Backend

| Layer | Technology | Rationale |
|---|---|---|
| Language | Python 3.11+ | Existing codebase; MT5 Python API |
| API | FastAPI | Async, type-safe, built-in OpenAPI |
| WebSocket | FastAPI WebSocket | Real-time dashboard updates |
| Database ORM | SQLAlchemy 2.x + Alembic | Migrations, type-safe queries |
| Database | SQLite (dev) → PostgreSQL (prod) | Zero-config start, production-ready path |
| Task scheduler | APScheduler | Market data polling, retraining cycles |
| Validation | Pydantic v2 | Config, API schemas, LLM output validation |
| LLM client | OpenAI / Anthropic / Ollama | Pluggable provider |
| Config / secrets | python-dotenv + `.env` | No secrets in source |
| Logging | structlog | Structured JSON logs |
| Testing | pytest + pytest-asyncio | Unit + integration |

### Frontend

| Layer | Technology | Rationale |
|---|---|---|
| UI | Jinja2 templates + HTMX | Server-side, no build step required |
| Charts | Lightweight Charts (TradingView) | Professional candle charts |
| Styling | TailwindCSS (CDN) | No build step required |

### MT5 (Windows only)

The MT5 Python API is Windows-only. The architecture wraps it behind a clean interface so the rest of the system remains testable on Linux/Mac using a mock adapter.

---

## 4. Module Structure

```
Trading-gold/
├── .env                        # secrets (never commit)
├── .env.example                # template
├── config.py                   # legacy (kept, will be superseded)
├── run.py                      # legacy bot (kept, will be refactored)
├── gui.py                      # legacy GUI (kept)
│
├── core/                       # NEW: production engine
│   ├── __init__.py
│   ├── settings.py             # Pydantic Settings (reads .env)
│   ├── database/
│   │   ├── __init__.py
│   │   ├── engine.py           # SQLAlchemy engine + session factory
│   │   ├── models.py           # ORM models
│   │   └── repository.py       # Data-access layer
│   ├── mt5/
│   │   ├── __init__.py
│   │   ├── adapter.py          # MT5Adapter (real + mock)
│   │   ├── connection.py       # Connection manager + reconnect
│   │   └── types.py            # MT5 data types (Pydantic)
│   ├── market/
│   │   ├── __init__.py
│   │   ├── data_engine.py      # Candle fetching, indicator pipeline
│   │   ├── indicators.py       # All technical indicators
│   │   ├── regime.py           # Market regime detector
│   │   └── scanner.py          # Multi-symbol scanner
│   ├── strategy/
│   │   ├── __init__.py
│   │   ├── base.py             # BaseStrategy interface
│   │   ├── ensemble.py         # Strategy ensemble + scoring
│   │   ├── trend.py            # TrendFollowingStrategy
│   │   ├── momentum.py         # MomentumStrategy
│   │   ├── breakout.py         # BreakoutStrategy
│   │   ├── mean_reversion.py   # MeanReversionStrategy
│   │   ├── market_structure.py # MarketStructureStrategy
│   │   └── volatility.py       # VolatilityStrategy
│   ├── ai/
│   │   ├── __init__.py
│   │   ├── ml_ensemble.py      # Existing scikit-learn ensemble (refactored)
│   │   ├── llm_client.py       # LLM provider abstraction
│   │   ├── llm_analyst.py      # Market analyst role
│   │   ├── llm_critic.py       # Risk critic role
│   │   ├── decision_engine.py  # Multi-model reconciliation
│   │   └── schemas.py          # LLM I/O Pydantic schemas
│   ├── risk/
│   │   ├── __init__.py
│   │   ├── engine.py           # RiskEngine (all hard limits)
│   │   ├── position_sizer.py   # Lot size calculator
│   │   └── limits.py           # Configurable limit registry
│   ├── execution/
│   │   ├── __init__.py
│   │   ├── engine.py           # ExecutionEngine
│   │   ├── order.py            # Order lifecycle
│   │   └── paper.py            # Paper trading simulator
│   ├── position/
│   │   ├── __init__.py
│   │   ├── monitor.py          # Continuous position monitor
│   │   ├── manager.py          # Position management actions
│   │   └── thesis.py           # Trade thesis validation
│   ├── backtesting/
│   │   ├── __init__.py
│   │   ├── engine.py           # Backtest runner
│   │   ├── simulator.py        # Order/fill simulator
│   │   └── metrics.py          # Performance metrics
│   ├── notifications/
│   │   ├── __init__.py
│   │   ├── notifier.py         # Notification dispatcher
│   │   ├── telegram.py         # Telegram adapter
│   │   └── webhook.py          # Generic webhook adapter
│   └── events/
│       ├── __init__.py
│       └── log.py              # Immutable event logger
│
├── api/                        # FastAPI application
│   ├── __init__.py
│   ├── main.py                 # FastAPI app factory
│   ├── auth.py                 # JWT authentication
│   ├── routers/
│   │   ├── account.py
│   │   ├── positions.py
│   │   ├── trades.py
│   │   ├── signals.py
│   │   ├── strategies.py
│   │   ├── risk.py
│   │   ├── scanner.py
│   │   ├── settings.py
│   │   └── control.py          # START/STOP/EMERGENCY
│   └── websocket.py            # WebSocket broadcaster
│
├── dashboard/                  # Web UI
│   ├── templates/
│   │   ├── base.html
│   │   ├── overview.html
│   │   ├── positions.html
│   │   ├── history.html
│   │   ├── ai_decisions.html
│   │   ├── strategy.html
│   │   ├── risk.html
│   │   ├── scanner.html
│   │   ├── settings.html
│   │   └── emergency.html
│   └── static/
│       ├── css/
│       └── js/
│
├── docs/                       # Documentation
│   ├── ARCHITECTURE.md         # ← this file
│   ├── TRADING_ENGINE.md
│   ├── RISK_ENGINE.md
│   ├── STRATEGIES.md
│   ├── AI_ENGINE.md
│   ├── BACKTESTING.md
│   ├── OPERATIONS.md
│   ├── SECURITY.md
│   └── ROADMAP.md
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── e2e/
│
├── alembic/                    # DB migrations
├── .env.example
├── requirements.txt            # (will be expanded)
└── pyproject.toml
```

---

## 5. Data Flow

```
Market tick / new bar
        │
        ▼
MT5Adapter.get_candles()
        │
        ▼
DataEngine.prepare()
  └─ Indicators
  └─ RegimeDetector
        │
        ▼
StrategyEnsemble.evaluate()
  ├─ TrendStrategy
  ├─ MomentumStrategy
  ├─ BreakoutStrategy
  ├─ MeanReversionStrategy
  ├─ MarketStructureStrategy
  └─ VolatilityStrategy
        │
        ▼
MLEnsemble.predict()     ←── retrain loop (background)
        │
        ▼
LLMDecisionEngine.decide()
  ├─ LLMAnalyst (market context)
  └─ LLMCritic  (risk review)
        │
        ▼
RiskEngine.approve()?
  ├─ Position size check
  ├─ Daily loss check
  ├─ Drawdown check
  ├─ Max positions check
  ├─ Spread check
  ├─ Margin check
  └─ Loss streak check
        │
     APPROVED
        │
        ▼
ExecutionEngine.send_order()
  └─ MT5Adapter.order_send()
        │
        ▼
Position created
        │
        ▼
PositionMonitor (every tick)
  ├─ ThesisValidator
  ├─ TrailingStop
  ├─ Breakeven
  ├─ PartialClose
  └─ ForceClose rules
        │
        ▼
EventLog + Database + Notifications
```

---

## 6. Reuse Strategy

| Existing Component | Reuse Decision |
|---|---|
| Indicator logic (`_prepare`, `build_features`) | ✅ Port to `core/market/indicators.py` |
| Scoring system (`_score_direction`, sub-scores) | ✅ Port to `core/strategy/` modules |
| AI ensemble (`AIEnsemble`, `_make_pipeline`) | ✅ Port to `core/ai/ml_ensemble.py` |
| Position sizing (`_size_lot`) | ✅ Port to `core/risk/position_sizer.py` |
| Position management (`_manage_positions`) | ✅ Port to `core/position/manager.py` |
| Notifier (Telegram) | ✅ Port to `core/notifications/telegram.py` |
| CONFIG dict | ⚠️ Migrate to `.env` + Pydantic Settings |
| MT5 connection / reconnect | ✅ Port to `core/mt5/connection.py` |
| `gui.py` / `position_dashboard.py` | ⚠️ Keep for now; superseded by web dashboard |
