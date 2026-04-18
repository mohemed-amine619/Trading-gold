# -*- coding: utf-8 -*-
import sys
import time
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QLineEdit,
    QTableWidget,
    QTableWidgetItem,
    QGroupBox,
    QFormLayout,
    QCheckBox,
)
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QColor
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from sklearn.ensemble import RandomForestClassifier
import requests
from datetime import datetime, date


# ═══════════════════════════════════════════════════
#  AI MODEL
# ═══════════════════════════════════════════════════
class AIModel:
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=10
        )
        self.trained = False

    def preprocess(self, df):
        df = df.copy()
        if "close" not in df.columns or len(df) < 200:
            return pd.DataFrame()

        # RSI
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = gain / loss
        df["RSI"] = 100 - (100 / (1 + rs))

        # ATR
        df["H-L"] = df["high"] - df["low"]
        df["H-C"] = abs(df["high"] - df["close"].shift())
        df["L-C"] = abs(df["low"] - df["close"].shift())
        df["TR"] = df[["H-L", "H-C", "L-C"]].max(axis=1)
        df["ATR"] = df["TR"].rolling(14).mean()

        # Trend filters
        df["SMA20"] = df["close"].rolling(20).mean()
        df["SMA50"] = df["close"].rolling(50).mean()
        df["SMA200"] = df["close"].rolling(200).mean()
        df["Trend"] = (df["SMA50"] > df["SMA200"]).astype(int)

        # Momentum & volatility
        df["Momentum"] = df["close"] - df["close"].shift(10)
        df["Vol_ratio"] = df["ATR"] / df["close"].rolling(50).std().replace(0, np.nan)

        # MACD
        ema12 = df["close"].ewm(span=12).mean()
        ema26 = df["close"].ewm(span=26).mean()
        df["MACD"] = ema12 - ema26
        df["Signal_line"] = df["MACD"].ewm(span=9).mean()
        df["MACD_hist"] = df["MACD"] - df["Signal_line"]

        # Bollinger band position
        sma20 = df["close"].rolling(20).mean()
        std20 = df["close"].rolling(20).std()
        df["BB_pos"] = (df["close"] - sma20) / (2 * std20)

        # Target: price higher 3 bars later AND by at least 0.05%
        future = df["close"].shift(-3)
        df["Target"] = (
            (future > df["close"]) & ((future - df["close"]) / df["close"] > 0.0005)
        ).astype(int)

        df.dropna(inplace=True)
        return df

    @property
    def features(self):
        return [
            "RSI",
            "ATR",
            "SMA50",
            "SMA200",
            "Momentum",
            "MACD_hist",
            "BB_pos",
            "Vol_ratio",
            "Trend",
        ]

    def train(self, df):
        if len(df) < 50:
            return
        X = df[self.features]
        y = df["Target"]
        self.model.fit(X, y)
        self.trained = True

    def predict(self, df):
        if df.empty or len(df) < 200:
            return 0, 0.5
        if not self.trained:
            self.train(df.iloc[:-1])
        last = df.iloc[[-1]]
        prob = self.model.predict_proba(last[self.features])[0][1]
        # Higher thresholds = fewer but higher-quality signals
        signal = 1 if prob > 0.75 else -1 if prob < 0.25 else 0
        return signal, prob


# ═══════════════════════════════════════════════════
#  RISK MANAGER
# ═══════════════════════════════════════════════════
class RiskManager:
    """Centralised risk controls. Bot checks here before every trade."""

    def __init__(self, config):
        self.config = config
        self._reset_daily()

    def _reset_daily(self):
        self.day = date.today()
        self.daily_pnl = 0.0
        self.daily_trades = 0

    def _check_day_rollover(self):
        if date.today() != self.day:
            self._reset_daily()

    def record_close(self, profit):
        self._check_day_rollover()
        self.daily_pnl += profit
        self.daily_trades += 1

    def can_open_trade(self, current_positions: int) -> tuple[bool, str]:
        self._check_day_rollover()
        c = self.config

        if current_positions >= c["max_exposure"]:
            return False, f"Max exposure reached ({c['max_exposure']} positions)"

        if self.daily_pnl <= -abs(c["daily_loss_limit"]):
            return (
                False,
                f"Daily loss limit hit (${self.daily_pnl:.2f}). Bot paused for today.",
            )

        if self.daily_trades >= c["max_daily_trades"]:
            return False, f"Max daily trades reached ({c['max_daily_trades']})"

        return True, "OK"

    def lot_size(self, sl_points) -> float:
        """Kelly-adjusted position sizing with hard cap."""
        info = mt5.account_info()
        if not info:
            return 0.01
        balance = info.balance
        risk_money = balance * self.config["risk_pct"]
        sym_info = mt5.symbol_info(self.config["symbol"])
        if not sym_info:
            return 0.01
        point = sym_info.point
        lot = risk_money / max(sl_points * point * 100, 1)
        lot = round(max(0.01, min(lot, self.config["max_lot"])), 2)
        return lot


# ═══════════════════════════════════════════════════
#  MT5 BOT
# ═══════════════════════════════════════════════════
class MT5Bot:
    def __init__(self, config):
        self.config = config
        self.ai = AIModel()
        self.risk = RiskManager(config)
        self.running = False
        self.trade_history = []  # list of closed profits
        self.stats = {"total": 0, "success": 0, "fail": 0}
        self.log_lines = []

        if not mt5.initialize():
            self._log("❌ MT5 Init Failed")
        else:
            if self.config["account"]:
                result = mt5.login(
                    self.config["account"],
                    password=self.config["password"],
                    server=self.config["server"],
                )
                self._log("✅ Logged in" if result else "❌ Login failed")
            mt5.symbol_select(self.config["symbol"], True)

    # ── LOGGING ──────────────────────────────────────
    def _log(self, msg):
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line)
        self.log_lines.append(line)
        if len(self.log_lines) > 200:
            self.log_lines.pop(0)

    # ── TELEGRAM ─────────────────────────────────────
    def send_telegram(self, msg):
        if not self.config.get("telegram_token"):
            return
        try:
            url = f"https://api.telegram.org/bot{self.config['telegram_token']}/sendMessage"
            requests.post(
                url,
                data={"chat_id": self.config["telegram_chat_id"], "text": msg},
                timeout=5,
            )
        except Exception as e:
            self._log(f"Telegram error: {e}")

    # ── DATA ─────────────────────────────────────────
    def get_data(self, timeframe=None, count=500):
        tf = timeframe or self.config["timeframe"]
        rates = mt5.copy_rates_from_pos(self.config["symbol"], tf, 0, count)
        if rates is None or len(rates) == 0:
            return pd.DataFrame()
        df = pd.DataFrame(rates)
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"], unit="s")
        return df

    # ── MULTI-TIMEFRAME CONFIRMATION ─────────────────
    def multi_tf_confirm(self, signal: int) -> bool:
        """Check that higher timeframe trend agrees with signal."""
        df_h1 = self.get_data(mt5.TIMEFRAME_H1, 210)
        if df_h1.empty:
            return True  # can't confirm, allow anyway
        df_h1 = self.ai.preprocess(df_h1)
        if df_h1.empty:
            return True
        sma50 = df_h1["SMA50"].iloc[-1]
        sma200 = df_h1["SMA200"].iloc[-1]
        if signal == 1 and sma50 < sma200:
            return False  # H1 downtrend — skip buy
        if signal == -1 and sma50 > sma200:
            return False  # H1 uptrend  — skip sell
        return True

    # ── EXECUTE TRADE ─────────────────────────────────
    def execute_trade(self, signal, df):
        tick = mt5.symbol_info_tick(self.config["symbol"])
        info = mt5.symbol_info(self.config["symbol"])
        if not tick or not info:
            return

        atr = df["ATR"].iloc[-1]
        sl_pts = max(int(atr * 2 / info.point), 50)  # min 50-point SL
        tp_pts = sl_pts * self.config["rr_ratio"]  # configurable R:R
        lot = self.risk.lot_size(sl_pts)
        price = tick.ask if signal == 1 else tick.bid
        sl = price - sl_pts * info.point if signal == 1 else price + sl_pts * info.point
        tp = price + tp_pts * info.point if signal == 1 else price - tp_pts * info.point
        order_t = mt5.ORDER_TYPE_BUY if signal == 1 else mt5.ORDER_TYPE_SELL
        label = "BUY" if signal == 1 else "SELL"

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.config["symbol"],
            "volume": lot,
            "type": order_t,
            "price": price,
            "sl": round(sl, info.digits),
            "tp": round(tp, info.digits),
            "magic": self.config["magic_number"],
            "comment": "GoldAIBot",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            self.stats["total"] += 1
            msg = f"✅ {label} | Price: {price} | Lot: {lot} | SL: {round(sl,info.digits)} | TP: {round(tp,info.digits)}"
            self._log(msg)
            self.send_telegram(msg)
        else:
            self._log(f"❌ Order failed: retcode={result.retcode}")

    # ── TRAILING STOP ─────────────────────────────────
    def _trail_position(self, pos, tick, info):
        half_range = (
            (pos.tp - pos.sl) / 2
            if pos.type == mt5.ORDER_TYPE_BUY
            else (pos.sl - pos.tp) / 2
        )
        if pos.type == mt5.ORDER_TYPE_BUY:
            new_sl = tick.bid - half_range
            if new_sl > pos.sl + info.point:
                mt5.order_send(
                    {
                        "action": mt5.TRADE_ACTION_SLTP,
                        "position": pos.ticket,
                        "sl": round(new_sl, info.digits),
                        "tp": pos.tp,
                    }
                )
        else:
            new_sl = tick.ask + half_range
            if new_sl < pos.sl - info.point:
                mt5.order_send(
                    {
                        "action": mt5.TRADE_ACTION_SLTP,
                        "position": pos.ticket,
                        "sl": round(new_sl, info.digits),
                        "tp": pos.tp,
                    }
                )

    # ── MONITOR POSITIONS ─────────────────────────────
    def monitor_positions(self):
        positions = mt5.positions_get(symbol=self.config["symbol"])
        if not positions:
            return
        info = mt5.symbol_info(self.config["symbol"])
        for pos in positions:
            if pos.magic != self.config["magic_number"]:
                continue
            tick = mt5.symbol_info_tick(self.config["symbol"])
            if not tick or not info:
                continue

            # Trailing stop
            if self.config["trailing"]:
                self._trail_position(pos, tick, info)

            # Auto-close on profit/loss targets
            should_close = (
                pos.profit >= self.config["profit_target"]
                or pos.profit <= self.config["loss_limit"]
            )
            if should_close:
                close_t = (
                    mt5.ORDER_TYPE_SELL
                    if pos.type == mt5.ORDER_TYPE_BUY
                    else mt5.ORDER_TYPE_BUY
                )
                result = mt5.order_send(
                    {
                        "action": mt5.TRADE_ACTION_DEAL,
                        "symbol": self.config["symbol"],
                        "volume": pos.volume,
                        "type": close_t,
                        "position": pos.ticket,
                        "magic": pos.magic,
                        "type_time": mt5.ORDER_TIME_GTC,
                        "type_filling": mt5.ORDER_FILLING_IOC,
                    }
                )
                if result.retcode == mt5.TRADE_RETCODE_DONE:
                    profit = pos.profit
                    self.trade_history.append(profit)
                    self.risk.record_close(profit)
                    if profit > 0:
                        self.stats["success"] += 1
                    else:
                        self.stats["fail"] += 1
                    msg = f"💰 Closed | Profit: {profit:+.2f}$ | Daily P&L: {self.risk.daily_pnl:+.2f}$"
                    self._log(msg)
                    self.send_telegram(msg)
                else:
                    self._log(f"❌ Close failed: {result.retcode}")

    # ── MAIN STEP ─────────────────────────────────────
    def run_step(self):
        if not self.running:
            return 0, 0.5, self.stats

        # Monitor existing positions first
        self.monitor_positions()

        # Check if daily limits allow new trades
        positions = mt5.positions_get(symbol=self.config["symbol"]) or []
        bot_positions = [p for p in positions if p.magic == self.config["magic_number"]]
        allowed, reason = self.risk.can_open_trade(len(bot_positions))

        if not allowed:
            self._log(f"⛔ No trade: {reason}")
            return 0, 0.5, self.stats

        # Get data and predict
        df = self.get_data()
        if df.empty:
            return 0, 0.5, self.stats
        df = self.ai.preprocess(df)
        signal, confidence = self.ai.predict(df)

        if signal != 0:
            # Multi-timeframe confirmation
            if self.multi_tf_confirm(signal):
                self.execute_trade(signal, df)
            else:
                self._log(
                    f"⚠️ Signal {'BUY' if signal==1 else 'SELL'} rejected — higher TF trend conflict"
                )

        return signal, confidence, self.stats


# ═══════════════════════════════════════════════════
#  GUI
# ═══════════════════════════════════════════════════
class BotGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pro Gold AI Bot — Enhanced Risk Edition")
        self.config = {
            "account": 261009880,
            "password": "Amine2002@",
            "server": "Exness-MT5Trial16",
            "symbol": "BTCUSDm",
            "timeframe": mt5.TIMEFRAME_M5,
            "risk_pct": 0.01,
            "max_exposure": 2,
            "max_lot": 1.0,
            "profit_target": 3.0,
            "loss_limit": -5.0,
            "trailing": True,
            "magic_number": 123456,
            "daily_loss_limit": -50.0,
            "max_daily_trades": 10,
            "rr_ratio": 2.0,
            "telegram_token": "8423946950:AAF0Ja88p_52coyDsh48nQD1yNZW7NwAgck",
            "telegram_chat_id": "6476316022",
        }
        self.bot = MT5Bot(self.config)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_bot)
        self.init_ui()

    # ── BUILD UI ──────────────────────────────────────
    def init_ui(self):
        root = QVBoxLayout()

        # ---- Config panel ----
        cfg_group = QGroupBox("Configuration")
        form = QFormLayout()

        self.account_input = QLineEdit(str(self.config["account"]))
        self.password_input = QLineEdit(self.config["password"])
        self.password_input.setEchoMode(QLineEdit.Password)
        self.server_input = QLineEdit(self.config["server"])
        self.symbol_input = QLineEdit(self.config["symbol"])
        self.risk_input = QLineEdit(str(self.config["risk_pct"]))
        self.max_exposure_input = QLineEdit(str(self.config["max_exposure"]))
        self.max_lot_input = QLineEdit(str(self.config["max_lot"]))
        self.profit_input = QLineEdit(str(self.config["profit_target"]))
        self.loss_input = QLineEdit(str(self.config["loss_limit"]))
        self.daily_loss_input = QLineEdit(str(self.config["daily_loss_limit"]))
        self.max_daily_trades_input = QLineEdit(str(self.config["max_daily_trades"]))
        self.rr_input = QLineEdit(str(self.config["rr_ratio"]))
        self.trailing_check = QCheckBox()
        self.trailing_check.setChecked(self.config["trailing"])
        self.telegram_token_input = QLineEdit(self.config["telegram_token"])
        self.telegram_chat_input = QLineEdit(self.config["telegram_chat_id"])

        form.addRow("MT5 Account:", self.account_input)
        form.addRow("Password:", self.password_input)
        form.addRow("Server:", self.server_input)
        form.addRow("Symbol:", self.symbol_input)
        form.addRow("Risk % per trade:", self.risk_input)
        form.addRow("Max Exposure (pos):", self.max_exposure_input)
        form.addRow("Max Lot Size:", self.max_lot_input)
        form.addRow("Profit Target $:", self.profit_input)
        form.addRow("Loss Limit $ (pos):", self.loss_input)
        form.addRow("Daily Loss Limit $:", self.daily_loss_input)
        form.addRow("Max Daily Trades:", self.max_daily_trades_input)
        form.addRow("R:R Ratio:", self.rr_input)
        form.addRow("Trailing Stop:", self.trailing_check)
        form.addRow("Telegram Token:", self.telegram_token_input)
        form.addRow("Telegram Chat ID:", self.telegram_chat_input)

        cfg_group.setLayout(form)
        root.addWidget(cfg_group)

        # ---- Buttons ----
        btn_row = QHBoxLayout()
        self.start_btn = QPushButton("▶  Start Bot")
        self.stop_btn = QPushButton("⏹  Stop Bot")
        self.start_btn.clicked.connect(self.start_bot)
        self.stop_btn.clicked.connect(self.stop_bot)
        self.start_btn.setStyleSheet(
            "background:#27ae60;color:white;font-weight:bold;padding:6px"
        )
        self.stop_btn.setStyleSheet(
            "background:#c0392b;color:white;font-weight:bold;padding:6px"
        )
        btn_row.addWidget(self.start_btn)
        btn_row.addWidget(self.stop_btn)
        root.addLayout(btn_row)

        # ---- Status labels ----
        self.conf_label = QLabel("AI Confidence: —")
        self.stats_label = QLabel("Total: 0 | Win: 0 | Loss: 0")
        self.daily_label = QLabel("Daily P&L: $0.00 | Trades today: 0")
        self.circuit_label = QLabel("")
        self.circuit_label.setStyleSheet("color:red;font-weight:bold")
        for lbl in [
            self.conf_label,
            self.stats_label,
            self.daily_label,
            self.circuit_label,
        ]:
            root.addWidget(lbl)

        # ---- Position table ----
        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels(
            ["Ticket", "Type", "Volume", "Price", "Profit", "SL / TP"]
        )
        root.addWidget(self.table)

        # ---- Chart ----
        self.figure = Figure(figsize=(8, 3))
        self.canvas = FigureCanvas(self.figure)
        root.addWidget(self.canvas)

        # ---- Log ----
        self.log_table = QTableWidget()
        self.log_table.setColumnCount(1)
        self.log_table.setHorizontalHeaderLabels(["Activity Log"])
        self.log_table.horizontalHeader().setStretchLastSection(True)
        self.log_table.setMaximumHeight(130)
        root.addWidget(self.log_table)

        self.setLayout(root)

    # ── START ─────────────────────────────────────────
    def start_bot(self):
        try:
            self.config["account"] = int(self.account_input.text())
            self.config["password"] = self.password_input.text()
            self.config["server"] = self.server_input.text()
            self.config["symbol"] = self.symbol_input.text()
            self.config["risk_pct"] = float(self.risk_input.text())
            self.config["max_exposure"] = int(self.max_exposure_input.text())
            self.config["max_lot"] = float(self.max_lot_input.text())
            self.config["profit_target"] = float(self.profit_input.text())
            self.config["loss_limit"] = float(self.loss_input.text())
            self.config["daily_loss_limit"] = float(self.daily_loss_input.text())
            self.config["max_daily_trades"] = int(self.max_daily_trades_input.text())
            self.config["rr_ratio"] = float(self.rr_input.text())
            self.config["trailing"] = self.trailing_check.isChecked()
            self.config["telegram_token"] = self.telegram_token_input.text()
            self.config["telegram_chat_id"] = self.telegram_chat_input.text()
        except ValueError as e:
            self.circuit_label.setText(f"⚠️ Config error: {e}")
            return

        # Re-init bot with new config
        self.bot = MT5Bot(self.config)
        self.bot.running = True
        self.timer.start(2000)
        self.circuit_label.setText("")

    # ── STOP ──────────────────────────────────────────
    def stop_bot(self):
        self.bot.running = False
        self.timer.stop()
        self.circuit_label.setText("⏹ Bot stopped.")

    # ── PERIODIC UPDATE ───────────────────────────────
    def update_bot(self):
        signal, confidence, stats = self.bot.run_step()

        # Status
        conf_color = (
            "#27ae60" if signal == 1 else "#c0392b" if signal == -1 else "#7f8c8d"
        )
        self.conf_label.setText(
            f"AI Confidence: {confidence:.2%}  |  Signal: {'BUY 📈' if signal==1 else 'SELL 📉' if signal==-1 else 'FLAT ⏸'}"
        )
        self.conf_label.setStyleSheet(f"color:{conf_color};font-weight:bold")
        self.stats_label.setText(
            f"Total: {stats['total']} | Win: {stats['success']} | Loss: {stats['fail']}"
        )

        # Daily stats
        r = self.bot.risk
        pnl_color = "#27ae60" if r.daily_pnl >= 0 else "#c0392b"
        self.daily_label.setText(
            f"Daily P&L: <span style='color:{pnl_color}'>${r.daily_pnl:+.2f}</span>  |  Trades today: {r.daily_trades}"
        )

        # Circuit-breaker warning
        if r.daily_pnl <= -abs(self.config["daily_loss_limit"]):
            self.circuit_label.setText(
                "🚨 DAILY LOSS LIMIT HIT — No new trades until tomorrow."
            )
        else:
            self.circuit_label.setText("")

        # Position table
        positions = mt5.positions_get(symbol=self.config["symbol"]) or []
        self.table.setRowCount(len(positions))
        for i, pos in enumerate(positions):
            label = "BUY" if pos.type == mt5.ORDER_TYPE_BUY else "SELL"
            profit_item = QTableWidgetItem(f"{pos.profit:+.2f}")
            profit_item.setForeground(
                QColor("#27ae60") if pos.profit >= 0 else QColor("#c0392b")
            )
            self.table.setItem(i, 0, QTableWidgetItem(str(pos.ticket)))
            self.table.setItem(i, 1, QTableWidgetItem(label))
            self.table.setItem(i, 2, QTableWidgetItem(str(pos.volume)))
            self.table.setItem(i, 3, QTableWidgetItem(f"{pos.price_open:.5f}"))
            self.table.setItem(i, 4, profit_item)
            self.table.setItem(i, 5, QTableWidgetItem(f"{pos.sl:.5f} / {pos.tp:.5f}"))

        # Equity curve
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        if self.bot.trade_history:
            cumulative = np.cumsum(self.bot.trade_history)
            color = "green" if cumulative[-1] >= 0 else "red"
            ax.plot(cumulative, color=color, linewidth=1.5, label="Cumulative P&L $")
            ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
            ax.fill_between(
                range(len(cumulative)),
                cumulative,
                0,
                where=(cumulative >= 0),
                alpha=0.15,
                color="green",
            )
            ax.fill_between(
                range(len(cumulative)),
                cumulative,
                0,
                where=(cumulative < 0),
                alpha=0.15,
                color="red",
            )
            ax.set_ylabel("Cumulative P&L $")
            ax.set_xlabel("Closed Trades")
            ax.set_title("Bot Equity Curve")
            ax.legend(fontsize=8)
        self.canvas.draw()

        # Log panel
        lines = self.bot.log_lines[-20:]
        self.log_table.setRowCount(len(lines))
        for i, line in enumerate(reversed(lines)):
            self.log_table.setItem(i, 0, QTableWidgetItem(line))


# ═══════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════
if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = BotGUI()
    gui.resize(1280, 900)
    gui.show()
    sys.exit(app.exec_())
