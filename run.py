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
)
from PyQt5.QtCore import QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from sklearn.ensemble import RandomForestClassifier
import requests


# ---------------- AI MODEL -----------------
class AIModel:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=200, max_depth=10)
        self.trained = False

    def preprocess(self, df):
        df = df.copy()
        if "close" not in df.columns or len(df) < 200:
            return pd.DataFrame()
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = gain / loss
        df["RSI"] = 100 - (100 / (1 + rs))
        df["H-L"] = df["high"] - df["low"]
        df["H-C"] = abs(df["high"] - df["close"].shift())
        df["L-C"] = abs(df["low"] - df["close"].shift())
        df["TR"] = df[["H-L", "H-C", "L-C"]].max(axis=1)
        df["ATR"] = df["TR"].rolling(14).mean()
        df["SMA50"] = df["close"].rolling(50).mean()
        df["SMA200"] = df["close"].rolling(200).mean()
        df["Momentum"] = df["close"] - df["close"].shift(10)
        df["Target"] = (df["close"].shift(-3) > df["close"]).astype(int)
        df.dropna(inplace=True)
        return df

    def train(self, df):
        X = df[["RSI", "ATR", "SMA50", "SMA200", "Momentum"]]
        y = df["Target"]
        self.model.fit(X, y)
        self.trained = True

    def predict(self, df):
        if df.empty or len(df) < 200:
            return 0, 0.5
        if not self.trained:
            self.train(df.iloc[:-1])
        last = df.iloc[[-1]]
        prob = self.model.predict_proba(
            last[["RSI", "ATR", "SMA50", "SMA200", "Momentum"]]
        )[0][1]
        signal = 1 if prob > 0.7 else -1 if prob < 0.3 else 0
        return signal, prob


# ---------------- MT5 BOT -----------------
class MT5Bot:
    def __init__(self, config):
        self.config = config
        self.ai = AIModel()
        self.running = False
        self.trade_history = []
        self.stats = {"total": 0, "success": 0, "fail": 0}

        if not mt5.initialize():
            print("❌ MT5 Init Failed")
        else:
            if self.config["account"]:
                mt5.login(
                    self.config["account"],
                    password=self.config["password"],
                    server=self.config["server"],
                )
            mt5.symbol_select(self.config["symbol"], True)

    # ---------- TELEGRAM ----------
    def send_telegram(self, msg):
        if not self.config["telegram_token"]:
            return
        try:
            url = f"https://api.telegram.org/bot{self.config['telegram_token']}/sendMessage"
            requests.post(
                url, data={"chat_id": self.config["telegram_chat_id"], "text": msg}
            )
        except Exception as e:
            print("Telegram error:", e)

    # ---------- DATA ----------
    def get_data(self):
        rates = mt5.copy_rates_from_pos(
            self.config["symbol"], self.config["timeframe"], 0, 500
        )
        df = pd.DataFrame(rates)
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"], unit="s")
        return df

    # ---------- LOT SIZE ----------
    def lot_size(self, sl_points):
        balance = mt5.account_info().balance
        risk_money = balance * self.config["risk_pct"]
        point = mt5.symbol_info(self.config["symbol"]).point
        lot = risk_money / (sl_points * point * 100)
        return round(max(0.01, lot), 2)

    # ---------- CHECK CAN TRADE ----------
    def can_trade(self):
        positions = mt5.positions_get(symbol=self.config["symbol"])
        bot_positions = (
            [p for p in positions if p.magic == self.config["magic_number"]]
            if positions
            else []
        )
        return len(bot_positions) < self.config["max_exposure"]

    # ---------- EXECUTE TRADE ----------
    def execute_trade(self, signal, df):
        tick = mt5.symbol_info_tick(self.config["symbol"])
        info = mt5.symbol_info(self.config["symbol"])
        if not tick or not info:
            return

        atr = df["ATR"].iloc[-1]
        sl_points = int(atr * 2 / info.point)
        tp_points = sl_points * 2
        lot = self.lot_size(sl_points)
        price = tick.ask if signal == 1 else tick.bid
        sl = (
            price - sl_points * info.point
            if signal == 1
            else price + sl_points * info.point
        )
        tp = (
            price + tp_points * info.point
            if signal == 1
            else price - tp_points * info.point
        )
        order_type = mt5.ORDER_TYPE_BUY if signal == 1 else mt5.ORDER_TYPE_SELL

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.config["symbol"],
            "volume": lot,
            "type": order_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "magic": self.config["magic_number"],
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            self.stats["total"] += 1
            self.send_telegram(
                f"✅ Trade executed: {'BUY' if signal==1 else 'SELL'} Price: {price}"
            )
        else:
            print("❌ Trade error:", result.retcode)

    # ---------- MONITOR POSITIONS ----------
    def monitor_positions(self):
        positions = mt5.positions_get(symbol=self.config["symbol"])
        if not positions:
            return
        for pos in positions:
            if pos.magic != self.config["magic_number"]:
                continue
            tick = mt5.symbol_info_tick(self.config["symbol"])
            if not tick:
                continue

            # Trailing
            if self.config["trailing"]:
                if pos.type == mt5.ORDER_TYPE_BUY:
                    new_sl = max(pos.sl, tick.bid - (pos.tp - pos.sl) / 2)
                    if new_sl > pos.sl:
                        mt5.order_send(
                            {
                                "action": mt5.TRADE_ACTION_SLTP,
                                "position": pos.ticket,
                                "sl": new_sl,
                                "tp": pos.tp,
                            }
                        )
                elif pos.type == mt5.ORDER_TYPE_SELL:
                    new_sl = min(pos.sl, tick.ask + (pos.sl - pos.tp) / 2)
                    if new_sl < pos.sl:
                        mt5.order_send(
                            {
                                "action": mt5.TRADE_ACTION_SLTP,
                                "position": pos.ticket,
                                "sl": new_sl,
                                "tp": pos.tp,
                            }
                        )

            # Auto-close
            try:
                if (
                    pos.profit >= self.config["profit_target"]
                    or pos.profit <= self.config["loss_limit"]
                ):
                    close_type = (
                        mt5.ORDER_TYPE_SELL
                        if pos.type == mt5.ORDER_TYPE_BUY
                        else mt5.ORDER_TYPE_BUY
                    )
                    result = mt5.order_send(
                        {
                            "action": mt5.TRADE_ACTION_DEAL,
                            "symbol": self.config["symbol"],
                            "volume": pos.volume,
                            "type": close_type,
                            "position": pos.ticket,
                            "magic": pos.magic,
                            "type_time": mt5.ORDER_TIME_GTC,
                            "type_filling": mt5.ORDER_FILLING_IOC,
                        }
                    )
                    if result.retcode == mt5.TRADE_RETCODE_DONE:
                        self.stats["success"] += 1 if pos.profit > 0 else 0
                        self.stats["fail"] += 1 if pos.profit < 0 else 0
                        self.trade_history.append(pos.profit)
                        self.send_telegram(
                            f"💰 Position closed. Profit: {pos.profit:.2f}$"
                        )
                    else:
                        print("❌ Close failed:", result.retcode)
            except Exception as e:
                print("Error closing position:", e)

    def run_step(self):
        if not self.running:
            return 0, 0.5, self.stats
        df = self.get_data()
        if df.empty:
            return 0, 0.5, self.stats
        df = self.ai.preprocess(df)
        signal, confidence = self.ai.predict(df)
        if signal != 0 and self.can_trade():
            self.execute_trade(signal, df)
        self.monitor_positions()
        return signal, confidence, self.stats


# ---------------- GUI -----------------
class BotGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pro Gold AI Bot")
        # Default config
        self.config = {
            "account": 261009880,
            "password": "Amine2002@",
            "server": "Exness-MT5Trial16",
            "symbol": "XAUUSDm",
            "timeframe": mt5.TIMEFRAME_M5,
            "risk_pct": 0.01,
            "max_exposure": 20,
            "profit_target": 3.0,
            "loss_limit": -5.0,
            "trailing": True,
            "magic_number": 123456,
            "telegram_token": "8423946950:AAF0Ja88p_52coyDsh48nQD1yNZW7NwAgck",
            "telegram_chat_id": "6476316022",
        }
        self.bot = MT5Bot(self.config)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_bot)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        config_group = QGroupBox("Configuration")
        form = QFormLayout()

        self.account_input = QLineEdit(str(self.config["account"]))
        self.password_input = QLineEdit(str(self.config["password"]))
        self.server_input = QLineEdit(str(self.config["server"]))
        self.symbol_input = QLineEdit(self.config["symbol"])
        self.risk_input = QLineEdit(str(self.config["risk_pct"]))
        self.max_exposure_input = QLineEdit(str(self.config["max_exposure"]))
        self.profit_input = QLineEdit(str(self.config["profit_target"]))
        self.loss_input = QLineEdit(str(self.config["loss_limit"]))
        self.trailing_input = QLineEdit(str(self.config["trailing"]))
        self.telegram_token_input = QLineEdit(str(self.config["telegram_token"]))
        self.telegram_chat_input = QLineEdit(str(self.config["telegram_chat_id"]))

        form.addRow("MT5 Account:", self.account_input)
        form.addRow("Password:", self.password_input)
        form.addRow("Server:", self.server_input)
        form.addRow("Symbol:", self.symbol_input)
        form.addRow("Risk %:", self.risk_input)
        form.addRow("Max Exposure:", self.max_exposure_input)
        form.addRow("Profit Target $:", self.profit_input)
        form.addRow("Loss Limit $:", self.loss_input)
        form.addRow("Trailing:", self.trailing_input)
        form.addRow("Telegram Token:", self.telegram_token_input)
        form.addRow("Telegram Chat ID:", self.telegram_chat_input)

        config_group.setLayout(form)
        layout.addWidget(config_group)

        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("Start Bot")
        self.start_btn.clicked.connect(self.start_bot)
        self.stop_btn = QPushButton("Stop Bot")
        self.stop_btn.clicked.connect(self.stop_bot)
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.stop_btn)
        layout.addLayout(btn_layout)

        self.conf_label = QLabel("AI Confidence: 0.0")
        self.stats_label = QLabel("Total Trades: 0 | Success: 0 | Fail: 0")
        layout.addWidget(self.conf_label)
        layout.addWidget(self.stats_label)

        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(
            ["Ticket", "Type", "Volume", "Profit", "SL/TP"]
        )
        layout.addWidget(self.table)

        self.figure = Figure(figsize=(8, 4))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        self.setLayout(layout)

    def start_bot(self):
        self.config["account"] = int(self.account_input.text())
        self.config["password"] = self.password_input.text()
        self.config["server"] = self.server_input.text()
        self.config["symbol"] = self.symbol_input.text()
        self.config["risk_pct"] = float(self.risk_input.text())
        self.config["max_exposure"] = int(self.max_exposure_input.text())
        self.config["profit_target"] = float(self.profit_input.text())
        self.config["loss_limit"] = float(self.loss_input.text())
        self.config["trailing"] = self.trailing_input.text().lower() in [
            "true",
            "1",
            "yes",
        ]
        self.config["telegram_token"] = self.telegram_token_input.text()
        self.config["telegram_chat_id"] = self.telegram_chat_input.text()
        self.bot.running = True
        self.timer.start(2000)

    def stop_bot(self):
        self.bot.running = False
        self.timer.stop()

    def update_bot(self):
        signal, confidence, stats = self.bot.run_step()
        self.conf_label.setText(f"AI Confidence: {confidence:.2f}")
        self.stats_label.setText(
            f"Total Trades: {stats['total']} | Success: {stats['success']} | Fail: {stats['fail']}"
        )

        positions = mt5.positions_get(symbol=self.config["symbol"])
        self.table.setRowCount(len(positions) if positions else 0)
        if positions:
            for i, pos in enumerate(positions):
                self.table.setItem(i, 0, QTableWidgetItem(str(pos.ticket)))
                self.table.setItem(
                    i,
                    1,
                    QTableWidgetItem(
                        "BUY" if pos.type == mt5.ORDER_TYPE_BUY else "SELL"
                    ),
                )
                self.table.setItem(i, 2, QTableWidgetItem(str(pos.volume)))
                self.table.setItem(i, 3, QTableWidgetItem(f"{pos.profit:.2f}"))
                self.table.setItem(i, 4, QTableWidgetItem(f"{pos.sl:.2f}/{pos.tp:.2f}"))

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        if self.bot.trade_history:
            cumulative = np.cumsum(self.bot.trade_history)
            ax.plot(cumulative, label="Cumulative Profit $", color="blue")
            ax.set_ylabel("Cumulative Profit $")
            ax.set_xlabel("Closed Trades")
            ax.set_title("Bot Performance")
            ax.legend()
        self.canvas.draw()


# ---------------- RUN -----------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = BotGUI()
    gui.resize(1200, 800)
    gui.show()
    sys.exit(app.exec_())
