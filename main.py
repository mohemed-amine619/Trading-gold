import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
import logging
import threading
import winsound
import requests
import customtkinter as ctk
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from tkinter import messagebox

# --- 1. GUI LOGGING HANDLER ---
class GuiLogHandler(logging.Handler):
    def __init__(self, text_widget):
        super().__init__()
        self.text_widget = text_widget

    def emit(self, record):
        msg = self.format(record)
        try:
            self.text_widget.configure(state="normal")
            self.text_widget.insert("end", msg + "\n")
            self.text_widget.see("end")j
            self.text_widget.configure(state="disabled")
        except Exception:
            pass

# --- 2. LOGIC CLASSES ---

class NotificationHandler:
    def __init__(self, config):
        self.config = config

    def send(self, message, type="INFO"):
        if self.config.get("enable_sound", True):
            try:
                if type == "TRADE": winsound.Beep(1000, 200)
                elif type == "CLOSE": winsound.Beep(800, 400)
                elif type == "ERROR": winsound.Beep(400, 1000)
            except: pass

        token = self.config.get("telegram_token")
        chat_id = self.config.get("telegram_chat_id")
        if token and chat_id:
            try:
                url = f"https://api.telegram.org/bot{token}/sendMessage"
                requests.post(url, data={"chat_id": chat_id, "text": message}, timeout=5)
            except Exception as e:
                logging.getLogger("Bot").error(f"Telegram Fail: {e}")

class AIModel:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, min_samples_split=10, random_state=42)
        self.is_trained = False
        self.logger = logging.getLogger("Bot")

    def preprocess_data(self, df):
        df = df.copy()
        df["SMA_50"] = df["close"].rolling(window=50).mean()
        df["SMA_20"] = df["close"].rolling(window=20).mean()
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df["RSI"] = 100 - (100 / (1 + rs))
        df["h-l"] = df["high"] - df["low"]
        df["h-pc"] = abs(df["high"] - df["close"].shift(1))
        df["l-pc"] = abs(df["low"] - df["close"].shift(1))
        df["tr"] = df[["h-l", "h-pc", "l-pc"]].max(axis=1)
        df["ATR"] = df["tr"].rolling(window=14).mean()
        df["Dist_SMA50"] = df["close"] - df["SMA_50"]
        df["Candle_Size"] = df["close"] - df["open"]
        df["Target"] = (df["close"].shift(-1) > df["close"]).astype(int)
        df.dropna(inplace=True)
        return df

    def predict(self, df):
        if len(df) < 100: return 0
        if not self.is_trained:
            self.model.fit(df.iloc[:-1][["RSI", "Dist_SMA50", "Candle_Size", "ATR", "SMA_20"]], df.iloc[:-1]["Target"])
            self.is_trained = True
            self.logger.info("AI Trained.")

        last = df.iloc[[-1]][["RSI", "Dist_SMA50", "Candle_Size", "ATR", "SMA_20"]]
        if last.isnull().values.any(): return 0
        
        probs = self.model.predict_proba(last)[0]
        prob_up = probs[1]
        
        current_price = df.iloc[-1]["close"]
        self.logger.info(f"AI Prediction: Prob UP = {prob_up:.2f} (Price: {current_price})")

        # --- EXPLICIT BUY/SELL LOGIC ---
        if prob_up > 0.55:
            self.logger.info(f"AI Signals BUY (Confidence: {prob_up:.2f})")
            return 1
        elif prob_up < 0.45:
            self.logger.info(f"AI Signals SELL (Confidence: {1-prob_up:.2f})")
            return -1
        else:
            self.logger.info("AI is uncertain. No trade.")
            return 0

class MT5TradingBot:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("Bot")
        self.ai = AIModel()
        self.notifier = NotificationHandler(config)
        self.running = False
        self.last_bar_time = None
        self.failed_tickets = {}
        self.dry_run_tickets = set()
    
    def connect(self):
        if not mt5.initialize():
            self.logger.error(f"MT5 Init Failed: {mt5.last_error()}")
            return False
        
        authorized = mt5.login(self.config["account"], password=self.config["password"], server=self.config["server"])
        if authorized:
            self.logger.info(f"Connected. Balance: {mt5.account_info().balance}")
            if not mt5.symbol_select(self.config["symbol"], True):
                self.logger.error(f"Failed to select symbol '{self.config['symbol']}'.")
                return False
        else:
            self.logger.error(f"Login Failed: {mt5.last_error()}")
            return False
        return True

    def check_safety_shields(self):
        account = mt5.account_info()
        if account is None: return True
        drawdown_threshold = account.balance * (1 - (self.config["max_drawdown_pct"] / 100))
        if account.equity < drawdown_threshold:
            msg = "STOP: Equity dropped below 50%! Emergency Halt."
            self.logger.critical(msg)
            self.notifier.send(msg, type="ERROR")
            self.panic_close_all()
            return False
        return True

    def panic_close_all(self):
        positions = mt5.positions_get()
        if positions:
            for pos in positions:
                self.close_position(pos, reason="EMERGENCY STOP")
        self.running = False

    # --- EXECUTION LOGIC (SEPARATED BUY/SELL) ---
    def execute_trade(self, signal, atr_value=0):
        if not self.running: return

        symbol = self.config["symbol"]
        lot = self.config["lot_size"]
        tick = mt5.symbol_info_tick(symbol)
        if not tick: return
        
        s_info = mt5.symbol_info(symbol)
        if not s_info: return

        # Spread Check
        spread_points = (tick.ask - tick.bid) / s_info.point
        if spread_points > self.config["max_spread_points"]:
            self.logger.warning(f"Spread {spread_points} too high. Waiting.")
            return

        point = s_info.point
        digits = s_info.digits

        # 1. SETUP VARIABLES
        price = 0.0
        sl = 0.0
        tp = 0.0
        type_order = 0

        # 2. CALCULATE BUY vs SELL
        if signal == 1:
            # --- BUY LOGIC ---
            price = tick.ask
            type_order = mt5.ORDER_TYPE_BUY
            
            if self.config["use_atr_stops"] and atr_value > 0:
                sl_dist = atr_value * self.config["atr_multiplier"]
                sl = price - sl_dist
                tp_dist = self.config["take_profit_points"] * point
                tp = price + tp_dist
            else:
                sl = price - (self.config["stop_loss_points"] * point)
                tp = price + (self.config["take_profit_points"] * point)

        elif signal == -1:
            # --- SELL LOGIC ---
            price = tick.bid
            type_order = mt5.ORDER_TYPE_SELL
            
            if self.config["use_atr_stops"] and atr_value > 0:
                sl_dist = atr_value * self.config["atr_multiplier"]
                sl = price + sl_dist
                tp_dist = self.config["take_profit_points"] * point
                tp = price - tp_dist
            else:
                sl = price + (self.config["stop_loss_points"] * point)
                tp = price - (self.config["take_profit_points"] * point)

        # Rounding
        sl = round(sl, digits)
        tp = round(tp, digits)

        # 3. DRY RUN CHECK
        if self.config.get("dry_run", False):
            side = "BUY" if signal == 1 else "SELL"
            self.logger.info(f"DRY RUN: Would have opened {side} @ {price} | SL: {sl} | TP: {tp}")
            return

        # 4. SEND ORDER
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot,
            "type": type_order,
            "price": price,
            "sl": float(sl),
            "tp": float(tp),
            "magic": self.config["magic_number"],
            "comment": "RealAI",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            side = "BUY" if signal == 1 else "SELL"
            msg = f"OPENED {side} @ {price}"
            self.logger.info(msg)
            self.notifier.send(msg, type="TRADE")
        elif result.retcode == 10026:
            self.logger.critical("AutoTrading Blocked! Check MT5 Settings.")
        else:
            self.logger.error(f"Order Failed: {result.comment}")

    def close_position(self, position, reason="Signal"):
        if not self.running: return False
        
        tick = mt5.symbol_info_tick(position.symbol)
        if not tick: return False

        # Explicit Type Logic
        close_type = mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
        close_price = tick.bid if position.type == mt5.ORDER_TYPE_BUY else tick.ask

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": position.symbol,
            "volume": position.volume,
            "type": close_type,
            "position": position.ticket,
            "price": close_price,
            "magic": self.config["magic_number"],
            "comment": f"Close-{reason}"[:31],
        }

        if self.config.get("dry_run", False):
            if position.ticket not in self.dry_run_tickets:
                self.logger.info(f"DRY RUN: Closing {position.ticket}")
                self.dry_run_tickets.add(position.ticket)
            return True

        res = mt5.order_send(request)
        if res.retcode == mt5.TRADE_RETCODE_DONE:
            self.notifier.send(f"CLOSED {position.ticket}. Profit: {position.profit}", type="CLOSE")
            self.logger.info(f"Closed {position.ticket} ({reason})")
            return True
        return False

    def modify_sl(self, pos, new_sl):
        if self.config.get("dry_run", False):
             self.logger.info(f"DRY RUN: Move SL {pos.ticket} -> {new_sl}")
             return

        req = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": pos.ticket,
            "symbol": pos.symbol,
            "sl": new_sl,
            "tp": pos.tp,
            "magic": self.config["magic_number"]
        }
        mt5.order_send(req)
        self.logger.info(f"SL Moved -> {new_sl}")

    def manage_positions(self, df=None):
        positions = mt5.positions_get(symbol=self.config["symbol"])
        if positions is None: return 0

        point = mt5.symbol_info(self.config["symbol"]).point
        be_trigger = self.config["breakeven_trigger_points"] * point
        be_cushion = self.config["breakeven_cushion_points"] * point
        trail_dist = self.config["trailing_stop_points"] * point
        min_step = self.config["min_sl_step_points"] * point

        for pos in positions:
            if not self.running: break
            if pos.magic != self.config["magic_number"]: continue

            # Hard Profit Target
            if pos.profit >= self.config["target_profit_money"]:
                self.close_position(pos, reason="Profit Target Hit")
                continue

            tick = mt5.symbol_info_tick(self.config["symbol"])
            if not tick: continue

            # --- BUY POSITION MANAGEMENT ---
            if pos.type == mt5.ORDER_TYPE_BUY:
                current_price = tick.bid
                distance_moved = current_price - pos.price_open

                # A. Breakeven
                if distance_moved > be_trigger:
                    new_sl = pos.price_open + be_cushion
                    if pos.sl < new_sl:
                        self.modify_sl(pos, new_sl)
                        continue
                
                # B. Trailing
                if distance_moved > be_trigger:
                    calculated_sl = current_price - trail_dist
                    if calculated_sl > (pos.sl + min_step):
                        self.modify_sl(pos, calculated_sl)

            # --- SELL POSITION MANAGEMENT ---
            elif pos.type == mt5.ORDER_TYPE_SELL:
                current_price = tick.ask
                distance_moved = pos.price_open - current_price

                # A. Breakeven
                if distance_moved > be_trigger:
                    new_sl = pos.price_open - be_cushion
                    if pos.sl == 0.0 or pos.sl > new_sl:
                        self.modify_sl(pos, new_sl)
                        continue
                
                # B. Trailing
                if distance_moved > be_trigger:
                    calculated_sl = current_price + trail_dist
                    if pos.sl == 0.0 or calculated_sl < (pos.sl - min_step):
                        self.modify_sl(pos, calculated_sl)
        
        return len(positions)

    def run(self):
        self.running = True
        if not self.connect(): 
            self.running = False
            return
        
        self.logger.info("Bot Running...")
        while self.running:
            try:
                if not self.check_safety_shields(): break
                
                rates = mt5.copy_rates_from_pos(self.config["symbol"], self.config["timeframe"], 0, 1000)
                if rates is not None:
                    df = pd.DataFrame(rates)
                    df["time"] = pd.to_datetime(df["time"], unit="s")
                    
                    self.manage_positions(df)

                    if self.last_bar_time != df.iloc[-1]["time"]:
                        self.last_bar_time = df.iloc[-1]["time"]
                        df_proc = self.ai.preprocess_data(df)
                        signal = self.ai.predict(df_proc)
                        atr = df_proc.iloc[-1]["ATR"]
                        
                        if signal != 0 and len(mt5.positions_get(symbol=self.config["symbol"])) < self.config["max_exposure"]:
                            self.execute_trade(signal, atr)
                            
                time.sleep(self.config["sleep_interval"])
            except Exception as e:
                self.logger.error(f"Error: {e}")
                time.sleep(2)
        mt5.shutdown()
        self.logger.info("Bot Stopped")

# --- 3. GUI APPLICATION ---
class TradingApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("blue")
        
        self.title("Fullstack Master - AI Bot PRO")
        self.geometry("1100x750")
        
        self.bot_instance = None
        self.bot_thread = None
        self.entries = {}

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.sidebar = ctk.CTkScrollableFrame(self, width=300, label_text="CONFIGURATION")
        self.sidebar.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        
        ctk.CTkLabel(self.main_frame, text="Live Logs", font=("Arial", 16, "bold")).pack(pady=5)
        self.log_box = ctk.CTkTextbox(self.main_frame, font=("Consolas", 12))
        self.log_box.pack(fill="both", expand=True, padx=5, pady=5)
        
        self.setup_logging()
        self.build_config_form()
        self.add_control_buttons()

    def setup_logging(self):
        logger = logging.getLogger("Bot")
        logger.setLevel(logging.INFO)
        if logger.hasHandlers(): logger.handlers.clear()
        handler = GuiLogHandler(self.log_box)
        handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s', '%H:%M:%S'))
        logger.addHandler(handler)

    def add_input(self, key, label, default, is_password=False):
        ctk.CTkLabel(self.sidebar, text=label, anchor="w").pack(fill="x", pady=(5,0))
        entry = ctk.CTkEntry(self.sidebar, show="*" if is_password else None)
        entry.insert(0, str(default))
        entry.pack(fill="x", pady=(0,5))
        self.entries[key] = entry

    def add_checkbox(self, key, label, default=False):
        var = ctk.BooleanVar(value=default)
        cb = ctk.CTkCheckBox(self.sidebar, text=label, variable=var)
        cb.pack(fill="x", pady=5)
        self.entries[key] = var

    def build_config_form(self):
        ctk.CTkLabel(self.sidebar, text="--- CONNECTION ---", text_color="cyan").pack(pady=10)
        self.add_input("account", "Account ID", "261009880")
        self.add_input("password", "Password", "Amine2002@", is_password=True)
        self.add_input("server", "Server Name", "Exness-MT5Trial16")
        
        ctk.CTkLabel(self.sidebar, text="--- TRADING ---", text_color="cyan").pack(pady=10)
        self.add_input("symbol", "Symbol", "BTCUSDm")
        self.add_input("lot_size", "Lot Size", "0.5")
        self.add_input("magic_number", "Magic Number", "234000")
        self.add_input("sleep_interval", "Sleep (sec)", "0.5")
        
        ctk.CTkLabel(self.sidebar, text="--- RISK & TARGETS ---", text_color="cyan").pack(pady=10)
        self.add_input("target_profit_money", "Target Profit ($)", "50.0")
        self.add_input("stop_loss_points", "Stop Loss (Points)", "10000")
        self.add_input("take_profit_points", "Take Profit (Points)", "20000")
        self.add_input("max_spread_points", "Max Spread (Points)", "5000")
        self.add_input("max_exposure", "Max Positions", "10")
        self.add_input("max_drawdown_pct", "Max Drawdown %", "50.0")

        ctk.CTkLabel(self.sidebar, text="--- TRAILING STOP ---", text_color="cyan").pack(pady=10)
        self.add_input("breakeven_trigger_points", "BE Trigger (Points)", "15000")
        self.add_input("breakeven_cushion_points", "BE Cushion (Points)", "2000")
        self.add_input("trailing_stop_points", "Trailing Dist (Points)", "20000")
        self.add_input("min_sl_step_points", "Min Step (Points)", "100")
        
        ctk.CTkLabel(self.sidebar, text="--- VOLATILITY (ATR) ---", text_color="cyan").pack(pady=10)
        self.add_checkbox("use_atr_stops", "Use ATR for SL", True)
        self.add_input("atr_multiplier", "ATR Multiplier", "3.0")

        ctk.CTkLabel(self.sidebar, text="--- TELEGRAM ---", text_color="cyan").pack(pady=10)
        self.add_input("telegram_token", "Bot Token", "8423946950:AAF0Ja88p_52coyDsh48nQD1yNZW7NwAgck")
        self.add_input("telegram_chat_id", "Chat ID", "6476316022")

        ctk.CTkLabel(self.sidebar, text="--- SYSTEM ---", text_color="cyan").pack(pady=10)
        self.add_checkbox("enable_sound", "Enable Sound Alerts", True)
        self.add_checkbox("dry_run", "Dry Run (Simulate Only)", False)

    def add_control_buttons(self):
        self.btn_start = ctk.CTkButton(self.sidebar, text="START BOT", fg_color="green", height=40, command=self.start_bot)
        self.btn_start.pack(pady=20, fill="x")
        self.btn_stop = ctk.CTkButton(self.sidebar, text="STOP BOT", fg_color="red", height=40, state="disabled", command=self.stop_bot)
        self.btn_stop.pack(pady=5, fill="x")

    def get_config_from_gui(self):
        try:
            return {
                "account": int(self.entries["account"].get()),
                "password": self.entries["password"].get(),
                "server": self.entries["server"].get(),
                "symbol": self.entries["symbol"].get(),
                "lot_size": float(self.entries["lot_size"].get()),
                "magic_number": int(self.entries["magic_number"].get()),
                "sleep_interval": float(self.entries["sleep_interval"].get()),
                "target_profit_money": float(self.entries["target_profit_money"].get()),
                "stop_loss_points": int(self.entries["stop_loss_points"].get()),
                "take_profit_points": int(self.entries["take_profit_points"].get()),
                "max_spread_points": int(self.entries["max_spread_points"].get()),
                "max_exposure": int(self.entries["max_exposure"].get()),
                "max_drawdown_pct": float(self.entries["max_drawdown_pct"].get()),
                "breakeven_trigger_points": int(self.entries["breakeven_trigger_points"].get()),
                "breakeven_cushion_points": int(self.entries["breakeven_cushion_points"].get()),
                "trailing_stop_points": int(self.entries["trailing_stop_points"].get()),
                "min_sl_step_points": int(self.entries["min_sl_step_points"].get()),
                "use_atr_stops": self.entries["use_atr_stops"].get(),
                "atr_multiplier": float(self.entries["atr_multiplier"].get()),
                "telegram_token": self.entries["telegram_token"].get(),
                "telegram_chat_id": self.entries["telegram_chat_id"].get(),
                "enable_sound": self.entries["enable_sound"].get(),
                "dry_run": self.entries["dry_run"].get(),
                "timeframe": mt5.TIMEFRAME_M15
            }
        except ValueError as e:
            messagebox.showerror("Format Error", f"Please check your numbers.\n\nError: {e}")
            return None

    def start_bot(self):
        config = self.get_config_from_gui()
        if not config: return
        self.bot_instance = MT5TradingBot(config)
        self.bot_thread = threading.Thread(target=self.bot_instance.run, daemon=True)
        self.bot_thread.start()
        self.btn_start.configure(state="disabled")
        self.btn_stop.configure(state="normal")

    def stop_bot(self):
        if self.bot_instance:
            self.bot_instance.running = False
            logging.getLogger("Bot").info("Stop Signal Sent...")
        self.btn_start.configure(state="normal")
        self.btn_stop.configure(state="disabled")

if __name__ == "__main__":
    app = TradingApp()
    app.mainloop()