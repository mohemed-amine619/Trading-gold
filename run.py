import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
import logging
from datetime import datetime, timedelta
import threading
import winsound  # For Sound Alerts on Windows
import requests  # For Telegram API
from sklearn.ensemble import RandomForestClassifier  # REAL AI MODEL

# --- CONFIGURATION ---
CONFIG = {
    # ------------------------------------------------------------------
    # ⚠️ CRITICAL: ACCOUNT MUST ALLOW ALGO TRADING
    # ------------------------------------------------------------------
    "account": 261009880,
    "password": "Amine2002@",
    "server": "Exness-MT5Trial16",
    "symbol": "BTCUSDm",  # Ensure this matches your broker
    "timeframe": mt5.TIMEFRAME_M15,
    # --- NOTIFICATIONS ---
    "enable_sound": False,
    "telegram_token": "8423946950:AAF0Ja88p_52coyDsh48nQD1yNZW7NwAgck",
    "telegram_chat_id": "6476316022",
    # --- STRATEGY: REAL AI (MACHINE LEARNING) ---
    "lot_size": 0.5,
    "target_profit_money": 50.0,  # Cash Target: Close if profit > $40
    # --- AUTO CLOSE SETTINGS ---
    "close_on_reversal": False,
    # --- VOLATILITY SETTINGS ---
    "use_atr_stops": True,
    "atr_multiplier": 3.0,
    # --- TRAILING STOP STRATEGY (CORRECTED FOR BTC) ---
    # 1. Breakeven: If price moves 15000 points ($150) in favor
    "breakeven_trigger_points": 15000,
    # Lock in $20 (2,000 points) profit when triggered
    "breakeven_cushion_points": 2000,
    # Trail the price by $200 (20,000 points)
    "trailing_stop_points": 20000,
    # 3. Anti-Spam: Only move SL if the new level is at least 100 points different
    "min_sl_step_points": 100,
    # --- RISK MANAGEMENT ---
    "stop_loss_points": 10000,
    "take_profit_points": 20000,
    "max_spread_points": 5000,
    "max_drawdown_pct": 50.0,
    "magic_number": 234000,
    "sleep_interval": 0.5,
    "max_exposure": 10,
    "dry_run": False,  # Set to False to trade with real money
}

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("trading_bot.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# --- NOTIFICATION MODULE ---
class NotificationHandler:
    def __init__(self, config):
        self.config = config

    def send(self, message, type="INFO"):
        # --- 1. SOUND ALERT (Windows Only) ---
        if self.config.get("enable_sound", True):
            try:
                if type == "TRADE":
                    winsound.Beep(1000, 200)
                elif type == "CLOSE":
                    winsound.Beep(800, 400)
                elif type == "ERROR":
                    winsound.Beep(400, 1000)
            except Exception:
                pass

        # --- 2. TELEGRAM MESSAGE ---
        token = self.config.get("telegram_token")
        chat_id = self.config.get("telegram_chat_id")

        if token and chat_id:
            try:
                url = f"https://api.telegram.org/bot{token}/sendMessage"
                data = {"chat_id": chat_id, "text": message}

                response = requests.post(url, data=data, timeout=10)

                if response.status_code != 200:
                    logger.error(f"TELEGRAM ERROR: {response.status_code}")

            except requests.exceptions.ConnectionError:
                logger.error("TELEGRAM CONNECT FAIL: Check Internet or DNS.")
            except Exception as e:
                logger.error(f"TELEGRAM FAILED: {e}")


# --- REAL AI PREDICTION MODULE ---
class AIModel:
    def __init__(self):
        # Initialize a Random Forest Classifier
        self.model = RandomForestClassifier(
            n_estimators=100, min_samples_split=10, random_state=42
        )
        self.is_trained = False
        logger.info("Real AI Model (Random Forest) initialized.")

    def preprocess_data(self, df):
        # 1. Feature Engineering
        df = df.copy()

        # Technical Indicators
        df["SMA_50"] = df["close"].rolling(window=50).mean()
        df["SMA_20"] = df["close"].rolling(window=20).mean()

        # RSI
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df["RSI"] = 100 - (100 / (1 + rs))

        # ATR (Volatility)
        df["h-l"] = df["high"] - df["low"]
        df["h-pc"] = abs(df["high"] - df["close"].shift(1))
        df["l-pc"] = abs(df["low"] - df["close"].shift(1))
        df["tr"] = df[["h-l", "h-pc", "l-pc"]].max(axis=1)
        df["ATR"] = df["tr"].rolling(window=14).mean()

        # Derived Features
        df["Dist_SMA50"] = df["close"] - df["SMA_50"]
        df["Candle_Size"] = df["close"] - df["open"]

        # TARGET (What we want the AI to predict)
        # 1 if Next Candle Close > Current Close (Price went Up)
        df["Target"] = (df["close"].shift(-1) > df["close"]).astype(int)

        df.dropna(inplace=True)
        return df

    def train(self, df):
        """Trains the model on historical data."""
        logger.info(f"Training AI on {len(df)} historical candles...")

        # Features columns
        features = ["RSI", "Dist_SMA50", "Candle_Size", "ATR", "SMA_20"]
        X = df[features]
        y = df["Target"]

        # Train the model
        self.model.fit(X, y)
        self.is_trained = True
        logger.info("AI Training Complete. Model is ready to predict.")

    def predict(self, df):
        # We need enough data to train + predict
        if len(df) < 100:
            logger.warning("Not enough data to train AI.")
            return 0

        # Train on the first run
        if not self.is_trained:
            training_data = df.iloc[:-1]
            self.train(training_data)

        # Prepare current data for prediction
        last_candle = df.iloc[[-1]]
        features = ["RSI", "Dist_SMA50", "Candle_Size", "ATR", "SMA_20"]
        X_new = last_candle[features]

        # Ask AI for Probability
        probs = self.model.predict_proba(X_new)[0]
        prob_up = probs[1]

        current_price = last_candle["close"].values[0]
        logger.info(
            f"AI Prediction: Probability UP = {prob_up:.2f} (Price: {current_price})"
        )

        # --- AI DECISION LOGIC ---
        if prob_up > 0.60:  # 60% Confidence it goes UP
            logger.info(f"AI Signals BUY (Confidence: {prob_up:.2f})")
            return 1
        elif prob_up < 0.40:  # <40% Confidence UP (means >60% Down)
            logger.info(f"AI Signals SELL (Confidence: {1-prob_up:.2f})")
            return -1
        else:
            logger.info("AI is uncertain. No trade.")
            return 0


# --- MAIN TRADING BOT ---
class MT5TradingBot:
    def __init__(self, config):
        self.config = config
        self.ai_model = AIModel()
        self.notifier = NotificationHandler(config)
        self.running = False
        self.last_bar_time = None

        # Anti-Spam: Track failed close attempts
        self.failed_tickets = {}
        # Anti-Spam: Track simulated dry run closes
        self.dry_run_tickets = set()

    def connect(self):
        if not mt5.initialize():
            logger.error(f"MT5 Init Failed: {mt5.last_error()}")
            self.notifier.send("MT5 Init Failed!", type="ERROR")
            return False

        authorized = mt5.login(
            self.config["account"],
            password=self.config["password"],
            server=self.config["server"],
        )

        if authorized:
            account_info = mt5.account_info()
            logger.info(f"Connected. Balance: {account_info.balance}")

            symbol = self.config["symbol"]
            if not mt5.symbol_select(symbol, True):
                msg = f"ERROR: Failed to select symbol '{symbol}'."
                logger.error(msg)
                self.notifier.send(msg, type="ERROR")
                return False

            self.notifier.send(
                f"Real AI Bot ON. Balance: {account_info.balance}", type="INFO"
            )
        else:
            logger.error(f"Login Failed: {mt5.last_error()}")
            self.notifier.send("Login Failed!", type="ERROR")

        return authorized

    def check_trading_allowed(self):
        if not mt5.terminal_info().trade_allowed:
            msg = "CRITICAL: 'Algo Trading' button in MT5 is OFF!"
            logger.warning(msg)
            self.notifier.send(msg, type="ERROR")
            return False
        return True

    def check_safety_shields(self):
        account = mt5.account_info()
        if account is None:
            return True

        drawdown_threshold = account.balance * (
            1 - (self.config["max_drawdown_pct"] / 100)
        )
        if account.equity < drawdown_threshold:
            msg = f"STOP: Equity dropped below 50%! Emergency Halt."
            logger.critical(msg)
            self.notifier.send(msg, type="ERROR")
            self.panic_close_all()
            return False
        return True

    def panic_close_all(self):
        positions = mt5.positions_get()
        if positions:
            count = 0
            for pos in positions:
                if count > 5:
                    break
                success = self.close_position(pos, reason="EMERGENCY STOP")
                count += 1
                if not success:
                    break

        self.running = False

    def get_market_data(self, n_candles=1000):
        rates = mt5.copy_rates_from_pos(
            self.config["symbol"], self.config["timeframe"], 0, n_candles
        )
        if rates is None or len(rates) == 0:
            logger.warning(
                f"Data fetch failed for {self.config['symbol']}! Check symbol name."
            )
            return pd.DataFrame()

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        return df

    def execute_trade(self, signal, atr_value=0):
        if not self.running:
            return

        symbol = self.config["symbol"]
        lot = self.config["lot_size"]

        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            return

        # --- FIX: SAFE SYMBOL INFO FETCH (PREVENTS CRASH) ---
        s_info = mt5.symbol_info(symbol)
        if s_info is None:
            logger.error(f"Cannot get symbol info for {symbol}. Connection lost?")
            return

        spread_points = (tick.ask - tick.bid) / s_info.point
        if spread_points > self.config["max_spread_points"]:
            logger.warning(f"Spread {spread_points} too high. Waiting.")
            return

        price = tick.ask if signal == 1 else tick.bid
        action = mt5.TRADE_ACTION_DEAL
        type_order = mt5.ORDER_TYPE_BUY if signal == 1 else mt5.ORDER_TYPE_SELL

        point = s_info.point
        digits = s_info.digits

        if self.config["use_atr_stops"] and atr_value > 0:
            sl_distance = atr_value * self.config["atr_multiplier"]
            sl = price - sl_distance if signal == 1 else price + sl_distance
            tp_dist = self.config["take_profit_points"] * point
            tp = price + tp_dist if signal == 1 else price - tp_dist
        else:
            sl_dist = self.config["stop_loss_points"] * point
            tp_dist = self.config["take_profit_points"] * point
            sl = price - sl_dist if signal == 1 else price + sl_dist
            tp = price + tp_dist if signal == 1 else price - tp_dist

        sl = round(sl, digits)
        tp = round(tp, digits)

        if self.config.get("dry_run", False):
            side = "BUY" if signal == 1 else "SELL"
            logger.info(
                f"DRY RUN: Would have opened {side} @ {price} | SL: {sl} | TP: {tp}"
            )
            return

        request = {
            "action": action,
            "symbol": symbol,
            "volume": lot,
            "type": type_order,
            "price": price,
            "sl": 0.0,
            "tp": 0.0,
            "magic": self.config["magic_number"],
            "comment": "RealAI",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)

        if result is None:
            logger.error("Order Send failed. Result is None.")
            return

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            if result.retcode == mt5.TRADE_RETCODE_INVALID_FILL:
                request["type_filling"] = mt5.ORDER_FILLING_FOK
                mt5.order_send(request)
            elif result.retcode == 10026:
                msg = "CRITICAL: AutoTrading Blocked (10026). Check Account!"
                logger.critical(msg)
                self.notifier.send(msg, type="ERROR")
                self.running = False
        else:
            side = "BUY" if signal == 1 else "SELL"
            msg = f"OPENED {side} @ {price}"
            logger.info(msg)
            self.notifier.send(msg, type="TRADE")

            time.sleep(0.5)
            modify_request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "position": result.order,
                "symbol": symbol,
                "sl": sl,
                "tp": tp,
                "magic": self.config["magic_number"],
            }
            mt5.order_send(modify_request)

    def close_position(self, position, reason="Signal"):
        if not self.running:
            return False

        if self.config.get("dry_run", False):
            if position.ticket not in self.dry_run_tickets:
                logger.info(
                    f"DRY RUN: Would have closed position {position.ticket} ({reason})"
                )
                self.dry_run_tickets.add(position.ticket)
            return True

        # Anti-Spam: Don't retry failed closes endlessly
        if position.ticket in self.failed_tickets:
            if self.failed_tickets[position.ticket] >= 3:
                return False

        tick = mt5.symbol_info_tick(position.symbol)
        if tick is None:
            return False

        safe_comment = f"Close-{reason}"[:31]

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": position.symbol,
            "volume": position.volume,
            "type": (
                mt5.ORDER_TYPE_SELL
                if position.type == mt5.ORDER_TYPE_BUY
                else mt5.ORDER_TYPE_BUY
            ),
            "position": position.ticket,
            "price": tick.bid if position.type == mt5.ORDER_TYPE_BUY else tick.ask,
            "magic": self.config["magic_number"],
            "comment": safe_comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)

        if result is None:
            self.failed_tickets[position.ticket] = (
                self.failed_tickets.get(position.ticket, 0) + 1
            )
            return False

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            if result.retcode == mt5.TRADE_RETCODE_INVALID_FILL:
                request["type_filling"] = mt5.ORDER_FILLING_FOK
                result = mt5.order_send(request)

            if result.retcode != mt5.TRADE_RETCODE_DONE:
                logger.error(
                    f"FAILED to CLOSE {position.ticket}. Error: {result.comment} ({result.retcode})"
                )
                self.failed_tickets[position.ticket] = (
                    self.failed_tickets.get(position.ticket, 0) + 1
                )
                return False

        if position.ticket in self.failed_tickets:
            del self.failed_tickets[position.ticket]

        msg = f"CLOSED {position.ticket}. Profit: {position.profit}."
        logger.info(msg)
        self.notifier.send(msg, type="CLOSE")
        return True

    def modify_sl(self, position, new_sl):
        if self.config.get("dry_run", False):
            sl_key = f"{position.ticket}_sl"
            if sl_key not in self.dry_run_tickets:
                logger.info(
                    f"DRY RUN: Would have moved SL for {position.ticket} to {new_sl}"
                )
                self.dry_run_tickets.add(sl_key)
            return

        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": position.ticket,
            "symbol": position.symbol,
            "sl": new_sl,
            "tp": position.tp,
            "magic": self.config["magic_number"],
        }
        res = mt5.order_send(request)
        if res.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info(f"SL Moved for {position.ticket} -> {new_sl}")

    # --- IMPROVED TRAILING LOGIC ---
    def manage_positions(self, df=None):
        positions = mt5.positions_get(symbol=self.config["symbol"])
        if positions is None:
            return 0

        symbol_info = mt5.symbol_info(self.config["symbol"])
        if not symbol_info:
            return 0

        point = symbol_info.point

        # Thresholds converting points to price differences
        be_trigger = self.config["breakeven_trigger_points"] * point
        be_cushion = self.config["breakeven_cushion_points"] * point
        trail_dist = self.config["trailing_stop_points"] * point
        min_step = self.config["min_sl_step_points"] * point

        for pos in positions:
            if not self.running:
                break
            if pos.magic != self.config["magic_number"] and pos.magic != 0:
                continue

            # 1. HARD PROFIT TARGET (Cash based)
            if pos.profit >= self.config["target_profit_money"]:
                self.close_position(pos, reason="Profit Target Hit")
                continue

            # 2. CALCULATE CURRENT PRICE
            current_tick = mt5.symbol_info_tick(self.config["symbol"])
            if not current_tick:
                continue

            bid = current_tick.bid
            ask = current_tick.ask

            # --- BUY POSITION MANAGEMENT ---
            if pos.type == mt5.ORDER_TYPE_BUY:
                current_price = bid
                distance_moved = current_price - pos.price_open

                # A. Breakeven Logic (First Line of Defense)
                # If price moved X points up, move SL to Entry + Cushion
                if distance_moved > be_trigger:
                    new_sl = pos.price_open + be_cushion
                    # Only move if current SL is below the new BreakEven level
                    if pos.sl < new_sl:
                        self.modify_sl(pos, new_sl)
                        continue  # Done for this tick

                # B. Trailing Stop Logic (Profit Protection)
                # If we are well in profit, drag the SL up behind the price
                if distance_moved > be_trigger:
                    calculated_sl = current_price - trail_dist

                    # LOGIC: Only move SL UP, never down.
                    # And only move if the jump is bigger than 'min_step' (Anti-Spam)
                    if calculated_sl > (pos.sl + min_step):
                        self.modify_sl(pos, calculated_sl)

            # --- SELL POSITION MANAGEMENT ---
            elif pos.type == mt5.ORDER_TYPE_SELL:
                current_price = ask
                distance_moved = pos.price_open - current_price

                # A. Breakeven Logic
                if distance_moved > be_trigger:
                    new_sl = pos.price_open - be_cushion
                    # Only move if current SL is above the new BreakEven level (or 0)
                    if pos.sl == 0.0 or pos.sl > new_sl:
                        self.modify_sl(pos, new_sl)
                        continue

                # B. Trailing Stop Logic
                if distance_moved > be_trigger:
                    calculated_sl = current_price + trail_dist

                    # LOGIC: Only move SL DOWN, never up.
                    if pos.sl == 0.0 or calculated_sl < (pos.sl - min_step):
                        self.modify_sl(pos, calculated_sl)

        return len(positions)

    def run(self):
        self.running = True
        if not self.connect():
            return
        self.check_trading_allowed()

        logger.info("REAL AI Bot STARTED.")

        try:
            while self.running:
                if not self.check_safety_shields():
                    break

                if len(self.failed_tickets) > 50:
                    self.failed_tickets.clear()

                df = self.get_market_data(n_candles=1000)
                open_positions_count = self.manage_positions(df=df)

                if (
                    self.running
                    and open_positions_count < self.config["max_exposure"]
                    and not df.empty
                ):
                    current_bar_time = df.iloc[-1]["time"]
                    if self.last_bar_time != current_bar_time:
                        self.last_bar_time = current_bar_time

                        df_processed = self.ai_model.preprocess_data(df)
                        signal = self.ai_model.predict(df_processed)

                        last_atr = df_processed.iloc[-1]["ATR"]
                        if signal != 0:
                            self.execute_trade(signal, atr_value=last_atr)

                time.sleep(self.config["sleep_interval"])

        except KeyboardInterrupt:
            logger.info("Bot stopping...")
        finally:
            mt5.shutdown()
            logger.info("MT5 Closed.")


if __name__ == "__main__":
    bot = MT5TradingBot(CONFIG)
    bot.run()
