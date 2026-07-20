import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
import logging
from datetime import datetime
import winsound
import requests
from sklearn.ensemble import RandomForestClassifier

# --- CONFIGURATION (GOLD EDITION) ---
CONFIG = {
    "account": 261009880,
    "password": "Amine2002@",
    "server": "Exness-MT5Trial16",
    "symbol": "XAUUSDm",
    "timeframe": mt5.TIMEFRAME_M15,
    "enable_sound": True,
    "telegram_token": "8423946950:AAF0Ja88p_52coyDsh48nQD1yNZW7NwAgck",
    "telegram_chat_id": "6476316022",
    "lot_size": 0.01,
    "daily_profit_target": 50.0,
    "close_on_reversal": False,
    "use_atr_stops": True,
    "atr_multiplier": 2.0,
    "breakeven_trigger_points": 3000,
    "breakeven_cushion_points": 500,
    "trailing_stop_points": 2000,
    "min_sl_step_points": 200,
    "stop_loss_points": 5000,
    "take_profit_points": 15000,
    "max_spread_points": 800,
    "max_drawdown_pct": 20.0,
    "magic_number": 888999,
    "sleep_interval": 0.5,
    "max_exposure": 1,
    "dry_run": False,
}

# --- LOGGING ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("trading_bot.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# --- NOTIFICATIONS ---
class NotificationHandler:
    def __init__(self, config):
        self.config = config

    def send(self, message, type="INFO"):
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

        token = self.config.get("telegram_token")
        chat_id = self.config.get("telegram_chat_id")
        if token and chat_id:
            try:
                url = f"https://api.telegram.org/bot{token}/sendMessage"
                data = {"chat_id": chat_id, "text": message}
                response = requests.post(url, data=data, timeout=10)
                if response.status_code != 200:
                    logger.error(f"TELEGRAM ERROR: {response.status_code}")
            except Exception as e:
                logger.error(f"TELEGRAM FAILED: {e}")


# --- AI MODEL ---
class AIModel:
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100, min_samples_split=10, random_state=42
        )
        self.is_trained = False
        logger.info("Real AI Model (Random Forest) initialized.")

    def preprocess_data(self, df):
        df = df.copy()
        df["SMA_50"] = df["close"].rolling(50).mean()
        df["SMA_20"] = df["close"].rolling(20).mean()
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df["RSI"] = 100 - (100 / (1 + rs))
        df["h-l"] = df["high"] - df["low"]
        df["h-pc"] = abs(df["high"] - df["close"].shift(1))
        df["l-pc"] = abs(df["low"] - df["close"].shift(1))
        df["tr"] = df[["h-l", "h-pc", "l-pc"]].max(axis=1)
        df["ATR"] = df["tr"].rolling(14).mean()
        df["Dist_SMA50"] = df["close"] - df["SMA_50"]
        df["Candle_Size"] = df["close"] - df["open"]
        df["Target"] = (df["close"].shift(-1) > df["close"]).astype(int)
        df.dropna(inplace=True)
        return df

    def train(self, df):
        features = ["RSI", "Dist_SMA50", "Candle_Size", "ATR", "SMA_20"]
        X = df[features]
        y = df["Target"]
        self.model.fit(X, y)
        self.is_trained = True
        logger.info("AI Training Complete.")

    def predict(self, df):
        if len(df) < 100:
            return 0
        if not self.is_trained:
            self.train(df.iloc[:-1])
        last_candle = df.iloc[[-1]]
        features = ["RSI", "Dist_SMA50", "Candle_Size", "ATR", "SMA_20"]
        prob_up = self.model.predict_proba(last_candle[features])[0][1]
        current_price = last_candle["close"].values[0]
        logger.info(
            f"AI Prediction: Probability UP = {prob_up:.2f} (Price: {current_price})"
        )
        if prob_up > 0.6:
            return 1
        elif prob_up < 0.4:
            return -1
        return 0


# --- MT5 TRADING BOT ---
class MT5TradingBot:
    def __init__(self, config):
        self.config = config
        self.ai_model = AIModel()
        self.notifier = NotificationHandler(config)
        self.running = False
        self.last_bar_time = None
        self.start_balance = 0

    def connect(self):
        # Use the exact path where your MT5 is actually installed
        mt5_path = r"C:\Users\mohamed.bougrioua\Trading\terminal64.exe" 
        
        logger.info(f"Attempting to initialize MT5 from {mt5_path} (this may take up to 60 seconds)...")
        # Pass the specific path so Python doesn't get lost
        # timeout is in milliseconds, default is 60000 (60s), increasing to 120000 (120s)
        if not mt5.initialize(path=mt5_path, timeout=120000):
            logger.error(f"MT5 Init Failed: {mt5.last_error()}")
            return False
            
        logger.info("MT5 initialized successfully. Attempting to log in...")
        authorized = mt5.login(
            self.config["account"],
            password=self.config["password"],
            server=self.config["server"],
        )
        
        if authorized:
            logger.info("MT5 login successful.")
        else:
            logger.error(f"MT5 Login Failed: {mt5.last_error()}")
            
        return authorized

    def get_market_data(self, n_candles=500):
        rates = mt5.copy_rates_from_pos(
            self.config["symbol"], self.config["timeframe"], 0, n_candles
        )
        if rates is None:
            return pd.DataFrame()
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        return df

    def dynamic_lot(self):
        lot = self.config.get("lot_size", 0.01)
        return lot

    def execute_trade(self, signal):
        symbol = self.config["symbol"]
        volume = self.dynamic_lot()
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return
        price = tick.ask if signal == 1 else tick.bid
        order_type = mt5.ORDER_TYPE_BUY if signal == 1 else mt5.ORDER_TYPE_SELL
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": order_type,
            "price": price,
            "magic": self.config["magic_number"],
            "comment": "AI Trade",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        retries = 0
        while retries < 5:
            result = mt5.order_send(request)
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(
                    f"Trade executed: {'BUY' if signal==1 else 'SELL'} @ {price}"
                )
                self.notifier.send(
                    f"Trade executed: {'BUY' if signal==1 else 'SELL'} @ {price}",
                    "TRADE",
                )
                return
            elif result and result.retcode in [10019, 10018]:
                retries += 1
                time.sleep(0.5)
            else:
                logger.error(
                    f"Trade failed: Retcode {result.retcode if result else 'N/A'}"
                )
                return

    def manage_positions(self):
        positions = mt5.positions_get(symbol=self.config["symbol"])
        if not positions:
            return 0
        daily_profit = sum([pos.profit for pos in positions])
        if daily_profit >= self.config["daily_profit_target"]:
            self.running = False
            self.close_all_positions(
                reason=f"Daily Profit ${self.config['daily_profit_target']}"
            )
            return 0
        for pos in positions:
            if pos.profit >= self.config.get("take_profit_points", 20000) / 1000:
                self.close_position(pos, reason="TP hit")
        return len(positions)

    def close_position(self, position, reason="Signal"):
        tick = mt5.symbol_info_tick(position.symbol)
        if tick is None:
            return False
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
            "comment": f"Close-{reason}"[:31],
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info(f"CLOSED {position.ticket}. Profit: {position.profit}")
            self.notifier.send(
                f"CLOSED {position.ticket}. Profit: {position.profit}", "CLOSE"
            )
            return True
        logger.error(
            f"Failed to close position {position.ticket} | Retcode: {result.retcode if result else 'N/A'}"
        )
        return False

    def close_all_positions(self, reason="Emergency"):
        positions = mt5.positions_get(symbol=self.config["symbol"])
        if not positions:
            return
        for pos in positions:
            self.close_position(pos, reason)

    def run(self):
        self.running = True
        if not self.connect():
            return
        logger.info("REAL AI Bot STARTED.")
        try:
            while self.running:
                df = self.get_market_data(500)
                if df.empty:
                    time.sleep(self.config["sleep_interval"])
                    continue
                if self.manage_positions() < self.config["max_exposure"]:
                    current_bar_time = df.iloc[-1]["time"]
                    if self.last_bar_time != current_bar_time:
                        self.last_bar_time = current_bar_time
                        df_processed = self.ai_model.preprocess_data(df)
                        signal = self.ai_model.predict(df_processed)
                        if signal != 0:
                            self.execute_trade(signal)
                time.sleep(self.config["sleep_interval"])
        except KeyboardInterrupt:
            logger.info("Bot stopping...")
        finally:
            self.close_all_positions(reason="Shutdown")
            mt5.shutdown()
            logger.info("MT5 Closed.")


if __name__ == "__main__":
    bot = MT5TradingBot(CONFIG)
    bot.run()