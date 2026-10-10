import MetaTrader5 as mt5
import time

print("Attempting to connect to MT5...")
if not mt5.initialize(timeout=30000):
    print("Initialize failed, error code:", mt5.last_error())
else:
    print("SUCCESS! Connected to MT5.")
    print("Terminal info:", mt5.terminal_info())
    mt5.shutdown()
