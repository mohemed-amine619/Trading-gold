@echo off
echo ==========================================
echo FIXING METATRADER 5 IPC TIMEOUTS
echo ==========================================
echo.
echo 1. Killing all hidden MetaTrader 5 processes...
taskkill /F /IM terminal64.exe /T >nul 2>&1
echo Done.
echo.
echo 2. Please open MetaTrader 5 manually now.
echo Wait until you see your charts and are logged in.
pause
echo.
echo 3. Testing connection...
python -c "import MetaTrader5 as mt5; print('Connecting...'); print('SUCCESS!' if mt5.initialize(timeout=30000) else f'FAILED: {mt5.last_error()}')"
echo.
echo If SUCCESS, you can now run your bot: py run.py
pause
