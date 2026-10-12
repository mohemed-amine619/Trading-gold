@echo off
    title Gold Trading Dashboard
    echo 🏆 Installing dependencies...
    pip install dash dash-bootstrap-components plotly pandas numpy --quiet
    
    echo.
    echo 🚀 Starting Gold Trading Dashboard...
    echo 📌 Open http://127.0.0.1:8050
    echo.
    python position_dashboard.py
    pause
