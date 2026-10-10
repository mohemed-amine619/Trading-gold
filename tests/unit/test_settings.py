import os
import pytest
from core.settings import Settings, TradingMode, LLMProvider, get_settings, reload_settings

def test_default_settings():
    os.environ.clear() # clear env for isolated testing
    settings = reload_settings()
    
    assert settings.trading_mode == TradingMode.PAPER
    assert settings.is_paper is True
    assert settings.is_live is False
    assert settings.symbols == ["XAUUSDm"]
    assert settings.llm_provider == LLMProvider.NONE
    assert settings.llm_enabled is False

def test_parse_symbols():
    os.environ["SYMBOLS"] = " EURUSD, GBPUSD , XAUUSD "
    settings = reload_settings()
    assert settings.symbols == ["EURUSD", "GBPUSD", "XAUUSD"]
    os.environ.clear()

def test_live_trading_validation():
    os.environ.clear()
    os.environ["TRADING_MODE"] = "live"
    os.environ["MT5_PASSWORD"] = "secret"
    os.environ["MT5_SERVER"] = "broker-server"
    os.environ["DASHBOARD_SECRET_KEY"] = "secure-key"
    
    settings = reload_settings()
    assert settings.is_live is True
    
    # Missing required live params
    os.environ.clear()
    os.environ["TRADING_MODE"] = "live"
    with pytest.raises(ValueError, match="MT5_PASSWORD is required for live trading"):
        reload_settings()
        
    os.environ["MT5_PASSWORD"] = "secret"
    with pytest.raises(ValueError, match="MT5_SERVER is required for live trading"):
        reload_settings()
        
    os.environ["MT5_SERVER"] = "broker-server"
    with pytest.raises(ValueError, match="DASHBOARD_SECRET_KEY must be changed before live trading"):
        reload_settings()
    
    os.environ.clear()

def test_as_legacy_config():
    os.environ.clear()
    settings = reload_settings()
    legacy = settings.as_legacy_config()
    assert legacy["symbol"] == "XAUUSDm"
    assert legacy["dry_run"] is True
