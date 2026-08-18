"""
gui_main.py - PyQt5 dashboard for the trading bot. Builds to a Windows .exe
via the PyInstaller spec in trading_bot.spec (see build_exe.bat).

Architecture
------------
The bot's asyncio loop runs inside BotThread (a QThread). The GUI never talks
to MT5 directly:
  * bot -> GUI : callbacks (account/positions snapshot, signal feed) bridged
                 to Qt signals (queued across threads automatically)
  * GUI -> bot : coroutines enqueued with asyncio.run_coroutine_threadsafe()
                 (stop, close symbol, close all)

Tabs
----
  Dashboard - account stats, start/stop, dry-run toggle, live risk
              parameters, symbol selection
  Positions - live table of open positions + close / close-all
  Signals   - recent strategy signal feed
  Logs      - rotating console fed by the python logging module

Safety notes
------------
  * config.DRY_RUN (tickable on the Dashboard) simulates all fills with zero
    risk - leave it ON until you have watched several signals cycle through.
  * Risk parameters edited on the Dashboard take effect on the NEXT decision
    (they are re-read every poll cycle). Symbol changes require a bot restart.
"""
import asyncio
import logging
import sys

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QApplication, QCheckBox, QDoubleSpinBox, QGridLayout, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMainWindow,
    QMessageBox, QPlainTextEdit, QPushButton, QSpinBox, QTabWidget,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

import config
from core.bot import Callbacks, TradingBot
from core.credentials_store import (clear_credentials, load_credentials,
                                    save_credentials)

logger = logging.getLogger(__name__)

MONO_FONT = "Consolas, Menlo, DejaVu Sans Mono"


class QtLogHandler(logging.Handler):
    """Routing log records from the worker thread to the GUI console."""

    def __init__(self, owner):
        super().__init__()
        self.owner = owner

    def emit(self, record):
        try:
            self.owner.log_emitted.emit(self.format(record))
        except Exception:
            pass  # an alert that fails must never kill the bot


class BotThread(QThread):
    """Runs the asyncio TradingBot loop in a background thread."""

    status_changed = pyqtSignal(str)
    snapshot_ready = pyqtSignal(dict)
    signal_fired = pyqtSignal(str)
    log_emitted = pyqtSignal(str)
    bot_stopped = pyqtSignal(bool)   # True = clean stop

    def __init__(self, cfg, parent=None):
        super().__init__(parent)
        self.cfg = cfg
        self._loop = None
        self.bot = None
        self.error_message = ""

    # ------------------------------------------------------------------
    # QThread body (worker thread)
    # ------------------------------------------------------------------
    def run(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        # Route python logging into the GUI console. The handler is created
        # HERE so its emit() runs in this thread (thread-safe Qt emission).
        handler = QtLogHandler(self)
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-8s %(message)s"))
        logging.getLogger().addHandler(handler)

        callbacks = Callbacks(on_snapshot=self._on_snapshot,
                              on_signal=self._on_signal)
        self.bot = TradingBot(self.cfg, callbacks=callbacks)
        ok = False
        try:
            self.status_changed.emit("Starting MT5 terminal...")
            self._loop.run_until_complete(self.bot.run_until_stopped())
            ok = True
        except Exception as exc:
            self.error_message = repr(exc)
            logger.exception("Bot thread crashed")
            self.status_changed.emit(f"CRASHED: {exc}")
        finally:
            logging.getLogger().removeHandler(handler)
            try:
                self._loop.run_until_complete(self.bot.engine.disconnect())
            except Exception:
                pass
            self._loop.close()
            self.bot_stopped.emit(ok)

    # ------------------------------------------------------------------
    # bot -> GUI callbacks (called from this worker thread)
    # ------------------------------------------------------------------
    def _on_snapshot(self, account, positions):
        self.snapshot_ready.emit({"account": account, "positions": positions})

    def _on_signal(self, signal):
        text = (f"{signal['time']}  {signal['symbol']:<8} "
                f"{'BUY ' if signal['direction'] == 1 else 'SELL '}  "
                f"close={signal['close']:.5f}  {signal['reason']}")
        self.signal_fired.emit(text)

    # ------------------------------------------------------------------
    # GUI -> bot (called from the main thread)
    # ------------------------------------------------------------------
    def stop_bot(self):
        if self._loop and self.bot:
            asyncio.run_coroutine_threadsafe(self.bot.request_stop(), self._loop)

    def close_symbol(self, symbol):
        if self._loop and self.bot:
            asyncio.run_coroutine_threadsafe(self.bot.close_symbol(symbol), self._loop)

    def close_all(self):
        if self._loop and self.bot:
            asyncio.run_coroutine_threadsafe(self.bot.close_all(), self._loop)


class MainWindow(QMainWindow):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.thread = None
        self._build_ui()
        self._apply_risk_to_ui()
        self._load_credentials()
        self._set_status("Idle - configure and press Start")
        self._update_controls()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self):
        self.setWindowTitle("Gold Trading Bot")
        self.resize(980, 640)

        tabs = QTabWidget()

        # ---- Dashboard tab -------------------------------------------------
        dash = QWidget()
        layout = QVBoxLayout(dash)

        control_row = QHBoxLayout()
        self.btn_start = QPushButton("Start Bot")
        self.btn_start.clicked.connect(self._start_bot)
        self.btn_stop = QPushButton("Stop Bot")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._stop_bot)
        self.chk_dry_run = QCheckBox("Dry-run (paper trading)")
        self.chk_dry_run.setChecked(self.cfg.DRY_RUN)
        self.chk_dry_run.stateChanged.connect(
            lambda state: setattr(self.cfg, "DRY_RUN", bool(state)))
        control_row.addWidget(self.btn_start)
        control_row.addWidget(self.btn_stop)
        control_row.addStretch(1)
        control_row.addWidget(self.chk_dry_run)
        layout.addLayout(control_row)

        # MT5 terminal credentials (stored in credentials.json next to the app)
        conn_box = QGroupBox("MT5 connection (saved locally, applied on Start)")
        conn_grid = QGridLayout(conn_box)
        self.edit_path = QLineEdit()
        self.edit_path.setPlaceholderText("auto-detect (empty)")
        self.edit_login = QLineEdit()
        self.edit_login.setPlaceholderText("e.g. 261009880")
        self.edit_password = QLineEdit()
        self.edit_password.setEchoMode(QLineEdit.Password)
        self.chk_show_pw = QCheckBox("Show")
        self.chk_show_pw.toggled.connect(
            lambda on: self.edit_password.setEchoMode(
                QLineEdit.Normal if on else QLineEdit.Password))
        self.edit_server = QLineEdit()
        self.edit_server.setPlaceholderText("e.g. Exness-MT5Trial16")
        btn_save_creds = QPushButton("Save credentials")
        btn_save_creds.clicked.connect(self._save_credentials)
        btn_clear_creds = QPushButton("Clear")
        btn_clear_creds.clicked.connect(self._clear_credentials)
        conn_grid.addWidget(QLabel("Terminal path:"), 0, 0)
        conn_grid.addWidget(self.edit_path, 0, 1, 1, 3)
        conn_grid.addWidget(QLabel("Login:"), 1, 0)
        conn_grid.addWidget(self.edit_login, 1, 1)
        conn_grid.addWidget(QLabel("Password:"), 1, 2)
        conn_grid.addWidget(self.edit_password, 1, 3)
        conn_grid.addWidget(self.chk_show_pw, 1, 4)
        conn_grid.addWidget(QLabel("Server:"), 2, 0)
        conn_grid.addWidget(self.edit_server, 2, 1, 1, 3)
        conn_grid.addWidget(btn_save_creds, 3, 1)
        conn_grid.addWidget(btn_clear_creds, 3, 2)
        conn_grid.setColumnStretch(3, 1)
        layout.addWidget(conn_box)

        # account stats grid
        acc_box = QGroupBox("Account")
        acc_grid = QGridLayout(acc_box)
        self.lbl_server = QLabel("-"); self.lbl_login = QLabel("-")
        self.lbl_balance = QLabel("-"); self.lbl_equity = QLabel("-")
        self.lbl_margin = QLabel("-"); self.lbl_free = QLabel("-")
        self.lbl_profit = QLabel("-"); self.lbl_currency = QLabel("-")
        acc_grid.addWidget(QLabel("Server:"), 0, 0)
        acc_grid.addWidget(self.lbl_server, 0, 1)
        acc_grid.addWidget(QLabel("Login:"), 0, 2)
        acc_grid.addWidget(self.lbl_login, 0, 3)
        acc_grid.addWidget(QLabel("Currency:"), 0, 4)
        acc_grid.addWidget(self.lbl_currency, 0, 5)
        acc_grid.addWidget(QLabel("Balance:"), 1, 0)
        acc_grid.addWidget(self.lbl_balance, 1, 1)
        acc_grid.addWidget(QLabel("Equity:"), 1, 2)
        acc_grid.addWidget(self.lbl_equity, 1, 3)
        acc_grid.addWidget(QLabel("Margin:"), 1, 4)
        acc_grid.addWidget(self.lbl_margin, 1, 5)
        acc_grid.addWidget(QLabel("Free margin:"), 2, 0)
        acc_grid.addWidget(self.lbl_free, 2, 1)
        acc_grid.addWidget(QLabel("Open P/L:"), 2, 2)
        acc_grid.addWidget(self.lbl_profit, 2, 3)
        acc_grid.setColumnStretch(6, 1)
        layout.addWidget(acc_box)

        # risk parameters + symbols
        mid_row = QHBoxLayout()

        risk_box = QGroupBox("Risk parameters (applied live)")
        risk_grid = QGridLayout(risk_box)
        self.spin_risk = QDoubleSpinBox()
        self.spin_risk.setRange(0.1, 10.0); self.spin_risk.setDecimals(1)
        self.spin_risk.setSuffix(" %")
        self.spin_max_pos = QSpinBox(); self.spin_max_pos.setRange(1, 20)
        self.spin_deviation = QSpinBox(); self.spin_deviation.setRange(0, 500)
        self.spin_trail = QDoubleSpinBox()
        self.spin_trail.setRange(1.0, 10.0); self.spin_trail.setDecimals(1)
        risk_grid.addWidget(QLabel("Risk per trade:"), 0, 0)
        risk_grid.addWidget(self.spin_risk, 0, 1)
        risk_grid.addWidget(QLabel("Max positions:"), 1, 0)
        risk_grid.addWidget(self.spin_max_pos, 1, 1)
        risk_grid.addWidget(QLabel("Slippage (points):"), 2, 0)
        risk_grid.addWidget(self.spin_deviation, 2, 1)
        risk_grid.addWidget(QLabel("Trail ATR mult:"), 3, 0)
        risk_grid.addWidget(self.spin_trail, 3, 1)
        btn_apply = QPushButton("Apply")
        btn_apply.clicked.connect(self._apply_risk)
        risk_grid.addWidget(btn_apply, 4, 0, 1, 2)
        mid_row.addWidget(risk_box)

        sym_box = QGroupBox("Symbols (effective on next Start)")
        sym_v = QVBoxLayout(sym_box)
        self.symbol_list = QListWidget()
        for symbol in self.cfg.SYMBOLS:
            item = QListWidgetItem(symbol)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            self.symbol_list.addItem(item)
        sym_v.addWidget(self.symbol_list)
        mid_row.addWidget(sym_box)

        strat_box = QGroupBox("Strategy")
        strat_v = QVBoxLayout(strat_box)
        strat_v.addWidget(QLabel("EMA 50/200 crossover + RSI filter"))
        strat_v.addWidget(QLabel(f"Entry TF: {self.cfg.ENTRY_TIMEFRAME}"))
        strat_v.addWidget(QLabel(f"Filter TF: {self.cfg.FILTER_TIMEFRAME}"))
        strat_v.addWidget(QLabel("Signals on closed candles only"))
        strat_v.addWidget(QLabel("ATR trailing stop + margin checks"))
        strat_v.addStretch(1)
        mid_row.addWidget(strat_box)

        layout.addLayout(mid_row)
        layout.addStretch(1)

        # ---- Positions tab --------------------------------------------------
        pos_tab = QWidget()
        pos_v = QVBoxLayout(pos_tab)
        self.pos_table = QTableWidget(0, 8)
        self.pos_table.setHorizontalHeaderLabels(
            ["Ticket", "Symbol", "Dir", "Volume", "Entry", "SL", "TP", "P/L"])
        self.pos_table.horizontalHeader().setStretchLastSection(True)
        pos_v.addWidget(self.pos_table)
        pos_buttons = QHBoxLayout()
        btn_close_sel = QPushButton("Close selected")
        btn_close_sel.clicked.connect(self._close_selected)
        btn_close_all = QPushButton("Close ALL positions")
        btn_close_all.setStyleSheet("background-color:#c0392b; color:white;")
        btn_close_all.clicked.connect(self._close_all)
        pos_buttons.addWidget(btn_close_sel)
        pos_buttons.addStretch(1)
        pos_buttons.addWidget(btn_close_all)
        pos_v.addLayout(pos_buttons)

        # ---- Signals tab -----------------------------------------------------
        sig_tab = QWidget()
        sig_v = QVBoxLayout(sig_tab)
        self.signal_list = QListWidget()
        self.signal_list.setFont(QFont(MONO_FONT, 9))
        sig_v.addWidget(self.signal_list)

        # ---- Logs tab --------------------------------------------------------
        log_tab = QWidget()
        log_v = QVBoxLayout(log_tab)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(5000)
        self.log_view.setFont(QFont(MONO_FONT, 9))
        log_v.addWidget(self.log_view)
        log_buttons = QHBoxLayout()
        btn_clear = QPushButton("Clear log")
        btn_clear.clicked.connect(self.log_view.clear)
        log_buttons.addStretch(1)
        log_buttons.addWidget(btn_clear)
        log_v.addLayout(log_buttons)

        tabs.addTab(dash, "Dashboard")
        tabs.addTab(pos_tab, "Positions")
        tabs.addTab(sig_tab, "Signals")
        tabs.addTab(log_tab, "Logs")
        self.setCentralWidget(tabs)

        self.statusBar().showMessage("Idle")
        self._status_label = QLabel()
        self.statusBar().addPermanentWidget(self._status_label)

    # ------------------------------------------------------------------
    # dashboard helpers
    # ------------------------------------------------------------------
    def _apply_risk_to_ui(self):
        self.spin_risk.setValue(self.cfg.RISK_PER_TRADE * 100.0)
        self.spin_max_pos.setValue(self.cfg.MAX_OPEN_POSITIONS)
        self.spin_deviation.setValue(self.cfg.DEVIATION_POINTS)
        self.spin_trail.setValue(self.cfg.TRAIL_ATR_MULT)

    def _apply_risk(self):
        self.cfg.RISK_PER_TRADE = self.spin_risk.value() / 100.0
        self.cfg.MAX_OPEN_POSITIONS = self.spin_max_pos.value()
        self.cfg.DEVIATION_POINTS = self.spin_deviation.value()
        self.cfg.TRAIL_ATR_MULT = self.spin_trail.value()
        self._set_status("Risk parameters applied (live)")

    def _selected_symbols(self):
        return [self.symbol_list.item(i).text()
                for i in range(self.symbol_list.count())
                if self.symbol_list.item(i).checkState() == Qt.Checked]

    # ------------------------------------------------------------------
    # credentials (persisted in credentials.json next to the app)
    # ------------------------------------------------------------------
    def _load_credentials(self):
        """Pre-fill the connection fields: config.py defaults, overridden by
        any saved credentials.json."""
        saved = load_credentials()
        self.edit_path.setText(
            saved.get("mt5_path") or self.cfg.MT5_PATH or "")
        self.edit_login.setText(
            saved.get("login") or (str(self.cfg.MT5_LOGIN) if self.cfg.MT5_LOGIN else ""))
        self.edit_password.setText(
            saved.get("password") or self.cfg.MT5_PASSWORD or "")
        self.edit_server.setText(
            saved.get("server") or self.cfg.MT5_SERVER or "")

    def _apply_credentials(self):
        """Push the fields into config so MT5Engine.connect() uses them."""
        self.cfg.MT5_PATH = self.edit_path.text().strip() or None
        login_text = self.edit_login.text().strip()
        self.cfg.MT5_LOGIN = int(login_text) if login_text.isdigit() else None
        self.cfg.MT5_PASSWORD = self.edit_password.text() or None
        self.cfg.MT5_SERVER = self.edit_server.text().strip() or None

    def _save_credentials(self):
        self._apply_credentials()
        payload = {
            "mt5_path": self.cfg.MT5_PATH or "",
            "login": str(self.cfg.MT5_LOGIN) if self.cfg.MT5_LOGIN else "",
            "password": self.cfg.MT5_PASSWORD or "",
            "server": self.cfg.MT5_SERVER or "",
        }
        if save_credentials(payload):
            self._set_status("Credentials saved to credentials.json (plain text - keep it safe)")
        else:
            self._set_status("Failed to save credentials")

    def _clear_credentials(self):
        self.edit_path.clear(); self.edit_login.clear()
        self.edit_password.clear(); self.edit_server.clear()
        clear_credentials()
        self._set_status("Saved credentials cleared")

    # ------------------------------------------------------------------
    # start / stop
    # ------------------------------------------------------------------
    def _start_bot(self):
        symbols = self._selected_symbols()
        if not symbols:
            QMessageBox.warning(self, "No symbols",
                                "Check at least one symbol on the Dashboard.")
            return
        if self.thread is not None and self.thread.isRunning():
            QMessageBox.information(self, "Already running",
                                    "Stop the bot first, then start again.")
            return
        self._apply_credentials()
        self.cfg.SYMBOLS = symbols
        self.thread = BotThread(self.cfg)
        self.thread.status_changed.connect(self._set_status)
        self.thread.snapshot_ready.connect(self._on_snapshot)
        self.thread.signal_fired.connect(self.signal_list.addItem)
        self.thread.log_emitted.connect(self.log_view.appendPlainText)
        self.thread.bot_stopped.connect(self._on_bot_stopped)
        self.thread.start()
        self._update_controls()

    def _stop_bot(self):
        if self.thread and self.thread.isRunning():
            self._set_status("Stopping (waits for current cycle)...")
            self.thread.stop_bot()
        self._update_controls()

    def _on_bot_stopped(self, clean: bool):
        self._set_status("Stopped cleanly" if clean
                         else f"Bot thread exited with error: {self.thread.error_message}")
        self._update_controls()
        if not clean:
            QMessageBox.critical(self, "Bot error", self.thread.error_message)

    def _update_controls(self):
        running = self.thread is not None and self.thread.isRunning()
        self.btn_start.setEnabled(not running)
        self.btn_stop.setEnabled(running)

    # ------------------------------------------------------------------
    # telemetry
    # ------------------------------------------------------------------
    def _on_snapshot(self, data):
        acc = data.get("account")
        if acc:
            self.lbl_server.setText(str(acc.get("server", "-")))
            self.lbl_login.setText(str(acc.get("login", "-")))
            self.lbl_currency.setText(str(acc.get("currency", "-")))
            self.lbl_balance.setText(f"{acc.get('balance', 0):,.2f}")
            self.lbl_equity.setText(f"{acc.get('equity', 0):,.2f}")
            self.lbl_margin.setText(f"{acc.get('margin', 0):,.2f}")
            self.lbl_free.setText(f"{acc.get('margin_free', 0):,.2f}")
            self.lbl_profit.setText(f"{acc.get('profit', 0):,.2f}")

        positions = data.get("positions", [])
        self.pos_table.setRowCount(len(positions))
        for row, p in enumerate(positions):
            self.pos_table.setItem(row, 0, QTableWidgetItem(str(p["ticket"])))
            self.pos_table.setItem(row, 1, QTableWidgetItem(p["symbol"]))
            self.pos_table.setItem(row, 2, QTableWidgetItem(
                "BUY" if p["direction"] == 1 else "SELL"))
            self.pos_table.setItem(row, 3, QTableWidgetItem(f"{p['volume']:.2f}"))
            self.pos_table.setItem(row, 4, QTableWidgetItem(f"{p['entry']:.5f}"))
            self.pos_table.setItem(row, 5, QTableWidgetItem(
                f"{p['sl']:.5f}" if p["sl"] else "-"))
            self.pos_table.setItem(row, 6, QTableWidgetItem(
                f"{p['tp']:.5f}" if p["tp"] else "-"))
            profit_item = QTableWidgetItem(f"{p['profit']:,.2f}")
            profit_item.setForeground(Qt.darkRed if p["profit"] < 0 else Qt.darkGreen)
            self.pos_table.setItem(row, 7, profit_item)

    # ------------------------------------------------------------------
    # manual position management
    # ------------------------------------------------------------------
    def _close_selected(self):
        row = self.pos_table.currentRow()
        if row < 0:
            QMessageBox.information(self, "Nothing selected",
                                    "Select a position row first.")
            return
        symbol = self.pos_table.item(row, 1).text()
        if self.thread and self.thread.isRunning():
            self.thread.close_symbol(symbol)

    def _close_all(self):
        if not (self.thread and self.thread.isRunning()):
            return
        answer = QMessageBox.question(
            self, "Close all",
            "Close ALL open positions with market orders?",
            QMessageBox.Yes | QMessageBox.No)
        if answer == QMessageBox.Yes:
            self.thread.close_all()

    # ------------------------------------------------------------------
    # misc
    # ------------------------------------------------------------------
    def _set_status(self, message: str):
        self._status_label.setText(message)

    def closeEvent(self, event):
        if self.thread is not None and self.thread.isRunning():
            answer = QMessageBox.question(
                self, "Exit",
                "The bot is running. Stop it and exit?",
                QMessageBox.Yes | QMessageBox.No)
            if answer != QMessageBox.Yes:
                event.ignore()
                return
            self.thread.stop_bot()
            self.thread.wait(8000)
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow(config)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
