"""
gui.py – Full control panel for the XAUUSDm AI trading bot (run.py).

Features:
  • Tabbed configuration editor covering every CONFIG key from run.py
  • Save / Load configuration as JSON (config.json by default)
  • Start / Stop the bot (runs StandaloneAIBot in a background thread)
  • Live status panel: connection, balance, equity, open positions, last signal
  • Live log viewer (streams the bot's own logger output in real time)
  • Emergency "Close all positions" button

Run with:  python gui.py
Must be in the SAME FOLDER as run.py (it imports StandaloneAIBot / CONFIG /
logger from it). Requires the same dependencies as run.py (MetaTrader5,
pandas, pandas_ta, scikit-learn, requests) plus the standard library's
tkinter (ships with Python on Windows).
"""

import copy
import json
import logging
import multiprocessing
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    mt5 = None
    MT5_AVAILABLE = False

# run.py (StandaloneAIBot, CONFIG, logger) is imported LAZILY in a background
# thread after the window is already on screen. It pulls in pandas / sklearn /
# pandas_ta / MetaTrader5, which are the slow part of startup — importing them
# eagerly at module load would delay the very first frame the user sees.
botmod = None

DEFAULT_JSON_PATH = Path(__file__).with_name("config.json")

# ---------------------------------------------------------------------------
# Timeframe mapping (name shown in GUI <-> raw MT5 constant stored in JSON)
# ---------------------------------------------------------------------------
def _tf_map() -> dict:
    if MT5_AVAILABLE:
        return {
            "M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15, "M30": mt5.TIMEFRAME_M30,
            "H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1,
        }
    # Fallback raw constants so the editor still works without MT5 installed
    return {"M1": 1, "M5": 5, "M15": 15, "M30": 30,
            "H1": 16385, "H4": 16388, "D1": 16408}


TF_MAP = _tf_map()
TF_MAP_REV = {v: k for k, v in TF_MAP.items()}
TF_OPTIONS = list(TF_MAP.keys())

# ---------------------------------------------------------------------------
# Field schema: (config_key, label, kind, section)
# kind in: str, password, int, float, bool, timeframe
# ---------------------------------------------------------------------------
FIELDS = [
    # --- Connexion ---
    ("account",  "Numero de compte",       "int",      "Connexion"),
    ("password", "Mot de passe",           "password", "Connexion"),
    ("server",   "Serveur",                "str",      "Connexion"),
    ("mt5_path", "Chemin terminal64.exe",  "str",      "Connexion"),

    # --- Instrument & session ---
    ("symbol",             "Symbole",                  "str",       "Instrument"),
    ("timeframe",          "Timeframe entree",         "timeframe", "Instrument"),
    ("h1_tf",              "Timeframe tendance H1",    "timeframe", "Instrument"),
    ("h4_tf",              "Timeframe tendance H4",    "timeframe", "Instrument"),
    ("n_candles",          "Nb bougies chargees",      "int",       "Instrument"),
    ("use_session_filter", "Filtrer par session",      "bool",      "Instrument"),
    ("london_start_utc",   "Londres debut (UTC)",      "int",       "Instrument"),
    ("london_end_utc",     "Londres fin (UTC)",        "int",       "Instrument"),
    ("ny_start_utc",       "New York debut (UTC)",     "int",       "Instrument"),
    ("ny_end_utc",         "New York fin (UTC)",       "int",       "Instrument"),

    # --- IA ---
    ("ai_confidence_threshold", "Seuil de confiance IA",            "float", "IA"),
    ("ai_train_bars",           "Bougies d'entrainement",           "int",   "IA"),
    ("ai_retrain_bars",         "Re-entrainer tous les N bougies",  "int",   "IA"),

    # --- Strategie / score ---
    ("min_signal_score", "Score minimum d'entree", "float", "Strategie"),
    ("adx_min",          "ADX minimum",            "float", "Strategie"),
    ("adx_strong",       "ADX fort",               "float", "Strategie"),

    # --- Risque ---
    ("risk_per_trade",    "Risque par trade (fraction)", "float", "Risque"),
    ("sl_atr_mult",       "SL (x ATR)",                   "float", "Risque"),
    ("tp_rr_ratio",       "TP (ratio R:R)",               "float", "Risque"),
    ("dynamic_risk",      "Risque dynamique (ADX)",       "bool",  "Risque"),
    ("min_risk_scale",    "Echelle risque min",           "float", "Risque"),
    ("max_risk_scale",    "Echelle risque max",           "float", "Risque"),
    ("max_margin_usage",  "Utilisation marge max",        "float", "Risque"),
    ("max_open_pos",      "Positions ouvertes max",       "int",   "Risque"),
    ("max_daily_dd",      "Drawdown journalier max",      "float", "Risque"),
    ("max_trades_per_day","Trades max par jour",          "int",   "Risque"),

    # --- Trailing / breakeven / partial ---
    ("trail_atr_mult",       "Trailing stop (x ATR)",         "float", "Trailing"),
    ("trail_activate_atr",   "Activation trailing (x ATR)",   "float", "Trailing"),
    ("trail_tighten1_atr",   "Palier 1 (x ATR)",              "float", "Trailing"),
    ("trail_tighten1_mult",  "Palier 1 - nouveau trail",      "float", "Trailing"),
    ("trail_tighten2_atr",   "Palier 2 (x ATR)",              "float", "Trailing"),
    ("trail_tighten2_mult",  "Palier 2 - nouveau trail",      "float", "Trailing"),
    ("enable_breakeven",         "Activer breakeven",             "bool",  "Trailing"),
    ("breakeven_trigger_atr",    "Declencheur breakeven (x ATR)", "float", "Trailing"),
    ("enable_partial_close",     "Activer cloture partielle",     "bool",  "Trailing"),
    ("partial_close_trigger_atr","Palier 1 partiel (x ATR)",      "float", "Trailing"),
    ("partial_close_fraction",   "Fraction fermee palier 1",      "float", "Trailing"),
    ("partial_close2_trigger_atr","Palier 2 partiel (x ATR)",     "float", "Trailing"),
    ("partial_close2_fraction",  "Fraction fermee palier 2",      "float", "Trailing"),

    # --- Sortie / gestion ---
    ("close_on_opposite_signal",  "Fermer sur signal oppose",             "bool",  "Sortie"),
    ("opposite_signal_min_score", "Score min signal oppose",              "float", "Sortie"),
    ("enable_trend_flip_close",   "Fermer sur retournement H1",           "bool",  "Sortie"),
    ("close_old_profitable",      "Fermer anciennes positions gagnantes", "bool",  "Sortie"),
    ("max_position_age_hours",    "Age max position (h)",                 "float", "Sortie"),
    ("equity_protect_close_all",  "Tout fermer si DD atteint",            "bool",  "Sortie"),
    ("server_utc_offset_hours",   "Decalage UTC serveur (h)",             "float", "Sortie"),

    # --- Power-up (decisions profit/perte automatiques) ---
    ("max_loss_per_trade_pct",   "Stop dur (% equity / trade)",     "float", "Power-Up"),
    ("profit_lock_activate_atr", "Activation verrouillage (x ATR)", "float", "Power-Up"),
    ("profit_giveback_pct",      "Giveback max avant cloture",      "float", "Power-Up"),
    ("cut_losing_after_hours",   "Couper perdant apres (h)",        "float", "Power-Up"),

    # --- Execution ---
    ("lot_size",             "Lot fallback",                       "float", "Execution"),
    ("magic_number",         "Magic number",                       "int",   "Execution"),
    ("deviation",            "Deviation (points)",                 "int",   "Execution"),
    ("dry_run",              "Mode simulation (dry-run)",          "bool",  "Execution"),
    ("sleep_interval",       "Intervalle boucle (s)",              "float", "Execution"),
    ("max_slippage_retries", "Essais si requote",                  "int",   "Execution"),
    ("max_spread_points",    "Spread max en points (0=desactive)", "int",   "Execution"),

    # --- Notifications ---
    ("enable_sound",     "Son active",       "bool",     "Notifications"),
    ("telegram_token",   "Token Telegram",   "password", "Notifications"),
    ("telegram_chat_id", "Chat ID Telegram", "str",      "Notifications"),
]

SECTIONS_ORDER = ["Connexion", "Instrument", "IA", "Strategie", "Risque",
                   "Trailing", "Sortie", "Power-Up", "Execution", "Notifications"]


# ---------------------------------------------------------------------------
# Logging bridge: feeds run.py's logger into a queue the GUI can poll
# ---------------------------------------------------------------------------
class QueueLogHandler(logging.Handler):
    def __init__(self, log_queue: "queue.Queue[str]"):
        super().__init__()
        self.log_queue = log_queue
        self.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                                             datefmt="%H:%M:%S"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.log_queue.put_nowait(self.format(record))
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Scrollable tab helper (some sections have many fields)
# ---------------------------------------------------------------------------
def make_scrollable_tab(notebook: ttk.Notebook, title: str) -> ttk.Frame:
    outer = ttk.Frame(notebook)
    notebook.add(outer, text=title)
    canvas = tk.Canvas(outer, borderwidth=0, highlightthickness=0)
    scrollbar = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
    inner = ttk.Frame(canvas)
    inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.create_window((0, 0), window=inner, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    def _on_mousewheel(event):
        canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
    canvas.bind_all("<MouseWheel>", _on_mousewheel)  # Windows
    return inner


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------
class BotGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("AI Trading Bot - XAUUSDm - Panneau de controle")
        self.root.geometry("1180x760")

        self.bot: "botmod.StandaloneAIBot | None" = None
        self.bot_thread: "threading.Thread | None" = None
        self.log_queue: "queue.Queue[str]" = queue.Queue()
        self.vars: dict = {}

        self._build_layout()
        self.start_btn.config(state="disabled", text="Chargement des modules...")
        self._poll_logs()
        self._poll_status()
        # Heavy imports (pandas/sklearn/pandas_ta/run.py) happen off the UI
        # thread, AFTER the window is already visible, so the app opens
        # instantly instead of blocking on the ML stack's import time.
        threading.Thread(target=self._load_botmod_async, daemon=True).start()

    # ------------------------------------------------------------------
    # Lazy loading of run.py (and its heavy dependencies)
    # ------------------------------------------------------------------
    def _load_botmod_async(self) -> None:
        global botmod
        try:
            import run as _run
            botmod = _run
            self.root.after(0, self._on_botmod_loaded)
        except Exception as exc:
            self.root.after(0, lambda: self._on_botmod_failed(exc))

    def _on_botmod_loaded(self) -> None:
        self._attach_log_handler()
        self._load_initial_config()
        self.start_btn.config(text="Demarrer le bot",
                               state="normal" if MT5_AVAILABLE else "disabled")
        self._log_local("Modules charges, bot pret.")

    def _on_botmod_failed(self, exc: Exception) -> None:
        self.start_btn.config(text="Erreur de chargement", state="disabled")
        self._log_local(f"Echec du chargement de run.py: {exc}")
        messagebox.showerror("Erreur", f"Impossible de charger run.py :\n{exc}")

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    def _build_layout(self) -> None:
        main = ttk.Frame(self.root, padding=8)
        main.pack(fill="both", expand=True)

        left = ttk.Frame(main)
        left.pack(side="left", fill="both", expand=True)

        right = ttk.Frame(main, width=360)
        right.pack(side="right", fill="y", padx=(10, 0))
        right.pack_propagate(False)

        # --- config notebook (left) ---
        self.notebook = ttk.Notebook(left)
        self.notebook.pack(fill="both", expand=True)
        section_frames = {}
        for section in SECTIONS_ORDER:
            section_frames[section] = make_scrollable_tab(self.notebook, section)

        for key, label, kind, section in FIELDS:
            self._add_field(section_frames[section], key, label, kind)

        # --- JSON save/load bar (bottom of left) ---
        jsonbar = ttk.Frame(left, padding=(0, 8, 0, 0))
        jsonbar.pack(fill="x")
        ttk.Button(jsonbar, text="Charger JSON...", command=self.load_json_dialog).pack(side="left")
        ttk.Button(jsonbar, text="Sauvegarder JSON...", command=self.save_json_dialog).pack(side="left", padx=6)
        ttk.Button(jsonbar, text="Sauvegarder (config.json)", command=self.save_default_json).pack(side="left")
        self.json_path_label = ttk.Label(jsonbar, text=str(DEFAULT_JSON_PATH))
        self.json_path_label.pack(side="left", padx=10)

        # --- control panel (right) ---
        ctrl = ttk.LabelFrame(right, text="Controle du bot", padding=10)
        ctrl.pack(fill="x")

        self.start_btn = ttk.Button(ctrl, text="Demarrer le bot", command=self.start_bot)
        self.start_btn.pack(fill="x", pady=2)
        self.stop_btn = ttk.Button(ctrl, text="Arreter le bot", command=self.stop_bot, state="disabled")
        self.stop_btn.pack(fill="x", pady=2)
        self.close_all_btn = ttk.Button(ctrl, text="Fermer toutes les positions",
                                         command=self.close_all_positions, state="disabled")
        self.close_all_btn.pack(fill="x", pady=(10, 2))

        if not MT5_AVAILABLE:
            ttk.Label(ctrl, text="MetaTrader5 non installe - demarrage desactive",
                      foreground="red", wraplength=320).pack(fill="x", pady=(6, 0))

        # --- status panel ---
        status = ttk.LabelFrame(right, text="Statut en direct", padding=10)
        status.pack(fill="x", pady=(10, 0))

        self.status_vars = {
            "state":     tk.StringVar(value="Arrete"),
            "connected": tk.StringVar(value="-"),
            "balance":   tk.StringVar(value="-"),
            "equity":    tk.StringVar(value="-"),
            "positions": tk.StringVar(value="-"),
            "trades":    tk.StringVar(value="-"),
            "last_bar":  tk.StringVar(value="-"),
            "signal":    tk.StringVar(value="-"),
        }
        rows = [
            ("Etat",              "state"),
            ("Connecte",          "connected"),
            ("Solde",             "balance"),
            ("Equity",            "equity"),
            ("Positions ouvertes","positions"),
            ("Trades aujourd'hui","trades"),
            ("Derniere bougie",   "last_bar"),
            ("Dernier signal",    "signal"),
        ]
        for i, (label, key) in enumerate(rows):
            ttk.Label(status, text=label + " :").grid(row=i, column=0, sticky="w", pady=1)
            ttk.Label(status, textvariable=self.status_vars[key], font=("TkDefaultFont", 9, "bold")).grid(
                row=i, column=1, sticky="w", padx=(6, 0), pady=1)

        # --- results / performance panel: is the bot actually winning? ---
        results = ttk.LabelFrame(right, text="Resultats (depuis le demarrage du bot)", padding=10)
        results.pack(fill="x", pady=(10, 0))

        self.result_vars = {
            "closed_trades": tk.StringVar(value="-"),
            "wins_losses":   tk.StringVar(value="-"),
            "win_rate":      tk.StringVar(value="-"),
            "total_pnl":     tk.StringVar(value="-"),
            "best_trade":    tk.StringVar(value="-"),
            "worst_trade":   tk.StringVar(value="-"),
            "session_pnl":   tk.StringVar(value="-"),
        }
        result_rows = [
            ("Trades fermes",        "closed_trades"),
            ("Gagnants / Perdants",  "wins_losses"),
            ("Taux de reussite",     "win_rate"),
            ("P&L total (trades)",   "total_pnl"),
            ("Meilleur trade",       "best_trade"),
            ("Pire trade",           "worst_trade"),
            ("P&L session (equity)","session_pnl"),
        ]
        self.result_pnl_label = None
        for i, (label, key) in enumerate(result_rows):
            ttk.Label(results, text=label + " :").grid(row=i, column=0, sticky="w", pady=1)
            lbl = ttk.Label(results, textvariable=self.result_vars[key], font=("TkDefaultFont", 9, "bold"))
            lbl.grid(row=i, column=1, sticky="w", padx=(6, 0), pady=1)
            if key in ("total_pnl", "session_pnl"):
                setattr(self, f"_{key}_label", lbl)
        ttk.Label(results, text="Vert = le bot va dans le bon sens. Rouge = ca part mal.",
                  font=("TkDefaultFont", 8), foreground="#888").grid(
            row=len(result_rows), column=0, columnspan=2, sticky="w", pady=(6, 0))

        # --- log viewer ---
        logframe = ttk.LabelFrame(right, text="Logs en direct", padding=6)
        logframe.pack(fill="both", expand=True, pady=(10, 0))
        self.log_text = ScrolledText(logframe, wrap="word", height=20, state="disabled",
                                      bg="#111", fg="#ddd", font=("Consolas", 9))
        self.log_text.pack(fill="both", expand=True)

    def _add_field(self, parent: ttk.Frame, key: str, label: str, kind: str) -> None:
        row = ttk.Frame(parent, padding=(8, 4))
        row.pack(fill="x")
        ttk.Label(row, text=label, width=32, anchor="w").pack(side="left")

        if kind == "bool":
            var = tk.BooleanVar()
            ttk.Checkbutton(row, variable=var).pack(side="left")
        elif kind == "timeframe":
            var = tk.StringVar()
            ttk.Combobox(row, textvariable=var, values=TF_OPTIONS, width=10,
                         state="readonly").pack(side="left")
        elif kind == "password":
            var = tk.StringVar()
            entry = ttk.Entry(row, textvariable=var, width=30, show="*")
            entry.pack(side="left")

            def toggle(entry=entry):
                entry.config(show="" if entry.cget("show") == "*" else "*")
            ttk.Button(row, text="afficher/masquer", command=toggle).pack(side="left", padx=4)
        else:  # str, int, float -> plain entry, validated on save
            var = tk.StringVar()
            ttk.Entry(row, textvariable=var, width=30).pack(side="left")

        self.vars[key] = (var, kind)

    # ------------------------------------------------------------------
    # Config <-> widgets
    # ------------------------------------------------------------------
    def _populate_vars(self, cfg: dict) -> None:
        for key, (var, kind) in self.vars.items():
            if key not in cfg:
                continue
            value = cfg[key]
            if kind == "bool":
                var.set(bool(value))
            elif kind == "timeframe":
                var.set(TF_MAP_REV.get(value, "M15"))
            else:
                var.set("" if value is None else str(value))

    def _collect_config(self) -> dict:
        """Read every widget and rebuild a full CONFIG dict. Raises ValueError
        with a readable message if a field can't be converted."""
        if botmod is None:
            raise ValueError("Modules encore en cours de chargement, patiente un instant.")
        cfg = copy.deepcopy(botmod.CONFIG)
        for key, (var, kind) in self.vars.items():
            raw = var.get()
            try:
                if kind == "bool":
                    cfg[key] = bool(raw)
                elif kind == "timeframe":
                    cfg[key] = TF_MAP[raw]
                elif kind == "int":
                    cfg[key] = int(float(raw)) if str(raw).strip() != "" else 0
                elif kind == "float":
                    cfg[key] = float(raw) if str(raw).strip() != "" else 0.0
                else:
                    cfg[key] = raw
            except (ValueError, KeyError) as exc:
                raise ValueError(f"Valeur invalide pour '{key}': {raw!r} ({exc})")
        return cfg

    def _load_initial_config(self) -> None:
        if DEFAULT_JSON_PATH.exists():
            try:
                cfg = json.loads(DEFAULT_JSON_PATH.read_text(encoding="utf-8"))
                merged = copy.deepcopy(botmod.CONFIG)
                merged.update(cfg)
                self._populate_vars(merged)
                self._log_local(f"Configuration chargee depuis {DEFAULT_JSON_PATH}")
                return
            except Exception as exc:
                self._log_local(f"Impossible de charger {DEFAULT_JSON_PATH}: {exc}")
        self._populate_vars(botmod.CONFIG)
        self._log_local("Configuration par defaut chargee (run.py CONFIG).")

    # ------------------------------------------------------------------
    # JSON save / load
    # ------------------------------------------------------------------
    def save_default_json(self) -> None:
        self._save_json(DEFAULT_JSON_PATH)

    def save_json_dialog(self) -> None:
        path = filedialog.asksaveasfilename(defaultextension=".json",
                                             filetypes=[("JSON", "*.json")],
                                             initialfile="config.json")
        if path:
            self._save_json(Path(path))

    def _save_json(self, path: Path) -> None:
        try:
            cfg = self._collect_config()
        except ValueError as exc:
            messagebox.showerror("Configuration invalide", str(exc))
            return
        try:
            path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
            self.json_path_label.config(text=str(path))
            self._log_local(f"Configuration sauvegardee dans {path}")
            messagebox.showinfo("Sauvegarde", f"Configuration sauvegardee :\n{path}\n\n"
                                 "Attention: le mot de passe et le token Telegram sont "
                                 "stockes en clair dans ce fichier.")
        except Exception as exc:
            messagebox.showerror("Erreur de sauvegarde", str(exc))

    def load_json_dialog(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("JSON", "*.json")])
        if not path:
            return
        try:
            cfg = json.loads(Path(path).read_text(encoding="utf-8"))
            merged = copy.deepcopy(botmod.CONFIG)
            merged.update(cfg)
            self._populate_vars(merged)
            self.json_path_label.config(text=path)
            self._log_local(f"Configuration chargee depuis {path}")
        except Exception as exc:
            messagebox.showerror("Erreur de chargement", str(exc))

    # ------------------------------------------------------------------
    # Bot control
    # ------------------------------------------------------------------
    def start_bot(self) -> None:
        if botmod is None:
            messagebox.showwarning("Patiente", "Les modules sont encore en cours de chargement.")
            return
        if self.bot_thread and self.bot_thread.is_alive():
            return
        try:
            cfg = self._collect_config()
        except ValueError as exc:
            messagebox.showerror("Configuration invalide", str(exc))
            return

        self.bot = botmod.StandaloneAIBot(cfg)
        self.bot_thread = threading.Thread(target=self.bot.run, daemon=True)
        self.bot_thread.start()

        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.close_all_btn.config(state="normal")
        self.status_vars["state"].set("Demarrage...")
        self._log_local("Bot demarre depuis le GUI.")

    def stop_bot(self) -> None:
        if self.bot:
            self.bot.running = False
            self.status_vars["state"].set("Arret en cours...")
            self._log_local("Arret demande depuis le GUI (fin de boucle en cours).")
        self.stop_btn.config(state="disabled")

    def close_all_positions(self) -> None:
        if not self.bot:
            return
        if messagebox.askyesno("Confirmer", "Fermer TOUTES les positions ouvertes maintenant ?"):
            threading.Thread(target=self.bot._close_all, args=("manual GUI close",), daemon=True).start()
            self._log_local("Fermeture manuelle de toutes les positions demandee.")

    # ------------------------------------------------------------------
    # Logging bridge
    # ------------------------------------------------------------------
    def _attach_log_handler(self) -> None:
        handler = QueueLogHandler(self.log_queue)
        botmod.logger.addHandler(handler)

    def _log_local(self, msg: str) -> None:
        """Log a GUI-only message (not from the bot's logger)."""
        self.log_queue.put_nowait(f"[GUI] {msg}")

    def _poll_logs(self) -> None:
        MAX_LINES = 800
        updated = False
        try:
            while True:
                line = self.log_queue.get_nowait()
                self.log_text.config(state="normal")
                self.log_text.insert("end", line + "\n")
                updated = True
                self.log_queue.task_done()
        except queue.Empty:
            pass
        if updated:
            # trim from the top if the log grows too long
            n_lines = int(self.log_text.index("end-1c").split(".")[0])
            if n_lines > MAX_LINES:
                self.log_text.delete("1.0", f"{n_lines - MAX_LINES}.0")
            self.log_text.see("end")
            self.log_text.config(state="disabled")
        self.root.after(300, self._poll_logs)

    # ------------------------------------------------------------------
    # Status polling (reads bot.get_status(), never touches MT5 directly
    # from this thread — the bot thread is the only MT5 caller)
    # ------------------------------------------------------------------
    def _poll_status(self) -> None:
        if self.bot is not None:
            st = self.bot.get_status()
            running = self.bot_thread.is_alive() if self.bot_thread else False

            if running:
                self.status_vars["state"].set("En cours" if not self.bot.cfg.get("dry_run") else "En cours (dry-run)")
            else:
                self.status_vars["state"].set("Arrete")
                self.start_btn.config(state="normal" if MT5_AVAILABLE else "disabled")
                self.stop_btn.config(state="disabled")
                self.close_all_btn.config(state="disabled")

            self.status_vars["connected"].set("Oui" if st.get("connected") else "Non")
            self.status_vars["balance"].set(f"{st.get('balance', 0.0):.2f}")
            self.status_vars["equity"].set(f"{st.get('equity', 0.0):.2f}")
            self.status_vars["positions"].set(str(st.get("open_positions", 0)))
            self.status_vars["trades"].set(str(st.get("trades_today", 0)))
            self.status_vars["last_bar"].set(str(st.get("last_bar", "-")))
            direction = st.get("last_direction", 0)
            dir_txt = {1: "ACHAT", -1: "VENTE", 0: "aucun"}.get(direction, "-")
            self.status_vars["signal"].set(
                f"{dir_txt} | score={st.get('last_score', 0):.0f} | {st.get('last_signal', '-')}")

        self.root.after(1000, self._poll_status)

    # ------------------------------------------------------------------
    def on_close(self) -> None:
        if self.bot_thread and self.bot_thread.is_alive():
            if not messagebox.askyesno("Quitter", "Le bot tourne encore. Arreter et quitter ?"):
                return
            self.stop_bot()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    app = BotGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == "__main__":
    # REQUIRED when this app is frozen with PyInstaller/cx_Freeze and uses
    # multiprocessing (scikit-learn/joblib can spawn worker processes even
    # at n_jobs=1 in some code paths, e.g. the loky resource tracker).
    # Without this, each spawned worker process re-executes the frozen exe
    # from the top and reopens the ENTIRE Tkinter app — which is exactly
    # what produces the "many bot windows" symptom. Must be the very first
    # call in this block, before anything else.
    multiprocessing.freeze_support()
    main()