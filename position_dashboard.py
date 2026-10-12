"""
Position Dashboard — لوحة مستقلة لمراقبة الصفقات المفتوحة والإحصائيات
=====================================================================
سكريبت مستقل تمامًا عن run.py و gui.py.
يتصل بنفس حساب MT5 المفتوح على الترمينال (لا يحتاج بيانات دخول إذا
الترمينال شغال ومسجّل دخول أصلاً).

المتطلبات:
    pip install MetaTrader5

التشغيل:
    python position_dashboard.py

إذا بدك تتصل بحساب مختلف عن الترمينال المفتوح، عدّل القيم بالأسفل
في MT5_LOGIN / MT5_PASSWORD / MT5_SERVER (اتركها None لاستخدام
الاتصال الحالي بالترمينال).
"""

import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime, timedelta

try:
    import MetaTrader5 as mt5
except ImportError:
    raise SystemExit("مكتبة MetaTrader5 غير مثبّتة. شغّل: pip install MetaTrader5")

# ------------------------------------------------------------------
# إعدادات اختيارية للاتصال (اتركها None لاستخدام الترمينال المفتوح)
# ------------------------------------------------------------------
MT5_LOGIN = None
MT5_PASSWORD = None
MT5_SERVER = None

REFRESH_MS = 1000  # كل كم مللي ثانية يحدّث الجدول (1000 = ثانية وحدة)

BG_DARK = "#1e1e2e"
BG_PANEL = "#282838"
FG_TEXT = "#e0e0e0"
GREEN = "#2ecc71"
RED = "#e74c3c"
GRAY = "#888888"


class PositionDashboard(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Position Dashboard — لوحة الصفقات المباشرة")
        self.geometry("1100x600")
        self.configure(bg=BG_DARK)

        self._connect_mt5()
        self._build_ui()
        self._refresh_loop()

    # ---------------- الاتصال بـ MT5 ----------------
    def _connect_mt5(self):
        ok = mt5.initialize()
        if MT5_LOGIN and MT5_PASSWORD and MT5_SERVER:
            ok = mt5.login(MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER)

        if not ok:
            messagebox.showerror(
                "خطأ اتصال",
                f"ما قدرت أتصل بـ MT5.\n{mt5.last_error()}\n"
                "تأكد إنه الترمينال شغال ومسجّل دخول.",
            )
            raise SystemExit("فشل الاتصال بـ MT5")

    # ---------------- بناء الواجهة ----------------
    def _build_ui(self):
        # -------- لوحة الإحصائيات العلوية --------
        stats_frame = tk.Frame(self, bg=BG_PANEL, pady=10)
        stats_frame.pack(fill="x", padx=10, pady=(10, 5))

        self.stat_labels = {}
        stats = [
            ("balance", "الرصيد"),
            ("equity", "Equity"),
            ("open_pl", "ربح/خسارة الصفقات المفتوحة"),
            ("win_rate", "نسبة النجاح"),
            ("wins", "صفقات رابحة"),
            ("losses", "صفقات خاسرة"),
            ("total_pl", "إجمالي الربح/الخسارة (مغلقة)"),
        ]
        for i, (key, label) in enumerate(stats):
            box = tk.Frame(stats_frame, bg=BG_PANEL)
            box.grid(row=0, column=i, padx=15)
            tk.Label(box, text=label, bg=BG_PANEL, fg=GRAY, font=("Segoe UI", 9)).pack()
            val = tk.Label(box, text="--", bg=BG_PANEL, fg=FG_TEXT, font=("Segoe UI", 13, "bold"))
            val.pack()
            self.stat_labels[key] = val

        # -------- جدول الصفقات المفتوحة --------
        table_frame = tk.Frame(self, bg=BG_DARK)
        table_frame.pack(fill="both", expand=True, padx=10, pady=5)

        columns = ("ticket", "symbol", "type", "volume", "open_price",
                   "current_price", "sl", "tp", "profit", "pips", "pct_balance", "duration")
        headers = {
            "ticket": "التذكرة", "symbol": "الرمز", "type": "النوع",
            "volume": "الحجم", "open_price": "سعر الدخول",
            "current_price": "السعر الحالي", "sl": "SL", "tp": "TP",
            "profit": "الربح/الخسارة ($)", "pips": "النقاط (pips)",
            "pct_balance": "% من الرصيد", "duration": "المدة",
        }

        style = ttk.Style()
        style.theme_use("default")
        style.configure("Treeview", background=BG_PANEL, fieldbackground=BG_PANEL,
                         foreground=FG_TEXT, rowheight=28, font=("Segoe UI", 10))
        style.configure("Treeview.Heading", background="#33334a", foreground=FG_TEXT,
                         font=("Segoe UI", 10, "bold"))
        style.map("Treeview", background=[("selected", "#44445a")])

        self.tree = ttk.Treeview(table_frame, columns=columns, show="headings")
        for col in columns:
            self.tree.heading(col, text=headers[col])
            self.tree.column(col, width=100, anchor="center")
        self.tree.column("symbol", width=90)
        self.tree.pack(fill="both", expand=True, side="left")

        scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")

        self.tree.tag_configure("profit", foreground=GREEN)
        self.tree.tag_configure("loss", foreground=RED)
        self.tree.tag_configure("flat", foreground=GRAY)

        # -------- شريط سفلي: آخر تحديث + زر إغلاق طارئ --------
        bottom = tk.Frame(self, bg=BG_DARK)
        bottom.pack(fill="x", padx=10, pady=(0, 10))

        self.last_update_label = tk.Label(bottom, text="", bg=BG_DARK, fg=GRAY, font=("Segoe UI", 9))
        self.last_update_label.pack(side="left")

        emergency_btn = tk.Button(
            bottom, text="🛑 إغلاق كل الصفقات", bg=RED, fg="white",
            font=("Segoe UI", 10, "bold"), relief="flat", padx=12, pady=4,
            command=self._close_all_positions,
        )
        emergency_btn.pack(side="right")

    # ---------------- منطق التحديث ----------------
    def _refresh_loop(self):
        try:
            self._update_stats()
            self._update_positions_table()
        except Exception as e:
            self.last_update_label.config(text=f"خطأ بالتحديث: {e}", fg=RED)
        finally:
            self.after(REFRESH_MS, self._refresh_loop)

    def _update_stats(self):
        account = mt5.account_info()
        if account is None:
            return

        positions = mt5.positions_get() or ()
        open_pl = sum(p.profit for p in positions)

        # إحصائيات الصفقات المغلقة (آخر 30 يوم كمثال، عدّلها إذا بدك)
        from_date = datetime.now() - timedelta(days=30)
        to_date = datetime.now() + timedelta(days=1)
        deals = mt5.history_deals_get(from_date, to_date) or ()

        # ناخد فقط صفقات الخروج (entry == DEAL_ENTRY_OUT) عشان نحسب الربح الفعلي لكل صفقة مغلقة
        closed_profits = [d.profit for d in deals if d.entry == mt5.DEAL_ENTRY_OUT]
        wins = sum(1 for p in closed_profits if p > 0)
        losses = sum(1 for p in closed_profits if p < 0)
        total_closed = wins + losses
        win_rate = (wins / total_closed * 100) if total_closed else 0
        total_pl = sum(closed_profits)

        self.stat_labels["balance"].config(text=f"${account.balance:,.2f}")
        self.stat_labels["equity"].config(text=f"${account.equity:,.2f}")
        self.stat_labels["open_pl"].config(
            text=f"${open_pl:,.2f}", fg=GREEN if open_pl >= 0 else RED
        )
        self.stat_labels["win_rate"].config(text=f"{win_rate:.1f}%")
        self.stat_labels["wins"].config(text=str(wins), fg=GREEN)
        self.stat_labels["losses"].config(text=str(losses), fg=RED)
        self.stat_labels["total_pl"].config(
            text=f"${total_pl:,.2f}", fg=GREEN if total_pl >= 0 else RED
        )

        self.last_update_label.config(
            text=f"آخر تحديث: {datetime.now().strftime('%H:%M:%S')}", fg=GRAY
        )

    def _update_positions_table(self):
        positions = mt5.positions_get() or ()

        # امسح الصفوف القديمة
        for row in self.tree.get_children():
            self.tree.delete(row)

        for p in positions:
            symbol_info = mt5.symbol_info(p.symbol)
            point = symbol_info.point if symbol_info else 0.0001
            digits = symbol_info.digits if symbol_info else 5

            price_open = p.price_open
            price_current = p.price_current
            trade_type = "شراء (Buy)" if p.type == mt5.ORDER_TYPE_BUY else "بيع (Sell)"

            # حساب النقاط (pips)
            diff = (price_current - price_open) if p.type == mt5.ORDER_TYPE_BUY else (price_open - price_current)
            pips = diff / point if point else 0

            account = mt5.account_info()
            pct_balance = (p.profit / account.balance * 100) if account and account.balance else 0

            duration = datetime.now() - datetime.fromtimestamp(p.time)
            hours, rem = divmod(int(duration.total_seconds()), 3600)
            minutes = rem // 60
            duration_str = f"{hours}س {minutes}د"

            tag = "profit" if p.profit > 0 else ("loss" if p.profit < 0 else "flat")

            self.tree.insert("", "end", values=(
                p.ticket,
                p.symbol,
                trade_type,
                f"{p.volume:.2f}",
                f"{price_open:.{digits}f}",
                f"{price_current:.{digits}f}",
                f"{p.sl:.{digits}f}" if p.sl else "-",
                f"{p.tp:.{digits}f}" if p.tp else "-",
                f"{p.profit:,.2f}",
                f"{pips:+.1f}",
                f"{pct_balance:+.2f}%",
                duration_str,
            ), tags=(tag,))

    # ---------------- زر الطوارئ ----------------
    def _close_all_positions(self):
        if not messagebox.askyesno("تأكيد", "متأكد إنك بدك تسكر كل الصفقات المفتوحة؟"):
            return

        positions = mt5.positions_get() or ()
        closed, failed = 0, 0
        for p in positions:
            tick = mt5.symbol_info_tick(p.symbol)
            if tick is None:
                failed += 1
                continue
            order_type = mt5.ORDER_TYPE_SELL if p.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
            price = tick.bid if order_type == mt5.ORDER_TYPE_SELL else tick.ask
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "position": p.ticket,
                "symbol": p.symbol,
                "volume": p.volume,
                "type": order_type,
                "price": price,
                "deviation": 20,
                "magic": p.magic,
                "comment": "Emergency close - dashboard",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            result = mt5.order_send(request)
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                closed += 1
            else:
                failed += 1

        messagebox.showinfo("النتيجة", f"تم إغلاق {closed} صفقة.\nفشل إغلاق {failed} صفقة.")

    def destroy(self):
        mt5.shutdown()
        super().destroy()


if __name__ == "__main__":
    app = PositionDashboard()
    app.mainloop()