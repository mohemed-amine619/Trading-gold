"""
scripts/collect_diagnostics.py — gather what is needed to review how the bot is
really performing, into one zip with secrets removed.

Run it on the machine where the bot and the MT5 terminal are running:

    .venv\\Scripts\\python.exe scripts\\collect_diagnostics.py --days 7

It READS the MT5 deal history, the bot's log, its trade database, its saved risk
state and the model-training history. It never sends orders and never changes
any file except writing the output zip under ``diagnostics\\``.

The zip holds (nothing secret):
  summary.md           findings, trade statistics, execution health
  stats.json           the same numbers, machine-readable
  closed_trades.csv    every closed round trip from the broker history
  log_tail.txt         the end of trading_bot.log, scrubbed
  config_redacted.env  your .env with passwords, tokens and the login masked
  account.json         demo/real, balance, equity (login masked)
  trades_db.csv, risk_state/, nn/   whatever exists

Send summary.md (or the whole zip) to whoever is reviewing the bot. Look inside
before you send it: the scrubbing is thorough, not a guarantee.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sqlite3
import sys
import zipfile
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.diagnostics.confidence import confidence_buckets           # noqa: E402
from core.diagnostics.logstats import parse_log                      # noqa: E402
from core.diagnostics.redact import (                                 # noqa: E402
    login_from_env, mask_account, redact_env_text, scrub_text, secret_values_from_env,
)
from core.diagnostics.report import build_summary                     # noqa: E402
from core.diagnostics.trades import Trade, pair_deals, trade_stats    # noqa: E402

_DEAL_FIELDS = ("position_id", "type", "entry", "time", "volume", "price", "profit",
                "commission", "swap", "fee", "symbol", "magic", "reason", "comment")
_ACCOUNT_TYPES = {0: "demo", 1: "contest", 2: "real"}


def _deal_dict(deal) -> dict:
    if hasattr(deal, "_asdict"):
        raw = deal._asdict()
        return {k: raw.get(k) for k in _DEAL_FIELDS}
    return {k: getattr(deal, k, None) for k in _DEAL_FIELDS}


def read_env(root: Path) -> str:
    path = root / ".env"
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


def read_tail(path: Path, n: int) -> List[str]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", errors="replace") as fh:
        return list(deque(fh, maxlen=n))


def fetch_mt5(mt5, days: int, now: datetime) -> dict:
    """Account facts and deals, read-only. Returns ``{"account": {}, "deals": []}``."""
    if not mt5.initialize():
        return {"account": {}, "deals": [], "error": f"MT5 initialize failed: {mt5.last_error()}"}
    try:
        info = mt5.account_info()
        account = {}
        if info is not None:
            account = {
                "type": _ACCOUNT_TYPES.get(int(getattr(info, "trade_mode", -1)), "unknown"),
                "login": mask_account(getattr(info, "login", "")),
                "server": str(getattr(info, "server", "")),
                "currency": str(getattr(info, "currency", "")),
                "leverage": getattr(info, "leverage", None),
                "balance": getattr(info, "balance", None),
                "equity": getattr(info, "equity", None),
            }
        deals = mt5.history_deals_get(now - timedelta(days=days), now + timedelta(days=1)) or ()
        return {"account": account, "deals": [_deal_dict(d) for d in deals]}
    finally:
        mt5.shutdown()


def _csv_text(rows: List[dict], fields: Optional[List[str]] = None) -> str:
    buf = io.StringIO()
    if not rows and not fields:
        return ""
    writer = csv.DictWriter(buf, fieldnames=fields or list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


def _read_trades_db(root: Path, env_text: str) -> List[dict]:
    url = next((ln.split("=", 1)[1].strip() for ln in env_text.splitlines()
                if ln.strip().startswith("DATABASE_URL=")), "sqlite:///./data/trading.db")
    rel = url.replace("sqlite:///", "", 1) if url.startswith("sqlite:///") else "data/trading.db"
    path = (root / rel).resolve() if not Path(rel).is_absolute() else Path(rel)
    if not path.exists():
        return []
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5.0)
        conn.row_factory = sqlite3.Row
        rows = [dict(r) for r in conn.execute("SELECT * FROM trades ORDER BY id")]
        conn.close()
        return rows
    except sqlite3.Error:
        return []


def _env_value(env_text: str, key: str) -> Optional[str]:
    for line in env_text.splitlines():
        if line.strip().startswith(key + "="):
            return line.split("=", 1)[1].strip().strip("\"'")
    return None


def collect(root: Path, out_dir: Path, days: int = 7, tail: int = 20000,
            magic: Optional[int] = None, use_env_magic: bool = True, mt5=None,
            now: Optional[datetime] = None, log_name: str = "trading_bot.log") -> Path:
    now = now or datetime.now(timezone.utc)
    env_text = read_env(root)
    secrets = secret_values_from_env(env_text)
    login = login_from_env(env_text)
    clean = lambda text: scrub_text(text, secrets, [login])             # noqa: E731

    if magic is None and use_env_magic:
        value = _env_value(env_text, "MAGIC_NUMBER")
        magic = int(value) if value and value.isdigit() else None

    notes: List[str] = []
    history = {"account": {}, "deals": []}
    if mt5 is None:
        try:
            import MetaTrader5 as mt5                                   # Windows only
        except ImportError:
            mt5 = None
    if mt5 is None:
        notes.append("MetaTrader5 is not available here: no broker history, trade statistics are empty.")
    else:
        history = fetch_mt5(mt5, days, now)
        if history.get("error"):
            notes.append(history["error"])
    trades: List[Trade] = pair_deals(history["deals"], magic=magic)
    if magic is not None:
        notes.append(f"Only trades with magic number {magic} are included.")

    log_lines = [clean(line) for line in read_tail(root / log_name, tail)]
    if not log_lines:
        notes.append(f"No {log_name} found or it is empty.")
    log = parse_log(log_lines)
    stats = trade_stats(trades)
    conf = confidence_buckets(log["entries"], trades)

    account = history["account"]
    meta = {"generated": now.strftime("%Y-%m-%d %H:%M UTC"), "days": days,
            "account_type": account.get("type", "unknown"), "account": account.get("login", ""),
            "balance": account.get("balance", "?"), "equity": account.get("equity", "?")}

    members: Dict[str, str] = {
        "summary.md": build_summary(meta, stats, log, notes, conf),
        "stats.json": json.dumps({"meta": meta, "trades": stats, "log": log, "confidence": conf},
                                 indent=2, default=str),
        "closed_trades.csv": _csv_text([t.to_row() for t in trades],
                                       list(Trade.__dataclass_fields__)),
        "log_tail.txt": "".join(log_lines),
        "config_redacted.env": redact_env_text(env_text),
        "account.json": json.dumps(account, indent=2, default=str),
    }
    db_rows = _read_trades_db(root, env_text)
    if db_rows:
        members["trades_db.csv"] = _csv_text(db_rows)
    state_dir = root / (_env_value(env_text, "RISK_STATE_DIR") or "data/state")
    for path in sorted(state_dir.glob("risk_*.json")) if state_dir.exists() else []:
        members[f"risk_state/{path.name}"] = path.read_text(encoding="utf-8", errors="replace")
    for path in sorted((root / "data" / "models").glob("*.history.jsonl")):
        members[f"nn/{path.name}"] = path.read_text(encoding="utf-8", errors="replace")

    members = {name: clean(text) for name, text in members.items()}
    for name, text in members.items():                                  # defence in depth
        for secret in secrets:
            if secret in text:
                raise RuntimeError(f"refusing to write the bundle: a secret value is still in {name}")

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"diagnostics_{now.strftime('%Y%m%d_%H%M%S')}.zip"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, text in members.items():
            zf.writestr(name, text)
    (out_dir / "latest_summary.md").write_text(members["summary.md"], encoding="utf-8")
    return path


def main(argv: Optional[List[str]] = None, mt5=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="the bot folder (default: here)")
    parser.add_argument("--out", type=Path, default=None, help="default: <root>/diagnostics")
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--tail", type=int, default=20000, help="log lines to keep")
    parser.add_argument("--magic", type=int, default=None)
    parser.add_argument("--all-magics", action="store_true",
                        help="include manual trades and other bots, not just the bot's magic number")
    args = parser.parse_args(argv)

    root = args.root.resolve()
    try:
        path = collect(root, args.out or root / "diagnostics", args.days, args.tail,
                       args.magic, use_env_magic=not args.all_magics, mt5=mt5)
    except RuntimeError as exc:
        print(f"error: {exc}")
        return 1
    print(f"wrote {path}")
    print(f"read first: {path.parent / 'latest_summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
