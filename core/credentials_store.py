"""
credentials_store.py - persistence for the MT5 credentials entered in the GUI.

Stores mt5_path / login / password / server in a JSON file next to the
executable (or the working directory in dev mode) so the user never has to
edit config.py.

Security note: the file is PLAIN TEXT. Anyone with access to the machine can
read it - treat it like you would a password stored in config.py. Never share
it, and prefer a demo/paper account password for daily use.
"""
import json
import logging
import os
import sys

logger = logging.getLogger(__name__)

FIELDS = ("mt5_path", "login", "password", "server")


def base_dir() -> str:
    """Where credentials.json lives: next to the exe when frozen, else cwd."""
    if getattr(sys, "frozen", False):  # packaged with PyInstaller
        return os.path.dirname(sys.executable)
    return os.getcwd()


def file_path() -> str:
    return os.path.join(base_dir(), "credentials.json")


def _blank() -> dict:
    return {k: "" for k in FIELDS}


def load_credentials() -> dict:
    """Return saved credentials (all fields as strings, '' when absent)."""
    try:
        with open(file_path(), encoding="utf-8") as fh:
            data = json.load(fh)
        return {k: data.get(k, "") for k in FIELDS}
    except FileNotFoundError:
        return _blank()
    except Exception as exc:
        logger.warning("Could not read credentials file: %s", exc)
        return _blank()


def save_credentials(credentials: dict) -> bool:
    try:
        with open(file_path(), "w", encoding="utf-8") as fh:
            json.dump({k: credentials.get(k, "") for k in FIELDS}, fh, indent=2)
        return True
    except Exception as exc:
        logger.error("Could not save credentials: %s", exc)
        return False


def clear_credentials() -> None:
    try:
        os.remove(file_path())
    except FileNotFoundError:
        pass
    except Exception as exc:
        logger.warning("Could not remove credentials file: %s", exc)
