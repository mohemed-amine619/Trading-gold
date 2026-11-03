"""
core/diagnostics/redact.py — remove secrets before anything is written to a
diagnostics bundle that the operator will send to someone else.

Three layers, because a secret can leak by name, by pattern, or by value:

* ``redact_env_text``  blanks the VALUE of any env key whose NAME looks secret
  (password, token, secret, api key, private) and masks the account login;
* ``scrub_text``       removes ``password=...``-style pairs and bearer tokens from
  free text such as a log file;
* both accept the actual secret values, so a secret that is logged somewhere
  unexpected is still removed by value.

Pure functions, standard library only.
"""

from __future__ import annotations

import re
from typing import Iterable, List

REDACTED = "<redacted>"

_SECRET_KEY = re.compile(r"(PASSWORD|PASSWD|TOKEN|SECRET|API_?KEY|PRIVATE)", re.IGNORECASE)
_LOGIN_KEY = re.compile(r"^(?:export\s+)?MT5_LOGIN$", re.IGNORECASE)

_PAIR = re.compile(
    r"(?i)\b(password|passwd|pwd|token|secret|api[_-]?key)(\"?'?\s*[:=]\s*\"?'?)([^\s\"',;]+)")
_BEARER = re.compile(r"(?i)(authorization:\s*bearer\s+)\S+")
_DASH_TOKEN = re.compile(r"(?i)(x-dashboard-token:\s*)\S+")

_MIN_SECRET_LEN = 4        # shorter values would blank ordinary words


def mask_account(login) -> str:
    """``261009880`` -> ``***9880``. Empty or short values are fully masked."""
    text = str(login or "").strip()
    return "***" + text[-4:] if len(text) > 4 else "***"


def secret_values_from_env(text: str) -> List[str]:
    """The values of every secret-looking key in an env file's text."""
    values: List[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.replace("export ", "").strip()
        value = value.strip().strip("\"'")
        if _SECRET_KEY.search(key) and len(value) >= _MIN_SECRET_LEN:
            values.append(value)
    return values


def login_from_env(text: str) -> str:
    for raw in text.splitlines():
        line = raw.strip()
        if "=" in line and not line.startswith("#"):
            key, _, value = line.partition("=")
            if _LOGIN_KEY.match(key.strip()):
                return value.strip().strip("\"'")
    return ""


def redact_env_text(text: str) -> str:
    """Blank secret VALUES and mask the login; keep every other line as written."""
    out = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            out.append(raw)
            continue
        key, _, value = raw.partition("=")
        bare = key.replace("export ", "").strip()
        if _SECRET_KEY.search(bare):
            out.append(f"{key}={REDACTED}" if value.strip() else raw)
        elif _LOGIN_KEY.match(bare):
            out.append(f"{key}={mask_account(value.strip().strip(chr(34) + chr(39)))}")
        else:
            out.append(raw)
    return "\n".join(out) + ("\n" if text.endswith("\n") else "")


def scrub_text(text: str, secrets: Iterable[str] = (), mask_values: Iterable[str] = ()) -> str:
    """Remove secrets from free text.

    ``secrets`` are replaced by value; ``mask_values`` (account numbers) are
    replaced by their masked form. Pair and bearer patterns are removed even
    when the value is not known.
    """
    for secret in sorted({s for s in secrets if s and len(s) >= _MIN_SECRET_LEN},
                         key=len, reverse=True):
        text = text.replace(secret, REDACTED)
    for value in {v for v in mask_values if v and len(str(v)) > 4}:
        text = text.replace(str(value), mask_account(value))
    text = _PAIR.sub(lambda m: f"{m.group(1)}{m.group(2)}{REDACTED}", text)
    text = _BEARER.sub(lambda m: f"{m.group(1)}{REDACTED}", text)
    text = _DASH_TOKEN.sub(lambda m: f"{m.group(1)}{REDACTED}", text)
    return text
