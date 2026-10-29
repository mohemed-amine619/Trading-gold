"""
core/accounts/service.py — users, licences, activations, dashboard sessions and
the per-customer settings history.

Every function takes an open SQLAlchemy ``Session`` and leaves committing to the
caller, so the web layer decides transaction boundaries. Nothing here touches
the broker or imports MetaTrader5, so the whole module is testable on Linux.

Failures raise ``AccountError``. The message is written to be shown to the
customer, so it never reveals whether an email address exists.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional, Tuple

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from core.accounts.security import (
    DUMMY_PASSWORD_HASH,
    MIN_PASSWORD_LENGTH,
    hash_password,
    hash_secret,
    new_secret,
    verify_password,
)
from core.config.schema import validate_payload
from core.database.models import (
    DashboardSession,
    Installation,
    Licence,
    SettingsRevision,
    User,
)

ROLES = ("owner", "admin", "customer")
LICENCE_KEY_PREFIX = "GLD-"
DEFAULT_SESSION_TTL = timedelta(hours=12)


class AccountError(Exception):
    """A rule the customer must be told about. Safe to show in the dashboard."""


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _aware(value: datetime) -> datetime:
    """SQLite returns naive datetimes. Treat them as UTC, which is how they were written."""
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


def _normalise_email(email: str) -> str:
    cleaned = (email or "").strip().lower()
    if len(cleaned) > 254 or "@" not in cleaned or cleaned.startswith("@") or cleaned.endswith("@"):
        raise AccountError("enter a valid email address")
    return cleaned


# ─────────────────────────────────────────────────────────────────────────────
# Users and sessions
# ─────────────────────────────────────────────────────────────────────────────

def create_user(session: Session, email: str, password: str, role: str = "customer") -> User:
    if role not in ROLES:
        raise AccountError(f"role must be one of {', '.join(ROLES)}")
    address = _normalise_email(email)
    if len(password or "") < MIN_PASSWORD_LENGTH:
        raise AccountError(f"password must be at least {MIN_PASSWORD_LENGTH} characters")
    exists = session.scalar(select(User.id).where(User.email == address))
    if exists is not None:
        raise AccountError("an account with this email already exists")
    user = User(email=address, password_hash=hash_password(password), role=role)
    session.add(user)
    session.flush()
    return user


def authenticate(session: Session, email: str, password: str) -> Optional[User]:
    """Return the user if the credentials are valid and the account is active."""
    try:
        address = _normalise_email(email)
    except AccountError:
        verify_password(password or "", DUMMY_PASSWORD_HASH)
        return None
    user = session.scalar(select(User).where(User.email == address))
    if user is None:
        verify_password(password or "", DUMMY_PASSWORD_HASH)
        return None
    if not verify_password(password or "", user.password_hash):
        return None
    if not user.is_active:
        return None
    user.last_login_at = _now()
    return user


def issue_session(session: Session, user: User, ttl: timedelta = DEFAULT_SESSION_TTL) -> str:
    """Create a dashboard session. Returns the raw token, which is never stored."""
    token = new_secret()
    now = _now()
    session.add(DashboardSession(
        user_id=user.id,
        token_hash=hash_secret(token),
        created_at=now,
        expires_at=now + ttl,
    ))
    session.flush()
    return token


def resolve_session(session: Session, token: Optional[str]) -> Optional[User]:
    if not token:
        return None
    row = session.scalar(
        select(DashboardSession).where(DashboardSession.token_hash == hash_secret(token)))
    if row is None or row.revoked or _aware(row.expires_at) <= _now():
        return None
    user = session.get(User, row.user_id)
    if user is None or not user.is_active:
        return None
    return user


def revoke_session(session: Session, token: str) -> bool:
    row = session.scalar(
        select(DashboardSession).where(DashboardSession.token_hash == hash_secret(token)))
    if row is None or row.revoked:
        return False
    row.revoked = True
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Licences
# ─────────────────────────────────────────────────────────────────────────────

def create_licence(session: Session, user: User, plan: str = "standard",
                   max_installations: int = 1,
                   expires_at: Optional[datetime] = None) -> Tuple[Licence, str]:
    """Issue a licence. Returns ``(licence, raw_key)``; the raw key is shown once."""
    if max_installations < 1:
        raise AccountError("a licence must allow at least one installation")
    raw_key = LICENCE_KEY_PREFIX + new_secret(24)
    licence = Licence(
        user_id=user.id,
        key_hash=hash_secret(raw_key),
        plan=plan,
        max_installations=max_installations,
        expires_at=expires_at,
    )
    session.add(licence)
    session.flush()
    return licence, raw_key


def licence_problem(licence: Licence, now: Optional[datetime] = None) -> Optional[str]:
    """Why this licence cannot be used right now, or None if it is valid."""
    now = now or _now()
    if licence.status != "active":
        return "this licence has been revoked"
    if licence.expires_at is not None and _aware(licence.expires_at) <= now:
        return "this licence has expired"
    return None


def find_licence(session: Session, raw_key: str) -> Optional[Licence]:
    if not raw_key:
        return None
    return session.scalar(select(Licence).where(Licence.key_hash == hash_secret(raw_key.strip())))


def activate_installation(session: Session, raw_key: str, machine_id: str,
                          label: str = "", now: Optional[datetime] = None) -> Installation:
    """Bind a machine to a licence, up to the licence's seat limit.

    Activating the same machine again returns its existing installation. It
    does not use up a second seat.
    """
    now = now or _now()
    licence = find_licence(session, raw_key)
    if licence is None:
        raise AccountError("unknown licence key")
    problem = licence_problem(licence, now)
    if problem:
        raise AccountError(problem)
    if not machine_id:
        raise AccountError("a machine identifier is required")

    fingerprint = hash_secret(machine_id)
    existing = session.scalar(select(Installation).where(
        Installation.licence_id == licence.id,
        Installation.fingerprint_hash == fingerprint))
    if existing is not None:
        if existing.revoked:
            raise AccountError("this machine has been removed from the licence")
        existing.last_seen_at = now
        return existing

    in_use = session.scalar(select(func.count()).select_from(Installation).where(
        Installation.licence_id == licence.id,
        Installation.revoked.is_(False)))
    if in_use >= licence.max_installations:
        raise AccountError(
            f"this licence is already active on {licence.max_installations} machine(s)")

    installation = Installation(
        licence_id=licence.id,
        fingerprint_hash=fingerprint,
        label=(label or "")[:128],
        created_at=now,
        last_seen_at=now,
    )
    session.add(installation)
    session.flush()
    return installation


def revoke_installation(session: Session, installation: Installation) -> None:
    installation.revoked = True


# ─────────────────────────────────────────────────────────────────────────────
# Settings history
# ─────────────────────────────────────────────────────────────────────────────

def latest_settings(session: Session, user_id: str) -> Dict:
    row = session.scalar(
        select(SettingsRevision).where(SettingsRevision.user_id == user_id)
        .order_by(SettingsRevision.id.desc()).limit(1))
    return json.loads(row.settings_json) if row is not None else {}


def record_settings(session: Session, user: User, changes: dict,
                    installation: Optional[Installation] = None,
                    changed_by: Optional[User] = None, note: str = "") -> SettingsRevision:
    """Validate a change against the parameter schema and append a new revision.

    ``changes`` may be partial. It is merged onto the customer's latest settings,
    and the full result is stored. Each revision is therefore self-contained,
    and history can be restored by copying one row.
    """
    if installation is not None and installation.licence.user_id != user.id:
        raise AccountError("that installation does not belong to this customer")

    values, errors = validate_payload(changes)
    if errors:
        raise AccountError("; ".join(errors))

    merged = {**latest_settings(session, user.id), **values}
    revision = SettingsRevision(
        user_id=user.id,
        installation_id=installation.id if installation is not None else None,
        settings_json=json.dumps(merged, sort_keys=True),
        changed_by=(changed_by or user).id,
        note=(note or "")[:255],
        created_at=_now(),
    )
    session.add(revision)
    session.flush()
    return revision
