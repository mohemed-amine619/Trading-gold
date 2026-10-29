"""
tests/unit/test_accounts.py — users, licences, activations, sessions and the
settings history.

Runs against an in-memory SQLite database. Nothing here touches a broker, the
network, or data/ on disk.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from core.accounts import security
from core.accounts.service import (
    AccountError,
    activate_installation,
    authenticate,
    create_licence,
    create_user,
    issue_session,
    latest_settings,
    record_settings,
    resolve_session,
    revoke_installation,
    revoke_session,
)
from core.database.models import Base, DashboardSession, Installation, Licence, SettingsRevision, User

PASSWORD = "correct horse battery"


@pytest.fixture
def db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()
    engine.dispose()


@pytest.fixture
def customer(db):
    user = create_user(db, "Buyer@Example.com ", PASSWORD)
    db.commit()
    return user


# ─────────────────────────────────────────────────────────────────────────────
# Passwords
# ─────────────────────────────────────────────────────────────────────────────

def test_password_is_never_stored_in_plain_text(db, customer):
    assert PASSWORD not in customer.password_hash
    assert security.verify_password(PASSWORD, customer.password_hash)


def test_wrong_password_does_not_verify(customer):
    assert not security.verify_password("wrong password!!", customer.password_hash)


def test_garbage_stored_hash_fails_closed():
    assert not security.verify_password(PASSWORD, "not-a-hash")
    assert not security.verify_password(PASSWORD, "bcrypt$1$2$3$aa$bb")


def test_short_password_is_refused(db):
    with pytest.raises(ValueError):
        security.hash_password("short")


def test_same_password_gets_different_salts():
    a = security.hash_password(PASSWORD)
    b = security.hash_password(PASSWORD)
    assert a != b


# ─────────────────────────────────────────────────────────────────────────────
# Users and authentication
# ─────────────────────────────────────────────────────────────────────────────

def test_email_is_normalised(customer):
    assert customer.email == "buyer@example.com"


def test_duplicate_email_is_refused_regardless_of_case(db, customer):
    with pytest.raises(AccountError, match="already exists"):
        create_user(db, "BUYER@example.com", PASSWORD)


def test_invalid_email_is_refused(db):
    with pytest.raises(AccountError, match="valid email"):
        create_user(db, "not-an-email", PASSWORD)


def test_unknown_role_is_refused(db):
    with pytest.raises(AccountError, match="role"):
        create_user(db, "a@b.com", PASSWORD, role="superuser")


def test_authenticate_accepts_correct_credentials(db, customer):
    user = authenticate(db, "buyer@example.com", PASSWORD)
    assert user is not None and user.id == customer.id
    assert user.last_login_at is not None


@pytest.mark.parametrize("email, password", [
    ("buyer@example.com", "wrong password!!"),
    ("nobody@example.com", PASSWORD),
    ("bad-email", PASSWORD),
])
def test_authenticate_rejects_bad_credentials(db, customer, email, password):
    assert authenticate(db, email, password) is None


def test_inactive_account_cannot_sign_in(db, customer):
    customer.is_active = False
    db.commit()
    assert authenticate(db, "buyer@example.com", PASSWORD) is None


# ─────────────────────────────────────────────────────────────────────────────
# Dashboard sessions
# ─────────────────────────────────────────────────────────────────────────────

def test_session_resolves_to_its_user(db, customer):
    token = issue_session(db, customer)
    assert resolve_session(db, token).id == customer.id


def test_raw_session_token_is_not_stored(db, customer):
    token = issue_session(db, customer)
    rows = db.scalars(select(DashboardSession)).all()
    assert all(row.token_hash != token for row in rows)
    assert rows[0].token_hash == security.hash_secret(token)


def test_expired_session_is_rejected(db, customer):
    token = issue_session(db, customer, ttl=timedelta(seconds=-1))
    assert resolve_session(db, token) is None


def test_revoked_session_is_rejected(db, customer):
    token = issue_session(db, customer)
    assert revoke_session(db, token) is True
    assert resolve_session(db, token) is None


def test_session_of_deactivated_user_is_rejected(db, customer):
    token = issue_session(db, customer)
    customer.is_active = False
    assert resolve_session(db, token) is None


def test_missing_or_unknown_token_is_rejected(db, customer):
    issue_session(db, customer)
    assert resolve_session(db, None) is None
    assert resolve_session(db, "") is None
    assert resolve_session(db, "made-up-token") is None


# ─────────────────────────────────────────────────────────────────────────────
# Licences
# ─────────────────────────────────────────────────────────────────────────────

def test_licence_key_is_prefixed_and_stored_only_as_a_hash(db, customer):
    licence, raw = create_licence(db, customer)
    db.commit()
    assert raw.startswith("GLD-")
    assert licence.key_hash == security.hash_secret(raw)
    assert raw not in licence.key_hash


def test_zero_seat_licence_is_refused(db, customer):
    with pytest.raises(AccountError):
        create_licence(db, customer, max_installations=0)


def test_activation_with_unknown_key_is_refused(db):
    with pytest.raises(AccountError, match="unknown licence key"):
        activate_installation(db, "GLD-nope", "machine-1")


def test_activation_requires_a_machine_id(db, customer):
    _, raw = create_licence(db, customer)
    with pytest.raises(AccountError, match="machine identifier"):
        activate_installation(db, raw, "")


def test_revoked_licence_cannot_activate(db, customer):
    licence, raw = create_licence(db, customer)
    licence.status = "revoked"
    with pytest.raises(AccountError, match="revoked"):
        activate_installation(db, raw, "machine-1")


def test_expired_licence_cannot_activate(db, customer):
    now = datetime(2026, 10, 29, tzinfo=timezone.utc)
    _, raw = create_licence(db, customer, expires_at=now - timedelta(days=1))
    with pytest.raises(AccountError, match="expired"):
        activate_installation(db, raw, "machine-1", now=now)


def test_seat_limit_is_enforced(db, customer):
    _, raw = create_licence(db, customer, max_installations=2)
    activate_installation(db, raw, "pc-a")
    activate_installation(db, raw, "pc-b")
    with pytest.raises(AccountError, match="2 machine"):
        activate_installation(db, raw, "pc-c")


def test_reactivating_the_same_machine_does_not_use_a_second_seat(db, customer):
    _, raw = create_licence(db, customer, max_installations=1)
    first = activate_installation(db, raw, "pc-a", label="office")
    again = activate_installation(db, raw, "pc-a")
    assert again.id == first.id


def test_machine_identifier_is_stored_hashed(db, customer):
    _, raw = create_licence(db, customer)
    installation = activate_installation(db, raw, "SERIAL-123")
    assert installation.fingerprint_hash == security.hash_secret("SERIAL-123")
    assert "SERIAL-123" not in installation.fingerprint_hash


def test_revoked_machine_cannot_reactivate_and_frees_no_seat(db, customer):
    _, raw = create_licence(db, customer, max_installations=1)
    installation = activate_installation(db, raw, "pc-a")
    revoke_installation(db, installation)
    with pytest.raises(AccountError, match="removed"):
        activate_installation(db, raw, "pc-a")


def test_revoking_a_machine_frees_its_seat(db, customer):
    _, raw = create_licence(db, customer, max_installations=1)
    installation = activate_installation(db, raw, "pc-a")
    revoke_installation(db, installation)
    replacement = activate_installation(db, raw, "pc-b")
    assert replacement.revoked is False


# ─────────────────────────────────────────────────────────────────────────────
# Settings history
# ─────────────────────────────────────────────────────────────────────────────

def test_settings_are_validated_against_the_schema(db, customer):
    with pytest.raises(AccountError):
        record_settings(db, customer, {"risk_per_trade": 0.9})  # far above the 5% ceiling


def test_unknown_setting_is_refused(db, customer):
    with pytest.raises(AccountError, match="unknown parameter"):
        record_settings(db, customer, {"not_a_real_key": 1})


def test_partial_change_is_merged_onto_the_latest_settings(db, customer):
    record_settings(db, customer, {"risk_per_trade": 0.01}, note="first")
    record_settings(db, customer, {"max_open_pos": 2}, note="second")
    settings = latest_settings(db, customer.id)
    assert settings["risk_per_trade"] == pytest.approx(0.01)
    assert settings["max_open_pos"] == 2


def test_history_is_append_only(db, customer):
    record_settings(db, customer, {"risk_per_trade": 0.01})
    record_settings(db, customer, {"risk_per_trade": 0.02})
    rows = db.scalars(select(SettingsRevision).order_by(SettingsRevision.id)).all()
    assert len(rows) == 2
    assert "0.01" in rows[0].settings_json and "0.02" in rows[1].settings_json


def test_installation_must_belong_to_the_customer(db, customer):
    other = create_user(db, "other@example.com", PASSWORD)
    _, raw = create_licence(db, other)
    their_machine = activate_installation(db, raw, "their-pc")
    with pytest.raises(AccountError, match="does not belong"):
        record_settings(db, customer, {"risk_per_trade": 0.01}, installation=their_machine)


def test_revision_records_who_made_the_change(db, customer):
    admin = create_user(db, "support@example.com", PASSWORD, role="admin")
    revision = record_settings(db, customer, {"risk_per_trade": 0.01}, changed_by=admin)
    assert revision.changed_by == admin.id
    assert revision.user_id == customer.id


def test_user_table_has_no_plain_password_column():
    assert "password" not in {c.name for c in User.__table__.columns}
    assert "password_hash" in {c.name for c in User.__table__.columns}
    assert "key" not in {c.name for c in Licence.__table__.columns}
    assert "key_hash" in {c.name for c in Licence.__table__.columns}
    assert "token" not in {c.name for c in DashboardSession.__table__.columns}
    assert "token_hash" in {c.name for c in DashboardSession.__table__.columns}
    assert "fingerprint_hash" in {c.name for c in Installation.__table__.columns}


# ─────────────────────────────────────────────────────────────────────────────
# Administrator CLI
# ─────────────────────────────────────────────────────────────────────────────

import io  # noqa: E402

from core.accounts import cli  # noqa: E402
from core.accounts.service import find_licence  # noqa: E402


def _run_cli(db_session_factory, argv, password=PASSWORD):
    out = io.StringIO()
    code = cli.run(argv, db_session_factory, password_source=lambda: password, out=out)
    return code, out.getvalue()


@pytest.fixture
def factory():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    maker = sessionmaker(bind=engine)
    yield maker
    engine.dispose()


def test_cli_creates_an_owner(factory):
    code, out = _run_cli(factory, ["create-owner", "--email", "owner@example.com"])
    assert code == 0
    assert "created owner owner@example.com" in out
    s = factory()
    assert s.scalar(select(User).where(User.role == "owner")) is not None
    s.close()


def test_cli_issues_a_working_licence_key(factory):
    _run_cli(factory, ["create-customer", "--email", "buyer@example.com"])
    code, out = _run_cli(factory, ["issue-licence", "--email", "buyer@example.com",
                                   "--seats", "2", "--days", "30"])
    assert code == 0
    key = next(line.split(": ", 1)[1] for line in out.splitlines() if line.startswith("key "))
    assert key.startswith("GLD-")
    s = factory()
    licence = find_licence(s, key)
    assert licence is not None and licence.max_installations == 2
    assert licence.expires_at is not None
    s.close()


def test_cli_refuses_a_licence_for_an_unknown_customer(factory):
    code, out = _run_cli(factory, ["issue-licence", "--email", "nobody@example.com"])
    assert code == 1


def test_cli_refuses_a_short_password_without_creating_the_user(factory, capsys):
    code, _ = _run_cli(factory, ["create-customer", "--email", "a@example.com"],
                       password="short")
    assert code == 1
    assert "at least" in capsys.readouterr().err
    s = factory()
    assert s.scalar(select(User)) is None
    s.close()


def test_cli_lists_users_without_secrets(factory):
    _run_cli(factory, ["create-owner", "--email", "owner@example.com"])
    code, out = _run_cli(factory, ["list-users"])
    assert code == 0
    assert "owner@example.com\towner\tactive" in out
    assert "scrypt$" not in out
