"""
core/accounts/cli.py — administrator commands for accounts and licences.

Until the web dashboard has an admin screen, this is how the owner creates
accounts and issues licence keys. Run from the repository root:

    python -m core.accounts.cli create-owner --email you@example.com
    python -m core.accounts.cli create-customer --email buyer@example.com
    python -m core.accounts.cli issue-licence --email buyer@example.com --seats 1 --days 365
    python -m core.accounts.cli list-users

Passwords are read with getpass, never from the command line, so they do not
end up in shell history. A licence key is printed once. Only its hash is kept.
"""

from __future__ import annotations

import argparse
import getpass
import sys
from datetime import datetime, timedelta, timezone
from typing import Callable, List, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from core.accounts.service import AccountError, create_licence, create_user
from core.database import engine as db_engine
from core.database.models import User


def _session_factory(database_url: str) -> Callable[[], Session]:
    db_engine.init_db(database_url, create_tables=True)
    # Read the attribute after init_db: it is None until the engine exists.
    return db_engine._SyncSession


def _prompt_password() -> str:
    first = getpass.getpass("Password: ")
    second = getpass.getpass("Repeat password: ")
    if first != second:
        raise AccountError("passwords do not match")
    return first


def _find_user(session: Session, email: str) -> User:
    user = session.scalar(select(User).where(User.email == email.strip().lower()))
    if user is None:
        raise AccountError(f"no account for {email}")
    return user


def run(argv: List[str], make_session: Callable[[], Session],
        password_source: Callable[[], str] = _prompt_password,
        out=sys.stdout) -> int:
    parser = argparse.ArgumentParser(prog="python -m core.accounts.cli")
    sub = parser.add_subparsers(dest="command", required=True)

    for name, role in (("create-owner", "owner"), ("create-customer", "customer")):
        p = sub.add_parser(name, help=f"create a {role} account")
        p.add_argument("--email", required=True)
        p.set_defaults(role=role)

    lic = sub.add_parser("issue-licence", help="issue a licence key to a customer")
    lic.add_argument("--email", required=True)
    lic.add_argument("--seats", type=int, default=1, help="machines allowed to run on this key")
    lic.add_argument("--plan", default="standard")
    lic.add_argument("--days", type=int, default=0, help="validity in days; 0 means no expiry")

    sub.add_parser("list-users", help="list accounts")

    args = parser.parse_args(argv)
    session = make_session()
    try:
        if args.command in ("create-owner", "create-customer"):
            user = create_user(session, args.email, password_source(), role=args.role)
            session.commit()
            print(f"created {user.role} {user.email}", file=out)

        elif args.command == "issue-licence":
            user = _find_user(session, args.email)
            expires = (datetime.now(timezone.utc) + timedelta(days=args.days)
                       if args.days > 0 else None)
            licence, raw_key = create_licence(session, user, plan=args.plan,
                                              max_installations=args.seats,
                                              expires_at=expires)
            session.commit()
            print(f"licence for {user.email}: {args.seats} seat(s), plan {args.plan}", file=out)
            print(f"key (shown once): {raw_key}", file=out)

        elif args.command == "list-users":
            for user in session.scalars(select(User).order_by(User.created_at)):
                state = "active" if user.is_active else "disabled"
                print(f"{user.email}\t{user.role}\t{state}", file=out)
        return 0
    except (AccountError, ValueError) as exc:
        session.rollback()
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        session.close()


def main(argv: Optional[List[str]] = None) -> int:
    import os
    database_url = os.environ.get("DATABASE_URL", "sqlite:///./data/trading.db")
    factory = _session_factory(database_url)
    return run(argv if argv is not None else sys.argv[1:], factory)


if __name__ == "__main__":
    raise SystemExit(main())
