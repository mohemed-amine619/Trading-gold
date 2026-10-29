"""users, licences, installations, dashboard sessions, settings history

Revision ID: c3d9e1f0a7b2
Revises: aedbfe3f600c
Create Date: 2026-10-29 09:00:00

Adds only new tables. No existing table is altered, so this is safe to apply
on a database that already runs the bot.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c3d9e1f0a7b2'
down_revision: Union[str, None] = 'aedbfe3f600c'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table('users',
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('email', sa.String(length=254), nullable=False),
        sa.Column('password_hash', sa.String(length=255), nullable=False),
        sa.Column('role', sa.String(length=16), nullable=False),
        sa.Column('is_active', sa.Boolean(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('last_login_at', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_users_email', 'users', ['email'], unique=True)

    op.create_table('licences',
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('user_id', sa.String(length=36), nullable=False),
        sa.Column('key_hash', sa.String(length=64), nullable=False),
        sa.Column('plan', sa.String(length=32), nullable=False),
        sa.Column('max_installations', sa.Integer(), nullable=False),
        sa.Column('status', sa.String(length=16), nullable=False),
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.id']),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_licences_user_id', 'licences', ['user_id'], unique=False)
    op.create_index('ix_licences_key_hash', 'licences', ['key_hash'], unique=True)

    op.create_table('installations',
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('licence_id', sa.String(length=36), nullable=False),
        sa.Column('fingerprint_hash', sa.String(length=64), nullable=False),
        sa.Column('label', sa.String(length=128), nullable=False),
        sa.Column('revoked', sa.Boolean(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('last_seen_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['licence_id'], ['licences.id']),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('licence_id', 'fingerprint_hash'),
    )
    op.create_index('ix_installations_licence_id', 'installations', ['licence_id'], unique=False)

    op.create_table('dashboard_sessions',
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('user_id', sa.String(length=36), nullable=False),
        sa.Column('token_hash', sa.String(length=64), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('revoked', sa.Boolean(), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.id']),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_dashboard_sessions_user_id', 'dashboard_sessions', ['user_id'], unique=False)
    op.create_index('ix_dashboard_sessions_token_hash', 'dashboard_sessions', ['token_hash'], unique=True)

    op.create_table('settings_revisions',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('user_id', sa.String(length=36), nullable=False),
        sa.Column('installation_id', sa.String(length=36), nullable=True),
        sa.Column('settings_json', sa.Text(), nullable=False),
        sa.Column('changed_by', sa.String(length=36), nullable=True),
        sa.Column('note', sa.String(length=255), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(['changed_by'], ['users.id']),
        sa.ForeignKeyConstraint(['installation_id'], ['installations.id']),
        sa.ForeignKeyConstraint(['user_id'], ['users.id']),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_settings_revisions_user_id', 'settings_revisions', ['user_id'], unique=False)


def downgrade() -> None:
    op.drop_index('ix_settings_revisions_user_id', table_name='settings_revisions')
    op.drop_table('settings_revisions')
    op.drop_index('ix_dashboard_sessions_token_hash', table_name='dashboard_sessions')
    op.drop_index('ix_dashboard_sessions_user_id', table_name='dashboard_sessions')
    op.drop_table('dashboard_sessions')
    op.drop_index('ix_installations_licence_id', table_name='installations')
    op.drop_table('installations')
    op.drop_index('ix_licences_key_hash', table_name='licences')
    op.drop_index('ix_licences_user_id', table_name='licences')
    op.drop_table('licences')
    op.drop_index('ix_users_email', table_name='users')
    op.drop_table('users')
