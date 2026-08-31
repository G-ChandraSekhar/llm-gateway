"""rename budget columns from cents to micros

Cents rounded every realistic small request to 0 (a typical gpt-4o-mini
call costs a few thousandths of a cent), making spend tracking useless.
Micros (millionths of a dollar) give six decimal places of integer
precision instead. This assumes no real spend data exists yet under the
old cents semantics — fine for this project's stage, would need a real
data-migration (multiply existing values by 10000) if run against a
database with live cents-denominated values.

Revision ID: 0002
Revises: 0001
Create Date: 2026-08-31

"""
from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.alter_column("api_keys", "budget_limit_cents", new_column_name="budget_limit_micros")
    op.alter_column("api_keys", "spent_cents", new_column_name="spent_micros")


def downgrade() -> None:
    op.alter_column("api_keys", "budget_limit_micros", new_column_name="budget_limit_cents")
    op.alter_column("api_keys", "spent_micros", new_column_name="spent_cents")
