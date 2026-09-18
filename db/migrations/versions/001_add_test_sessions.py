"""add_test_sessions_and_calibration

Revision ID: 001_add_test_sessions
Revises:
Create Date: 2026-09-17 19:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
import uuid

revision = "001_add_test_sessions"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    enum_test_type = sa.Enum("SPRINT", "AGILITY", "DRIBBLING", "BALL_CONTROL", "PASSING", "SHOOTING", "ENDURANCE", name="test_type")
    enum_test_type_mapped = sa.Enum("SPRINT", "AGILITY", "DRIBBLING", "BALL_CONTROL", "PASSING", "SHOOTING", "ENDURANCE", name="test_type_mapped")
    enum_calibration_status = sa.Enum("NOT_CALIBRATED", "PARTIAL", "CALIBRATED", name="calibration_status")

    enum_test_type.create(op.get_bind(), checkfirst=True)
    enum_test_type_mapped.create(op.get_bind(), checkfirst=True)
    enum_calibration_status.create(op.get_bind(), checkfirst=True)

    op.create_table(
        "test_sessions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("uuid_generate_v4()")),
        sa.Column("player_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("players.id", ondelete="CASCADE"), nullable=False),
        sa.Column("session_date", sa.DateTime(), nullable=False),
        sa.Column("location", sa.String(255), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("status", sa.String(50), nullable=False, server_default="active"),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.text("now()"), nullable=False),
    )
    op.create_index(op.f("ix_test_sessions_player_id"), "test_sessions", ["player_id"], unique=False)

    op.create_table(
        "test_session_test_types",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("uuid_generate_v4()")),
        sa.Column("test_session_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("test_sessions.id", ondelete="CASCADE"), nullable=False),
        sa.Column("test_type", enum_test_type, nullable=False),
        sa.Column("config", postgresql.JSONB(), nullable=True),
        sa.Column("status", sa.String(50), nullable=False, server_default="pending"),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.text("now()"), nullable=False),
    )
    op.create_index(op.f("ix_test_session_test_types_test_session_id"), "test_session_test_types", ["test_session_id"], unique=False)

    op.create_table(
        "calibration_configs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("uuid_generate_v4()")),
        sa.Column("player_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("players.id", ondelete="CASCADE"), nullable=False),
        sa.Column("camera_position", postgresql.JSONB(), nullable=True),
        sa.Column("camera_height_m", sa.Float(), nullable=True),
        sa.Column("field_coordinates", postgresql.JSONB(), nullable=True),
        sa.Column("known_distances", postgresql.JSONB(), nullable=True),
        sa.Column("pitch_dimensions", postgresql.JSONB(), nullable=True),
        sa.Column("test_markers", postgresql.JSONB(), nullable=True),
        sa.Column("start_point", postgresql.JSONB(), nullable=True),
        sa.Column("finish_point", postgresql.JSONB(), nullable=True),
        sa.Column("target_positions", postgresql.JSONB(), nullable=True),
        sa.Column("calibration_status", enum_calibration_status, nullable=False, server_default="NOT_CALIBRATED"),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.text("now()"), nullable=False),
    )
    op.create_index(op.f("ix_calibration_configs_player_id"), "calibration_configs", ["player_id"], unique=False)

    op.create_table(
        "analysis_test_mappings",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("uuid_generate_v4()")),
        sa.Column("analysis_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("analyses.id", ondelete="CASCADE"), nullable=False),
        sa.Column("test_session_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("test_sessions.id", ondelete="SET NULL"), nullable=True),
        sa.Column("test_type", enum_test_type_mapped, nullable=True),
        sa.Column("player_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("players.id", ondelete="CASCADE"), nullable=False),
        sa.Column("status", sa.String(50), nullable=False, server_default="pending"),
        sa.Column("metadata", postgresql.JSONB(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("now()"), nullable=False),
    )
    op.create_index(op.f("ix_analysis_test_mappings_analysis_id"), "analysis_test_mappings", ["analysis_id"], unique=False)
    op.create_index(op.f("ix_analysis_test_mappings_test_session_id"), "analysis_test_mappings", ["test_session_id"], unique=False)

    op.create_table(
        "footiq_player_ids",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("uuid_generate_v4()")),
        sa.Column("player_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("players.id", ondelete="CASCADE"), nullable=False),
        sa.Column("footiq_id", sa.String(30), nullable=False),
        sa.Column("issued_at", sa.DateTime(), nullable=True),
        sa.Column("status", sa.String(50), nullable=False, server_default="active"),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("now()"), nullable=False),
    )
    op.create_unique_constraint(op.f("uq_footiq_player_ids_footiq_id"), "footiq_player_ids", ["footiq_id"])
    op.create_index(op.f("ix_footiq_player_ids_player_id"), "footiq_player_ids", ["player_id"], unique=False)


    op.add_column("analysis_jobs", sa.Column("test_session_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("test_sessions.id", ondelete="SET NULL"), nullable=True))
    op.create_index(op.f("ix_analysis_jobs_test_session_id"), "analysis_jobs", ["test_session_id"], unique=False)


def downgrade():
    op.drop_table("footiq_player_ids")
    op.drop_table("analysis_test_mappings")
    op.drop_table("calibration_configs")
    op.drop_table("test_session_test_types")
    op.drop_table("test_sessions")
    op.execute("DROP TYPE IF EXISTS test_type")
    op.execute("DROP TYPE IF EXISTS test_type_mapped")
    op.execute("DROP TYPE IF EXISTS calibration_status")
