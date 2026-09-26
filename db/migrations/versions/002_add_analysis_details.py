"""add_analysis_overlay_and_simulation_data

Revision ID: 002_add_analysis_details
Revises: 001_add_test_sessions
Create Date: 2026-09-25 13:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "002_add_analysis_details"
down_revision = "001_add_test_sessions"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("analyses", sa.Column("overlay_data", postgresql.JSONB(), nullable=True))
    op.add_column("analyses", sa.Column("simulation_data", postgresql.JSONB(), nullable=True))
    op.add_column("analyses", sa.Column("cv_repo_used", sa.String(100), nullable=True))
    op.add_column("analyses", sa.Column("subject_track_id", sa.Integer(), nullable=True))
    op.add_column("analyses", sa.Column("selection_method", sa.String(50), nullable=True))
    op.add_column("analyses", sa.Column("pipeline_warnings", postgresql.JSONB(), nullable=True))
    op.add_column("analyses", sa.Column("video_url", sa.String(500), nullable=True))
    op.add_column("analyses", sa.Column("analysis_duration_s", sa.Float(), nullable=True))

    op.add_column("video_uploads", sa.Column("public_url", sa.String(500), nullable=True))

    op.create_table(
        "analysis_frames",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("uuid_generate_v4()")),
        sa.Column("analysis_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("analyses.id", ondelete="CASCADE"), nullable=False),
        sa.Column("frame_number", sa.Integer(), nullable=False),
        sa.Column("timestamp_s", sa.Float(), nullable=False),
        sa.Column("player_pixel_x", sa.Float(), nullable=True),
        sa.Column("player_pixel_y", sa.Float(), nullable=True),
        sa.Column("player_bbox", postgresql.JSONB(), nullable=True),
        sa.Column("player_conf", sa.Float(), nullable=True),
        sa.Column("player_track_id", sa.Integer(), nullable=True),
        sa.Column("pitch_x_cm", sa.Float(), nullable=True),
        sa.Column("pitch_y_cm", sa.Float(), nullable=True),
        sa.Column("speed_kmh", sa.Float(), nullable=True),
        sa.Column("ball_pixel_x", sa.Float(), nullable=True),
        sa.Column("ball_pixel_y", sa.Float(), nullable=True),
        sa.Column("possession", sa.Boolean(), nullable=True),
        sa.Column("event", sa.String(100), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("now()"), nullable=False),
    )
    op.create_index(op.f("ix_analysis_frames_analysis_id"), "analysis_frames", ["analysis_id"], unique=False)
    op.create_index(op.f("ix_analysis_frames_frame_number"), "analysis_frames", ["analysis_id", "frame_number"], unique=False)
    op.create_index(op.f("ix_analysis_frames_timestamp"), "analysis_frames", ["analysis_id", "timestamp_s"], unique=False)


def downgrade():
    op.drop_index(op.f("ix_analysis_frames_timestamp"), table_name="analysis_frames")
    op.drop_index(op.f("ix_analysis_frames_frame_number"), table_name="analysis_frames")
    op.drop_index(op.f("ix_analysis_frames_analysis_id"), table_name="analysis_frames")
    op.drop_table("analysis_frames")

    op.drop_column("video_uploads", "public_url")
    op.drop_column("analyses", "analysis_duration_s")
    op.drop_column("analyses", "video_url")
    op.drop_column("analyses", "pipeline_warnings")
    op.drop_column("analyses", "selection_method")
    op.drop_column("analyses", "subject_track_id")
    op.drop_column("analyses", "cv_repo_used")
    op.drop_column("analyses", "simulation_data")
    op.drop_column("analyses", "overlay_data")
