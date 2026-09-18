from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    Uuid,
    func,
    Enum as SQLEnum,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

import enum


class Base(DeclarativeBase):
    pass


class UserRole(str, enum.Enum):
    player = "player"
    owner = "owner"
    club = "club"
    agent = "agent"
    academy = "academy"


class OfferStatus(str, enum.Enum):
    draft = "draft"
    sent = "sent"
    viewed = "viewed"
    negotiating = "negotiating"
    accepted = "accepted"
    rejected = "rejected"
    withdrawn = "withdrawn"


class AnalysisStatus(str, enum.Enum):
    uploaded = "uploaded"
    queued = "queued"
    preprocessing = "preprocessing"
    detecting = "detecting"
    tracking = "tracking"
    calibrating = "calibrating"
    calculating = "calculating"
    generating_report = "generating_report"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"


class VerificationStatus(str, enum.Enum):
    unverified = "unverified"
    pending = "pending"
    verified = "verified"
    rejected = "rejected"


# ---------------------------------------------------------------------------
# Users / Profiles
# ---------------------------------------------------------------------------

class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    hashed_password: Mapped[Optional[str]] = mapped_column(String(255))
    role: Mapped[UserRole] = mapped_column(SQLEnum(UserRole, name="user_role"), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    is_superuser: Mapped[bool] = mapped_column(Boolean, default=False)
    last_login_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())

    profile: Mapped[Optional[Profile]] = relationship("Profile", back_populates="user", uselist=False)
    memberships: Mapped[list[OrganizationMember]] = relationship("OrganizationMember", back_populates="user")


class Profile(Base):
    __tablename__ = "profiles"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"), unique=True)
    full_name: Mapped[Optional[str]] = mapped_column(String(200))
    phone: Mapped[Optional[str]] = mapped_column(String(50))
    avatar_url: Mapped[Optional[str]] = mapped_column(String(500))
    bio: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())

    user: Mapped[User] = relationship("User", back_populates="profile")


# ---------------------------------------------------------------------------
# Organizations
# ---------------------------------------------------------------------------

class Organization(Base):
    __tablename__ = "organizations"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    type: Mapped[str] = mapped_column(String(50), nullable=False)  # club | academy | agency | owner_org
    country: Mapped[Optional[str]] = mapped_column(String(100))
    city: Mapped[Optional[str]] = mapped_column(String(100))
    logo_url: Mapped[Optional[str]] = mapped_column(String(500))
    website: Mapped[Optional[str]] = mapped_column(String(255))
    contact_email: Mapped[Optional[str]] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())

    members: Mapped[list[OrganizationMember]] = relationship("OrganizationMember", back_populates="organization")


class OrganizationMember(Base):
    __tablename__ = "organization_members"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    organization_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("organizations.id"))
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"))
    role: Mapped[str] = mapped_column(String(50), default="member")  # owner | admin | member
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    organization: Mapped[Organization] = relationship("Organization", back_populates="members")
    user: Mapped[User] = relationship("User", back_populates="memberships")


# ---------------------------------------------------------------------------
# Football domain
# ---------------------------------------------------------------------------

class Player(Base):
    __tablename__ = "players"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    user_id: Mapped[Optional[uuid.UUID]] = mapped_column(ForeignKey("users.id"), nullable=True)
    organization_id: Mapped[Optional[uuid.UUID]] = mapped_column(ForeignKey("organizations.id"), nullable=True)
    agent_id: Mapped[Optional[uuid.UUID]] = mapped_column(ForeignKey("agents.id"), nullable=True)

    first_name: Mapped[str] = mapped_column(String(100))
    last_name: Mapped[str] = mapped_column(String(100))
    date_of_birth: Mapped[Optional[datetime]] = mapped_column(DateTime)
    nationality: Mapped[Optional[str]] = mapped_column(String(100))
    position: Mapped[str] = mapped_column(String(50))
    secondary_positions: Mapped[Optional[list]] = mapped_column(JSON)
    preferred_foot: Mapped[Optional[str]] = mapped_column(String(10))
    height_cm: Mapped[Optional[float]] = mapped_column(Float)
    weight_kg: Mapped[Optional[float]] = mapped_column(Float)
    academy: Mapped[Optional[str]] = mapped_column(String(200))

    rating: Mapped[Optional[float]] = mapped_column(Float)
    technical_rating: Mapped[Optional[float]] = mapped_column(Float)
    tactical_rating: Mapped[Optional[float]] = mapped_column(Float)
    physical_rating: Mapped[Optional[float]] = mapped_column(Float)
    mental_rating: Mapped[Optional[float]] = mapped_column(Float)

    verification_status: Mapped[VerificationStatus] = mapped_column(
        SQLEnum(VerificationStatus, name="verification_status"), default=VerificationStatus.unverified
    )
    profile_image_url: Mapped[Optional[str]] = mapped_column(String(500))
    strengths: Mapped[Optional[list]] = mapped_column(JSON)
    development_areas: Mapped[Optional[list]] = mapped_column(JSON)
    bio: Mapped[Optional[str]] = mapped_column(Text)
    is_public: Mapped[bool] = mapped_column(Boolean, default=False)

    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())

    agent: Mapped[Optional[Agent]] = relationship("Agent", back_populates="players")
    videos: Mapped[list[VideoUpload]] = relationship("VideoUpload", back_populates="player")
    analyses: Mapped[list[Analysis]] = relationship("Analysis", back_populates="player")
    match_actions: Mapped[list[MatchAction]] = relationship("MatchAction", back_populates="player")


class Agent(Base):
    __tablename__ = "agents"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    user_id: Mapped[Optional[uuid.UUID]] = mapped_column(ForeignKey("users.id"), nullable=True)
    organization_id: Mapped[Optional[uuid.UUID]] = mapped_column(ForeignKey("organizations.id"), nullable=True)
    agency_name: Mapped[str] = mapped_column(String(200))
    bio: Mapped[Optional[str]] = mapped_column(Text)
    email: Mapped[Optional[str]] = mapped_column(String(255))
    phone: Mapped[Optional[str]] = mapped_column(String(50))
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())

    players: Mapped[list[Player]] = relationship("Player", back_populates="agent")


class Academy(Base):
    __tablename__ = "academies"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    organization_id: Mapped[Optional[uuid.UUID]] = mapped_column(ForeignKey("organizations.id"), nullable=True)
    name: Mapped[str] = mapped_column(String(200))
    country: Mapped[Optional[str]] = mapped_column(String(100))
    description: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())


class Club(Base):
    __tablename__ = "clubs"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    organization_id: Mapped[Optional[uuid.UUID]] = mapped_column(ForeignKey("organizations.id"), nullable=True)
    name: Mapped[str] = mapped_column(String(200))
    country: Mapped[Optional[str]] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())


# ---------------------------------------------------------------------------
# Video & Analysis
# ---------------------------------------------------------------------------

class VideoUpload(Base):
    __tablename__ = "video_uploads"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    player_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("players.id"))
    uploaded_by: Mapped[Optional[uuid.UUID]] = mapped_column(ForeignKey("users.id"), nullable=True)
    filename: Mapped[str] = mapped_column(String(255))
    original_path: Mapped[Optional[str]] = mapped_column(String(500))
    storage_path: Mapped[Optional[str]] = mapped_column(String(500))
    mime_type: Mapped[Optional[str]] = mapped_column(String(100))
    size_bytes: Mapped[Optional[int]] = mapped_column(Integer)
    duration_seconds: Mapped[Optional[float]] = mapped_column(Float)
    width: Mapped[Optional[int]] = mapped_column(Integer)
    height: Mapped[Optional[int]] = mapped_column(Integer)
    match_name: Mapped[Optional[str]] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    player: Mapped[Player] = relationship("Player", back_populates="videos")
    analysis_jobs: Mapped[list[AnalysisJob]] = relationship("AnalysisJob", back_populates="video")


class AnalysisJob(Base):
    __tablename__ = "analysis_jobs"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    video_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("video_uploads.id"))
    player_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("players.id"))
    status: Mapped[AnalysisStatus] = mapped_column(
        SQLEnum(AnalysisStatus, name="analysis_status"), default=AnalysisStatus.queued
    )
    error: Mapped[Optional[str]] = mapped_column(Text)
    worker: Mapped[Optional[str]] = mapped_column(String(100))
    test_session_id: Mapped[Optional[uuid.UUID]] = mapped_column(Uuid, ForeignKey("test_sessions.id", ondelete="SET NULL"), nullable=True)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())

    video: Mapped[VideoUpload] = relationship("VideoUpload", back_populates="analysis_jobs")
    analysis: Mapped[Optional[Analysis]] = relationship("Analysis", back_populates="job", uselist=False)
    test_session: Mapped[Optional[TestSession]] = relationship("TestSession")


class Analysis(Base):
    __tablename__ = "analyses"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    job_id: Mapped[Optional[uuid.UUID]] = mapped_column(ForeignKey("analysis_jobs.id"), nullable=True)
    player_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("players.id"))
    match_name: Mapped[Optional[str]] = mapped_column(String(255))
    date: Mapped[Optional[datetime]] = mapped_column(DateTime)

    overall_rating: Mapped[Optional[float]] = mapped_column(Float)
    technical: Mapped[Optional[float]] = mapped_column(Float)
    tactical: Mapped[Optional[float]] = mapped_column(Float)
    physical: Mapped[Optional[float]] = mapped_column(Float)
    mental: Mapped[Optional[float]] = mapped_column(Float)

    summary: Mapped[Optional[str]] = mapped_column(Text)
    strengths: Mapped[Optional[list]] = mapped_column(JSON)
    development_areas: Mapped[Optional[list]] = mapped_column(JSON)

    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())

    job: Mapped[Optional[AnalysisJob]] = relationship("AnalysisJob", back_populates="analysis")
    player: Mapped[Player] = relationship("Player", back_populates="analyses")
    actions: Mapped[list[MatchAction]] = relationship("MatchAction", back_populates="analysis")
    heatmap_points: Mapped[list[HeatmapPoint]] = relationship("HeatmapPoint", back_populates="analysis")


class HeatmapPoint(Base):
    __tablename__ = "heatmap_points"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    analysis_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("analyses.id"))
    player_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("players.id"))
    x: Mapped[float] = mapped_column(Float, nullable=False)
    y: Mapped[float] = mapped_column(Float, nullable=False)
    intensity: Mapped[float] = mapped_column(Float, nullable=False)
    frame_number: Mapped[Optional[int]] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    analysis: Mapped[Analysis] = relationship("Analysis", back_populates="heatmap_points")
    player: Mapped[Player] = relationship("Player")


class MatchAction(Base):
    __tablename__ = "match_actions"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    analysis_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("analyses.id"))
    player_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("players.id"))
    minute: Mapped[Optional[int]] = mapped_column(Integer)
    type: Mapped[str] = mapped_column(String(100))
    description: Mapped[Optional[str]] = mapped_column(String(255))
    category: Mapped[Optional[str]] = mapped_column(String(100))
    success: Mapped[Optional[bool]] = mapped_column(Boolean)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    analysis: Mapped[Analysis] = relationship("Analysis", back_populates="actions")
    player: Mapped[Player] = relationship("Player", back_populates="match_actions")


class PhysicalMetric(Base):
    __tablename__ = "physical_metrics"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    analysis_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("analyses.id"))
    player_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("players.id"))
    total_distance_m: Mapped[Optional[float]] = mapped_column(Float)
    max_speed_ms: Mapped[Optional[float]] = mapped_column(Float)
    avg_speed_ms: Mapped[Optional[float]] = mapped_column(Float)
    sprint_count: Mapped[Optional[int]] = mapped_column(Integer)
    hi_run_count: Mapped[Optional[int]] = mapped_column(Integer)
    acceleration_profile: Mapped[Optional[dict]] = mapped_column(JSON)
    speed_zones: Mapped[Optional[dict]] = mapped_column(JSON)
    fatigue_index: Mapped[Optional[float]] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class TechnicalMetric(Base):
    __tablename__ = "technical_metrics"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    analysis_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("analyses.id"))
    player_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("players.id"))
    pass_completion_pct: Mapped[Optional[float]] = mapped_column(Float)
    dribble_success_pct: Mapped[Optional[float]] = mapped_column(Float)
    shot_accuracy_pct: Mapped[Optional[float]] = mapped_column(Float)
    cross_accuracy_pct: Mapped[Optional[float]] = mapped_column(Float)
    duels_won_pct: Mapped[Optional[float]] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class TacticalMetric(Base):
    __tablename__ = "tactical_metrics"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    analysis_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("analyses.id"))
    player_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("players.id"))
    ppda_contribution: Mapped[Optional[float]] = mapped_column(Float)
    press_count: Mapped[Optional[int]] = mapped_column(Integer)
    press_success_rate: Mapped[Optional[float]] = mapped_column(Float)
    pitch_control_contribution: Mapped[Optional[float]] = mapped_column(Float)
    dangerous_zone_occupancy: Mapped[Optional[float]] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


# ---------------------------------------------------------------------------
# Scouting / Workflow
# ---------------------------------------------------------------------------

class Watchlist(Base):
    __tablename__ = "watchlists"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    owner_user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"))
    name: Mapped[str] = mapped_column(String(200))
    description: Mapped[Optional[str]] = mapped_column(Text)
    is_public: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())


class WatchlistItem(Base):
    __tablename__ = "watchlist_items"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    watchlist_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("watchlists.id"))
    player_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("players.id"))
    note: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class ScoutNote(Base):
    __tablename__ = "scout_notes"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    author_user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"))
    player_id: Mapped[Optional[uuid.UUID]] = mapped_column(ForeignKey("players.id"), nullable=True)
    content: Mapped[str] = mapped_column(Text)
    is_private: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())


# ---------------------------------------------------------------------------
# Offers
# ---------------------------------------------------------------------------

class Offer(Base):
    __tablename__ = "offers"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    player_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("players.id"))
    club_id: Mapped[Optional[uuid.UUID]] = mapped_column(ForeignKey("clubs.id"), nullable=True)
    agent_id: Mapped[Optional[uuid.UUID]] = mapped_column(ForeignKey("agents.id"), nullable=True)
    sender_user_id: Mapped[Optional[uuid.UUID]] = mapped_column(ForeignKey("users.id"), nullable=True)
    status: Mapped[OfferStatus] = mapped_column(
        SQLEnum(OfferStatus, name="offer_status"), default=OfferStatus.draft
    )
    type: Mapped[str] = mapped_column(String(50))  # trial | contract | loan | transfer
    message: Mapped[Optional[str]] = mapped_column(Text)
    contract_length_months: Mapped[Optional[int]] = mapped_column(Integer)
    date: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())


# ---------------------------------------------------------------------------
# Messaging
# ---------------------------------------------------------------------------

class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())


class ConversationParticipant(Base):
    __tablename__ = "conversation_participants"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    conversation_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("conversations.id"))
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    conversation_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("conversations.id"))
    sender_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"))
    content: Mapped[str] = mapped_column(Text)
    attachment_url: Mapped[Optional[str]] = mapped_column(String(500))
    is_read: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


# ---------------------------------------------------------------------------
# Notifications
# ---------------------------------------------------------------------------

class Notification(Base):
    __tablename__ = "notifications"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"))
    title: Mapped[str] = mapped_column(String(200))
    body: Mapped[str] = mapped_column(Text)
    type: Mapped[Optional[str]] = mapped_column(String(50))
    read: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


# ---------------------------------------------------------------------------
# Test Sessions & Standardized Testing
# ---------------------------------------------------------------------------


class TestType(str, enum.Enum):
    sprint = "SPRINT"
    agility = "AGILITY"
    dribbling = "DRIBBLING"
    ball_control = "BALL_CONTROL"
    passing = "PASSING"
    shooting = "SHOOTING"
    endurance = "ENDURANCE"


class AnalysisStatusNew(str, enum.Enum):
    queued = "QUEUED"
    processing = "PROCESSING"
    completed = "COMPLETED"
    failed = "FAILED"
    partial = "PARTIAL"


class CalibrationStatus(str, enum.Enum):
    not_calibrated = "NOT_CALIBRATED"
    partial = "PARTIAL"
    calibrated = "CALIBRATED"


class TestSession(Base):
    __tablename__ = "test_sessions"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    player_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("players.id"), nullable=False)
    session_date: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    location: Mapped[Optional[str]] = mapped_column(String(255))
    notes: Mapped[Optional[str]] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(50), default="active")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())

    player: Mapped[Player] = relationship("Player", back_populates="test_sessions")
    tests: Mapped[list["TestSessionTestType"]] = relationship("TestSessionTestType", back_populates="test_session")
    analyses: Mapped[list["AnalysisTestMapping"]] = relationship("AnalysisTestMapping", back_populates="test_session")


class TestSessionTestType(Base):
    __tablename__ = "test_session_test_types"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    test_session_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("test_sessions.id"), nullable=False)
    test_type: Mapped[TestType] = mapped_column(SQLEnum(TestType, name="test_type"), nullable=False)
    config: Mapped[Optional[dict]] = mapped_column(JSON)
    status: Mapped[str] = mapped_column(String(50), default="pending")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())

    test_session: Mapped[TestSession] = relationship("TestSession", back_populates="tests")


class CalibrationConfig(Base):
    __tablename__ = "calibration_configs"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    player_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("players.id"), nullable=False)
    camera_position: Mapped[Optional[dict]] = mapped_column(JSON)
    camera_height_m: Mapped[Optional[float]] = mapped_column(Float)
    field_coordinates: Mapped[Optional[dict]] = mapped_column(JSON)
    known_distances: Mapped[Optional[dict]] = mapped_column(JSON)
    pitch_dimensions: Mapped[Optional[dict]] = mapped_column(JSON)
    test_markers: Mapped[Optional[dict]] = mapped_column(JSON)
    start_point: Mapped[Optional[dict]] = mapped_column(JSON)
    finish_point: Mapped[Optional[dict]] = mapped_column(JSON)
    target_positions: Mapped[Optional[dict]] = mapped_column(JSON)
    calibration_status: Mapped[CalibrationStatus] = mapped_column(
        SQLEnum(CalibrationStatus, name="calibration_status"),
        default=CalibrationStatus.not_calibrated,
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())

    player: Mapped[Player] = relationship("Player", back_populates="calibration")


class AnalysisTestMapping(Base):
    __tablename__ = "analysis_test_mappings"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    analysis_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("analyses.id"), nullable=False)
    test_session_id: Mapped[Optional[uuid.UUID]] = mapped_column(ForeignKey("test_sessions.id"), nullable=True)
    test_type: Mapped[Optional[TestType]] = mapped_column(SQLEnum(TestType, name="test_type_mapped"), nullable=True)
    player_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("players.id"), nullable=False)
    status: Mapped[str] = mapped_column(String(50), default="pending")
    metadata_: Mapped[Optional[dict]] = mapped_column(JSON, name="metadata")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    analysis: Mapped[Analysis] = relationship("Analysis", back_populates="test_mappings")
    test_session: Mapped[Optional[TestSession]] = relationship("TestSession", back_populates="analyses")
    player: Mapped[Player] = relationship("Player")


class FootIQPlayerId(Base):
    __tablename__ = "footiq_player_ids"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    player_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("players.id"), nullable=False)
    footiq_id: Mapped[str] = mapped_column(String(30), unique=True, nullable=False)
    issued_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    status: Mapped[str] = mapped_column(String(50), default="active")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    player: Mapped[Player] = relationship("Player", back_populates="footiq_ids")


# Add relationships to Player
Player.test_sessions = relationship("TestSession", back_populates="player")
Player.calibration = relationship("CalibrationConfig", back_populates="player")
Player.footiq_ids = relationship("FootIQPlayerId", back_populates="player")
Analysis.test_mappings = relationship("AnalysisTestMapping", back_populates="analysis")
