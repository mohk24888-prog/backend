from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, ConfigDict, Field


class TimestampedModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    created_at: datetime
    updated_at: datetime


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class TokenPayload(BaseModel):
    sub: str
    exp: datetime
    iat: datetime


class LoginRequest(BaseModel):
    email: str
    password: str


class User(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    email: str
    hashed_password: Optional[str] = None
    role: str = "player"
    is_active: bool = True
    is_superuser: bool = False
    last_login_at: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime


class RegisterRequest(BaseModel):
    email: str
    password: str
    role: str = "player"
    full_name: Optional[str] = None


# ---------------------------------------------------------------------------
# Users / Profiles
# ---------------------------------------------------------------------------

class ProfileCreate(BaseModel):
    full_name: Optional[str] = None
    phone: Optional[str] = None
    bio: Optional[str] = None


class ProfileUpdate(BaseModel):
    full_name: Optional[str] = None
    phone: Optional[str] = None
    avatar_url: Optional[str] = None
    bio: Optional[str] = None


class ProfileRead(TimestampedModel):
    user_id: uuid.UUID
    full_name: Optional[str] = None
    phone: Optional[str] = None
    avatar_url: Optional[str] = None
    bio: Optional[str] = None


# ---------------------------------------------------------------------------
# Players
# ---------------------------------------------------------------------------

class PlayerCreate(BaseModel):
    first_name: str
    last_name: str
    date_of_birth: Optional[datetime] = None
    nationality: Optional[str] = None
    position: str
    secondary_positions: Optional[list[str]] = None
    preferred_foot: Optional[str] = None
    height_cm: Optional[float] = None
    weight_kg: Optional[float] = None
    academy: Optional[str] = None
    bio: Optional[str] = None
    is_public: bool = False


class PlayerUpdate(BaseModel):
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    date_of_birth: Optional[datetime] = None
    nationality: Optional[str] = None
    position: Optional[str] = None
    secondary_positions: Optional[list[str]] = None
    preferred_foot: Optional[str] = None
    height_cm: Optional[float] = None
    weight_kg: Optional[float] = None
    academy: Optional[str] = None
    rating: Optional[float] = None
    technical_rating: Optional[float] = None
    tactical_rating: Optional[float] = None
    physical_rating: Optional[float] = None
    mental_rating: Optional[float] = None
    strengths: Optional[list[str]] = None
    development_areas: Optional[list[str]] = None
    bio: Optional[str] = None
    profile_image_url: Optional[str] = None
    is_public: Optional[bool] = None


class PlayerRead(TimestampedModel):
    user_id: Optional[uuid.UUID] = None
    organization_id: Optional[uuid.UUID] = None
    agent_id: Optional[uuid.UUID] = None
    first_name: str
    last_name: str
    date_of_birth: Optional[datetime] = None
    nationality: Optional[str] = None
    position: str
    secondary_positions: Optional[list[str]] = None
    preferred_foot: Optional[str] = None
    height_cm: Optional[float] = None
    weight_kg: Optional[float] = None
    academy: Optional[str] = None
    rating: Optional[float] = None
    technical_rating: Optional[float] = None
    tactical_rating: Optional[float] = None
    physical_rating: Optional[float] = None
    mental_rating: Optional[float] = None
    verification_status: str
    profile_image_url: Optional[str] = None
    strengths: Optional[list[str]] = None
    development_areas: Optional[list[str]] = None
    bio: Optional[str] = None
    is_public: bool = False


# ---------------------------------------------------------------------------
# Academies / Agents / Clubs
# ---------------------------------------------------------------------------


class AcademyRead(TimestampedModel):
    organization_id: Optional[uuid.UUID] = None
    name: str
    country: Optional[str] = None
    description: Optional[str] = None


class AgentRead(TimestampedModel):
    user_id: Optional[uuid.UUID] = None
    organization_id: Optional[uuid.UUID] = None
    agency_name: str
    bio: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None


class ClubRead(TimestampedModel):
    organization_id: Optional[uuid.UUID] = None
    name: str
    country: Optional[str] = None


# ---------------------------------------------------------------------------
# Videos
# ---------------------------------------------------------------------------

class VideoUploadCreate(BaseModel):
    filename: str
    mime_type: Optional[str] = None
    size_bytes: Optional[int] = None
    duration_seconds: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    match_name: Optional[str] = None


class VideoUploadRead(TimestampedModel):
    id: uuid.UUID
    player_id: uuid.UUID
    uploaded_by: Optional[uuid.UUID] = None
    filename: str
    storage_path: Optional[str] = None
    mime_type: Optional[str] = None
    size_bytes: Optional[int] = None
    duration_seconds: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    match_name: Optional[str] = None


# ---------------------------------------------------------------------------
# Analyses
# ---------------------------------------------------------------------------

class AnalysisCreate(BaseModel):
    video_id: uuid.UUID
    player_id: uuid.UUID
    match_name: Optional[str] = None


class AnalysisRead(TimestampedModel):
    id: uuid.UUID
    job_id: Optional[uuid.UUID] = None
    player_id: uuid.UUID
    video_id: Optional[uuid.UUID] = None
    match_name: Optional[str] = None
    date: Optional[datetime] = None
    overall_rating: Optional[float] = None
    technical: Optional[float] = None
    tactical: Optional[float] = None
    physical: Optional[float] = None
    mental: Optional[float] = None
    summary: Optional[str] = None
    strengths: Optional[list[str]] = None
    development_areas: Optional[list[str]] = None


class AnalysisJobRead(TimestampedModel):
    id: uuid.UUID
    video_id: uuid.UUID
    player_id: uuid.UUID
    status: str
    error: Optional[str] = None
    worker: Optional[str] = None
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None


# ---------------------------------------------------------------------------
# Heatmaps / Actions / Metrics
# ---------------------------------------------------------------------------

class HeatmapPointCreate(BaseModel):
    x: float
    y: float
    intensity: float
    frame_number: Optional[int] = None


class HeatmapPointRead(TimestampedModel):
    id: uuid.UUID
    analysis_id: uuid.UUID
    player_id: uuid.UUID
    x: float
    y: float
    intensity: float
    frame_number: Optional[int] = None


class MatchActionCreate(BaseModel):
    minute: Optional[int] = None
    type: str
    description: Optional[str] = None
    category: Optional[str] = None
    success: Optional[bool] = None


class MatchActionRead(TimestampedModel):
    id: uuid.UUID
    analysis_id: uuid.UUID
    player_id: uuid.UUID
    minute: Optional[int] = None
    type: str
    description: Optional[str] = None
    category: Optional[str] = None
    success: Optional[bool] = None


class PhysicalMetricCreate(BaseModel):
    total_distance_m: Optional[float] = None
    max_speed_ms: Optional[float] = None
    avg_speed_ms: Optional[float] = None
    sprint_count: Optional[int] = None
    hi_run_count: Optional[int] = None
    acceleration_profile: Optional[dict] = None
    speed_zones: Optional[dict] = None
    fatigue_index: Optional[float] = None


class PhysicalMetricRead(TimestampedModel):
    id: uuid.UUID
    analysis_id: uuid.UUID
    player_id: uuid.UUID
    total_distance_m: Optional[float] = None
    max_speed_ms: Optional[float] = None
    avg_speed_ms: Optional[float] = None
    sprint_count: Optional[int] = None
    hi_run_count: Optional[int] = None
    acceleration_profile: Optional[dict] = None
    speed_zones: Optional[dict] = None
    fatigue_index: Optional[float] = None


class TechnicalMetricCreate(BaseModel):
    pass_completion_pct: Optional[float] = None
    dribble_success_pct: Optional[float] = None
    shot_accuracy_pct: Optional[float] = None
    cross_accuracy_pct: Optional[float] = None
    duels_won_pct: Optional[float] = None


class TechnicalMetricRead(TimestampedModel):
    id: uuid.UUID
    analysis_id: uuid.UUID
    player_id: uuid.UUID
    pass_completion_pct: Optional[float] = None
    dribble_success_pct: Optional[float] = None
    shot_accuracy_pct: Optional[float] = None
    cross_accuracy_pct: Optional[float] = None
    duels_won_pct: Optional[float] = None


class TacticalMetricCreate(BaseModel):
    ppda_contribution: Optional[float] = None
    press_count: Optional[int] = None
    press_success_rate: Optional[float] = None
    pitch_control_contribution: Optional[float] = None
    dangerous_zone_occupancy: Optional[float] = None


class TacticalMetricRead(TimestampedModel):
    id: uuid.UUID
    analysis_id: uuid.UUID
    player_id: uuid.UUID
    ppda_contribution: Optional[float] = None
    press_count: Optional[int] = None
    press_success_rate: Optional[float] = None
    pitch_control_contribution: Optional[float] = None
    dangerous_zone_occupancy: Optional[float] = None


# ---------------------------------------------------------------------------
# Offers
# ---------------------------------------------------------------------------

class OfferCreate(BaseModel):
    player_id: uuid.UUID
    club_id: Optional[uuid.UUID] = None
    agent_id: Optional[uuid.UUID] = None
    type: str = Field(..., pattern="^(trial|contract|loan|transfer)$")
    message: Optional[str] = None
    contract_length_months: Optional[int] = None


class OfferRead(TimestampedModel):
    id: uuid.UUID
    player_id: uuid.UUID
    club_id: Optional[uuid.UUID] = None
    agent_id: Optional[uuid.UUID] = None
    sender_user_id: Optional[uuid.UUID] = None
    status: str
    type: str
    message: Optional[str] = None
    contract_length_months: Optional[int] = None
    date: datetime
    updated_at: datetime


# ---------------------------------------------------------------------------
# Watchlists / Notes
# ---------------------------------------------------------------------------

class WatchlistCreate(BaseModel):
    name: str
    description: Optional[str] = None
    is_public: bool = False


class WatchlistRead(TimestampedModel):
    id: uuid.UUID
    owner_user_id: uuid.UUID
    name: str
    description: Optional[str] = None
    is_public: bool = False


class WatchlistItemCreate(BaseModel):
    watchlist_id: uuid.UUID
    player_id: uuid.UUID
    note: Optional[str] = None


class WatchlistItemRead(TimestampedModel):
    id: uuid.UUID
    watchlist_id: uuid.UUID
    player_id: uuid.UUID
    note: Optional[str] = None


class ScoutNoteCreate(BaseModel):
    player_id: Optional[uuid.UUID] = None
    content: str
    is_private: bool = True


class ScoutNoteRead(TimestampedModel):
    id: uuid.UUID
    author_user_id: uuid.UUID
    player_id: Optional[uuid.UUID] = None
    content: str
    is_private: bool = True


# ---------------------------------------------------------------------------
# Messaging
# ---------------------------------------------------------------------------

class ConversationRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    created_at: datetime
    updated_at: datetime


class MessageCreate(BaseModel):
    content: str
    attachment_url: Optional[str] = None


class MessageRead(TimestampedModel):
    id: uuid.UUID
    conversation_id: uuid.UUID
    sender_id: uuid.UUID
    content: str
    attachment_url: Optional[str] = None
    is_read: bool = False
    created_at: datetime


# ---------------------------------------------------------------------------
# Notifications
# ---------------------------------------------------------------------------

class NotificationCreate(BaseModel):
    title: str
    body: str
    type: Optional[str] = None


class NotificationRead(TimestampedModel):
    id: uuid.UUID
    user_id: uuid.UUID
    title: str
    body: str
    type: Optional[str] = None
    read: bool = False
    created_at: datetime


# ---------------------------------------------------------------------------
# Test Sessions & Testing
# ---------------------------------------------------------------------------


class TestType(str, Enum):
    sprint = "SPRINT"
    agility = "AGILITY"
    dribbling = "DRIBBLING"
    ball_control = "BALL_CONTROL"
    passing = "PASSING"
    shooting = "SHOOTING"
    endurance = "ENDURANCE"


class TestSessionCreate(BaseModel):
    player_id: uuid.UUID
    session_date: datetime
    location: Optional[str] = None
    notes: Optional[str] = None


class TestSessionRead(TimestampedModel):
    player_id: uuid.UUID
    session_date: datetime
    location: Optional[str] = None
    notes: Optional[str] = None
    status: str


class TestSessionTestTypeCreate(BaseModel):
    test_type: TestType
    config: Optional[dict] = None


class TestSessionTestTypeRead(TimestampedModel):
    id: uuid.UUID
    test_session_id: uuid.UUID
    test_type: TestType
    config: Optional[dict] = None
    status: str


class CalibrationConfigCreate(BaseModel):
    player_id: uuid.UUID
    camera_position: Optional[dict] = None
    camera_height_m: Optional[float] = None
    field_coordinates: Optional[dict] = None
    known_distances: Optional[dict] = None
    pitch_dimensions: Optional[dict] = None
    test_markers: Optional[dict] = None
    start_point: Optional[dict] = None
    finish_point: Optional[dict] = None
    target_positions: Optional[dict] = None
    calibration_status: Optional[str] = None


class CalibrationConfigRead(TimestampedModel):
    id: uuid.UUID
    player_id: uuid.UUID
    camera_position: Optional[dict] = None
    camera_height_m: Optional[float] = None
    field_coordinates: Optional[dict] = None
    known_distances: Optional[dict] = None
    pitch_dimensions: Optional[dict] = None
    test_markers: Optional[dict] = None
    start_point: Optional[dict] = None
    finish_point: Optional[dict] = None
    target_positions: Optional[dict] = None
    calibration_status: str


class FootIQPlayerIdCreate(BaseModel):
    player_id: uuid.UUID


class FootIQPlayerIdRead(TimestampedModel):
    id: uuid.UUID
    player_id: uuid.UUID
    footiq_id: str
    issued_at: Optional[datetime] = None
    status: str


class AnalysisTestMappingRead(TimestampedModel):
    id: uuid.UUID
    analysis_id: uuid.UUID
    test_session_id: Optional[uuid.UUID] = None
    test_type: Optional[TestType] = None
    player_id: uuid.UUID
    status: str
    metadata: Optional[dict] = None
