-- FootIQ Supabase Schema
-- Run this in the Supabase SQL editor.
-- No mock data is seeded. Only technical/system extensions are created.

-- Extensions
create extension if not exists "uuid-ossp";
create extension if not exists pgcrypto;

-- Enums
create type user_role as enum ('player', 'owner', 'club', 'agent', 'academy');
create type offer_status as enum ('draft', 'sent', 'viewed', 'negotiating', 'accepted', 'rejected', 'withdrawn');
create type analysis_status as enum ('uploaded', 'queued', 'preprocessing', 'detecting', 'tracking', 'calibrating', 'calculating', 'generating_report', 'completed', 'failed', 'cancelled');
create type verification_status as enum ('unverified', 'pending', 'verified', 'rejected');

-- Helper: updated_at trigger
create or replace function set_updated_at()
returns trigger as $$
begin
  new.updated_at = now();
  return new;
end;
$$ language plpgsql;

-- Users
create table users (
  id uuid primary key default uuid_generate_v4(),
  email text unique not null,
  hashed_password text,
  role user_role not null default 'player',
  is_active boolean not null default true,
  is_superuser boolean not null default false,
  last_login_at timestamptz,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create trigger trg_users_updated_at
  before update on users for each row execute function set_updated_at();

-- Profiles
create table profiles (
  id uuid primary key default uuid_generate_v4(),
  user_id uuid unique not null references users(id) on delete cascade,
  full_name text,
  phone text,
  avatar_url text,
  bio text,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create trigger trg_profiles_updated_at
  before update on profiles for each row execute function set_updated_at();

-- Organizations
create table organizations (
  id uuid primary key default uuid_generate_v4(),
  name text not null,
  type text not null,
  country text,
  city text,
  logo_url text,
  website text,
  contact_email text,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create trigger trg_organizations_updated_at
  before update on organizations for each row execute function set_updated_at();

-- Organization Members
create table organization_members (
  id uuid primary key default uuid_generate_v4(),
  organization_id uuid not null references organizations(id) on delete cascade,
  user_id uuid not null references users(id) on delete cascade,
  role text not null default 'member',
  created_at timestamptz not null default now(),
  unique(organization_id, user_id)
);

-- Players
create table players (
  id uuid primary key default uuid_generate_v4(),
  user_id uuid references users(id) on delete set null,
  organization_id uuid references organizations(id) on delete set null,
  agent_id uuid references agents(id) on delete set null,
  first_name text not null,
  last_name text not null,
  date_of_birth timestamptz,
  nationality text,
  position text not null,
  secondary_positions jsonb,
  preferred_foot text,
  height_cm float8,
  weight_kg float8,
  academy text,
  rating float8,
  technical_rating float8,
  tactical_rating float8,
  physical_rating float8,
  mental_rating float8,
  verification_status verification_status not null default 'unverified',
  profile_image_url text,
  strengths jsonb,
  development_areas jsonb,
  bio text,
  is_public boolean not null default false,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create trigger trg_players_updated_at
  before update on players for each row execute function set_updated_at();

-- Agents
create table agents (
  id uuid primary key default uuid_generate_v4(),
  user_id uuid references users(id) on delete set null,
  organization_id uuid references organizations(id) on delete set null,
  agency_name text not null,
  bio text,
  email text,
  phone text,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create trigger trg_agents_updated_at
  before update on agents for each row execute function set_updated_at();

-- Academies
create table academies (
  id uuid primary key default uuid_generate_v4(),
  organization_id uuid references organizations(id) on delete set null,
  name text not null,
  country text,
  description text,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create trigger trg_academies_updated_at
  before update on academies for each row execute function set_updated_at();

-- Clubs
create table clubs (
  id uuid primary key default uuid_generate_v4(),
  organization_id uuid references organizations(id) on delete set null,
  name text not null,
  country text,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create trigger trg_clubs_updated_at
  before update on clubs for each row execute function set_updated_at();

-- Video Uploads
create table video_uploads (
  id uuid primary key default uuid_generate_v4(),
  player_id uuid not null references players(id) on delete cascade,
  uploaded_by uuid references users(id) on delete set null,
  filename text not null,
  original_path text,
  storage_path text,
  mime_type text,
  size_bytes bigint,
  duration_seconds float8,
  width int,
  height int,
  match_name text,
  created_at timestamptz not null default now()
);

-- Analysis Jobs
create table analysis_jobs (
  id uuid primary key default uuid_generate_v4(),
  video_id uuid not null references video_uploads(id) on delete cascade,
  player_id uuid not null references players(id) on delete cascade,
  status analysis_status not null default 'queued',
  error text,
  worker text,
  started_at timestamptz,
  finished_at timestamptz,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create trigger trg_analysis_jobs_updated_at
  before update on analysis_jobs for each row execute function set_updated_at();

-- Analyses
create table analyses (
  id uuid primary key default uuid_generate_v4(),
  job_id uuid references analysis_jobs(id) on delete set null,
  player_id uuid not null references players(id) on delete cascade,
  match_name text,
  date timestamptz,
  overall_rating float8,
  technical float8,
  tactical float8,
  physical float8,
  mental float8,
  summary text,
  strengths jsonb,
  development_areas jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create trigger trg_analyses_updated_at
  before update on analyses for each row execute function set_updated_at();

-- Heatmap Points
create table heatmap_points (
  id uuid primary key default uuid_generate_v4(),
  analysis_id uuid not null references analyses(id) on delete cascade,
  player_id uuid not null references players(id) on delete cascade,
  x float8 not null,
  y float8 not null,
  intensity float8 not null,
  frame_number int,
  created_at timestamptz not null default now()
);

-- Match Actions
create table match_actions (
  id uuid primary key default uuid_generate_v4(),
  analysis_id uuid not null references analyses(id) on delete cascade,
  player_id uuid not null references players(id) on delete cascade,
  minute int,
  type text not null,
  description text,
  category text,
  success boolean,
  created_at timestamptz not null default now()
);

-- Metrics
create table physical_metrics (
  id uuid primary key default uuid_generate_v4(),
  analysis_id uuid not null references analyses(id) on delete cascade,
  player_id uuid not null references players(id) on delete cascade,
  total_distance_m float8,
  max_speed_ms float8,
  avg_speed_ms float8,
  sprint_count int,
  hi_run_count int,
  acceleration_profile jsonb,
  speed_zones jsonb,
  fatigue_index float8,
  created_at timestamptz not null default now()
);

create table technical_metrics (
  id uuid primary key default uuid_generate_v4(),
  analysis_id uuid not null references analyses(id) on delete cascade,
  player_id uuid not null references players(id) on delete cascade,
  pass_completion_pct float8,
  dribble_success_pct float8,
  shot_accuracy_pct float8,
  cross_accuracy_pct float8,
  duels_won_pct float8,
  created_at timestamptz not null default now()
);

create table tactical_metrics (
  id uuid primary key default uuid_generate_v4(),
  analysis_id uuid not null references analyses(id) on delete cascade,
  player_id uuid not null references players(id) on delete cascade,
  ppda_contribution float8,
  press_count int,
  press_success_rate float8,
  pitch_control_contribution float8,
  dangerous_zone_occupancy float8,
  created_at timestamptz not null default now()
);

-- Scouting
create table watchlists (
  id uuid primary key default uuid_generate_v4(),
  owner_user_id uuid not null references users(id) on delete cascade,
  name text not null,
  description text,
  is_public boolean not null default false,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table watchlist_items (
  id uuid primary key default uuid_generate_v4(),
  watchlist_id uuid not null references watchlists(id) on delete cascade,
  player_id uuid not null references players(id) on delete cascade,
  note text,
  created_at timestamptz not null default now(),
  unique(watchlist_id, player_id)
);

create table scout_notes (
  id uuid primary key default uuid_generate_v4(),
  author_user_id uuid not null references users(id) on delete cascade,
  player_id uuid references players(id) on delete set null,
  content text not null,
  is_private boolean not null default true,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

-- Offers
create table offers (
  id uuid primary key default uuid_generate_v4(),
  player_id uuid not null references players(id) on delete cascade,
  club_id uuid references clubs(id) on delete set null,
  agent_id uuid references agents(id) on delete set null,
  sender_user_id uuid references users(id) on delete set null,
  status offer_status not null default 'draft',
  type text not null,
  message text,
  contract_length_months int,
  date timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create trigger trg_offers_updated_at
  before update on offers for each row execute function set_updated_at();

-- Messaging
create table conversations (
  id uuid primary key default uuid_generate_v4(),
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table conversation_participants (
  id uuid primary key default uuid_generate_v4(),
  conversation_id uuid not null references conversations(id) on delete cascade,
  user_id uuid not null references users(id) on delete cascade,
  created_at timestamptz not null default now(),
  unique(conversation_id, user_id)
);

create table messages (
  id uuid primary key default uuid_generate_v4(),
  conversation_id uuid not null references conversations(id) on delete cascade,
  sender_id uuid not null references users(id) on delete cascade,
  content text not null,
  attachment_url text,
  is_read boolean not null default false,
  created_at timestamptz not null default now()
);

-- Notifications
create table notifications (
  id uuid primary key default uuid_generate_v4(),
  user_id uuid not null references users(id) on delete cascade,
  title text not null,
  body text not null,
  type text,
  read boolean not null default false,
  created_at timestamptz not null default now()
);

-- Indexes
create index idx_players_user_id on players(user_id);
create index idx_players_organization_id on players(organization_id);
create index idx_players_agent_id on players(agent_id);
create index idx_players_position on players(position);
create index idx_video_uploads_player_id on video_uploads(player_id);
create index idx_analysis_jobs_player_id on analysis_jobs(player_id);
create index idx_analysis_jobs_status on analysis_jobs(status);
create index idx_analyses_player_id on analyses(player_id);
create index idx_heatmap_points_analysis_id on heatmap_points(analysis_id);
create index idx_match_actions_analysis_id on match_actions(analysis_id);
create index idx_offers_player_id on offers(player_id);
create index idx_offers_status on offers(status);
create index idx_messages_conversation_id on messages(conversation_id);
create index idx_notifications_user_id on notifications(user_id);
create index idx_scout_notes_player_id on scout_notes(player_id);
create index idx_watchlist_items_watchlist_id on watchlist_items(watchlist_id);

-- RLS: Enable
alter table users enable row level security;
alter table profiles enable row level security;
alter table organizations enable row level security;
alter table organization_members enable row level security;
alter table players enable row level security;
alter table agents enable row level security;
alter table academies enable row level security;
alter table clubs enable row level security;
alter table video_uploads enable row level security;
alter table analysis_jobs enable row level security;
alter table analyses enable row level security;
alter table heatmap_points enable row level security;
alter table match_actions enable row level security;
alter table physical_metrics enable row level security;
alter table technical_metrics enable row level security;
alter table tactical_metrics enable row level security;
alter table watchlists enable row level security;
alter table watchlist_items enable row level security;
alter table scout_notes enable row level security;
alter table offers enable row level security;
alter table conversations enable row level security;
alter table conversation_participants enable row level security;
alter table messages enable row level security;
alter table notifications enable row level security;

-- RLS Policies
create policy "Users read own profile" on users for select using (auth.uid() = id);
create policy "Users update own profile" on users for update using (auth.uid() = id);
create policy "Users insert own profile" on profiles for insert with check (auth.uid() = user_id);
create policy "Users read own profile" on profiles for select using (auth.uid() = user_id);
create policy "Users update own profile" on profiles for update using (auth.uid() = user_id);
create policy "Users read own conversations" on conversations for select using (exists (select 1 from conversation_participants where conversation_id = conversations.id and user_id = auth.uid()));
create policy "Participants read messages" on messages for select using (auth.uid() = sender_id or exists (select 1 from conversation_participants where conversation_id = messages.conversation_id and user_id = auth.uid()));
create policy "Participants insert messages" on messages for insert with check (auth.uid() = sender_id);
create policy "Users read own notifications" on notifications for select using (auth.uid() = user_id);
create policy "Users update own notifications" on notifications for update using (auth.uid() = user_id);

-- RLS: Public read for public players
create policy "Public players are viewable" on players for select using (is_public = true);

-- RLS: Watchlists
create policy "Users own watchlists" on watchlists for select using (auth.uid() = owner_user_id);
create policy "Users own watchlist items" on watchlist_items for select using (exists (select 1 from watchlists where id = watchlist_id and owner_user_id = auth.uid()));
create policy "Users own watchlist items insert" on watchlist_items for insert with check (exists (select 1 from watchlists where id = watchlist_id and owner_user_id = auth.uid()));

-- RLS: Scout notes
create policy "Users own notes" on scout_notes for select using (auth.uid() = author_user_id);
create policy "Users own notes insert" on scout_notes for insert with check (auth.uid() = author_user_id);

-- Service Role bypass (for backend API)
-- Note: Supabase service role key bypasses RLS automatically.
