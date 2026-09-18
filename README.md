# FootIQ Backend

FootIQ — Football Intelligence Platform production backend.

## Stack

- **API**: Python 3.11 + FastAPI + Pydantic v2 + SQLAlchemy 2.x
- **Database**: Supabase PostgreSQL (local dev uses Docker PostgreSQL)
- **Auth**: Supabase Auth (email/password, Google OAuth, Apple Sign In)
- **Storage**: Supabase Storage (player photos, videos, reports, thumbnails)
- **Realtime**: Supabase Realtime + Celery task status updates
- **AI**: YOLOv10 tracking, SAM2 segmentation, OCR, metrics calculation
- **Queue**: Celery + Redis for async video analysis
- **Infrastructure**: Docker + Docker Compose

## Quick Start

```bash
# 1. Copy environment
cp .env.example .env

# 2. Start infrastructure
docker compose up -d

# 3. Run migrations (after Supabase is configured)
alembic upgrade head --ini db/migrations/alembic.ini

# 4. Start API
uvicorn api.main:app --reload --port 8000

# 5. Start worker
celery -A workers.celery_app worker --loglevel=info --concurrency=1
```

## Architecture

```
FootIQ Flutter Frontend (Supabase client)
        │
        ▼
Supabase Auth + Storage + Realtime
        │
        ▼
FootIQ FastAPI (JWT validation, business logic)
        │
        ├── PostgreSQL (Supabase) — source of truth
        ├── Redis + Celery — async video jobs
        └── GPU Worker — YOLO/tracking/metrics
```

## Environment Variables

See `.env.example` for all available variables. Never commit real secrets.

## Supabase Configuration

### Google OAuth
1. Go to Supabase Dashboard → Authentication → Providers → Google
2. Add your Google OAuth Client ID and Secret
3. Add authorized redirect URI: `https://<your-project>.supabase.co/auth/v1/callback`

### Apple Sign In
1. Go to Supabase Dashboard → Authentication → Providers → Apple
2. Add your Apple Service ID and Team ID
3. Add the Apple Key ID and private key
4. Configure redirect URIs in Apple Developer portal

## API Endpoints

- `GET /health` — health check
- `GET /health/ready` — readiness check
- `POST /api/v1/auth/register` — register user
- `POST /api/v1/auth/login` — login
- `GET /api/v1/users/me` — current user
- `POST/GET /api/v1/players` — player CRUD
- `POST /api/v1/videos` — upload video
- `POST /api/v1/analyses` — create analysis
- `GET /api/v1/analyses/{id}` — get analysis
- `GET /api/v1/discover/players` — discover players
- `POST /api/v1/watchlists` — create watchlist
- `POST /api/v1/offers` — create offer
- `POST/GET /api/v1/conversations/{id}/messages` — messaging
- `GET /api/v1/notifications` — notifications
- `POST /api/v1/scout-notes` — scout notes
- `GET /api/v1/metrics/{analysis_id}/{type}` — metrics

## GPU Support

The worker automatically detects CUDA:
- `DEVICE=auto` — uses CUDA if available, falls back to CPU
- `DEVICE=cuda` — force CUDA
- `DEVICE=cpu` — force CPU

Model configuration via environment:
- `YOLO_MODEL=yolov10n.pt`
- `CONFIDENCE_THRESHOLD=0.35`
- `TRACKER=bytetrack`
- `FRAME_SKIP=2`

## Deployment to Render

1. Push code to repository
2. Create Render Web Service for API
3. Set environment variables from `.env.example`
4. Add `release_command: alembic upgrade head --ini db/migrations/alembic.ini`
5. For AI worker, use separate Render service or dedicated GPU server

## Testing

```bash
pytest tests/ -v
```

## Logging

Logs go to stdout with structured format. Set `LOG_LEVEL` environment variable.

Never logs passwords, OAuth secrets, or service-role keys.
