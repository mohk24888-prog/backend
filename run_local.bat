@echo off
set DATABASE_URL=sqlite+aiosqlite:///./footiq_test.db
set SUPABASE_SERVICE_ROLE_KEY=
set SUPABASE_JWT_SECRET=
set SUPABASE_URL=
set SUPABASE_ANON_KEY=
set REDIS_URL=redis://localhost:6379/0
set CELERY_BROKER_URL=redis://localhost:6379/0
set CELERY_RESULT_BACKEND=redis://localhost:6379/1
set RAW_DIR=./footiq_test_raw
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
