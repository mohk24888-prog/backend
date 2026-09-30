-- Drop all tables in public schema
-- Run this in Supabase SQL Editor: https://supabase.com/dashboard/project/ehsthcapklzkomnatuhj/editor

DO $$
DECLARE
    rec RECORD;
BEGIN
    -- Disable RLS on all tables
    FOR rec IN (SELECT tablename FROM pg_tables WHERE schemaname = 'public')
    LOOP
        EXECUTE 'ALTER TABLE public.' || rec.tablename || ' DISABLE ROW LEVEL SECURITY';
    END LOOP;

    -- Drop all tables
    FOR rec IN (SELECT tablename FROM pg_tables WHERE schemaname = 'public')
    LOOP
        EXECUTE 'DROP TABLE IF EXISTS public.' || rec.tablename || ' CASCADE';
    END LOOP;

    -- Drop all custom types
    FOR rec IN (SELECT typname FROM pg_type WHERE typnamespace = (SELECT oid FROM pg_namespace WHERE nspname = 'public') AND typtype = 'e')
    LOOP
        EXECUTE 'DROP TYPE IF EXISTS public.' || rec.typname || ' CASCADE';
    END LOOP;

    -- Drop all sequences
    FOR rec IN (SELECT sequence_name FROM information_schema.sequences WHERE sequence_schema = 'public')
    LOOP
        EXECUTE 'DROP SEQUENCE IF EXISTS public.' || rec.sequence_name || ' CASCADE';
    END LOOP;

    -- Drop all functions
    FOR rec IN (SELECT routine_name FROM information_schema.routines WHERE routine_schema = 'public' AND routine_type = 'FUNCTION')
    LOOP
        EXECUTE 'DROP FUNCTION IF EXISTS public.' || rec.routine_name || '() CASCADE';
    END LOOP;
END $$;

-- Enable extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS pg_trgm;
