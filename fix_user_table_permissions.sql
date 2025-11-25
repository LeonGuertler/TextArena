-- SQL script to fix permissions for the or_agent_users table
-- Run this in the Supabase SQL Editor
-- IMPORTANT: Run this as the postgres superuser or with admin privileges

-- 0. Grant schema-level permissions (this is often the issue!)
GRANT USAGE ON SCHEMA public TO service_role;
GRANT ALL ON SCHEMA public TO service_role;
GRANT CREATE ON SCHEMA public TO service_role;

-- 1. Disable RLS (if you're using service_role key, RLS can cause issues)
ALTER TABLE IF EXISTS public.or_agent_users DISABLE ROW LEVEL SECURITY;

-- 2. Grant all permissions to service_role on the table
GRANT ALL ON public.or_agent_users TO service_role;
GRANT ALL ON public.or_agent_users TO postgres;

-- 3. Grant sequence permissions (if using SERIAL/identity for index column)
-- Check if sequence exists first, then grant
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_class WHERE relname = 'or_agent_users_index_seq') THEN
        GRANT USAGE, SELECT ON SEQUENCE or_agent_users_index_seq TO service_role;
        GRANT USAGE, SELECT ON SEQUENCE or_agent_users_index_seq TO postgres;
    END IF;
END $$;

-- 4. Verify table exists and show its structure
SELECT 
    table_name,
    column_name,
    data_type,
    is_nullable
FROM information_schema.columns
WHERE table_schema = 'public' 
  AND table_name = 'or_agent_users'
ORDER BY ordinal_position;

-- 5. Check RLS status
SELECT 
    schemaname,
    tablename,
    rowsecurity as rls_enabled
FROM pg_tables
WHERE schemaname = 'public' 
  AND tablename = 'or_agent_users';

