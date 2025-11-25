-- Diagnostic SQL to check table permissions
-- Run this in Supabase SQL Editor to see current state

-- 1. Check if table exists and its structure
SELECT 
    'Table Structure' as check_type,
    table_name,
    column_name,
    data_type,
    is_nullable,
    column_default
FROM information_schema.columns
WHERE table_schema = 'public' 
  AND table_name = 'or_agent_users'
ORDER BY ordinal_position;

-- 2. Check RLS status
SELECT 
    'RLS Status' as check_type,
    schemaname,
    tablename,
    rowsecurity as rls_enabled
FROM pg_tables
WHERE schemaname = 'public' 
  AND tablename = 'or_agent_users';

-- 3. Check existing RLS policies
SELECT 
    'RLS Policies' as check_type,
    schemaname,
    tablename,
    policyname,
    permissive,
    roles,
    cmd,
    qual,
    with_check
FROM pg_policies
WHERE schemaname = 'public' 
  AND tablename = 'or_agent_users';

-- 4. Check table privileges
SELECT 
    'Table Privileges' as check_type,
    grantee,
    privilege_type
FROM information_schema.table_privileges
WHERE table_schema = 'public' 
  AND table_name = 'or_agent_users'
ORDER BY grantee, privilege_type;

-- 5. Check sequence privileges (for index column)
SELECT 
    'Sequence Privileges' as check_type,
    sequence_schema,
    sequence_name,
    grantee,
    privilege_type
FROM information_schema.usage_privileges
WHERE sequence_schema = 'public' 
  AND sequence_name LIKE 'or_agent_users%'
ORDER BY sequence_name, grantee;


