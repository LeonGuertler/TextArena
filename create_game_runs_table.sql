-- SQL script to create game_runs table (if it doesn't exist)
-- This table is used by the existing log_run() method
-- Run this in the Supabase SQL Editor

-- ============================================================================
-- Create game_runs table (for legacy logging)
-- ============================================================================
-- Note: user_id is TEXT (not UUID) to match the code which passes strings
-- This allows for both UUID strings and other identifiers like "anonymous"
CREATE TABLE IF NOT EXISTS public.game_runs (
  id uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
  run_id uuid,
  user_id text NOT NULL,  -- Changed from uuid to text to match code
  mode text NOT NULL,
  guidance_frequency integer,
  final_reward numeric,
  log_text text,
  created_at timestamp with time zone DEFAULT timezone('utc', now())
);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_game_runs_user_id ON public.game_runs(user_id);
CREATE INDEX IF NOT EXISTS idx_game_runs_run_id ON public.game_runs(run_id);
CREATE INDEX IF NOT EXISTS idx_game_runs_mode ON public.game_runs(mode);
CREATE INDEX IF NOT EXISTS idx_game_runs_created_at ON public.game_runs(created_at);

-- ============================================================================
-- Grant permissions to service_role
-- ============================================================================
GRANT ALL ON public.game_runs TO service_role;

-- ============================================================================
-- Enable Row Level Security (optional)
-- ============================================================================
ALTER TABLE public.game_runs ENABLE ROW LEVEL SECURITY;

-- ============================================================================
-- RLS Policies for game_runs
-- ============================================================================

-- Policy: Service role can do everything
CREATE POLICY "Service role full access to game_runs"
  ON public.game_runs
  FOR ALL
  TO service_role
  USING (true)
  WITH CHECK (true);

-- Policy: Authenticated users can read their own runs
CREATE POLICY "Users can read their own game_runs"
  ON public.game_runs
  FOR SELECT
  TO authenticated
  USING (user_id = auth.uid()::text);

-- ============================================================================
-- Verification queries
-- ============================================================================

-- Check table structure
SELECT 
    'game_runs structure' as check_type,
    column_name,
    data_type,
    is_nullable,
    column_default
FROM information_schema.columns
WHERE table_schema = 'public' 
  AND table_name = 'game_runs'
ORDER BY ordinal_position;

-- Check RLS status
SELECT 
    'RLS Status' as check_type,
    schemaname,
    tablename,
    rowsecurity as rls_enabled
FROM pg_tables
WHERE schemaname = 'public' 
  AND tablename = 'game_runs';

-- Check RLS policies
SELECT 
    'RLS Policies' as check_type,
    schemaname,
    tablename,
    policyname,
    cmd,
    roles
FROM pg_policies
WHERE schemaname = 'public' 
  AND tablename = 'game_runs'
ORDER BY policyname;

