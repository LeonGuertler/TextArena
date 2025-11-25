-- ============================================================================
-- Complete SQL script to set up game logging tables with RLS
-- Run this in the Supabase SQL Editor to create all required tables
-- ============================================================================

-- ============================================================================
-- 1. Create game_steps table (for step-by-step logging)
-- ============================================================================
CREATE TABLE IF NOT EXISTS public.game_steps (
  id uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_index integer NOT NULL,
  user_uuid text NOT NULL,
  instance text NOT NULL,
  mode text NOT NULL,
  period integer NOT NULL,
  inventory_decision jsonb NOT NULL,  -- Decision dict, empty {} for guidance entries
  total_reward numeric NOT NULL,
  input_prompt text,
  output_prompt text,
  or_recommendation jsonb,
  guidance_message text,  -- Human guidance text (for Mode C guidance entries)
  run_id uuid,
  step_type text DEFAULT 'decision',  -- 'decision' or 'guidance'
  timestamp timestamp with time zone DEFAULT timezone('utc', now())
);

CREATE INDEX IF NOT EXISTS idx_game_steps_user_uuid ON public.game_steps(user_uuid);
CREATE INDEX IF NOT EXISTS idx_game_steps_user_index ON public.game_steps(user_index);
CREATE INDEX IF NOT EXISTS idx_game_steps_run_id ON public.game_steps(run_id);
CREATE INDEX IF NOT EXISTS idx_game_steps_timestamp ON public.game_steps(timestamp);
CREATE INDEX IF NOT EXISTS idx_game_steps_instance_mode ON public.game_steps(instance, mode);
CREATE INDEX IF NOT EXISTS idx_game_steps_step_type ON public.game_steps(step_type);

-- ============================================================================
-- 2. Create game_completions table (for end-of-game logging)
-- ============================================================================
CREATE TABLE IF NOT EXISTS public.game_completions (
  id uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_index integer NOT NULL,
  user_uuid text NOT NULL,
  instance text NOT NULL,
  mode text NOT NULL,
  total_reward numeric NOT NULL,
  run_id uuid,
  timestamp timestamp with time zone DEFAULT timezone('utc', now())
);

CREATE INDEX IF NOT EXISTS idx_game_completions_user_uuid ON public.game_completions(user_uuid);
CREATE INDEX IF NOT EXISTS idx_game_completions_user_index ON public.game_completions(user_index);
CREATE INDEX IF NOT EXISTS idx_game_completions_run_id ON public.game_completions(run_id);
CREATE INDEX IF NOT EXISTS idx_game_completions_timestamp ON public.game_completions(timestamp);
CREATE INDEX IF NOT EXISTS idx_game_completions_instance_mode ON public.game_completions(instance, mode);

-- ============================================================================
-- 3. Grant permissions to service_role
-- ============================================================================
GRANT ALL ON public.game_steps TO service_role;
GRANT ALL ON public.game_completions TO service_role;

-- ============================================================================
-- 4. Enable Row Level Security
-- ============================================================================
ALTER TABLE public.game_steps ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.game_completions ENABLE ROW LEVEL SECURITY;

-- ============================================================================
-- 5. RLS Policies for game_steps
-- ============================================================================

DROP POLICY IF EXISTS "Service role full access to game_steps" ON public.game_steps;
DROP POLICY IF EXISTS "Users can read their own game_steps via lookup" ON public.game_steps;

CREATE POLICY "Service role full access to game_steps"
  ON public.game_steps
  FOR ALL
  TO service_role
  USING (true)
  WITH CHECK (true);

CREATE POLICY "Users can read their own game_steps via lookup"
  ON public.game_steps
  FOR SELECT
  TO authenticated
  USING (
    EXISTS (
      SELECT 1 FROM public.or_agent_users
      WHERE or_agent_users.uuid = game_steps.user_uuid
      AND or_agent_users.user_id = auth.uid()::text
    )
  );

-- ============================================================================
-- 6. RLS Policies for game_completions
-- ============================================================================

DROP POLICY IF EXISTS "Service role full access to game_completions" ON public.game_completions;
DROP POLICY IF EXISTS "Users can read their own game_completions via lookup" ON public.game_completions;

CREATE POLICY "Service role full access to game_completions"
  ON public.game_completions
  FOR ALL
  TO service_role
  USING (true)
  WITH CHECK (true);

CREATE POLICY "Users can read their own game_completions via lookup"
  ON public.game_completions
  FOR SELECT
  TO authenticated
  USING (
    EXISTS (
      SELECT 1 FROM public.or_agent_users
      WHERE or_agent_users.uuid = game_completions.user_uuid
      AND or_agent_users.user_id = auth.uid()::text
    )
  );

-- ============================================================================
-- Verification queries (run these to verify setup)
-- ============================================================================

-- Check all table structures
SELECT 
    'Table Structures' as check_type,
    table_name,
    column_name,
    data_type,
    is_nullable
FROM information_schema.columns
WHERE table_schema = 'public' 
  AND table_name IN ('game_steps', 'game_completions')
ORDER BY table_name, ordinal_position;

-- Check RLS status
SELECT 
    'RLS Status' as check_type,
    schemaname,
    tablename,
    rowsecurity as rls_enabled
FROM pg_tables
WHERE schemaname = 'public' 
  AND tablename IN ('game_steps', 'game_completions')
ORDER BY tablename;

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
  AND tablename IN ('game_steps', 'game_completions')
ORDER BY tablename, policyname;

