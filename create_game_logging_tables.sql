-- SQL script to create game_steps and game_completions tables with RLS policies
-- Run this in the Supabase SQL Editor

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
  inventory_decision jsonb NOT NULL,
  total_reward numeric NOT NULL,
  input_prompt text,
  output_prompt text,
  or_recommendation jsonb,
  run_id uuid,
  timestamp timestamp with time zone DEFAULT timezone('utc', now())
);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_game_steps_user_uuid ON public.game_steps(user_uuid);
CREATE INDEX IF NOT EXISTS idx_game_steps_user_index ON public.game_steps(user_index);
CREATE INDEX IF NOT EXISTS idx_game_steps_run_id ON public.game_steps(run_id);
CREATE INDEX IF NOT EXISTS idx_game_steps_timestamp ON public.game_steps(timestamp);
CREATE INDEX IF NOT EXISTS idx_game_steps_instance_mode ON public.game_steps(instance, mode);

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

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_game_completions_user_uuid ON public.game_completions(user_uuid);
CREATE INDEX IF NOT EXISTS idx_game_completions_user_index ON public.game_completions(user_index);
CREATE INDEX IF NOT EXISTS idx_game_completions_run_id ON public.game_completions(run_id);
CREATE INDEX IF NOT EXISTS idx_game_completions_timestamp ON public.game_completions(timestamp);
CREATE INDEX IF NOT EXISTS idx_game_completions_instance_mode ON public.game_completions(instance, mode);

-- ============================================================================
-- 3. Grant permissions to service_role (for backend inserts)
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

-- Policy: Service role can do everything (bypasses RLS anyway, but good for clarity)
CREATE POLICY "Service role full access to game_steps"
  ON public.game_steps
  FOR ALL
  TO service_role
  USING (true)
  WITH CHECK (true);

-- Policy: Authenticated users can read their own steps
-- Note: This assumes user_uuid matches auth.uid() or can be looked up from auth.users
-- If your user_uuid is different, adjust the condition accordingly
CREATE POLICY "Users can read their own game_steps"
  ON public.game_steps
  FOR SELECT
  TO authenticated
  USING (
    -- Match if user_uuid exists in or_agent_users table and matches current user
    EXISTS (
      SELECT 1 FROM public.or_agent_users
      WHERE or_agent_users.uuid = game_steps.user_uuid
      AND or_agent_users.user_id = auth.uid()::text
    )
    OR
    -- Fallback: direct match if user_uuid is the auth.uid (if you store it that way)
    user_uuid = auth.uid()::text
  );

-- Policy: Authenticated users can insert their own steps (if needed from frontend)
-- Note: Backend typically uses service_role, but this allows frontend direct inserts if needed
CREATE POLICY "Users can insert their own game_steps"
  ON public.game_steps
  FOR INSERT
  TO authenticated
  WITH CHECK (
    EXISTS (
      SELECT 1 FROM public.or_agent_users
      WHERE or_agent_users.uuid = game_steps.user_uuid
      AND or_agent_users.user_id = auth.uid()::text
    )
    OR
    user_uuid = auth.uid()::text
  );

-- ============================================================================
-- 6. RLS Policies for game_completions
-- ============================================================================

-- Policy: Service role can do everything
CREATE POLICY "Service role full access to game_completions"
  ON public.game_completions
  FOR ALL
  TO service_role
  USING (true)
  WITH CHECK (true);

-- Policy: Authenticated users can read their own completions
CREATE POLICY "Users can read their own game_completions"
  ON public.game_completions
  FOR SELECT
  TO authenticated
  USING (
    EXISTS (
      SELECT 1 FROM public.or_agent_users
      WHERE or_agent_users.uuid = game_completions.user_uuid
      AND or_agent_users.user_id = auth.uid()::text
    )
    OR
    user_uuid = auth.uid()::text
  );

-- Policy: Authenticated users can insert their own completions (if needed from frontend)
CREATE POLICY "Users can insert their own game_completions"
  ON public.game_completions
  FOR INSERT
  TO authenticated
  WITH CHECK (
    EXISTS (
      SELECT 1 FROM public.or_agent_users
      WHERE or_agent_users.uuid = game_completions.user_uuid
      AND or_agent_users.user_id = auth.uid()::text
    )
    OR
    user_uuid = auth.uid()::text
  );

-- ============================================================================
-- 7. Optional: Grant SELECT to anon role if you want unauthenticated reads
-- (Usually not recommended, but uncomment if needed)
-- ============================================================================
-- GRANT SELECT ON public.game_steps TO anon;
-- GRANT SELECT ON public.game_completions TO anon;

-- ============================================================================
-- Verification queries (run these to verify setup)
-- ============================================================================

-- Check table structure
SELECT 
    'game_steps structure' as check_type,
    column_name,
    data_type,
    is_nullable
FROM information_schema.columns
WHERE table_schema = 'public' 
  AND table_name = 'game_steps'
ORDER BY ordinal_position;

SELECT 
    'game_completions structure' as check_type,
    column_name,
    data_type,
    is_nullable
FROM information_schema.columns
WHERE table_schema = 'public' 
  AND table_name = 'game_completions'
ORDER BY ordinal_position;

-- Check RLS status
SELECT 
    'RLS Status' as check_type,
    schemaname,
    tablename,
    rowsecurity as rls_enabled
FROM pg_tables
WHERE schemaname = 'public' 
  AND tablename IN ('game_steps', 'game_completions');

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

