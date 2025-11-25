-- Simplified SQL script to create game_steps and game_completions tables with RLS
-- This version uses simpler policies - adjust based on your authentication setup
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
-- 5. RLS Policies - SIMPLIFIED VERSION
-- ============================================================================
-- Note: service_role bypasses RLS, so these policies mainly affect authenticated/anonymous users

-- Policy: Service role can do everything
CREATE POLICY "Service role full access to game_steps"
  ON public.game_steps
  FOR ALL
  TO service_role
  USING (true)
  WITH CHECK (true);

CREATE POLICY "Service role full access to game_completions"
  ON public.game_completions
  FOR ALL
  TO service_role
  USING (true)
  WITH CHECK (true);

-- ============================================================================
-- OPTION A: If user_uuid in game_steps matches auth.uid() directly
-- ============================================================================
-- Uncomment these if your user_uuid is the same as Supabase auth.uid()

-- CREATE POLICY "Users can read their own game_steps"
--   ON public.game_steps
--   FOR SELECT
--   TO authenticated
--   USING (user_uuid = auth.uid()::text);

-- CREATE POLICY "Users can read their own game_completions"
--   ON public.game_completions
--   FOR SELECT
--   TO authenticated
--   USING (user_uuid = auth.uid()::text);

-- ============================================================================
-- OPTION B: If you need to look up user_uuid from or_agent_users table
-- ============================================================================
-- Uncomment these if user_uuid needs to be looked up from or_agent_users

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
-- OPTION C: Allow all authenticated users to read (less secure, but simpler)
-- ============================================================================
-- Uncomment these if you want all authenticated users to read all data
-- (Not recommended for production, but useful for development)

-- CREATE POLICY "Authenticated users can read all game_steps"
--   ON public.game_steps
--   FOR SELECT
--   TO authenticated
--   USING (true);

-- CREATE POLICY "Authenticated users can read all game_completions"
--   ON public.game_completions
--   FOR SELECT
--   TO authenticated
--   USING (true);

-- ============================================================================
-- OPTION D: Disable RLS entirely (if you only use service_role)
-- ============================================================================
-- If your backend always uses service_role and you don't need user-level access control,
-- you can disable RLS:

-- ALTER TABLE public.game_steps DISABLE ROW LEVEL SECURITY;
-- ALTER TABLE public.game_completions DISABLE ROW LEVEL SECURITY;

