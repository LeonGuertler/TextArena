-- Migration script to add step_type and guidance_message columns to existing game_steps table
-- Run this in the Supabase SQL Editor if the table already exists

-- ============================================================================
-- Add step_type column (if it doesn't exist)
-- ============================================================================
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_schema = 'public' 
        AND table_name = 'game_steps' 
        AND column_name = 'step_type'
    ) THEN
        ALTER TABLE public.game_steps 
        ADD COLUMN step_type text DEFAULT 'decision';
        
        -- Update existing rows to have step_type = 'decision'
        UPDATE public.game_steps 
        SET step_type = 'decision' 
        WHERE step_type IS NULL;
    END IF;
END $$;

-- ============================================================================
-- Add guidance_message column (if it doesn't exist)
-- ============================================================================
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_schema = 'public' 
        AND table_name = 'game_steps' 
        AND column_name = 'guidance_message'
    ) THEN
        ALTER TABLE public.game_steps 
        ADD COLUMN guidance_message text;
    END IF;
END $$;

-- ============================================================================
-- Create index on step_type (if it doesn't exist)
-- ============================================================================
CREATE INDEX IF NOT EXISTS idx_game_steps_step_type ON public.game_steps(step_type);

-- ============================================================================
-- Verification queries
-- ============================================================================

-- Check that columns were added
SELECT 
    'Column Check' as check_type,
    column_name,
    data_type,
    is_nullable,
    column_default
FROM information_schema.columns
WHERE table_schema = 'public' 
  AND table_name = 'game_steps'
  AND column_name IN ('step_type', 'guidance_message')
ORDER BY column_name;

-- Check index exists
SELECT 
    'Index Check' as check_type,
    indexname,
    indexdef
FROM pg_indexes
WHERE schemaname = 'public' 
  AND tablename = 'game_steps'
  AND indexname = 'idx_game_steps_step_type';

