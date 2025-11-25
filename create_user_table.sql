-- SQL script to create the or_agent_users table in Supabase
-- Run this in the Supabase SQL Editor

-- Create the table if it doesn't exist
CREATE TABLE IF NOT EXISTS public.or_agent_users (
    uuid TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    name TEXT NOT NULL,
    index SERIAL NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc', now())
);

-- Create an index on user_id for faster lookups
CREATE INDEX IF NOT EXISTS idx_or_agent_users_user_id ON public.or_agent_users(user_id);

-- Create an index on uuid for faster lookups
CREATE INDEX IF NOT EXISTS idx_or_agent_users_uuid ON public.or_agent_users(uuid);

-- Grant permissions to service_role (should already have access, but this ensures it)
-- Note: Service role key bypasses RLS, so RLS policies are optional
GRANT ALL ON public.or_agent_users TO service_role;
-- SERIAL creates a sequence automatically named: or_agent_users_index_seq
GRANT USAGE, SELECT ON SEQUENCE or_agent_users_index_seq TO service_role;

-- Optional: Enable RLS if you want additional security
-- ALTER TABLE public.or_agent_users ENABLE ROW LEVEL SECURITY;

-- Optional: If RLS is enabled, create a policy for service_role (though it bypasses RLS anyway)
-- CREATE POLICY "Service role can do everything" ON public.or_agent_users
--     FOR ALL
--     TO service_role
--     USING (true)
--     WITH CHECK (true);

