# Fullstack Vending Machine Demo

This demo turns the original `llm_csv_demo.py` experiment into a small fullstack app. It uses
FastAPI for the backend, Supabase for authentication and persistence, and a minimal HTML/JS
frontend built with supabase-js.

## Project Layout

- `backend/simulation.py` – reusable helpers that wrap the vending machine environment, capturing
  transcript events and supporting both Mode 1 (Daily Feedback) and Mode 2 (Periodic Guidance).
- `backend/app.py` – FastAPI application exposing endpoints for creating runs, exchanging
  messages/guidance, and recording final actions. Persists finished runs to Supabase.
- `backend/token_verifier.py` – fetches Supabase JWKS and validates JWTs from the frontend.
- `backend/supabase_client.py` – minimal supabase-py wrapper for inserting completed runs.
- `frontend/index.html` – static HTML interface with login, mode selection, transcript view, and
  chat/guidance controls.

The demo always loads `examples/demand.csv` for demand data.

## Supabase Setup

1. Create a Supabase project and note the **Project URL**, **anon/public key**, and **service role key**.
2. Create a table `game_runs`:

   ```sql
   create table if not exists public.game_runs (
     id uuid primary key default uuid_generate_v4(),
     run_id uuid,
     user_id uuid not null,
     mode text not null,
     guidance_frequency int,
     final_reward numeric,
     log_text text,
     created_at timestamp with time zone default timezone('utc', now())
   );
   ```

3. Enable Row Level Security and add a policy allowing inserts via the service role key (or leave
   RLS disabled while experimenting).

4. In Supabase Authentication settings, ensure email/password signups are enabled.

## Backend Configuration

The backend reads these environment variables (load them via `.env` or OS env vars):

- `SUPABASE_URL` – Supabase project URL.
- `SUPABASE_SERVICE_ROLE_KEY` – service role key.
- `SUPABASE_ANON_KEY` – public anon key (used to feed `/config.js`).
- `OPENAI_API_KEY` – OpenAI key for the VM agent.

You can create a `.env` next to `app.py` (or export the variables in your shell), then launch the
fullstack app with:

```bash
python app.py
```

This loads the `.env`, starts uvicorn on port 8000, and opens the frontend automatically.

## Frontend Configuration

The backend serves a `config.js` endpoint that injects the Supabase URL/anon key for the browser, so
no manual edits to `index.html` are required.

## Usage Flow

1. Sign up or log in using email/password. The frontend uses supabase-js and stores the session.
2. Choose Mode 1 (Daily Feedback) or Mode 2 (Periodic Guidance). Mode 2 allows custom guidance
   frequency (default 5).
3. Press **Start Run**. The backend creates a simulation session bound to your user ID.
4. The transcript panel mirrors backend events (observations, agent proposals, demand actions, etc.).
5. Mode 1:
   - Use the chat box to exchange messages with the agent adviser.
   - When ready, submit a final decision JSON (e.g. `{ "action": {"chips(Regular)": 100} }`).
   - The backend uses the human decision to progress the environment and complete the run.
6. Mode 2:
   - The agent runs automatically until a guidance checkpoint is due.
   - When status indicates “waiting for guidance”, provide a message; the agent resumes afterward.

Upon completion the backend writes a row in `game_runs` containing the transcript and final reward.

## Notes & Testing Tips

- Ensure your OpenAI API key is set; the demo uses `gpt-4o-mini`.
- Supabase JWT verification pulls JWKS from `SUPABASE_URL/auth/v1/keys`. Make sure the backend has
  outbound network access.
- For quick testing without RLS, you can run the backend with a mocked `get_auth_context` that
  injects a fixed user ID. Replace with the real verifier when integrating.
- Inspect backend logs for demand and agent outputs when debugging prompts.
- To run Mode 1 through multiple dialogue turns, keep sending chat messages until satisfied, then
  submit the final action.

