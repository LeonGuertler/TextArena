# Scalability Analysis: 40 Concurrent Users with 1 Worker

## Current Architecture

### Components
- **FastAPI backend** with 1 uvicorn worker
- **In-memory RUN_STORE** dictionary (no expiration/cleanup)
- **Background threads** for LLM generation (daemon threads)
- **SSE connections** for real-time streaming
- **Polling** every 2 seconds per user
- **OpenAI API** calls via individual client instances (no connection pooling)

### Per-Session Resource Usage
- **Mode B**: ~1 LLM call per user action (user-controlled pace)
- **Mode C**: ~20 LLM calls per game (1 per period, auto-play)
- **Memory**: Full game state, transcript, environment state per session
- **SSE**: 1 persistent connection per active user
- **Threads**: 1 background thread per active LLM generation

---

## Bottlenecks & Issues

### 🔴 Critical Issues

#### 1. **OpenAI API Rate Limits**
- **Problem**: Each agent creates its own `OpenAI()` client instance
- **Rate Limits**:
  - Tier 1: 500 requests/minute (RPM)
  - Tier 2: 3,500 RPM
- **Impact**: 
  - 40 users × 20 periods (Mode C) = 800 LLM calls
  - If all start simultaneously: **800 calls in ~10-20 minutes**
  - **Risk**: Rate limit exceeded → failed requests → retries → cascading failures

#### 2. **Memory Leak: RUN_STORE Never Cleans Up**
- **Problem**: `RUN_STORE` dictionary grows indefinitely
- **Impact**: 
  - Each session: ~1-5 MB (game state + transcript)
  - 40 sessions: ~40-200 MB
  - After 1000 sessions: ~1-5 GB
  - **Risk**: Memory exhaustion, OOM kills

#### 3. **Python GIL Limits True Parallelism**
- **Problem**: Python's Global Interpreter Lock allows only 1 thread to execute Python bytecode at a time
- **Impact**: 
  - LLM API calls are I/O-bound (network waits), so threading helps
  - BUT: 40 concurrent threads competing for GIL can cause context switching overhead
  - CPU-bound operations (JSON parsing, state updates) block other threads
- **Risk**: Degraded performance under high concurrency

### ⚠️ Moderate Issues

#### 4. **No Connection Pooling**
- **Problem**: Each `OpenAI()` client creates new HTTP connections
- **Impact**: 
  - Connection overhead for each LLM call
  - TCP connection limits (OS-dependent, typically ~65k)
- **Risk**: Connection exhaustion under extreme load

#### 5. **SSE Connection Limits**
- **Current**: 40 persistent SSE connections
- **Impact**: 
  - Each connection holds a request handler
  - FastAPI/uvicorn can handle hundreds, but monitoring needed
- **Risk**: Lower priority, but should monitor connection count

#### 6. **Polling Overhead**
- **Current**: 40 users × 1 request/2 seconds = 20 requests/second
- **Impact**: 
  - Each poll reads from `RUN_STORE` (fast, in-memory)
  - Minimal CPU overhead
- **Risk**: Low, but adds up with more users

#### 7. **No Request Timeout/Retry Limits**
- **Problem**: LLM calls can hang indefinitely
- **Impact**: 
  - Threads stuck waiting for API response
  - No timeout → resource exhaustion
- **Risk**: Thread pool exhaustion

---

## Capacity Estimates

### ✅ What Works Well
- **SSE connections**: 40 is fine (can handle 100+)
- **Polling**: 20 req/s is manageable
- **Memory per session**: Reasonable (~1-5 MB)
- **Threading model**: Works for I/O-bound LLM calls

### ⚠️ What May Struggle
- **Concurrent LLM calls**: 40 simultaneous calls may hit rate limits
- **Memory growth**: No cleanup → slow memory leak
- **GIL contention**: 40 threads competing can cause slowdowns

### 🔴 What Will Fail
- **Rate limits**: If all 40 users start Mode C simultaneously → 800 calls in short time → rate limit exceeded
- **Memory**: After days/weeks without cleanup → OOM

---

## Recommendations

### 🚨 High Priority (Required for 40 Users)

#### 1. **Add RUN_STORE Cleanup**
```python
# Add TTL-based cleanup
import time
from datetime import datetime, timedelta

RUN_STORE: Dict[str, RunEntry] = {}
RUN_STORE_CLEANUP_INTERVAL = 300  # 5 minutes
RUN_STORE_TTL = 3600  # 1 hour

def cleanup_expired_runs():
    """Remove runs older than TTL."""
    now = time.time()
    expired = [
        run_id for run_id, entry in RUN_STORE.items()
        if now - entry.created_at > RUN_STORE_TTL
    ]
    for run_id in expired:
        del RUN_STORE[run_id]
    return len(expired)

# Run cleanup periodically
@app.on_event("startup")
async def start_cleanup_task():
    async def cleanup_loop():
        while True:
            await asyncio.sleep(RUN_STORE_CLEANUP_INTERVAL)
            cleaned = cleanup_expired_runs()
            if cleaned > 0:
                logger.info(f"Cleaned up {cleaned} expired runs")
    asyncio.create_task(cleanup_loop())
```

#### 2. **Add Rate Limiting/Queue for LLM Calls**
```python
from asyncio import Semaphore
import asyncio

# Limit concurrent LLM calls
MAX_CONCURRENT_LLM_CALLS = 10  # Adjust based on API tier
llm_semaphore = Semaphore(MAX_CONCURRENT_LLM_CALLS)

async def rate_limited_llm_call(agent, prompt):
    async with llm_semaphore:
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, agent, prompt)
```

#### 3. **Add Request Timeouts**
```python
# In OpenAI client initialization
self.client = OpenAI(
    api_key=api_key,
    timeout=30.0,  # 30 second timeout
    max_retries=3
)
```

### ⚠️ Medium Priority (Improves Stability)

#### 4. **Connection Pooling**
```python
# Reuse OpenAI client instances
from functools import lru_cache

@lru_cache(maxsize=1)
def get_openai_client():
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Use shared client in agents
```

#### 5. **Add Monitoring/Logging**
```python
# Track active sessions, LLM call rate, memory usage
@app.get("/metrics")
def get_metrics():
    return {
        "active_sessions": len(RUN_STORE),
        "memory_mb": get_memory_usage(),
        "concurrent_llm_calls": get_active_thread_count(),
    }
```

#### 6. **Graceful Degradation**
```python
# Return 503 if overloaded
MAX_ACTIVE_SESSIONS = 100

@app.post("/runs")
def start_run(...):
    if len(RUN_STORE) >= MAX_ACTIVE_SESSIONS:
        raise HTTPException(
            status_code=503,
            detail="Server at capacity. Please try again later."
        )
```

### 💡 Low Priority (Nice to Have)

#### 7. **Use AsyncIO Instead of Threading**
- Replace `threading.Thread` with `asyncio.create_task()`
- Better for I/O-bound operations
- Reduces GIL contention

#### 8. **Add Redis for Shared State**
- If scaling to multiple workers
- Replace in-memory `RUN_STORE` with Redis
- Enables horizontal scaling

---

## Testing Recommendations

### Load Testing
1. **Simulate 40 concurrent users**
   ```bash
   # Use locust or similar
   locust -f load_test.py --users 40 --spawn-rate 5
   ```

2. **Monitor**:
   - Memory usage over time
   - LLM API rate limit errors
   - Response times
   - Failed requests

3. **Test Scenarios**:
   - All users start Mode C simultaneously
   - Mixed Mode B and Mode C
   - Long-running sessions (1+ hour)

---

## Conclusion

### Can It Handle 40 Users? **Maybe, with caveats:**

✅ **Will work IF**:
- Users don't all start simultaneously
- OpenAI API tier supports 40+ concurrent calls
- Sessions are cleaned up periodically
- Memory is monitored

❌ **Will fail IF**:
- All 40 users start Mode C at once → rate limit exceeded
- No cleanup → memory leak → OOM after days
- API tier too low → rate limits hit

### Recommended Actions:
1. **Implement cleanup** (required)
2. **Add rate limiting** (required)
3. **Add timeouts** (required)
4. **Monitor metrics** (highly recommended)
5. **Test with load testing** (before production)

### For Production:
- **Consider multiple workers** (2-4) for better parallelism
- **Use Redis** for shared state if multi-worker
- **Add proper monitoring** (Prometheus, Grafana)
- **Set up alerts** for rate limits, memory, errors

