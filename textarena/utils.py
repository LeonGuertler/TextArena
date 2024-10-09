"""Utilities used across games and environments.
Expect functionality to be spun off into separate modules in the future."""

import asyncio
import os

import aiohttp
import backoff
import dotenv

RETRY_EXCEPTIONS = (aiohttp.ClientError, asyncio.TimeoutError, AssertionError)

dotenv.load_dotenv()
OPEN_ROUTER_TOKEN = os.getenv("OPEN_ROUTER_TOKEN")


@backoff.on_exception(backoff.expo, RETRY_EXCEPTIONS, max_tries=5, jitter=None)
# @utils.file_cache(verbose=True) # not implemented yet but can it is really cool...
async def open_router_generate(
    text: str, model_string: str, message_history: list[dict] | None, **gen_kwargs
) -> str:
    """Generate a response from a model behind openrouter model."""
    async with aiohttp.ClientSession(raise_for_status=True) as session:
        if not message_history:
            message_history = []
        response = await session.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPEN_ROUTER_TOKEN}",
            },
            json={
                "model": model_string,  # Optional
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful game-playing assistant.",
                    }
                ]
                + message_history
                + [
                    {"role": "user", "content": text},
                ],
                **gen_kwargs,  # https://openrouter.ai/docs/parameters
            },
            timeout=100,
        )
        response_json = await response.json()
        if (
            not response_json or "choices" not in response_json
        ):  # check if response is valid
            raise AssertionError("Invalid response")
        return response_json["choices"][0]["message"]["content"]


def batch_open_router_generate(
    texts: list[str],
    model_string: str,
    message_history: list[dict] | None,
    batch_size: int = 8,
    **gen_kwargs,
) -> list[str]:
    """G"""
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        batch_responses = asyncio.run(
            asyncio.gather(
                *[
                    open_router_generate(
                        text=text,
                        model_string=model_string,
                        message_history=message_history,
                        **gen_kwargs,
                    )
                    for text in batch_texts
                ]
            )
        )
        if i == 0:
            responses = batch_responses
        else:
            responses += batch_responses
    return responses


DEFAULT_GEN_KWARGS = {
    "max_tokens": 150,
    "n": 1,
    "stop": None,
    "temperature": 0.7,
}
