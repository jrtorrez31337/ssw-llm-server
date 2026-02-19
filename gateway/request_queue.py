"""Request queue — holds requests for unloaded models until they become available."""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid

import redis.asyncio as redis

log = logging.getLogger(__name__)

QUEUE_KEY_PREFIX = "ssw:queue:"
RESPONSE_KEY_PREFIX = "ssw:response:"
DEFAULT_WAIT_TIMEOUT = 180  # seconds


class RequestQueue:
    """Redis-based request queue with pub/sub response delivery."""

    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client

    async def enqueue_and_wait(
        self, model: str, request_body: dict, timeout: float = DEFAULT_WAIT_TIMEOUT
    ) -> tuple[str, dict | None]:
        """Subscribe first, then enqueue, then wait. Prevents pub/sub race."""
        request_id = str(uuid.uuid4())
        channel_name = f"{RESPONSE_KEY_PREFIX}{request_id}"

        # Subscribe BEFORE enqueuing to prevent race
        pubsub = self.redis.pubsub()
        await pubsub.subscribe(channel_name)

        # Now enqueue
        entry = json.dumps({
            "request_id": request_id,
            "model": model,
            "body": request_body,
            "ts": time.time(),
        })
        await self.redis.rpush(f"{QUEUE_KEY_PREFIX}{model}", entry)
        log.info("Enqueued request %s for model %s (subscribed first)", request_id, model)

        # Wait for response
        try:
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    msg = await asyncio.wait_for(
                        pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0),
                        timeout=min(remaining, 2.0),
                    )
                except asyncio.TimeoutError:
                    continue
                if msg and msg["type"] == "message":
                    return request_id, json.loads(msg["data"])
            return request_id, None
        finally:
            await pubsub.unsubscribe(channel_name)
            await pubsub.aclose()

    async def publish_response(self, request_id: str, response: dict):
        """Publish response for a queued request."""
        channel = f"{RESPONSE_KEY_PREFIX}{request_id}"
        await self.redis.publish(channel, json.dumps(response))

    async def drain_queue(self, model: str) -> list[dict]:
        """Pop all pending requests for a model."""
        key = f"{QUEUE_KEY_PREFIX}{model}"
        entries = []
        while True:
            raw = await self.redis.lpop(key)
            if raw is None:
                break
            entries.append(json.loads(raw))
        if entries:
            log.info("Drained %d queued requests for %s", len(entries), model)
        return entries

    async def queue_length(self, model: str) -> int:
        return await self.redis.llen(f"{QUEUE_KEY_PREFIX}{model}")
