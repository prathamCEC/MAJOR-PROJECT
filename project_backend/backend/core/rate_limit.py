"""
In-memory sliding window rate limiter for FastAPI.
Provides brute-force protection for authentication and sensitive API endpoints.
"""

from collections import defaultdict
from datetime import datetime, timezone
import threading
import time
from typing import Callable, Dict, List
from fastapi import HTTPException, Request, status

from .logging_config import logger


class SlidingWindowRateLimiter:
    """
    Thread-safe in-memory sliding window rate limiter.
    Maintains timestamp logs per client key (IP + endpoint).
    """

    def __init__(self):
        self._lock = threading.Lock()
        # key -> list of float epoch timestamps
        self._requests: Dict[str, List[float]] = defaultdict(list)
        self._last_cleanup = time.time()

    def _cleanup(self, now: float, max_window: float = 3600.0) -> None:
        """Periodically purge expired timestamp entries to prevent memory growth."""
        if now - self._last_cleanup > 300.0:  # Every 5 minutes
            expired_keys = []
            for key, timestamps in self._requests.items():
                self._requests[key] = [t for t in timestamps if now - t < max_window]
                if not self._requests[key]:
                    expired_keys.append(key)
            for k in expired_keys:
                del self._requests[k]
            self._last_cleanup = now

    def check(self, key: str, max_requests: int, window_seconds: int) -> tuple[bool, int]:
        """
        Check if an incoming request exceeds the allowable threshold.
        Returns: (is_allowed: bool, retry_after_seconds: int)
        """
        now = time.time()
        window_start = now - window_seconds

        with self._lock:
            self._cleanup(now, float(window_seconds))
            timestamps = self._requests[key]

            # Retain only timestamps within the current sliding window
            valid_timestamps = [t for t in timestamps if t > window_start]
            self._requests[key] = valid_timestamps

            if len(valid_timestamps) >= max_requests:
                earliest = valid_timestamps[0]
                retry_after = max(1, int(earliest + window_seconds - now))
                return False, retry_after

            # Record this valid request
            valid_timestamps.append(now)
            return True, 0


# Global singleton instance
rate_limiter_store = SlidingWindowRateLimiter()


def rate_limit(max_requests: int = 10, window_seconds: int = 60) -> Callable:
    """
    FastAPI dependency factory to enforce rate limits per client IP.

    Example:
        @router.post("/login", dependencies=[Depends(rate_limit(max_requests=5, window_seconds=60))])
    """

    async def dependency(request: Request) -> None:
        # Determine client IP (respecting proxy headers if present)
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            client_ip = forwarded.split(",")[0].strip()
        elif request.client:
            client_ip = request.client.host
        else:
            client_ip = "127.0.0.1"

        endpoint = request.url.path
        rate_key = f"{client_ip}:{endpoint}"

        allowed, retry_after = rate_limiter_store.check(
            key=rate_key,
            max_requests=max_requests,
            window_seconds=window_seconds,
        )

        if not allowed:
            logger.warning(
                f"Rate limit exceeded for client {client_ip} on {endpoint}. "
                f"Limit: {max_requests}/{window_seconds}s. Retry after: {retry_after}s."
            )
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Too many requests. Rate limit exceeded. Please try again in {retry_after} seconds.",
                headers={"Retry-After": str(retry_after)},
            )

    return dependency
