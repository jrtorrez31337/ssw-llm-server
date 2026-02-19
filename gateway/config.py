"""Gateway configuration — pool management and settings."""

from __future__ import annotations

import threading

from pydantic_settings import BaseSettings


class WorkerPool:
    """Thread-safe round-robin pool of worker URLs."""

    def __init__(self, name: str, urls: list[str] | None = None):
        self.name = name
        self._urls: list[str] = list(urls) if urls else []
        self._index = 0
        self.healthy: set[str] = set(self._urls)
        self._lock = threading.Lock()

    def next(self) -> str | None:
        with self._lock:
            healthy_urls = [u for u in self._urls if u in self.healthy]
            if not healthy_urls:
                return None
            url = healthy_urls[self._index % len(healthy_urls)]
            self._index += 1
            return url

    def all_urls(self) -> list[str]:
        with self._lock:
            return list(self._urls)

    def add_worker(self, url: str) -> None:
        with self._lock:
            if url not in self._urls:
                self._urls.append(url)
                self.healthy.add(url)

    def remove_worker(self, url: str) -> None:
        with self._lock:
            if url in self._urls:
                self._urls.remove(url)
            self.healthy.discard(url)

    def mark_healthy(self, url: str) -> None:
        with self._lock:
            if url in self._urls:
                self.healthy.add(url)

    def mark_unhealthy(self, url: str) -> None:
        with self._lock:
            self.healthy.discard(url)

    def worker_count(self) -> int:
        with self._lock:
            return len(self._urls)

    def healthy_count(self) -> int:
        with self._lock:
            return len([u for u in self._urls if u in self.healthy])

    def is_healthy(self, url: str) -> bool:
        with self._lock:
            return url in self.healthy

    def health_snapshot(self) -> dict[str, bool]:
        with self._lock:
            return {url: url in self.healthy for url in self._urls}


class PoolManager:
    """Manages all model worker pools."""

    def __init__(self):
        self._pools: dict[str, WorkerPool] = {}
        self._lock = threading.Lock()

    def get_pool(self, name: str) -> WorkerPool | None:
        with self._lock:
            return self._pools.get(name)

    def get_or_create_pool(self, name: str) -> WorkerPool:
        with self._lock:
            if name not in self._pools:
                self._pools[name] = WorkerPool(name)
            return self._pools[name]

    def register_worker(self, pool_name: str, url: str) -> WorkerPool:
        pool = self.get_or_create_pool(pool_name)
        pool.add_worker(url)
        return pool

    def unregister_worker(self, pool_name: str, url: str) -> bool:
        pool = self.get_pool(pool_name)
        if pool is None:
            return False
        pool.remove_worker(url)
        return True

    def all_pools(self) -> dict[str, WorkerPool]:
        with self._lock:
            return dict(self._pools)

    def loaded_model_names(self) -> list[str]:
        with self._lock:
            return [name for name, pool in self._pools.items() if pool.worker_count() > 0]


class Settings(BaseSettings):
    # Legacy worker URLs (backward compat)
    heavy_workers: str = "http://heavy-0:8000,http://heavy-1:8000"
    light_workers: str = "http://light-0:8000,http://light-1:8000"

    # Registry
    models_dir: str = "/models"
    models_yaml_path: str = "/app/models.yaml"

    # Dynamic port range (for loader)
    dynamic_port_start: int = 9000
    dynamic_port_end: int = 9099

    # Redis
    redis_url: str = "redis://redis:6379/0"

    # Timeouts (seconds)
    worker_timeout: float = 120.0
    worker_connect_timeout: float = 5.0
    health_check_interval: float = 10.0

    # Metrics
    metrics_max_entries: int = 10000

    model_config = {"env_prefix": "GATEWAY_"}


settings = Settings()
