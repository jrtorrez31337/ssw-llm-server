from pydantic_settings import BaseSettings


class WorkerPool:
    def __init__(self, name: str, urls: list[str]):
        self.name = name
        self.urls = urls
        self._index = 0
        self.healthy: set[str] = set(urls)

    def next(self) -> str | None:
        """Round-robin through healthy workers. Returns None if all down."""
        healthy_urls = [u for u in self.urls if u in self.healthy]
        if not healthy_urls:
            return None
        url = healthy_urls[self._index % len(healthy_urls)]
        self._index += 1
        return url

    def all_urls(self) -> list[str]:
        return list(self.urls)


class Settings(BaseSettings):
    # Worker URLs (comma-separated)
    heavy_workers: str = "http://heavy-0:8000,http://heavy-1:8000"
    light_workers: str = "http://light-0:8000,http://light-1:8000"

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


def build_pools() -> dict[str, WorkerPool]:
    return {
        "heavy": WorkerPool("heavy", settings.heavy_workers.split(",")),
        "light": WorkerPool("light", settings.light_workers.split(",")),
    }
