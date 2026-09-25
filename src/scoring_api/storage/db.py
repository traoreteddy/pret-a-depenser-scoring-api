"""Accès PostgreSQL asynchrone (psycopg 3 + pool) et application du schéma."""

from __future__ import annotations

from importlib import resources

from psycopg_pool import AsyncConnectionPool

from scoring_api.config import Settings


def make_pool(settings: Settings) -> AsyncConnectionPool:
    if not settings.database_url:
        msg = "DATABASE_URL manquant"
        raise ValueError(msg)
    return AsyncConnectionPool(
        conninfo=settings.database_url,
        min_size=settings.db_pool_min,
        max_size=settings.db_pool_max,
        open=False,
        timeout=5.0,
        kwargs={"application_name": settings.app_name, "connect_timeout": 5},
    )


def schema_sql() -> str:
    return resources.files("scoring_api.storage").joinpath("schema.sql").read_text(encoding="utf-8")


async def ensure_schema(pool: AsyncConnectionPool) -> None:
    async with pool.connection() as conn:
        await conn.execute(schema_sql())  # type: ignore[arg-type]
        await conn.commit()


async def ping(pool: AsyncConnectionPool, timeout: float = 2.0) -> bool:
    try:
        async with pool.connection(timeout=timeout) as conn:
            await conn.execute("SELECT 1")
        return True
    except Exception:
        return False
