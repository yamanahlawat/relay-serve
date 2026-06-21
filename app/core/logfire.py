import logfire
from fastapi import FastAPI

from app.core.config import settings
from app.core.database.session import async_engine


def configure_logfire(app: FastAPI) -> None:
    """
    Configures Logfire for the application.
    """
    logfire.configure(
        token=settings.LOGFIRE_TOKEN.get_secret_value(),
        environment=settings.ENVIRONMENT.value,
    )
    # Application-level instrumentation
    logfire.instrument_pydantic_ai()
    logfire.instrument_mcp()
    logfire.instrument_fastapi(app=app)
    # Infrastructure-level instrumentation (DB, driver, cache)
    logfire.instrument_sqlalchemy(engine=async_engine)
    logfire.instrument_asyncpg()
    logfire.instrument_redis()
    # Add any additional Logfire configurations here
