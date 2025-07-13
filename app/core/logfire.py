import logfire

from app.core.config import settings


def configure_logfire() -> None:
    """
    Configures Logfire for the application.
    """
    logfire.configure(
        token=settings.LOGFIRE_TOKEN.get_secret_value(),
        environment=settings.ENVIRONMENT.value,
    )
    logfire.instrument_pydantic_ai()
    logfire.instrument_mcp()
    # Add any additional Logfire configurations here
