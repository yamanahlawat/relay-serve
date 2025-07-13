import sentry_sdk

from app.core.config import settings


def init_sentry() -> None:
    """
    Initialize Sentry error tracking
    """
    sentry_sdk.init(
        dsn=str(settings.SENTRY_DSN),
        environment=settings.ENVIRONMENT.value,
        traces_sample_rate=1.0,
        profiles_sample_rate=1.0,
    )
