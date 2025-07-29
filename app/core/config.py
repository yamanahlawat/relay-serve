from functools import lru_cache
from pathlib import Path

from pydantic import HttpUrl, PostgresDsn, RedisDsn, SecretStr, ValidationInfo, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core.constants import Environment, StorageProvider


class DatabaseSettings(BaseSettings):
    HOST: str
    USER: str
    PASSWORD: SecretStr
    DB: str
    PORT: int = 5432
    DSN: PostgresDsn | None = None

    @field_validator("DSN", mode="after")
    def assemble_db_connection(cls, value: PostgresDsn | None, info: ValidationInfo) -> PostgresDsn:
        """
        Assembles the database connection URL from the individual components.
        """
        if isinstance(value, str):
            return value
        values = info.data
        return PostgresDsn.build(
            scheme="postgresql+asyncpg",
            username=values.get("USER"),
            password=values.get("PASSWORD").get_secret_value(),
            host=values.get("HOST"),
            port=values.get("PORT"),
            path=values.get("DB"),
        )


class RedisSettings(BaseSettings):
    HOST: str
    PORT: int
    DB: int
    DSN: RedisDsn | None = None

    @field_validator("DSN", mode="after")
    def assemble_redis_connection(cls, value: RedisDsn | None, info: ValidationInfo) -> RedisDsn:
        """
        Assembles the Redis connection URL from the individual components.
        """
        if isinstance(value, str):
            return value
        values = info.data
        return RedisDsn.build(
            scheme="redis",
            host=values.get("HOST", ""),
            port=values.get("PORT"),
            path=str(values.get("DB")),
        )


class Settings(BaseSettings):
    """
    Handles config and settings for relay serve.
    Fetches the config from environment variables and .env file
    """

    # Base URL for the application
    BASE_URL: HttpUrl

    # the endpoint for api docs and all endpoints
    API_URL: str = "/api"

    # Enables or disables debug mode
    DEBUG: bool = False

    # The current environment
    ENVIRONMENT: Environment = Environment.LOCAL

    # List of allowed CORS origins
    ALLOWED_CORS_ORIGINS: list[HttpUrl] = []

    # Sentry Configuration
    SENTRY_DSN: HttpUrl | None = None
    SENTRY_AUTH_TOKEN: SecretStr | None = None

    # Database Settings
    DATABASE: DatabaseSettings

    # Redis
    REDIS: RedisSettings

    # File Storage
    STORAGE_PROVIDER: StorageProvider = StorageProvider.LOCAL
    FILE_STORAGE_PATH: Path = Path("/uploads")

    # Logfire
    LOGFIRE_TOKEN: SecretStr | None = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        case_sensitive=True,
        extra="allow",
    )


@lru_cache
def get_settings() -> Settings:
    """
    Retrieves and caches the application settings.
    """
    return Settings()


settings = get_settings()
