from functools import lru_cache
from pathlib import Path
from typing import Self

from pydantic import HttpUrl, PostgresDsn, RedisDsn, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core.constants import Environment, StorageProvider


class DatabaseSettings(BaseSettings):
    HOST: str
    USER: str
    PASSWORD: SecretStr
    DB: str
    PORT: int = 5432
    DSN: PostgresDsn | None = None

    @model_validator(mode="after")
    def assemble_db_connection(self) -> Self:
        """
        Assemble the database connection URL from the individual components
        when an explicit DSN is not provided.
        """
        if self.DSN is None:
            self.DSN = PostgresDsn.build(
                scheme="postgresql+asyncpg",
                username=self.USER,
                password=self.PASSWORD.get_secret_value(),
                host=self.HOST,
                port=self.PORT,
                path=self.DB,
            )
        return self


class RedisSettings(BaseSettings):
    HOST: str
    PORT: int
    DB: int
    DSN: RedisDsn | None = None

    @model_validator(mode="after")
    def assemble_redis_connection(self) -> Self:
        """
        Assemble the Redis connection URL from the individual components
        when an explicit DSN is not provided.
        """
        if self.DSN is None:
            self.DSN = RedisDsn.build(
                scheme="redis",
                host=self.HOST,
                port=self.PORT,
                path=str(self.DB),
            )
        return self


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
