from functools import lru_cache
from pathlib import Path

from pydantic import Field, HttpUrl, PostgresDsn, RedisDsn, SecretStr, ValidationInfo, field_validator
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


class LLMSettings(BaseSettings):
    """Settings for LLM configuration."""

    # Default model settings
    DEFAULT_TEMPERATURE: float = Field(default=0.7, ge=0.0, le=2.0)
    DEFAULT_MAX_TOKENS: int = 2048
    DEFAULT_TOP_P: float = Field(default=0.9, ge=0.0, le=1.0)

    # Agent configuration
    DEFAULT_RETRIES: int = 3
    ENABLE_LOGFIRE: bool = False


class Mem0Settings(BaseSettings):
    """Settings for mem0 memory management."""

    # Vector store configuration
    VECTOR_STORE_PROVIDER: str = "qdrant"
    VECTOR_STORE_HOST: str = "localhost"
    VECTOR_STORE_PORT: int = 6333
    VECTOR_STORE_COLLECTION_NAME: str = "relay_memories"

    # Graph store configuration
    ENABLE_GRAPH_MEMORY: bool = True
    GRAPH_STORE_PROVIDER: str = "neo4j"
    GRAPH_STORE_URL: str = "bolt://localhost:7687"
    GRAPH_STORE_USERNAME: str = "neo4j"
    GRAPH_STORE_PASSWORD: SecretStr | None = None

    # LLM configuration for memory processing
    MEMORY_LLM_PROVIDER: str = "openai"
    MEMORY_LLM_MODEL: str = "gpt-4o-mini"
    MEMORY_LLM_TEMPERATURE: float = 0.0

    # Embedder configuration
    EMBEDDER_PROVIDER: str = "openai"
    EMBEDDER_MODEL: str = "text-embedding-3-small"
    EMBEDDER_DIMENSIONS: int = 1536

    # Memory management settings
    MEMORY_DECAY_RATE: float = 0.99
    MAX_MEMORY_AGE_DAYS: int = 90


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

    # Search
    TAVILY_SEARCH_API_KEY: SecretStr | None = None

    # AI Configuration
    LLM: LLMSettings = LLMSettings()
    MEM0: Mem0Settings = Mem0Settings()

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
