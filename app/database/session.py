from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.core.config import settings
from app.core.constants import Environment

echo = settings.ENVIRONMENT == Environment.LOCAL

# an Engine, which the Session will use for connection resources
async_engine = create_async_engine(url=str(settings.DATABASE.DSN), pool_pre_ping=True, echo=echo)

AsyncSessionLocal = async_sessionmaker(bind=async_engine, autocommit=False, autoflush=False, expire_on_commit=False)
