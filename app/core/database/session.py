from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.core.config import settings

# an Engine, which the Session will use for connection resources
async_engine = create_async_engine(url=str(settings.DATABASE.DSN), pool_pre_ping=True, echo=False)

AsyncSessionLocal = async_sessionmaker(bind=async_engine, autocommit=False, autoflush=False, expire_on_commit=False)
