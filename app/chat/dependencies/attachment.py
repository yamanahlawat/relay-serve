from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.chat.services.attachment import AttachmentService
from app.database.dependencies import get_db_session


async def get_attachment_service(db: AsyncSession = Depends(get_db_session)) -> AttachmentService:
    """
    Get the attachment service instance with database dependency.
    """
    return AttachmentService(db=db)
