from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.attachment.service import AttachmentService
from app.core.database.dependencies import get_db_session


async def get_attachment_service(db: AsyncSession = Depends(get_db_session)) -> AttachmentService:
    """
    Get the attachment service instance with database dependency.
    """
    return AttachmentService(db=db)
