from typing import Annotated

from fastapi import Depends

from app.attachment.service import AttachmentService
from app.core.database.dependencies import DBSessionDep


async def get_attachment_service(db: DBSessionDep) -> AttachmentService:
    """
    Get the attachment service instance with database dependency.
    """
    return AttachmentService(db=db)


AttachmentServiceDep = Annotated[AttachmentService, Depends(get_attachment_service)]
