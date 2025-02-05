from uuid import UUID

from fastapi import UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.chat.constants import AttachmentType
from app.chat.crud import crud_attachment
from app.chat.models import Attachment
from app.chat.schemas.attachment import AttachmentCreate
from app.storages.utils import get_storage


class AttachmentService:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db
        self.storage = get_storage()

    def get_attachment_type(self, content_type: str) -> AttachmentType:
        """
        Determine attachment type from content type
        """
        if content_type.startswith("image/"):
            return AttachmentType.IMAGE
        elif content_type.startswith("video/"):
            return AttachmentType.VIDEO
        elif content_type.startswith("audio/"):
            return AttachmentType.AUDIO
        elif content_type.startswith("application/") or content_type.startswith("text/"):
            return AttachmentType.DOCUMENT
        raise ValueError(f"Unsupported content type: {content_type}")

    async def create_attachment(self, file: UploadFile, message_id: UUID, session_id: UUID) -> Attachment:
        """
        Create a single attachment
        """
        # Save file and get storage path
        storage_path = await self.storage.save_file_to_folder(file=file, folder=f"{session_id}/{message_id}")

        # Create attachment record
        attachment_type = self.get_attachment_type(file.content_type)
        attachment_create = AttachmentCreate(
            message_id=message_id,
            file_name=file.filename,
            file_size=file.size,
            mime_type=file.content_type,
            type=attachment_type,
            storage_path=str(storage_path),
        )
        return await crud_attachment.create(db=self.db, obj_in=attachment_create)

    async def create_attachments(self, files: list[UploadFile], message_id: UUID, session_id: UUID) -> list[Attachment]:
        """
        Create multiple attachments for a message
        """
        attachments = []
        for file in files:
            attachment = await self.create_attachment(file=file, message_id=message_id, session_id=session_id)
            attachments.append(attachment)
        return attachments
