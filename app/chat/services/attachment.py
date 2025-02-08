import asyncio

from fastapi import HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.chat.constants import AttachmentType
from app.chat.crud import crud_attachment
from app.chat.models import Attachment
from app.chat.schemas.attachment import AttachmentCreate
from app.files.image.constants import ImageLimits
from app.files.image.processor import ImageProcessor
from app.files.storage.utils import get_storage


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

    async def create_attachment(self, folder: str, file: UploadFile) -> Attachment:
        """
        Create a single attachment
        """
        content_type = file.content_type
        attachment_type = self.get_attachment_type(content_type=content_type)

        # Process image if it's an image file
        if attachment_type == AttachmentType.IMAGE:
            try:
                file = await ImageProcessor.process_image(
                    file=file,
                    limits=ImageLimits(
                        max_width=1024,
                        max_height=1024,
                        max_file_size=20 * 1024 * 1024,  # 20MB
                    ),
                )
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Failed to process image attachment: {e}")

        # Save file and get storage path
        storage_path = await self.storage.save_file_to_folder(file=file, folder=folder)

        # Create attachment record
        attachment_create = AttachmentCreate(
            file_name=file.filename,
            file_size=file.size,
            mime_type=content_type,
            type=attachment_type,
            storage_path=str(storage_path),
        )

        return await crud_attachment.create(db=self.db, obj_in=attachment_create)

    async def bulk_create_attachments(self, folder: str, files: list[UploadFile]) -> list[Attachment]:
        """
        Create multiple attachments
        """
        tasks = [self.create_attachment(folder=folder, file=file) for file in files]
        return await asyncio.gather(*tasks)
