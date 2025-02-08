from fastapi import APIRouter, Depends, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from app.chat.dependencies.attachment import get_attachment_service
from app.chat.models.attachment import Attachment
from app.chat.schemas.attachment import AttachmentRead
from app.chat.services.attachment import AttachmentService
from app.files.storage.utils import get_storage, normalize_filename, sanitize_filename

router = APIRouter(prefix="/attachments", tags=["Attachments"])


@router.post("/", response_model=list[AttachmentRead])
async def upload_attachments(
    folder: str,
    files: list[UploadFile],
    service: AttachmentService = Depends(get_attachment_service),
) -> list[Attachment]:
    return await service.bulk_create_attachments(folder=folder, files=files)


@router.get("/{folder:path}/{filename}", response_class=StreamingResponse)
async def serve_attachment(folder: str, filename: str) -> StreamingResponse:
    """
    Serves an attachment file.
    The folder should represent the storage folder (e.g. 'session_id/message_id'),
    and filename is the stored file name (including the UUID prefix).
    """
    storage = get_storage()
    try:
        # Normalize the filename for lookup
        normalized_filename = normalize_filename(filename=filename)
        file_generator = await storage.get_file(folder=folder, original_filename=normalized_filename)
        safe_filename = sanitize_filename(filename=filename)
        return StreamingResponse(
            file_generator,
            media_type="application/octet-stream",
            headers={"Content-Disposition": f"attachment; filename={safe_filename}"},
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found")
