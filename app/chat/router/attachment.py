from fastapi import APIRouter, Depends, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from app.chat.dependencies.attachment import get_attachment_service
from app.chat.models.attachment import Attachment
from app.chat.schemas.attachment import AttachmentRead
from app.chat.services.attachment import AttachmentService
from app.files.storage.utils import get_storage, normalize_filename, sanitize_filename

router = APIRouter(prefix="/attachments", tags=["Attachments"])


@router.post("/{folder}/", response_model=AttachmentRead)
async def upload_attachment(
    folder: str,
    file: UploadFile,
    service: AttachmentService = Depends(get_attachment_service),
) -> Attachment:
    """
    Upload a single file attachment.
    Args:
        folder: Storage folder path (e.g. 'session_id/message_id')
        file: File to upload
    Returns:
        Uploaded attachment details
    Raises:
        HTTPException: If file type not supported or upload fails
    """
    return await service.create_attachment(folder=folder, file=file)


@router.get("/{folder:path}/{filename}/", response_class=StreamingResponse)
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
