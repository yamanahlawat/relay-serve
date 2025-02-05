from app.chat.models import Attachment
from app.chat.schemas import AttachmentCreate, AttachmentUpdate
from app.database.crud import CRUDBase


class CRUDAttachment(CRUDBase[Attachment, AttachmentCreate, AttachmentUpdate]):
    """
    CRUD operations for Attachments.
    """

    pass


crud_attachment = CRUDAttachment(model=Attachment)
