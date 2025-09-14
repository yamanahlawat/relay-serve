from app.attachment.model import Attachment
from app.attachment.schema import AttachmentCreate, AttachmentUpdate
from app.core.database.crud import CRUDBase


class CRUDAttachment(CRUDBase[Attachment, AttachmentCreate, AttachmentUpdate]):
    """
    CRUD operations for Attachments.
    """

    pass


crud_attachment = CRUDAttachment(model=Attachment)
