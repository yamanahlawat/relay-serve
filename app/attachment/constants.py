from app.core.constants import BaseEnum


class AttachmentType(BaseEnum):
    """
    Enum for attachment types
    """

    IMAGE = "image"
    VIDEO = "video"
    DOCUMENT = "document"
    AUDIO = "audio"
