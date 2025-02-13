import unicodedata

from app.core.config import settings
from app.core.constants import StorageProvider
from app.files.storage.interface import StorageBackend
from app.files.storage.local import LocalStorage


def get_storage() -> StorageBackend:
    """
    Get the storage backend based on the configured provider.
    """
    if settings.STORAGE_PROVIDER == StorageProvider.LOCAL:
        return LocalStorage(base_path=settings.FILE_STORAGE_PATH)
    elif settings.STORAGE_PROVIDER == StorageProvider.S3:
        # TODO: Implement S3 storage
        raise NotImplementedError("S3 storage not yet implemented")
    else:
        raise ValueError("Invalid storage provider configured")


def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename by removing non-ASCII characters.
    Args:
        filename (str): The filename to sanitize
    Returns:
        str: The sanitized filename
    """
    # Normalize and remove non-ASCII characters
    return unicodedata.normalize("NFKD", filename).encode("ascii", "ignore").decode("ascii")


def normalize_filename(filename: str) -> str:
    """
    Normalize a filename to use NFC (composed) Unicode normalization.
    Args:
        filename (str): The filename to normalize
    Returns:
        str: The normalized filename
    """
    return unicodedata.normalize("NFC", filename)


def get_attachment_download_url(storage_path: str) -> str:
    """
    Get an absolute download URL for an attachment.
    """
    if settings.STORAGE_PROVIDER == StorageProvider.LOCAL:
        # Here, we assume storage_path is something like:
        # "/uploads/8a540b73-.../ba194f98-da60-44a9-.../filename"
        # The endpoint URL becomes:
        # {BASE_URL}{API_URL}/v1/attachments/<folder>/<filename>
        return f"{str(settings.BASE_URL).rstrip('/')}{settings.API_URL}/v1/attachments{storage_path}/"
    elif settings.STORAGE_PROVIDER == StorageProvider.S3:
        raise NotImplementedError("S3 URL generation not yet implemented")
    else:
        raise ValueError("Invalid storage provider configured")
