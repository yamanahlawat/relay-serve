import unicodedata

from app.core.config import settings
from app.core.constants import StorageProvider
from app.storages.interface import StorageBackend
from app.storages.local import LocalStorage


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
    # Normalize and remove non-ASCII characters
    return unicodedata.normalize("NFKD", filename).encode("ascii", "ignore").decode("ascii")


def normalize_filename(filename: str) -> str:
    return unicodedata.normalize("NFC", filename)
