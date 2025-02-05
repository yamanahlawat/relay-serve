from pathlib import Path
from typing import AsyncGenerator, Protocol

from fastapi import UploadFile


class StorageBackend(Protocol):
    async def save_file(self, file: UploadFile, *path_segments: str) -> Path:
        """
        Save an UploadFile to storage using the provided path segments.
        Returns the Path where the file was saved.
        """
        ...

    async def save_file_to_folder(self, file: UploadFile, folder: str) -> Path:
        """
        Save an UploadFile to a specific folder (e.g. 'session_id/message_id').
        Returns the Path where the file was saved.
        """
        ...

    async def get_file(self, folder: str, original_filename: str) -> AsyncGenerator[bytes, None]:
        """
        Retrieve a file from storage.

        For local storage this may return a FileResponse (streaming the file) or
        an absolute URL, while for S3 this may return a pre-signed URL.
        """
        ...

    def get_absolute_url(self, folder: str, original_filename: str) -> str:
        """
        Return an absolute URL that the frontend can use to download or display the file.
        """
        ...
