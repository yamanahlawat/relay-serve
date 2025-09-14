# app/storages/local.py
from pathlib import Path
from typing import AsyncGenerator
from uuid import uuid4

import aiofiles
from fastapi import UploadFile

from app.core.config import settings
from app.core.storage.interface import StorageBackend


class LocalStorage(StorageBackend):
    def __init__(self, base_path: Path) -> None:
        """
        Initialize the local storage service with a base path.
        Example: base_path = Path("/uploads")
        """
        self.base_path = base_path

    def generate_file_path(self, *path_segments: str, original_filename: str | None) -> Path:
        """
        Generates a unique file path by prefixing the original filename with a UUID.
        """
        unique_name = f"{uuid4().hex}_{original_filename}"
        return self.base_path.joinpath(*path_segments, unique_name)

    async def save_file(self, file: UploadFile, *path_segments: str) -> Path:
        """
        Saves the file in chunks to avoid loading the entire file into memory.
        """
        destination = self.generate_file_path(*path_segments, original_filename=file.filename)
        destination.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(destination, "wb") as out_file:
            while True:
                chunk = await file.read(1024 * 1024)  # 1 MB chunks
                if not chunk:
                    break
                await out_file.write(chunk)
        await file.close()
        return destination

    async def save_file_to_folder(self, file: UploadFile, folder: str) -> Path:
        """
        Saves a file using a folder path (e.g. 'session_id/message_id').
        The folder string may include a duplicate base folder name (e.g. "uploads"); if so, it is removed.
        """
        path_segments = folder.split("/")
        return await self.save_file(file, *path_segments)

    def find_file_path(self, folder: str, original_filename: str) -> Path | None:
        """
        Constructs the expected file path and checks if it exists.
        If the folder string already contains the base folder name, remove it first.
        """
        # Split the folder into parts
        parts = folder.split("/")
        # Remove the duplicate base folder if present (e.g. "uploads")
        if parts and parts[0].lower() == self.base_path.name.lower():
            parts = parts[1:]
        folder_path = self.base_path.joinpath(*parts)
        # Construct the full file path by joining folder_path with the original filename
        file_path = folder_path.joinpath(original_filename)
        if file_path.exists():
            return file_path
        return None

    async def get_file(self, folder: str, original_filename: str) -> AsyncGenerator[bytes, None]:
        """
        Retrieves a file from local storage as an asynchronous generator of file chunks.
        This generator is intended for use by a router to stream the file to the client.
        """
        file_path = self.find_file_path(folder, original_filename)
        if not file_path or not file_path.exists():
            raise FileNotFoundError(f"File '{original_filename}' not found in folder '{folder}'.")

        async def file_iterator() -> AsyncGenerator[bytes, None]:
            async with aiofiles.open(file_path, mode="rb") as f:
                while True:
                    chunk = await f.read(1024 * 1024)  # 1 MB chunks
                    if not chunk:
                        break
                    yield chunk

        return file_iterator()

    def get_absolute_url(self, folder: str, original_filename: str) -> str:
        """
        Constructs an absolute URL that points to the endpoint serving this file.
        It uses settings.BASE_URL, settings.API_URL, and assumes the attachments router is mounted at:
          {BASE_URL}{API_URL}/v1/attachments
        The folder string is used as provided.
        """
        file_path = self.find_file_path(folder=folder, original_filename=original_filename)
        if not file_path:
            raise FileNotFoundError(f"File '{original_filename}' not found in folder '{folder}'.")
        filename = file_path.name
        # Build URL: e.g. "http://localhost:8000/api/v1/attachments/<folder>/<filename>"
        return f"{str(settings.BASE_URL).rstrip('/')}{settings.API_URL}/v1/attachments/{folder}/{filename}/"
