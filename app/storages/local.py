from pathlib import Path
from uuid import uuid4

import aiofiles
from fastapi import UploadFile


class LocalStorage:
    def __init__(self, base_path: Path) -> None:
        self.base_path = base_path

    def generate_file_path(self, *path_segments: str, original_filename: str | None) -> Path:
        """
        Generates a unique file path by prefixing the original filename with a UUID.
        """
        unique_name = f"{uuid4().hex}_{original_filename}"
        return self.base_path.joinpath(*path_segments, unique_name)

    async def save_file(self, file: UploadFile, *path_segments: str) -> Path:
        """
        Saves an UploadFile to the destination built from the given path segments.
        Writes the file in chunks to avoid loading the entire file in memory.
        """
        destination = self.generate_file_path(*path_segments, original_filename=file.filename)
        # Ensure the destination directory exists
        destination.parent.mkdir(parents=True, exist_ok=True)

        # Write the file in 1MB chunks
        async with aiofiles.open(destination, "wb") as out_file:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                await out_file.write(chunk)
        await file.close()
        return destination

    async def save_file_to_folder(self, file: UploadFile, folder: str) -> Path:
        """
        Accepts a folder string (e.g., "session_id/message_id") and saves the file to that folder.
        """
        # Split the folder string into segments
        path_segments = folder.split("/")
        return await self.save_file(file, *path_segments)

    def find_file_path(self, folder: str, original_filename: str) -> Path | None:
        """
        Searches the specified folder (e.g. "session_id/message_id") for a file whose name ends with
        '_{original_filename}'. Returns the first matching file path, or None if not found.
        """
        folder_path = self.base_path.joinpath(*folder.split("/"))
        # The naming convention adds a UUID and an underscore before the original filename.
        pattern = f"*_{original_filename}"
        for file_path in folder_path.glob(pattern):
            return file_path  # Return the first match
        return None

    def get_saved_file_path(self, folder: str, original_filename: str) -> Path:
        """
        Searches the specified folder (e.g. "session_id/message_id") for a file whose name ends with
        '_{original_filename}' (the naming convention used when saving the file).
        Returns the matching file path if found, otherwise raises a FileNotFoundError.
        """
        file_path = self.find_file_path(folder, original_filename)
        if file_path is None:
            raise FileNotFoundError(f"File ending with '_{original_filename}' not found in folder '{folder}'.")
        return file_path
