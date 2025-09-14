import io

from fastapi import HTTPException, UploadFile
from PIL import Image, ImageOps

from app.core.image.constants import ImageLimits


class ImageProcessor:
    @staticmethod
    def check_needs_processing(image: Image.Image, file_size: int, limits: ImageLimits) -> bool:
        """
        Return True if the image dimensions exceed the allowed limits or if the file size is too high.
        """
        width, height = image.size
        return width > limits.max_width or height > limits.max_height or file_size > limits.max_file_size

    @staticmethod
    def _save_image(image: Image.Image, fmt: str, quality: int) -> bytes:
        """
        Helper method to save an image to an in-memory buffer using the specified format and quality.
        Progressive and optimize flags are enabled.
        """
        output = io.BytesIO()
        image.save(output, format=fmt, quality=quality, optimize=True, progressive=True)
        return output.getvalue()

    @staticmethod
    async def process_image(file: UploadFile, limits: ImageLimits) -> UploadFile:
        """
        Process an uploaded image according to the provided limits.

        Steps:
          1. Read the file into memory.
          2. Open the image and auto-orient it using EXIF data.
          3. If the image already meets the limits, wrap and return its raw bytes as an UploadFile.
          4. Otherwise, convert to RGB (if needed) and resize using the LANCZOS filter while preserving aspect ratio.
          5. Save the image using the primary quality setting.
          6. If the result still exceeds the file size limit, try the fallback quality.
          7. Return a new UploadFile instance containing the (possibly processed) image bytes.
        """
        try:
            content = await file.read()
            file_size = len(content)

            # Open the image from memory and auto-correct orientation.
            with Image.open(io.BytesIO(content)) as image:
                image = ImageOps.exif_transpose(image)
                orig_width, orig_height = image.size
                orig_mode = image.mode

                # If image already meets limits, return a new UploadFile wrapping original bytes.
                if not ImageProcessor.check_needs_processing(image, file_size, limits):
                    return UploadFile(
                        file=io.BytesIO(content),
                        filename=file.filename,
                        size=file_size,
                    )

                # Convert to RGB if needed.
                if orig_mode in ("RGBA", "P"):
                    image = image.convert("RGB")

                # Resize if dimensions exceed limits.
                if orig_width > limits.max_width or orig_height > limits.max_height:
                    ratio = min(limits.max_width / orig_width, limits.max_height / orig_height)
                    new_size = (int(orig_width * ratio), int(orig_height * ratio))
                    image = image.resize(new_size, Image.Resampling.LANCZOS)

                # Save with primary quality.
                processed_bytes = ImageProcessor._save_image(image, limits.format, limits.quality)

                # If still too large, try fallback quality.
                if len(processed_bytes) > limits.max_file_size:
                    processed_bytes = ImageProcessor._save_image(image, limits.format, limits.fallback_quality)
                    if len(processed_bytes) > limits.max_file_size:
                        raise HTTPException(
                            status_code=400,
                            detail=(
                                f"Unable to compress image to acceptable size. "
                                f"Final size: {len(processed_bytes) / (1024 * 1024):.1f}MB, "
                                f"Limit: {limits.max_file_size / (1024 * 1024):.1f}MB"
                            ),
                        )

                return UploadFile(
                    file=io.BytesIO(processed_bytes),
                    filename=file.filename,
                    size=len(processed_bytes),
                )

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error processing image: {e}")
