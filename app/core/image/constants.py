from pydantic import BaseModel


class ImageLimits(BaseModel):
    """
    Image processing limits
    """

    max_width: int
    max_height: int
    max_file_size: int
    format: str = "JPEG"
    quality: int = 100
    fallback_quality: int = 80
