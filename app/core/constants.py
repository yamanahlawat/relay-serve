from enum import Enum
from typing import Any


class BaseEnum(str, Enum):
    @classmethod
    def list(cls) -> list[Any]:
        return list(map(lambda item: item.value, cls))


class Environment(BaseEnum):
    """
    An enumeration representing the various environments in which the application can run.
    """

    LOCAL = "local"
    PRODUCTION = "production"
