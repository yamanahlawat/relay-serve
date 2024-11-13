from app.database.crud import CRUDBase
from app.providers.models.provider import LLMProvider
from app.providers.schemas.provider import ProviderCreate, ProviderUpdate


class CRUDProvider(CRUDBase[LLMProvider, ProviderCreate, ProviderUpdate]):
    """
    CRUD operations for LLM providers.
    """

    pass


crud_provider = CRUDProvider(model=LLMProvider)
