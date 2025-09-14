from app.core.database.crud import CRUDBase
from app.provider.model import LLMProvider
from app.provider.schema import ProviderCreate, ProviderUpdate


class CRUDProvider(CRUDBase[LLMProvider, ProviderCreate, ProviderUpdate]):
    """
    CRUD operations for LLM providers.
    """

    pass


crud_provider = CRUDProvider(model=LLMProvider)
