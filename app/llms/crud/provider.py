from app.database.crud import CRUDBase
from app.llms.models import LLMProvider
from app.llms.schemas import ProviderCreate, ProviderUpdate


class CRUDProvider(CRUDBase[LLMProvider, ProviderCreate, ProviderUpdate]):
    """
    CRUD operations for LLM providers.
    """

    pass


crud_provider = CRUDProvider(model=LLMProvider)
