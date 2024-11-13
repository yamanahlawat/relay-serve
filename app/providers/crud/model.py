from app.database.crud import CRUDBase
from app.providers.models import LLMModel
from app.providers.schemas import ModelCreate, ModelUpdate


class CRUDModel(CRUDBase[LLMModel, ModelCreate, ModelUpdate]):
    """
    CRUD operations for LLM Models.
    """

    pass


crud_model = CRUDModel(model=LLMModel)
