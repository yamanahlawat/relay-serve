from typing import Any, Generic, Sequence, Type, TypeVar
from uuid import UUID

from pydantic import BaseModel
from sqlalchemy import desc, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm.attributes import InstrumentedAttribute

from app.database.base import Base

ModelType = TypeVar("ModelType", bound=Base)
CreateSchemaType = TypeVar("CreateSchemaType", bound=BaseModel)
UpdateSchemaType = TypeVar("UpdateSchemaType", bound=BaseModel)


class CRUDBase(Generic[ModelType, CreateSchemaType, UpdateSchemaType]):
    """
    Base class for CRUD operations: create, read, update, delete.
    It's meant to be extended by specific model CRUD classes,
    providing basic CRUD operations for a given SQLAlchemy model.
    """

    def __init__(self, model: Type[ModelType]) -> None:
        """
        Initialize the CRUDBase class with the model type.
        Args:
            model (Type[ModelType]): SQLAlchemy model class
        """
        self.model = model

    async def get(self, db: AsyncSession, id: UUID) -> ModelType | None:
        """
        Get a specific record by id.
        Args:
            db (AsyncSession): Database session
            id (UUID): Id of the record to fetch
        Returns:
            Instance of the ModelType if found, else None
        """
        return await db.get(self.model, id)

    async def filter(
        self,
        db: AsyncSession,
        *,
        order_on: list[InstrumentedAttribute] | None = None,
        offset: int = 0,
        limit: int = 10,
        filters: list[InstrumentedAttribute] | None = None,
    ) -> Sequence[ModelType]:
        """
        Get multiple records from the database based on the provided filters and
        pagination parameters.
        Args:
            db (AsyncSession): Database session
            order_on (list[InstrumentedAttribute] | None, optional): Ordering of records. Defaults to None.
            offset (int, optional): Number of records to skip. Defaults to 0.
            limit (int, optional): Maximum number of records to retrieve. Defaults to 10.
            filters (list[InstrumentedAttribute] | None, optional): Filters to apply. Defaults to None.
        Returns:
            Sequence[ModelType]: List of instances of the ModelType
        """
        if order_on is None:
            # Default ordering by created_at
            order_on = [desc(self.model.created_at)]
        query = select(self.model)
        if filters:
            query = query.where(*filters)
        query = query.order_by(*order_on).offset(offset).limit(limit)
        items = await db.scalars(query)
        return items.all()

    async def create(self, db: AsyncSession, *, obj_in: CreateSchemaType) -> ModelType:
        """
        Create a new record with provided input data.
        Args:
            db (AsyncSession): Database session
            obj_in (CreateSchemaType): Pydantic schema model with the data to create a new record
        Returns:
            Instance of the ModelType for the created record
        """
        db_obj = self.model(**obj_in.model_dump(mode="json"))
        db.add(db_obj)
        await db.commit()
        await db.refresh(instance=db_obj)
        return db_obj

    async def bulk_create(self, db: AsyncSession, *, objs_in: list[CreateSchemaType]) -> list[ModelType]:
        """
        Create multiple new records with provided input data.
        Args:
            db (AsyncSession): Database session
            objs_in (list[CreateSchemaType]): List of Pydantic schema models with the data to create new records
        Returns:
            List of instances of the ModelType for the created records
        """
        db_objs = [self.model(**obj_in.model_dump()) for obj_in in objs_in]
        db.add_all(db_objs)
        await db.commit()
        return db_objs

    async def update(self, db: AsyncSession, *, id: Any, obj_in: UpdateSchemaType) -> ModelType | None:
        """
        Update a specific record by id.
        Args:
            db (AsyncSession): Database session
            id (Any): Id of the record to update
            obj_in (UpdateSchemaType): Pydantic schema model with the data to update
        Returns:
            ModelType | None: Instance of the ModelType for the updated record if found, else None
        """
        obj_in_data = obj_in.model_dump(mode="json", exclude_unset=True)
        query = update(self.model).where(self.model.id == id).values(**obj_in_data).returning(self.model)
        db_obj = await db.scalar(query)
        await db.commit()
        return db_obj

    async def delete(self, db: AsyncSession, *, id: UUID) -> ModelType | None:
        """
        Delete a specific record by id.
        Args:
            db (AsyncSession): Database session
            id (int): Id of the record to delete
        Returns:
            Instance of the ModelType for the deleted record if found, else None
        """
        db_obj = await db.get(self.model, id)
        await db.delete(db_obj)
        await db.commit()
        return db_obj
