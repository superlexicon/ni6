from pydantic import BaseModel


from typing import Generic, TypeVar

T = TypeVar("T")


class DataResponse(BaseModel, Generic[T]):
    data: T


class ErrorResponse(BaseModel):
    message: str
