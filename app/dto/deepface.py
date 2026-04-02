from pydantic import BaseModel


class DeepfaceResponse(BaseModel):
    verify: bool
    percentage: float
    model: str
