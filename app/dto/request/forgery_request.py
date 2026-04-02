from pydantic import BaseModel, Field


class DetectForgeryRequest(BaseModel):
    image: str = Field(
        ...,
        description="Base64 encoded string of the image file (supports common formats like JPEG, PNG)",
    )
    request_fields: str = Field(
        ...,
        description="Comma separated list of fields to extract from the image",
    )
