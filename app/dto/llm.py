from pydantic import BaseModel
from typing import List


class ResumeData(BaseModel):
    name: str
    email: str
    companies: List[str]
    designations: List[str]
