"""
Resume Extraction Request/Response Models

This module defines Pydantic models for resume extraction endpoints.
Uses base64 encoded file data in JSON requests (consistent with other endpoints).
"""

import base64
from typing import Optional, Dict, Any, List, Union
from pydantic import BaseModel, Field, field_validator


class ResumeExtractionRequest(BaseModel):
    """Request model for resume extraction using base64 encoded file"""

    file_data: str = Field(..., description="Base64 encoded resume/CV file content")
    filename: str = Field(..., description="Original filename")
    include_raw_entities: bool = Field(
        default=False,
        description="If true, return raw entity extraction results along with structured data"
    )

    @field_validator('file_data')
    @classmethod
    def validate_base64(cls, v: str) -> str:
        """Ensure file_data is valid Base64"""
        try:
            base64.b64decode(v, validate=True)
        except Exception:
            raise ValueError("file_data must be a valid Base64 encoded string")
        return v


class EducationEntry(BaseModel):
    """Education entry model"""
    degree: Optional[str] = None
    institution: Optional[str] = None
    year: Optional[str] = None


class WorkExperienceEntry(BaseModel):
    """Work experience entry model"""
    company: Optional[str] = None
    position: Optional[str] = None
    start_year: Optional[str] = None
    end_year: Optional[str] = None


class ResumeExtractionResponse(BaseModel):
    """Response model for resume extraction"""
    # Personal Information
    full_name: Optional[str] = None
    email: Optional[str] = None
    phone_number: Optional[str] = None
    address: Optional[str] = None
    linkedin_url: Optional[str] = None
    website: Optional[str] = None

    # Professional Summary
    professional_title: Optional[str] = None
    summary: Optional[str] = None

    # Education
    education_entries: Optional[list[EducationEntry]] = None
    highest_degree: Optional[str] = None

    # Work Experience
    work_experience_entries: Optional[list[WorkExperienceEntry]] = None
    years_of_experience: Optional[int] = None
    current_position: Optional[str] = None
    current_employer: Optional[str] = None

    # Skills
    skills: Optional[list[str]] = None
    certifications: Optional[list[str]] = None
    languages: Optional[list[str]] = None

    # Confidence Scores - supports both:
    # - Dict[str, float] (legacy format, 0-100)
    # - Dict[str, dict] (new format with 'overall_confidence' and 'sources')
    confidence_scores: Dict[str, Union[float, Dict[str, Any]]] = {}

    # Raw entities (optional)
    raw_entities: Optional[list[Dict[str, Any]]] = None
