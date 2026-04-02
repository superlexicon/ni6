from pydantic import BaseModel
from typing import Optional, Dict, List, Any, Union


class ResumeData(BaseModel):
    """Standard fields extracted from resume/CV documents"""

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
    education_entries: Optional[List[Dict]] = None
    highest_degree: Optional[str] = None

    # Work Experience
    work_experience_entries: Optional[List[Dict]] = None
    years_of_experience: Optional[int] = None
    current_position: Optional[str] = None
    current_employer: Optional[str] = None

    # Skills
    skills: Optional[List[str]] = None
    certifications: Optional[List[str]] = None
    languages: Optional[List[str]] = None

    # Confidence Scores - supports both:
    # - Dict[str, float] (legacy format, 0-100)
    # - Dict[str, dict] (new format with 'overall_confidence' and 'sources')
    confidence_scores: Dict[str, Union[float, Dict[str, Any]]] = {}
