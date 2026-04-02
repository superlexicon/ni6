from pydantic import BaseModel
from typing import List, Optional, Dict, Any


class DetailedCheckResult(BaseModel):
    """Detailed result for each individual forgery detection check"""
    name: str                                    # "DQ", "Adaptive", "EXIF As Language", etc.
    raw_score: float                             # Raw confidence score from the method
    research_threshold: float                    # Research-based threshold from literature
    analysis: str                                # Human-readable analysis of what the score means
    detected_forgery: bool                       # Whether forgery was detected based on threshold
    research_paper: Optional[str] = None         # Reference to research paper if applicable
    methodology: Optional[str] = None            # Brief description of detection methodology
    processing_time: Optional[float] = None      # Time taken for this check (seconds)


class DetailedPhotoHolmesResults(BaseModel):
    """Comprehensive detailed results from all PhotoHolmes methods with research context"""
    checks: List[DetailedCheckResult]
    total_checks_run: int
    checks_with_detections: int
    overall_forgery_probability: float
    processing_summary: str
    research_sources: List[str] = []  # List of research papers referenced


class DetailedForgeryResponse(BaseModel):
    """Enhanced forgery detection response with detailed analysis"""
    # Existing fields
    ai_generated: Dict[str, Any]
    photoshopped: Dict[str, Any]
    extracted_data: Optional[dict] = None

    # Enhanced detailed results
    detailed_photoholmes_results: DetailedPhotoHolmesResults
    analysis_summary: str                          # Overall summary of all findings
    recommendation: str                            # Actionable recommendation based on results
    confidence_level: str                          # "High", "Medium", "Low" based on overall confidence