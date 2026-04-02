from pydantic import BaseModel

from typing import Optional, Dict, Any


class AIGeneratedData(BaseModel):
    authenticity_percentage: float
    verdict: str
    is_ai_generated: bool
    reason: str
    confidence: Optional[float] = None


class PhotoShoppedData(BaseModel):
    is_photoshopped: bool
    verdict: str
    authenticity_percentage: float
    reason: str
    confidence: Optional[float] = None


# Individual PhotoHolmes method results
class DQMethodData(BaseModel):
    max_probability: float


class NoiseSnifferData(BaseModel):
    noise_confidence_score: float


class PsccnetMethodData(BaseModel):
    psccnet_confidence_score: float


class AdaptiveMethodData(BaseModel):
    tampered_ratio: float


class CatnetMethodData(BaseModel):
    catnet_confidence_score: float


class ExifAsLanguageData(BaseModel):
    exif_confidence_score: float
    exif_analysis: Optional[Dict[str, Any]] = None


class FocalMethodData(BaseModel):
    focal_confidence_score: float


class SplicebusterMethodData(BaseModel):
    splicebuster_confidence_score: float


class TruforMethodData(BaseModel):
    trufor_confidence_score: Optional[float] = None  # None when skipped on CPU
    trufor_heatmap_score: Optional[float] = None


class ZeroMethodData(BaseModel):
    zero_confidence_score: float


class PhotoHolmesResults(BaseModel):
    """Comprehensive results from all 10 PhotoHolmes methods"""
    # Original 4 methods
    dq: Optional[DQMethodData] = None
    adaptive: Optional[AdaptiveMethodData] = None
    noisesniffer: Optional[NoiseSnifferData] = None
    psccnet: Optional[PsccnetMethodData] = None

    # Additional 6 methods
    catnet: Optional[CatnetMethodData] = None
    exif_as_language: Optional[ExifAsLanguageData] = None
    focal: Optional[FocalMethodData] = None
    splicebuster: Optional[SplicebusterMethodData] = None
    trufor: Optional[TruforMethodData] = None
    zero: Optional[ZeroMethodData] = None

    # Overall assessment
    total_methods_run: int = 0
    methods_with_detections: int = 0
    overall_forgery_probability: float = 0.0


class ForgeryAndPhotoshoppedResponse(BaseModel):
    ai_generated:  AIGeneratedData
    photoshopped: PhotoShoppedData
    photoholmes_results: Optional[PhotoHolmesResults] = None
    extracted_data: Optional[dict] = None
