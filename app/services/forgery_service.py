import asyncio

from .image_validation_service import ImageValidationService
from .comprehensive_photoholmes_service import ComprehensivePhotoHolmesService
from .detailed_analysis_service import DetailedAnalysisService
from app.core import (AIGeneratedAnalyzer, logger)
from app.core import (PhotoShoppedAnalyzer)
from app.dto import (
    AIGeneratedData,
    ForgeryAndPhotoshoppedResponse,
    PhotoShoppedData,
    DetectForgeryRequest,
    PhotoHolmesResults,
    DetailedForgeryResponse,
)
from app.utils import DecodeBase64


class ForgeryService:
    def __init__(
        self,
        decode_base64: DecodeBase64,
        aigenerated_analyzer: AIGeneratedAnalyzer,
        photoshopped_analyzer: PhotoShoppedAnalyzer,
        image_validation_service: ImageValidationService,
        comprehensive_photoholmes_service: ComprehensivePhotoHolmesService,
    ):
        self.decode_base64 = decode_base64
        self.logger = logger
        self.aigenerated_analyzer = aigenerated_analyzer
        self.photoshopped_analyzer = photoshopped_analyzer
        self.image_validation_service = image_validation_service
        self.comprehensive_photoholmes_service = comprehensive_photoholmes_service
        self.detailed_analysis_service = DetailedAnalysisService()

    async def detect_forgery(self,
                             request: DetectForgeryRequest,
                             text_require: bool) -> ForgeryAndPhotoshoppedResponse:
        image_content = self.decode_base64.decode_base64(request.image)

        try:
            processed_image = await self.image_validation_service.validate_and_convert_image(
                image_bytes=image_content
            )
        except ValueError as e:
            self.logger.error(f"Image validation failed: {str(e)}")
            raise

        # Run comprehensive PhotoHolmes analysis with all 10 methods
        photoholmes_results: PhotoHolmesResults = await self.comprehensive_photoholmes_service.run_all_methods(processed_image)

        # Use DQ and Adaptive methods for AI generation analysis
        ai_generated: AIGeneratedData = self.aigenerated_analyzer.analyze(
            photoholmes_results.dq.max_probability if photoholmes_results.dq else 0.0,
            photoholmes_results.adaptive.tampered_ratio if photoholmes_results.adaptive else 0.0
        )

        # Use NoiseSniffer and PSCCNet for photoshopping analysis
        photoshopped: PhotoShoppedData = self.photoshopped_analyzer.analyze(
            photoholmes_results.noisesniffer.noise_confidence_score if photoholmes_results.noisesniffer else 0.0,
            photoholmes_results.psccnet.psccnet_confidence_score if photoholmes_results.psccnet else 0.0,
        )

        if text_require:
            self.logger.info("Text required, but data extraction temporarily disabled")
            return ForgeryAndPhotoshoppedResponse(
                ai_generated=ai_generated,
                photoshopped=photoshopped,
                photoholmes_results=photoholmes_results,
                extracted_data=None
            )
        else:
            return ForgeryAndPhotoshoppedResponse(
                ai_generated=ai_generated,
                photoshopped=photoshopped,
                photoholmes_results=photoholmes_results,
            )

    async def detect_forgery_detailed(self,
                                   request: DetectForgeryRequest,
                                   text_require: bool) -> DetailedForgeryResponse:
        """Detect forgery with detailed research-backed analysis"""
        image_content = self.decode_base64.decode_base64(request.image)

        try:
            processed_image = await self.image_validation_service.validate_and_convert_image(
                image_bytes=image_content
            )
        except ValueError as e:
            self.logger.error(f"Image validation failed: {str(e)}")
            raise

        # Run comprehensive PhotoHolmes analysis with all methods
        photoholmes_results: PhotoHolmesResults = await self.comprehensive_photoholmes_service.run_all_methods(processed_image)

        # Use DQ and Adaptive methods for AI generation analysis
        ai_generated: AIGeneratedData = self.aigenerated_analyzer.analyze(
            photoholmes_results.dq.max_probability if photoholmes_results.dq else 0.0,
            photoholmes_results.adaptive.tampered_ratio if photoholmes_results.adaptive else 0.0
        )

        # Use NoiseSniffer and PSCCNet for photoshopping analysis
        photoshopped: PhotoShoppedData = self.photoshopped_analyzer.analyze(
            photoholmes_results.noisesniffer.noise_confidence_score if photoholmes_results.noisesniffer else 0.0,
            photoholmes_results.psccnet.psccnet_confidence_score if photoholmes_results.psccnet else 0.0,
        )

        # Convert to dictionaries for detailed response
        ai_generated_data = {
            "authenticity_percentage": ai_generated.authenticity_percentage,
            "verdict": ai_generated.verdict,
            "is_ai_generated": ai_generated.is_ai_generated,
            "reason": ai_generated.reason,
            "confidence": ai_generated.confidence
        }

        photoshopped_data = {
            "is_photoshopped": photoshopped.is_photoshopped,
            "verdict": photoshopped.verdict,
            "authenticity_percentage": photoshopped.authenticity_percentage,
            "reason": photoshopped.reason,
            "confidence": photoshopped.confidence
        }

        extracted_data = None
        if text_require:
            self.logger.info("Text required, but data extraction temporarily disabled")

        # Create detailed response with research context
        detailed_response = self.detailed_analysis_service.create_detailed_response(
            photoholmes_results=photoholmes_results,
            ai_generated_data=ai_generated_data,
            photoshopped_data=photoshopped_data,
            extracted_data=extracted_data
        )

        return detailed_response
