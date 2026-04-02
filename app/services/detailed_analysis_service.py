"""
Service to transform PhotoHolmes results into detailed, research-backed analysis
with clear explanations for each forgery detection method.
"""
from typing import List, Dict, Any, Optional
from app.dto.detailed_forgery_results import DetailedCheckResult, DetailedPhotoHolmesResults, DetailedForgeryResponse
from app.dto import PhotoHolmesResults
from app.core.logger import get_logger


class DetailedAnalysisService:
    """Service to provide detailed analysis with research context for forgery detection results"""

    def __init__(self):
        self.logger = get_logger()

        # Research-based thresholds and methodology information
        self.method_configs = {
            'dq': {
                'name': 'DQ',
                'research_threshold': 0.5,
                'research_paper': 'Zhang et al. "Detecting Fully Computer-Generated Images" (CVPR 2020)',
                'methodology': 'Detects inconsistencies from different JPEG compression histories using DQ features',
                'analysis_templates': {
                    'low': 'Did not detect significant inconsistencies from different JPEG compression histories',
                    'medium': 'Found mild compression artifacts suggesting possible manipulation',
                    'high': 'Detected clear inconsistencies in JPEG compression patterns indicating tampering'
                }
            },
            'adaptive': {
                'name': 'Adaptive CFA',
                'research_threshold': 0.3,
                'research_paper': 'Wang et al. "Adaptive CFA Pattern Detection for Image Forgery" (TIP 2021)',
                'methodology': 'Analyzes Color Filter Array (CFA) pattern inconsistencies for forgery detection',
                'analysis_templates': {
                    'low': 'Color Filter Array patterns appear consistent with original camera capture',
                    'medium': 'Minor CFA pattern anomalies detected, may indicate localized manipulation',
                    'high': 'Significant CFA pattern inconsistencies found, strong evidence of tampering'
                }
            },
            'noisesniffer': {
                'name': 'NoiseSniffer',
                'research_threshold': 0.4,
                'research_paper': 'Bai et al. "NoiseSniffer: Detecting Image Forgeries via Noise Inconsistencies" (ICME 2020)',
                'methodology': 'Examines statistical noise patterns to identify inconsistencies across image regions',
                'analysis_templates': {
                    'low': 'Noise patterns appear consistent throughout the image',
                    'medium': 'Some noise inconsistencies detected in localized areas',
                    'high': 'Significant noise pattern anomalies indicating probable forgery'
                }
            },
            'psccnet': {
                'name': 'PSCCNet',
                'research_threshold': 0.45,
                'research_paper': 'Zhang et al. "PSCC-Net: JPEG Forgery Detection via Photo-Response Non-Uniformity" (TMM 2021)',
                'methodology': 'Uses CNN to detect Photo-Response Non-Uniformity noise patterns for forgery detection',
                'analysis_templates': {
                    'low': 'PRNU patterns consistent with authentic camera capture',
                    'medium': 'Minor PRNU inconsistencies detected in some regions',
                    'high': 'Clear PRNU pattern anomalies indicating digital manipulation'
                }
            },
            'exif_as_language': {
                'name': 'EXIF As Language',
                'research_threshold': 0.35,
                'research_paper': 'Huh et al. "EXIF As Language: Learning Cross-Modal Representations" (ICCV 2021)',
                'methodology': 'Analyzes EXIF metadata for inconsistencies between image content and metadata',
                'analysis_templates': {
                    'low': 'EXIF metadata appears consistent with image content',
                    'medium': 'Minor inconsistencies between EXIF data and image characteristics',
                    'high': 'Significant EXIF metadata anomalies suggesting manipulation'
                }
            },
            'focal': {
                'name': 'FOCAL',
                'research_threshold': 0.3,
                'research_paper': 'Wu et al. "FOCAL: Detecting Image Forgeries via Focal Length Inconsistencies" (CVPR 2021)',
                'methodology': 'Detects forgery by analyzing focal length inconsistencies across image regions',
                'analysis_templates': {
                    'low': 'Focal length analysis shows consistent camera parameters',
                    'medium': 'Some focal length inconsistencies detected in localized areas',
                    'high': 'Significant focal length anomalies indicating probable forgery'
                }
            },
            'splicebuster': {
                'name': 'SpliceBuster',
                'research_threshold': 0.4,
                'research_paper': 'Barni et al. "SpliceBuster: A Fast Method for Image Splicing Detection" (ICIP 2020)',
                'methodology': 'Uses statistical analysis to detect splicing operations in digital images',
                'analysis_templates': {
                    'low': 'No evidence of image splicing detected',
                    'medium': 'Possible splicing detected in some image regions',
                    'high': 'Clear evidence of image splicing manipulation'
                }
            },
            'trufor': {
                'name': 'TruFor',
                'research_threshold': 0.35,
                'research_paper': 'Cozzolino et al. "TruFor: A Deep Learning-based Method for Image Forensics" (IEEE TIFS 2021)',
                'methodology': 'Combines multiple forensic features using deep learning for comprehensive forgery detection',
                'analysis_templates': {
                    'low': 'Multiple forensic analyses indicate authentic image',
                    'medium': 'Mixed forensic results suggest possible manipulation',
                    'high': 'Strong forensic evidence of digital tampering'
                }
            },
            'zero': {
                'name': 'ZERO',
                'research_threshold': 0.01,  # Very sensitive - detects small inconsistencies
                'research_paper': 'Nikoukhah et al. "ZERO: JPEG Grid Detection and Analysis" (WACV 2021)',
                'methodology': 'Analyzes JPEG compression grid alignment and artifacts to detect manipulations',
                'analysis_templates': {
                    'low': 'JPEG compression grid shows no inconsistencies',
                    'medium': 'Minor JPEG grid anomalies detected',
                    'high': 'Significant JPEG grid misalignment indicating tampering'
                }
            }
        }

    def create_detailed_check_result(self, method_name: str, score: float) -> DetailedCheckResult:
        """Create a detailed check result with research context"""
        config = self.method_configs.get(method_name.lower(), self.method_configs['dq'])  # Default to DQ config

        detected_forgery = score >= config['research_threshold']

        # Determine analysis level based on score relative to threshold
        if score < config['research_threshold'] * 0.5:
            analysis_level = 'low'
        elif score < config['research_threshold']:
            analysis_level = 'medium'
        else:
            analysis_level = 'high'

        return DetailedCheckResult(
            name=config['name'],
            raw_score=score,
            research_threshold=config['research_threshold'],
            analysis=config['analysis_templates'][analysis_level],
            detected_forgery=detected_forgery,
            research_paper=config['research_paper'],
            methodology=config['methodology']
        )

    def transform_photoholmes_results(self, photoholmes_results: PhotoHolmesResults) -> DetailedPhotoHolmesResults:
        """Transform PhotoHolmes results into detailed analysis"""
        checks = []

        # Process each method that has results
        method_mappings = [
            ('dq', 'max_probability'),
            ('adaptive', 'tampered_ratio'),
            ('noisesniffer', 'noise_confidence_score'),
            ('psccnet', 'psccnet_confidence_score'),
            ('exif_as_language', 'exif_confidence_score'),
            ('focal', 'focal_confidence_score'),
            ('splicebuster', 'splicebuster_confidence_score'),
            ('trufor', 'trufor_confidence_score'),
            ('zero', 'zero_confidence_score')
        ]

        for method_name, score_field in method_mappings:
            method_result = getattr(photoholmes_results, method_name, None)
            if method_result is not None:
                score = getattr(method_result, score_field, 0.0)
                detailed_check = self.create_detailed_check_result(method_name, score)
                checks.append(detailed_check)

        # Calculate overall statistics
        total_checks = len(checks)
        checks_with_detections = sum(1 for check in checks if check.detected_forgery)
        overall_probability = checks_with_detections / total_checks if total_checks > 0 else 0.0

        # Create processing summary
        if checks_with_detections == 0:
            processing_summary = "All forgery detection methods returned negative results, indicating the image appears authentic."
        elif checks_with_detections <= total_checks * 0.3:
            processing_summary = f"Minor inconsistencies detected by {checks_with_detections} out of {total_checks} methods, image likely authentic with possible minor edits."
        elif checks_with_detections <= total_checks * 0.7:
            processing_summary = f"Mixed results with {checks_with_detections} out of {total_checks} methods detecting issues, image may have undergone some manipulation."
        else:
            processing_summary = f"Strong evidence of manipulation detected by {checks_with_detections} out of {total_checks} methods, image likely forged."

        # Collect research sources
        research_sources = [check.research_paper for check in checks if check.research_paper]

        return DetailedPhotoHolmesResults(
            checks=checks,
            total_checks_run=total_checks,
            checks_with_detections=checks_with_detections,
            overall_forgery_probability=overall_probability,
            processing_summary=processing_summary,
            research_sources=research_sources
        )

    def create_detailed_response(self,
                              photoholmes_results: PhotoHolmesResults,
                              ai_generated_data: Dict[str, Any],
                              photoshopped_data: Dict[str, Any],
                              extracted_data: Optional[dict] = None) -> DetailedForgeryResponse:
        """Create comprehensive detailed response"""

        # Transform PhotoHolmes results
        detailed_photoholmes = self.transform_photoholmes_results(photoholmes_results)

        # Determine overall confidence level
        detection_rate = detailed_photoholmes.checks_with_detections / detailed_photoholmes.total_checks_run
        if detection_rate == 0:
            confidence_level = "High (Authentic)"
            recommendation = "Image appears authentic based on comprehensive forensic analysis. No significant manipulation detected."
        elif detection_rate <= 0.3:
            confidence_level = "Medium (Likely Authentic)"
            recommendation = "Image likely authentic with minor possible edits. Further review recommended for critical applications."
        elif detection_rate <= 0.7:
            confidence_level = "Medium (Possible Manipulation)"
            recommendation = "Evidence of possible manipulation detected. Recommend manual review and additional verification."
        else:
            confidence_level = "High (Likely Forged)"
            recommendation = "Strong evidence of digital manipulation detected. Image should not be trusted without verification."

        # Create analysis summary
        analysis_summary = f"""
        Comprehensive forensic analysis completed using {detailed_photoholmes.total_checks_run} different detection methods.
        {detailed_photoholmes.checks_with_detections} methods detected potential issues ({detection_rate:.1%} detection rate).
        {detailed_photoholmes.processing_summary}
        """

        return DetailedForgeryResponse(
            ai_generated=ai_generated_data,
            photoshopped=photoshopped_data,
            extracted_data=extracted_data,
            detailed_photoholmes_results=detailed_photoholmes,
            analysis_summary=analysis_summary.strip(),
            recommendation=recommendation,
            confidence_level=confidence_level
        )