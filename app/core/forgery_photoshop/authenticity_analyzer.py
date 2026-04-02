from app.dto import AuthenticityData
from app.config.forgery_thresholds import get_thresholds


class AuthenticityAnalyzer:
    """
    Analyzes image authenticity using multiple detectors.

    Returns raw detector scores along with document-type-specific thresholds.
    Higher raw scores indicate MORE likelihood of forgery.
    """

    def analyze(
        self,
        dq_max_probability: float,
        adaptive_tampered_ratio: float,
        psccnet_confidence: float,
        noise_confidence: float,
        document_type: str = 'default'
    ) -> AuthenticityData:
        """
        Evaluate forgery detection scores against document-type-specific thresholds.

        Args:
            dq_max_probability: DQMethod score (0-1, higher = more likely AI)
            adaptive_tampered_ratio: AdaptiveMethod score (0-1, higher = more likely AI)
            psccnet_confidence: PSCCNET score (0-1, higher = more manipulation)
            noise_confidence: NoiseSniffer score (0-1, higher = more manipulation)
            document_type: Type of document for threshold selection

        Returns:
            AuthenticityData with raw scores, thresholds, and per-detector status
        """
        # Validate inputs
        scores = [dq_max_probability, adaptive_tampered_ratio, psccnet_confidence, noise_confidence]
        if not all(isinstance(score, (int, float)) for score in scores):
            raise ValueError("All confidence scores must be numbers")

        if not all(0 <= score <= 1 for score in scores):
            raise ValueError("All confidence scores must be in range [0, 1]")

        # Get thresholds for this document type
        thresholds = get_thresholds(document_type)

        # Determine detector statuses
        failed_detectors = []
        warning_detectors = []

        # Evaluate DQ (AI Generation)
        dq_status = 'pass'
        if dq_max_probability >= thresholds['dq']['problem']:
            dq_status = 'fail'
            failed_detectors.append('dq')
        elif dq_max_probability >= thresholds['dq']['warning']:
            dq_status = 'warning'
            warning_detectors.append('dq')

        # Evaluate Adaptive (AI Generation)
        adaptive_status = 'pass'
        if adaptive_tampered_ratio >= thresholds['adaptive']['problem']:
            adaptive_status = 'fail'
            failed_detectors.append('adaptive')
        elif adaptive_tampered_ratio >= thresholds['adaptive']['warning']:
            adaptive_status = 'warning'
            warning_detectors.append('adaptive')

        # Evaluate NoiseSniffer (Manipulation)
        noise_status = 'pass'
        if noise_confidence >= thresholds['noisesniffer']['problem']:
            noise_status = 'fail'
            failed_detectors.append('noisesniffer')
        elif noise_confidence >= thresholds['noisesniffer']['warning']:
            noise_status = 'warning'
            warning_detectors.append('noisesniffer')

        # Evaluate PSCCNET (Splicing/Copy-Paste)
        psccnet_status = 'pass'
        if psccnet_confidence >= thresholds['psccnet']['problem']:
            psccnet_status = 'fail'
            failed_detectors.append('psccnet')
        elif psccnet_confidence >= thresholds['psccnet']['warning']:
            psccnet_status = 'warning'
            warning_detectors.append('psccnet')

        # Determine overall decision
        if failed_detectors:
            overall_decision = 'fail'
        elif warning_detectors:
            overall_decision = 'warning'
        else:
            overall_decision = 'pass'

        return AuthenticityData(
            dq={
                'raw_score': dq_max_probability,
                'problem_threshold': thresholds['dq']['problem']
            },
            adaptive={
                'raw_score': adaptive_tampered_ratio,
                'problem_threshold': thresholds['adaptive']['problem']
            },
            noisesniffer={
                'raw_score': noise_confidence,
                'problem_threshold': thresholds['noisesniffer']['problem']
            },
            psccnet={
                'raw_score': psccnet_confidence,
                'problem_threshold': thresholds['psccnet']['problem']
            },
            overall_decision=overall_decision,
            failed_detectors=failed_detectors,
            warning_detectors=warning_detectors
        )

    # _evaluate_detector method removed - now using simplified direct mapping
    # Results are returned as simple {raw_score: float, problem_threshold: float} format
