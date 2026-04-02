from app.dto.forgery_photoshop import PhotoShoppedData


class PhotoShoppedAnalyzer:
    """
    Analyzes images for photoshop manipulation using multiple detectors.

    Returns a confidence score (0-100) indicating authenticity.
    Higher confidence = more likely the image is authentic and unmodified.
    """

    def analyze(
        self,
        noise_confidence: float,
        psccnet_confidence: float
    ) -> PhotoShoppedData:
        """
        Calculate authenticity confidence from detector scores.

        Args:
            noise_confidence: NoiseSniffer confidence (0-1, higher = more manipulation)
            psccnet_confidence: PSCCNET confidence (0-1, higher = more manipulation)

        Returns:
            PhotoShoppedData with authenticity confidence (inverted) and reasoning
        """
        # Validate inputs
        if not all(isinstance(score, (int, float)) for score in [psccnet_confidence, noise_confidence]):
            raise ValueError("Confidence scores must be numbers")

        if not all(0 <= score <= 1 for score in [psccnet_confidence, noise_confidence]):
            raise ValueError("Confidence scores must be in range [0, 1]")

        # Calculate weighted manipulation score
        # PSCCNET is more reliable (60%), NoiseSniffer as support (40%)
        manipulation_score = (psccnet_confidence * 0.6) + (noise_confidence * 0.4)
        manipulation_score = max(0.0, min(manipulation_score, 1.0))

        # Invert to get authenticity confidence (1 - manipulation = authenticity)
        authenticity_normalized = 1.0 - manipulation_score

        # Convert to 0-100 scale
        confidence = round(authenticity_normalized * 100, 2)

        # Generate reason based on confidence level
        if confidence >= 70:
            reason = f"High authenticity: PSCCNET={psccnet_confidence:.2f}, NoiseSniffer={noise_confidence:.2f}"
        elif confidence >= 40:
            reason = f"Moderate authenticity: PSCCNET={psccnet_confidence:.2f}, NoiseSniffer={noise_confidence:.2f}"
        else:
            reason = f"Low authenticity (likely manipulated): PSCCNET={psccnet_confidence:.2f}, NoiseSniffer={noise_confidence:.2f}"

        return PhotoShoppedData(
            confidence=confidence,
            reason=reason
        )
