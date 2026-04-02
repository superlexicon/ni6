from app.dto.forgery_photoshop import AIGeneratedData


class AIGeneratedAnalyzer:
    """
    Analyzes images for AI generation using multiple detectors.

    Returns a confidence score (0-100) indicating authenticity.
    Higher confidence = more likely the image is real and not AI-generated.
    """

    def analyze(
        self,
        dq_max_probability: float,
        adaptive_tampered_ratio: float,
    ) -> AIGeneratedData:
        """
        Calculate authenticity confidence from detector scores.

        Args:
            dq_max_probability: DQMethod max probability (0-1, higher = more likely AI)
            adaptive_tampered_ratio: AdaptiveMethod tampered ratio (0-1, higher = more likely AI)

        Returns:
            AIGeneratedData with authenticity confidence (inverted) and reasoning
        """
        # Input validation
        if not (0 <= dq_max_probability <= 1):
            raise ValueError("dq_max_probability must be between 0 and 1")
        if not (0 <= adaptive_tampered_ratio <= 1):
            raise ValueError("adaptive_tampered_ratio must be between 0 and 1")

        # Calculate weighted AI generation score
        # DQ is more reliable for AI detection (70%), Adaptive as support (30%)
        ai_score = (dq_max_probability * 0.7) + (adaptive_tampered_ratio * 0.3)
        ai_score = max(0.0, min(ai_score, 1.0))

        # Invert to get authenticity confidence (1 - AI = authentic)
        authenticity_normalized = 1.0 - ai_score

        # Convert to 0-100 scale
        confidence = round(authenticity_normalized * 100, 2)

        # Generate reason based on confidence level
        if confidence >= 70:
            reason = f"High authenticity: DQ={dq_max_probability:.2f}, Adaptive={adaptive_tampered_ratio:.2f}"
        elif confidence >= 40:
            reason = f"Moderate authenticity: DQ={dq_max_probability:.2f}, Adaptive={adaptive_tampered_ratio:.2f}"
        else:
            reason = f"Low authenticity (likely AI-generated): DQ={dq_max_probability:.2f}, Adaptive={adaptive_tampered_ratio:.2f}"

        return AIGeneratedData(
            confidence=confidence,
            reason=reason
        )
