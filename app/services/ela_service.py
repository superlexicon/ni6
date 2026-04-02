from PIL import Image
import io
import numpy as np
from app.core import logger


class ELAService:
    """
    Error Level Analysis (ELA) Service
    Detects image manipulation by analyzing JPEG compression artifacts
    """

    def __init__(self):
        self.logger = logger

    async def analyze(self, content: bytes, quality: int = 90) -> float:
        """
        Perform Error Level Analysis on an image

        Args:
            content: Image bytes
            quality: JPEG quality level for resave (default 90)

        Returns:
            Manipulation score (0-100, higher = more suspicious)
        """
        try:
            # Open original image
            original = Image.open(io.BytesIO(content))

            # Convert to RGB if needed
            if original.mode != 'RGB':
                original = original.convert('RGB')

            # Resave at specified quality
            temp_buffer = io.BytesIO()
            original.save(temp_buffer, 'JPEG', quality=quality)
            temp_buffer.seek(0)

            # Load resaved image
            resaved = Image.open(temp_buffer)

            # Calculate pixel-level differences
            original_array = np.array(original, dtype=np.float64)
            resaved_array = np.array(resaved, dtype=np.float64)

            # Compute absolute difference
            diff = np.abs(original_array - resaved_array)

            # Calculate metrics
            mean_diff = np.mean(diff)
            max_diff = np.max(diff)
            std_diff = np.std(diff)

            # Calculate manipulation score
            # Higher differences indicate more manipulation
            # Normalize to 0-100 scale
            manipulation_score = min(100.0, (mean_diff / 255.0) * 200)

            # Adjust based on maximum difference (spotty high differences = manipulation)
            if max_diff > 50:
                manipulation_score += min(20.0, (max_diff / 255.0) * 50)

            # Adjust based on standard deviation (uneven differences = manipulation)
            if std_diff > 10:
                manipulation_score += min(15.0, (std_diff / 25.5) * 15)

            # Cap at 100
            manipulation_score = min(100.0, manipulation_score)

            self.logger.info(
                f"ELA Analysis: mean_diff={mean_diff:.2f}, "
                f"max_diff={max_diff:.2f}, std_diff={std_diff:.2f}, "
                f"score={manipulation_score:.2f}"
            )

            return round(manipulation_score, 2)

        except Exception as e:
            self.logger.error(f"Error in ELA analysis: {str(e)}")
            # Return neutral score on error
            return 50.0
