"""
Passport Reference Checker - Compare submitted passport against country-specific templates.

v2.1 Simplified to comparison-based checks only:
- Ghost photo and security thread detection removed (presence-only, not reliable)
- Comparison focuses on features that can be reliably compared to reference:
  - Guilloche patterns (FFT + texture analysis)
  - Color profile (statistics-based comparison)

v2.0 Updated for Dynamic Region Exclusion Pipeline:
- In the new pipeline (v2.0), the submitted image is already cleaned:
  - User text regions (values) have been removed by Step 3
  - MRZ region has been removed by Step 3
  - Face region has been removed by Step 4

Used in Step 5 of the strict linear passport processing pipeline (v2.0).

Configuration: Uses a single config.json with all countries, each with their own
template image file (e.g., THA.jpg, SGP.jpg) in the same directory.
"""

import io
import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from PIL import Image

from app.core.logger import get_logger


class PassportReferenceChecker:
    """
    Compare submitted passport against country-specific reference template.

    v2.0 Updated for Dynamic Region Exclusion Pipeline:
    - Submitted image should already have user-specific regions removed
    - text_regions parameter can be empty if image is pre-cleaned
    - Comparison focuses on security features (guilloche, ghost photo, threads)

    Configuration: Uses a single config.json with all countries. Each country
    has its own template image file (e.g., THA.jpg, SGP.jpg) in the same directory.
    """

    def __init__(self):
        self.logger = get_logger()
        self.templates_dir = Path(__file__).parent.parent / "reference_templates" / "passports"

        # Import settings to get configurable threshold
        from app.config.reference_config import reference_settings

        # Default weights for scoring (used if not specified in config)
        # Only includes features that can reliably compare to reference
        self.default_weights = {
            "guilloche": 0.50,
            "color_profile": 0.50
        }

        # Default threshold for pass/fail (from configurable settings)
        self.default_threshold = reference_settings.similarity_threshold

        # Cache for loaded templates
        self._template_cache: Dict[str, Dict[str, Any]] = {}

        # Cached config (loaded once)
        self._config: Optional[Dict[str, Any]] = None
        self._config_loaded = False

    def compare(
        self,
        submitted_image: bytes,
        text_regions: List[Dict],
        country_code: str
    ) -> Dict[str, Any]:
        """
        Compare submitted passport against reference template.

        v2.0: In the new pipeline, the submitted image is already cleaned
        (user text, MRZ, and face regions removed). text_regions can be empty.

        Args:
            submitted_image: Passport image bytes (preferably cleaned)
            text_regions: List of {"bbox": [x1,y1,x2,y2], "text": "..."} to EXCLUDE
                         Can be empty if image is already cleaned.
            country_code: ISO 3-letter country code

        Returns:
            {
                "passed": bool,
                "similarity_score": float,
                "reason": str,
                "region_scores": {...}
            }
        """
        self.logger.info(f"Starting reference comparison (v2.0) for country: {country_code}")

        # 1. Load reference template for country
        reference = self._load_reference(country_code)
        if not reference:
            # No template available - skip the check (pass with reason)
            self.logger.info(f"No reference template for country: {country_code} - skipping reference check")
            return {
                "passed": True,
                "similarity_score": 1.0,
                "reason": f"Skipped - no reference template for country: {country_code}",
                "region_scores": {},
                "threshold": self.default_threshold
            }

        # 2. Decode submitted image
        try:
            submitted_np = self._decode_image(submitted_image)
        except Exception as e:
            self.logger.error(f"Failed to decode submitted image: {e}")
            return {
                "passed": False,
                "similarity_score": 0.0,
                "reason": f"Failed to decode image: {str(e)}",
                "region_scores": {},
                "threshold": reference.get("threshold", self.default_threshold)
            }

        # 3. Create mask excluding all text regions
        mask = self._create_text_exclusion_mask(
            image_shape=submitted_np.shape,
            text_regions=text_regions
        )

        # 4. Get reference image
        reference_np = reference.get("image")
        if reference_np is None:
            return {
                "passed": False,
                "similarity_score": 0.0,
                "reason": "Reference template has no image",
                "region_scores": {},
                "threshold": reference.get("threshold", self.default_threshold)
            }

        # 5. Align submitted image to reference (handle rotation, scaling)
        aligned = self._align_to_reference(submitted_np, reference_np)

        # 6. Compare non-text regions
        scores = {}
        regions = reference.get("regions", {})

        # 6a. Guilloche pattern comparison (FFT fingerprint)
        guilloche_region = regions.get("guilloche")
        if guilloche_region:
            scores["guilloche"] = self._compare_guilloche(
                aligned, reference_np, mask, guilloche_region
            )
        else:
            scores["guilloche"] = 0.5  # Default if no region defined

        # 6b. Overall color profile
        scores["color_profile"] = self._compare_color_profile(
            aligned, reference_np, mask
        )

        # 7. Weighted overall score
        weights = reference.get("weights", self.default_weights)
        overall_score = self._weighted_score(scores, weights)

        # 8. Determine pass/fail
        threshold = reference.get("threshold", self.default_threshold)
        passed = overall_score >= threshold

        result = {
            "passed": passed,
            "similarity_score": round(overall_score, 4),
            "reason": "Visual similarity verified" if passed else "Visual similarity below threshold",
            "region_scores": {k: round(v, 4) for k, v in scores.items()},
            "threshold": threshold
        }

        self.logger.info(
            f"Reference comparison result: passed={passed}, score={overall_score:.4f}, "
            f"threshold={threshold}, scores={scores}"
        )

        return result

    def _load_config(self) -> Optional[Dict[str, Any]]:
        """Load the single config.json with all countries (cached)."""
        if self._config_loaded:
            return self._config

        config_path = self.templates_dir / "config.json"
        if not config_path.exists():
            self.logger.warning(f"No config.json found at: {config_path}")
            self._config_loaded = True
            return None

        try:
            with open(config_path, 'r') as f:
                self._config = json.load(f)
            self.logger.info(f"Loaded reference config with {len(self._config.get('countries', {}))} countries")
        except Exception as e:
            self.logger.error(f"Failed to load config.json: {e}")
            self._config = None

        self._config_loaded = True
        return self._config

    def _load_reference(self, country_code: str) -> Optional[Dict[str, Any]]:
        """
        Load reference template for a country.

        Uses single config.json with all countries, and loads the country's
        template image from {COUNTRY_CODE}.jpg in the same directory.
        """
        country_code = country_code.upper()

        # Check cache first
        if country_code in self._template_cache:
            return self._template_cache[country_code]

        # Load the main config
        config = self._load_config()
        if not config:
            return None

        # Get default settings
        default_settings = config.get('default', {})

        # Look up country in config
        countries = config.get('countries', {})
        if country_code not in countries:
            self.logger.warning(f"No reference template config for country: {country_code}")
            return None

        country_config = countries[country_code]

        # Merge country config with defaults
        merged_config = {
            "country_code": country_code,
            "name": country_config.get("name", country_code),
            "threshold": country_config.get("threshold", default_settings.get("threshold", self.default_threshold)),
            "weights": {**default_settings.get("weights", self.default_weights), **country_config.get("weights", {})},
            "regions": country_config.get("regions", {})
        }

        # Load reference image from {COUNTRY_CODE}.jpg in same directory
        template_path = self.templates_dir / f"{country_code}.jpg"
        if not template_path.exists():
            # Try PNG as fallback
            template_path = self.templates_dir / f"{country_code}.png"

        if template_path.exists():
            try:
                reference_image = np.array(Image.open(template_path).convert('RGB'))
                merged_config["image"] = reference_image
                self.logger.info(f"Loaded template image for {country_code}: {template_path}")
            except Exception as e:
                self.logger.error(f"Failed to load template image for {country_code}: {e}")
                merged_config["image"] = None
        else:
            self.logger.warning(f"No template image found at: {template_path}")
            merged_config["image"] = None

        # Cache the result
        self._template_cache[country_code] = merged_config
        return merged_config

    def _decode_image(self, image_bytes: bytes) -> np.ndarray:
        """Decode image bytes to numpy array."""
        with Image.open(io.BytesIO(image_bytes)) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            return np.array(img)

    def _create_text_exclusion_mask(
        self,
        image_shape: Tuple[int, int, int],
        text_regions: List[Dict]
    ) -> np.ndarray:
        """
        Create binary mask where 0 = text region (exclude), 1 = compare.

        Args:
            image_shape: (height, width, channels)
            text_regions: List of {"bbox": [x1,y1,x2,y2], "text": "..."} normalized 0-1

        Returns:
            Binary mask of shape (height, width)
        """
        height, width = image_shape[:2]
        mask = np.ones((height, width), dtype=np.uint8)

        for region in text_regions:
            bbox = region.get('bbox', [])
            if len(bbox) != 4:
                continue

            # Handle both normalized (0-1) and pixel coordinates
            x1, y1, x2, y2 = bbox

            # If coordinates are normalized (0-1), convert to pixels
            if all(0 <= c <= 1 for c in [x1, y1, x2, y2]):
                x1 = int(x1 * width)
                y1 = int(y1 * height)
                x2 = int(x2 * width)
                y2 = int(y2 * height)
            else:
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Clamp to image bounds
            x1 = max(0, min(x1, width))
            x2 = max(0, min(x2, width))
            y1 = max(0, min(y1, height))
            y2 = max(0, min(y2, height))

            # Exclude this region (add padding around text for safety)
            padding = 3  # pixels
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(width, x2 + padding)
            y2 = min(height, y2 + padding)

            mask[y1:y2, x1:x2] = 0

        text_coverage = 1.0 - (np.sum(mask) / (height * width))
        self.logger.debug(f"Text exclusion mask created: {text_coverage*100:.1f}% of image masked")

        return mask

    def _align_to_reference(
        self,
        submitted: np.ndarray,
        reference: np.ndarray
    ) -> np.ndarray:
        """
        Align submitted image to reference (handle rotation, scaling).

        Currently implements basic resizing. Can be extended with feature matching.
        """
        # Simple resize to match reference dimensions
        if submitted.shape[:2] != reference.shape[:2]:
            pil_submitted = Image.fromarray(submitted)
            pil_submitted = pil_submitted.resize(
                (reference.shape[1], reference.shape[0]),
                Image.Resampling.LANCZOS
            )
            return np.array(pil_submitted)
        return submitted

    def _extract_region(
        self,
        image: np.ndarray,
        region: Dict,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Extract a region from the image.

        Args:
            image: Image as numpy array
            region: {"x1": 0.1, "y1": 0.3, "x2": 0.5, "y2": 0.7} normalized coords
            normalize: If True, coordinates are 0-1, else pixels

        Returns:
            Cropped region
        """
        height, width = image.shape[:2]

        if normalize:
            x1 = int(region.get("x1", 0) * width)
            y1 = int(region.get("y1", 0) * height)
            x2 = int(region.get("x2", 1) * width)
            y2 = int(region.get("y2", 1) * height)
        else:
            x1 = int(region.get("x1", 0))
            y1 = int(region.get("y1", 0))
            x2 = int(region.get("x2", width))
            y2 = int(region.get("y2", height))

        # Clamp to bounds
        x1 = max(0, min(x1, width))
        x2 = max(0, min(x2, width))
        y1 = max(0, min(y1, height))
        y2 = max(0, min(y2, height))

        return image[y1:y2, x1:x2].copy()

    def _create_black_region_mask(self, image_gray: np.ndarray, threshold: int = 30) -> np.ndarray:
        """
        Create mask where 1 = valid pixels (non-black), 0 = black/excluded.

        Black regions are from:
        - Template: black bars masking user data
        - Submitted: black fill from cleaning pipeline
        """
        return (image_gray > threshold).astype(np.uint8)

    def _compare_guilloche(
        self,
        aligned: np.ndarray,
        reference: np.ndarray,
        mask: np.ndarray,
        region: Dict
    ) -> float:
        """
        Compare guilloche patterns, excluding black regions from both images.

        Combines multiple checks:
        1. High-frequency FFT (fine pattern details) - on full images, FFT handles black regions well
        2. Masked correlation (on valid pixels only)
        3. Texture analysis (on valid pixels only)
        4. Histogram similarity (on valid pixels only)
        """
        try:
            # Extract regions
            aligned_region = self._extract_region(aligned, region)
            reference_region = self._extract_region(reference, region)

            # Resize to match if needed
            if aligned_region.shape != reference_region.shape:
                pil_aligned = Image.fromarray(aligned_region)
                pil_aligned = pil_aligned.resize(
                    (reference_region.shape[1], reference_region.shape[0]),
                    Image.Resampling.LANCZOS
                )
                aligned_region = np.array(pil_aligned)

            # Convert to grayscale
            aligned_gray = np.mean(aligned_region, axis=2).astype(np.float64)
            reference_gray = np.mean(reference_region, axis=2).astype(np.float64)

            # Create combined mask excluding black regions from BOTH images
            mask_aligned = self._create_black_region_mask(aligned_gray)
            mask_reference = self._create_black_region_mask(reference_gray)
            combined_mask = (mask_aligned & mask_reference).astype(bool)

            # Log valid pixel ratio for debugging
            valid_ratio = np.sum(combined_mask) / combined_mask.size
            self.logger.debug(f"Guilloche: {valid_ratio*100:.1f}% valid pixels for comparison")

            # Extract only valid pixels
            aligned_valid = aligned_gray[combined_mask]
            reference_valid = reference_gray[combined_mask]

            # 1. High-Frequency FFT (on full images, FFT handles black regions well)
            hf_fft_score = self._high_freq_fft_similarity(aligned_gray, reference_gray)

            # 2. Masked correlation on valid pixels only
            ssim_score = self._masked_correlation(aligned_valid, reference_valid)

            # 3. Texture on valid pixels only
            texture_score = self._masked_texture_similarity(aligned_valid, reference_valid)

            # 4. Histogram similarity on valid pixels only
            edge_score = self._masked_edge_similarity(aligned_valid, reference_valid)

            # Combine with weights
            combined = (
                0.30 * hf_fft_score +     # FFT works well
                0.25 * ssim_score +       # Masked correlation
                0.25 * texture_score +    # Masked texture
                0.20 * edge_score         # Masked edge (histogram)
            )
            combined = float(np.clip(combined, 0, 1))

            self.logger.debug(
                f"Guilloche: valid={valid_ratio*100:.1f}%, HF-FFT={hf_fft_score:.3f}, "
                f"SSIM={ssim_score:.3f}, Texture={texture_score:.3f}, Edge={edge_score:.3f} → {combined:.3f}"
            )
            return combined

        except Exception as e:
            self.logger.error(f"Guilloche comparison failed: {e}")
            return 0.5

    def _masked_correlation(self, aligned_valid: np.ndarray, reference_valid: np.ndarray) -> float:
        """Pearson correlation on valid pixels only."""
        if len(aligned_valid) < 100:
            return 0.5

        # Normalize
        a_norm = aligned_valid - np.mean(aligned_valid)
        r_norm = reference_valid - np.mean(reference_valid)

        std_a = np.std(a_norm)
        std_r = np.std(r_norm)

        if std_a < 1e-10 or std_r < 1e-10:
            return 0.5

        corr = np.mean(a_norm * r_norm) / (std_a * std_r)
        return float(np.clip((corr + 1) / 2, 0, 1))  # Map [-1,1] to [0,1]

    def _masked_texture_similarity(self, aligned_valid: np.ndarray, reference_valid: np.ndarray) -> float:
        """Compare texture statistics on valid pixels."""
        if len(aligned_valid) < 100:
            return 0.5

        # Compare statistical texture features
        var_a = np.var(aligned_valid)
        var_r = np.var(reference_valid)

        skew_a = self._skewness(aligned_valid)
        skew_r = self._skewness(reference_valid)

        kurt_a = self._kurtosis(aligned_valid)
        kurt_r = self._kurtosis(reference_valid)

        # Ratio-based comparison (robust to magnitude differences)
        var_score = min(var_a, var_r) / (max(var_a, var_r) + 1e-10)
        skew_score = 1 - min(abs(skew_a - skew_r), 2) / 2  # Difference normalized
        kurt_score = 1 - min(abs(kurt_a - kurt_r), 10) / 10  # Difference normalized

        return float(np.clip((var_score + skew_score + kurt_score) / 3, 0, 1))

    def _masked_edge_similarity(self, aligned_valid: np.ndarray, reference_valid: np.ndarray) -> float:
        """Compare edge statistics on valid pixels using histogram distribution."""
        if len(aligned_valid) < 100:
            return 0.5

        # Compute gradient statistics on valid pixels
        # Since we don't have spatial info, use intensity distribution as proxy
        hist_a, _ = np.histogram(aligned_valid, bins=32, range=(0, 255))
        hist_r, _ = np.histogram(reference_valid, bins=32, range=(0, 255))

        # Normalize histograms
        hist_a = hist_a / (np.sum(hist_a) + 1e-10)
        hist_r = hist_r / (np.sum(hist_r) + 1e-10)

        # Histogram intersection
        return float(np.sum(np.minimum(hist_a, hist_r)))

    def _skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of distribution."""
        mean = np.mean(data)
        std = np.std(data)
        if std < 1e-10:
            return 0.0
        return float(np.mean(((data - mean) / std) ** 3))

    def _kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of distribution."""
        mean = np.mean(data)
        std = np.std(data)
        if std < 1e-10:
            return 0.0
        return float(np.mean(((data - mean) / std) ** 4) - 3)

    def _high_freq_fft_similarity(self, aligned_gray: np.ndarray, reference_gray: np.ndarray) -> float:
        """Compare high-frequency FFT components (where guilloche patterns live)."""
        aligned_fft = np.fft.fft2(aligned_gray)
        reference_fft = np.fft.fft2(reference_gray)

        aligned_fft = np.fft.fftshift(aligned_fft)
        reference_fft = np.fft.fftshift(reference_fft)

        aligned_mag = np.abs(aligned_fft)
        reference_mag = np.abs(reference_fft)

        # High-pass filter
        rows, cols = aligned_mag.shape
        crow, ccol = rows // 2, cols // 2
        radius = min(rows, cols) // 8

        y, x = np.ogrid[:rows, :cols]
        high_pass_mask = (x - ccol)**2 + (y - crow)**2 > radius**2

        aligned_hf = aligned_mag[high_pass_mask].flatten()
        reference_hf = reference_mag[high_pass_mask].flatten()

        aligned_norm = aligned_hf / (np.linalg.norm(aligned_hf) + 1e-10)
        reference_norm = reference_hf / (np.linalg.norm(reference_hf) + 1e-10)

        return float(np.clip(np.dot(aligned_norm, reference_norm), 0, 1))


    def _compare_color_profile(
        self,
        aligned: np.ndarray,
        reference: np.ndarray,
        mask: np.ndarray
    ) -> float:
        """
        Compare overall color profiles on non-text regions.
        Excludes black regions from both images for accurate comparison.
        Uses color statistics (mean, std) which is more robust to lighting differences.
        """
        try:
            # Resize mask to match aligned image if needed
            if mask.shape != aligned.shape[:2]:
                from PIL import Image as PILImage
                mask_pil = PILImage.fromarray(mask * 255)
                mask_pil = mask_pil.resize((aligned.shape[1], aligned.shape[0]), PILImage.Resampling.NEAREST)
                mask = np.array(mask_pil) > 128

            # Convert to grayscale for black region detection
            aligned_gray = np.mean(aligned, axis=2)
            reference_gray = np.mean(reference, axis=2)

            # Create black region masks for each image independently
            mask_aligned_black = self._create_black_region_mask(aligned_gray).astype(bool)
            mask_reference_black = self._create_black_region_mask(reference_gray).astype(bool)

            # For aligned image: exclude text AND black regions
            aligned_valid_mask = mask & mask_aligned_black
            # For reference image: exclude text AND black regions
            reference_valid_mask = mask & mask_reference_black

            # Log valid pixel ratios
            aligned_valid_ratio = np.sum(aligned_valid_mask) / aligned_valid_mask.size
            reference_valid_ratio = np.sum(reference_valid_mask) / reference_valid_mask.size
            self.logger.debug(f"Color profile: aligned={aligned_valid_ratio*100:.1f}% valid, reference={reference_valid_ratio*100:.1f}% valid")

            # Extract valid pixels from each image independently
            aligned_valid = aligned[aligned_valid_mask]
            reference_valid = reference[reference_valid_mask]

            if len(aligned_valid) < 100 or len(reference_valid) < 100:
                return 0.5

            # Compare color statistics (more robust to lighting differences)
            scores = []
            for channel in range(3):  # R, G, B
                # Compute statistics for each channel
                mean_a = np.mean(aligned_valid[:, channel])
                mean_r = np.mean(reference_valid[:, channel])
                std_a = np.std(aligned_valid[:, channel])
                std_r = np.std(reference_valid[:, channel])

                # Mean comparison (ratio-based, robust to magnitude)
                mean_ratio = min(mean_a, mean_r) / (max(mean_a, mean_r) + 1e-10)
                # Std comparison (texture similarity)
                std_ratio = min(std_a, std_r) / (max(std_a, std_r) + 1e-10)

                # Combine mean and std scores
                channel_score = (mean_ratio + std_ratio) / 2
                scores.append(channel_score)

            # Average across channels
            similarity = float(np.mean(scores))

            self.logger.debug(f"Color profile similarity (stats-based): {similarity:.4f}")
            return similarity

        except Exception as e:
            self.logger.error(f"Color profile comparison failed: {e}")
            return 0.5  # Neutral score on error

    def _weighted_score(
        self,
        scores: Dict[str, float],
        weights: Dict[str, float]
    ) -> float:
        """Calculate weighted overall score."""
        total_weight = 0.0
        weighted_sum = 0.0

        for key, weight in weights.items():
            if key in scores:
                weighted_sum += scores[key] * weight
                total_weight += weight

        if total_weight == 0:
            return 0.0

        return weighted_sum / total_weight

    def has_template(self, country_code: str) -> bool:
        """Check if a template exists for a country in the config."""
        country_code = country_code.upper()
        config = self._load_config()
        if not config:
            return False
        return country_code in config.get('countries', {})


# Global instance
passport_reference_checker = PassportReferenceChecker()
