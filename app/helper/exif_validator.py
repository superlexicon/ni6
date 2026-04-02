from PIL import Image
from PIL.ExifTags import TAGS
import io
from typing import List, Tuple
from app.core import logger


class ExifValidationResult:
    """Result of EXIF metadata analysis"""

    def __init__(
        self,
        score: float,
        has_editing_software: bool,
        findings: List[str],
        metadata: dict
    ):
        self.score = score
        self.has_editing_software = has_editing_software
        self.findings = findings
        self.metadata = metadata


class ExifValidator:
    """Validates document authenticity using EXIF metadata analysis"""

    # Known image editing software indicators
    EDITING_SOFTWARE = [
        'photoshop',
        'gimp',
        'paint.net',
        'paintshop',
        'affinity',
        'pixlr',
        'photoscape',
        'inkscape',
        'illustrator',
        'corel',
        'sketch',
        'figma',
        'canva'
    ]

    def __init__(self):
        self.logger = logger

    async def analyze(self, content: bytes) -> ExifValidationResult:
        """
        Analyze EXIF metadata for authenticity indicators

        Args:
            content: Image bytes

        Returns:
            ExifValidationResult with score and findings
        """
        try:
            image = Image.open(io.BytesIO(content))

            # Extract EXIF data
            exif_data = self._extract_exif(image)

            if not exif_data:
                self.logger.warning("No EXIF data found in image")
                return ExifValidationResult(
                    score=50.0,  # Neutral score - missing EXIF is suspicious but not conclusive
                    has_editing_software=False,
                    findings=["No EXIF metadata found (could indicate editing or screenshot)"],
                    metadata={}
                )

            # Analyze EXIF for authenticity
            score, has_editing, findings = self._analyze_exif_data(exif_data)

            return ExifValidationResult(
                score=score,
                has_editing_software=has_editing,
                findings=findings,
                metadata=exif_data
            )

        except Exception as e:
            self.logger.error(f"Error analyzing EXIF data: {str(e)}")
            return ExifValidationResult(
                score=50.0,
                has_editing_software=False,
                findings=[f"Error analyzing EXIF: {str(e)}"],
                metadata={}
            )

    def _extract_exif(self, image: Image.Image) -> dict:
        """
        Extract EXIF metadata from image

        Returns:
            Dictionary of EXIF tag names and values
        """
        try:
            exif = image.getexif()
            if not exif:
                return {}

            exif_data = {}
            for tag_id, value in exif.items():
                tag_name = TAGS.get(tag_id, tag_id)
                # Convert bytes to string if needed
                if isinstance(value, bytes):
                    try:
                        value = value.decode('utf-8', errors='ignore')
                    except:
                        value = str(value)
                exif_data[tag_name] = value

            return exif_data

        except Exception as e:
            self.logger.warning(f"Could not extract EXIF: {str(e)}")
            return {}

    def _analyze_exif_data(self, exif_data: dict) -> Tuple[float, bool, List[str]]:
        """
        Analyze EXIF data for authenticity indicators

        Returns:
            Tuple of (score, has_editing_software, findings_list)
        """
        score = 100.0
        has_editing_software = False
        findings = []

        # Check for editing software
        software_fields = ['Software', 'ProcessingSoftware', 'HostComputer']
        for field in software_fields:
            if field in exif_data:
                software_value = str(exif_data[field]).lower()

                # Check against known editing software
                for editor in self.EDITING_SOFTWARE:
                    if editor in software_value:
                        has_editing_software = True
                        score -= 40.0
                        findings.append(f"Editing software detected: {exif_data[field]}")
                        break

                if not has_editing_software:
                    findings.append(f"Software: {exif_data[field]}")

        # Check for camera/device info
        if 'Make' in exif_data and 'Model' in exif_data:
            make = exif_data['Make']
            model = exif_data['Model']
            findings.append(f"Captured with {make} {model}")
            score += 10.0  # Bonus for having device info
        elif 'Make' in exif_data or 'Model' in exif_data:
            device = exif_data.get('Make') or exif_data.get('Model')
            findings.append(f"Device: {device}")
            score += 5.0
        else:
            score -= 15.0
            findings.append("No camera/device information found")

        # Check for timestamp data
        datetime_fields = ['DateTime', 'DateTimeOriginal', 'DateTimeDigitized']
        datetime_count = sum(1 for field in datetime_fields if field in exif_data)

        if datetime_count >= 2:
            findings.append("Multiple timestamps present (good sign)")
            score += 5.0
        elif datetime_count == 1:
            findings.append("Single timestamp present")
        else:
            score -= 10.0
            findings.append("No timestamp information found")

        # Check for GPS data
        if 'GPSInfo' in exif_data or 'GPS' in str(exif_data):
            findings.append("GPS data present")
            score += 5.0

        # Check for thumbnail
        if 'thumbnail' in str(exif_data).lower() or 'Thumbnail' in exif_data:
            findings.append("Thumbnail present")
            score += 3.0

        # Check for orientation
        if 'Orientation' in exif_data:
            score += 2.0

        # Ensure score is in valid range
        score = max(0.0, min(100.0, score))

        return score, has_editing_software, findings
