from typing import Dict
from app.repositories.face_biometrics_repository import FaceBiometricsRepository, DuplicateFaceError
from app.repositories.user_key_repository import UserKeyRepository
from app.services.face_extraction_service import FaceExtractionService
from app.core.logger import get_logger
from app.helper.face_recognition_factory import get_model_name

class VoluntarySelfieService:
    """Service for voluntary selfie submissions"""

    def __init__(self):
        self.logger = get_logger()
        self.face_biometrics_repo = FaceBiometricsRepository()
        self.user_key_repo = UserKeyRepository()

    async def submit_voluntary_selfie(
        self,
        public_key: str,
        selfie_image_bytes: bytes
    ) -> Dict:
        """
        Submit voluntary selfie to improve face matching accuracy.

        Steps:
        1. Verify user exists in system
        2. Extract face from selfie
        3. Validate face quality
        4. Store in face_biometrics

        Args:
            public_key: User's public key
            selfie_image_bytes: Selfie image

        Returns:
            Dict with success status and quality metrics
        """

        # Step 1: Verify user exists
        user_key = self.user_key_repo.get_key_by_public_key(public_key)
        if not user_key:
            raise ValueError("User not found - complete verification first")

        mobile_number = user_key.get('mobile_number')
        country_code = user_key.get('country_code')

        # Step 2: Extract face
        face_extraction_service = FaceExtractionService()

        extracted_face = await face_extraction_service.extract_face_embedding(
            image_bytes=selfie_image_bytes,
            public_key=public_key,
            document_type="voluntary_selfie"
        )

        if not extracted_face or not extracted_face.get('face_image_b64'):
            raise ValueError("Face extraction failed - no face detected")

        # Step 3: Validate quality
        quality_score = extracted_face.get('extraction_confidence', 0.0)
        anti_spoofing_score = extracted_face.get('anti_spoofing_score', 0.0)

        if quality_score < 0.7:
            raise ValueError(
                f"Face quality too low: {quality_score:.2f}. "
                "Please submit a clearer image."
            )

        if anti_spoofing_score < 0.7:
            raise ValueError(
                f"Liveness check failed: {anti_spoofing_score:.2f}. "
                "Please submit a live selfie."
            )

        # Step 4: Store in face_biometrics
        user_identity_id = extracted_face.get('user_identity_id')
        face_embedding = extracted_face.get('face_embedding')

        if not user_identity_id:
            raise ValueError("No user identity ID found")
        if not face_embedding:
            raise ValueError("No face embedding extracted")

        try:
            biometric_id = self.face_biometrics_repo.create_face_biometric(
                user_identity_id=user_identity_id,
                face_embedding=face_embedding,
                model_name=get_model_name()
            )
        except DuplicateFaceError as e:
            raise ValueError(f"This face is already registered in the system: {str(e)}")

        if not biometric_id:
            raise ValueError("Failed to store face biometric")

        self.logger.info(
            f"✅ Stored voluntary selfie for user {public_key} "
            f"(quality: {quality_score:.2f}, liveness: {anti_spoofing_score:.2f})"
        )

        return {
            'success': True,
            'message': 'Voluntary selfie submitted successfully',
            'face_quality_score': quality_score,
            'anti_spoofing_score': anti_spoofing_score,
            'biometric_id': biometric_id
        }
