
from app.core import logger
from .deepface_helper import DeepfaceHelper
from .doctr.document_text_extractor import DocumentTextExtractor
from app.repositories import OTPRepository


class KeyHelper:
    def __init__(self,
                 document_text_extractor: DocumentTextExtractor,
                 otp_repository: OTPRepository,
                 deepface_helper: DeepfaceHelper
                 ):
        self.logger = logger
        self.document_text_extractor = document_text_extractor
        self.otp_repository = otp_repository
        self.deepface_helper = deepface_helper

    async def doctr_verify(self, recovery_image: bytes, email: str) -> bool:
        text = await self.document_text_extractor.extract_text(recovery_image)
        otp = self.otp_repository.get_otp_by_email(email)
        if otp is None:
            return False
        if otp['random_number'] != text:
            return False
        return True
