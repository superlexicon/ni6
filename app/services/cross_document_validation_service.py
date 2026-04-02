from typing import List, Dict, Optional, Any, Tuple
from pydantic import BaseModel

from app.utils.string_matching import jaro_winkler_similarity
import asyncio

from app.schemas import (
    PassportData,
    BankStatementData,
    SelfieOTPData,
    TaxStatementData,
    WealthDeclarationData,
    ResumeData
)
from app.schemas.id_card_schema import IDCardData
from app.dto.document_analysis import CrossValidationResult
from app.helper.deepface_helper import DeepfaceHelper, convert_numpy_types
from app.repositories import OTPRepository, UserIdentityRepository
from app.core import logger


class DocumentData(BaseModel):
    """Container for document data and metadata"""
    filename: str
    document_type: str
    extracted_data: Any  # Union of all schema types


class CrossDocumentValidationService:
    """
    Service for validating consistency across multiple documents.
    Performs smart field matching considering each document type's schema.
    """

    def __init__(self, otp_repository: OTPRepository, user_identity_repo: UserIdentityRepository):
        self.logger = logger
        self.deepface_helper = DeepfaceHelper()
        self.otp_repository = otp_repository
        self.user_identity_repo = user_identity_repo

    async def validate(
        self,
        documents: List[DocumentData],
        document_contents: List[Tuple[str, bytes, bool]],
        client_public_key: str
    ) -> CrossValidationResult:
        """
        Validate consistency across multiple documents.

        Args:
            documents: List of DocumentData with extracted information
            document_contents: List of (filename, bytes, is_pdf) for face matching
            client_public_key: Client's public key for OTP lookup from database

        Returns:
            CrossValidationResult with validation details
        """
        try:
            findings = []

            # Extract all names from all documents (most common field)
            names = self._extract_all_names(documents)
            name_consistency_score = self._fuzzy_match_all(names, 'name') if len(names) > 1 else 100.0

            if names:
                findings.append(f"Names found: {', '.join(names)}")
                if len(names) > 1:
                    findings.append(f"Name consistency: {name_consistency_score:.1f}%")

            # Extract dates of birth from passport (single source - no cross-check)
            passport_dob = self._extract_passport_dob(documents)
            dob_age_consistency = None
            if passport_dob:
                findings.append(f"Passport DOB: {passport_dob}")

            # Extract addresses from documents that actually have them
            addresses = self._extract_all_addresses(documents)
            address_consistency_score = self._fuzzy_match_all(addresses, 'address') if len(addresses) > 1 else 100.0

            if addresses:
                findings.append(f"Addresses found: {len(addresses)} documents with address")
                if len(addresses) > 1:
                    findings.append(f"Address consistency: {address_consistency_score:.1f}%")

            # Age estimation from face analysis (if selfie available)
            age_estimation_result = await self._estimate_age_from_selfie(documents, document_contents)
            if age_estimation_result:
                findings.extend(age_estimation_result['findings'])
                dob_age_consistency = age_estimation_result['dob_age_consistency']

            # Face matching between selfie and passport
            face_match_score = await self._match_faces(documents, document_contents, client_public_key)

            if face_match_score is not None:
                findings.append(f"Face match score: {face_match_score:.1f}%")

            # OTP verification
            otp_verified = self._verify_otp(documents, client_public_key)

            # Only add finding if there's a selfie
            has_selfie = any(doc.document_type == 'selfie' for doc in documents)
            if has_selfie:
                findings.append(f"OTP verification: {'Passed' if otp_verified else 'Failed'}")

            # Overall validation - only check fields that actually exist in multiple documents
            overall_match = True  # Start with true, only fail if actual inconsistencies found

            # Name consistency only matters if multiple names exist
            if len(names) > 1 and name_consistency_score < 70.0:
                overall_match = False

            # Address consistency only matters if multiple addresses exist
            if len(addresses) > 1 and address_consistency_score < 70.0:
                overall_match = False

            # Face matching only matters if both selfie and passport exist
            if face_match_score is not None and face_match_score < 70.0:
                overall_match = False

            # OTP verification only matters if selfie exists
            if has_selfie and not otp_verified:
                overall_match = False

            # Age-DOB consistency is a bonus check, not requirement
            if dob_age_consistency is False:
                findings.append("Warning: Face age estimation doesn't match passport DOB")

            face_score_str = f"{face_match_score:.1f}" if face_match_score is not None else "N/A"
            self.logger.info(
                f"Cross-validation: Name={name_consistency_score:.1f}% ({len(names)} docs), "
                f"Address={address_consistency_score:.1f}% ({len(addresses)} docs), "
                f"Face={face_score_str}%, "
                f"Overall={'PASS' if overall_match else 'FAIL'}"
            )

            return CrossValidationResult(
                name_consistency_score=name_consistency_score,
                dob_consistency=True,  # Always true since only single source (passport)
                address_consistency_score=address_consistency_score,
                face_match_score=face_match_score,
                otp_verified=otp_verified,
                overall_match=overall_match,
                detailed_findings=findings
            )

        except Exception as e:
            self.logger.error(f"Cross-document validation failed: {str(e)}")
            return CrossValidationResult(
                name_consistency_score=0.0,
                dob_consistency=False,
                address_consistency_score=0.0,
                otp_verified=False,
                overall_match=False,
                detailed_findings=[f"Validation error: {str(e)}"]
            )

    def _extract_all_names(self, documents: List[DocumentData]) -> List[str]:
        """Extract names from all documents"""
        names = []

        for doc in documents:
            data = doc.extracted_data

            # Passport
            if isinstance(data, PassportData) and data.full_name:
                names.append(data.full_name)
            # Bank statement
            elif isinstance(data, BankStatementData) and data.account_holder_name:
                names.append(data.account_holder_name)
            # Tax statement
            elif isinstance(data, TaxStatementData) and data.taxpayer_name:
                names.append(data.taxpayer_name)
            # Wealth declaration
            elif isinstance(data, WealthDeclarationData) and data.declarant_name:
                names.append(data.declarant_name)
            # Resume
            elif isinstance(data, ResumeData) and data.full_name:
                names.append(data.full_name)
            # ID cards (PAN, UAE TRC, etc.)
            elif isinstance(data, IDCardData) and data.full_name:
                names.append(data.full_name)

        return names

    def _extract_passport_dob(self, documents: List[DocumentData]) -> Optional[str]:
        """Extract date of birth from passport (single source)"""
        for doc in documents:
            data = doc.extracted_data
            # Passport
            if isinstance(data, PassportData) and data.date_of_birth:
                return data.date_of_birth
        return None

    def _extract_all_addresses(self, documents: List[DocumentData]) -> List[str]:
        """Extract addresses from all documents"""
        addresses = []

        for doc in documents:
            data = doc.extracted_data

            # Bank statement
            if isinstance(data, BankStatementData) and data.address:
                addresses.append(data.address)
            # Tax statement
            elif isinstance(data, TaxStatementData) and data.address:
                addresses.append(data.address)
            # Resume
            elif isinstance(data, ResumeData) and data.address:
                addresses.append(data.address)

        return addresses

    def _fuzzy_match_all(self, values: List[str], field_type: str) -> float:
        """
        Fuzzy match all values and return average similarity score.

        Args:
            values: List of values to compare
            field_type: Type of field ('name', 'address', etc.)

        Returns:
            Average similarity score (0-100)
        """
        if not values or len(values) < 2:
            return 100.0  # Single value = consistent by default

        # Compare all pairs and average the scores
        total_score = 0.0
        comparisons = 0

        for i in range(len(values)):
            for j in range(i + 1, len(values)):
                similarity = jaro_winkler_similarity(values[i], values[j])
                total_score += similarity * 100
                comparisons += 1

        avg_score = total_score / comparisons if comparisons > 0 else 100.0

        self.logger.info(
            f"Fuzzy match for {field_type}: {avg_score:.1f}% "
            f"(compared {comparisons} pairs)"
        )

        return avg_score

    def _exact_match_all(self, values: List[str]) -> bool:
        """
        Check if all values match exactly (after normalization).

        Args:
            values: List of values to compare

        Returns:
            True if all values match exactly
        """
        if not values or len(values) < 2:
            return True  # Single value = consistent

        # Normalize values (remove spaces, convert to lowercase)
        normalized = [v.replace(' ', '').lower() for v in values]

        # Check if all are identical
        return len(set(normalized)) == 1

    async def _match_faces(
        self,
        documents: List[DocumentData],
        document_contents: List[Tuple[str, bytes, bool, Optional[str], Optional[str]]],
        client_public_key: str
    ) -> Optional[float]:
        """
        Match faces between selfie and passport.

        Args:
            documents: List of DocumentData
            document_contents: List of (filename, bytes, is_pdf, explicit_file_type, explicit_document_type)
            client_public_key: Client's public key for database lookup

        Returns:
            Face match score (0-100) or None if no face documents found
        """
        try:
            # Find selfie and passport documents
            selfie_content = None
            passport_content = None

            for doc, (filename, content, is_pdf, explicit_file_type, explicit_document_type) in zip(documents, document_contents):
                if doc.document_type == 'selfie' and not is_pdf:
                    selfie_content = content
                elif doc.document_type == 'passport' and not is_pdf:
                    passport_content = content

            if not selfie_content:
                self.logger.info("No selfie found for face matching")
                return None

            # Prioritized face matching logic
            if passport_content:
                # Case 1: Passport present - match selfie directly against passport
                self.logger.info("Passport found - performing direct selfie-to-passport face matching")

                results = await self.deepface_helper.verify_faces(
                    selfie_content,
                    passport_content
                )
            else:
                # Case 2: No passport - lookup face from database using client_public_key
                self.logger.info("No passport found - performing selfie-to-database face matching")

                # Get stored face biometric from database
                stored_face_data = await self.user_identity_repo.get_face_biometric_by_public_key(client_public_key)

                if not stored_face_data or not stored_face_data.face_embedding:
                    self.logger.warning(f"No stored face biometric found for client: {client_public_key[:8]}...")
                    return None

                # Perform face matching against stored embedding
                results = await self.deepface_helper.verify_selfie_against_records(
                    selfie_image=selfie_content,
                    stored_embedding=stored_face_data.face_embedding,
                    verification_request=None  # Not needed for basic matching
                )

            if not results:
                return 0.0

            # Handle different result formats from the two matching methods
            if passport_content:
                # Direct face-to-face matching: results is a list of FaceVerificationResult objects
                avg_match = sum(r.percentage for r in results) / len(results)
            else:
                # Database matching: results is a FaceVerificationResponse object
                if hasattr(results, 'face_match_result') and results.face_match_result:
                    # Convert distance to similarity percentage for reporting (inverted relationship)
                    avg_match = (1.0 - results.face_match_result.distance_score) * 100
                else:
                    avg_match = 0.0

            self.logger.info(f"Face match average: {avg_match:.1f}%")
            return avg_match

        except Exception as e:
            self.logger.warning(f"Face matching failed: {str(e)}")
            return None

    def _verify_otp(
        self,
        documents: List[DocumentData],
        client_public_key: str
    ) -> bool:
        """
        Verify OTP from selfie matches expected value from database.

        Args:
            documents: List of DocumentData
            client_public_key: Client's public key for OTP lookup

        Returns:
            True if OTP matches or no selfie
        """
        # Find selfie document
        selfie_data = None
        for doc in documents:
            if doc.document_type == 'selfie':
                data = doc.extracted_data
                if isinstance(data, SelfieOTPData):
                    selfie_data = data
                    break

        # No selfie found
        if not selfie_data:
            return True  # No OTP to verify

        # Fetch expected OTP from database using client public key
        otp_record = self.otp_repository.get_otp_by_public_key(client_public_key)
        if not otp_record:
            self.logger.warning(f"No OTP found in database for public key: {client_public_key[:16]}...")
            return False

        expected_otp = otp_record['random_number']

        # Verify match
        matches = selfie_data.otp_number == expected_otp

        if not matches:
            self.logger.warning(
                f"OTP mismatch: expected {expected_otp}, "
                f"got {selfie_data.otp_number}"
            )

        return matches

    async def _estimate_age_from_selfie(
        self,
        documents: List[DocumentData],
        document_contents: List[Tuple[str, bytes, bool, Optional[str], Optional[str]]]
    ) -> Optional[Dict[str, Any]]:
        """
        Estimate age from selfie using DeepFace and compare with passport DOB.

        NOTE: Disabled for performance - age estimation adds significant latency.

        Args:
            documents: List of DocumentData
            document_contents: List of (filename, bytes, is_pdf, explicit_file_type, explicit_document_type)

        Returns:
            Dict with age estimation findings and DOB consistency check, or None if no selfie
        """
        # Age estimation disabled for performance
        return None

        try:
            # Find selfie document
            selfie_content = None
            for doc, (filename, content, is_pdf, explicit_file_type, explicit_document_type) in zip(documents, document_contents):
                if doc.document_type == 'selfie' and not is_pdf:
                    selfie_content = content
                    break

            if not selfie_content:
                return None  # No selfie to analyze

            # Extract passport DOB for comparison
            passport_dob = self._extract_passport_dob(documents)
            if not passport_dob:
                self.logger.info("No passport DOB found for age comparison")
                return None

            # Use DeepFace for age estimation
            try:
                # Load image for DeepFace analysis
                img_array = self.deepface_helper.load_and_convert_image(selfie_content)

                # Analyze face for age using DeepFace
                analysis = await self.deepface_helper.analyze_face(
                    img_array=img_array,
                    actions=['age'],
                    enforce_detection=True,  # Fix: Ensure valid face detection
                    detector_backend="retinaface"  # Standardized to retinaface (preloaded at startup)
                )

                # Convert numpy types to native Python types
                analysis = convert_numpy_types(analysis)

                if not analysis or len(analysis) == 0:
                    self.logger.warning("No face detected for age estimation")
                    return None

                # Extract estimated age (DeepFace returns list of dicts)
                estimated_age = analysis[0].get('age', None)
                if estimated_age is None:
                    self.logger.warning("Could not estimate age from selfie")
                    return None

                # Calculate actual age from passport DOB
                from datetime import datetime
                from dateutil.parser import parse as parse_date

                try:
                    # Parse passport DOB (various formats supported)
                    dob_date = parse_date(passport_dob)
                    current_date = datetime.now()
                    actual_age = current_date.year - dob_date.year - (
                        (current_date.month, current_date.day) < (dob_date.month, dob_date.day)
                    )
                except Exception as e:
                    self.logger.warning(f"Could not parse passport DOB '{passport_dob}': {str(e)}")
                    return None

                # Compare estimated age with actual age
                age_difference = abs(estimated_age - actual_age)
                age_consistent = age_difference <= 5  # Allow 5-year tolerance

                findings = [
                    f"Face age estimation: {estimated_age} years",
                    f"Actual age from passport: {actual_age} years",
                    f"Age difference: {age_difference} years"
                ]

                if age_consistent:
                    findings.append("Age estimation matches passport DOB")
                else:
                    findings.append("Age estimation differs from passport DOB")

                return {
                    'findings': findings,
                    'estimated_age': estimated_age,
                    'actual_age': actual_age,
                    'age_difference': age_difference,
                    'dob_age_consistency': age_consistent
                }

            except Exception as e:
                self.logger.warning(f"Age estimation failed: {str(e)}")
                return None

        except Exception as e:
            self.logger.error(f"Age estimation analysis failed: {str(e)}")
            return None
