#!/usr/bin/env python3
"""
Unified Verification Test Script

Consolidates testing across four verification modes:
1. Full user verification - Process all documents with cross-checks
2. Passport-only - Test passports across all users (auto-skips face matching)
3. Bank statement-only - Test bank statements (skips passport cross-checks)
4. Auto mode - Test documents using filename-based hints

Usage:
    # Mode 1: Full user verification (with cross-checks)
    poetry run python scripts/test_verification.py --mode full --user user_004

    # Mode 2: Passport-only across all users (auto-skips face matching)
    poetry run python scripts/test_verification.py --mode passport

    # Mode 3: Bank statement-only across all users (skips passport cross-checks)
    poetry run python scripts/test_verification.py --mode bank

    # Mode 4: Auto mode with filename-based hints (e.g., trc_AE.jpg, tax_SG_IRAS.pdf)
    poetry run python scripts/test_verification.py --mode auto --user user_004

    # Common flags
    --skip-photoholmes     # Skip forgery detection
    --user user_001        # Specific user(s), can be repeated
    --folder scripts/test_data  # Base folder path
    --detailed             # Verbose output

    # Filename convention for auto mode: {doc_type}_{country}_{entity}.{ext}
    # Shorthand doc types: trc (tax_residency_certificate), tax (tax_return),
    #                      id (id_card), pan (pan_card), dl (driving_license), bank (bank_statement)
    # Examples: trc_AE.jpg, tax_SG_IRAS.pdf, pan_IN.jpg, id_SG.jpg
"""

import os
import sys
import asyncio
import argparse
import base64
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

# Set environment variable BEFORE importing modules that depend on it
os.environ['USE_LINE_BY_LINE'] = 'true'

from app.core.logger import get_logger


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class DocumentResult:
    """Result of testing a single document."""
    user: str
    filename: str
    document_type: str
    success: bool
    confidence: float = 0.0
    extracted_data: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    is_missing: bool = False
    elapsed_seconds: float = 0.0


@dataclass
class UserVerificationResult:
    """Result of verifying a single user in full mode."""
    user: str
    selfie_result: Optional[DocumentResult] = None
    passport_result: Optional[DocumentResult] = None
    bank_statement_results: List[DocumentResult] = field(default_factory=list)
    other_results: List[DocumentResult] = field(default_factory=list)
    cross_checks: Dict[str, Any] = field(default_factory=dict)


# ============================================================
# DOCUMENT TYPE DETECTION
# ============================================================

class DocumentTypeDetector:
    """Detect document types from filenames."""

    # Document type detection patterns
    DOC_TYPE_PATTERNS = {
        'selfie': ['selfie', 'selfy', 'photo', 'portrait'],
        'passport': ['passport', 'pasport', 'pas'],
        'id_card': ['id_card', 'idcard', 'national_id', 'nric', 'idcard'],
        'bank_statement': ['bank', 'statement'],
        'tax_statement': ['tax', 'irs', 'return'],
        'tax_residency_certificate': ['trc', 'tax_residency'],
        'pan_card': ['pan_card', 'pancard', 'pan'],
    }

    # Display names
    DOC_TYPE_NAMES = {
        'selfie': 'Selfie',
        'passport': 'Passport',
        'id_card': 'ID Card',
        'bank_statement': 'Bank Stmt',
        'tax_statement': 'Tax Return',
        'tax_residency_certificate': 'Tax Residency',
        'pan_card': 'PAN Card',
        'other': 'Other',
    }

    @classmethod
    def detect(cls, filename: str) -> str:
        """
        Auto-detect document type from filename patterns.

        Priority order: selfie > passport > id_card > bank_statement > tax_statement > tax_residency_certificate > pan_card
        """
        filename_lower = filename.lower()

        # Check each type in priority order
        for doc_type in ['selfie', 'passport', 'id_card', 'bank_statement', 'tax_statement', 'tax_residency_certificate', 'pan_card']:
            patterns = cls.DOC_TYPE_PATTERNS.get(doc_type, [])
            if any(pattern in filename_lower for pattern in patterns):
                return doc_type

        return 'other'

    @classmethod
    def get_display_name(cls, doc_type: str) -> str:
        """Get display name for document type."""
        return cls.DOC_TYPE_NAMES.get(doc_type, doc_type.title())


class MockUserIdentityRepository:
    """Mock user identity repository for testing name matching."""

    def __init__(self, full_name: str = None):
        self.full_name = full_name

    def get_user_by_id(self, user_identity_id: str) -> Optional[Dict[str, Any]]:
        """Return mock user data with the provided full_name."""
        if self.full_name:
            return {'full_name': self.full_name}
        return None


class FilenameHintParser:
    """Parse document hints from filename using convention: {doc_type}_{country}_{entity}.{ext}"""

    # Shorthand mapping
    SHORTHAND_MAP = {
        'trc': 'tax_residency_certificate',
        'tax': 'tax_return',
        'id': 'id_card',
        'pan': 'pan_card',
        'dl': 'driving_license',
        'bank': 'bank_statement',
    }

    @classmethod
    def parse(cls, filename: str) -> Dict[str, Any]:
        """Parse hints from filename. Returns dict with document_type, country, entity."""
        stem = Path(filename).stem
        parts = stem.split('_')

        hints = {'document_type': 'auto', 'country': 'auto', 'entity': 'auto'}

        if len(parts) >= 1 and parts[0]:
            hints['document_type'] = cls.SHORTHAND_MAP.get(parts[0].lower(), parts[0].lower())

        if len(parts) >= 2 and parts[1]:
            hints['country'] = parts[1].upper()

        if len(parts) >= 3 and parts[2]:
            hints['entity'] = parts[2].upper()

        return hints


# ============================================================
# FILE DISCOVERY
# ============================================================

class FileDiscovery:
    """Discover files in user folders."""

    FILE_PATTERNS = ['*.[Jj][Pp][Gg]', '*.[Jj][Pp][Ee][Gg]', '*.[Pp][Nn][Gg]', '*.[Pp][Dd][Ff]']

    @classmethod
    def scan_user_folder(cls, folder: Path) -> Dict[str, List[Path]]:
        """
        Scan a user folder and return files grouped by document type.

        Returns:
            Dict mapping document type to list of file paths
        """
        files_by_type = defaultdict(list)

        for pattern in cls.FILE_PATTERNS:
            for file_path in folder.glob(pattern):
                if file_path.name.startswith('.'):
                    continue
                doc_type = DocumentTypeDetector.detect(file_path.name)
                files_by_type[doc_type].append(file_path)

        return dict(files_by_type)

    @classmethod
    def get_user_folders(cls, base_folder: Path, filter_users: Optional[List[str]] = None) -> List[Path]:
        """
        Get all user folders from base folder.

        Args:
            base_folder: Base folder containing user folders
            filter_users: Optional list of specific users to include

        Returns:
            List of user folder paths
        """
        if not base_folder.exists():
            return []

        user_folders = sorted([f for f in base_folder.iterdir() if f.is_dir() and not f.name.startswith('.')])

        if filter_users:
            user_folders = [f for f in user_folders if f.name in filter_users]

        return user_folders


# ============================================================
# DOCUMENT TESTERS
# ============================================================

class PassportTester:
    """Test passport processing through the strict pipeline."""

    def __init__(self):
        self.logger = get_logger()

    async def test_passport(
        self,
        passport_path: Path,
        user_identity_id: Optional[str] = None,
        skip_face_matching: bool = True,
        verbose: bool = True
    ) -> DocumentResult:
        """
        Test a single passport through the strict pipeline.

        Args:
            passport_path: Path to passport image
            user_identity_id: Optional user identity ID (for face matching)
            skip_face_matching: Whether to skip face matching
            verbose: Print detailed output

        Returns:
            DocumentResult with extraction results
        """
        from app.services.sequential_passport_service import SequentialPassportService
        from app.repositories.user_key_repository import UserKeyRepository
        from app.repositories.user_identity_repository import UserIdentityRepository
        from app.repositories.face_biometrics_repository import FaceBiometricsRepository
        from app.core.db.database import get_db_connection_context
        from ecdsa import SigningKey, SECP256k1
        import uuid

        user_data = None
        original_skip_face = None

        try:
            # Configure skip_face_matching if requested
            if skip_face_matching:
                from app.config.verification_config import verification_settings
                original_skip_face = verification_settings.skip_face_matching
                verification_settings.skip_face_matching = True
                if verbose:
                    print("Note: Face matching SKIPPED (passport-only mode)")

            # Create test user for pipeline
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')
            test_id = passport_path.stem[:20]

            user_identity_id = str(uuid.uuid4())
            private_key = SigningKey.generate(curve=SECP256k1)
            public_key = private_key.get_verifying_key()
            pub_key_hex = public_key.to_string().hex()

            mobile_number = "+9999999999"
            country_code = "TST"

            with get_db_connection_context() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        INSERT INTO user_identity_index (id, verification_state)
                        VALUES (%s, 1)
                    """, (user_identity_id,))

                    cursor.execute("""
                        INSERT INTO user_keys
                        (id, mobile_number, country_code, user_public_key, user_identity_id)
                        VALUES (UUID(), %s, %s, %s, %s)
                    """, (mobile_number, country_code, pub_key_hex, user_identity_id))

                    conn.commit()

            user_data = {
                "user_identity_id": user_identity_id,
                "client_public_key": pub_key_hex
            }

            # Load passport image
            with open(passport_path, 'rb') as f:
                file_data = base64.b64encode(f.read()).decode('utf-8')

            if verbose:
                print(f"\n{'='*80}")
                print(f"Testing Passport: {passport_path.name}")
                print(f"{'='*80}")

            # Run the strict pipeline
            start_time = datetime.now()
            passport_service = SequentialPassportService()

            result = await passport_service.process_passport_strict(
                client_public_key=user_data["client_public_key"],
                file_data=file_data,
                filename=passport_path.name,
                iv="test_iv_placeholder",
                callback_url=None,
                document_type="passport"
            )

            elapsed = (datetime.now() - start_time).total_seconds()

            # Extract key data
            extracted_data = {}
            if result.extracted_data:
                extracted_data = {
                    'full_name': result.extracted_data.get('full_name'),
                    'passport_number': result.extracted_data.get('passport_number') or result.extracted_data.get('number'),
                    'country_code': result.extracted_data.get('country_code'),
                    'dob': result.extracted_data.get('dob'),
                    'sex': result.extracted_data.get('sex'),
                    'expiry': result.extracted_data.get('expiry'),
                }

            # Add other checks
            other_checks = {}
            if result.other_checks:
                other_checks = {
                    'face_match_confidence': result.other_checks.get('face_match_confidence'),
                    'similarity_score': result.other_checks.get('similarity_score'),
                    'osint_risk_score': result.other_checks.get('osint_risk_score'),
                    'document_expiry_valid': result.other_checks.get('document_expiry_valid'),
                }

            if result.forgery_checks:
                other_checks['forgery_checks'] = result.forgery_checks

            extracted_data['other_checks'] = other_checks

            # Extract similarity score for easy access
            similarity_score = other_checks.get('similarity_score')
            if similarity_score is not None:
                extracted_data['similarity_score'] = similarity_score

            if verbose:
                status = "✓ PASSED" if result.result else "✗ FAILED"
                print(f"\nResult: {status}")
                print(f"Processing Time: {elapsed:.2f}s")
                if result.error:
                    print(f"Error: {result.error}")

                # Print key fields
                for key, value in extracted_data.items():
                    if key == 'other_checks':
                        continue
                    if value is not None:
                        value_str = str(value)
                        if len(value_str) > 50:
                            value_str = value_str[:47] + "..."
                        print(f"  {key}: {value_str}")

                # Print reference similarity if available
                if similarity_score is not None:
                    threshold = result.other_checks.get('reference_threshold', 0.65) if result.other_checks else 0.65
                    sim_status = "✓" if similarity_score >= threshold else "✗"
                    print(f"  Reference Similarity: {sim_status} {similarity_score:.4f} (threshold: {threshold})")

            return DocumentResult(
                user="",  # Set by caller
                filename=passport_path.name,
                document_type='passport',
                success=result.result,
                confidence=85.0 if result.result else 0.0,  # Pipeline doesn't return confidence
                extracted_data=extracted_data,
                elapsed_seconds=elapsed,
                error_message=result.error if not result.result else None
            )

        except Exception as e:
            self.logger.error(f"Passport test failed for {passport_path}: {e}")
            import traceback
            traceback.print_exc()
            return DocumentResult(
                user="",
                filename=passport_path.name,
                document_type='passport',
                success=False,
                confidence=0.0,
                error_message=str(e)
            )

        finally:
            # Cleanup test user
            if user_data:
                try:
                    with get_db_connection_context() as conn:
                        with conn.cursor() as cursor:
                            cursor.execute("DELETE FROM face_biometrics WHERE user_identity_id = %s", (user_data["user_identity_id"],))
                            cursor.execute("DELETE FROM user_keys WHERE user_identity_id = %s", (user_data["user_identity_id"],))
                            cursor.execute("DELETE FROM user_identity_index WHERE id = %s", (user_data["user_identity_id"],))
                            conn.commit()
                except Exception as e:
                    self.logger.warning(f"Failed to cleanup test user: {e}")

            # Restore original skip_face_matching setting
            if original_skip_face is not None:
                from app.config.verification_config import verification_settings
                verification_settings.skip_face_matching = original_skip_face


class BankStatementTester:
    """Test bank statement processing."""

    def __init__(self):
        self.logger = get_logger()

    async def test_bank_statement(
        self,
        file_path: Path,
        passport_name: Optional[str] = None,
        verbose: bool = True,
        extractor: str = 'default'
    ) -> DocumentResult:
        """
        Test a bank statement using the unified validation pipeline.

        This uses the same validate_from_file() method that the API uses,
        ensuring consistent validation behavior between tests and production.

        Args:
            file_path: Path to bank statement file
            passport_name: Optional passport name for cross-check
            verbose: Print detailed output
            extractor: Extractor to use ('default' or 'spatial')

        Returns:
            DocumentResult with extraction results
        """
        # Use spatial extractor if requested
        if extractor == 'spatial':
            return await self._test_with_spatial_extractor(
                file_path, passport_name, verbose
            )

        from app.services.sequential_bank_statement_service import SequentialBankStatementService

        try:
            with open(file_path, 'rb') as f:
                file_bytes = f.read()

            if verbose:
                print(f"\n{'='*80}")
                print(f"Testing Bank Statement: {file_path.name}")
                print(f"{'='*80}")

            # Use the SINGLE ENTRY POINT for bank statement extraction
            # Both API and test script use this exact same method
            result = await SequentialBankStatementService.extract_from_file(
                file_bytes=file_bytes,
                filename=file_path.name,
            )

            elapsed = result['elapsed_seconds']
            is_valid = result['success']
            error_message = result['error_message']
            extracted_data = result['extracted_data']
            confidence_data = result['confidence_data']
            validation_results = result['validation_results']

            # Calculate average confidence
            if confidence_data:
                field_confidences = [v.get('overall_confidence', 0) for v in confidence_data.values() if isinstance(v, dict)]
                avg_confidence = (sum(field_confidences) / len(field_confidences) * 100) if field_confidences else 0.0
            else:
                avg_confidence = 0.0

            # Add name cross-check if passport name provided (separate from main validation)
            if passport_name and extracted_data.get('account_holder_name'):
                from app.utils.string_matching import fuzzy_match_names
                similarity_score = round(fuzzy_match_names(passport_name, extracted_data['account_holder_name']) * 100, 1)
                extracted_data['name_cross_check'] = {
                    'passport_name': passport_name,
                    'bank_name': extracted_data['account_holder_name'],
                    'similarity': similarity_score,
                    'passed': similarity_score >= 70.0
                }

            if verbose:
                status = "✓ PASSED" if is_valid else "✗ FAILED"
                print(f"\nResult: {status}")
                print(f"Processing Time: {elapsed:.2f}s")
                print(f"Confidence: {avg_confidence:.1f}%")

                for key, value in extracted_data.items():
                    if key in ['validation', 'validation_results']:
                        continue
                    if key == 'name_cross_check':
                        ncc = value
                        chk = "✓" if ncc['passed'] else "✗"
                        print(f"  Name Cross-Check: {chk} ({ncc['similarity']:.1f}%)")
                        continue
                    if value is not None:
                        value_str = str(value)
                        if len(value_str) > 50:
                            value_str = value_str[:47] + "..."
                        print(f"  {key}: {value_str}")

                # Print validation details
                if validation_results:
                    print("\n  Validation Results:")
                    for check_name, check_data in validation_results.items():
                        if isinstance(check_data, dict):
                            if 'valid' in check_data:
                                chk_status = "✓" if check_data['valid'] else "✗"
                                print(f"    {check_name}: {chk_status}")
                                if not check_data['valid']:
                                    if 'error' in check_data:
                                        print(f"      Error: {check_data['error']}")
                                    if 'missing' in check_data:
                                        print(f"      Missing: {', '.join(check_data['missing'])}")
                                    if 'missing_fields' in check_data:
                                        print(f"      Missing: {', '.join(check_data['missing_fields'])}")
                            elif 'components' in check_data:
                                # Address components
                                chk_status = "✓" if check_data.get('valid') else "✗"
                                print(f"    {check_name}: {chk_status}")
                                if check_data.get('components'):
                                    print(f"      Components: {check_data['components']}")
                                if check_data.get('missing'):
                                    print(f"      Missing: {', '.join(check_data['missing'])}")

                if not is_valid and error_message:
                    print(f"\n  Validation Error: {error_message}")

            return DocumentResult(
                user="",
                filename=file_path.name,
                document_type='bank_statement',
                success=is_valid,
                confidence=avg_confidence,
                extracted_data=extracted_data,
                elapsed_seconds=elapsed,
                error_message=error_message
            )

        except Exception as e:
            self.logger.error(f"Bank statement test failed: {e}")
            if verbose:
                print(f"\n✗ FAILED: {str(e)}")
            return DocumentResult(
                user="",
                filename=file_path.name,
                document_type='bank_statement',
                success=False,
                confidence=0.0,
                error_message=str(e)
            )

    async def _test_with_spatial_extractor(
        self,
        file_path: Path,
        passport_name: Optional[str] = None,
        verbose: bool = True
    ) -> DocumentResult:
        """
        Test a bank statement using the spatial extractor.

        This uses the new geometry-based extraction algorithm.

        Args:
            file_path: Path to bank statement file
            passport_name: Optional passport name for cross-check
            verbose: Print detailed output

        Returns:
            DocumentResult with extraction results
        """
        from app.helper.extractors.spatial_bank_statement_extractor import (
            SpatialBankStatementExtractor, ExtractionResult
        )
        import time

        try:
            with open(file_path, 'rb') as f:
                file_bytes = f.read()

            if verbose:
                print(f"\n{'='*80}")
                print(f"Testing Bank Statement (SPATIAL): {file_path.name}")
                print(f"{'='*80}")

            # Use the spatial extractor
            start_time = time.time()
            extractor = SpatialBankStatementExtractor()
            result = extractor.extract(str(file_path), max_pages=1)
            elapsed = time.time() - start_time

            # Convert ExtractionResult to dict format
            extracted_data = {
                'account_holder_name': result.account_holder_name,
                'account_holder_address': result.account_holder_address,  # Now street address only
                # Address components extracted from full address
                'address_city': result.address_city,
                'address_state': result.address_state,
                'address_postal': result.address_postal,
                'address_country': result.address_country,
                'account_number': result.account_number,
                'bank_name': result.bank_name,
                'bank_country': result.bank_country,
                'bank_code': result.bank_code,  # SWIFT/IFSC code for display in test output
                'currency': result.currency,
                'statement_date': result.statement_date,
                'ifsc_code': result.ifsc_code,
                'swift_code': result.swift_code,
            }

            # Add extraction method indicator
            extracted_data['_extraction_method'] = 'spatial'

            # Add raw values for debugging
            extracted_data['_raw_values'] = result.raw_values

            # Determine success (basic validation)
            is_valid = bool(result.account_holder_name or result.account_number)

            if verbose:
                status = "✓ PASSED" if is_valid else "✗ FAILED"
                print(f"\nResult: {status}")
                print(f"Processing Time: {elapsed:.2f}s")

                for key, value in extracted_data.items():
                    if key.startswith('_'):
                        continue
                    if value is not None:
                        value_str = str(value)
                        if len(value_str) > 50:
                            value_str = value_str[:47] + "..."
                        print(f"  {key}: {value_str}")

            return DocumentResult(
                user="",
                filename=file_path.name,
                document_type='bank_statement',
                success=is_valid,
                confidence=85.0 if is_valid else 0.0,
                extracted_data=extracted_data,
                elapsed_seconds=elapsed,
                error_message=None if is_valid else "Could not extract required fields"
            )

        except Exception as e:
            self.logger.error(f"Spatial bank statement test failed: {e}")
            if verbose:
                print(f"\n✗ FAILED: {str(e)}")
            import traceback
            traceback.print_exc()
            return DocumentResult(
                user="",
                filename=file_path.name,
                document_type='bank_statement',
                success=False,
                confidence=0.0,
                error_message=str(e)
            )


# ============================================================
# AUTO DOCUMENT TESTER
# ============================================================

class AutoDocumentTester:
    """Test auto-document processing with hints from filename."""

    def __init__(self):
        self.logger = get_logger()

    async def test_auto_document(
        self,
        file_path: Path,
        hints: Dict[str, Any],
        verbose: bool = True
    ) -> DocumentResult:
        """Test a single document through auto-document processing."""
        from app.services.generic_document_service import GenericDocumentService

        start_time = time.time()

        try:
            # Read and encode file
            with open(file_path, 'rb') as f:
                file_bytes = f.read()
                file_data_b64 = base64.b64encode(file_bytes).decode('utf-8')

            # Prepare file_data in the expected format
            file_data = {
                "file_data": file_data_b64,
                "file_type": file_path.suffix[1:].lower(),  # e.g., "jpg", "png", "pdf"
            }

            service = GenericDocumentService(user_identity_repo=None)

            result = await service.process_auto_document(
                file_data=file_data,
                client_public_key="",
                user_identity_id="test",
                hints=hints if any(v != 'auto' for v in hints.values()) else None
            )

            elapsed = time.time() - start_time

            if verbose:
                print(f"\n{'='*60}")
                print(f"File: {file_path.name}")
                print(f"Hints: {hints}")
                print(f"Result: {'PASS' if result.result else 'FAIL'}")
                if result.error:
                    print(f"Error: {result.error}")
                print(f"Time: {elapsed:.2f}s")

                extracted = result.extracted_data or {}
                if extracted:
                    print("Extracted:")
                    for key, value in extracted.items():
                        if value is not None:
                            print(f"  {key}: {value}")

            return DocumentResult(
                user="", filename=file_path.name,
                document_type=hints.get('document_type', 'auto'),
                success=result.result,
                confidence=100.0 if result.result else 0.0,
                extracted_data=result.extracted_data or {},
                elapsed_seconds=elapsed,
                error_message=result.error
            )

        except Exception as e:
            elapsed = time.time() - start_time
            if verbose:
                print(f"\n{'='*60}")
                print(f"FAILED: {file_path.name}")
                print(f"Error: {str(e)}")
            return DocumentResult(
                user="", filename=file_path.name,
                document_type=hints.get('document_type', 'auto'),
                success=False,
                error_message=str(e),
                elapsed_seconds=elapsed
            )


class PanCardTester:
    """Test PAN card processing using the ID card service."""

    def __init__(self):
        self.logger = get_logger()

    async def test_pan_card(
        self,
        file_path: Path,
        passport_name: Optional[str] = None,
        passport_dob: Optional[str] = None,
        verbose: bool = True
    ) -> DocumentResult:
        """
        Test a PAN card using the SequentialIDCardService.

        Args:
            file_path: Path to PAN card image
            passport_name: Name from passport for cross-checking
            passport_dob: Date of birth from passport for cross-checking
            verbose: Print detailed output

        Returns:
            DocumentResult with extraction results
        """
        from app.services.sequential_id_card_service import SequentialIDCardService
        import time

        try:
            with open(file_path, 'rb') as f:
                file_bytes = f.read()

            if verbose:
                print(f"\n{'='*80}")
                print(f"Testing PAN Card: {file_path.name}")
                print(f"{'='*80}")

            start_time = time.time()

            # Use SequentialIDCardService.validate_from_file() to test the PAN card
            service = SequentialIDCardService()
            validation_result = await service.validate_from_file(
                file_bytes=file_bytes,
                filename=file_path.name,
                skip_photoholmes=True
            )

            elapsed = time.time() - start_time

            is_valid = validation_result.get('is_valid', False)
            extracted_data = validation_result.get('extracted_data', {})
            validation_checks = validation_result.get('validation_checks', {})
            error_message = validation_result.get('error_message')
            avg_confidence = validation_result.get('average_confidence', 0.0)

            # Cross-check with passport data if provided
            cross_check_passed = True
            cross_check_details = {}

            if passport_name and extracted_data.get('full_name'):
                from app.utils.string_matching import fuzzy_match_names
                name_similarity = round(fuzzy_match_names(passport_name, extracted_data['full_name']) * 100, 1)
                name_match = name_similarity >= 70.0

                cross_check_details['name_cross_check'] = {
                    'passport_name': passport_name,
                    'pan_name': extracted_data['full_name'],
                    'similarity': name_similarity,
                    'passed': name_match
                }
                cross_check_passed = cross_check_passed and name_match

                if verbose:
                    chk = "✓" if name_match else "✗"
                    print(f"  Name Cross-Check: {chk} ({name_similarity:.1f}%)")

            if passport_dob and extracted_data.get('date_of_birth'):
                # Compare dates - both should be in YYYY-MM-DD format
                dob_match = passport_dob == extracted_data['date_of_birth']

                cross_check_details['dob_cross_check'] = {
                    'passport_dob': passport_dob,
                    'pan_dob': extracted_data['date_of_birth'],
                    'passed': dob_match
                }
                cross_check_passed = cross_check_passed and dob_match

                if verbose:
                    chk = "✓" if dob_match else "✗"
                    print(f"  DOB Cross-Check: {chk} (Passport: {passport_dob}, PAN: {extracted_data['date_of_birth']})")

            # Add cross-check details to extracted_data
            extracted_data.update(cross_check_details)

            # If cross-checks fail, mark overall result as failed
            if not cross_check_passed and is_valid:
                is_valid = False
                if not error_message:
                    error_message = "Cross-check with passport failed"

            if verbose:
                print(f"\nResult: {'PASS' if is_valid else 'FAIL'}")
                print(f"Time: {elapsed:.2f}s")
                print(f"Confidence: {avg_confidence:.1f}%")
                if extracted_data:
                    print("\nExtracted:")
                    for key, value in extracted_data.items():
                        if key in ['validation', 'validation_results', 'name_cross_check', 'dob_cross_check']:
                            continue
                        if value is not None:
                            print(f"  {key}: {value}")
                if validation_checks:
                    print("\nValidation Checks:")
                    for check, value in validation_checks.items():
                        print(f"  {check}: {value}")
                if error_message:
                    print(f"\nError: {error_message}")

            return DocumentResult(
                user="",
                filename=file_path.name,
                document_type='pan_card',
                success=is_valid,
                confidence=avg_confidence * 100,
                extracted_data=extracted_data,
                elapsed_seconds=elapsed,
                error_message=error_message
            )

        except Exception as e:
            self.logger.error(f"PAN card test failed: {e}")
            if verbose:
                print(f"\n✗ FAILED: {str(e)}")
            return DocumentResult(
                user="",
                filename=file_path.name,
                document_type='pan_card',
                success=False,
                confidence=0.0,
                error_message=str(e)
            )


# ============================================================
# VERIFICATION ORCHESTRATOR
# ============================================================

class VerificationOrchestrator:
    """Orchestrates verification testing across modes."""

    def __init__(self, extractor: str = 'default'):
        self.logger = get_logger()
        self.passport_tester = PassportTester()
        self.bank_tester = BankStatementTester()
        self.pan_card_tester = PanCardTester()
        self.extractor = extractor  # 'default' or 'spatial'

    async def run_full_mode(
        self,
        user_folders: List[Path],
        skip_photoholmes: bool = False,
        skip_face_matching: bool = False,
        verbose: bool = True
    ) -> Dict[str, UserVerificationResult]:
        """
        Run full user verification with cross-checks.

        For each user:
        1. Process selfie (if present)
        2. Process passport (if present) - with face matching if selfie available
        3. Process bank statements (if present) - with name cross-check against passport
        4. Process other documents
        5. Generate cross-check summary
        """
        results = {}

        for user_folder in user_folders:
            user_name = user_folder.name
            files_by_type = FileDiscovery.scan_user_folder(user_folder)

            if verbose:
                print(f"\n{'#'*80}")
                print(f"# USER: {user_name}")
                print(f"{'#'*80}")

            user_result = UserVerificationResult(user=user_name)

            # 1. Process selfie first (for face matching later)
            selfie_files = files_by_type.get('selfie', [])
            if selfie_files:
                selfie_path = selfie_files[0]  # Use first selfie
                result = await self._process_selfie(selfie_path, verbose)
                user_result.selfie_result = result

            # 2. Process passport (with face matching if selfie was processed)
            passport_files = files_by_type.get('passport', [])
            if passport_files:
                passport_path = passport_files[0]  # Use first passport
                has_selfie = user_result.selfie_result is not None and user_result.selfie_result.success

                result = await self.passport_tester.test_passport(
                    passport_path,
                    skip_face_matching=skip_face_matching or not has_selfie,
                    verbose=verbose
                )
                result.user = user_name
                user_result.passport_result = result

                # Add face matching cross-check
                if has_selfie and not skip_face_matching:
                    face_conf = result.extracted_data.get('other_checks', {}).get('face_match_confidence')
                    user_result.cross_checks['face_match'] = {
                        'selfie': user_result.selfie_result.filename,
                        'passport': result.filename,
                        'confidence': face_conf,
                        'passed': face_conf is not None and face_conf >= 70.0
                    }

            # 3. Process bank statements (with name cross-check against passport)
            passport_name = None
            # Use passport name even if passport validation failed - we still need the name for matching
            if user_result.passport_result and user_result.passport_result.extracted_data:
                passport_name = user_result.passport_result.extracted_data.get('full_name')

            bank_files = files_by_type.get('bank_statement', [])
            for bank_path in bank_files:
                result = await self.bank_tester.test_bank_statement(
                    bank_path,
                    passport_name=passport_name,
                    verbose=verbose
                )
                result.user = user_name
                user_result.bank_statement_results.append(result)

                # Add name cross-check to user result
                if 'name_cross_check' in result.extracted_data:
                    ncc = result.extracted_data['name_cross_check']
                    user_result.cross_checks[f'name_match_{bank_path.name}'] = {
                        'passport_name': ncc['passport_name'],
                        'bank_name': ncc['bank_name'],
                        'similarity': ncc['similarity'],
                        'passed': ncc['passed']
                    }

            # 4. Process other document types (with passport name for matching)
            id_files = files_by_type.get('id_card', [])
            for id_path in id_files:
                result = await self._process_id_card(id_path, verbose, user_name, passport_name)
                result.user = user_name
                user_result.other_results.append(result)

                # Add name cross-check to user result
                # GenericDocumentService stores this as '_name_match'
                if '_name_match' in result.extracted_data:
                    nm = result.extracted_data['_name_match']
                    user_result.cross_checks[f'id_name_match_{id_path.name}'] = {
                        'passport_name': nm.get('passport_name'),
                        'id_name': nm.get('extracted_name'),
                        'similarity': nm.get('score', 0),
                        'passed': nm.get('is_valid', False)
                    }

            tax_files = files_by_type.get('tax_statement', [])
            for tax_path in tax_files:
                result = await self._process_tax_statement(tax_path, verbose)
                result.user = user_name
                user_result.other_results.append(result)

            # Process Tax Residency Certificates (with cross-check against passport)
            trc_files = files_by_type.get('tax_residency_certificate', [])
            for trc_path in trc_files:
                result = await self._process_tax_residency_certificate(trc_path, verbose, user_name, passport_name)
                result.user = user_name
                user_result.other_results.append(result)

                # Add name cross-check to user result
                # GenericDocumentService stores this as '_name_match'
                if '_name_match' in result.extracted_data:
                    nm = result.extracted_data['_name_match']
                    user_result.cross_checks[f'trc_name_match_{trc_path.name}'] = {
                        'passport_name': nm.get('passport_name'),
                        'trc_name': nm.get('extracted_name'),
                        'similarity': nm.get('score', 0),
                        'passed': nm.get('is_valid', False)
                    }
                elif 'name_cross_check' in result.extracted_data:
                    # Legacy format for compatibility
                    ncc = result.extracted_data['name_cross_check']
                    user_result.cross_checks[f'trc_name_match_{trc_path.name}'] = {
                        'passport_name': ncc.get('passport_name'),
                        'trc_name': ncc.get('document_name'),
                        'similarity': ncc.get('similarity'),
                        'passed': ncc.get('passed', False)
                    }

            # Process PAN cards (with cross-check against passport)
            passport_name = None
            passport_dob = None
            if user_result.passport_result and user_result.passport_result.success:
                passport_name = user_result.passport_result.extracted_data.get('full_name')
                # Passport may have 'dob' or 'date_of_birth'
                passport_dob = user_result.passport_result.extracted_data.get('dob') or \
                               user_result.passport_result.extracted_data.get('date_of_birth')

            pan_files = files_by_type.get('pan_card', [])
            for pan_path in pan_files:
                result = await self.pan_card_tester.test_pan_card(
                    pan_path,
                    passport_name=passport_name,
                    passport_dob=passport_dob,
                    verbose=verbose
                )
                result.user = user_name
                user_result.other_results.append(result)

                # Add cross-checks to user result
                if 'name_cross_check' in result.extracted_data:
                    ncc = result.extracted_data['name_cross_check']
                    user_result.cross_checks[f'pan_name_match_{pan_path.name}'] = {
                        'passport_name': ncc['passport_name'],
                        'pan_name': ncc['pan_name'],
                        'similarity': ncc['similarity'],
                        'passed': ncc['passed']
                    }
                if 'dob_cross_check' in result.extracted_data:
                    dcc = result.extracted_data['dob_cross_check']
                    user_result.cross_checks[f'pan_dob_match_{pan_path.name}'] = {
                        'passport_dob': dcc['passport_dob'],
                        'pan_dob': dcc['pan_dob'],
                        'passed': dcc['passed']
                    }

            results[user_name] = user_result

        return results

    async def run_passport_mode(
        self,
        base_folder: Path,
        filter_users: Optional[List[str]] = None,
        skip_photoholmes: bool = False,
        verbose: bool = True
    ) -> List[DocumentResult]:
        """
        Run passport-only mode across all users.

        Auto-skips face matching since no selfie is processed.
        """
        results = []
        user_folders = FileDiscovery.get_user_folders(base_folder, filter_users)

        print(f"\n{'='*80}")
        print(f"PASSPORT-ONLY MODE - Scanning {len(user_folders)} user(s)")
        print(f"{'='*80}")
        print("Note: Face matching auto-skipped in passport-only mode")

        for user_folder in user_folders:
            files_by_type = FileDiscovery.scan_user_folder(user_folder)
            passport_files = files_by_type.get('passport', [])

            for passport_path in passport_files:
                if verbose:
                    print(f"\nUser: {user_folder.name}")

                result = await self.passport_tester.test_passport(
                    passport_path,
                    skip_face_matching=True,  # Always skip in passport-only mode
                    verbose=verbose
                )
                result.user = user_folder.name
                results.append(result)

        return results

    async def run_bank_mode(
        self,
        base_folder: Path,
        filter_users: Optional[List[str]] = None,
        verbose: bool = True
    ) -> List[DocumentResult]:
        """
        Run bank statement-only mode across all users.

        Skips passport name cross-checks.
        """
        results = []
        user_folders = FileDiscovery.get_user_folders(base_folder, filter_users)

        print(f"\n{'='*80}")
        print(f"BANK STATEMENT-ONLY MODE - Scanning {len(user_folders)} user(s)")
        print(f"{'='*80}")
        print("Note: Passport name cross-checks skipped in bank-only mode")

        for user_folder in user_folders:
            files_by_type = FileDiscovery.scan_user_folder(user_folder)
            bank_files = files_by_type.get('bank_statement', [])

            for bank_path in bank_files:
                if verbose:
                    print(f"\nUser: {user_folder.name}")

                result = await self.bank_tester.test_bank_statement(
                    bank_path,
                    passport_name=None,  # Skip name cross-check in bank-only mode
                    verbose=verbose,
                    extractor=self.extractor
                )
                result.user = user_folder.name
                results.append(result)

        return results

    async def run_pan_card_mode(
        self,
        base_folder: Path,
        filter_users: Optional[List[str]] = None,
        skip_photoholmes: bool = False,
        verbose: bool = True
    ) -> List[DocumentResult]:
        """
        Run PAN card-only mode across all users.

        Skips passport cross-checks.
        """
        results = []
        user_folders = FileDiscovery.get_user_folders(base_folder, filter_users)

        print(f"\n{'='*80}")
        print(f"PAN CARD-ONLY MODE - Scanning {len(user_folders)} user(s)")
        print(f"{'='*80}")

        for user_folder in user_folders:
            files_by_type = FileDiscovery.scan_user_folder(user_folder)
            pan_files = files_by_type.get('pan_card', [])

            for pan_path in pan_files:
                if verbose:
                    print(f"\nUser: {user_folder.name}")

                result = await self.pan_card_tester.test_pan_card(
                    pan_path,
                    passport_name=None,  # Skip cross-checks in pan_card-only mode
                    passport_dob=None,
                    verbose=verbose
                )
                result.user = user_folder.name
                results.append(result)

        return results

    async def run_auto_mode(
        self,
        user_folders: List[Path],
        auto_tester: 'AutoDocumentTester',
        verbose: bool = True
    ) -> Dict[str, List[DocumentResult]]:
        """Run auto-document testing using filename-based hints."""
        results = defaultdict(list)

        for user_folder in user_folders:
            user_name = user_folder.name

            if verbose:
                print(f"\n{'#'*80}")
                print(f"# USER: {user_name}")
                print(f"{'#'*80}")

            for pattern in FileDiscovery.FILE_PATTERNS:
                for file_path in user_folder.glob(pattern):
                    if file_path.name.startswith('.'):
                        continue

                    hints = FilenameHintParser.parse(file_path.name)

                    # Only process if filename has BOTH meaningful document_type AND country
                    # This prevents processing files like "selfie.jpg" which would parse as
                    # document_type='selfie', country='auto'
                    if hints['document_type'] != 'auto' and hints['country'] != 'auto':
                        result = await auto_tester.test_auto_document(
                            file_path=file_path, hints=hints, verbose=verbose
                        )
                        result.user = user_name
                        results[user_name].append(result)

        return dict(results)

    async def _process_selfie(self, selfie_path: Path, verbose: bool) -> DocumentResult:
        """Process selfie and store embedding for face matching."""
        from app.services.face_extraction_service import FaceExtractionService

        try:
            if verbose:
                print(f"\n{'='*80}")
                print(f"Processing Selfie: {selfie_path.name}")
                print(f"{'='*80}")

            start_time = datetime.now()

            with open(selfie_path, 'rb') as f:
                selfie_bytes = f.read()

            face_service = FaceExtractionService()
            result = await face_service.extract_face_embedding(
                image_bytes=selfie_bytes,
                public_key=None,
                user_identity_id=None,  # Test mode
                document_type="selfie"
            )

            elapsed = (datetime.now() - start_time).total_seconds()

            success = result is not None

            if verbose:
                status = "✓ PASSED" if success else "✗ FAILED"
                print(f"\nResult: {status}")
                print(f"Processing Time: {elapsed:.2f}s")
                if success:
                    print(f"  Face embedding extracted successfully")

            return DocumentResult(
                user="",
                filename=selfie_path.name,
                document_type='selfie',
                success=success,
                confidence=100.0 if success else 0.0,
                elapsed_seconds=elapsed,
                error_message=None if success else "Failed to extract face embedding"
            )

        except Exception as e:
            self.logger.error(f"Selfie processing failed: {e}")
            return DocumentResult(
                user="",
                filename=selfie_path.name,
                document_type='selfie',
                success=False,
                confidence=0.0,
                error_message=str(e)
            )

    async def _process_id_card(self, id_path: Path, verbose: bool, user_id: str = "test_user", passport_name: str = None) -> DocumentResult:
        """Process ID card using the new auto-detection system (GenericDocumentService)."""
        from app.services.generic_document_service import GenericDocumentService

        try:
            if verbose:
                print(f"\n{'='*80}")
                print(f"Processing ID Card: {id_path.name}")
                print(f"{'='*80}")

            start_time = datetime.now()

            with open(id_path, 'rb') as f:
                file_bytes = f.read()

            # Use the new auto-detection system
            # Create mock repository with passport name for name matching
            mock_repo = MockUserIdentityRepository(passport_name) if passport_name else None
            service = GenericDocumentService(user_identity_repo=mock_repo)
            file_data = {
                "file_data": base64.b64encode(file_bytes).decode(),
                "file_type": id_path.suffix[1:].lower(),  # e.g., "jpg", "png"
            }

            # Parse hints from filename (e.g., id_AE.jpg, id_SG.jpg → country='AE'/'SG')
            # FilenameHintParser.parse() returns {'document_type': ..., 'country': ..., 'entity': ...}
            parsed_hints = FilenameHintParser.parse(id_path.name)
            # Filter out 'auto' placeholder values
            hints_to_pass = {k: v for k, v in parsed_hints.items() if v != 'auto'}
            # Fallback to document_type hint if nothing meaningful parsed
            if not hints_to_pass:
                hints_to_pass = {"document_type": "id_card"}

            if verbose:
                print(f"Filename hints: {parsed_hints} -> Passing: {hints_to_pass}")

            # Process with auto-detection
            result = await service.process_auto_document(
                file_data=file_data,
                client_public_key="",  # Not needed for extraction
                user_identity_id=user_id,  # Use actual user_id for passport name matching
                hints=hints_to_pass,
            )

            elapsed = (datetime.now() - start_time).total_seconds()

            # Extract data from the new result format
            extracted_data = result.extracted_data or {}

            # Get detection info from the _detection metadata we added
            detection_info = extracted_data.get("_detection", {})
            detected_type = detection_info.get("document_type", "id_card")

            elapsed = (datetime.now() - start_time).total_seconds()

            # Extract data from the new result format
            extracted_data = result.extracted_data or {}

            # Get detection info from the _detection metadata we added
            detection_info = extracted_data.get("_detection", {})
            detected_type = detection_info.get("document_type", "id_card")
            detected_country = detection_info.get("country", "")
            detected_entity = detection_info.get("entity", "")
            schema_id = detection_info.get("schema_id", "")
            detection_confidence = detection_info.get("detection_confidence", 0.0)

            # Build standardized extracted data
            standardized_data = {
                'document_type': detected_type,
                'detected_country': detected_country,
                'detected_entity': detected_entity,
                'schema_id': schema_id,
                'number': extracted_data.get('pan_number') or extracted_data.get('id_number') or extracted_data.get('nric_number') or extracted_data.get('identification_number'),
                'full_name': extracted_data.get('full_name'),
                'dob': extracted_data.get('date_of_birth') or extracted_data.get('dob'),
                'sex': extracted_data.get('sex') or extracted_data.get('gender'),
                'father_name': extracted_data.get('father_name'),
            }

            # Preserve _name_match for cross-check display (even though it's internal metadata)
            if '_name_match' in extracted_data:
                standardized_data['_name_match'] = extracted_data['_name_match']

            # Remove None values for cleaner output
            standardized_data = {k: v for k, v in standardized_data.items() if v is not None}

            if verbose:
                # Determine success based on having extracted meaningful data
                has_number = bool(standardized_data.get('number'))
                has_name = bool(standardized_data.get('full_name'))
                status = "✓ PASSED" if (has_number or has_name) else "✗ FAILED"
                print(f"\nResult: {status}")
                print(f"Processing Time: {elapsed:.2f}s")
                print(f"Detection Confidence: {detection_confidence:.1%}")
                print(f"Detected: type={detected_type}, country={detected_country}, entity={detected_entity}")
                if schema_id:
                    print(f"Schema: {schema_id}")
                print("Extracted Fields:")
                for key, value in extracted_data.items():
                    if key != '_detection' and value is not None:
                        print(f"  {key}: {value}")

            # Success if we got either a number or name
            success = bool(standardized_data.get('number') or standardized_data.get('full_name'))

            return DocumentResult(
                user="",
                filename=id_path.name,
                document_type=detected_type,
                success=success,
                confidence=detection_confidence * 100,
                extracted_data=standardized_data,
                elapsed_seconds=elapsed
            )

        except Exception as e:
            self.logger.error(f"ID card processing failed: {e}")
            return DocumentResult(
                user="",
                filename=id_path.name,
                document_type='id_card',
                success=False,
                confidence=0.0,
                error_message=str(e)
            )

    async def _process_tax_statement(self, tax_path: Path, verbose: bool) -> DocumentResult:
        """Process tax statement."""
        from app.helper.extractors.tax_statement_extractor import TaxStatementExtractor

        try:
            if verbose:
                print(f"\n{'='*80}")
                print(f"Processing Tax Statement: {tax_path.name}")
                print(f"{'='*80}")

            start_time = datetime.now()

            with open(tax_path, 'rb') as f:
                file_bytes = f.read()

            is_pdf = tax_path.suffix.lower() == '.pdf'
            extractor = TaxStatementExtractor()
            result = await extractor.extract(file_bytes, is_pdf)

            elapsed = (datetime.now() - start_time).total_seconds()

            # Calculate confidence
            if result.confidence_scores:
                avg_confidence = sum(result.confidence_scores.values()) / len(result.confidence_scores)
            else:
                avg_confidence = 0.0

            extracted_data = {
                'taxpayer_name': result.taxpayer_name,
                'tax_id': result.tax_id,
                'tax_year': result.tax_year,
                'gross_income': result.gross_income,
                'taxable_income': result.taxable_income,
                'tax_paid': result.tax_paid,
            }

            if verbose:
                status = "✓ PASSED" if result.taxpayer_name else "✗ FAILED"
                print(f"\nResult: {status}")
                print(f"Processing Time: {elapsed:.2f}s")
                for key, value in extracted_data.items():
                    if value is not None:
                        print(f"  {key}: {value}")

            return DocumentResult(
                user="",
                filename=tax_path.name,
                document_type='tax_statement',
                success=bool(result.taxpayer_name),
                confidence=avg_confidence,
                extracted_data=extracted_data,
                elapsed_seconds=elapsed
            )

        except Exception as e:
            self.logger.error(f"Tax statement processing failed: {e}")
            return DocumentResult(
                user="",
                filename=tax_path.name,
                document_type='tax_statement',
                success=False,
                confidence=0.0,
                error_message=str(e)
            )

    async def _process_tax_residency_certificate(self, trc_path: Path, verbose: bool, user_id: str = "test_user", passport_name: str = None) -> DocumentResult:
        """Process tax residency certificate using the auto-detection system."""
        from app.services.generic_document_service import GenericDocumentService

        try:
            if verbose:
                print(f"\n{'='*80}")
                print(f"Processing Tax Residency Certificate: {trc_path.name}")
                print(f"{'='*80}")

            start_time = datetime.now()

            with open(trc_path, 'rb') as f:
                file_bytes = f.read()

            # Use the auto-detection system
            # Create mock repository with passport name for name matching
            mock_repo = MockUserIdentityRepository(passport_name) if passport_name else None
            service = GenericDocumentService(user_identity_repo=mock_repo)
            file_data = {
                "file_data": base64.b64encode(file_bytes).decode(),
                "file_type": trc_path.suffix[1:].lower(),  # e.g., "jpg", "png"
            }

            # Parse hints from filename (e.g., trc_AE.jpg → country='AE')
            # FilenameHintParser.parse() returns {'document_type': ..., 'country': ..., 'entity': ...}
            parsed_hints = FilenameHintParser.parse(trc_path.name)
            # Filter out 'auto' placeholder values
            hints_to_pass = {k: v for k, v in parsed_hints.items() if v != 'auto'}
            # Fallback to document_type hint if no country/entity parsed
            if not hints_to_pass:
                hints_to_pass = {"document_type": "tax_residency_certificate"}

            if verbose:
                print(f"Filename hints: {parsed_hints} -> Passing: {hints_to_pass}")

            # Process with auto-detection
            result = await service.process_auto_document(
                file_data=file_data,
                client_public_key="",  # Not needed for extraction
                user_identity_id=user_id,  # Use actual user_id for passport name matching
                hints=hints_to_pass,
            )

            elapsed = (datetime.now() - start_time).total_seconds()

            # Extract data from the new result format
            extracted_data = result.extracted_data or {}

            # Get detection info from the _detection metadata
            detection_info = extracted_data.get("_detection", {})
            detected_type = detection_info.get("document_type", "tax_residency_certificate")
            detected_country = detection_info.get("country", "")
            detected_entity = detection_info.get("entity", "")
            schema_id = detection_info.get("schema_id", "")
            detection_confidence = detection_info.get("detection_confidence", 0.0)

            # Build standardized extracted data for TRC
            standardized_data = {
                'document_type': detected_type,
                'detected_country': detected_country,
                'detected_entity': detected_entity,
                'schema_id': schema_id,
                'full_name': extracted_data.get('full_name'),
                'certificate_number': extracted_data.get('certificate_number'),
                'passport_number': extracted_data.get('passport_number'),
                'valid_from': extracted_data.get('valid_from'),
                'valid_until': extracted_data.get('valid_until'),
                'nationality': extracted_data.get('nationality'),
            }

            # Preserve _name_match for cross-check display (even though it's internal metadata)
            if '_name_match' in extracted_data:
                standardized_data['_name_match'] = extracted_data['_name_match']

            # Remove None values for cleaner output
            standardized_data = {k: v for k, v in standardized_data.items() if v is not None}

            if verbose:
                # Determine success based on having extracted meaningful data
                has_name = bool(standardized_data.get('full_name'))
                status = "✓ PASSED" if has_name else "✗ FAILED"
                print(f"\nResult: {status}")
                print(f"Processing Time: {elapsed:.2f}s")
                print(f"Detection Confidence: {detection_confidence:.1%}")
                print(f"Detected: type={detected_type}, country={detected_country}, entity={detected_entity}")
                if schema_id:
                    print(f"Schema: {schema_id}")
                print("Extracted Fields:")
                for key, value in extracted_data.items():
                    if key != '_detection' and value is not None:
                        print(f"  {key}: {value}")

            # Success if we got a name
            success = has_name
            confidence = detection_confidence * 100 if success else 0.0

            return DocumentResult(
                user="",
                filename=trc_path.name,
                document_type='tax_residency_certificate',
                success=success,
                confidence=confidence,
                extracted_data=standardized_data,
                elapsed_seconds=elapsed
            )

        except Exception as e:
            self.logger.error(f"Tax residency certificate processing failed: {e}")
            return DocumentResult(
                user="",
                filename=trc_path.name,
                document_type='tax_residency_certificate',
                success=False,
                confidence=0.0,
                error_message=str(e)
            )


# ============================================================
# RESULT PRINTER
# ============================================================

class ResultPrinter:
    """Print verification results in formatted tables."""

    @staticmethod
    def print_full_mode_summary(results: Dict[str, UserVerificationResult], detailed: bool = False):
        """Print summary for full mode verification."""
        print("\n" + "=" * 100)
        print(" " * 30 + "FULL VERIFICATION RESULTS")
        print("=" * 100)

        for user_name, user_result in sorted(results.items()):
            print(f"\n{'─'*100}")
            print(f"USER: {user_name}")
            print(f"{'─'*100}")

            # Document summary
            docs_summary = []

            if user_result.selfie_result:
                sr = user_result.selfie_result
                status = "✓" if sr.success else "✗"
                docs_summary.append(f"Selfie: {status}")

            if user_result.passport_result:
                pr = user_result.passport_result
                status = "✓" if pr.success else "✗"
                name = pr.extracted_data.get('full_name', '-') or '-'
                docs_summary.append(f"Passport: {status} ({name[:20]})")

            for br in user_result.bank_statement_results:
                status = "✓" if br.success else "✗"
                bank = br.extracted_data.get('bank_name', '-') or '-'
                docs_summary.append(f"Bank: {status} ({bank[:15]})")

            for ot in user_result.other_results:
                status = "✓" if ot.success else "✗"
                docs_summary.append(f"{ot.document_type}: {status}")

            print("Documents: " + " | ".join(docs_summary))

            # Cross-checks
            if user_result.cross_checks:
                print("\nCross-Checks:")
                for check_name, check_data in user_result.cross_checks.items():
                    if check_name == 'face_match':
                        status = "✓ PASSED" if check_data.get('passed') else "✗ FAILED"
                        conf = check_data.get('confidence', 0) or 0
                        print(f"  Face Match: {status} ({conf:.1f}%)")
                    elif check_name.startswith('id_name_match_'):
                        status = "✓ PASSED" if check_data.get('passed') else "✗ FAILED"
                        sim = check_data.get('similarity', 0) or 0
                        print(f"  ID Card Name Match: {status} ({sim:.1f}%)")
                    elif check_name.startswith('trc_name_match_'):
                        status = "✓ PASSED" if check_data.get('passed') else "✗ FAILED"
                        sim = check_data.get('similarity', 0) or 0
                        print(f"  TRC Name Match: {status} ({sim:.1f}%)")
                    elif check_name.startswith('name_match_'):
                        status = "✓ PASSED" if check_data.get('passed') else "✗ FAILED"
                        sim = check_data.get('similarity', 0) or 0
                        print(f"  Name Match (Bank): {status} ({sim:.1f}%)")
                    elif check_name.startswith('pan_name_match_'):
                        status = "✓ PASSED" if check_data.get('passed') else "✗ FAILED"
                        sim = check_data.get('similarity', 0) or 0
                        print(f"  PAN Name Match: {status} ({sim:.1f}%)")
                    elif check_name.startswith('pan_dob_match_'):
                        status = "✓ PASSED" if check_data.get('passed') else "✗ FAILED"
                        passport_dob = check_data.get('passport_dob', '')
                        pan_dob = check_data.get('pan_dob', '')
                        print(f"  PAN DOB Match: {status} (Passport: {passport_dob}, PAN: {pan_dob})")

        # Overall summary
        print("\n" + "=" * 100)
        print("SUMMARY")
        print("=" * 100)

        total_users = len(results)
        passed_users = sum(
            1 for r in results.values()
            if (r.passport_result is None or r.passport_result.success)
            and all(br.success for br in r.bank_statement_results)
        )

        print(f"Total Users: {total_users}")
        print(f"Fully Verified: {passed_users}")
        print(f"Partial/Failed: {total_users - passed_users}")

    @staticmethod
    def print_document_results(results: List[DocumentResult], mode_name: str, detailed: bool = False):
        """Print summary for document-only modes (passport/bank/pan_card)."""
        # Adjust table width based on mode
        if mode_name.lower() == 'passport':
            header = f"\n{'User':<12} {'Filename':<25} {'Status':<10} {'Similarity':<12} {'Time':<10}"
            separator = "-" * 85
            print("\n" + "=" * 100)
            print(" " * 30 + f"{mode_name.upper()} MODE RESULTS")
            print("=" * 100)
        elif mode_name.lower() == 'pan card':
            header = f"\n{'User':<12} {'Filename':<20} {'Status':<10} {'PAN Number':<12} {'Full Name':<25} {'Father Name':<20} {'DOB':<12} {'Type':<8} {'Time':<8}"
            separator = "-" * 145
            print("\n" + "=" * 145)
            print(" " * 45 + f"{mode_name.upper()} MODE RESULTS")
            print("=" * 145)
        else:
            # Bank mode with detailed columns (including address components)
            header = f"\n{'User':<10} {'Filename':<20} {'Status':<8} {'Bank':<10} {'Holder':<16} {'Address':<20} {'City':<12} {'State':<10} {'Postal':<8} {'Ctry':<4} {'Acct#':<12} {'Curr':<4} {'Stmt Date':<12}"
            separator = "-" * 162
            print("\n" + "=" * 162)
            print(" " * 50 + f"{mode_name.upper()} MODE RESULTS")
            print("=" * 150)

        passed = sum(1 for r in results if r.success)
        failed = len(results) - passed
        total = len(results)

        # Summary table
        print(header)
        print(separator)

        for result in results:
            status = "✓ PASS" if result.success else "✗ FAIL"

            if mode_name.lower() == 'passport':
                time_str = f"{result.elapsed_seconds:.2f}s"
                # Show similarity score for passport mode
                sim_score = result.extracted_data.get('similarity_score')
                if sim_score is not None:
                    # Get threshold from other_checks or use default
                    threshold = result.extracted_data.get('other_checks', {}).get('reference_threshold', 0.5)
                    sim_status = "✓" if sim_score >= threshold else "✗"
                    sim_str = f"{sim_status} {sim_score:.3f}"
                else:
                    sim_str = "-"
                filename = result.filename[:23] + ".." if len(result.filename) > 25 else result.filename
                print(f"{result.user:<12} {filename:<25} {status:<10} {sim_str:<12} {time_str:<10}")
            elif mode_name.lower() == 'pan card':
                time_str = f"{result.elapsed_seconds:.2f}s"
                # PAN card specific fields
                pan_number = result.extracted_data.get('pan_number', '') or '-'
                full_name = result.extracted_data.get('full_name', '') or '-'
                father_name = result.extracted_data.get('father_name', '') or '-'
                dob = result.extracted_data.get('date_of_birth', '') or '-'
                pan_type = result.extracted_data.get('pan_card_type', '') or '-'

                # Truncate long values
                full_name = full_name[:23] + '..' if len(full_name) > 25 else full_name
                father_name = father_name[:18] + '..' if len(father_name) > 20 else father_name
                filename = result.filename[:18] + '..' if len(result.filename) > 20 else result.filename

                print(f"{result.user:<12} {filename:<20} {status:<10} {pan_number:<12} {full_name:<25} {father_name:<20} {dob:<12} {pan_type:<8} {time_str:<8}")
            else:
                # Bank mode with detected values
                bank = result.extracted_data.get('bank_code', '') or '-'  # Show SWIFT code instead of bank name
                holder = result.extracted_data.get('account_holder_name', '') or '-'
                # Address components
                street = result.extracted_data.get('address', '') or '-'
                city = result.extracted_data.get('address_city', '') or '-'
                state = result.extracted_data.get('address_state', '') or '-'
                postal = result.extracted_data.get('address_postal', '') or '-'
                addr_ctry = result.extracted_data.get('address_country', '') or '-'
                acct = result.extracted_data.get('account_number', '') or '-'
                curr = result.extracted_data.get('currency', '') or '-'
                stmt_date = result.extracted_data.get('statement_date', '') or '-'

                # Truncate long values
                bank = bank[:9] if len(bank) > 10 else bank
                holder = holder[:14] + '..' if len(holder) > 16 else holder
                street = street[:18] + '..' if len(street) > 20 else street
                city = city[:10] + '..' if len(city) > 12 else city
                state = state[:8] + '..' if len(state) > 10 else state
                postal = postal[:8]
                addr_ctry = addr_ctry[:4]
                acct = acct[:10] + '..' if len(acct) > 12 else acct
                curr = curr[:4]
                filename = result.filename[:18] + '..' if len(result.filename) > 20 else result.filename

                print(f"{result.user:<10} {filename:<20} {status:<8} {bank:<10} {holder:<16} {street:<20} {city:<12} {state:<10} {postal:<8} {addr_ctry:<4} {acct:<12} {curr:<4} {stmt_date:<12}")

        print(separator)
        print(f"\nTotal: {total} | Passed: {passed} | Failed: {failed}")

        if total > 0:
            success_rate = (passed / total) * 100
            avg_time = sum(r.elapsed_seconds for r in results) / total
            print(f"Success Rate: {success_rate:.1f}% | Avg Time: {avg_time:.2f}s")

        # Detailed output if requested
        if detailed:
            print("\n" + "=" * 120)
            print("DETAILED RESULTS")
            print("=" * 120)

            for result in results:
                print(f"\n{'─'*100}")
                print(f"File: {result.filename} (User: {result.user})")
                print(f"Status: {'SUCCESS' if result.success else 'FAILED'}")

                if result.error_message:
                    print(f"Error: {result.error_message}")

                if result.extracted_data:
                    print("\nExtracted Fields:")
                    for key, value in result.extracted_data.items():
                        if key in ['validation', 'other_checks', 'validation_results']:
                            continue
                        if value is not None:
                            value_str = str(value)
                            if len(value_str) > 80:
                                value_str = value_str[:77] + "..."
                            print(f"  {key}: {value_str}")

                    # Show validation results for bank statements
                    if mode_name.lower() in ['bank', 'bank statement']:
                        val_results = result.extracted_data.get('validation_results', {})
                        if val_results:
                            print("\nValidation Results:")
                            for check_name, check_data in val_results.items():
                                if isinstance(check_data, dict):
                                    is_valid = check_data.get('valid', 'N/A')
                                    status_icon = "✓" if is_valid is True else "✗" if is_valid is False else "?"
                                    print(f"  {check_name}: {status_icon} {is_valid}")
                                    # Show additional details
                                    if check_name == 'address_components':
                                        components = check_data.get('components', {})
                                        if components:
                                            print(f"    Components: {components}")
                                        missing = check_data.get('missing', [])
                                        if missing:
                                            print(f"    Missing: {missing}")
                                    elif check_name == 'bank_lookup':
                                        swift = check_data.get('swift_code')
                                        if swift:
                                            print(f"    SWIFT: {swift}")
                                        error = check_data.get('error')
                                        if error:
                                            print(f"    Error: {error}")
                                    elif check_name == 'account_number':
                                        method = result.extracted_data.get('account_number_extraction_method')
                                        if method:
                                            print(f"    Extraction Method: {method}")
                                        error = check_data.get('error')
                                        if error:
                                            print(f"    Error: {error}")
                                else:
                                    print(f"  {check_name}: {check_data}")


# ============================================================
# MAIN ENTRY POINT
# ============================================================

async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Unified Verification Test Script',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full verification with cross-checks for specific user
  poetry run python scripts/test_verification.py --mode full --user user_004

  # Full verification for all users
  poetry run python scripts/test_verification.py --mode full

  # Passport-only across all users (auto-skips face matching)
  poetry run python scripts/test_verification.py --mode passport

  # Bank statement-only with default extractor
  poetry run python scripts/test_verification.py --mode bank

  # Bank statement-only with spatial extractor
  poetry run python scripts/test_verification.py --mode bank --extractor spatial

  # Bank statement for specific user with spatial extractor
  poetry run python scripts/test_verification.py --mode bank --user user_004 --extractor spatial

  # Auto mode with filename-based hints for specific user
  poetry run python scripts/test_verification.py --mode auto --user user_004

  # Auto mode across all users
  poetry run python scripts/test_verification.py --mode auto

  # Filename convention: {doc_type}_{country}_{entity}.{ext}
  # Shorthand: trc, tax, id, pan, dl, bank
  # Examples: trc_AE.jpg, tax_SG_IRAS.pdf, pan_IN.jpg, id_SG.jpg

  # Skip forgery detection (useful if PyTorch has issues)
  poetry run python scripts/test_verification.py --mode passport --skip-photoholmes

  # Detailed output
  poetry run python scripts/test_verification.py --mode full --detailed
        """
    )

    parser.add_argument(
        '--mode', '-m',
        type=str,
        choices=['full', 'passport', 'bank', 'pan_card', 'auto'],
        default='full',
        help='Test mode: full (all docs with cross-checks), passport (passports only), bank (bank statements only), pan_card (PAN cards only), auto (filename-based hints)'
    )

    parser.add_argument(
        '--user', '-u',
        type=str,
        action='append',
        help='Specific user(s) to test (can be specified multiple times)'
    )

    parser.add_argument(
        '--folder', '-f',
        type=str,
        default='scripts/test_data',
        help='Base folder containing user folders (default: scripts/test_data)'
    )

    parser.add_argument(
        '--skip-photoholmes',
        action='store_true',
        help='Skip PhotoHolmes forgery detection (useful if PyTorch has compatibility issues)'
    )

    parser.add_argument(
        '--skip-face',
        action='store_true',
        help='Skip face matching (useful when testing without selfie image)'
    )

    parser.add_argument(
        '--detailed', '-d',
        action='store_true',
        help='Show detailed results for each document'
    )

    parser.add_argument(
        '--extractor', '-e',
        type=str,
        choices=['default', 'spatial'],
        default='default',
        help='Bank statement extractor: default (GLiNER-based) or spatial (geometry-based)'
    )

    # OCR engine argument removed - DocTR is now the only OCR engine

    args = parser.parse_args()

    # Configure skip flags
    if args.skip_photoholmes:
        from app.config.verification_config import verification_settings
        verification_settings.skip_photoholmes = True
        print("⚠️  PhotoHolmes forgery detection DISABLED via --skip-photoholmes")

    if args.skip_face:
        from app.config.verification_config import verification_settings
        verification_settings.skip_face_matching = True
        print("⚠️  Face matching DISABLED via --skip-face")

    # Validate folder exists
    base_folder = Path(args.folder)
    if not base_folder.exists():
        print(f"Error: Folder not found: {args.folder}")
        return

    # Get user folders
    filter_users = args.user
    user_folders = FileDiscovery.get_user_folders(base_folder, filter_users)

    if not user_folders:
        print(f"No user folders found in: {args.folder}")
        if filter_users:
            print(f"Looking for users: {', '.join(filter_users)}")
        return

    # Create orchestrator and run appropriate mode
    # Using DocTR OCR engine for passport processing
    print(f"🔧 Using DocTR engine for passport processing")
    orchestrator = VerificationOrchestrator(extractor=args.extractor)

    if args.mode == 'full':
        print(f"\n{'#'*80}")
        print(f"# UNIFIED VERIFICATION TEST - FULL MODE")
        print(f"{'#'*80}")
        print(f"Base Folder: {args.folder}")
        print(f"Users to test: {len(user_folders)}")
        if filter_users:
            print(f"Filtered to: {', '.join(filter_users)}")

        results = await orchestrator.run_full_mode(
            user_folders,
            skip_photoholmes=args.skip_photoholmes,
            skip_face_matching=args.skip_face,
            verbose=not args.detailed
        )

        ResultPrinter.print_full_mode_summary(results, detailed=args.detailed)

    elif args.mode == 'passport':
        results = await orchestrator.run_passport_mode(
            base_folder,
            filter_users=filter_users,
            skip_photoholmes=args.skip_photoholmes,
            verbose=not args.detailed
        )

        if not results:
            print("No passport files found in user folders")
            return

        ResultPrinter.print_document_results(results, "Passport", detailed=args.detailed)

    elif args.mode == 'bank':
        results = await orchestrator.run_bank_mode(
            base_folder,
            filter_users=filter_users,
            verbose=not args.detailed
        )

        if not results:
            print("No bank statement files found in user folders")
            return

        ResultPrinter.print_document_results(results, "Bank Statement", detailed=args.detailed)

    elif args.mode == 'pan_card':
        results = await orchestrator.run_pan_card_mode(
            base_folder,
            filter_users=filter_users,
            skip_photoholmes=args.skip_photoholmes,
            verbose=not args.detailed
        )

        if not results:
            print("No PAN card files found in user folders")
            return

        ResultPrinter.print_document_results(results, "PAN Card", detailed=args.detailed)

    elif args.mode == 'auto':
        auto_tester = AutoDocumentTester()
        results = await orchestrator.run_auto_mode(
            user_folders=user_folders,
            auto_tester=auto_tester,
            verbose=not args.detailed
        )

        if not results:
            print("No files with hint-based naming found in user folders")
            return

        # Print summary for each user
        for user_name, user_results in results.items():
            print(f"\n{'='*60}")
            print(f"User: {user_name}")
            print(f"Files processed: {len(user_results)}")
            print(f"Success: {sum(1 for r in user_results if r.success)}")
            print(f"Failed: {sum(1 for r in user_results if not r.success)}")


if __name__ == "__main__":
    asyncio.run(main())
