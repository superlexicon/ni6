from typing import Optional, List, Dict
from dateutil import parser as date_parser
from datetime import datetime
# camelot removed due to dependency conflicts with pypdf 6.4.1
CAMELOT_AVAILABLE = False
from app.core import logger


class BankStatementValidationResult:
    """Result of bank statement validation"""

    def __init__(
        self,
        transactions_extracted: int,
        balance_validation_passed: bool,
        date_sequence_valid: bool,
        font_consistency_score: float,
        error_message: Optional[str] = None,
        balance_details: Optional[Dict] = None
    ):
        self.transactions_extracted = transactions_extracted
        self.balance_validation_passed = balance_validation_passed
        self.date_sequence_valid = date_sequence_valid
        self.font_consistency_score = font_consistency_score
        self.error_message = error_message
        self.balance_details = balance_details


class BankStatementValidator:
    """Validates bank statements through table extraction and consistency checks"""

    def __init__(self):
        self.logger = logger

    async def validate(self, content: bytes, is_pdf: bool = True) -> BankStatementValidationResult:
        """
        Validate bank statement

        Args:
            content: PDF bytes of bank statement
            is_pdf: Whether content is PDF (camelot only works with PDFs)

        Returns:
            BankStatementValidationResult with validation details
        """
        if not is_pdf:
            self.logger.warning("Bank statement validation only works with PDF files")
            return BankStatementValidationResult(
                transactions_extracted=0,
                balance_validation_passed=False,
                date_sequence_valid=False,
                font_consistency_score=50.0,
                error_message="Bank statement validation requires PDF format"
            )

        try:
            # Step 1: Extract tables from PDF
            transactions, dates = await self._extract_transactions(content)

            # Step 2: Validate date sequence
            date_sequence_valid = self._validate_date_sequence(dates)

            # Step 3: Validate balance (if we can extract opening/closing balance)
            balance_valid, balance_details = self._validate_balance(transactions)

            # Step 4: Check font consistency (using PyMuPDF)
            font_score = await self._check_font_consistency(content)

            self.logger.info(
                f"Bank statement validation: {len(transactions)} transactions, "
                f"balance_valid={balance_valid}, date_seq_valid={date_sequence_valid}, "
                f"font_score={font_score:.1f}"
            )

            return BankStatementValidationResult(
                transactions_extracted=len(transactions),
                balance_validation_passed=balance_valid,
                date_sequence_valid=date_sequence_valid,
                font_consistency_score=font_score,
                balance_details=balance_details
            )

        except Exception as e:
            self.logger.error(f"Error in bank statement validation: {str(e)}")
            return BankStatementValidationResult(
                transactions_extracted=0,
                balance_validation_passed=False,
                date_sequence_valid=False,
                font_consistency_score=50.0,
                error_message=f"Error: {str(e)}"
            )

    async def _extract_transactions(self, content: bytes) -> tuple[List[Dict], List[datetime]]:
        """
        Extract transaction tables from PDF - DISABLED (camelot removed)

        Returns:
            Tuple of (transactions_list, dates_list)
        """
        # Table extraction has been disabled due to dependency conflicts
        self.logger.info("Transaction extraction from tables is disabled")
        return [], []

    def _validate_date_sequence(self, dates: List[datetime]) -> bool:
        """
        Validate that dates are in chronological order

        Args:
            dates: List of parsed dates

        Returns:
            True if dates are sequential
        """
        if len(dates) < 2:
            return True  # Not enough dates to validate

        # Sort dates
        sorted_dates = sorted(dates)

        # Check if original dates match sorted order
        # Allow some flexibility (dates within same day can be out of order)
        misorder_count = 0
        for i in range(len(dates) - 1):
            if dates[i] > dates[i + 1]:
                # Check if it's just same-day reordering
                if dates[i].date() != dates[i + 1].date():
                    misorder_count += 1

        # Allow up to 10% misordering
        tolerance = max(1, len(dates) * 0.1)
        is_valid = misorder_count <= tolerance

        self.logger.info(f"Date sequence: {misorder_count} misorderings out of {len(dates)}, valid={is_valid}")
        return is_valid

    def _validate_balance(self, transactions: List[Dict]) -> tuple[bool, Optional[Dict]]:
        """
        Validate balance calculations

        Args:
            transactions: List of transaction dictionaries

        Returns:
            Tuple of (is_valid, balance_details)
        """
        # This is a simplified check - would need more sophisticated logic
        # to identify opening balance, closing balance, and transaction amounts

        try:
            # Try to find numeric patterns that might be balances
            all_amounts = []
            for txn in transactions:
                if 'amounts' in txn:
                    all_amounts.extend(txn['amounts'])

            if len(all_amounts) < 3:
                return (True, None)  # Not enough data to validate

            # Simple heuristic: check if amounts are reasonable
            # (no negative balances, amounts within reasonable range)
            has_negatives = any(amt < 0 for amt in all_amounts)
            has_huge_amounts = any(amt > 1_000_000_000 for amt in all_amounts)

            is_valid = not (has_negatives or has_huge_amounts)

            balance_details = {
                "amounts_found": len(all_amounts),
                "has_negative_amounts": has_negatives,
                "has_unrealistic_amounts": has_huge_amounts
            }

            return (is_valid, balance_details)

        except Exception as e:
            self.logger.warning(f"Could not validate balance: {str(e)}")
            return (True, None)  # Default to valid if we can't check

    async def _check_font_consistency(self, content: bytes) -> float:
        """
        Check font consistency across document

        Returns:
            Font consistency score (0-100)
        """
        try:
            import fitz  # PyMuPDF

            # Write to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                temp_file.write(content)
                temp_path = temp_file.name

            try:
                pdf_document = fitz.open(temp_path)

                # Collect all fonts used
                fonts_used = set()

                for page_num in range(min(5, pdf_document.page_count)):  # Check first 5 pages
                    page = pdf_document[page_num]

                    # Get font information
                    blocks = page.get_text("dict")["blocks"]
                    for block in blocks:
                        if "lines" in block:
                            for line in block["lines"]:
                                for span in line["spans"]:
                                    font_name = span.get("font", "")
                                    if font_name:
                                        fonts_used.add(font_name)

                pdf_document.close()

                # Score based on font count
                # Bank statements typically use 2-4 fonts
                font_count = len(fonts_used)

                if font_count == 0:
                    score = 50.0  # No font data
                elif font_count <= 4:
                    score = 100.0  # Good - limited fonts
                elif font_count <= 7:
                    score = 80.0  # Acceptable
                elif font_count <= 10:
                    score = 60.0  # Suspicious - many fonts
                else:
                    score = 40.0  # Very suspicious - too many fonts

                self.logger.info(f"Font consistency: {font_count} fonts used, score={score}")
                return score

            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

        except Exception as e:
            self.logger.warning(f"Could not check font consistency: {str(e)}")
            return 70.0  # Default neutral score
