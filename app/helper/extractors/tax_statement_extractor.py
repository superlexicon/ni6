import re
from typing import Optional

from app.schemas.tax_statement_schema import TaxStatementData
from app.helper.doctr.document_text_extractor import DocumentTextExtractor
from app.core import logger


class TaxStatementExtractor:
    """Extract all standard fields from tax statement documents"""

    def __init__(self):
        self.logger = logger
        self.text_extractor = DocumentTextExtractor()

    async def extract(self, content: bytes, is_pdf: bool = True) -> TaxStatementData:
        """
        Extract all standard tax statement fields.

        Args:
            content: Document bytes
            is_pdf: Whether content is PDF

        Returns:
            TaxStatementData with extracted fields and confidence scores
        """
        try:
            # Extract text
            text = await self.text_extractor.extract_text(content, is_pdf=is_pdf)

            tax_data = TaxStatementData()

            # Extract taxpayer name
            name_match = re.search(
                r'(?:Tax\s*payer|Name)\s*:?\s*(.+?)(?:\n|$)',
                text,
                re.IGNORECASE
            )
            if name_match:
                tax_data.taxpayer_name = name_match.group(1).strip()
                tax_data.confidence_scores['taxpayer_name'] = 75.0

            # Extract tax ID/SSN
            tax_id_match = re.search(
                r'(?:Tax\s*ID|SSN|Social\s*Security)\s*:?\s*([0-9\-]+)',
                text,
                re.IGNORECASE
            )
            if tax_id_match:
                tax_data.tax_id = tax_id_match.group(1).strip()
                tax_data.confidence_scores['tax_id'] = 80.0

            # Extract tax year
            year_match = re.search(
                r'(?:Tax\s*Year|Year)\s*:?\s*([0-9]{4})',
                text,
                re.IGNORECASE
            )
            if year_match:
                tax_data.tax_year = year_match.group(1)
                tax_data.confidence_scores['tax_year'] = 85.0

            # Extract income amounts
            gross_match = re.search(
                r'(?:Gross\s*Income|Total\s*Income)\s*:?\s*\$?([0-9,]+\.?[0-9]*)',
                text,
                re.IGNORECASE
            )
            if gross_match:
                tax_data.gross_income = float(gross_match.group(1).replace(',', ''))
                tax_data.confidence_scores['gross_income'] = 70.0

            taxable_match = re.search(
                r'Taxable\s*Income\s*:?\s*\$?([0-9,]+\.?[0-9]*)',
                text,
                re.IGNORECASE
            )
            if taxable_match:
                tax_data.taxable_income = float(taxable_match.group(1).replace(',', ''))
                tax_data.confidence_scores['taxable_income'] = 70.0

            # Extract tax paid
            tax_paid_match = re.search(
                r'(?:Tax\s*Paid|Total\s*Tax)\s*:?\s*\$?([0-9,]+\.?[0-9]*)',
                text,
                re.IGNORECASE
            )
            if tax_paid_match:
                tax_data.tax_paid = float(tax_paid_match.group(1).replace(',', ''))
                tax_data.confidence_scores['tax_paid'] = 70.0

            # Extract filing date
            filing_date_match = re.search(
                r'(?:Filing\s*Date|Date\s*Filed)\s*:?\s*([0-9]{1,2}[/-][0-9]{1,2}[/-][0-9]{2,4})',
                text,
                re.IGNORECASE
            )
            if filing_date_match:
                tax_data.filing_date = filing_date_match.group(1)
                tax_data.confidence_scores['filing_date'] = 75.0

            # Extract tax authority
            authority_match = re.search(
                r'(Internal\s*Revenue\s*Service|IRS|Tax\s*Authority)',
                text,
                re.IGNORECASE
            )
            if authority_match:
                tax_data.tax_authority = authority_match.group(1).strip()
                tax_data.confidence_scores['tax_authority'] = 70.0

            # Store raw OCR text for debugging/auditing
            tax_data.raw_data = text

            return tax_data

        except Exception as e:
            self.logger.error(f"Tax statement extraction failed: {str(e)}")
            return TaxStatementData(confidence_scores={})
