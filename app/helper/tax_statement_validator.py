from pydantic import BaseModel
from datetime import datetime

from app.schemas.tax_statement_schema import TaxStatementData
from app.core import logger


class TaxStatementValidationResult(BaseModel):
    """Validation result for tax statement"""
    has_required_fields: bool
    tax_year_valid: bool
    amounts_consistent: bool
    overall_valid: bool
    validation_score: float  # 0-100
    findings: list[str] = []


class TaxStatementValidator:
    """Validate tax statement documents"""

    def __init__(self):
        self.logger = logger

    async def validate(self, tax_data: TaxStatementData) -> TaxStatementValidationResult:
        """
        Validate tax statement document.

        Args:
            tax_data: Extracted tax statement data

        Returns:
            TaxStatementValidationResult with validation details
        """
        try:
            findings = []
            validation_score = 0.0

            # Check required fields
            required_fields = [
                'taxpayer_name',
                'tax_year',
                'gross_income'
            ]

            has_required_fields = all(
                getattr(tax_data, field) is not None
                for field in required_fields
            )

            if has_required_fields:
                findings.append("All required fields present")
                validation_score += 40.0
            else:
                missing = [f for f in required_fields if getattr(tax_data, f) is None]
                findings.append(f"Missing required fields: {', '.join(missing)}")

            # Validate tax year (should be reasonable)
            tax_year_valid = False
            if tax_data.tax_year:
                try:
                    year = int(tax_data.tax_year)
                    current_year = datetime.now().year
                    tax_year_valid = 1900 <= year <= current_year

                    if tax_year_valid:
                        findings.append(f"Tax year valid: {year}")
                        validation_score += 20.0
                    else:
                        findings.append(f"Tax year out of range: {year}")
                except ValueError:
                    findings.append(f"Invalid tax year format: {tax_data.tax_year}")

            # Validate amount consistency
            amounts_consistent = True
            if tax_data.gross_income and tax_data.taxable_income:
                if tax_data.taxable_income > tax_data.gross_income:
                    amounts_consistent = False
                    findings.append("Inconsistency: Taxable income exceeds gross income")
                else:
                    findings.append("Income amounts consistent")
                    validation_score += 20.0

            # Check for tax authority markers
            if tax_data.tax_authority:
                findings.append(f"Tax authority identified: {tax_data.tax_authority}")
                validation_score += 10.0

            # Check confidence scores
            if tax_data.confidence_scores:
                avg_confidence = sum(tax_data.confidence_scores.values()) / len(tax_data.confidence_scores)
                findings.append(f"Average extraction confidence: {avg_confidence:.1f}%")
                validation_score += min(10.0, avg_confidence / 10)

            # Overall validation
            overall_valid = (
                has_required_fields and
                tax_year_valid and
                amounts_consistent
            )

            self.logger.info(
                f"Tax statement validation: Required={has_required_fields}, "
                f"Year={tax_year_valid}, Consistent={amounts_consistent}, "
                f"Score={validation_score:.1f}%"
            )

            return TaxStatementValidationResult(
                has_required_fields=has_required_fields,
                tax_year_valid=tax_year_valid,
                amounts_consistent=amounts_consistent,
                overall_valid=overall_valid,
                validation_score=validation_score,
                findings=findings
            )

        except Exception as e:
            self.logger.error(f"Tax statement validation failed: {str(e)}")
            return TaxStatementValidationResult(
                has_required_fields=False,
                tax_year_valid=False,
                amounts_consistent=False,
                overall_valid=False,
                validation_score=0.0,
                findings=[f"Validation error: {str(e)}"]
            )
