"""
Generic document schemas for common document types.

This module defines base schemas for document types that are not passport,
bank_statement, or selfie. These schemas can be used as-is or extended
with country-specific and entity-specific variants.

Document Types:
- tax_return: Tax assessment documents, tax returns
- id_card: National ID cards (not passport, not selfie)
- driving_license: Driver's license documents
- utility_bill: Utility bills (electricity, water, gas, internet)
- payslip: Employee payslip documents
- insurance_policy: Insurance policy documents
- employment_letter: Employment verification letters
- residence_proof: Proof of residence documents
"""

from typing import Dict, List, Optional, Set

from .base import DocumentTypeSchema, GLINER2Schema
from . import SchemaRegistry


# ============================================================================
# TAX RETURN SCHEMAS
# ============================================================================

TAX_RETURN_GENERIC = DocumentTypeSchema(
    schema_id="tax_return",
    document_type="tax_return",
    document_type_name="Tax Return",
    extraction_schema=GLINER2Schema(
        fields={
            "taxpayer_name": "The full legal name of the taxpayer",
            "tax_id_number": "The tax identification number, tax ID, or taxpayer identification number",
            "assessment_year": "The year of assessment or tax year",
            "tax_period": "The tax period covered by the return",
            "total_income": "The total income amount or gross income",
            "taxable_income": "The taxable income amount after deductions",
            "tax_payable": "The total tax payable or tax due amount",
            "tax_paid": "The amount of tax already paid",
            "tax_refund": "The refund amount or overpayment",
            "filing_date": "The date the tax return was filed",
            "issuing_authority": "The name of the tax authority issuing the document",
        },
        threshold=0.2,
    ),
    detection_patterns={
        "keywords": [
            "tax return", "income tax", "tax assessment", "notice of assessment",
            "tax statement", "tax computation", "taxable income", "tax payable",
            "assessment year", "tax year", "federal tax", "state tax"
        ],
        "labels": [
            "tax authority", "revenue authority", "tax department", "inland revenue"
        ],
    },
    required_fields=["taxpayer_name"],
    optional_fields=[
        "tax_id_number", "assessment_year", "total_income", "tax_payable",
        "tax_paid", "tax_refund", "filing_date", "issuing_authority"
    ],
    priority=10,
)


# ============================================================================
# ID CARD SCHEMAS (National ID cards, not passport, not selfie)
# ============================================================================

ID_CARD_GENERIC = DocumentTypeSchema(
    schema_id="id_card",
    document_type="id_card",
    document_type_name="National ID Card",
    extraction_schema=GLINER2Schema(
        fields={
            "full_name": "The cardholder's full legal name",
            "given_name": "The first name or given name",
            "family_name": "The last name, surname, or family name",
            "date_of_birth": "The date of birth of the cardholder",
            "place_of_birth": "The place of birth city or country",
            "sex": "The gender or sex of the cardholder",
            "identification_number": "The national ID number or identification number",
            "national_id_number": "The national identification number unique to the person",
            "issue_date": "The date the ID card was issued",
            "expiry_date": "The expiration date of the ID card",
            "nationality": "The nationality of the cardholder",
            "address": "The residential address of the cardholder",
            "blood_group": "The blood group of the cardholder",
        },
        threshold=0.2,
    ),
    detection_patterns={
        "keywords": [
            "identity card", "national id", "identification card", "id card",
            "national identity", "citizen card", "personal identity"
        ],
        "labels": [
            "government", "identity", "citizen", "national registration"
        ],
    },
    required_fields=["identification_number"],
    optional_fields=[
        "full_name", "given_name", "family_name", "date_of_birth",
        "sex", "issue_date", "expiry_date", "nationality", "address"
    ],
    priority=10,
)


# ============================================================================
# DRIVING LICENSE SCHEMAS
# ============================================================================

DRIVING_LICENSE_GENERIC = DocumentTypeSchema(
    schema_id="driving_license",
    document_type="driving_license",
    document_type_name="Driving License",
    extraction_schema=GLINER2Schema(
        fields={
            "full_name": "The license holder's full legal name",
            "date_of_birth": "The date of birth of the license holder",
            "license_number": "The driving license number or driver's license number",
            "issue_date": "The date the license was issued",
            "expiry_date": "The expiration date of the license",
            "vehicle_class": "The class of vehicle the license is valid for",
            "vehicle_categories": "The categories of vehicles the holder can drive",
            "issuing_authority": "The authority that issued the license",
            "address": "The residential address of the license holder",
            "sex": "The gender or sex of the license holder",
            "restrictions": "Any restrictions on the license",
            "endorsements": "Any endorsements on the license",
        },
        threshold=0.2,
    ),
    detection_patterns={
        "keywords": [
            "driver's license", "driving licence", "driving license",
            "driver license", "motor vehicle license", "vehicle licence",
            "class", "driving permit", "motor vehicle department"
        ],
        "labels": [
            "transport", "motor vehicles", "driver and vehicle licensing",
            "department of motor vehicles", "road transport"
        ],
    },
    required_fields=["license_number", "full_name"],
    optional_fields=[
        "date_of_birth", "issue_date", "expiry_date", "vehicle_class",
        "vehicle_categories", "issuing_authority", "address", "sex"
    ],
    priority=10,
)


# ============================================================================
# UTILITY BILL SCHEMAS
# ============================================================================

UTILITY_BILL_GENERIC = DocumentTypeSchema(
    schema_id="utility_bill",
    document_type="utility_bill",
    document_type_name="Utility Bill",
    extraction_schema=GLINER2Schema(
        fields={
            "customer_name": "The name of the customer or account holder",
            "account_number": "The customer account number",
            "service_address": "The address where the service is provided",
            "billing_period": "The period covered by the bill",
            "bill_date": "The date the bill was issued",
            "due_date": "The payment due date",
            "amount_due": "The total amount due for payment",
            "amount_paid": "The amount already paid",
            "previous_balance": "The balance from the previous billing period",
            "current_charges": "The charges for the current period",
            "utility_type": "The type of utility (electricity, water, gas, internet)",
            "provider_name": "The name of the utility provider",
            "customer_id": "The customer ID or customer reference number",
            "meter_number": "The meter number for utility consumption",
            "usage": "The usage amount or consumption",
        },
        threshold=0.2,
    ),
    detection_patterns={
        "keywords": [
            "utility bill", "electricity bill", "water bill", "gas bill",
            "internet bill", "phone bill", "utility statement", "bill statement",
            "amount due", "payment due", "billing period", "service address"
        ],
        "labels": [
            "electric utility", "water utility", "gas utility", "utility provider",
            "telecommunications", "internet service provider"
        ],
    },
    required_fields=["customer_name", "service_address"],
    optional_fields=[
        "account_number", "bill_date", "due_date", "amount_due",
        "utility_type", "provider_name", "billing_period"
    ],
    priority=10,
)


# ============================================================================
# PAYSLIP SCHEMAS
# ============================================================================

PAYSLIP_GENERIC = DocumentTypeSchema(
    schema_id="payslip",
    document_type="payslip",
    document_type_name="Payslip",
    extraction_schema=GLINER2Schema(
        fields={
            "employee_name": "The full name of the employee",
            "employee_id": "The employee ID or staff number",
            "pay_period": "The pay period covered by the payslip",
            "pay_date": "The payment date or date of payment",
            "gross_salary": "The gross salary or gross pay amount",
            "net_salary": "The net salary or take-home pay amount",
            "basic_salary": "The basic salary amount",
            "total_deductions": "The total amount of deductions",
            "total_earnings": "The total earnings before deductions",
            "employer_name": "The name of the employer or company",
            "tax_deduction": "The income tax deduction amount",
            "social_security": "The social security or provident fund contribution",
            "income_tax": "The income tax withheld",
        },
        threshold=0.2,
    ),
    detection_patterns={
        "keywords": [
            "payslip", "pay slip", "salary slip", "pay statement",
            "earnings statement", "pay advice", "payroll", "wage statement",
            "net pay", "gross pay", "pay period", "year to date"
        ],
        "labels": [
            "payroll", "human resources", "compensation", "salary"
        ],
    },
    required_fields=["employee_name", "employer_name"],
    optional_fields=[
        "employee_id", "pay_period", "pay_date", "gross_salary",
        "net_salary", "basic_salary", "total_deductions"
    ],
    priority=10,
)


# ============================================================================
# INSURANCE POLICY SCHEMAS
# ============================================================================

INSURANCE_POLICY_GENERIC = DocumentTypeSchema(
    schema_id="insurance_policy",
    document_type="insurance_policy",
    document_type_name="Insurance Policy",
    extraction_schema=GLINER2Schema(
        fields={
            "policyholder_name": "The name of the policyholder",
            "policy_number": "The insurance policy number",
            "policy_type": "The type of insurance policy",
            "insurance_company": "The name of the insurance company",
            "issue_date": "The date the policy was issued",
            "expiry_date": "The policy expiration date",
            "coverage_amount": "The sum assured or coverage amount",
            "premium_amount": "The premium amount to be paid",
            "premium_frequency": "The frequency of premium payment",
            "beneficiary_name": "The name of the beneficiary",
            "agent_name": "The name of the insurance agent",
        },
        threshold=0.2,
    ),
    detection_patterns={
        "keywords": [
            "insurance policy", "policy document", "insurance certificate",
            "policy number", "sum assured", "insurance company",
            "policyholder", "beneficiary", "premium", "coverage"
        ],
        "labels": [
            "insurance", "assurance", "policy", "coverage"
        ],
    },
    required_fields=["policyholder_name", "policy_number", "insurance_company"],
    optional_fields=[
        "policy_type", "issue_date", "expiry_date", "coverage_amount",
        "premium_amount", "beneficiary_name"
    ],
    priority=10,
)


# ============================================================================
# EMPLOYMENT LETTER SCHEMAS
# ============================================================================

EMPLOYMENT_LETTER_GENERIC = DocumentTypeSchema(
    schema_id="employment_letter",
    document_type="employment_letter",
    document_type_name="Employment Letter",
    extraction_schema=GLINER2Schema(
        fields={
            "employee_name": "The full name of the employee",
            "employer_name": "The name of the employer or company",
            "employment_start_date": "The start date of employment",
            "job_title": "The job title or position",
            "department": "The department or division",
            "employment_type": "The type of employment (full-time, part-time, contract)",
            "annual_salary": "The annual salary amount",
            "salary_currency": "The currency of the salary",
            "letter_date": "The date of the employment letter",
            "signatory_name": "The name of the person signing the letter",
            "signatory_title": "The title of the person signing the letter",
        },
        threshold=0.2,
    ),
    detection_patterns={
        "keywords": [
            "employment letter", "offer letter", "employment contract",
            "appointment letter", "job offer", "offer of employment",
            "employment verification", "proof of employment", "work confirmation"
        ],
        "labels": [
            "human resources", "employment", "appointment", "offer"
        ],
    },
    required_fields=["employee_name", "employer_name", "employment_start_date"],
    optional_fields=[
        "job_title", "department", "employment_type", "annual_salary",
        "letter_date", "signatory_name"
    ],
    priority=10,
)


# ============================================================================
# RESIDENCE PROOF SCHEMAS
# ============================================================================

RESIDENCE_PROOF_GENERIC = DocumentTypeSchema(
    schema_id="residence_proof",
    document_type="residence_proof",
    document_type_name="Proof of Residence",
    extraction_schema=GLINER2Schema(
        fields={
            "resident_name": "The name of the resident",
            "address": "The residential address",
            "document_date": "The date of the proof of residence document",
            "issuing_authority": "The authority issuing the proof",
            "document_type": "The type of document used as proof",
            "valid_from": "The date from which the proof is valid",
            "valid_until": "The date until which the proof is valid",
        },
        threshold=0.2,
    ),
    detection_patterns={
        "keywords": [
            "proof of residence", "address proof", "residence verification",
            "proof of address", "residency proof", "domicile certificate"
        ],
        "labels": [
            "residence", "address verification", "domicile"
        ],
    },
    required_fields=["resident_name", "address"],
    optional_fields=["document_date", "issuing_authority", "valid_from", "valid_until"],
    priority=10,
)


# ============================================================================
# REGISTRATION
# ============================================================================

def register_generic_schemas() -> None:
    """Register all generic document schemas in the registry."""
    schemas = [
        TAX_RETURN_GENERIC,
        ID_CARD_GENERIC,
        DRIVING_LICENSE_GENERIC,
        UTILITY_BILL_GENERIC,
        PAYSLIP_GENERIC,
        INSURANCE_POLICY_GENERIC,
        EMPLOYMENT_LETTER_GENERIC,
        RESIDENCE_PROOF_GENERIC,
    ]

    for schema in schemas:
        SchemaRegistry.register(schema)


# Auto-register on module import
register_generic_schemas()


__all__ = [
    "TAX_RETURN_GENERIC",
    "ID_CARD_GENERIC",
    "DRIVING_LICENSE_GENERIC",
    "UTILITY_BILL_GENERIC",
    "PAYSLIP_GENERIC",
    "INSURANCE_POLICY_GENERIC",
    "EMPLOYMENT_LETTER_GENERIC",
    "RESIDENCE_PROOF_GENERIC",
    "register_generic_schemas",
]
