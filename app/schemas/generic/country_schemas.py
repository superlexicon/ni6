"""
Country-specific document schemas.

This module extends generic schemas with country-specific patterns, field requirements,
 and validation rules. Each country schema inherits from a generic schema and adds:
- Country-specific keywords and labels for detection
- Country-specific field patterns (e.g., NRIC format for Singapore)
- Country-specific required/optional fields

Supported Countries:
- SG: Singapore
- IN: India
- US: United States
- MY: Malaysia
- TH: Thailand
"""

from typing import Dict, List, Optional, Set

from .base import DocumentTypeSchema, GLINER2Schema
from . import SchemaRegistry


# ============================================================================
# SINGAPORE (SG) SCHEMAS
# ============================================================================

# Singapore Tax Return (IRAS)
TAX_RETURN_SG_IRAS = DocumentTypeSchema(
    schema_id="tax_return:SG:iras",
    document_type="tax_return",
    document_type_name="Tax Return",
    country_code="SG",
    country_name="Singapore",
    entity="iras",
    entity_name="Inland Revenue Authority of Singapore",
    extraction_schema=GLINER2Schema(
        fields={
            "taxpayer_name": "The taxpayer's full legal name",
            "tax_id_number": "The NRIC number or FIN number (e.g., S1234567D or T1234567G)",
            "assessment_year": "The year of assessment (e.g., 2024)",
            "tax_reference_number": "The tax reference number",
            "total_income": "The total income amount or gross income",
            "taxable_income": "The taxable income amount after deductions",
            "tax_payable": "The total tax payable or tax due amount",
            "tax_paid": "The amount of tax already paid through withholding",
            "tax_refund": "The refund amount or overpayment",
            "notice_date": "The date of the notice of assessment",
            "filing_status": "The filing status (individual, joint, etc.)",
        },
        threshold=0.2,
    ),
    detection_patterns={
        "keywords": [
            "iras", "inland revenue authority of singapore", "income tax",
            "notice of assessment", "noa", "tax assessment", "assessment year",
            "singapore tax", "sg tax", "income tax act", "gst registration"
        ],
        "labels": [
            "inland revenue authority of singapore", "iras", "comptroller of income tax"
        ],
        "patterns": [
            r"iras",
            r"inland revenue authority",
            r"notice of assessment",
            r"YA\s+\d{4}",  # Assessment Year
        ],
    },
    required_fields=["taxpayer_name"],
    optional_fields=[
        "tax_id_number", "assessment_year", "total_income", "tax_payable",
        "tax_paid", "tax_refund", "notice_date"
    ],
    priority=100,
)

# Generic Singapore Tax Return (any issuer)
TAX_RETURN_SG = DocumentTypeSchema(
    schema_id="tax_return:SG",
    document_type="tax_return",
    document_type_name="Tax Return",
    country_code="SG",
    country_name="Singapore",
    extraction_schema=GLINER2Schema(
        fields={
            "taxpayer_name": "The taxpayer's full legal name",
            "tax_id_number": "The NRIC number or FIN number (e.g., S1234567D)",
            "assessment_year": "The year of assessment",
            "total_income": "The total income amount",
            "tax_payable": "The tax payable amount",
            "issuing_authority": "The issuing tax authority",
        },
        threshold=0.2,
    ),
    detection_patterns={
        "keywords": [
            "singapore", "sg", "income tax", "tax return", "assessment",
            "iras", "inland revenue"
        ],
        "labels": ["singapore tax authority", "tax board singapore"],
    },
    required_fields=["taxpayer_name"],
    optional_fields=["tax_id_number", "assessment_year", "total_income", "tax_payable"],
    priority=50,
)

# Singapore ID Card (NRIC/FIN)
ID_CARD_SG = DocumentTypeSchema(
    schema_id="id_card:SG",
    document_type="id_card",
    document_type_name="National ID Card",
    country_code="SG",
    country_name="Singapore",
    extraction_schema=GLINER2Schema(
        fields={
            "full_name": "The cardholder's full legal name",
            "nric_number": "The NRIC number in format S/T + 7 digits + check letter (e.g., S1234567D)",
            "fin_number": "The Foreign Identification Number",
            "citizenship_status": "The citizenship status (citizen, PR, foreigner)",
            "date_of_birth": "The date of birth",
            "sex": "The gender or sex",
            "race": "The race",
            "address": "The registered address",
            "issue_date": "The date of issue",
        },
        threshold=0.2,
    ),
    detection_patterns={
        "keywords": [
            "national registration identity card", "nric", "identity card",
            "singapore", "fin", "foreign identification number", "citizen",
            "permanent resident", "singapore citizen"
        ],
        "labels": [
            "national registration department", "immigration and checkpoints authority",
            "ica", "singapore government"
        ],
        "patterns": [
            r"[STFG]\d{7}[A-Z]",  # NRIC/FIN format
        ],
    },
    required_fields=["nric_number", "full_name"],
    optional_fields=["date_of_birth", "sex", "address", "citizenship_status"],
    priority=50,
)

# Singapore Driving License
DRIVING_LICENSE_SG = DocumentTypeSchema(
    schema_id="driving_license:SG",
    document_type="driving_license",
    document_type_name="Driving License",
    country_code="SG",
    country_name="Singapore",
    extraction_schema=GLINER2Schema(
        fields={
            "full_name": "The license holder's full legal name",
            "license_number": "The driving license number",
            "date_of_birth": "The date of birth",
            "issue_date": "The issue date of the license",
            "expiry_date": "The expiration date",
            "vehicle_class": "The license class (e.g., 3, 3A, 2, 2B)",
            "address": "The residential address",
            "blood_group": "The blood group",
        },
        threshold=0.2,
    ),
    detection_patterns={
        "keywords": [
            "driving licence", "singapore", "class", "traffic police",
            "motor vehicle", "driving license singapore"
        ],
        "labels": ["singapore traffic police", "traffic police"],
    },
    required_fields=["license_number", "full_name"],
    optional_fields=["date_of_birth", "issue_date", "expiry_date", "vehicle_class"],
    priority=50,
)

# Singapore Utility Bill Patterns
UTILITY_BILL_SG_SP = DocumentTypeSchema(
    schema_id="utility_bill:SG:sp",
    document_type="utility_bill",
    document_type_name="Utility Bill",
    country_code="SG",
    country_name="Singapore",
    entity="sp",
    entity_name="SP Services / Singapore Power",
    extraction_schema=GLINER2Schema(
        fields={
            "customer_name": "The customer name",
            "account_number": "The SP account number",
            "service_address": "The service address",
            "billing_period": "The billing period",
            "bill_date": "The bill date",
            "due_date": "The payment due date",
            "amount_due": "The total amount due",
            "electricity_consumption": "The electricity consumption in kWh",
            "gas_consumption": "The gas consumption",
            "water_consumption": "The water consumption in cubic meters",
            "mains_number": "The SP mains number",
        },
        threshold=0.2,
    ),
    detection_patterns={
        "keywords": [
            "sp services", "singapore power", "s&p", "utilities bill",
            "electricity bill", "gas bill", "water bill", "sp group"
        ],
        "labels": ["sp services", "singapore power", "sp group"],
    },
    required_fields=["customer_name", "service_address"],
    optional_fields=["account_number", "bill_date", "amount_due", "billing_period"],
    priority=100,
)


# ============================================================================
# INDIA (IN) SCHEMAS
# ============================================================================

# India PAN Card (most common ID)
ID_CARD_IN_PAN = DocumentTypeSchema(
    schema_id="id_card:IN:pan",
    document_type="id_card",
    document_type_name="PAN Card",
    country_code="IN",
    country_name="India",
    entity="pan",
    entity_name="Permanent Account Number",
    extraction_schema=GLINER2Schema(
        fields={
            "full_name": "The cardholder's full name as shown on PAN card",
            "pan_number": "The Permanent Account Number in format 5 letters + 4 digits + 1 letter (e.g., ABCDE1234F)",
            "father_name": "The father's name",
            "date_of_birth": "The date of birth",
            "pan_card_type": "The type of PAN card (individual, company, etc.)",
        },
        threshold=0.2,
    ),
    detection_patterns={
        "keywords": [
            "permanent account number", "pan card", "income tax department",
            "पैन कार्ड", "आयकर विभाग"
        ],
        "labels": ["income tax department", "income tax department government of india"],
        "patterns": [
            r"[A-Z]{5}[0-9]{4}[A-Z]",  # PAN format
        ],
    },
    required_fields=["pan_number", "full_name"],
    optional_fields=["father_name", "date_of_birth"],
    priority=100,
)

# India Aadhaar Card
ID_CARD_IN_AADHAAR = DocumentTypeSchema(
    schema_id="id_card:IN:aadhaar",
    document_type="id_card",
    document_type_name="Aadhaar Card",
    country_code="IN",
    country_name="India",
    entity="aadhaar",
    entity_name="Unique Identification Authority of India",
    extraction_schema=GLINER2Schema(
        fields={
            "full_name": "The resident's full name",
            "aadhaar_number": "The 12-digit Aadhaar number",
            "date_of_birth": "The date of birth",
            "gender": "The gender (male/female/transgender)",
            "address": "The address",
            "enrollment_date": "The enrollment date",
            "download_date": "The download date for e-Aadhaar",
        },
        threshold=0.2,
    ),
    detection_patterns={
        "keywords": [
            "aadhaar", "uidai", "unique identification authority of india",
            "आधार", "भारतीय विशिष्ट पहचान प्राधिकरण"
        ],
        "labels": [
            "unique identification authority of india",
            "uidai",
            "government of india"
        ],
        "patterns": [
            r"\d{4}\s?\d{4}\s?\d{4}",  # Aadhaar format (with/without spaces)
        ],
    },
    required_fields=["aadhaar_number", "full_name"],
    optional_fields=["date_of_birth", "gender", "address"],
    priority=100,
)

# India Tax Return
TAX_RETURN_IN = DocumentTypeSchema(
    schema_id="tax_return:IN",
    document_type="tax_return",
    document_type_name="Income Tax Return",
    country_code="IN",
    country_name="India",
    extraction_schema=GLINER2Schema(
        fields={
            "taxpayer_name": "The taxpayer's full name",
            "pan_number": "The Permanent Account Number",
            "assessment_year": "The assessment year (e.g., AY 2024-25)",
            "total_income": "The total income",
            "tax_payable": "The tax payable",
            "refund": "The refund amount",
            "itr_form": "The ITR form type (ITR-1, ITR-2, etc.)",
            "filing_date": "The date of filing",
        },
        threshold=0.2,
    ),
    detection_patterns={
        "keywords": [
            "income tax return", "itr", "assessment year", "ay",
            "income tax department", "form itr", "income tax india"
        ],
        "labels": ["income tax department", "incometax", "income tax india"],
    },
    required_fields=["taxpayer_name", "pan_number"],
    optional_fields=["assessment_year", "total_income", "tax_payable", "refund"],
    priority=50,
)

# India Driving License
DRIVING_LICENSE_IN = DocumentTypeSchema(
    schema_id="driving_license:IN",
    document_type="driving_license",
    document_type_name="Driving License",
    country_code="IN",
    country_name="India",
    extraction_schema=GLINER2Schema(
        fields={
            "full_name": "The license holder's full name",
            "license_number": "The driving license number",
            "date_of_birth": "The date of birth",
            "issue_date": "The issue date",
            "expiry_date": "The expiry date",
            "vehicle_class": "The class of vehicle (e.g., MCWG, LMV)",
            "address": "The permanent address",
            "blood_group": "The blood group",
            "issuing_authority": "The issuing RTO",
        },
        threshold=0.2,
    ),
    detection_patterns={
        "keywords": [
            "driving licence", "driver license", "rto", "regional transport office",
            "transport department", "motor vehicles department"
        ],
        "labels": ["rto", "regional transport office", "transport authority"],
    },
    required_fields=["license_number", "full_name"],
    optional_fields=["date_of_birth", "issue_date", "expiry_date", "vehicle_class"],
    priority=50,
)


# ============================================================================
# UNITED STATES (US) SCHEMAS
# ============================================================================

# US Tax Return (IRS)
TAX_RETURN_US_IRS = DocumentTypeSchema(
    schema_id="tax_return:US:irs",
    document_type="tax_return",
    document_type_name="Tax Return",
    country_code="US",
    country_name="United States",
    entity="irs",
    entity_name="Internal Revenue Service",
    extraction_schema=GLINER2Schema(
        fields={
            "taxpayer_name": "The taxpayer's full name",
            "ssn": "The Social Security Number or Tax ID",
            "tax_year": "The tax year (e.g., 2023)",
            "filing_status": "The filing status (single, married filing jointly, etc.)",
            "adjusted_gross_income": "The adjusted gross income (AGI)",
            "total_income": "The total income",
            "tax_liability": "The total tax liability",
            "federal_tax": "The federal income tax",
            "tax_withheld": "The tax already withheld",
            "refund": "The refund amount",
            "amount_owed": "The amount owed",
            "form_type": "The IRS form type (1040, 1040A, 1040EZ, etc.)",
        },
        threshold=0.2,
    ),
    detection_patterns={
        "keywords": [
            "internal revenue service", "irs", "department of the treasury",
            "form 1040", "income tax return", "tax year", "federal income tax",
            "social security number", "w-2", "wage and tax statement"
        ],
        "labels": [
            "internal revenue service",
            "irs",
            "department of the treasury - internal revenue service"
        ],
    },
    required_fields=["taxpayer_name"],
    optional_fields=["ssn", "tax_year", "total_income", "tax_liability", "refund"],
    priority=100,
)

# US Social Security Card
ID_CARD_US_SSN = DocumentTypeSchema(
    schema_id="id_card:US:ssn",
    document_type="id_card",
    document_type_name="Social Security Card",
    country_code="US",
    country_name="United States",
    entity="ssa",
    entity_name="Social Security Administration",
    extraction_schema=GLINER2Schema(
        fields={
            "full_name": "The cardholder's full legal name",
            "ssn": "The Social Security Number in format XXX-XX-XXXX",
            "card_number": "The social security number",
        },
        threshold=0.2,
    ),
    detection_patterns={
        "keywords": [
            "social security", "social security administration", "ssa",
            "social security card", "social security number", "not valid for employment"
        ],
        "labels": ["social security administration", "ssa", "united states government"],
    },
    required_fields=["ssn", "full_name"],
    optional_fields=[],
    priority=100,
)

# US Driver's License
DRIVING_LICENSE_US = DocumentTypeSchema(
    schema_id="driving_license:US",
    document_type="driving_license",
    document_type_name="Driver's License",
    country_code="US",
    country_name="United States",
    extraction_schema=GLINER2Schema(
        fields={
            "full_name": "The license holder's full name",
            "license_number": "The driver's license number",
            "date_of_birth": "The date of birth",
            "issue_date": "The issue date",
            "expiry_date": "The expiration date",
            "vehicle_class": "The license class (e.g., Class C, Class M)",
            "address": "The residential address",
            "sex": "The gender or sex",
            "height": "The height",
            "eye_color": "The eye color",
            "issuing_state": "The issuing state (e.g., CA, TX, NY)",
        },
        threshold=0.2,
    ),
    detection_patterns={
        "keywords": [
            "driver license", "driver's license", "department of motor vehicles",
            "dmv", "operator's license", "class", "state of"
        ],
        "labels": [
            "department of motor vehicles",
            "dmv",
            "driver license"
        ],
    },
    required_fields=["license_number", "full_name"],
    optional_fields=["date_of_birth", "issue_date", "expiry_date", "vehicle_class", "issuing_state"],
    priority=50,
)


# ============================================================================
# MALAYSIA (MY) SCHEMAS
# ============================================================================

# Malaysia ID Card (MyKad)
ID_CARD_MY_MYKAD = DocumentTypeSchema(
    schema_id="id_card:MY:mykad",
    document_type="id_card",
    document_type_name="MyKad",
    country_code="MY",
    country_name="Malaysia",
    entity="mykad",
    entity_name="Malaysian Identity Card",
    extraction_schema=GLINER2Schema(
        fields={
            "full_name": "The cardholder's full name",
            "id_number": "The 12-digit MyKad number (YYMMDD-PB-###G)",
            "date_of_birth": "The date of birth",
            "sex": "The gender or sex",
            "nationality": "The nationality (Warganegara)",
            "address": "The permanent address",
            "religion": "The religion",
            "issue_date": "The date of issue",
        },
        threshold=0.2,
    ),
    detection_patterns={
        "keywords": [
            "mykad", "kad pengenalan", "malaysia", "jabatan pendaftaran negara",
            "jpn", "warganegara", "penduduk"
        ],
        "labels": [
            "jabatan pendaftaran negara",
            "jpn",
            "national registration department"
        ],
        "patterns": [
            r"\d{2}\d{2}\d{2}-\d{2}-\d{4}",  # MyKad format
            r"\d{12}",  # Alternative MyKad format
        ],
    },
    required_fields=["id_number", "full_name"],
    optional_fields=["date_of_birth", "sex", "address"],
    priority=100,
)

# Malaysia Tax Return
TAX_RETURN_MY = DocumentTypeSchema(
    schema_id="tax_return:MY",
    document_type="tax_return",
    document_type_name="Tax Return",
    country_code="MY",
    country_name="Malaysia",
    extraction_schema=GLINER2Schema(
        fields={
            "taxpayer_name": "The taxpayer's full name",
            "tax_reference_number": "The tax reference number",
            "identification_number": "The MyKad number",
            "tax_year": "The year of assessment",
            "total_income": "The total income",
            "tax_payable": "The tax payable",
            "tax_paid": "The tax paid",
        },
        threshold=0.2,
    ),
    detection_patterns={
        "keywords": [
            "hasil cukai", "income tax", "lembaga hasil dalam negeri",
            "lhdn", "malaysia tax", "borang be", "form e"
        ],
        "labels": [
            "lembaga hasil dalam negeri",
            "lhdn",
            "inland revenue board of malaysia"
        ],
    },
    required_fields=["taxpayer_name"],
    optional_fields=["tax_reference_number", "tax_year", "total_income", "tax_payable"],
    priority=50,
)


# ============================================================================
# UNITED ARAB EMIRATES (AE) SCHEMAS
# ============================================================================

# UAE Tax Residency Certificate
ID_CARD_AE_TRC = DocumentTypeSchema(
    schema_id="id_card:AE:trc",
    document_type="id_card",
    document_type_name="Tax Residency Certificate",
    country_code="AE",
    country_name="United Arab Emirates",
    entity="trc",
    entity_name="Ministry of Finance",
    extraction_schema=GLINER2Schema(
        fields={
            "full_name": "The taxpayer's full legal name",
            "application_number": "The application number or reference number of the certificate",
            "certificate_number": "The certificate number (alternative to application_number)",
            "passport_number": "The passport number of the taxpayer",
            "valid_from": "The validity start date of the certificate",
            "valid_until": "The validity end date or expiry date of the certificate",
            "nationality": "The nationality of the taxpayer",
        },
        threshold=0.2,
    ),
    detection_patterns={
        "keywords": [
            "tax residency certificate", "trc", "tax resident",
            "ministry of finance", "uae", "united arab emirates",
        ],
        "labels": ["ministry of finance", "federal tax authority"],
        "patterns": [
            r"tax\s+residenc",
            r"certificate\s+of\s+residence",
        ],
    },
    required_fields=["full_name"],
    optional_fields=["application_number", "certificate_number", "passport_number", "valid_from", "valid_until", "nationality"],
    priority=100,
)

# UAE Tax Residency Certificate - Individual (distinct from Emirates ID)
TAX_RESIDENCY_CERTIFICATE_AE = DocumentTypeSchema(
    schema_id="tax_residency_certificate:AE:trc",
    document_type="tax_residency_certificate",
    document_type_name="Tax Residency Certificate - Individual",
    country_code="AE",
    country_name="United Arab Emirates",
    entity="trc",
    entity_name="Ministry of Finance",
    extraction_schema=GLINER2Schema(
        fields={
            "full_name": "The taxpayer's full legal name",
            "certificate_number": "The certificate/reference number (TRC number)",
            "passport_number": "The passport number of the taxpayer",
            "valid_from": "The validity start date",
            "valid_until": "The validity end date or expiry date",
            "nationality": "The nationality of the taxpayer",
        },
        threshold=0.2,
    ),
    detection_patterns={
        "keywords": [
            "tax residency certificate", "trc", "tax resident",
            "ministry of finance", "uae", "united arab emirates",
            "certificate of residence", "tax domicile",
        ],
        "labels": ["ministry of finance", "federal tax authority"],
        "patterns": [
            r"tax\s+residenc",
            r"certificate\s+of\s+residence",
        ],
    },
    required_fields=["full_name"],
    optional_fields=["certificate_number", "passport_number", "valid_from", "valid_until", "nationality"],
    priority=100,
)

# Generic UAE ID Card (Emirates ID)
ID_CARD_AE = DocumentTypeSchema(
    schema_id="id_card:AE",
    document_type="id_card",
    document_type_name="UAE ID Card",
    country_code="AE",
    country_name="United Arab Emirates",
    extraction_schema=GLINER2Schema(
        fields={
            "full_name": "The cardholder's full name in English or Arabic",
            "id_number": "The Emirates ID number (15 digits)",
            "date_of_birth": "The date of birth",
            "nationality": "The nationality",
            "sex": "The gender",
            "issue_date": "The issue date",
            "expiry_date": "The expiry date",
        },
        threshold=0.2,
    ),
    detection_patterns={
        "keywords": ["emirates id", "uae id", "identity card"],
        "labels": ["federal authority for identity and citizenship"],
        "patterns": [r"\d{3}-\d{4}-\d{7}-\d{1}"],  # Emirates ID format
    },
    required_fields=["id_number", "full_name"],
    optional_fields=["date_of_birth", "nationality", "sex"],
    priority=50,
)


# ============================================================================
# THAILAND (TH) SCHEMAS
# ============================================================================

# Thailand ID Card
ID_CARD_TH = DocumentTypeSchema(
    schema_id="id_card:TH",
    document_type="id_card",
    document_type_name="Thai ID Card",
    country_code="TH",
    country_name="Thailand",
    extraction_schema=GLINER2Schema(
        fields={
            "full_name": "The cardholder's full name in Thai or English",
            "id_number": "The 13-digit Thai identification number",
            "date_of_birth": "The date of birth",
            "sex": "The gender or sex",
            "address": "The address",
            "issue_date": "The issue date",
            "expiry_date": "The expiry date",
        },
        threshold=0.2,
    ),
    detection_patterns={
        "keywords": [
            "บัตรประจำตัวประชาชน", "thai id", "thai identification card",
            "thailand citizen", "สำนักทะเบียนกลาง"
        ],
        "labels": ["ministry of interior", "thai government"],
        "patterns": [
            r"\d{1}-\d{4}-\d{5}-\d{2}-\d{1}",  # Thai ID format
            r"\d{13}",  # Alternative Thai ID format
        ],
    },
    required_fields=["id_number", "full_name"],
    optional_fields=["date_of_birth", "sex", "address"],
    priority=50,
)


# ============================================================================
# REGISTRATION
# ============================================================================

def register_country_schemas() -> None:
    """Register all country-specific schemas in the registry."""
    schemas = [
        # Singapore
        TAX_RETURN_SG_IRAS,
        TAX_RETURN_SG,
        ID_CARD_SG,
        DRIVING_LICENSE_SG,
        UTILITY_BILL_SG_SP,
        # India
        ID_CARD_IN_PAN,
        ID_CARD_IN_AADHAAR,
        TAX_RETURN_IN,
        DRIVING_LICENSE_IN,
        # United States
        TAX_RETURN_US_IRS,
        ID_CARD_US_SSN,
        DRIVING_LICENSE_US,
        # Malaysia
        ID_CARD_MY_MYKAD,
        TAX_RETURN_MY,
        # Thailand
        ID_CARD_TH,
        # United Arab Emirates
        ID_CARD_AE_TRC,
        ID_CARD_AE,
        TAX_RESIDENCY_CERTIFICATE_AE,
    ]

    for schema in schemas:
        SchemaRegistry.register(schema)


# Auto-register on module import
register_country_schemas()


__all__ = [
    # Singapore
    "TAX_RETURN_SG_IRAS",
    "TAX_RETURN_SG",
    "ID_CARD_SG",
    "DRIVING_LICENSE_SG",
    "UTILITY_BILL_SG_SP",
    # India
    "ID_CARD_IN_PAN",
    "ID_CARD_IN_AADHAAR",
    "TAX_RETURN_IN",
    "DRIVING_LICENSE_IN",
    # United States
    "TAX_RETURN_US_IRS",
    "ID_CARD_US_SSN",
    "DRIVING_LICENSE_US",
    # Malaysia
    "ID_CARD_MY_MYKAD",
    "TAX_RETURN_MY",
    # Thailand
    "ID_CARD_TH",
    # United Arab Emirates
    "ID_CARD_AE_TRC",
    "ID_CARD_AE",
    "TAX_RESIDENCY_CERTIFICATE_AE",
    "register_country_schemas",
]
