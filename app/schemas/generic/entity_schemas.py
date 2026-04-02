"""
Entity-specific document schemas.

This module defines schemas for specific entities (banks, institutions, organizations).
These are the most specific schemas in the hierarchy, combining document type,
country, and entity for highly targeted extraction.

Entity types:
- Banks: DBS, POSB, UOB, OCBC, SBI, Chase, etc.
- Government Agencies: IRAS, CPF, ICA, etc.
- Utility Companies: SP Power, PUB, etc.
- Insurance Companies: AIA, Prudential, Great Eastern, etc.
"""

from typing import Dict, List, Optional, Set

from .base import DocumentTypeSchema, GLINER2Schema
from . import SchemaRegistry


# ============================================================================
# SINGAPORE BANK ENTITIES
# ============================================================================

# DBS Bank Singapore Statement
BANK_STATEMENT_SG_DBS = DocumentTypeSchema(
    schema_id="bank_statement:SG:dbs",
    document_type="bank_statement",
    document_type_name="Bank Statement",
    country_code="SG",
    country_name="Singapore",
    entity="dbs",
    entity_name="DBS Bank",
    extraction_schema=GLINER2Schema(
        fields={
            "bank_name": "The name of the bank (DBS Bank, Development Bank of Singapore)",
            "account_holder_name": "The account holder's full name",
            "account_number": "The DBS account number (usually 10 digits)",
            "cif_number": "The Customer Information File number",
            "address": "The customer's mailing address",
            "currency": "The account currency (SGD, USD, etc.)",
            "statement_date": "The statement date or period",
            "branch": "The branch name or code",
            "account_type": "The account type (savings, current, etc.)",
        },
        threshold=0.2,
    ),
    detection_patterns={
        "keywords": [
            "dbs bank", "development bank of singapore", "dbs",
            "posb-dbs", "dbs bank ltd", "live more, bank less"
        ],
        "labels": ["dbs bank", "development bank of singapore", "dbs"],
        "patterns": [
            r"dbs\s+bank",
            r"development\s+bank\s+of\s+singapore",
        ],
    },
    required_fields=["bank_name", "account_holder_name"],
    optional_fields=["account_number", "currency", "statement_date", "address"],
    priority=100,
)

# POSB Bank Singapore Statement
BANK_STATEMENT_SG_POSB = DocumentTypeSchema(
    schema_id="bank_statement:SG:posb",
    document_type="bank_statement",
    document_type_name="Bank Statement",
    country_code="SG",
    country_name="Singapore",
    entity="posb",
    entity_name="POSB Bank",
    extraction_schema=GLINER2Schema(
        fields={
            "bank_name": "The name of the bank (POSB Bank, Post Office Savings Bank)",
            "account_holder_name": "The account holder's full name",
            "account_number": "The POSB account number",
            "address": "The customer's mailing address",
            "currency": "The account currency",
            "statement_date": "The statement date",
            "passbook_number": "The passbook number",
            "branch": "The branch name",
        },
        threshold=0.2,
    ),
    detection_patterns={
        "keywords": [
            "posb", "posb bank", "post office savings bank",
            "posb everyday", "neighbours first, bankers second"
        ],
        "labels": ["posb bank", "posb"],
        "patterns": [r"posb\s+bank"],
    },
    required_fields=["bank_name", "account_holder_name"],
    optional_fields=["account_number", "currency", "statement_date", "address"],
    priority=100,
)

# UOB Bank Singapore Statement
BANK_STATEMENT_SG_UOB = DocumentTypeSchema(
    schema_id="bank_statement:SG:uob",
    document_type="bank_statement",
    document_type_name="Bank Statement",
    country_code="SG",
    country_name="Singapore",
    entity="uob",
    entity_name="United Overseas Bank",
    extraction_schema=GLINER2Schema(
        fields={
            "bank_name": "The bank name (UOB, United Overseas Bank)",
            "account_holder_name": "The account holder's full name",
            "account_number": "The UOB account number",
            "address": "The customer's address",
            "currency": "The account currency",
            "statement_date": "The statement date",
            "branch": "The branch name",
        },
        threshold=0.2,
    ),
    detection_patterns={
        "keywords": [
            "uob", "united overseas bank", "uob bank",
            "personal banking", "united overseas bank ltd"
        ],
        "labels": ["uob", "united overseas bank"],
    },
    required_fields=["bank_name", "account_holder_name"],
    optional_fields=["account_number", "currency", "statement_date", "address"],
    priority=100,
)

# OCBC Bank Singapore Statement
BANK_STATEMENT_SG_OCBC = DocumentTypeSchema(
    schema_id="bank_statement:SG:ocbc",
    document_type="bank_statement",
    document_type_name="Bank Statement",
    country_code="SG",
    country_name="Singapore",
    entity="ocbc",
    entity_name="OCBC Bank",
    extraction_schema=GLINER2Schema(
        fields={
            "bank_name": "The bank name (OCBC, Oversea-Chinese Banking Corporation)",
            "account_holder_name": "The account holder's full name",
            "account_number": "The OCBC account number",
            "address": "The customer's address",
            "currency": "The account currency",
            "statement_date": "The statement date",
        },
        threshold=0.2,
    ),
    detection_patterns={
        "keywords": [
            "ocbc", "ocbc bank", "oversea-chinese banking corporation",
            "ocbc bank ltd"
        ],
        "labels": ["ocbc", "ocbc bank", "oversea-chinese banking corporation"],
    },
    required_fields=["bank_name", "account_holder_name"],
    optional_fields=["account_number", "currency", "statement_date", "address"],
    priority=100,
)

# Citibank Singapore Statement
BANK_STATEMENT_SG_CITIBANK = DocumentTypeSchema(
    schema_id="bank_statement:SG:citibank",
    document_type="bank_statement",
    document_type_name="Bank Statement",
    country_code="SG",
    country_name="Singapore",
    entity="citibank",
    entity_name="Citibank Singapore",
    extraction_schema=GLINER2Schema(
        fields={
            "bank_name": "The bank name (Citibank, Citi)",
            "account_holder_name": "The account holder's full name",
            "account_number": "The Citibank account number",
            "address": "The customer's address",
            "currency": "The account currency",
            "statement_date": "The statement date",
        },
        threshold=0.2,
    ),
    detection_patterns={
        "keywords": ["citibank", "citi bank", "citi", "citi financial"],
        "labels": ["citibank", "citi"],
    },
    required_fields=["bank_name", "account_holder_name"],
    optional_fields=["account_number", "currency", "statement_date", "address"],
    priority=100,
)


# ============================================================================
# INDIA BANK ENTITIES
# ============================================================================

# SBI India Statement
BANK_STATEMENT_IN_SBI = DocumentTypeSchema(
    schema_id="bank_statement:IN:sbi",
    document_type="bank_statement",
    document_type_name="Bank Statement",
    country_code="IN",
    country_name="India",
    entity="sbi",
    entity_name="State Bank of India",
    extraction_schema=GLINER2Schema(
        fields={
            "bank_name": "The bank name (SBI, State Bank of India)",
            "account_holder_name": "The customer name or account holder name",
            "account_number": "The SBI account number",
            "cif_number": "The Customer Information File number",
            "address": "The customer's address",
            "currency": "The account currency (INR)",
            "statement_date": "The statement date",
            "branch": "The branch name",
        },
        threshold=0.2,
    ),
    detection_patterns={
        "keywords": [
            "state bank of india", "sbi", "sbi online",
            "state bank", "sbi bank", "भारतीय स्टेट बैंक"
        ],
        "labels": ["state bank of india", "sbi"],
        "patterns": [r"state\s+bank\s+of\s+india"],
    },
    required_fields=["bank_name", "account_holder_name"],
    optional_fields=["account_number", "currency", "statement_date", "address", "cif_number"],
    priority=100,
)

# HDFC Bank India Statement
BANK_STATEMENT_IN_HDFC = DocumentTypeSchema(
    schema_id="bank_statement:IN:hdfc",
    document_type="bank_statement",
    document_type_name="Bank Statement",
    country_code="IN",
    country_name="India",
    entity="hdfc",
    entity_name="HDFC Bank",
    extraction_schema=GLINER2Schema(
        fields={
            "bank_name": "The bank name (HDFC Bank)",
            "account_holder_name": "The customer name",
            "account_number": "The HDFC account number",
            "customer_id": "The customer ID",
            "address": "The customer's address",
            "currency": "The account currency",
            "statement_date": "The statement date",
        },
        threshold=0.2,
    ),
    detection_patterns={
        "keywords": [
            "hdfc bank", "hdfc", "hdfc bank ltd",
            "we understand your world"
        ],
        "labels": ["hdfc bank", "hdfc"],
    },
    required_fields=["bank_name", "account_holder_name"],
    optional_fields=["account_number", "currency", "statement_date", "address"],
    priority=100,
)

# ICICI Bank India Statement
BANK_STATEMENT_IN_ICICI = DocumentTypeSchema(
    schema_id="bank_statement:IN:icici",
    document_type="bank_statement",
    document_type_name="Bank Statement",
    country_code="IN",
    country_name="India",
    entity="icici",
    entity_name="ICICI Bank",
    extraction_schema=GLINER2Schema(
        fields={
            "bank_name": "The bank name (ICICI Bank)",
            "account_holder_name": "The customer name",
            "account_number": "The ICICI account number",
            "customer_id": "The customer ID",
            "address": "The customer's address",
            "currency": "The account currency",
            "statement_date": "The statement date",
        },
        threshold=0.2,
    ),
    detection_patterns={
        "keywords": ["icici bank", "icici", "icici bank ltd"],
        "labels": ["icici bank", "icici"],
    },
    required_fields=["bank_name", "account_holder_name"],
    optional_fields=["account_number", "currency", "statement_date", "address"],
    priority=100,
)

# Axis Bank India Statement
BANK_STATEMENT_IN_AXIS = DocumentTypeSchema(
    schema_id="bank_statement:IN:axis",
    document_type="bank_statement",
    document_type_name="Bank Statement",
    country_code="IN",
    country_name="India",
    entity="axis",
    entity_name="Axis Bank",
    extraction_schema=GLINER2Schema(
        fields={
            "bank_name": "The bank name (Axis Bank)",
            "account_holder_name": "The customer name",
            "account_number": "The Axis account number",
            "customer_id": "The customer ID or CRM number",
            "address": "The customer's address",
            "currency": "The account currency",
            "statement_date": "The statement date",
        },
        threshold=0.2,
    ),
    detection_patterns={
        "keywords": ["axis bank", "axis", "axis bank ltd"],
        "labels": ["axis bank", "axis"],
    },
    required_fields=["bank_name", "account_holder_name"],
    optional_fields=["account_number", "currency", "statement_date", "address"],
    priority=100,
)


# ============================================================================
# US BANK ENTITIES
# ============================================================================

# Chase Bank Statement
BANK_STATEMENT_US_CHASE = DocumentTypeSchema(
    schema_id="bank_statement:US:chase",
    document_type="bank_statement",
    document_type_name="Bank Statement",
    country_code="US",
    country_name="United States",
    entity="chase",
    entity_name="JPMorgan Chase Bank",
    extraction_schema=GLINER2Schema(
        fields={
            "bank_name": "The bank name (Chase, JPMorgan Chase)",
            "account_holder_name": "The customer name",
            "account_number": "The account number (last 4 digits shown)",
            "address": "The customer's address",
            "currency": "The account currency (USD)",
            "statement_date": "The statement date",
            "routing_number": "The routing number",
        },
        threshold=0.2,
    ),
    detection_patterns={
        "keywords": [
            "chase", "jpmorgan chase", "chase bank",
            "jpmorgan chase bank, n.a."
        ],
        "labels": ["chase", "jpmorgan chase"],
    },
    required_fields=["bank_name", "account_holder_name"],
    optional_fields=["account_number", "currency", "statement_date", "address"],
    priority=100,
)

# Bank of America Statement
BANK_STATEMENT_US_BOA = DocumentTypeSchema(
    schema_id="bank_statement:US:boa",
    document_type="bank_statement",
    document_type_name="Bank Statement",
    country_code="US",
    country_name="United States",
    entity="boa",
    entity_name="Bank of America",
    extraction_schema=GLINER2Schema(
        fields={
            "bank_name": "The bank name (Bank of America, BOA)",
            "account_holder_name": "The customer name",
            "account_number": "The account number",
            "address": "The customer's address",
            "currency": "The account currency (USD)",
            "statement_date": "The statement date",
            "routing_number": "The routing/transit number",
        },
        threshold=0.2,
    ),
    detection_patterns={
        "keywords": [
            "bank of america", "bank of america, n.a.", "boa",
            "bank of america logo"
        ],
        "labels": ["bank of america", "boa"],
    },
    required_fields=["bank_name", "account_holder_name"],
    optional_fields=["account_number", "currency", "statement_date", "address"],
    priority=100,
)

# Wells Fargo Statement
BANK_STATEMENT_US_WELLS_FARGO = DocumentTypeSchema(
    schema_id="bank_statement:US:wells_fargo",
    document_type="bank_statement",
    document_type_name="Bank Statement",
    country_code="US",
    country_name="United States",
    entity="wells_fargo",
    entity_name="Wells Fargo Bank",
    extraction_schema=GLINER2Schema(
        fields={
            "bank_name": "The bank name (Wells Fargo, Wells Fargo Bank)",
            "account_holder_name": "The customer name",
            "account_number": "The account number",
            "address": "The customer's address",
            "currency": "The account currency (USD)",
            "statement_date": "The statement date",
        },
        threshold=0.2,
    ),
    detection_patterns={
        "keywords": ["wells fargo", "wells fargo bank", "wells fargo logo"],
        "labels": ["wells fargo", "wells fargo bank"],
    },
    required_fields=["bank_name", "account_holder_name"],
    optional_fields=["account_number", "currency", "statement_date", "address"],
    priority=100,
)


# ============================================================================
# SINGAPORE GOVERNMENT ENTITIES
# ============================================================================

# CPF Statement (Central Provident Fund)
TAX_STATEMENT_SG_CPF = DocumentTypeSchema(
    schema_id="tax_statement:SG:cpf",
    document_type="tax_statement",
    document_type_name="CPF Statement",
    country_code="SG",
    country_name="Singapore",
    entity="cpf",
    entity_name="Central Provident Fund Board",
    extraction_schema=GLINER2Schema(
        fields={
            "member_name": "The CPF member's full name",
            "cpf_number": "The CPF account number or NRIC",
            "statement_date": "The statement date",
            "total_contributions": "The total CPF contributions",
            "oa_balance": "The Ordinary Account balance",
            "sa_balance": "The Special Account balance",
            "ma_balance": "The Medisave Account balance",
            "ra_balance": "The Retirement Account balance",
            "employer_name": "The employer's name",
            "contribution_month": "The month of contribution",
        },
        threshold=0.2,
    ),
    detection_patterns={
        "keywords": [
            "cpf", "central provident fund", "cpf board",
            "cpf statement", "cpf contribution", "ordinary account",
            "special account", "medisave account"
        ],
        "labels": ["cpf board", "central provident fund", "cpf"],
    },
    required_fields=["member_name"],
    optional_fields=[
        "cpf_number", "statement_date", "oa_balance", "sa_balance",
        "ma_balance", "total_contributions"
    ],
    priority=100,
)


# ============================================================================
# INSURANCE ENTITIES
# ============================================================================

# AIA Insurance Policy
INSURANCE_POLICY_SG_AIA = DocumentTypeSchema(
    schema_id="insurance_policy:SG:aia",
    document_type="insurance_policy",
    document_type_name="Insurance Policy",
    country_code="SG",
    country_name="Singapore",
    entity="aia",
    entity_name="AIA Singapore",
    extraction_schema=GLINER2Schema(
        fields={
            "policyholder_name": "The policyholder's full name",
            "policy_number": "The policy number",
            "insurance_company": "AIA or American International Assurance",
            "policy_type": "The type of insurance (life, health, investment)",
            "issue_date": "The policy issue date",
            "expiry_date": "The policy expiry date",
            "coverage_amount": "The sum assured or coverage amount",
            "premium_amount": "The premium amount",
            "agent_name": "The agent's name",
        },
        threshold=0.2,
    ),
    detection_patterns={
        "keywords": ["aia", "american international assurance", "aia singapore"],
        "labels": ["aia", "aia insurance", "american international assurance"],
    },
    required_fields=["policyholder_name", "policy_number", "insurance_company"],
    optional_fields=["policy_type", "issue_date", "coverage_amount", "premium_amount"],
    priority=100,
)

# Prudential Insurance Policy
INSURANCE_POLICY_SG_PRUDENTIAL = DocumentTypeSchema(
    schema_id="insurance_policy:SG:prudential",
    document_type="insurance_policy",
    document_type_name="Insurance Policy",
    country_code="SG",
    country_name="Singapore",
    entity="prudential",
    entity_name="Prudential Singapore",
    extraction_schema=GLINER2Schema(
        fields={
            "policyholder_name": "The policyholder's full name",
            "policy_number": "The policy number",
            "insurance_company": "Prudential Assurance",
            "policy_type": "The type of insurance",
            "issue_date": "The policy issue date",
            "coverage_amount": "The sum assured",
            "premium_amount": "The premium amount",
        },
        threshold=0.2,
    ),
    detection_patterns={
        "keywords": ["prudential", "prudential assurance", "prudential singapore"],
        "labels": ["prudential", "prudential assurance"],
    },
    required_fields=["policyholder_name", "policy_number", "insurance_company"],
    optional_fields=["policy_type", "issue_date", "coverage_amount", "premium_amount"],
    priority=100,
)

# Great Eastern Insurance Policy
INSURANCE_POLICY_SG_GREAT_EASTERN = DocumentTypeSchema(
    schema_id="insurance_policy:SG:great_eastern",
    document_type="insurance_policy",
    document_type_name="Insurance Policy",
    country_code="SG",
    country_name="Singapore",
    entity="great_eastern",
    entity_name="Great Eastern Singapore",
    extraction_schema=GLINER2Schema(
        fields={
            "policyholder_name": "The policyholder's full name",
            "policy_number": "The policy number",
            "insurance_company": "Great Eastern Life",
            "policy_type": "The type of insurance",
            "issue_date": "The policy issue date",
            "coverage_amount": "The sum assured",
            "premium_amount": "The premium amount",
        },
        threshold=0.2,
    ),
    detection_patterns={
        "keywords": ["great eastern", "great eastern life", "great eastern insurance"],
        "labels": ["great eastern", "great eastern life"],
    },
    required_fields=["policyholder_name", "policy_number", "insurance_company"],
    optional_fields=["policy_type", "issue_date", "coverage_amount", "premium_amount"],
    priority=100,
)


# ============================================================================
# UTILITY ENTITIES (Singapore)
# ============================================================================

# PUB Singapore Water Bill
UTILITY_BILL_SG_PUB = DocumentTypeSchema(
    schema_id="utility_bill:SG:pub",
    document_type="utility_bill",
    document_type_name="Water Bill",
    country_code="SG",
    country_name="Singapore",
    entity="pub",
    entity_name="PUB Singapore",
    extraction_schema=GLINER2Schema(
        fields={
            "customer_name": "The customer name",
            "account_number": "The PUB account number",
            "service_address": "The service address",
            "billing_period": "The billing period",
            "bill_date": "The bill date",
            "due_date": "The payment due date",
            "amount_due": "The total amount due",
            "water_consumption": "The water consumption in cubic meters",
        },
        threshold=0.2,
    ),
    detection_patterns={
        "keywords": ["pub", "pub singapore", "water bill", "water conservation tax"],
        "labels": ["pub", "pub singapore", "national water agency"],
    },
    required_fields=["customer_name", "service_address"],
    optional_fields=["account_number", "bill_date", "amount_due", "water_consumption"],
    priority=100,
)

# Singtel Bill
UTILITY_BILL_SG_SINGTEL = DocumentTypeSchema(
    schema_id="utility_bill:SG:singtel",
    document_type="utility_bill",
    document_type_name="Telecommunications Bill",
    country_code="SG",
    country_name="Singapore",
    entity="singtel",
    entity_name="Singtel Singapore",
    extraction_schema=GLINER2Schema(
        fields={
            "customer_name": "The customer name",
            "account_number": "The Singtel account number",
            "service_address": "The service address",
            "billing_period": "The billing period",
            "bill_date": "The bill date",
            "amount_due": "The total amount due",
            "mobile_number": "The mobile number",
            "plan_type": "The plan type",
        },
        threshold=0.2,
    ),
    detection_patterns={
        "keywords": ["singtel", "singapore telecommunications", "mobile bill"],
        "labels": ["singtel", "singtel singapore"],
    },
    required_fields=["customer_name", "service_address"],
    optional_fields=["account_number", "bill_date", "amount_due"],
    priority=100,
)

# StarHub Bill
UTILITY_BILL_SG_STARHUB = DocumentTypeSchema(
    schema_id="utility_bill:SG:starhub",
    document_type="utility_bill",
    document_type_name="Telecommunications Bill",
    country_code="SG",
    country_name="Singapore",
    entity="starhub",
    entity_name="StarHub Singapore",
    extraction_schema=GLINER2Schema(
        fields={
            "customer_name": "The customer name",
            "account_number": "The StarHub account number",
            "service_address": "The service address",
            "billing_period": "The billing period",
            "bill_date": "The bill date",
            "amount_due": "The total amount due",
        },
        threshold=0.2,
    ),
    detection_patterns={
        "keywords": ["starhub", "starhub singapore", "cable tv", "mobile bill"],
        "labels": ["starhub", "starhub limited"],
    },
    required_fields=["customer_name", "service_address"],
    optional_fields=["account_number", "bill_date", "amount_due"],
    priority=100,
)


# ============================================================================
# REGISTRATION
# ============================================================================

def register_entity_schemas() -> None:
    """Register all entity-specific schemas in the registry."""
    schemas = [
        # Singapore Banks
        BANK_STATEMENT_SG_DBS,
        BANK_STATEMENT_SG_POSB,
        BANK_STATEMENT_SG_UOB,
        BANK_STATEMENT_SG_OCBC,
        BANK_STATEMENT_SG_CITIBANK,
        # India Banks
        BANK_STATEMENT_IN_SBI,
        BANK_STATEMENT_IN_HDFC,
        BANK_STATEMENT_IN_ICICI,
        BANK_STATEMENT_IN_AXIS,
        # US Banks
        BANK_STATEMENT_US_CHASE,
        BANK_STATEMENT_US_BOA,
        BANK_STATEMENT_US_WELLS_FARGO,
        # Singapore Government
        TAX_STATEMENT_SG_CPF,
        # Insurance
        INSURANCE_POLICY_SG_AIA,
        INSURANCE_POLICY_SG_PRUDENTIAL,
        INSURANCE_POLICY_SG_GREAT_EASTERN,
        # Utilities (Singapore)
        UTILITY_BILL_SG_PUB,
        UTILITY_BILL_SG_SINGTEL,
        UTILITY_BILL_SG_STARHUB,
    ]

    for schema in schemas:
        SchemaRegistry.register(schema)


# Auto-register on module import
register_entity_schemas()


__all__ = [
    # Singapore Banks
    "BANK_STATEMENT_SG_DBS",
    "BANK_STATEMENT_SG_POSB",
    "BANK_STATEMENT_SG_UOB",
    "BANK_STATEMENT_SG_OCBC",
    "BANK_STATEMENT_SG_CITIBANK",
    # India Banks
    "BANK_STATEMENT_IN_SBI",
    "BANK_STATEMENT_IN_HDFC",
    "BANK_STATEMENT_IN_ICICI",
    "BANK_STATEMENT_IN_AXIS",
    # US Banks
    "BANK_STATEMENT_US_CHASE",
    "BANK_STATEMENT_US_BOA",
    "BANK_STATEMENT_US_WELLS_FARGO",
    # Singapore Government
    "TAX_STATEMENT_SG_CPF",
    # Insurance
    "INSURANCE_POLICY_SG_AIA",
    "INSURANCE_POLICY_SG_PRUDENTIAL",
    "INSURANCE_POLICY_SG_GREAT_EASTERN",
    # Utilities
    "UTILITY_BILL_SG_PUB",
    "UTILITY_BILL_SG_SINGTEL",
    "UTILITY_BILL_SG_STARHUB",
    "register_entity_schemas",
]
