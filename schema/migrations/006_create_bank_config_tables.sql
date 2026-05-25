-- ============================================================
-- Migration: Create Bank Config Tables
-- Version: v1.12.0
-- Date: 2025-05-24
-- Purpose: Create configuration tables for bank statement extraction
--          Stores currencies, address formats, field labels, state mappings
-- ============================================================

-- This migration creates supporting tables for bank statement processing
-- Replaces static configuration sections in config.json

-- ============================================================
-- Table 1: currencies
-- Stores currency definitions with account number length requirements
-- ============================================================

CREATE TABLE IF NOT EXISTS currencies (
    currency_code VARCHAR(3) PRIMARY KEY COMMENT 'ISO 4217 currency code',
    currency_name VARCHAR(100) NOT NULL,
    country_code VARCHAR(2) NOT NULL,
    account_number_min INT DEFAULT 8,
    account_number_max INT DEFAULT 20,
    is_active TINYINT(1) DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_currency_country (country_code),
    INDEX idx_currency_active (is_active)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
COMMENT='Currency definitions with account number validation rules';

-- ============================================================
-- Table 2: currency_name_map
-- Maps display names to ISO currency codes for extraction
-- ============================================================

CREATE TABLE IF NOT EXISTS currency_name_map (
    id INT AUTO_INCREMENT PRIMARY KEY,
    display_name VARCHAR(100) NOT NULL,
    currency_code VARCHAR(3) NOT NULL,
    is_common TINYINT(1) DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY idx_display_name (display_name),
    INDEX idx_currency_code (currency_code),
    FOREIGN KEY (currency_code) REFERENCES currencies(currency_code) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
COMMENT='Maps various currency display names to ISO codes';

-- ============================================================
-- Table 3: address_formats
-- Postal code patterns and validation rules by country
-- ============================================================

CREATE TABLE IF NOT EXISTS address_formats (
    country_code VARCHAR(2) PRIMARY KEY,
    postal_code_pattern VARCHAR(50),
    postal_code_length INT,
    is_required TINYINT(1) DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
COMMENT='Address format validation rules by country';

-- ============================================================
-- Table 4: state_to_country_map
-- Maps state/province names to country codes
-- ============================================================

CREATE TABLE IF NOT EXISTS state_to_country_map (
    id INT AUTO_INCREMENT PRIMARY KEY,
    state_name VARCHAR(100) NOT NULL,
    country_code VARCHAR(2) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY idx_state_name (state_name),
    INDEX idx_country_code (country_code)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
COMMENT='Maps state names to ISO country codes for inference';

-- ============================================================
-- Table 5: field_labels
-- Unified table for all extraction field labels
-- ============================================================

CREATE TABLE IF NOT EXISTS field_labels (
    id INT AUTO_INCREMENT PRIMARY KEY,
    label_text VARCHAR(100) NOT NULL,
    label_type VARCHAR(30) NOT NULL,
    priority INT DEFAULT 0,
    is_active TINYINT(1) DEFAULT 1,
    country_code VARCHAR(2) DEFAULT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY idx_label_type_text (label_type, label_text),
    INDEX idx_label_type (label_type),
    INDEX idx_label_active (is_active),
    CONSTRAINT chk_label_type CHECK (label_type IN (
        'account_number', 'account_holder_name', 'currency', 'iban',
        'statement_date', 'opening_balance', 'closing_balance', 'address'
    ))
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
COMMENT='Field labels for bank statement extraction';

-- ============================================================
-- Table 6: address_extraction_exceptions
-- Banks where address field should be skipped (bank address vs customer)
-- ============================================================

CREATE TABLE IF NOT EXISTS address_extraction_exceptions (
    bank_abbrev VARCHAR(50) PRIMARY KEY,
    reason TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (bank_abbrev) REFERENCES banks(abbrev) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
COMMENT='Banks where address label refers to bank address, not customer';

-- ============================================================
-- PERFORMANCE INDEXES FOR EXISTING BANK TABLES
-- ============================================================

-- Add composite indexes for optimized lookups
-- These indexes significantly improve lookup_by_name, detect_bank_in_text, and lookup_by_domain

CREATE INDEX IF NOT EXISTS idx_banks_abbrev_active ON banks(abbrev, is_active);
CREATE INDEX IF NOT EXISTS idx_bank_identifiers_exact ON bank_identifiers(identifier, is_validated);
CREATE INDEX IF NOT EXISTS idx_bank_identifiers_type_validated ON bank_identifiers(identifier_type, is_validated);
CREATE INDEX IF NOT EXISTS idx_bank_operations_composite ON bank_country_operations(bank_id, is_active, country_code);
CREATE INDEX IF NOT EXISTS idx_bank_identifiers_length ON bank_identifiers(identifier(100), is_validated);

-- Add covering index for bank identification queries
CREATE INDEX IF NOT EXISTS idx_bank_identifiers_lookup_covering ON bank_identifiers(identifier, identifier_type, is_validated, bank_id);

-- Add index for country-based filtering
CREATE INDEX IF NOT EXISTS idx_bank_identifiers_country ON bank_identifiers(country_code, is_validated);

-- ============================================================
-- VERIFICATION
-- ============================================================

-- Verify currencies table structure
-- SELECT COLUMN_NAME, DATA_TYPE, CHARACTER_MAXIMUM_LENGTH, COLUMN_COMMENT
-- FROM INFORMATION_SCHEMA.COLUMNS
-- WHERE TABLE_NAME = 'currencies'
-- ORDER BY ORDINAL_POSITION;

-- Verify field_labels table structure
-- SELECT COLUMN_NAME, DATA_TYPE, CHARACTER_MAXIMUM_LENGTH, COLUMN_COMMENT
-- FROM INFORMATION_SCHEMA.COLUMNS
-- WHERE TABLE_NAME = 'field_labels'
-- ORDER BY ORDINAL_POSITION;

-- Check if indexes were created successfully
-- SHOW INDEX FROM bank_identifiers WHERE Key_name LIKE 'idx_bank_%';

-- ============================================================
-- ROLLBACK
-- ============================================================

-- To rollback this migration:
-- DROP TABLE IF EXISTS address_extraction_exceptions;
-- DROP TABLE IF EXISTS field_labels;
-- DROP TABLE IF EXISTS state_to_country_map;
-- DROP TABLE IF EXISTS address_formats;
-- DROP TABLE IF EXISTS currency_name_map;
-- DROP TABLE IF EXISTS currencies;
