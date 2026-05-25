-- ============================================================
-- Migration: Create Bank Tables
-- Version: v1.12.0
-- Date: 2025-05-24
-- Purpose: Create core bank database tables for storing bank information
--          migrated from JSON config files
-- ============================================================

-- This migration creates the foundation for bank data management
-- Replaces static config.json with dynamic database tables

-- ============================================================
-- Table 1: banks
-- Core bank information
-- ============================================================

CREATE TABLE IF NOT EXISTS banks (
    id INT AUTO_INCREMENT PRIMARY KEY,
    abbrev VARCHAR(50) NOT NULL UNIQUE,
    full_name VARCHAR(255) NOT NULL,
    legal_name VARCHAR(255) DEFAULT NULL,
    is_active TINYINT(1) NOT NULL DEFAULT 1,
    is_multi_country TINYINT(1) NOT NULL DEFAULT 0,
    website_url TEXT DEFAULT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    created_by VARCHAR(100) NOT NULL DEFAULT 'migration',
    updated_by VARCHAR(100) DEFAULT NULL,
    deleted_at TIMESTAMP NULL DEFAULT NULL,
    INDEX idx_banks_active (is_active),
    INDEX idx_banks_abbrev (abbrev)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
COMMENT='Core bank information and metadata';

-- ============================================================
-- Table 2: bank_country_operations
-- Countries where each bank operates, with SWIFT codes
-- ============================================================

CREATE TABLE IF NOT EXISTS bank_country_operations (
    id INT AUTO_INCREMENT PRIMARY KEY,
    bank_id INT NOT NULL,
    country_code VARCHAR(2) NOT NULL,
    swift_codes JSON NOT NULL,
    local_names JSON DEFAULT NULL,
    is_primary_country TINYINT(1) NOT NULL DEFAULT 0,
    is_active TINYINT(1) NOT NULL DEFAULT 1,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE KEY idx_bank_country_lookup (bank_id, country_code),
    FOREIGN KEY (bank_id) REFERENCES banks(id) ON DELETE CASCADE,
    INDEX idx_bank_country_bank (bank_id),
    INDEX idx_bank_country_code (country_code)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
COMMENT='Bank operations by country with SWIFT codes';

-- ============================================================
-- Table 3: bank_identifiers
-- All identifiers that map to a bank (names, domains, keywords)
-- ============================================================

CREATE TABLE IF NOT EXISTS bank_identifiers (
    id INT AUTO_INCREMENT PRIMARY KEY,
    bank_id INT NOT NULL,
    identifier VARCHAR(500) NOT NULL,
    identifier_type VARCHAR(20) NOT NULL,
    country_code VARCHAR(2) DEFAULT NULL,
    is_validated TINYINT(1) NOT NULL DEFAULT 1,
    usage_count INT NOT NULL DEFAULT 0,
    last_used_at TIMESTAMP NULL DEFAULT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE KEY idx_identifier_unique (identifier, bank_id, identifier_type),
    FOREIGN KEY (bank_id) REFERENCES banks(id) ON DELETE CASCADE,
    INDEX idx_identifier_lookup (identifier, identifier_type),
    INDEX idx_identifier_bank (bank_id),
    INDEX idx_identifier_type (identifier_type),
    CONSTRAINT chk_identifier_type CHECK (identifier_type IN ('full_name', 'alternate_name', 'domain', 'email_domain', 'abbreviation'))
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
COMMENT='Bank identifiers for fuzzy matching and lookup';

-- ============================================================
-- Performance Indexes
-- ============================================================

CREATE INDEX IF NOT EXISTS idx_banks_abbrev_active ON banks(abbrev, is_active);
CREATE INDEX IF NOT EXISTS idx_bank_identifiers_exact ON bank_identifiers(identifier, is_validated);
CREATE INDEX IF NOT EXISTS idx_bank_identifiers_type_validated ON bank_identifiers(identifier_type, is_validated);
CREATE INDEX IF NOT EXISTS idx_bank_operations_composite ON bank_country_operations(bank_id, is_active, country_code);

-- ============================================================
-- VERIFICATION
-- ============================================================

-- Verify banks table structure
-- SELECT COLUMN_NAME, DATA_TYPE, CHARACTER_MAXIMUM_LENGTH, COLUMN_COMMENT
-- FROM INFORMATION_SCHEMA.COLUMNS
-- WHERE TABLE_NAME = 'banks'
-- ORDER BY ORDINAL_POSITION;

-- Verify bank_country_operations table structure
-- SELECT COLUMN_NAME, DATA_TYPE, CHARACTER_MAXIMUM_LENGTH, COLUMN_COMMENT
-- FROM INFORMATION_SCHEMA.COLUMNS
-- WHERE TABLE_NAME = 'bank_country_operations'
-- ORDER BY ORDINAL_POSITION;

-- Verify bank_identifiers table structure
-- SELECT COLUMN_NAME, DATA_TYPE, CHARACTER_MAXIMUM_LENGTH, COLUMN_COMMENT
-- FROM INFORMATION_SCHEMA.COLUMNS
-- WHERE TABLE_NAME = 'bank_identifiers'
-- ORDER BY ORDINAL_POSITION;

-- ============================================================
-- ROLLBACK
-- ============================================================

-- To rollback this migration:
-- DROP TABLE IF EXISTS bank_identifiers;
-- DROP TABLE IF EXISTS bank_country_operations;
-- DROP TABLE IF EXISTS banks;
