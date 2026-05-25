-- ============================================================
-- Migration: Migrate Bank Config Data
-- Version: v1.12.0
-- Date: 2025-05-24
-- Purpose: Migrate configuration data from config.json to database
--          Loads currencies, address formats, field labels, state mappings
-- ============================================================

-- This migration populates the configuration tables with data from config.json
-- Must be run AFTER: 006_create_bank_config_tables.sql
-- Bank Configuration Data Migration
-- Migrates remaining config.json data to database tables
-- Run this after creating the tables with create_bank_config_tables.sql

-- ============================================================
-- INSERT DATA INTO CURRENCIES TABLE
-- ============================================================

INSERT INTO currencies (currency_code, currency_name, country_code, account_number_min, account_number_max) VALUES
('SGD', 'Singapore Dollar', 'SG', 9, 12),
('INR', 'Indian Rupee', 'IN', 9, 18),
('MYR', 'Malaysian Ringgit', 'MY', 10, 14),
('THB', 'Thai Baht', 'TH', 10, 12),
('MMK', 'Myanmar Kyat', 'MM', 10, 16),
('IDR', 'Indonesian Rupiah', 'ID', 10, 16),
('USD', 'US Dollar', 'US', 8, 12),
('GBP', 'British Pound', 'GB', 8, 8),
('HKD', 'Hong Kong Dollar', 'HK', 8, 12),
('AED', 'UAE Dirham', 'AE', 8, 20),
('EUR', 'Euro', 'FR', 8, 20),
('SAR', 'Saudi Riyal', 'SA', 8, 20)
ON DUPLICATE KEY UPDATE currency_name=VALUES(currency_name), country_code=VALUES(country_code);

-- ============================================================
-- INSERT DATA INTO CURRENCY_NAME_MAP TABLE
-- ============================================================

INSERT INTO currency_name_map (display_name, currency_code, is_common) VALUES
('UAE DIRHAM', 'AED', 1),
('DIRHAM', 'AED', 1),
('EMIRATES DIRHAM', 'AED', 0),
('US DOLLAR', 'USD', 1),
('DOLLAR', 'USD', 1),
('EURO', 'EUR', 0),
('BRITISH POUND', 'GBP', 1),
('POUND STERLING', 'GBP', 0),
('INDIAN RUPEE', 'INR', 1),
('RUPEE', 'INR', 1),
('RS', 'INR', 0),
('SINGAPORE DOLLAR', 'SGD', 1),
('MALAYSIAN RINGGIT', 'MYR', 1),
('THAI BAHT', 'THB', 1),
('MYANMAR KYAT', 'MMK', 1),
('KYAT', 'MMK', 0),
('INDONESIAN RUPIAH', 'IDR', 1),
('HONG KONG DOLLAR', 'HKD', 1),
('SAUDI RIYAL', 'SAR', 0),
('RIYAL', 'SAR', 0)
ON DUPLICATE KEY UPDATE currency_code=VALUES(currency_code);

-- ============================================================
-- INSERT DATA INTO ADDRESS_FORMATS TABLE
-- ============================================================

INSERT INTO address_formats (country_code, postal_code_pattern, postal_code_length, is_required) VALUES
('SG', '\\d{6}', 6, 0),
('IN', '\\d{6}', 6, 1),
('MY', '\\d{5}', 5, 1),
('TH', '\\d{5}', 5, 1),
('MM', '\\d{5}', 5, 1),
('AE', '\\d{5}', 5, 0)
ON DUPLICATE KEY UPDATE postal_code_pattern=VALUES(postal_code_pattern), postal_code_length=VALUES(postal_code_length);

-- ============================================================
-- INSERT DATA INTO STATE_TO_COUNTRY_MAP TABLE
-- ============================================================

INSERT INTO state_to_country_map (state_name, country_code) VALUES
('Andhra Pradesh', 'IN'),
('Tamil Nadu', 'IN'),
('Tamilnadu', 'IN'),
('Uttarakhand', 'IN'),
('Uttaranchal', 'IN'),
('Odisha', 'IN'),
('Orissa', 'IN'),
('Singapore', 'SG'),
('Dubai', 'AE'),
('Abu Dhabi', 'AE'),
('Sharjah', 'AE'),
('Ajman', 'AE'),
('Umm Al Quwain', 'AE'),
('Ras Al Khaimah', 'AE'),
('Fujairah', 'AE'),
('Al Ain', 'AE')
ON DUPLICATE KEY UPDATE country_code=VALUES(country_code);

-- ============================================================
-- INSERT DATA INTO FIELD_LABELS TABLE
-- ============================================================

-- Account number labels (35 labels)
INSERT INTO field_labels (label_text, label_type, priority, is_active) VALUES
('ACCOUNT NUMBER', 'account_number', 0, 1),
('ACCOUNT NUMBER:', 'account_number', 0, 1),
('ACCOUNT NO', 'account_number', 0, 1),
('ACCOUNT NO.', 'account_number', 0, 1),
('ACCOUNT NUM', 'account_number', 0, 1),
('A/C NUMBER', 'account_number', 0, 1),
('A/C NO', 'account_number', 0, 1),
('A/C NO.', 'account_number', 0, 1),
('AC NUMBER', 'account_number', 0, 1),
('AC NO', 'account_number', 0, 1),
('AC NO.', 'account_number', 0, 1),
('ACCOUNT NO:', 'account_number', 0, 1),
('A/C NO:', 'account_number', 0, 1),
('ACCOUNTNUMBER', 'account_number', 0, 1),
('ACCOUNTNO', 'account_number', 0, 1),
('ACCOUNTNUM', 'account_number', 0, 1),
('ACCOUNTNO.', 'account_number', 0, 1),
('AC/NO', 'account_number', 0, 1),
('A / C NO', 'account_number', 0, 1),
('A / C NUMBER', 'account_number', 0, 1),
('ACC NO', 'account_number', 0, 1),
('ACC NO.', 'account_number', 0, 1),
('ACC NUMBER', 'account_number', 0, 1),
('ACCNT NO', 'account_number', 0, 1),
('ACCT NO', 'account_number', 0, 1),
('ACCT NO.', 'account_number', 0, 1),
('ACCT NUMBER', 'account_number', 0, 1),
('ACCOUNT REF', 'account_number', 0, 1),
('ACCOUNT REFERENCE', 'account_number', 0, 1),
('REF NO', 'account_number', 0, 1),
('REFERENCE NO', 'account_number', 0, 1),
('CARD NUMBER', 'account_number', 0, 1),
('PRIMARY CARD NUMBER', 'account_number', 0, 1),
('CARD NO', 'account_number', 0, 1)
ON DUPLICATE KEY UPDATE priority=VALUES(priority), is_active=VALUES(is_active);

-- Account holder name labels (23 labels)
INSERT INTO field_labels (label_text, label_type, priority, is_active) VALUES
('CUSTOMER NAME', 'account_holder_name', 0, 1),
('NAME OF ACCOUNT HOLDER', 'account_holder_name', 0, 1),
('A/C HOLDER NAME', 'account_holder_name', 0, 1),
('ACCOUNT HOLDER NAME:', 'account_holder_name', 0, 1),
('CUSTOMER NAME:', 'account_holder_name', 0, 1),
('A/C HOLDER NAME:', 'account_holder_name', 0, 1),
('PRIMARY ACCOUNT HOLDER NAME', 'account_holder_name', 0, 1),
('CARDHOLDER NAME', 'account_holder_name', 0, 1),
('CARD TYPE:', 'account_holder_name', 0, 1),
('CARD NUMBER:', 'account_holder_name', 0, 1),
('PRIMARY CARD NUMBER:', 'account_holder_name', 0, 1),
('NAME:', 'account_holder_name', 0, 1),
('CARD MEMBER', 'account_holder_name', 0, 1),
('PRIMARY CARDHOLDER', 'account_holder_name', 0, 1),
('MR.', 'account_holder_name', 0, 1),
('MRS.', 'account_holder_name', 0, 1),
('MS.', 'account_holder_name', 0, 1),
('DR.', 'account_holder_name', 0, 1),
('MISS.', 'account_holder_name', 0, 1),
('PROF.', 'account_holder_name', 0, 1),
('REV.', 'account_holder_name', 0, 1),
('HON.', 'account_holder_name', 0, 1)
ON DUPLICATE KEY UPDATE priority=VALUES(priority), is_active=VALUES(is_active);

-- Currency labels (5 labels)
INSERT INTO field_labels (label_text, label_type, priority, is_active) VALUES
('CURRENCY', 'currency', 0, 1),
('CURRENCY NAME', 'currency', 0, 1),
('Amount (', 'currency', 0, 1),
('CURRENCY CODE', 'currency', 0, 1),
('ACCOUNT DETAILS -', 'currency', 0, 1)
ON DUPLICATE KEY UPDATE priority=VALUES(priority), is_active=VALUES(is_active);

-- IBAN labels (3 labels)
INSERT INTO field_labels (label_text, label_type, priority, is_active) VALUES
('IBAN', 'iban', 0, 1),
('INTERNATIONAL BANK ACCOUNT NUMBER', 'iban', 0, 1),
('INTL BANK ACCOUNT NO', 'iban', 0, 1)
ON DUPLICATE KEY UPDATE priority=VALUES(priority), is_active=VALUES(is_active);

-- Statement date labels (9 labels)
INSERT INTO field_labels (label_text, label_type, priority, is_active) VALUES
('STATEMENT DATE', 'statement_date', 0, 1),
('STATEMENT DATE:', 'statement_date', 0, 1),
('DATE OF STATEMENT', 'statement_date', 0, 1),
('STATEMENT PERIOD', 'statement_date', 0, 1),
('STATEMENT PERIOD:', 'statement_date', 0, 1),
('PERIOD', 'statement_date', 0, 1),
('PERIOD:', 'statement_date', 0, 1),
('AS OF', 'statement_date', 0, 1),
('AS OF:', 'statement_date', 0, 1)
ON DUPLICATE KEY UPDATE priority=VALUES(priority), is_active=VALUES(is_active);

-- Opening balance labels (9 labels)
INSERT INTO field_labels (label_text, label_type, priority, is_active) VALUES
('OPENING BALANCE', 'opening_balance', 0, 1),
('OPENING BALANCE:', 'opening_balance', 0, 1),
('OPENING BAL:', 'opening_balance', 0, 1),
('PREVIOUS BALANCE', 'opening_balance', 0, 1),
('PREVIOUS BALANCE:', 'opening_balance', 0, 1),
('BRING FORWARD', 'opening_balance', 0, 1),
('BRING FORWARD:', 'opening_balance', 0, 1),
('BF', 'opening_balance', 0, 1),
('B/F', 'opening_balance', 0, 1)
ON DUPLICATE KEY UPDATE priority=VALUES(priority), is_active=VALUES(is_active);

-- Closing balance labels (11 labels)
INSERT INTO field_labels (label_text, label_type, priority, is_active) VALUES
('CLOSING BALANCE', 'closing_balance', 0, 1),
('CLOSING BALANCE:', 'closing_balance', 0, 1),
('CLOSING BAL:', 'closing_balance', 0, 1),
('CURRENT BALANCE', 'closing_balance', 0, 1),
('CURRENT BALANCE:', 'closing_balance', 0, 1),
('ENDING BALANCE', 'closing_balance', 0, 1),
('ENDING BALANCE:', 'closing_balance', 0, 1),
('CARRY FORWARD', 'closing_balance', 0, 1),
('CARRY FORWARD:', 'closing_balance', 0, 1),
('CF', 'closing_balance', 0, 1),
('C/F', 'closing_balance', 0, 1)
ON DUPLICATE KEY UPDATE priority=VALUES(priority), is_active=VALUES(is_active);

-- ============================================================
-- INSERT DATA INTO ADDRESS_EXTRACTION_EXCEPTIONS TABLE
-- ============================================================

INSERT INTO address_extraction_exceptions (bank_abbrev, reason) VALUES
('HDFC', 'Address label refers to bank address, not customer')
ON DUPLICATE KEY UPDATE reason=VALUES(reason);

-- ============================================================
-- VERIFICATION QUERIES
-- ============================================================

-- Run these to verify the migration was successful:
-- SELECT 'currencies' AS table_name, COUNT(*) AS row_count FROM currencies
-- UNION ALL
-- SELECT 'currency_name_map', COUNT(*) FROM currency_name_map
-- UNION ALL
-- SELECT 'address_formats', COUNT(*) FROM address_formats
-- UNION ALL
-- SELECT 'state_to_country_map', COUNT(*) FROM state_to_country_map
-- UNION ALL
-- SELECT 'field_labels', COUNT(*) FROM field_labels
-- UNION ALL
-- SELECT 'address_extraction_exceptions', COUNT(*) FROM address_extraction_exceptions;
