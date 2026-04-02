-- ===============================================
-- Rollback: v1.7.0 to v1.3.0
-- Description: Rollback migration from schema v1.7.0 to v1.3.0
-- Date: 2026-03-11
-- ===============================================
--
-- WARNING: This rollback will:
-- 1. Delete the user_keys_pending table (data will be lost)
-- 2. Add back device_identifier columns (without data restoration)
-- 3. Add back hash columns (without data restoration)
-- 4. Make country_code NOT NULL again (may fail if NULL values exist)
--
-- IMPORTANT: Only run this if you have a database backup!
-- Ideally, restore from backup instead of running this rollback.
--
-- ===============================================
-- Section 1: Remove user_keys_pending table (v1.7.0)
-- ===============================================

DROP TABLE IF EXISTS user_keys_pending;

-- ===============================================
-- Section 2: Revert country_code to NOT NULL (v1.4.0)
-- ===============================================

-- WARNING: This will fail if there are any NULL country_code values
-- Check first:
SELECT COUNT(*) AS null_country_codes
FROM user_keys
WHERE country_code IS NULL;

-- If count is 0, proceed with making it NOT NULL
-- If count > 0, update those records first or abort rollback
ALTER TABLE user_keys
MODIFY COLUMN country_code VARCHAR(10) NOT NULL;

-- ===============================================
-- Section 3: Add back device_identifier (v1.6.0)
-- ===============================================

-- Add device_identifier back to user_keys (without data)
ALTER TABLE user_keys
ADD COLUMN device_identifier VARCHAR(255) NOT NULL UNIQUE AFTER id;

-- Add index for device_identifier
CREATE INDEX idx_device_identifier ON user_keys(device_identifier);

-- Add device_identifier back to otp table (without data)
ALTER TABLE otp
ADD COLUMN device_identifier VARCHAR(255) NULL AFTER is_verified;

-- ===============================================
-- Section 4: Add back hash columns (v1.5.0)
-- ===============================================

-- Add passport_hash back to user_identity_index (without data)
ALTER TABLE user_identity_index
ADD COLUMN passport_hash VARCHAR(64) NULL UNIQUE AFTER pii_data_encrypted
COMMENT 'SHA-256 hash of passport_country:passport_number for uniqueness constraint (without storing plaintext passport number)';

-- Add document_hash back to document_submissions (without data)
ALTER TABLE document_submissions
ADD COLUMN document_hash VARCHAR(64) NOT NULL UNIQUE AFTER client_public_key;

-- ===============================================
-- Section 5: Remove new columns from otp table (v1.4.0)
-- ===============================================

-- Remove country_code from otp table
ALTER TABLE otp DROP COLUMN IF EXISTS country_code;

-- Remove encrypted_secret_share from otp table
ALTER TABLE otp DROP COLUMN IF EXISTS encrypted_secret_share;

-- ===============================================
-- Section 6: Verification queries
-- ===============================================

SELECT 'Rollback to v1.3.0 completed' as status;

-- Check user_keys table structure
DESCRIBE user_keys;

-- Check otp table structure
DESCRIBE otp;

-- Check user_identity_index table structure
DESCRIBE user_identity_index;

-- Check document_submissions table structure
DESCRIBE document_submissions;

-- Verify device_identifier exists
SELECT COUNT(*) AS device_identifier_in_user_keys
FROM information_schema.COLUMNS
WHERE TABLE_SCHEMA = DATABASE()
  AND COLUMN_NAME = 'device_identifier'
  AND TABLE_NAME = 'user_keys';
-- Expected result: 1

SELECT COUNT(*) AS device_identifier_in_otp
FROM information_schema.COLUMNS
WHERE TABLE_SCHEMA = DATABASE()
  AND COLUMN_NAME = 'device_identifier'
  AND TABLE_NAME = 'otp';
-- Expected result: 1

-- Verify passport_hash exists
SELECT COUNT(*) AS passport_hash_in_user_identity_index
FROM information_schema.COLUMNS
WHERE TABLE_SCHEMA = DATABASE()
  AND COLUMN_NAME = 'passport_hash'
  AND TABLE_NAME = 'user_identity_index';
-- Expected result: 1

-- Verify document_hash exists
SELECT COUNT(*) AS document_hash_in_document_submissions
FROM information_schema.COLUMNS
WHERE TABLE_SCHEMA = DATABASE()
  AND COLUMN_NAME = 'document_hash'
  AND TABLE_NAME = 'document_submissions';
-- Expected result: 1

-- Verify removed columns don't exist
SELECT COUNT(*) AS country_code_in_otp
FROM information_schema.COLUMNS
WHERE TABLE_SCHEMA = DATABASE()
  AND COLUMN_NAME = 'country_code'
  AND TABLE_NAME = 'otp';
-- Expected result: 0

SELECT COUNT(*) AS encrypted_secret_share_in_otp
FROM information_schema.COLUMNS
WHERE TABLE_SCHEMA = DATABASE()
  AND COLUMN_NAME = 'encrypted_secret_share'
  AND TABLE_NAME = 'otp';
-- Expected result: 0

SELECT COUNT(*) AS user_keys_pending_exists
FROM information_schema.TABLES
WHERE TABLE_SCHEMA = DATABASE()
  AND TABLE_NAME = 'user_keys_pending';
-- Expected result: 0

-- ===============================================
-- Notes
-- ===============================================
--
-- IMPORTANT: After rollback:
-- 1. device_identifier columns are empty (data was lost during drop)
-- 2. passport_hash and document_hash columns are empty (data was lost during drop)
-- 3. You will need to rebuild these columns from application data
-- 4. Consider restoring from backup instead of using this rollback
--
-- For a clean rollback, restore from the backup taken before migration:
-- mysql -u root -p im_osint < backup_before_v1.7.0_migration.sql
