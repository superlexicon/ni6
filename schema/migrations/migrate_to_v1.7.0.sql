-- ===============================================
-- Migration: v1.3.0 to v1.7.0
-- Description: Complete migration from schema v1.3.0 to v1.7.0
-- Date: 2026-03-11
-- ===============================================
--
-- IMPORTANT: Backup your database before running this migration!
-- mysqldump -u root -p im_osint > backup_before_v1.7.0_migration.sql
--
-- This migration includes all changes from:
-- - v1.4.0: Multi-Device Support (country_code, encrypted_secret_share)
-- - v1.5.0: Face Biometrics as Primary Identity (remove hash columns)
-- - v1.6.0: Device Identifier Removal
-- - v1.7.0: Pending Keys Table for Race Condition Fix
--
-- ===============================================
-- Section 1: Add new columns to otp table (v1.4.0)
-- ===============================================

-- Add country_code column to otp table (for multi-device linking)
ALTER TABLE otp
ADD COLUMN IF NOT EXISTS country_code VARCHAR(10) NULL COMMENT 'Country code for mobile number (for multi-device linking)' AFTER mobile_number;

-- Add encrypted_secret_share column to otp table (for multi-device linking)
ALTER TABLE otp
ADD COLUMN IF NOT EXISTS encrypted_secret_share TEXT NULL COMMENT 'Encrypted secret share (for multi-device linking)' AFTER public_key;

-- ===============================================
-- Section 2: Remove hash columns (v1.5.0)
-- ===============================================

-- Drop passport_hash from user_identity_index
-- This removes the uniqueness constraint on passport hashes
ALTER TABLE user_identity_index
DROP COLUMN IF EXISTS passport_hash;

-- Drop document_hash from document_submissions
-- This allows the same document to be submitted with different encryption per key
ALTER TABLE document_submissions
DROP COLUMN IF EXISTS document_hash;

-- ===============================================
-- Section 3: Remove device_identifier (v1.6.0)
-- ===============================================

-- Drop device_identifier index from user_keys
ALTER TABLE user_keys DROP INDEX IF EXISTS device_identifier;

-- Drop device_identifier column from user_keys
ALTER TABLE user_keys
DROP COLUMN IF EXISTS device_identifier;

-- Drop device_identifier column from otp table
ALTER TABLE otp
DROP COLUMN IF EXISTS device_identifier;

-- ===============================================
-- Section 4: Make country_code nullable in user_keys (v1.4.0)
-- ===============================================

-- Modify country_code to allow NULL values (for legacy records)
ALTER TABLE user_keys
MODIFY COLUMN country_code VARCHAR(10) NULL COMMENT 'Country code for mobile number (nullable for legacy records)';

-- ===============================================
-- Section 5: Add user_keys_pending table (v1.7.0)
-- ===============================================

-- Create staging table for user keys before verification
-- This fixes the race condition where broadcasts arrive at nodes 2 and 3
-- before their own signed OTP requests
CREATE TABLE IF NOT EXISTS user_keys_pending (
    id VARCHAR(255) PRIMARY KEY DEFAULT(UUID()),
    mobile_number VARCHAR(20) NOT NULL,
    country_code VARCHAR(10) NULL COMMENT 'Country code for mobile number (nullable for legacy records)',
    user_public_key TEXT NOT NULL,
    encrypted_secret_share TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_mobile_number (mobile_number),
    INDEX idx_country_code (country_code),
    INDEX idx_user_public_key_hash (user_public_key(64))
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
COMMENT='Staging table for user keys before verification';

-- ===============================================
-- Section 6: Verification queries
-- ===============================================

-- Verify all changes were applied successfully
SELECT 'Migration v1.3.0 to v1.7.0 completed' as status;

-- Check user_keys table structure
DESCRIBE user_keys;

-- Check otp table structure
DESCRIBE otp;

-- Check user_identity_index table structure
DESCRIBE user_identity_index;

-- Check document_submissions table structure
DESCRIBE document_submissions;

-- Check user_keys_pending table exists
DESCRIBE user_keys_pending;

-- Verify removed columns don't exist
SELECT COUNT(*) AS device_identifier_in_user_keys
FROM information_schema.COLUMNS
WHERE TABLE_SCHEMA = DATABASE()
  AND COLUMN_NAME = 'device_identifier'
  AND TABLE_NAME = 'user_keys';
-- Expected result: 0

SELECT COUNT(*) AS device_identifier_in_otp
FROM information_schema.COLUMNS
WHERE TABLE_SCHEMA = DATABASE()
  AND COLUMN_NAME = 'device_identifier'
  AND TABLE_NAME = 'otp';
-- Expected result: 0

SELECT COUNT(*) AS passport_hash_in_user_identity_index
FROM information_schema.COLUMNS
WHERE TABLE_SCHEMA = DATABASE()
  AND COLUMN_NAME = 'passport_hash'
  AND TABLE_NAME = 'user_identity_index';
-- Expected result: 0

SELECT COUNT(*) AS document_hash_in_document_submissions
FROM information_schema.COLUMNS
WHERE TABLE_SCHEMA = DATABASE()
  AND COLUMN_NAME = 'document_hash'
  AND TABLE_NAME = 'document_submissions';
-- Expected result: 0

-- Verify new columns exist
SELECT COUNT(*) AS country_code_in_otp
FROM information_schema.COLUMNS
WHERE TABLE_SCHEMA = DATABASE()
  AND COLUMN_NAME = 'country_code'
  AND TABLE_NAME = 'otp';
-- Expected result: 1

SELECT COUNT(*) AS encrypted_secret_share_in_otp
FROM information_schema.COLUMNS
WHERE TABLE_SCHEMA = DATABASE()
  AND COLUMN_NAME = 'encrypted_secret_share'
  AND TABLE_NAME = 'otp';
-- Expected result: 1

SELECT COUNT(*) AS user_keys_pending_exists
FROM information_schema.TABLES
WHERE TABLE_SCHEMA = DATABASE()
  AND TABLE_NAME = 'user_keys_pending';
-- Expected result: 1

-- ===============================================
-- Notes
-- ===============================================
--
-- After migration:
-- 1. Public key uniqueness is enforced by idx_user_public_key_hash index
-- 2. Multi-device support is handled by user_identity_id column
-- 3. Authentication uses ECDSA signature verification with public_key
-- 4. Face biometrics trigger provides identity uniqueness
-- 5. user_keys_pending table stores key data before verification to prevent race conditions
--
-- Cleanup Configuration:
-- The CleanupWorker automatically cleans up abandoned records:
-- - OTP records: Deleted after 168 hours (7 days) by default (configurable via CLEANUP_ABANDONED_HOURS)
-- - Pending keys: Deleted after 24 hours (hardcoded - these should be verified quickly)
--
-- To adjust cleanup intervals:
-- export CLEANUP_ABANDONED_HOURS=168  # OTP cleanup threshold in hours
--
-- To manually trigger cleanup:
-- See app/services/cleanup_worker.py
--
-- To rollback this migration, see: rollback_to_v1.3.0.sql
