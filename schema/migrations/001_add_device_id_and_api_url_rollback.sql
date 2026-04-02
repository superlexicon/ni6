-- Rollback Migration: Remove device_id and api_url columns
-- Date: 2025-03-27
-- Description: Removes device_id and api_url columns from user_keys_pending, user_keys, and otp tables
-- WARNING: This will permanently delete any data in these columns

-- ===============================================
-- Remove columns from otp table
-- ===============================================
ALTER TABLE otp DROP COLUMN device_id;
ALTER TABLE otp DROP COLUMN api_url;

-- ===============================================
-- Remove columns from user_keys table
-- ===============================================
-- First drop the indexes
ALTER TABLE user_keys DROP INDEX idx_device_id;
ALTER TABLE user_keys DROP INDEX idx_api_url;

-- Then drop the columns
ALTER TABLE user_keys DROP COLUMN device_id;
ALTER TABLE user_keys DROP COLUMN api_url;

-- ===============================================
-- Remove columns from user_keys_pending table
-- ===============================================
-- First drop the indexes
ALTER TABLE user_keys_pending DROP INDEX idx_device_id;
ALTER TABLE user_keys_pending DROP INDEX idx_api_url;

-- Then drop the columns
ALTER TABLE user_keys_pending DROP COLUMN device_id;
ALTER TABLE user_keys_pending DROP COLUMN api_url;

-- ===============================================
-- Verification query (optional - run to verify)
-- ===============================================
-- Check columns were removed successfully
-- SELECT COUNT(*) as columns_found
-- FROM INFORMATION_SCHEMA.COLUMNS
-- WHERE TABLE_NAME IN ('user_keys_pending', 'user_keys', 'otp')
--   AND COLUMN_NAME IN ('device_id', 'api_url');
-- Result should be 0 if rollback was successful
