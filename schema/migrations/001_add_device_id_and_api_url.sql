-- Migration: Add device_id and api_url columns for secret share tracking
-- Date: 2025-03-27
-- Description: Adds device_id and api_url columns to user_keys_pending, user_keys, and otp tables
--              to track which device a share belongs to and filter shares by API URL during recovery

-- ===============================================
-- Add columns to user_keys_pending table
-- ===============================================
ALTER TABLE user_keys_pending
ADD COLUMN device_id VARCHAR(255) NULL COMMENT 'Device identifier from client',
ADD COLUMN api_url VARCHAR(512) NULL COMMENT 'API URL that should receive this share';

-- Add indexes for efficient filtering
ALTER TABLE user_keys_pending
ADD INDEX idx_device_id (device_id),
ADD INDEX idx_api_url (api_url);

-- ===============================================
-- Add columns to user_keys table
-- ===============================================
ALTER TABLE user_keys
ADD COLUMN device_id VARCHAR(255) NULL COMMENT 'Device identifier from client',
ADD COLUMN api_url VARCHAR(512) NULL COMMENT 'API URL that received this share';

-- Add indexes for efficient filtering
ALTER TABLE user_keys
ADD INDEX idx_device_id (device_id),
ADD INDEX idx_api_url (api_url);

-- ===============================================
-- Add columns to otp table
-- ===============================================
ALTER TABLE otp
ADD COLUMN device_id VARCHAR(255) NULL COMMENT 'Device identifier from client',
ADD COLUMN api_url VARCHAR(512) NULL COMMENT 'API URL that received this share';

-- ===============================================
-- Verification queries (optional - run to verify)
-- ===============================================
-- Check columns were added successfully
-- SELECT COLUMN_NAME, DATA_TYPE, CHARACTER_MAXIMUM_LENGTH, COLUMN_COMMENT
-- FROM INFORMATION_SCHEMA.COLUMNS
-- WHERE TABLE_NAME IN ('user_keys_pending', 'user_keys', 'otp')
--   AND COLUMN_NAME IN ('device_id', 'api_url')
-- ORDER BY TABLE_NAME, COLUMN_NAME;
