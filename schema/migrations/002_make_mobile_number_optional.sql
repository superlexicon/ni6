-- Migration: Make mobile_number optional
-- Version: 1.10.0
-- Date: 2026-04-17
-- Description: Allow OTP requests without mobile numbers
-- Reason: OTP is already returned encrypted in the response (using hybrid encryption
--          with the client's public key). SMS delivery is not strictly necessary for
--          all use cases. The client_public_key now serves as the primary identifier.

-- ===============================================
-- Step 1: Drop UNIQUE constraint on otp.mobile_number
-- ===============================================
-- Check for and drop any UNIQUE constraint on mobile_number
-- The constraint name may vary; we use a safe approach that works regardless of name
-- MariaDB/MySQL syntax: ALTER TABLE ... DROP INDEX ...

-- First, try to drop by common index name
ALTER TABLE otp DROP INDEX IF EXISTS unique_mobile_number;

-- Also try dropping by the index name shown in schema (if it's unique)
ALTER TABLE otp DROP INDEX IF EXISTS mobile_number;

-- Note: If the index has a different name, run:
-- SHOW INDEX FROM otp WHERE Key_name LIKE '%mobile%';
-- Then manually: ALTER TABLE otp DROP INDEX <actual_index_name>;

-- ===============================================
-- Step 2: Allow NULL in otp.mobile_number
-- ===============================================
-- This command makes mobile_number nullable while keeping all other properties
ALTER TABLE otp MODIFY COLUMN mobile_number VARCHAR(20) NULL;

-- ===============================================
-- Step 3: Allow NULL in user_keys.mobile_number
-- ===============================================
-- This command makes mobile_number nullable while keeping all other properties
ALTER TABLE user_keys MODIFY COLUMN mobile_number VARCHAR(20) NULL;

-- ===============================================
-- Step 4: Allow NULL in user_keys_pending.mobile_number
-- ===============================================
-- This command makes mobile_number nullable while keeping all other properties
ALTER TABLE user_keys_pending MODIFY COLUMN mobile_number VARCHAR(20) NULL;

-- ===============================================
-- Verification Queries (run after migration to verify)
-- ===============================================
-- 1. Check otp table allows NULL mobile_number:
-- DESCRIBE otp;

-- 2. Check user_keys table allows NULL mobile_number:
-- DESCRIBE user_keys;

-- 3. Check user_keys_pending table allows NULL mobile_number:
-- DESCRIBE user_keys_pending;

-- 4. Verify no UNIQUE constraint on mobile_number in otp:
-- SHOW INDEX FROM otp WHERE Column_name = 'mobile_number' AND Non_unique = 0;
-- This should return an empty result set

-- ===============================================
-- Rollback (if needed)
-- ===============================================
-- To revert this migration:
--
-- -- Make mobile_number NOT NULL again (will fail if NULL values exist)
-- ALTER TABLE otp MODIFY COLUMN mobile_number VARCHAR(20) NOT NULL;
-- ALTER TABLE user_keys MODIFY COLUMN mobile_number VARCHAR(20) NOT NULL;
-- ALTER TABLE user_keys_pending MODIFY COLUMN mobile_number VARCHAR(20) NOT NULL;
--
-- -- Re-add UNIQUE constraint (will fail if duplicate values exist)
-- ALTER TABLE otp ADD CONSTRAINT unique_mobile_number UNIQUE (mobile_number);
