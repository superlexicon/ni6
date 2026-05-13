-- Migration: Add client_public_key to document_submissions
-- Version: 1.11.0
-- Date: 2026-05-13
-- Description: Add client_public_key column and index to document_submissions table
-- Reason: The code expects this column for multi-device support (filtering submissions
--          by the device that submitted them), but existing databases don't have it.

-- ===============================================
-- Step 1: Add client_public_key column
-- ===============================================
ALTER TABLE document_submissions
ADD COLUMN IF NOT EXISTS client_public_key VARCHAR(255) NULL
COMMENT 'Client public key for lookup'
AFTER user_identity_id;

-- ===============================================
-- Step 2: Add index on client_public_key
-- ===============================================
-- This index is used for queries filtering by client_public_key
CREATE INDEX IF NOT EXISTS idx_client_public_key ON document_submissions(client_public_key);

-- ===============================================
-- Verification Queries
-- ===============================================
-- 1. Check client_public_key column exists:
-- DESCRIBE document_submissions;

-- 2. Check idx_client_public_key index exists:
-- SHOW INDEX FROM document_submissions WHERE Key_name = 'idx_client_public_key';

-- ===============================================
-- Rollback (if needed)
-- ===============================================
-- To revert this migration:
--
-- DROP INDEX idx_client_public_key ON document_submissions;
-- ALTER TABLE document_submissions DROP COLUMN client_public_key;
