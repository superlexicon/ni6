-- ===============================================
-- Multi-Device Verification State Migration (v1.8.0)
-- ===============================================
-- This migration adds per-device verification state tracking to user_keys table.
-- Previously, verification_state was stored only in user_identity_index (shared across devices).
-- Now, each device (user_public_key) has its own verification_state and sequence_no.
--
-- Architecture:
-- - user_keys: Per-device state (each device has its own record)
-- - user_identity_index: Best unexpired state across devices (overall identity state)
-- - document_submissions: References user_keys for state (not user_identity_index)
--
-- Example Scenario:
-- - Device A (public_key_A): verification_state=3, sequence_no=3 (completed all steps)
-- - Device B (public_key_B): verification_state=1, sequence_no=1 (only selfie done)
-- - Result: Device A gets state 3, Device B gets state 1
-- ===============================================

-- Step 1: Add per-device tracking columns to user_keys
ALTER TABLE user_keys
ADD COLUMN verification_state TINYINT NOT NULL DEFAULT 0
    COMMENT 'Per-device verification state: 0=initial, 1=selfie, 2=passport, 3=complete',
ADD COLUMN sequence_no INT NOT NULL DEFAULT 0
    COMMENT 'Per-device submission progress: 0=initial, 1=selfie done, 2=passport data extracted, 3=complete';

-- Add index for efficient queries
ALTER TABLE user_keys
ADD INDEX idx_verification_state (verification_state);

-- Step 2: Initialize existing records with current user_identity_index state
-- This ensures existing data doesn't lose its state information
UPDATE user_keys uk
INNER JOIN user_identity_index uii ON uk.user_identity_id = uii.id
SET uk.verification_state = COALESCE(uii.verification_state, 0),
    uk.sequence_no = COALESCE(uii.sequence_no, 0)
WHERE uk.user_identity_id IS NOT NULL;

-- ===============================================
-- Verification Queries
-- ===============================================

-- Verify columns were added
SELECT
    user_public_key,
    verification_state,
    sequence_no,
    updated_at
FROM user_keys
LIMIT 10;

-- Check per-device states for a specific identity (example)
-- SELECT
--     user_public_key,
--     verification_state,
--     sequence_no,
--     updated_at
-- FROM user_keys
-- WHERE user_identity_id = 'your-identity-id-here'
-- ORDER BY verification_state DESC;

-- ===============================================
-- Rollback (if needed)
-- ===============================================
-- ALTER TABLE user_keys DROP COLUMN verification_state;
-- ALTER TABLE user_keys DROP COLUMN sequence_no;
-- ALTER TABLE user_keys DROP INDEX idx_verification_state;
