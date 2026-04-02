-- Rollback: Remove user_keys_pending table
-- Description: Remove the user_keys_pending table if reverting the migration

-- Drop the user_keys_pending table
DROP TABLE IF EXISTS user_keys_pending;
