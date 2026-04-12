-- IM-OSINT Database Schema
-- Version: 1.9.0
-- Created: 2025-11-27
-- Updated: 2026-04-12
-- MySQL Requirements: MariaDB 11.7+ (for VECTOR type)
-- Description: Complete schema for IM-OSINT KYC verification application
-- Encryption: ECIES (ephemeral key) for user-only PII decryption
-- Security: PII only stored encrypted in pii_data_encrypted column

-- Database and User Setup (run once as root)
-- CREATE DATABASE IF NOT EXISTS im_osint CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
-- CREATE USER IF NOT EXISTS 'app_user'@'%' IDENTIFIED BY 'secure_password';
-- GRANT ALL PRIVILEGES ON im_osint.* TO 'app_user'@'%';
-- FLUSH PRIVILEGES;

-- Switch to application database
-- USE im_osint;  -- Commented out: database is specified in mysql command

-- Drop existing tables for clean setup (uncomment for fresh install)
-- DROP TABLE IF EXISTS document_analysis_jobs;
-- DROP TABLE IF EXISTS document_submissions;
-- DROP TABLE IF EXISTS face_biometrics;
-- DROP TABLE IF EXISTS user_identity_index;
-- DROP TABLE IF EXISTS otp;
-- DROP TABLE IF EXISTS user_keys;
-- DROP TABLE IF EXISTS sanctions_entries;
-- DROP TABLE IF EXISTS sanctions_lists;
-- DROP TABLE IF EXISTS face_search_results;
-- DROP TABLE IF EXISTS face_search_matched_images;

-- ===============================================
-- Table: sanctions_lists
-- Description: Metadata table for sanctions lists (OFAC, EU, UN)
-- ===============================================
CREATE TABLE sanctions_lists (
    id INT AUTO_INCREMENT PRIMARY KEY,
    source_key VARCHAR(20) NOT NULL UNIQUE COMMENT 'ofac, eu, or un',
    source_name VARCHAR(255) NOT NULL COMMENT 'Human-readable name',
    source_url VARCHAR(500) NOT NULL COMMENT 'Download URL',
    last_sync_at TIMESTAMP NULL COMMENT 'Last successful sync time',
    last_sync_status ENUM('success', 'failed', 'partial') DEFAULT 'failed',
    last_sync_error TEXT NULL COMMENT 'Error message from last sync attempt',
    entry_count INT DEFAULT 0 COMMENT 'Number of entries in source',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_source_key (source_key),
    INDEX idx_last_sync_at (last_sync_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
COMMENT='Metadata table for sanctions lists (OFAC, EU, UN)';

-- Insert initial metadata for sanctions sources
INSERT INTO sanctions_lists (source_key, source_name, source_url, last_sync_status) VALUES
('ofac', 'OFAC Specially Designated Nationals (SDN) List', 'https://www.treasury.gov/ofac/downloads/sdn.csv', 'failed'),
('eu', 'EU Financial Sanctions Files (FSF) via OpenSanctions', 'https://data.opensanctions.org/datasets/latest/eu_fsf/targets.simple.csv', 'failed'),
('un', 'UN Security Council Consolidated List', 'https://unsolprodfiles.blob.core.windows.net/publiclegacyxmlfiles/EN/consolidated.xml', 'failed')
ON DUPLICATE KEY UPDATE
    source_name = VALUES(source_name),
    source_url = VALUES(source_url);

-- ===============================================
-- Table: sanctions_entries
-- Description: Individual entries from sanctions lists with parsed searchable fields
-- ===============================================
CREATE TABLE sanctions_entries (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    source_key VARCHAR(20) NOT NULL COMMENT 'ofac, eu, or un',
    name VARCHAR(500) NOT NULL COMMENT 'Original name from source',
    normalized_name VARCHAR(500) NOT NULL COMMENT 'Name normalized for search (lowercase, no titles)',
    entry_type VARCHAR(50) DEFAULT 'unknown' COMMENT 'Person, Organization, Entity, Vessel',
    program TEXT NULL COMMENT 'Sanctions program or list type',
    birth_date VARCHAR(100) NULL COMMENT 'Date of birth (YYYY-MM-DD or YYYY)',
    countries TEXT NULL COMMENT 'Associated countries (semicolon-separated)',
    address TEXT NULL COMMENT 'Address information',
    identifiers TEXT NULL COMMENT 'Passport, ID, or other identifiers',
    aliases TEXT NULL COMMENT 'Alternative names (semicolon-separated)',
    title VARCHAR(255) NULL COMMENT 'Title or position',
    sanctions_details TEXT NULL COMMENT 'Additional sanctions information',
    raw_data JSON NULL COMMENT 'Complete original data for reference',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,

    INDEX idx_source_key (source_key),
    INDEX idx_normalized_name (normalized_name),
    INDEX idx_entry_type (entry_type),
    FOREIGN KEY (source_key) REFERENCES sanctions_lists(source_key)
        ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
COMMENT='Individual entries from sanctions lists with parsed searchable fields';

-- ===============================================
-- Table: user_keys
-- Description: Stores device-based user authentication keys
-- ===============================================
CREATE TABLE user_keys (
    id VARCHAR(255) PRIMARY KEY DEFAULT(UUID()),
    mobile_number VARCHAR(20) NOT NULL,
    country_code VARCHAR(10) NULL COMMENT 'Country code for mobile number (nullable for legacy records)',
    user_public_key TEXT NOT NULL,
    encrypted_secret_share TEXT,
    user_identity_id VARCHAR(64) NULL COMMENT 'References user_identity_index.id - links user key to verified identity',
    verification_state TINYINT NOT NULL DEFAULT 0 COMMENT 'Per-device verification state: 0=initial, 1=selfie, 2=passport, 3=complete',
    sequence_no INT NOT NULL DEFAULT 0 COMMENT 'Per-device submission progress: 0=initial, 1=selfie done, 2=passport data extracted, 3=complete',
    device_id VARCHAR(255) NULL COMMENT 'Device identifier from client',
    api_url VARCHAR(512) NULL COMMENT 'API URL that received this share',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_mobile_number (mobile_number),
    INDEX idx_country_code (country_code),
    INDEX idx_user_public_key_hash (user_public_key(64)),
    INDEX idx_user_keys_user_identity_id (user_identity_id),
    INDEX idx_verification_state (verification_state),
    INDEX idx_device_id (device_id),
    INDEX idx_api_url (api_url)
);

-- ===============================================
-- Table: user_keys_pending
-- Description: Staging table for user keys before verification
-- Stores key data immediately upon OTP request (before verification)
-- Moved to user_keys after selfie verification passes
-- ===============================================
CREATE TABLE user_keys_pending (
    id VARCHAR(255) PRIMARY KEY DEFAULT(UUID()),
    mobile_number VARCHAR(20) NOT NULL,
    country_code VARCHAR(10) NULL COMMENT 'Country code for mobile number (nullable for legacy records)',
    user_public_key TEXT NOT NULL,
    encrypted_secret_share TEXT,
    device_id VARCHAR(255) NULL COMMENT 'Device identifier from client',
    api_url VARCHAR(512) NULL COMMENT 'API URL that should receive this share',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_mobile_number (mobile_number),
    INDEX idx_country_code (country_code),
    INDEX idx_user_public_key_hash (user_public_key(64)),
    INDEX idx_device_id (device_id),
    INDEX idx_api_url (api_url)
) COMMENT='Staging table for user keys before verification';

-- ===============================================
-- Table: otp
-- Description: One-time password verification system
--
-- Multi-device support: Stores pending OTP requests before verification.
-- Only verified users are inserted into user_keys table.
-- ===============================================
CREATE TABLE otp (
    id VARCHAR(255) PRIMARY KEY DEFAULT(UUID()),
    email VARCHAR(255) UNIQUE,
    public_key VARCHAR(255),
    mobile_number VARCHAR(20) UNIQUE,
    country_code VARCHAR(10) NULL COMMENT 'Country code for mobile number (for multi-device linking)',
    encrypted_secret_share TEXT NULL COMMENT 'Encrypted secret share (for multi-device linking)',
    device_id VARCHAR(255) NULL COMMENT 'Device identifier from client',
    api_url VARCHAR(512) NULL COMMENT 'API URL that should receive this share',
    random_number VARCHAR(10) NOT NULL,
    otp_id VARCHAR(255) UNIQUE,
    delivery_method VARCHAR(20) DEFAULT 'email',
    expires_at DATETIME NULL,
    attempts INT DEFAULT 0,
    max_attempts INT DEFAULT 3,
    is_verified BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_otp_mobile_number (mobile_number),
    INDEX idx_otp_otp_id (otp_id),
    INDEX idx_otp_expires_at (expires_at),
    INDEX idx_otp_public_key (public_key)
);

-- ===============================================
-- Table: user_identity_index
-- Description: Index of user identities - stores verification state
-- verification_state: 0=initial, 1=after selfie, 2=after passport, 3=complete
-- PII Encryption: ECIES (ephemeral key) - user-only decryption
-- ===============================================
CREATE TABLE user_identity_index (
    id VARCHAR(36) PRIMARY KEY DEFAULT(UUID()),
    verification_state TINYINT NOT NULL DEFAULT 0 COMMENT '0=initial, 1=selfie, 2=passport, 3=complete',
    sequence_no INT NOT NULL DEFAULT 0 COMMENT 'Submission progress: 0=initial, 1=selfie done, 2=passport data extracted, 3=complete. Tracks attempted submissions, verification_state tracks successful completions.',
    -- Encrypted PII column (ECIES encrypted with user's public key - user-only decryption)
    pii_data_encrypted TEXT NULL COMMENT 'PII encrypted with user public key (ECIES ephemeral key envelope) - only user can decrypt. Contains: date_of_birth, gender, passport_number, passport_country, bank_statement_address',
    -- Stored plaintext for display and matching (non-sensitive)
    full_name VARCHAR(255) COMMENT 'Full name for display and matching (stored plaintext for verification)',
    -- Non-PII columns
    passport_expiry_date DATE COMMENT 'Passport expiration date (non-PII, used for validation)',
    bank_statement_date DATE COMMENT 'Bank statement date (non-PII, used for age validation)',
    -- OSINT screening results
    worldcheck_screening_result JSON NULL COMMENT 'World Check One screening results',
    osint_screening_result JSON NULL COMMENT 'OSINT background search results (alternative to World Check)',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,

    INDEX idx_verification_state (verification_state),
    INDEX idx_sequence_no (sequence_no),
    INDEX idx_full_name (full_name)
);

-- ===============================================
-- Table: face_biometrics
-- Description: Stores face embeddings for key recovery verification
-- ===============================================
CREATE TABLE face_biometrics (
    id VARCHAR(36) PRIMARY KEY DEFAULT(UUID()),
    user_identity_id VARCHAR(36) NOT NULL COMMENT 'References user_identity_index.id',
    face_embedding JSON NOT NULL COMMENT 'Face embedding vector (legacy, kept for backward compatibility)',
    embedding_vec VECTOR(512) NOT NULL COMMENT 'Native VECTOR for similarity search (512 dims), used for HNSW indexing',
    model_name VARCHAR(100) NOT NULL DEFAULT 'deepface_vgg-face' COMMENT 'Face recognition model used (deepface_vgg-face, insightface_buffalo_l)',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Indexes
    INDEX idx_user_identity_id (user_identity_id),
    INDEX idx_created_at (created_at),
    INDEX idx_model_name (model_name),
    INDEX idx_model_user (model_name, user_identity_id),
    VECTOR INDEX idx_embedding_vec (embedding_vec) M=16 DISTANCE=cosine
) COMMENT='Face biometrics with VECTOR-based similarity search and model tracking';

-- Trigger to prevent same face registering under different identities
-- Allows: Same face, same user_identity_id (recovery selfies)
-- Rejects: Same face, DIFFERENT user_identity_id (fraud prevention)
-- IMPORTANT: Only compares embeddings from the SAME model (cross-model comparison is invalid)
DELIMITER //
CREATE TRIGGER trg_face_biometrics_cross_identity_check
BEFORE INSERT ON face_biometrics
FOR EACH ROW
BEGIN
    DECLARE existing_identity VARCHAR(36);

    -- Check if this face matches any existing face under a DIFFERENT identity
    -- Uses HNSW index for O(log n) lookup
    -- CRITICAL: Only compare embeddings from the SAME model
    SELECT user_identity_id INTO existing_identity
    FROM face_biometrics
    WHERE user_identity_id != NEW.user_identity_id
      AND model_name = NEW.model_name  -- SAME MODEL ONLY - cross-model comparison is invalid
      AND VEC_DISTANCE_COSINE(embedding_vec, NEW.embedding_vec) < 0.4
    LIMIT 1;

    IF existing_identity IS NOT NULL THEN
        SIGNAL SQLSTATE '45000'
        SET MESSAGE_TEXT = 'DUPLICATE_FACE: Face already registered under different identity',
            MYSQL_ERRNO = 1062;
    END IF;
END //
DELIMITER ;

-- ===============================================
-- Table: document_submissions
-- Description: Stores submitted documents and their analysis results
-- ===============================================
CREATE TABLE document_submissions (
    id VARCHAR(255) PRIMARY KEY DEFAULT(UUID()),
    user_identity_id VARCHAR(255) NULL,
    client_public_key VARCHAR(255) NULL COMMENT 'Client public key for lookup',
    filename VARCHAR(255),
    document_type VARCHAR(50),
    request_data JSON,
    job_id VARCHAR(36) NULL,
    submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    -- Encrypted PII column (ECIES encrypted with user's public key - user-only decryption)
    extracted_data_encrypted TEXT NULL COMMENT 'Extracted PII encrypted with user public key (ECIES ephemeral key envelope) - only user can decrypt. Contains: selfie (otp_number), passport (full_name, date_of_birth, passport_number, passport_country, nationality, place_of_birth, sex), bank_statement (account_holder_name, account_number, address)',
    -- Summary/metrics columns (non-PII, for queries)
    processing_time_seconds FLOAT NULL COMMENT 'Processing time in seconds',
    verification_state INT NULL COMMENT 'Verification state 0-3',
    sequence_no INT NULL COMMENT 'Sequence number 0-3',
    docs_auth_score FLOAT NULL COMMENT 'Document authentication score %',
    id_veri_score FLOAT NULL COMMENT 'Identity verification score %',
    forgery_checks_summary JSON NULL COMMENT 'Summary of forgery detection',
    other_checks_summary JSON NULL COMMENT 'Summary of validation checks',
    result_status BOOLEAN NULL COMMENT 'True=passed, False=failed',
    error_message TEXT NULL COMMENT 'Error message if processing failed',
    INDEX idx_user_identity (user_identity_id),
    INDEX idx_job_id (job_id),
    INDEX idx_client_public_key (client_public_key)
);

-- ===============================================
-- Table: document_analysis_jobs
-- Description: Job queue for document analysis tasks
-- ===============================================
CREATE TABLE document_analysis_jobs (
    id VARCHAR(36) PRIMARY KEY,
    client_public_key VARCHAR(255) NULL COMMENT 'Client public key for lookup by public key',
    user_identity_id VARCHAR(36) NULL COMMENT 'References user_identity_index.id - links job to user identity',
    status ENUM('pending', 'processing', 'completed', 'failed', 'callback_failed') DEFAULT 'pending',
    request_data JSON NOT NULL,
    response_data JSON NULL,
    error_message TEXT NULL,
    callback_url VARCHAR(500) NULL,
    retry_count INT DEFAULT 0,
    max_retries INT DEFAULT 3,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    started_at TIMESTAMP NULL,
    completed_at TIMESTAMP NULL,
    callback_attempted_at TIMESTAMP NULL,
    INDEX idx_status (status),
    INDEX idx_created_at (created_at),
    INDEX idx_callback_url (callback_url),
    INDEX idx_client_public_key (client_public_key),
    INDEX idx_user_identity_id (user_identity_id)
);

-- ===============================================
-- Schema Verification Queries
-- ===============================================

-- Verify all tables were created successfully
SELECT 'Tables created successfully' as status;

-- List all tables
SHOW TABLES;

-- Check table structures
DESCRIBE user_keys;
DESCRIBE otp;
DESCRIBE user_identity_index;
DESCRIBE face_biometrics;
DESCRIBE document_submissions;
DESCRIBE document_analysis_jobs;
DESCRIBE sanctions_lists;
DESCRIBE sanctions_entries;

-- Check indexes
SHOW INDEX FROM user_keys;
SHOW INDEX FROM otp;
SHOW INDEX FROM user_identity_index;
SHOW INDEX FROM face_biometrics;
SHOW INDEX FROM document_submissions;
SHOW INDEX FROM document_analysis_jobs;
SHOW INDEX FROM sanctions_lists;
SHOW INDEX FROM sanctions_entries;


-- ===============================================
-- Migration Notes
-- ===============================================
-- For existing databases upgrading to v1.3.0:

-- 1. Add ECIES encrypted PII column (if not exists):
-- ALTER TABLE user_identity_index
-- ADD COLUMN pii_data_encrypted TEXT NULL
-- COMMENT 'PII encrypted with user public key (ECIES ephemeral key envelope) - only user can decrypt';

-- 2. One-time data migration for existing records:
--    - Read existing plaintext PII columns (full_name, date_of_birth, etc.)
--    - Encrypt with ECIES using user's public key
--    - Store in pii_data_encrypted

-- 4. Drop deprecated plaintext PII columns (after migration verification):
-- ALTER TABLE user_identity_index DROP COLUMN passport_country;
-- ALTER TABLE user_identity_index DROP COLUMN passport_number;
-- ALTER TABLE user_identity_index DROP COLUMN full_name;
-- ALTER TABLE user_identity_index DROP COLUMN date_of_birth;
-- ALTER TABLE user_identity_index DROP COLUMN gender;
-- ALTER TABLE user_identity_index DROP COLUMN bank_statement_address;

-- 5. Drop legacy unique constraint:
-- ALTER TABLE user_identity_index DROP INDEX unique_passport;
-- ALTER TABLE user_identity_index DROP INDEX idx_passport_country;

-- ===============================================
-- Multi-Device Support Migration (v1.4.0)
-- ===============================================
-- For existing databases upgrading to support multi-device:

-- 1. Add country_code column to otp table (if not exists):
-- ALTER TABLE otp
-- ADD COLUMN country_code VARCHAR(10) NULL AFTER mobile_number
-- COMMENT 'Country code for mobile number (for multi-device linking)';

-- 2. Add encrypted_secret_share column to otp table (if not exists):
-- ALTER TABLE otp
-- ADD COLUMN encrypted_secret_share TEXT NULL AFTER public_key
-- COMMENT 'Encrypted secret share (for multi-device linking)';

-- 3. Update otp table queries to include new columns
--    - No data migration needed for existing records (new columns are nullable)
--    - New OTP requests will populate these columns
--    - Old OTP requests without these columns will continue to work

-- ===============================================
-- Face Biometrics as Primary Identity (v1.5.0)
-- ===============================================
-- For existing databases upgrading to use face biometrics as primary identifier:

-- 1. Remove hash columns (run migration file):
--    See schema/migrations/remove_hash_columns.sql
--
--    This removes:
--    - passport_hash from user_identity_index
--    - document_hash from document_submissions
--
-- 2. Face biometrics trigger provides identity uniqueness:
--    - trg_face_biometrics_cross_identity_check prevents same face under different IDs
--    - Multiple keys per user_identity_id enable multi-device support
--    - Same document can be submitted with different encryption per key

-- ===============================================
-- Device Identifier Removal (v1.6.0)
-- ===============================================
-- For existing databases upgrading to remove device_identifier:

-- 1. Remove device_identifier column from user_keys and otp tables:
--    See schema/migrations/remove_device_identifier.sql
--
--    This removes:
--    - device_identifier column from user_keys
--    - device_identifier column from otp
--
-- 2. Public key uniqueness is enforced by idx_user_public_key_hash index
-- 3. Multi-device support is handled by user_identity_id column
-- 4. Authentication uses ECDSA signature verification with public_key

-- ===============================================
-- Pending Keys Table for Multi-Device Race Condition Fix (v1.7.0)
-- ===============================================
-- For existing databases upgrading to fix encrypted_secret_share race condition:

-- 1. Add user_keys_pending table (run migration file):
--    See schema/migrations/add_user_keys_pending_table.sql
--
--    This creates:
--    - user_keys_pending table for staging key data before verification
--
-- 2. The fix addresses the race condition where broadcasts arrive at nodes 2 and 3
--    before their own signed OTP requests, causing encrypted_secret_share to be NULL.
--    Key data is now stored immediately in user_keys_pending upon OTP request,
--    then moved to user_keys after selfie verification passes.
--
-- 3. To rollback (remove the table):
--    See schema/migrations/rollback_user_keys_pending_table.sql

-- ===============================================
-- Migration Status (v1.7.0)
-- ===============================================
-- All previous migrations have been merged into this schema:
--
-- v1.4.0 - Multi-Device Support:
--   - country_code column added to otp and user_keys tables (nullable)
--   - encrypted_secret_share column added to otp table
--
-- v1.5.0 - Face Biometrics as Primary Identity:
--   - passport_hash removed from user_identity_index
--   - document_hash removed from document_submissions
--   - Face biometrics trigger provides identity uniqueness
--
-- v1.6.0 - Device Identifier Removal:
--   - device_identifier removed from user_keys and otp tables
--   - Public key uniqueness enforced by idx_user_public_key_hash index
--
-- v1.7.0 - Pending Keys Table for Race Condition Fix:
--   - user_keys_pending table added for staging key data before verification
--   - country_code made nullable in user_keys and user_keys_pending tables
--
-- For new installations, simply run this schema.sql file.
-- For existing databases, see individual migration files in schema/migrations/

-- ===============================================
-- Multi-Device Verification State Migration (v1.8.0)
-- ===============================================
-- For existing databases upgrading to support per-device verification state:

-- 1. Add verification_state and sequence_no columns to user_keys table (if not exists):
--    See schema/migrations/add_user_keys_verification_state.sql
--
--    This adds:
--    - verification_state TINYINT NOT NULL DEFAULT 0 to user_keys
--    - sequence_no INT NOT NULL DEFAULT 0 to user_keys
--    - idx_verification_state index to user_keys
--
-- 2. Initialize existing records with current user_identity_index state:
--    The migration script handles this automatically.
--
-- 3. Update application code to use user_keys for per-device state:
--    - UserKeyRepository gets new methods: get_verification_state, get_sequence_no, update_state_and_sequence
--    - VerificationStateService reads from user_keys instead of document_submissions
--    - Document processors update user_keys on completion
--
-- 4. To rollback (remove the columns):
--    See comments in schema/migrations/add_user_keys_verification_state.sql

-- ===============================================
-- Face Recognition Model Tracking (v1.9.0)
-- ===============================================
-- For existing databases upgrading to support multiple face recognition models:

-- 1. Add model_name column (run migration file):
--    See schema/migrations/add_model_name_to_face_biometrics.sql
--
--    This adds:
--    - model_name VARCHAR(100) NOT NULL DEFAULT 'deepface_vgg-face' to face_biometrics
--    - idx_model_name index for model filtering
--    - idx_model_user composite index for model + user queries

-- 2. Update existing records (assumes all are DeepFace):
--    The migration script handles this automatically.

-- 3. Update duplicate detection trigger (same model only):
--    The migration script recreates the trigger with model filtering.

-- 4. Key changes to application code:
--    - FaceBiometricsRepository now tracks and filters by model_name
--    - All face comparison queries include AND model_name = X filter
--    - Services pass model_name from factory's get_model_name()

-- 5. To rollback (remove model tracking):
--    Drop model_name column and indexes
--    Restore old trigger (without model filtering)

