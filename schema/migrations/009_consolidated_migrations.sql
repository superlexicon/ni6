-- ============================================================
-- Migration 009: Consolidated migrations (009-022)
-- Version: v1.20.0
-- Date: 2025-06-15
-- Purpose: Consolidate all local development migrations into single migration
--          for deployment to remote server (last migration was 008)
--
-- This consolidates:
-- - Bank schema simplification (013)
-- - Bank additions (014-015, 017-020)
-- - Drop obsolete tables (016, 022)
-- ============================================================

SET FOREIGN_KEY_CHECKS = 0;
SET AUTOCOMMIT = 0;

-- ============================================================
-- Step 1: Drop old bank-related tables
-- ============================================================

DROP TABLE IF EXISTS bank_layout_cache;
DROP TABLE IF EXISTS prompt_generation_history;
DROP TABLE IF EXISTS bank_gliner_prompts;
DROP TABLE IF EXISTS bank_extraction_config;
DROP TABLE IF EXISTS bank_identifiers;
DROP TABLE IF EXISTS bank_country_operations;
DROP TABLE IF EXISTS banks;

-- Drop validator-related tables (obsolete)
DROP TABLE IF EXISTS address_extraction_exceptions;
DROP TABLE IF EXISTS currency_name_map;
DROP TABLE IF EXISTS field_labels;
DROP TABLE IF EXISTS currencies;
DROP TABLE IF EXISTS address_formats;
DROP TABLE IF EXISTS state_to_country_map;
DROP TABLE IF EXISTS prompt_feedback;

-- ============================================================
-- Step 2: Create new simplified banks table
-- ============================================================

CREATE TABLE banks (
    id INT AUTO_INCREMENT PRIMARY KEY,
    swift_code VARCHAR(11) NOT NULL UNIQUE,
    country_code VARCHAR(2) NOT NULL,
    legal_name VARCHAR(255) NOT NULL,
    abbreviations TEXT,
    common_names TEXT,
    is_active TINYINT(1) DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,

    UNIQUE KEY idx_swift_code (swift_code),
    INDEX idx_country (country_code),
    INDEX idx_active (is_active),
    FULLTEXT INDEX idx_search_names (abbreviations, common_names)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
COMMENT='Simplified bank lookup table with unique SWIFT codes';

-- ============================================================
-- Step 3: Insert comprehensive bank data
-- ============================================================

-- Singapore banks
INSERT INTO banks (swift_code, country_code, legal_name, abbreviations, common_names, is_active) VALUES
('DBSSSGSG', 'SG', 'DBS Bank Ltd', 'DBS,DBSS,POSB', 'DBS Bank,Development Bank of Singapore,POSB Bank', 1),
('HSBCSGSG', 'SG', 'HSBC Singapore', 'HSBC', 'Hongkong and Shanghai Banking Corporation', 1),
('OCBCSGSG', 'SG', 'OCBC Bank', 'OCBC', 'Oversea-Chinese Banking Corporation', 1),
('UOVBSGSG', 'SG', 'UOB Singapore', 'UOB', 'United Overseas Bank', 1),
('CITISGSG', 'SG', 'Citibank Singapore', 'CITI', 'Citibank', 1);

-- Indian banks
INSERT INTO banks (swift_code, country_code, legal_name, abbreviations, common_names, is_active) VALUES
('HDFCINBB', 'IN', 'HDFC Bank Ltd', 'HDFC', 'HDFC Bank,Housing Development Finance Corporation', 1),
('ICICINBB', 'IN', 'ICICI Bank Ltd', 'ICICI,ICIC', 'ICICI Bank,Industrial Credit and Investment Corporation of India,AICici Bank', 1),
('SBININBB', 'IN', 'State Bank of India', 'SBI,SBIN', 'State Bank of India', 1),
('AXISINBB', 'IN', 'Axis Bank Ltd', 'AXIS,UTIB', 'Axis Bank', 1),
('PUNBINBB', 'IN', 'Punjab National Bank', 'PNB,PUNB', 'Punjab National Bank', 1),
('UBININBB', 'IN', 'Union Bank of India', 'UBI,UBIN', 'Union Bank of India', 1),
('BKIDINBB', 'IN', 'Bank of India', 'BOI,BKID', 'Bank of India', 1),
('MAHBINBB', 'IN', 'Bank of Maharashtra', 'MAHB,MAHAR', 'Bank of Maharashtra,mahabank,HERICE Bank of Maharashtra', 1),
('CANBINBB', 'IN', 'Canara Bank', 'CAN,CANR', 'Canara Bank', 1),
('CORPINBB', 'IN', 'Corporation Bank', 'CORP,CORPBNK', 'Corporation Bank', 1),
('ALLAINBB', 'IN', 'Allahabad Bank', 'ALLA,ALLA', 'Allahabad Bank', 1),
('ANDAINBB', 'IN', 'Andhra Pradesh Grameena Vikas Bank', 'ANDA,APGVB', 'Andhra Pradesh Grameena Vikas Bank', 1),
('BARBINBB', 'IN', 'Bank of Baroda', 'BOB,BARB', 'Bank of Baroda', 1),
('CABBINBB', 'IN', 'Central Bank of India', 'CBI,CABB', 'Central Bank of India', 1),
('CBININBB', 'IN', 'Central Bank of India', 'CBI,CBIN', 'Central Bank of India', 1),
('CIUBINBB', 'IN', 'City Union Bank', 'CUB,CIUB', 'City Union Bank', 1),
('DEUTINBB', 'IN', 'Deutsche Bank India', 'DEUT,DEUT', 'Deutsche Bank', 1),
('DLXBINBB', 'IN', 'Dhanlaxmi Bank', 'DLXB,DLX', 'Dhanlaxmi Bank', 1),
('DSKBINBB', 'IN', 'DKB Bank India', 'DSK,DSKB', 'DKB Bank', 1),
('FEDRINBB', 'IN', 'Federal Bank', 'FEDR,FDRL', 'Federal Bank', 1),
('INDBINBB', 'IN', 'Indian Bank', 'INDB,IND', 'Indian Bank', 1),
('IOBAINBB', 'IN', 'Indian Overseas Bank', 'IOB,IOBA', 'Indian Overseas Bank', 1),
('JAKAINBB', 'IN', 'Jammu and Kashmir Bank', 'JAKA,J&K', 'Jammu and Kashmir Bank', 1),
('KKBKINBB', 'IN', 'Kotak Mahindra Bank', 'KKBK,KOTAK', 'Kotak Mahindra Bank', 1),
('KVBLINBB', 'IN', 'Karur Vysya Bank', 'KVB,KVBL', 'Karur Vysya Bank', 1),
('LAVBINBB', 'IN', 'Lakshmi Vilas Bank', 'LAV,LAVB', 'Lakshmi Vilas Bank', 1),
('NRBLINBB', 'IN', 'Nainital Bank', 'NRBL,NRB', 'Nainital Bank', 1),
('RATNINBB', 'IN', 'RBL Bank', 'RATN,RBL', 'RBL Bank', 1),
('SVCBINBB', 'IN', 'Shivalik Small Finance Bank', 'SVCB,SVC', 'Shivalik Small Finance Bank', 1),
('TMBLINBB', 'IN', 'Tamilnad Mercantile Bank', 'TMB,TMBL', 'Tamilnad Mercantile Bank', 1),
('UCBAINBB', 'IN', 'UCO Bank', 'UCO,UCBA', 'UCO Bank', 1),
('VIJBINBB', 'IN', 'Vijaya Bank', 'VIJB,VIJ', 'Vijaya Bank', 1),
('YESBINBB', 'IN', 'Yes Bank', 'YESB,YES', 'Yes Bank', 1);

-- UAE banks
INSERT INTO banks (swift_code, country_code, legal_name, abbreviations, common_names, is_active) VALUES
('NBDQAQNA', 'AE', 'Emirates NBD', 'ENBD,NBD', 'Emirates NBD,National Bank of Dubai', 1),
('ADCBAEAA', 'AE', 'Abu Dhabi Commercial Bank', 'ADCB', 'Abu Dhabi Commercial Bank', 1),
('FABAAEAA', 'AE', 'First Abu Dhabi Bank', 'FAB', 'First Abu Dhabi Bank', 1),
('RAKBAEAD', 'AE', 'National Bank of Ras Al-Khaimah', 'RAK,NRAK,RAKBANK', 'RAKBANK,Rak Bank,Ras Al Khaimah National Bank,National Bank of Ras Al-Khaimah', 1),
('BBMEAEAD', 'AE', 'HSBC Bank Middle East Limited', 'HSBC,BBME,HSBCAE', 'HSBC,HSBC UAE,HSBC Bank Middle East,HSBC Bank UAE', 1),
('MASCQAQA', 'AE', 'Mashreq Bank', 'MASHREQ,MASH', 'Mashreq Bank', 1),
('DIBAAEAD', 'AE', 'Dubai Islamic Bank', 'DIB,DUBAI', 'Dubai Islamic Bank', 1),
('CBASAEAA', 'AE', 'Commercial Bank of Dubai', 'CBD,CBAS', 'Commercial Bank of Dubai', 1),
('ARBKAEAA', 'AE', 'Arab Bank', 'ARB,ARAB', 'Arab Bank', 1),
('NBFQAQNA', 'AE', 'National Bank of Fujairah', 'NBF', 'National Bank of Fujairah', 1);

-- Thailand banks
INSERT INTO banks (swift_code, country_code, legal_name, abbreviations, common_names, is_active) VALUES
('KRABORHK', 'TH', 'Krung Thai Bank Public Company Limited', 'KTB,KRUNGTHAI,KRUNG THAI', 'Krung Thai Bank,Krungthai Bank,Krung Thai Bank Public Company Limited,KTB', 1),
('BKKBTHBK', 'TH', 'Bangkok Bank', 'BBL,BKK', 'Bangkok Bank', 1),
('AYUDTHBK', 'TH', 'Krungsri Bank', 'BAY,AYUD', 'Krungsri Bank,Bank of Ayudhya', 1),
('KASITHBK', 'TH', 'Kasikornbank', 'KBank,KASI', 'Kasikornbank,KASIKORN', 1),
('TFUESTHH', 'TH', 'TMB Thanachart Bank', 'TTB,TFUS', 'TMB Thanachart Bank', 1);

-- Myanmar banks
INSERT INTO banks (swift_code, country_code, legal_name, abbreviations, common_names, is_active) VALUES
('AYABMMMY', 'MM', 'AYA Bank Public Company Limited', 'AYA,AYABANK,AYEYARWADY', 'AYA Bank,AYA Bank PCL,Ayeyarwady Bank,Ayeyarwady Bank Public Company Limited,AYA Bank Myanmar', 1),
('CBAYMMMY', 'MM', 'CB Bank', 'CB,CMM', 'CB Bank,Cooperative Bank', 1),
('KBZMMMMY', 'MM', 'KBZ Bank', 'KBZ,KANBAWZA', 'KBZ Bank,Kanbawza Bank', 1),
('MABBMMMY', 'MM', 'Myanmar Apex Bank', 'MAB,MABB', 'Myanmar Apex Bank', 1);

-- Malaysia banks
INSERT INTO banks (swift_code, country_code, legal_name, abbreviations, common_names, is_active) VALUES
('BBKEMYMY', 'MY', 'Malayan Banking Berhad (Maybank)', 'MBB,BBKE', 'Maybank,Malayan Banking', 1),
('CIBBMYMY', 'MY', 'CIMB Bank Berhad', 'CIMB,CIBB', 'CIMB Bank', 1),
('RHBBMYMY', 'MY', 'Hong Leong Bank', 'HLB,RHBB', 'Hong Leong Bank', 1),
('PBBEMYMY', 'MY', 'Public Bank Berhad', 'PBB,PBBE', 'Public Bank', 1),
('BIMBMYMY', 'MY', 'Bank Islam Malaysia', 'BIMB,BANKISLAM', 'Bank Islam Malaysia', 1);

-- Australia banks
INSERT INTO banks (swift_code, country_code, legal_name, abbreviations, common_names, is_active) VALUES
('NATAAU33', 'AU', 'National Australia Bank', 'NAB,NATA', 'National Australia Bank', 1),
('WPACAU2S', 'AU', 'Westpac Banking Corporation', 'WBC,WPAC', 'Westpac', 1),
('UNIAU2M1', 'AU', 'Australia and New Zealand Banking Group', 'ANZ,UNIA', 'ANZ Bank', 1),
('CTBAAU2S', 'AU', 'Commonwealth Bank of Australia', 'CBA,CTBA', 'Commonwealth Bank', 1);

-- UK banks
INSERT INTO banks (swift_code, country_code, legal_name, abbreviations, common_names, is_active) VALUES
('HBUKGB4B', 'GB', 'HSBC Bank plc', 'HSBC,HBUK', 'HSBC', 1),
('BARCGB22', 'GB', 'Barclays Bank plc', 'BARC,BARB', 'Barclays Bank', 1),
('NWBKGB2L', 'GB', 'National Westminster Bank', 'NatWest,NWBK', 'NatWest,National Westminster', 1),
('LOYDGB2L', 'GB', 'Lloyds Bank plc', 'LLOY,LOYD', 'Lloyds Bank', 1);

-- US banks
INSERT INTO banks (swift_code, country_code, legal_name, abbreviations, common_names, is_active) VALUES
('CITIUS33', 'US', 'Citibank N.A.', 'CITI', 'Citibank', 1),
('CHASUS33', 'US', 'JPMorgan Chase Bank', 'CHAS,JPM', 'JPMorgan Chase', 1),
('BOFAUS3N', 'US', 'Bank of America', 'BOA,BOFA', 'Bank of America', 1),
('WELFED6W', 'US', 'Wells Fargo Bank', 'WF,WELF', 'Wells Fargo', 1);

-- ============================================================
-- Step 4: Recreate dependent tables (for future use)
-- ============================================================

-- Note: These tables are recreated for potential future use
-- but are currently not used by the Qwen3-VL extraction pipeline

CREATE TABLE IF NOT EXISTS bank_extraction_config (
    id INT AUTO_INCREMENT PRIMARY KEY,
    bank_id INT NOT NULL,
    country_code VARCHAR(2) NOT NULL,
    default_threshold FLOAT DEFAULT 0.3,
    extraction_order JSON DEFAULT NULL,
    special_handling TEXT DEFAULT NULL,
    is_active TINYINT(1) DEFAULT 1,
    prompt_generation_status VARCHAR(50) DEFAULT 'pending',
    last_generated_at TIMESTAMP NULL,
    samples_processed INT DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE KEY idx_bank_country (bank_id, country_code),
    INDEX idx_status (prompt_generation_status),
    FOREIGN KEY (bank_id) REFERENCES banks(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS bank_gliner_prompts (
    id INT AUTO_INCREMENT PRIMARY KEY,
    bank_id INT NOT NULL,
    country_code VARCHAR(2) NOT NULL,
    entity_type VARCHAR(100) NOT NULL,
    prompt_description TEXT NOT NULL,
    entity_category VARCHAR(100) NOT NULL,
    threshold FLOAT DEFAULT 0.3,
    examples JSON DEFAULT NULL,
    validation_pattern VARCHAR(500) DEFAULT NULL,
    is_active TINYINT(1) DEFAULT 1,
    usage_count INT DEFAULT 0,
    last_used_at TIMESTAMP NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    created_by VARCHAR(100) DEFAULT 'system',
    UNIQUE KEY idx_bank_entity (bank_id, country_code, entity_type),
    INDEX idx_active (is_active),
    FOREIGN KEY (bank_id) REFERENCES banks(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS prompt_generation_history (
    id INT AUTO_INCREMENT PRIMARY KEY,
    bank_id INT NOT NULL,
    country_code VARCHAR(2) NOT NULL,
    generation_status VARCHAR(50) NOT NULL,
    llm_provider VARCHAR(50) DEFAULT NULL,
    llm_model VARCHAR(100) DEFAULT NULL,
    prompt_tokens INT DEFAULT 0,
    completion_tokens INT DEFAULT 0,
    total_tokens INT DEFAULT 0,
    generation_time_ms INT DEFAULT 0,
    error_message TEXT DEFAULT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_bank_country (bank_id, country_code),
    INDEX idx_status (generation_status),
    FOREIGN KEY (bank_id) REFERENCES banks(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

COMMIT;
SET FOREIGN_KEY_CHECKS = 1;

-- ============================================================
-- Verification queries
-- ============================================================

-- Verify total bank count
SELECT CONCAT('Total banks: ', COUNT(*)) as verification_result FROM banks;

-- Verify banks by country
SELECT country_code, COUNT(*) as bank_count FROM banks GROUP BY country_code ORDER BY country_code;

-- Verify FULLTEXT index exists
SHOW INDEX FROM banks WHERE Key_name = 'idx_search_names';

-- ============================================================
-- Summary
-- ============================================================
-- This migration consolidates the following changes:
-- 1. Simplified bank schema (3 tables → 1 banks table)
-- 2. Added comprehensive bank data for multiple countries
-- 3. Removed obsolete validator tables
-- 4. Removed bank_layout_cache table (replaced by Qwen3-VL extraction)
-- 5. Recreated dependent tables for future use
