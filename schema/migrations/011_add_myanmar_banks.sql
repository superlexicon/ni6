-- ============================================================
-- Migration 011: Add Myanmar Banks
-- Purpose: Add CB Bank and verify AYA Bank entries for Myanmar
--          banking system support
-- ============================================================

INSERT INTO banks (swift_code, country_code, legal_name, abbreviations, common_names, is_active) VALUES
-- CB Bank Myanmar
('CBANMMMY', 'MM', 'CB Bank Limited', 'CBB,CBAN', 'CB Bank,CB Bank Limited,CB Bank Myanmar', 1),
-- AYA Bank (Ayeyarwady Bank) - verify/update if needed
('AYABMMMY', 'MM', 'AYA Bank Limited', 'AYA,AYAB', 'AYA Bank,AYA Bank Limited,Ayeyarwady Bank', 1)
ON DUPLICATE KEY UPDATE
    legal_name = VALUES(legal_name),
    abbreviations = VALUES(abbreviations),
    common_names = VALUES(common_names),
    is_active = VALUES(is_active);
