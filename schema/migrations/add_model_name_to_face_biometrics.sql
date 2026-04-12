-- Migration: Add model_name to face_biometrics table
-- Version: 1.9.0
-- Date: 2026-04-12
-- Description: Track which face recognition model was used for each embedding
-- Reason: Embeddings from different models (DeepFace vs InsightFace) are incompatible
--          for comparison. We must track the model name to ensure we only compare
--          embeddings from the same model.

-- ===============================================
-- Step 1: Add model_name column
-- ===============================================
ALTER TABLE face_biometrics
ADD COLUMN model_name VARCHAR(100) NOT NULL DEFAULT 'deepface_vgg-face' COMMENT 'Face recognition model used (deepface_vgg-face, insightface_buffalo_l)';

-- ===============================================
-- Step 2: Add index for model-aware queries
-- ===============================================
CREATE INDEX idx_model_name ON face_biometrics(model_name);

-- ===============================================
-- Step 3: Update existing records (assume all are DeepFace)
-- ===============================================
UPDATE face_biometrics
SET model_name = 'deepface_vgg-face'
WHERE model_name IS NULL OR model_name = '';

-- ===============================================
-- Step 4: Update duplicate detection trigger to check same model only
-- ===============================================
DROP TRIGGER IF EXISTS trg_face_biometrics_cross_identity_check;

DELIMITER //
CREATE TRIGGER trg_face_biometrics_cross_identity_check
BEFORE INSERT ON face_biometrics
FOR EACH ROW
BEGIN
    DECLARE existing_identity VARCHAR(36);

    -- Check if this face matches any existing face under a DIFFERENT identity
    -- IMPORTANT: Only compare embeddings from the SAME model
    -- Cross-model comparisons are invalid and produce false negatives
    SELECT user_identity_id INTO existing_identity
    FROM face_biometrics
    WHERE user_identity_id != NEW.user_identity_id
      AND model_name = NEW.model_name  -- SAME MODEL ONLY
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
-- Step 5: Update composite index for model-aware queries
-- ===============================================
-- Add composite index for efficient model + user queries
CREATE INDEX idx_model_user ON face_biometrics(model_name, user_identity_id);
