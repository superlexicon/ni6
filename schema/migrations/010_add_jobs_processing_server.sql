-- Migration: add processing_server column to document_analysis_jobs
--
-- Tracks which instance owns a job in the peer job replication design:
--   NULL                     -> job belongs to this instance (created locally)
--   <origin INSTANCE_URL>    -> replicated shadow row owned by another instance;
--                               stored unprocessed until the result push arrives
ALTER TABLE document_analysis_jobs
    ADD COLUMN processing_server VARCHAR(500) NULL COMMENT 'URL of the instance that received the client request and is processing it (set on replicated shadow rows; NULL = owned locally)',
    ADD INDEX idx_processing_server (processing_server);
