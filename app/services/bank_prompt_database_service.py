"""
Bank Prompt Database Service

Manages bank-specific GLiNER2 prompts in database.
Provides CRUD operations for prompt storage and retrieval.
"""

import json
from typing import Optional, Dict, Any, List
from datetime import datetime

from app.core.db.database import get_db_connection
from app.core.logger import get_logger


logger = get_logger()


class BankPromptDatabaseService:
    """Manages bank-specific GLiNER2 prompts in database."""

    def get_bank_prompts(
        self,
        bank_id: int,
        country_code: str
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve bank-specific prompts from database.

        Args:
            bank_id: Bank ID
            country_code: ISO country code

        Returns:
            GLiNER2 schema dictionary or None if not found.
            Format:
            {
                "bank_id": int,
                "country_code": str,
                "default_threshold": float,
                "prompts": {
                    "entity_type": {
                        "description": str,
                        "entity": str,
                        "threshold": float,
                        "examples": list,
                        "pattern": str
                    }
                }
            }
        """
        conn = get_db_connection()
        try:
            cursor = conn.cursor(dictionary=True)

            # First check if config exists and is active
            config_query = """
                SELECT default_threshold, is_active, prompt_generation_status
                FROM bank_extraction_config
                WHERE bank_id = %s AND country_code = %s
                LIMIT 1
            """
            cursor.execute(config_query, (bank_id, country_code.upper()))
            config_result = cursor.fetchone()

            if not config_result or not config_result['is_active']:
                logger.debug(
                    f"No active config found for bank_id={bank_id}, country={country_code}"
                )
                return None

            # Get all prompts for this bank/country
            prompt_query = """
                SELECT entity_type, prompt_description, entity_category,
                       threshold, examples, validation_pattern
                FROM bank_gliner_prompts
                WHERE bank_id = %s AND country_code = %s AND is_active = 1
                ORDER BY entity_type
            """
            cursor.execute(prompt_query, (bank_id, country_code.upper()))
            prompt_results = cursor.fetchall()

            if not prompt_results:
                logger.debug(
                    f"No prompts found for bank_id={bank_id}, country={country_code}"
                )
                return None

            # Convert to GLiNER2 schema format
            prompts = {}
            for p in prompt_results:
                prompts[p['entity_type']] = {
                    'description': p['prompt_description'],
                    'entity': p['entity_category'],
                    'threshold': float(p['threshold']),
                    'examples': json.loads(p['examples']) if p['examples'] else [],
                    'pattern': p['validation_pattern']
                }

            logger.info(
                f"Found {len(prompts)} custom prompts for bank_id={bank_id}, country={country_code}"
            )

            return {
                'bank_id': bank_id,
                'country_code': country_code,
                'default_threshold': float(config_result['default_threshold']),
                'prompts': prompts
            }

        finally:
            conn.close()

    def save_bank_prompts(
        self,
        bank_id: int,
        country_code: str,
        prompts: List[Dict[str, Any]],
        extraction_config: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Save or update bank-specific prompts in database.

        Inserts into bank_gliner_prompts and bank_extraction_config tables.
        Updates existing prompts if they exist.

        Args:
            bank_id: Bank ID
            country_code: ISO country code
            prompts: List of prompt configurations
            extraction_config: Extraction configuration
            metadata: Optional generation metadata

        Returns:
            True if save successful, False otherwise
        """
        conn = get_db_connection()
        try:
            cursor = conn.cursor(dictionary=True)

            country_code = country_code.upper()

            # Start transaction
            conn.start_transaction()

            # Step 1: Save extraction config
            # Handle empty strings for JSON columns (convert to NULL)
            special_handling = extraction_config.get('special_handling')
            if isinstance(special_handling, str) and not special_handling.strip():
                special_handling = None

            config_query = """
                INSERT INTO bank_extraction_config
                    (bank_id, country_code, default_threshold, extraction_order,
                     special_handling, is_active, prompt_generation_status,
                     last_generated_at, samples_processed)
                VALUES (%s, %s, %s, %s, %s, %s, %s, NOW(), %s)
                ON DUPLICATE KEY UPDATE
                    default_threshold = VALUES(default_threshold),
                    extraction_order = VALUES(extraction_order),
                    special_handling = VALUES(special_handling),
                    prompt_generation_status = VALUES(prompt_generation_status),
                    last_generated_at = VALUES(last_generated_at),
                    samples_processed = VALUES(samples_processed) + samples_processed
            """
            cursor.execute(config_query, (
                extraction_config['bank_id'],
                extraction_config['country_code'],
                extraction_config['default_threshold'],
                extraction_config.get('extraction_order'),
                special_handling,
                extraction_config.get('is_active', 1),
                extraction_config.get('prompt_generation_status', 'completed'),
                extraction_config.get('samples_processed', 1)
            ))

            # Step 2: Save prompts (replace existing)
            # First delete old prompts for this bank/country
            delete_query = """
                DELETE FROM bank_gliner_prompts
                WHERE bank_id = %s AND country_code = %s
            """
            cursor.execute(delete_query, (bank_id, country_code))

            # Insert new prompts
            prompt_query = """
                INSERT INTO bank_gliner_prompts
                    (bank_id, country_code, entity_type, prompt_description,
                     entity_category, threshold, examples, validation_pattern,
                     is_active, created_by)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            prompt_values = []
            for prompt in prompts:
                prompt_values.append((
                    prompt['bank_id'],
                    prompt['country_code'],
                    prompt['entity_type'],
                    prompt['prompt_description'],
                    prompt['entity_category'],
                    prompt['threshold'],
                    prompt.get('examples'),
                    prompt.get('validation_pattern'),
                    prompt.get('is_active', 1),
                    prompt.get('created_by', 'llm_auto_generated')
                ))

            cursor.executemany(prompt_query, prompt_values)

            # Step 3: Save generation history if metadata provided
            if metadata:
                history_query = """
                    INSERT INTO prompt_generation_history
                        (bank_id, country_code, generation_status, llm_provider,
                         llm_model, prompt_tokens, completion_tokens, total_tokens,
                         generation_time_ms)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """
                cursor.execute(history_query, (
                    bank_id,
                    country_code,
                    'completed' if not metadata.get('error') else 'failed',
                    metadata.get('llm_provider'),
                    metadata.get('llm_model'),
                    metadata.get('prompt_tokens'),
                    metadata.get('completion_tokens'),
                    metadata.get('total_tokens'),
                    metadata.get('generation_time_ms')
                ))

            # Commit transaction
            conn.commit()

            logger.info(
                f"Saved {len(prompts)} prompts for bank_id={bank_id}, country={country_code}"
            )

            return True

        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to save prompts for bank_id={bank_id}: {str(e)}")
            return False

        finally:
            conn.close()

    def check_prompts_exist(
        self,
        bank_id: int,
        country_code: str
    ) -> bool:
        """
        Check if prompts exist for this bank/country combination.

        Args:
            bank_id: Bank ID
            country_code: ISO country code

        Returns:
            True if prompts exist and are active
        """
        conn = get_db_connection()
        try:
            cursor = conn.cursor(dictionary=True)

            query = """
                SELECT COUNT(*) as count
                FROM bank_gliner_prompts
                WHERE bank_id = %s AND country_code = %s AND is_active = 1
            """
            cursor.execute(query, (bank_id, country_code.upper()))
            result = cursor.fetchone()

            return result['count'] > 0

        finally:
            conn.close()

    def update_usage_stats(
        self,
        bank_id: int,
        country_code: str
    ) -> None:
        """
        Update usage_count and last_used_at for prompts.

        Args:
            bank_id: Bank ID
            country_code: ISO country code
        """
        conn = get_db_connection()
        try:
            cursor = conn.cursor(dictionary=True)

            # Start transaction for write operation
            conn.start_transaction()

            query = """
                UPDATE bank_gliner_prompts
                SET usage_count = usage_count + 1,
                    last_used_at = NOW()
                WHERE bank_id = %s AND country_code = %s AND is_active = 1
            """
            cursor.execute(query, (bank_id, country_code.upper()))
            conn.commit()

            logger.debug(f"Updated usage stats for bank_id={bank_id}, country={country_code}")

        except Exception as e:
            try:
                conn.rollback()
            except:
                pass
            logger.error(f"Failed to update usage stats: {str(e)}")

        finally:
            conn.close()

    def get_all_banks_with_prompts(
        self,
        is_active: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Get all banks that have prompts.

        Args:
            is_active: Filter for active prompts only

        Returns:
            List of banks with prompt counts
        """
        conn = get_db_connection()
        try:
            cursor = conn.cursor(dictionary=True)

            # Updated to use new schema column names:
            # - abbrev → abbreviations (first entry)
            # - full_name → legal_name
            query = """
                SELECT b.id,
                       SUBSTRING_INDEX(b.abbreviations, ',', 1) as abbrev,
                       b.legal_name as full_name,
                       bgp.country_code,
                       COUNT(bgp.id) as prompt_count,
                       MAX(bgp.last_used_at) as last_used,
                       SUM(bgp.usage_count) as total_usage
                FROM banks b
                JOIN bank_gliner_prompts bgp ON b.id = bgp.bank_id
                WHERE b.is_active = 1
                  AND (NOT %s OR bgp.is_active = 1)
                GROUP BY b.id, abbrev, full_name, bgp.country_code
                ORDER BY total_usage DESC, last_used DESC
            """
            cursor.execute(query, (is_active,))
            results = cursor.fetchall()

            return results

        finally:
            conn.close()

    def get_prompt_statistics(
        self,
        bank_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get statistics about prompts.

        Args:
            bank_id: Optional bank ID to filter by

        Returns:
            Statistics dictionary
        """
        conn = get_db_connection()
        try:
            cursor = conn.cursor(dictionary=True)

            # Get total counts
            if bank_id:
                count_query = """
                    SELECT COUNT(*) as total_prompts,
                           SUM(usage_count) as total_usage,
                           AVG(usage_count) as avg_usage,
                           COUNT(DISTINCT country_code) as country_count
                    FROM bank_gliner_prompts
                    WHERE bank_id = %s AND is_active = 1
                """
                cursor.execute(count_query, (bank_id,))
            else:
                count_query = """
                    SELECT COUNT(*) as total_prompts,
                           SUM(usage_count) as total_usage,
                           AVG(usage_count) as avg_usage,
                           COUNT(DISTINCT bank_id) as bank_count,
                           COUNT(DISTINCT country_code) as country_count
                    FROM bank_gliner_prompts
                    WHERE is_active = 1
                """
                cursor.execute(count_query)

            stats = cursor.fetchone()

            # Get entity type breakdown
            entity_query = """
                SELECT entity_type, COUNT(*) as count, AVG(threshold) as avg_threshold
                FROM bank_gliner_prompts
                WHERE is_active = 1
            """
            params = []
            if bank_id:
                entity_query += " AND bank_id = %s"
                params.append(bank_id)

            entity_query += " GROUP BY entity_type ORDER BY count DESC"
            cursor.execute(entity_query, params)
            entity_breakdown = cursor.fetchall()

            return {
                "total_prompts": stats['total_prompts'] or 0,
                "total_usage": stats['total_usage'] or 0,
                "avg_usage": float(stats['avg_usage']) if stats['avg_usage'] else 0,
                "bank_count": stats.get('bank_count', bank_id),
                "country_count": stats['country_count'] or 0,
                "entity_breakdown": entity_breakdown
            }

        finally:
            conn.close()


# Global service instance
bank_prompt_database_service = BankPromptDatabaseService()
