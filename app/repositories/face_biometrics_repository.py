from typing import Optional, Dict, Any, List
import json
import uuid
from datetime import datetime
import numpy as np
from app.core.logger import get_logger
from app.helper.vector_helper import VectorHelper


class DuplicateFaceError(Exception):
    """Raised when attempting to insert a face that already exists."""
    pass


class FaceBiometricsRepository:
    """
    Repository for face biometric data operations.

    Embeddings are stored with multiple representations:
    1. face_embedding (JSON) - Legacy plaintext embedding (kept for compatibility)
    2. embedding_vec (VECTOR) - Native MariaDB VECTOR for HNSW indexing (fast filtering)
    3. model_name (VARCHAR) - Which face recognition model was used (CRITICAL for compatibility)
    """

    def __init__(self):
        self.logger = get_logger()

    def create_face_biometric(
        self,
        user_identity_id: str,
        face_embedding: List[float],
        model_name: str = 'deepface_vgg-face'
    ) -> Optional[str]:
        """
        Store face biometric data with VECTOR column for similarity search.

        Args:
            user_identity_id: Reference to user_identity_index.id
            face_embedding: Face embedding vector (512 floats)
            model_name: Model used for extraction (e.g., 'deepface_vgg-face', 'insightface_buffalo_l')

        Returns:
            Biometric ID if successful, None otherwise

        Raises:
            DuplicateFaceError: If similar embedding already exists (cosine distance < 0.4)
        """
        from app.core.db.database import get_db_connection_context
        try:
            with get_db_connection_context() as conn:
                # Validate embedding before storing
                if not face_embedding or len(face_embedding) == 0:
                    self.logger.error("Cannot store empty face embedding")
                    return None

                # Check for all-zero or near-zero embeddings (invalid)
                if all(abs(v) < 1e-10 for v in face_embedding[:20]):
                    self.logger.error("Cannot store invalid face embedding (all zeros)")
                    return None

                cursor = conn.cursor()
                biometric_id = str(uuid.uuid4())
                embedding_hex = VectorHelper.to_hex(face_embedding)

                sql = """
                INSERT INTO face_biometrics
                (id, user_identity_id, face_embedding, embedding_vec, model_name, created_at)
                VALUES (%s, %s, %s, UNHEX(%s), %s, %s)
                """

                cursor.execute(sql, (
                    biometric_id,
                    user_identity_id,
                    json.dumps(face_embedding),
                    embedding_hex,
                    model_name,
                    datetime.now()
                ))

                conn.commit()
                cursor.close()
                self.logger.info(f"Created face biometric {biometric_id} for user {user_identity_id[:16]}... with model {model_name}")
                return biometric_id

        except Exception as e:
            # Check if trigger rejected the insert due to cross-identity duplicate face
            if 'DUPLICATE_FACE' in str(e):
                raise DuplicateFaceError("Face already registered under a different identity")
            self.logger.error(f"Error creating face biometric: {str(e)}")
            return None

    def get_embeddings_by_user_identity(self, user_identity_id: str) -> List[List[float]]:
        """
        Get all face embeddings for a user.

        Args:
            user_identity_id: Reference to user_identity_index.id

        Returns:
            List of face embedding vectors
        """
        from app.core.db.database import get_db_connection_context
        try:
            with get_db_connection_context() as conn:
                cursor = conn.cursor(dictionary=True)

                sql = """
                SELECT face_embedding
                FROM face_biometrics
                WHERE user_identity_id = %s
                ORDER BY created_at DESC
                """
                cursor.execute(sql, (user_identity_id,))

                results = cursor.fetchall()
                cursor.close()

                embeddings = []
                for result in results:
                    if result.get('face_embedding'):
                        embeddings.append(json.loads(result['face_embedding']))

                return embeddings

        except Exception as e:
            self.logger.error(f"Error getting face embeddings: {str(e)}")
            return []


    def count_by_user_identity(self, user_identity_id: str) -> int:
        """Count face biometric records for a user"""
        from app.core.db.database import get_db_connection_context
        try:
            with get_db_connection_context() as conn:
                cursor = conn.cursor()
                sql = "SELECT COUNT(*) FROM face_biometrics WHERE user_identity_id = %s"
                cursor.execute(sql, (user_identity_id,))
                result = cursor.fetchone()
                count = result[0] if result else 0
                cursor.close()
                return count
        except Exception as e:
            self.logger.error(f"Error counting face biometrics: {str(e)}")
            return 0

    def get_embeddings_by_user_identity_ordered(
        self,
        user_identity_id: str,
        limit: Optional[int] = None,
        model_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get face embeddings for a user ordered by created_at DESC.

        Used for key recovery face matching - returns embeddings in
        declining order of creation date.

        Args:
            user_identity_id: User identity ID
            limit: Optional limit on number of embeddings to return
            model_name: Optional filter by model name (returns all if None)

        Returns:
            List of dicts with 'id', 'embedding', 'created_at', 'model_name'
        """
        from app.core.db.database import get_db_connection_context
        try:
            with get_db_connection_context() as conn:
                cursor = conn.cursor(dictionary=True)

                sql = """
                SELECT id, face_embedding, created_at, model_name
                FROM face_biometrics
                WHERE user_identity_id = %s
                """
                params = [user_identity_id]

                if model_name:
                    sql += " AND model_name = %s"
                    params.append(model_name)

                sql += " ORDER BY created_at DESC"

                if limit:
                    sql += f" LIMIT {int(limit)}"

                cursor.execute(sql, params)
                results = cursor.fetchall()
                cursor.close()

                embeddings = []
                for result in results:
                    if result.get('face_embedding'):
                        embedding = json.loads(result['face_embedding']) if isinstance(result['face_embedding'], str) else result['face_embedding']
                        embeddings.append({
                            'id': result['id'],
                            'embedding': embedding,
                            'created_at': result['created_at'],
                            'model_name': result.get('model_name', 'deepface_vgg-face')
                        })

                self.logger.info(f"Retrieved {len(embeddings)} embeddings for user_identity_id: {user_identity_id[:16]}... (model: {model_name or 'all'})")
                return embeddings

        except Exception as e:
            self.logger.error(f"Error getting face embeddings ordered: {str(e)}")
            return []

    def check_duplicate_embedding(
        self,
        face_embedding: List[float],
        distance_threshold: float = 0.4,
        model_name: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Check if a similar face embedding already exists in the database.
        Uses MariaDB HNSW index for O(log n) lookup.
        IMPORTANT: Only checks within the SAME model.

        Args:
            face_embedding: The embedding to check
            distance_threshold: Max cosine distance (0.4 = similarity > 0.6)
            model_name: Model to use for comparison (uses all if None - NOT RECOMMENDED)

        Returns:
            Dict with match info if duplicate found, None otherwise
        """
        from app.core.db.database import get_db_connection_context
        try:
            with get_db_connection_context() as conn:
                cursor = conn.cursor(dictionary=True)
                embedding_hex = VectorHelper.to_hex(face_embedding)

                sql = f"""
                SELECT id, user_identity_id, model_name, VEC_DISTANCE_COSINE(embedding_vec, x'{embedding_hex}') AS distance
                FROM face_biometrics
                WHERE VEC_DISTANCE_COSINE(embedding_vec, x'{embedding_hex}') < %s
                """
                params = [distance_threshold]

                if model_name:
                    sql += " AND model_name = %s"
                    params.append(model_name)

                sql += " ORDER BY distance LIMIT 1"

                cursor.execute(sql, params)
                result = cursor.fetchone()
                cursor.close()

                if result:
                    return {
                        'id': result['id'],
                        'user_identity_id': result['user_identity_id'],
                        'model_name': result['model_name'],
                        'distance': result['distance']
                    }
                return None

        except Exception as e:
            self.logger.error(f"Error checking duplicate embedding: {str(e)}")
            return None

    def verify_face_matches_user(
        self,
        face_embedding: List[float],
        user_identity_id: str,
        distance_threshold: float = 0.4,
        model_name: Optional[str] = None
    ) -> tuple[bool, Optional[float]]:
        """
        Verify that a face embedding matches the user's stored face.

        Uses VECTOR HNSW index for fast lookup.
        IMPORTANT: Only compares embeddings from the SAME model.

        Args:
            face_embedding: The new face embedding to verify
            user_identity_id: User identity ID to check against
            distance_threshold: Max cosine distance for match (0.4 = similarity > 0.6)
            model_name: Model to use for comparison (uses latest if None)

        Returns:
            Tuple of (is_match, distance):
            - (True, value) if face matches user's stored face
            - (False, None) if no match or user has no stored face
        """
        return self._verify_face_matches_user_vector(face_embedding, user_identity_id, distance_threshold, model_name)

    def _verify_face_matches_user_vector(
        self,
        face_embedding: List[float],
        user_identity_id: str,
        distance_threshold: float = 0.4,
        model_name: Optional[str] = None
    ) -> tuple[bool, Optional[float]]:
        """
        Verify face match using VECTOR HNSW index (fast).

        Used for selfie resubmissions to ensure the new selfie is from
        the same person who registered the identity.

        IMPORTANT: Only compares embeddings from the SAME model.
        Cross-model comparisons are invalid and produce false negatives.
        """
        from app.core.db.database import get_db_connection_context
        try:
            with get_db_connection_context() as conn:
                cursor = conn.cursor(dictionary=True)
                embedding_hex = VectorHelper.to_hex(face_embedding)

                # Build SQL with optional model filter
                sql = f"""
                SELECT id, model_name, VEC_DISTANCE_COSINE(embedding_vec, x'{embedding_hex}') AS distance
                FROM face_biometrics
                WHERE user_identity_id = %s
                """
                params = [user_identity_id]

                if model_name:
                    sql += " AND model_name = %s"
                    params.append(model_name)

                sql += " ORDER BY distance LIMIT 1"

                cursor.execute(sql, params)
                result = cursor.fetchone()
                cursor.close()

                if not result:
                    self.logger.warning(f"No face biometrics found for user: {user_identity_id[:16]}... (model: {model_name or 'any'})")
                    return (False, None)

                distance = result['distance']
                is_match = distance < distance_threshold

                self.logger.info(
                    f"Face verification for user {user_identity_id[:16]}...: "
                    f"model={result['model_name']}, distance={distance:.4f}, "
                    f"threshold={distance_threshold}, match={is_match}"
                )

                return (is_match, distance)

        except Exception as e:
            self.logger.error(f"Error verifying face match: {str(e)}")
            return (False, None)
    def find_matching_identity(
        self,
        face_embedding: List[float],
        distance_threshold: float = 0.3,
        model_name: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Find matching identity by face embedding across all users.

        Uses MariaDB HNSW index for efficient O(log n) lookup.
        IMPORTANT: Only compares embeddings from the SAME model.

        Args:
            face_embedding: Face embedding to match
            distance_threshold: Max cosine distance (default 0.3 = 70% confidence)
            model_name: Model to use for comparison (uses all if None - NOT RECOMMENDED)

        Returns:
            Dict with 'identity_id', 'distance', 'similarity', 'matched_biometric_id', 'model_name' if found,
            None otherwise
        """
        from app.core.db.database import get_db_connection_context
        try:
            with get_db_connection_context() as conn:
                cursor = conn.cursor(dictionary=True)
                embedding_hex = VectorHelper.to_hex(face_embedding)

                # Build SQL with optional model filter
                sql = f"""
                SELECT
                id AS biometric_id,
                user_identity_id AS identity_id,
                model_name,
                VEC_DISTANCE_COSINE(embedding_vec, x'{embedding_hex}') AS distance
                FROM face_biometrics
                WHERE VEC_DISTANCE_COSINE(embedding_vec, x'{embedding_hex}') < %s
                """
                params = [distance_threshold]

                if model_name:
                    sql += " AND model_name = %s"
                    params.append(model_name)

                sql += " ORDER BY distance LIMIT 1"

                cursor.execute(sql, params)
                result = cursor.fetchone()
                cursor.close()

                if result:
                    similarity = 1.0 - result['distance']
                    self.logger.info(
                        f"Found matching identity: {result['identity_id'][:16]}..., "
                        f"model={result['model_name']}, distance={result['distance']:.4f} "
                        f"(similarity={similarity*100:.1f}%)"
                    )
                    return {
                        'identity_id': result['identity_id'],
                        'distance': result['distance'],
                        'similarity': similarity,
                        'matched_biometric_id': result['biometric_id'],
                        'model_name': result['model_name']
                    }

                self.logger.info(f"No matching identity found (threshold: distance<{distance_threshold}, model: {model_name or 'any'})")
                return None

        except Exception as e:
            self.logger.error(f"Error finding matching identity: {str(e)}")
            return None

    def update_face_biometric(
        self,
        user_identity_id: str,
        face_embedding: List[float],
        model_name: str = 'deepface_vgg-face'
    ) -> Optional[str]:
        """
        Update or add a face biometric for a user (used for resubmissions).

        This method adds a new face biometric record. The existing records
        are kept for history, but the new one will be the most recent.

        Args:
            user_identity_id: User identity ID
            face_embedding: New face embedding to store
            model_name: Model used for extraction

        Returns:
            Biometric ID if successful, None otherwise
        """
        from app.core.db.database import get_db_connection_context
        try:
            with get_db_connection_context() as conn:
                # Validate embedding before storing
                if not face_embedding or len(face_embedding) == 0:
                    self.logger.error("Cannot store empty face embedding")
                    return None

                # Check for all-zero or near-zero embeddings (invalid)
                if all(abs(v) < 1e-10 for v in face_embedding[:20]):
                    self.logger.error("Cannot store invalid face embedding (all zeros)")
                    return None

                cursor = conn.cursor()
                biometric_id = str(uuid.uuid4())
                embedding_hex = VectorHelper.to_hex(face_embedding)

                sql = """
                INSERT INTO face_biometrics
                (id, user_identity_id, face_embedding, embedding_vec, model_name, created_at)
                VALUES (%s, %s, %s, UNHEX(%s), %s, %s)
                """

                cursor.execute(sql, (
                    biometric_id,
                    user_identity_id,
                    json.dumps(face_embedding),
                    embedding_hex,
                    model_name,
                    datetime.now()
                ))

                conn.commit()
                cursor.close()
                self.logger.info(f"Updated face biometric {biometric_id} for user {user_identity_id[:16]}... with model {model_name}")
                return biometric_id

        except Exception as e:
            self.logger.error(f"Error updating face biometric: {str(e)}")
            return None

    def delete_user_embeddings(self, user_identity_id: str) -> int:
        """
        Delete all face embeddings for a user.

        Used during selfie resubmission to replace old embeddings with new ones.

        Args:
            user_identity_id: User identity ID

        Returns:
            Number of embeddings deleted
        """
        from app.core.db.database import get_db_connection_context
        try:
            with get_db_connection_context() as conn:
                cursor = conn.cursor()

                sql = """
                DELETE FROM face_biometrics
                WHERE user_identity_id = %s
                """

                cursor.execute(sql, (user_identity_id,))
                deleted_count = cursor.rowcount
                conn.commit()
                cursor.close()

                if deleted_count > 0:
                    self.logger.info(
                        f"Deleted {deleted_count} face embeddings for user {user_identity_id[:16]}..."
                    )
                return deleted_count

        except Exception as e:
            self.logger.error(f"Error deleting face embeddings: {str(e)}")
            return 0

    def get_model_name_for_user(
        self,
        user_identity_id: str
    ) -> Optional[str]:
        """
        Get the model name of the most recent face embedding for a user.

        Args:
            user_identity_id: User identity ID

        Returns:
            Model name (e.g., 'deepface_vgg-face', 'insightface_buffalo_l') or None if no embeddings found
        """
        from app.core.db.database import get_db_connection_context
        try:
            with get_db_connection_context() as conn:
                cursor = conn.cursor(dictionary=True)

                sql = """
                SELECT model_name
                FROM face_biometrics
                WHERE user_identity_id = %s
                ORDER BY created_at DESC
                LIMIT 1
                """

                cursor.execute(sql, (user_identity_id,))
                result = cursor.fetchone()
                cursor.close()

                if result and result.get('model_name'):
                    return result['model_name']

                self.logger.warning(f"No model name found for user: {user_identity_id[:16]}...")
                return None

        except Exception as e:
            self.logger.error(f"Error getting model name for user: {str(e)}")
            return None
