"""
Face Matching Service - Shared face matching logic with cosine similarity.

This service extracts duplicated face matching logic from KeyRecoveryService
and provides a reusable implementation for all services that need to match
faces against stored embeddings.

Uses cosine distance for matching:
- Cosine similarity = dot(embedding1, embedding2) / (norm1 * norm2)
- Cosine distance = 1 - cosine_similarity
- Lower distance = more similar faces (0 = identical, 1 = completely different)

Threshold is configurable via confidence percentage:
- Format: Confidence percentage (0-100)
- Conversion: cosine_distance = 1 - (confidence_percent / 100)
- Example: 70% confidence → distance threshold 0.3
"""

from typing import Dict, List, Optional
import numpy as np
from app.core.logger import get_logger


class FaceMatchingService:
    """
    Reusable face matching with cosine similarity.

    Eliminates code duplication across KeyRecoveryService, KeyManagementService,
    and other services that need face matching.
    """

    def __init__(self, confidence_threshold: float = 70.0):
        """
        Initialize face matching service with confidence threshold.

        Args:
            confidence_threshold: Face match confidence percentage (0-100)
                Higher = stricter matching (e.g., 80% = more strict than 70%)
        """
        self.logger = get_logger()

        # Convert confidence percentage to cosine distance threshold
        # confidence% = (1 - distance) * 100, so distance = 1 - (confidence / 100)
        self.face_match_threshold = 1.0 - (confidence_threshold / 100.0)
        self.face_match_confidence_threshold = confidence_threshold

        self.logger.info(
            f"FaceMatchingService initialized with threshold: "
            f"{self.face_match_confidence_threshold}% confidence "
            f"(cosine distance: {self.face_match_threshold:.3f})"
        )

    async def match_embedding(
        self,
        new_embedding: List[float],
        stored_embeddings: List[Dict],
        stop_on_first_match: bool = True
    ) -> Dict:
        """
        Match new embedding against stored embeddings using cosine similarity.

        Checks embeddings in order (newest first) and returns on first match
        if stop_on_first_match is True.

        Args:
            new_embedding: Face embedding from new image (list of floats)
            stored_embeddings: List of stored embeddings ordered by created_at DESC
                Each dict must have: 'id', 'embedding', 'created_at'
            stop_on_first_match: If True, return immediately on first match
                If False, check all embeddings and return best match

        Returns:
            Dict with:
            - match_found: bool
            - best_similarity: float (cosine similarity, higher = more similar)
            - matched_date: datetime of matched embedding (None if no match)
            - embeddings_checked: int
        """
        best_similarity = 0.0
        matched_date = None
        embeddings_checked = 0

        # Convert to numpy array for efficient computation
        new_embedding_array = np.array(new_embedding, dtype=np.float64)
        new_norm = np.linalg.norm(new_embedding_array)

        if new_norm == 0:
            self.logger.error("New embedding has zero magnitude")
            return {
                'match_found': False,
                'best_similarity': 0.0,
                'matched_date': None,
                'embeddings_checked': 0
            }

        for stored in stored_embeddings:
            embeddings_checked += 1
            stored_embedding = stored['embedding']
            stored_array = np.array(stored_embedding, dtype=np.float64)
            stored_norm = np.linalg.norm(stored_array)

            if stored_norm == 0:
                self.logger.warning(f"Stored embedding {stored['id']} has zero magnitude, skipping")
                continue

            # Calculate cosine similarity (1 = identical, 0 = orthogonal)
            cosine_similarity = np.dot(new_embedding_array, stored_array) / (new_norm * stored_norm)

            # Convert to distance for threshold comparison
            cosine_distance = 1 - cosine_similarity

            self.logger.debug(
                f"Embedding {embeddings_checked}: similarity={cosine_similarity:.4f} ({cosine_similarity*100:.1f}%), "
                f"distance={cosine_distance:.4f}, date={stored['created_at']}"
            )

            # Track best similarity
            if cosine_similarity > best_similarity:
                best_similarity = cosine_similarity

            # Check if match (distance below threshold)
            if cosine_distance < self.face_match_threshold:
                self.logger.info(
                    f"Match found at embedding {embeddings_checked}! "
                    f"Similarity: {cosine_similarity*100:.1f}% ({cosine_similarity:.4f}), "
                    f"distance: {cosine_distance:.4f}, "
                    f"threshold: {self.face_match_confidence_threshold}% ({self.face_match_threshold:.3f})"
                )
                return {
                    'match_found': True,
                    'best_similarity': cosine_similarity,
                    'matched_date': stored['created_at'],
                    'embeddings_checked': embeddings_checked
                }

            # If not stopping on first match, continue checking to find best
            if not stop_on_first_match and cosine_distance < self.face_match_threshold:
                # Update best match if this is better
                if matched_date is None or cosine_similarity > best_similarity:
                    best_similarity = cosine_similarity
                    matched_date = stored['created_at']

        # No match found after checking all embeddings
        return {
            'match_found': False,
            'best_similarity': best_similarity,
            'matched_date': matched_date,
            'embeddings_checked': embeddings_checked
        }

    def get_threshold_info(self) -> Dict:
        """
        Get current threshold configuration.

        Returns:
            Dict with confidence_percentage and cosine_distance_threshold
        """
        return {
            'confidence_percentage': self.face_match_confidence_threshold,
            'cosine_distance_threshold': self.face_match_threshold
        }
