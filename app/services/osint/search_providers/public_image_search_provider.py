"""
Public Image Search Provider for OSINT Background Search.

Searches DuckDuckGo for publicly available images of the applicant
and verifies them using facial recognition against their selfie.

Two-phase approach:
1. Search and match images (during selfie verification)
2. Analyze source URLs (during OSINT screening)
"""

import asyncio
import time
import hashlib
from typing import List, Optional, Dict, Any
from datetime import datetime
from ddgs import DDGS
import httpx
from app.core.logger import get_logger
from app.config.osint_config import osint_settings
from app.helper.deepface_helper import DeepfaceHelper


class PublicImageSearchProvider:
    """
    Public image search provider with facial recognition verification.

    Features:
    - DuckDuckGo image search
    - Face detection and matching using DeepFace
    - Source URL classification
    - Rate limiting and caching
    """

    def __init__(self):
        self.logger = get_logger()
        self.last_search_time = 0
        self.min_search_interval = osint_settings.min_delay_seconds
        self.cache = {}  # Simple in-memory cache

    def _get_cache_key(self, query: str) -> str:
        """Generate cache key for query."""
        return hashlib.md5(query.encode()).hexdigest()

    def _get_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get results from cache if available."""
        if cache_key in self.cache:
            cached = self.cache[cache_key]
            # Cache for 24 hours
            if (datetime.now() - cached.get('timestamp', datetime.now())).total_seconds() < 86400:
                self.logger.info(f"Cache hit for query: {cache_key[:8]}...")
                return cached.get('results')
        return None

    def _save_to_cache(self, cache_key: str, results: Dict[str, Any]):
        """Save results to cache."""
        self.cache[cache_key] = {
            'timestamp': datetime.now(),
            'results': results
        }

    async def search(
        self,
        full_name: str,
        country: Optional[str] = None,
        selfie_image_bytes: Optional[bytes] = None,
        max_results: int = 5,
        selfie_embedding: Optional[list] = None
    ) -> Dict[str, Any]:
        """
        Search for publicly available images and verify with facial recognition.

        Args:
            full_name: Person's full name
            country: Country name or code
            selfie_image_bytes: Selfie image bytes for face matching (legacy)
            max_results: Maximum number of image results to process (default: 5, reduced for faster processing)
            selfie_embedding: Pre-computed selfie embedding vector (preferred)

        Returns:
            {
                "search_queries": list,
                "images_found": int,
                "images_downloaded": int,
                "matches_found": int,
                "matched_images": list,
                "failed_downloads": int,
                "no_face_detected": int
            }
        """
        self.logger.debug(f"Starting public image search for: {full_name}")

        # Build search queries
        queries = self._build_search_queries(full_name, country)

        all_results = []
        search_queries_performed = []

        for query in queries:
            # Check cache first
            cache_key = self._get_cache_key(query)
            cached_results = self._get_from_cache(cache_key)
            if cached_results:
                all_results.extend(cached_results.get('results', []))
                search_queries_performed.append(query)
                continue

            # Perform image search with retry logic
            results = await self._search_images_with_retry(query, max_results)
            if results:
                all_results.extend(results)
                search_queries_performed.append(query)

                # Cache results
                self._save_to_cache(cache_key, {'results': results})

        # Remove duplicates by image URL
        seen_urls = set()
        unique_results = []
        for result in all_results:
            url = result.get('image', '')
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_results.append(result)

        # Process images: download, detect faces, match against selfie
        matched_page_urls = []
        matched_images = []  # Keep for backward compatibility
        failed_downloads = 0
        no_face_detected = 0
        faces_not_matched = 0  # Face detected but doesn't match selfie
        images_downloaded = 0

        # Get threshold from config
        from app.config.osint_config import osint_settings
        threshold = osint_settings.face_verified_similarity_threshold

        # Phase 1: Download all images first (for batch processing)
        downloaded_images = []  # Store with metadata for later matching

        for result in unique_results[:max_results]:
            image_url = result.get('image')
            source_url = result.get('url')

            try:
                self.logger.debug(f"Downloading image {images_downloaded + 1}: {image_url}")
                image_bytes = await self._download_image(image_url)
                if not image_bytes:
                    self.logger.warning(f"Failed to download {image_url}")
                    failed_downloads += 1
                    continue

                images_downloaded += 1
                downloaded_images.append({
                    'bytes': image_bytes,
                    'image_url': image_url,
                    'source_url': source_url,
                    'title': result.get('title', '')
                })
            except Exception as e:
                self.logger.warning(f"Failed to download image {image_url}: {str(e)}")
                failed_downloads += 1
                continue

        self.logger.info(f"Downloaded {images_downloaded} images, starting batch face detection")

        # Phase 2: Batch process with selfie_embedding (preferred)
        if selfie_embedding and downloaded_images:
            from app.deepface.deepface import DeepFace
            import numpy as np
            import io
            from PIL import Image

            def load_and_convert_image(img_bytes: bytes) -> np.ndarray:
                try:
                    with Image.open(io.BytesIO(img_bytes)) as img:
                        return np.array(img.convert("RGB"))
                except Exception as e:
                    raise Exception(f"Failed to load image: {str(e)}")

            # Convert all to numpy arrays
            img_arrays = []
            for img_data in downloaded_images:
                try:
                    img_array = load_and_convert_image(img_data['bytes'])
                    img_arrays.append(img_array)
                except Exception as e:
                    self.logger.warning(f"Failed to convert image: {str(e)}")
                    img_arrays.append(None)

            # Filter out failed conversions
            valid_indices = [i for i, arr in enumerate(img_arrays) if arr is not None]
            valid_arrays = [img_arrays[i] for i in valid_indices]

            if valid_arrays:
                # Batch extract embeddings (single call - much faster!)
                self.logger.info(f"Batch extracting embeddings for {len(valid_arrays)} images...")
                try:
                    # Add timeout to prevent hanging - max 60 seconds for batch processing
                    embeddings_batch = await asyncio.wait_for(
                        asyncio.to_thread(
                            DeepFace.represent,
                            img_path=valid_arrays,  # Pass list of arrays
                            model_name='Facenet512',
                            enforce_detection=False,
                            detector_backend='retinaface',
                            align=True
                        ),
                        timeout=60.0  # 60 second timeout for batch embedding extraction
                    )
                except asyncio.TimeoutError:
                    self.logger.warning(f"Batch embedding extraction timed out after 60 seconds - skipping image verification")
                    # Return results indicating timeout occurred
                    return {
                        "search_queries": search_queries_performed,
                        "images_found": len(unique_results),
                        "images_downloaded": images_downloaded,
                        "matches_found": 0,
                        "matched_images": [],
                        "failed_downloads": failed_downloads,
                        "faces_not_detected": no_face_detected,
                        "faces_not_matched": 0,
                        "timeout": True
                    }

                # Compare each embedding to selfie
                for batch_idx, embedding_result in enumerate(embeddings_batch):
                    original_idx = valid_indices[batch_idx]
                    img_data = downloaded_images[original_idx]
                    image_num = batch_idx + 1

                    if embedding_result and len(embedding_result) > 0:
                        face_confidence = embedding_result[0].get('face_confidence', 0.0)

                        # Skip low-confidence detections (dummy faces from enforce_detection=False)
                        if face_confidence < 0.5:
                            self.logger.warning(
                                f"[{image_num}/{len(valid_arrays)}] ✗ Low confidence face ({face_confidence:.3f}) - treating as no face"
                            )
                            no_face_detected += 1
                            continue

                        self.logger.info(
                            f"[{image_num}/{len(valid_arrays)}] Face detected: "
                            f"confidence={face_confidence:.3f}"
                        )

                        image_embedding = np.array(embedding_result[0]['embedding'])
                        selfie_embedding_array = np.array(selfie_embedding)

                        # Calculate cosine similarity
                        dot_product = np.dot(image_embedding, selfie_embedding_array)
                        norm1 = np.linalg.norm(image_embedding)
                        norm2 = np.linalg.norm(selfie_embedding_array)
                        similarity = dot_product / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0.0

                        self.logger.info(
                            f"[{image_num}/{len(valid_arrays)}] Face verification: "
                            f"similarity={similarity:.3f}, match={similarity >= threshold}, threshold={threshold}"
                        )

                        if similarity >= threshold:
                            # Face matched - record the page URL
                            source_type = self._classify_source_url(img_data['source_url'])
                            confidence = "high" if similarity >= 0.90 else "verified"

                            self.logger.info(f"[{image_num}/{len(valid_arrays)}] ✓ FACE MATCHED! {img_data['image_url']}")

                            matched_page_urls.append({
                                "page_url": img_data['source_url'],
                                "image_url": img_data['image_url'],
                                "source_type": source_type,
                                "title": img_data['title'],
                                "similarity": round(similarity, 3),
                                "confidence": confidence
                            })
                        else:
                            self.logger.debug(f"[{image_num}/{len(valid_arrays)}] ✗ Face doesn't match (similarity too low)")
                            faces_not_matched += 1
                    else:
                        self.logger.warning(f"[{image_num}/{len(valid_arrays)}] ✗ No face detected in {img_data['image_url']}")
                        no_face_detected += 1

                # Count failed conversions as no face
                no_face_detected += len(downloaded_images) - len(valid_indices)

        # Phase 3: Legacy path for selfie_image_bytes (process one-by-one)
        elif selfie_image_bytes and downloaded_images:
            from app.helper.deepface_helper import DeepfaceHelper

            for img_data in downloaded_images:
                try:
                    match_result = await DeepfaceHelper.verify_face_with_embedding(
                        stored_embedding=None,  # Will extract from selfie_image_bytes
                        verification_image=img_data['bytes'],
                        threshold=threshold,
                        model_name='Facenet512'
                    )

                    # For legacy path, we need to handle differently
                    # This path is rarely used now
                    self.logger.warning("Legacy selfie_image_bytes path not optimized for batch processing")
                except Exception as e:
                    self.logger.warning(f"Failed to process image: {str(e)}")
                    no_face_detected += 1

        # Build matched_images for backward compatibility
        matched_images = [
            {
                "image_url": m["image_url"],
                "source_url": m["page_url"],
                "source_type": m["source_type"],
                "title": m["title"],
                "similarity": m["similarity"],
                "confidence": m["confidence"]
            }
            for m in matched_page_urls
        ]

        self.logger.info(
            f"Public image search completed: "
            f"{images_downloaded} downloaded, {len(matched_page_urls)} matched, "
            f"{faces_not_matched} not matched, {no_face_detected} no face, {failed_downloads} failed"
        )

        return {
            "search_queries": search_queries_performed,
            "images_found": len(unique_results),
            "images_downloaded": images_downloaded,
            "matches_found": len(matched_page_urls),
            "matched_page_urls": matched_page_urls,  # NEW: Return page URLs
            "matched_images": matched_images,  # Keep for backward compatibility
            "failed_downloads": failed_downloads,
            "no_face_detected": no_face_detected,
            "faces_not_matched": faces_not_matched  # NEW: Faces detected but didn't match
        }

    def _build_search_queries(
        self,
        full_name: str,
        country: Optional[str]
    ) -> List[str]:
        """Build query variations for comprehensive image search."""
        queries = []

        # Basic queries
        queries.append(f'"{full_name}" photo')
        queries.append(f'"{full_name}" profile')

        # Name + country
        if country:
            queries.append(f'"{full_name}" {country}')

        return queries

    async def _search_images_with_retry(
        self,
        query: str,
        max_results: int,
        retries: int = 3
    ) -> Optional[List[Dict]]:
        """Perform image search with exponential backoff retry logic."""
        for attempt in range(retries):
            try:
                # Enforce rate limiting
                elapsed = time.time() - self.last_search_time
                if elapsed < self.min_search_interval:
                    await asyncio.sleep(self.min_search_interval - elapsed)

                with DDGS() as ddgs:
                    results = list(
                        ddgs.images(
                            query=query,
                            max_results=max_results,
                        )
                    )
                    self.last_search_time = time.time()
                    self.logger.info(f"Image search '{query}' returned {len(results)} results")
                    return results

            except Exception as e:
                self.logger.warning(f"Attempt {attempt + 1} failed for image search '{query}': {str(e)}")
                if attempt < retries - 1:
                    await asyncio.sleep(2**attempt)  # Exponential backoff
                else:
                    self.logger.error(f"All image search attempts failed for query: {query}")
                    return None

    async def _download_image(self, url: str) -> Optional[bytes]:
        """
        Download image from URL.

        Args:
            url: Image URL

        Returns:
            Image bytes or None if download failed
        """
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(url)
                response.raise_for_status()
                return response.content
        except Exception as e:
            self.logger.warning(f"Failed to download image from {url}: {str(e)}")
            return None

    async def _detect_and_match_face(
        self,
        image_bytes: bytes,
        selfie_image_bytes: Optional[bytes] = None,
        selfie_embedding: Optional[list] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Detect face in image and match against selfie using the proven helper method.

        Args:
            image_bytes: Downloaded image bytes
            selfie_image_bytes: Applicant's selfie bytes (legacy)
            selfie_embedding: Pre-computed selfie embedding vector (preferred)

        Returns:
            {
                "face_detected": bool,
                "similarity": float (0-1) if face detected
            }
        """
        try:
            # Use the same implementation as passport/selfie matching
            if selfie_embedding:
                # This is the working pattern from deepface_helper.py:639
                match_result = await DeepfaceHelper.verify_face_with_embedding(
                    stored_embedding=selfie_embedding,
                    verification_image=image_bytes,  # Web image bytes
                    threshold=0.31,  # Cosine distance threshold for Facenet512
                    model_name='Facenet512'
                )

                # Convert FaceMatchResult to expected format
                return {
                    "face_detected": True,
                    "similarity": round(match_result.match_confidence, 3)  # Confidence = similarity
                }

            # Legacy fallback for selfie_image_bytes
            if selfie_image_bytes:
                from app.deepface.deepface import DeepFace
                import numpy as np
                import io
                from PIL import Image

                def load_and_convert_image(img_bytes: bytes) -> np.ndarray:
                    try:
                        with Image.open(io.BytesIO(img_bytes)) as img:
                            return np.array(img.convert("RGB"))
                    except Exception as e:
                        raise Exception(f"Failed to load image: {str(e)}")

                img_array = load_and_convert_image(image_bytes)

                try:
                    embedding_result = await asyncio.to_thread(
                        DeepFace.represent,
                        img_path=img_array,
                        model_name='Facenet512',
                        enforce_detection=False,
                        detector_backend='retinaface',
                        align=True
                    )
                except Exception as e:
                    self.logger.warning(f"Face embedding extraction failed: {str(e)}")
                    return {"face_detected": False, "similarity": 0.0}

                if not embedding_result or len(embedding_result) == 0:
                    self.logger.warning("No face embedding result returned")
                    return {"face_detected": False, "similarity": 0.0}

                image_embedding = np.array(embedding_result[0]['embedding'])
                selfie_img_array = load_and_convert_image(selfie_image_bytes)
                selfie_embedding_result = await asyncio.to_thread(
                    DeepFace.represent,
                    img_path=selfie_img_array,
                    model_name='Facenet512',
                    enforce_detection=False,
                    detector_backend='retinaface',
                    align=True
                )

                if not selfie_embedding_result:
                    self.logger.error("Failed to generate embedding from selfie")
                    return {"face_detected": False, "similarity": 0.0}

                selfie_embedding_array = np.array(selfie_embedding_result[0]['embedding'])
                similarity = self._cosine_similarity(image_embedding, selfie_embedding_array)
                return {
                    "face_detected": True,
                    "similarity": round(similarity, 3)
                }

            return {"face_detected": False, "similarity": 0.0}

        except Exception as e:
            self.logger.warning(f"Face detection/matching failed: {str(e)}")
            return {"face_detected": False, "similarity": 0.0}

    def _cosine_similarity(self, vec1, vec2) -> float:
        """Calculate cosine similarity between two vectors."""
        import numpy as np
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        return dot_product / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0.0

    def _classify_source_url(self, url: str) -> str:
        """
        Classify source type from URL pattern.

        Returns:
            'social_media', 'company_website', 'news', 'forum', or 'other'
        """
        url_lower = url.lower()

        # Social media
        if 'linkedin.com/in/' in url_lower:
            return 'social_media'
        elif 'facebook.com/' in url_lower:
            return 'social_media'
        elif 'x.com/' in url_lower or 'twitter.com/' in url_lower:
            return 'social_media'
        elif 'instagram.com/' in url_lower:
            return 'social_media'

        # Company websites
        elif '/team/' in url_lower or '/about/' in url_lower:
            return 'company_website'

        # News sites
        elif any(news in url_lower for news in ['cnn', 'bbc', 'reuters', 'bloomberg', 'news']):
            return 'news'

        # Forums
        elif any(forum in url_lower for forum in ['reddit', 'quora', 'stackexchange']):
            return 'forum'

        else:
            return 'other'


# Global instance
public_image_search_provider = PublicImageSearchProvider()
