"""
Batch Processing Service for PhotoHolmes GPU Optimization

This module provides dynamic batching capabilities to maximize GPU utilization
for PhotoHolmes forgery detection methods.

Expected Improvement: 200-300% PhotoHolmes throughput on RTX 4000
"""

import asyncio
import time
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass
import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor
import torch
from app.core.logger import get_logger
from app.core.gpu_manager import get_gpu_manager, ModelType


@dataclass
class BatchItem:
    """Individual item in a processing batch."""
    item_id: str
    image_bytes: bytes
    metadata: Optional[Dict[str, Any]] = None
    callback: Optional[Callable] = None
    timestamp: float = 0.0


@dataclass
class BatchResult:
    """Result from batch processing."""
    item_id: str
    result: Any
    processing_time: float
    success: bool
    error_message: Optional[str] = None


class BatchProcessingService:
    """
    Dynamic batch processing service for PhotoHolmes methods.

    This service groups individual image processing requests into batches
    to maximize GPU utilization while maintaining acceptable latency.
    """

    def __init__(
        self,
        max_batch_size: int = 8,
        max_wait_time: float = 0.05,  # 50ms max wait for batch formation
        target_batch_size: int = 4
    ):
        self.logger = get_logger()
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.target_batch_size = target_batch_size

        self.gpu_manager = get_gpu_manager()
        self.executor = ThreadPoolExecutor(max_workers=4)

        # Batch queues for different methods/frameworks
        self.batch_queues: Dict[str, List[BatchItem]] = {}
        self.processing_locks: Dict[str, asyncio.Lock] = {}

        # Performance metrics
        self.metrics = {
            'total_batches_processed': 0,
            'total_items_processed': 0,
            'average_batch_size': 0.0,
            'average_processing_time': 0.0,
            'gpu_utilization': 0.0
        }

        self.logger.info(f"BatchProcessingService initialized: max_batch_size={max_batch_size}, "
                        f"target_batch_size={target_batch_size}, max_wait_time={max_wait_time}s")

    async def process_item(
        self,
        method_name: str,
        item_id: str,
        image_bytes: bytes,
        processing_func: Callable,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Process a single item using dynamic batching.

        Args:
            method_name: Name of the PhotoHolmes method
            item_id: Unique identifier for this item
            image_bytes: Image data to process
            processing_func: Function to process the item
            metadata: Optional metadata for the item

        Returns:
            Processing result for the individual item
        """
        try:
            # Initialize batch queue if needed
            if method_name not in self.batch_queues:
                self.batch_queues[method_name] = []
                self.processing_locks[method_name] = asyncio.Lock()

            # Create batch item
            batch_item = BatchItem(
                item_id=item_id,
                image_bytes=image_bytes,
                metadata=metadata,
                timestamp=time.time()
            )

            # Add to batch queue
            async with self.processing_locks[method_name]:
                self.batch_queues[method_name].append(batch_item)

                # Check if we should process the batch now
                should_process = (
                    len(self.batch_queues[method_name]) >= self.target_batch_size or
                    len(self.batch_queues[method_name]) >= self.max_batch_size
                )

                if should_process:
                    # Process the batch immediately
                    batch = self.batch_queues[method_name].copy()
                    self.batch_queues[method_name].clear()
                    return await self._process_batch(method_name, batch, processing_func, item_id)
                else:
                    # Wait for timeout or more items
                    batch_result = await self._wait_for_batch_completion(method_name, processing_func, item_id)
                    return batch_result

        except Exception as e:
            self.logger.error(f"Batch processing failed for {method_name}, item {item_id}: {e}")
            raise

    async def _wait_for_batch_completion(
        self,
        method_name: str,
        processing_func: Callable,
        target_item_id: str
    ) -> Any:
        """
        Wait for batch to complete either by timeout or reaching target size.
        """
        try:
            # Wait for either timeout or target batch size
            start_time = time.time()
            remaining_time = self.max_wait_time

            while remaining_time > 0:
                await asyncio.sleep(0.001)  # Small sleep to prevent busy waiting
                current_time = time.time()
                elapsed = current_time - start_time
                remaining_time = self.max_wait_time - elapsed

                async with self.processing_locks[method_name]:
                    # Check if we should process the batch
                    if (len(self.batch_queues[method_name]) >= self.target_batch_size or
                        remaining_time <= 0):

                        batch = self.batch_queues[method_name].copy()
                        self.batch_queues[method_name].clear()
                        return await self._process_batch(method_name, batch, processing_func, target_item_id)

            # Timeout reached - process whatever we have
            async with self.processing_locks[method_name]:
                batch = self.batch_queues[method_name].copy()
                self.batch_queues[method_name].clear()
                return await self._process_batch(method_name, batch, processing_func, target_item_id)

        except Exception as e:
            self.logger.error(f"Batch wait failed for {method_name}: {e}")
            raise

    async def _process_batch(
        self,
        method_name: str,
        batch: List[BatchItem],
        processing_func: Callable,
        target_item_id: str
    ) -> Any:
        """
        Process a batch of items using the specified processing function.

        Args:
            method_name: Name of the method
            batch: List of batch items to process
            processing_func: Function to process items
            target_item_id: ID of the item we're waiting for

        Returns:
            Result for the target item
        """
        if not batch:
            raise ValueError(f"Empty batch for method {method_name}")

        start_time = time.time()
        target_result = None

        try:
            self.logger.info(f"Processing batch of {len(batch)} items for {method_name}")

            # Get GPU memory info for adaptive batch processing
            gpu_info = self.gpu_manager.get_memory_info()
            available_memory = gpu_info.get('free', 0)

            # Adjust batch size if needed based on available memory
            effective_batch_size = min(len(batch), self._calculate_optimal_batch_size(method_name, available_memory))

            # Process batch in sub-batches if necessary
            sub_batches = [batch[i:i + effective_batch_size] for i in range(0, len(batch), effective_batch_size)]

            all_results = []

            for sub_batch in sub_batches:
                # Process this sub-batch
                sub_results = await self._process_sub_batch(method_name, sub_batch, processing_func)
                all_results.extend(sub_results)

            # Find the result for our target item
            for result in all_results:
                if result.item_id == target_item_id:
                    target_result = result.result
                    break

            if target_result is None:
                raise ValueError(f"Target item {target_item_id} not found in batch results")

            # Update metrics
            processing_time = time.time() - start_time
            self._update_metrics(len(batch), processing_time)

            self.logger.info(f"Batch completed for {method_name}: {len(batch)} items in {processing_time:.3f}s")

            return target_result

        except Exception as e:
            self.logger.error(f"Batch processing failed for {method_name}: {e}")
            raise

    async def _process_sub_batch(
        self,
        method_name: str,
        sub_batch: List[BatchItem],
        processing_func: Callable
    ) -> List[BatchResult]:
        """
        Process a sub-batch of items.

        Args:
            method_name: Name of the method
            sub_batch: Sub-batch to process
            processing_func: Processing function

        Returns:
            List of batch results
        """
        results = []

        try:
            if len(sub_batch) == 1:
                # Single item - process directly
                item = sub_batch[0]
                start_time = time.time()

                try:
                    result = await asyncio.to_thread(processing_func, item.image_bytes, item.metadata)
                    processing_time = time.time() - start_time

                    results.append(BatchResult(
                        item_id=item.item_id,
                        result=result,
                        processing_time=processing_time,
                        success=True
                    ))
                except Exception as e:
                    processing_time = time.time() - start_time
                    results.append(BatchResult(
                        item_id=item.item_id,
                        result=None,
                        processing_time=processing_time,
                        success=False,
                        error_message=str(e)
                    ))

            else:
                # Multiple items - batch process
                try:
                    # Prepare batch data
                    batch_images = []
                    batch_metadata = []

                    for item in sub_batch:
                        # Convert bytes to numpy array
                        nparr = np.frombuffer(item.image_bytes, np.uint8)
                        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        if image is not None:
                            batch_images.append(image)
                            batch_metadata.append(item.metadata or {})
                        else:
                            # Handle invalid image
                            results.append(BatchResult(
                                item_id=item.item_id,
                                result=None,
                                processing_time=0.0,
                                success=False,
                                error_message="Invalid image data"
                            ))

                    if batch_images:
                        # Process the batch
                        start_time = time.time()

                        # Check if processing function supports batch processing
                        if hasattr(processing_func, 'process_batch'):
                            # Use batch processing if available
                            batch_results = await asyncio.to_thread(
                                processing_func.process_batch,
                                batch_images,
                                batch_metadata
                            )
                            processing_time = time.time() - start_time

                            # Convert batch results to individual results
                            for i, (item, result) in enumerate(zip(sub_batch[:len(batch_images)], batch_results)):
                                results.append(BatchResult(
                                    item_id=item.item_id,
                                    result=result,
                                    processing_time=processing_time / len(batch_images),  # Average time
                                    success=True
                                ))
                        else:
                            # Fallback to individual processing within the batch
                            for i, (item, image, metadata) in enumerate(zip(sub_batch[:len(batch_images)], batch_images, batch_metadata)):
                                item_start = time.time()
                                try:
                                    # Convert image back to bytes for individual processing
                                    _, buffer = cv2.imencode('.jpg', image)
                                    item_bytes = buffer.tobytes()

                                    result = await asyncio.to_thread(processing_func, item_bytes, metadata)
                                    item_processing_time = time.time() - item_start

                                    results.append(BatchResult(
                                        item_id=item.item_id,
                                        result=result,
                                        processing_time=item_processing_time,
                                        success=True
                                    ))
                                except Exception as e:
                                    item_processing_time = time.time() - item_start
                                    results.append(BatchResult(
                                        item_id=item.item_id,
                                        result=None,
                                        processing_time=item_processing_time,
                                        success=False,
                                        error_message=str(e)
                                    ))

                except Exception as e:
                    self.logger.error(f"Sub-batch processing failed for {method_name}: {e}")
                    # Create error results for all items in this sub-batch
                    for item in sub_batch:
                        results.append(BatchResult(
                            item_id=item.item_id,
                            result=None,
                            processing_time=0.0,
                            success=False,
                            error_message=f"Batch processing failed: {str(e)}"
                        ))

        except Exception as e:
            self.logger.error(f"Sub-batch processing error for {method_name}: {e}")
            raise

        return results

    def _calculate_optimal_batch_size(self, method_name: str, available_memory: int) -> int:
        """
        Calculate optimal batch size based on available GPU memory and method requirements.

        Args:
            method_name: Name of the PhotoHolmes method
            available_memory: Available GPU memory in bytes

        Returns:
            Optimal batch size
        """
        # Memory requirements per method (in MB)
        method_memory_requirements = {
            'psccnet': 300,      # PSCCNet is memory intensive
            'trufor': 200,       # TruFor moderate memory usage
            'adaptive': 150,     # Adaptive method
            'noisesniffer': 100, # NoiseSniffer lightweight
            'dq': 80,           # DQ method very lightweight
            'focal': 120,       # Focal method
            'splicebuster': 100, # Splicebuster
            'zero': 400         # ZERO method (if enabled)
        }

        base_memory_per_item = method_memory_requirements.get(method_name, 150)  # Default 150MB
        memory_per_item_mb = base_memory_per_item

        # Reserve memory for system and other processes (2GB)
        usable_memory_mb = max(512, (available_memory // (1024 * 1024)) - 2048)

        # Calculate batch size based on memory
        memory_based_batch_size = max(1, usable_memory_mb // memory_per_item_mb)

        # Consider GPU utilization
        optimal_batch_size = min(
            self.target_batch_size,
            memory_based_batch_size,
            self.max_batch_size
        )

        # Ensure at least batch size of 2 for GPU utilization if memory allows
        if usable_memory_mb > memory_per_item_mb * 2:
            optimal_batch_size = max(2, optimal_batch_size)

        return optimal_batch_size

    def _update_metrics(self, batch_size: int, processing_time: float):
        """Update performance metrics."""
        self.metrics['total_batches_processed'] += 1
        self.metrics['total_items_processed'] += batch_size

        # Update running averages
        total_items = self.metrics['total_items_processed']
        current_avg = self.metrics['average_batch_size']
        self.metrics['average_batch_size'] = ((current_avg * (total_items - batch_size)) + batch_size) / total_items

        total_batches = self.metrics['total_batches_processed']
        current_time_avg = self.metrics['average_processing_time']
        self.metrics['average_processing_time'] = ((current_time_avg * (total_batches - 1)) + processing_time) / total_batches

    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        return self.metrics.copy()

    def reset_metrics(self):
        """Reset all performance metrics."""
        self.metrics = {
            'total_batches_processed': 0,
            'total_items_processed': 0,
            'average_batch_size': 0.0,
            'average_processing_time': 0.0,
            'gpu_utilization': 0.0
        }

    async def shutdown(self):
        """Shutdown the batch processing service."""
        # Clear all batch queues
        for method_name in self.batch_queues:
            async with self.processing_locks.get(method_name, asyncio.Lock()):
                self.batch_queues[method_name].clear()

        # Shutdown executor
        self.executor.shutdown(wait=True)

        self.logger.info("BatchProcessingService shutdown complete")


# Global batch processing service instance
_batch_service: Optional[BatchProcessingService] = None


def get_batch_processing_service() -> BatchProcessingService:
    """Get the global batch processing service instance."""
    global _batch_service
    if _batch_service is None:
        _batch_service = BatchProcessingService()
    return _batch_service