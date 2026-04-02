import asyncio
import io
import time
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
import torch
import threading
from contextlib import asynccontextmanager
from PIL import Image

from app.utils import ModelSummary
from app.services.batch_processing_service import get_batch_processing_service
from app.config.photoholmes_config import photoholmes_settings

from app.dto import (
    PhotoHolmesResults,
    DQMethodData,
    AdaptiveMethodData,
    NoiseSnifferData,
    PsccnetMethodData,
    CatnetMethodData,
    # ExifAsLanguageData,  # Temporarily disabled due to transformers version conflicts
    FocalMethodData,
    SplicebusterMethodData,
    TruforMethodData,
    ZeroMethodData
)
from app.core.logger import get_logger
from app.core.device_capabilities import get_device_capabilities
from app.core.shared_exif_extractor import get_shared_exif_extractor
from app.core.forgery_photoshop.forgery.dq_method import DQMethod
from app.core.forgery_photoshop.forgery.adaptive_method import AdaptiveMethod
from app.core.forgery_photoshop.photoshopped.noisesniffer_method import NoiseSnifferMethod
from app.core.forgery_photoshop.photoshopped.psccnet_method import PsccnetMethod
# from app.core.forgery_photoshop.photoshopped.imdlbenco_catnet_method import IMDLBenCoCatnetMethod  # Disabled per user request
# from app.core.forgery_photoshop.photoshopped.exif_as_language_method import ExifAsLanguageMethod  # Temporarily disabled due to transformers version conflicts
from app.core.forgery_photoshop.photoshopped.focal_method import FocalMethod  # Enabled despite ml_dtypes compatibility issue
from app.core.forgery_photoshop.photoshopped.splicebuster_method import SplicebusterMethod  # Enabled - onnxruntime dependency resolved
from app.core.forgery_photoshop.photoshopped.trufor_method import TruforMethod  # Enabled - onnxruntime dependency resolved
from app.core.forgery_photoshop.photoshopped.zero_method import ZeroMethod  # Enabled - onnxruntime dependency resolved

logger = get_logger()


class ComprehensivePhotoHolmesService:
    """
    Comprehensive PhotoHolmes service that runs all available methods and returns detailed scores.

    Implemented as a singleton to prevent multiple instances from loading the same heavy models.
    """

    _instance: Optional['ComprehensivePhotoHolmesService'] = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        """Create or return the singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, enable_zero_method: bool = False):
        # Only initialize once
        if self._initialized:
            logger.debug(f"ComprehensivePhotoHolmesService already initialized, returning singleton instance")
            return

        logger.info(f"Initializing ComprehensivePhotoHolmesService singleton (enable_zero_method={enable_zero_method})")
        self.logger = logger
        self.enable_zero_method = enable_zero_method

        # Get device capabilities for intelligent model selection
        self.device_caps = get_device_capabilities()
        self.recommendations = self.device_caps.get_recommended_models()

        # Initialize model summary for comprehensive reporting
        self.model_summary = ModelSummary()

        # Initialize shared EXIF extractor
        self.exif_extractor = get_shared_exif_extractor()

        # Store document type context for potential future use
        self.current_document_type = None

        # Initialize framework-specific memory locks for granular concurrent processing
        # Note: Locks created lazily to avoid event loop binding issues
        self._framework_locks = None
        self._locks_loop = None

        # Framework mapping for each method
        # Note: Splicebuster, TruFor, and ZERO are PyTorch-based, not ONNX-based
        self.method_frameworks = {
            'dq': 'pytorch',
            'adaptive': 'pytorch',
            'noisesniffer': 'pytorch',
            'psccnet': 'pytorch',
            'focal': 'pytorch',
            'splicebuster': 'pytorch',  # Fixed: was incorrectly marked as 'onnx'
            'trufor': 'pytorch',        # Fixed: was incorrectly marked as 'onnx'
            'zero': 'pytorch',          # Fixed: was incorrectly marked as 'onnx'
            'exif_as_language': 'pytorch'
        }

        # Initialize batch processing service for GPU optimization
        self.batch_service = get_batch_processing_service()
        self.logger.info("Batch processing service initialized for dynamic GPU optimization")

        self.logger.info(f"Device detected: {self.device_caps.get_capability_summary()}")
        self.logger.info(f"Model recommendations: {self.recommendations}")

        # Log ZERO method status
        if self.enable_zero_method:
            self.logger.info("ZERO method enabled")
        else:
            self.logger.info("ZERO method disabled for performance - will be optimized separately")

        # Initialize methods using smart loading order
        self.methods = {}
        loaded_methods, failed_methods, skipped_methods = self._load_methods_with_smart_order()

        # Print comprehensive startup summary
        device_info = {
            'device': self.device_caps.capabilities.get('device', 'Unknown'),
            'gpu_memory_gb': self.device_caps.capabilities.get('gpu_memory_gb', 0),
            'system_memory_gb': self.device_caps.capabilities.get('system_memory_gb', 0),
            'cpu_cores': self.device_caps.capabilities.get('cpu_cores', 0),
            'has_mps': self.device_caps.capabilities.get('has_mps', False),
            'has_cuda': self.device_caps.capabilities.get('has_cuda', False)
        }
        self.model_summary.print_startup_summary(loaded_methods, failed_methods, skipped_methods, device_info, self.recommendations)

        # Mark as initialized to prevent re-initialization
        self._initialized = True

    @property
    def framework_locks(self):
        """
        Lazily create asyncio locks bound to the current event loop.
        This prevents "Lock bound to different event loop" errors.
        """
        current_loop = None
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop - locks will be created when actually needed
            pass

        # Recreate locks if we're in a different event loop
        if self._framework_locks is None or self._locks_loop != current_loop:
            self._locks_loop = current_loop
            self._framework_locks = {
                'pytorch': asyncio.Lock(),
                'tensorflow': asyncio.Lock(),
                'onnx': asyncio.Lock(),
                'general': asyncio.Lock()
            }
            if current_loop:
                self.logger.debug(f"Created framework locks for event loop {id(current_loop)}")

        return self._framework_locks

    def _load_methods_with_smart_order(self) -> Tuple[List[str], List[str], List[str]]:
        """
        Load PhotoHolmes methods in an optimized order to maximize GPU memory efficiency.

        Loading strategy:
        1. Memory-intensive models first (when GPU memory is fresh)
        2. Moderate complexity models next
        3. Lightweight models last
        4. Graceful fallback on loading failures
        """
        self.logger.info("Loading PhotoHolmes methods using smart memory-efficient order...")

        # Define smart loading order with memory requirements and error handling
        loading_order = [
            # Tier 1: Memory-intensive models (load first when memory is fresh)
            ('psccnet', 'PSCCNet', lambda: PsccnetMethod(), 'memory_intensive'),
            ('focal', 'FOCAL', lambda: FocalMethod(), 'memory_intensive'),

            # Tier 2: Moderate complexity models
            ('adaptive', 'Adaptive', lambda: AdaptiveMethod(), 'moderate'),

            # Tier 3: ONNX-based models (can share memory efficiently)
            ('trufor', 'TruFor', lambda: TruforMethod(), 'onnx_based'),
            ('splicebuster', 'Splicebuster', lambda: SplicebusterMethod(), 'onnx_based'),

            # Tier 4: Lightweight models (load last)
            ('dq', 'DQ', lambda: DQMethod(), 'lightweight'),
            ('noisesniffer', 'NoiseSniffer', lambda: NoiseSnifferMethod(), 'lightweight'),

            # Tier 5: Optional ZERO method (if enabled)
            *([('zero', 'ZERO', lambda: ZeroMethod(), 'optional')] if self.enable_zero_method else []),

            # Tier 6: EXIF method (temporarily disabled)
            # ('exif_as_language', 'EXIF As Language', lambda: ExifAsLanguageMethod(), 'disabled'),
        ]

        successful_loads = 0
        failed_loads = 0
        failed_method_names = []
        skipped_method_names = []

        for key, display_name, factory, memory_tier in loading_order:
            try:
                # Check if method is recommended for this device
                if key == 'psccnet' and not self.recommendations['cnet']:
                    self.logger.info(f"Skipping {display_name} - not recommended for device")
                    skipped_method_names.append(key)
                    continue

                if key == 'focal' and not self.recommendations['focal']:
                    self.logger.info(f"Skipping {display_name} - not recommended for device")
                    skipped_method_names.append(key)
                    continue

                if key in ['adaptive'] and not self.recommendations['adaptive']:
                    self.logger.info(f"Skipping {display_name} - not recommended for device")
                    skipped_method_names.append(key)
                    continue

                if key in ['splicebuster', 'trufor', 'zero'] and not self.recommendations['doctr']:
                    self.logger.info(f"Skipping {display_name} - Doctr not available")
                    skipped_method_names.append(key)
                    continue

                # Skip TruFor on MPS - too slow, requires CUDA for practical performance
                if key == 'trufor' and self.device_caps.capabilities.get('has_mps', False) and not self.device_caps.capabilities.get('has_cuda', False):
                    self.logger.info(f"Skipping {display_name} - requires CUDA (MPS too slow)")
                    skipped_method_names.append(key)
                    continue

                if key in ['dq'] and not self.recommendations['dq']:
                    self.logger.info(f"Skipping {display_name} - DQ not recommended")
                    skipped_method_names.append(key)
                    continue

                if key in ['noisesniffer'] and not self.recommendations['noise_sniffer']:
                    self.logger.info(f"Skipping {display_name} - NoiseSniffer not recommended")
                    skipped_method_names.append(key)
                    continue

                # Monitor memory before loading
                before_memory = self._get_gpu_memory_info()

                self.logger.info(f"Loading {display_name} ({memory_tier})...")

                # Load the method
                method_instance = factory()
                self.methods[key] = method_instance

                # Monitor memory after loading
                after_memory = self._get_gpu_memory_info()
                memory_diff = after_memory['allocated'] - before_memory['allocated']

                self.logger.info(f"✅ {display_name} loaded successfully (+{memory_diff:.2f}GB)")
                successful_loads += 1

                # Aggressive cleanup after memory-intensive models
                if memory_tier == 'memory_intensive':
                    self._cleanup_gpu_memory(aggressive=False)  # Light cleanup to free fragmentation

            except Exception as e:
                self.logger.error(f"❌ Failed to load {display_name}: {str(e)}")
                failed_loads += 1
                failed_method_names.append(key)

                # Continue loading other methods - don't let one failure stop all
                continue

        self.logger.info(f"Smart model loading completed: {successful_loads} successful, {failed_loads} failed")

        # Log final memory state
        final_memory = self._get_gpu_memory_info()
        self.logger.info(f"Final memory state: {final_memory['allocated']:.2f}GB allocated, {final_memory['cached']:.2f}GB cached")

        # Log enabled methods
        enabled_methods = list(self.methods.keys())
        self.logger.info(f"Comprehensive PhotoHolmes Service initialized with {len(enabled_methods)} methods: {enabled_methods}")

        # Log disabled methods with reasons
        all_possible_methods = ['dq', 'adaptive', 'noisesniffer', 'psccnet',  # 'exif_as_language',
                               'focal', 'splicebuster', 'trufor', 'zero']
        disabled_methods = [m for m in all_possible_methods if m not in enabled_methods]
        if disabled_methods:
            self.logger.info(f"Disabled methods due to device constraints: {disabled_methods}")

        return enabled_methods, failed_method_names, skipped_method_names

    def _get_gpu_memory_info(self) -> Dict[str, float]:
        """Get current GPU memory information."""
        if torch.cuda.is_available():
            return {
                'allocated': torch.cuda.memory_allocated() / 1024**3,  # GB
                'cached': torch.cuda.memory_reserved() / 1024**3,     # GB
                'max_allocated': torch.cuda.max_memory_allocated() / 1024**3,  # GB
            }
        return {'allocated': 0, 'cached': 0, 'max_allocated': 0}

    def _cleanup_gpu_memory(self, aggressive: bool = False):
        """Clean up GPU memory to prevent fragmentation and OOM errors."""
        if torch.cuda.is_available():
            if aggressive:
                # More aggressive cleanup for memory-intensive methods
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                self.logger.debug("Performed aggressive GPU memory cleanup")
            else:
                # Standard cleanup
                torch.cuda.empty_cache()
                self.logger.debug("Performed standard GPU memory cleanup")

    def _preprocess_image_shared(self, image_bytes: bytes, target_size: int = 512) -> np.ndarray:
        """
        Decode and preprocess image ONCE for all PhotoHolmes methods.

        This eliminates redundant decoding/resizing that was happening 8+ times
        (once per method). All methods use 512x512 input size.

        Args:
            image_bytes: Raw image bytes
            target_size: Target size for resize (default 512, used by all methods)

        Returns:
            Preprocessed numpy array (RGB, shape: target_size x target_size x 3)
        """
        try:
            with Image.open(io.BytesIO(image_bytes)) as image:
                # Convert to RGB if needed
                if image.mode != 'RGB':
                    image = image.convert('RGB')

                # Resize to target size (512x512 for all methods)
                image = image.resize(
                    (target_size, target_size),
                    Image.Resampling.LANCZOS
                )

                # Convert to numpy array
                image_np = np.array(image)

                # Validate
                if image_np.size == 0 or np.any(np.isnan(image_np)) or np.any(np.isinf(image_np)):
                    raise ValueError("Invalid image data: contains invalid values")

                self.logger.debug(f"Shared preprocessing: decoded to {image_np.shape}")
                return image_np

        except Exception as e:
            self.logger.error(f"Shared image preprocessing failed: {str(e)}")
            raise ValueError(f"Failed to preprocess image: {str(e)}")

    @asynccontextmanager
    async def _framework_lock(self, framework: str, method_name: str):
        """Context manager for framework-specific memory locks."""
        lock = self.framework_locks.get(framework, self.framework_locks['general'])
        self.logger.debug(f"Acquiring {framework} lock for {method_name}")
        await lock.acquire()
        try:
            yield
        finally:
            lock.release()
            self.logger.debug(f"Released {framework} lock for {method_name}")

    def _monitor_memory_usage(self, method_name: str, before_memory: Dict[str, float], after_memory: Dict[str, float]):
        """Monitor and log memory usage for a method."""
        if torch.cuda.is_available():
            allocated_diff = after_memory['allocated'] - before_memory['allocated']
            cached_diff = after_memory['cached'] - before_memory['cached']

            if allocated_diff > 0.1:  # Log if more than 100MB allocated
                self.logger.info(f"{method_name} allocated {allocated_diff:.2f}GB GPU memory")

            if allocated_diff < -0.05:  # Log if memory was freed
                self.logger.info(f"{method_name} freed {abs(allocated_diff):.2f}GB GPU memory")

    async def run_all_methods(self, image_bytes: bytes, document_type: str = None) -> PhotoHolmesResults:
        """
        Run all 10 PhotoHolmes methods on the provided image.

        Args:
            image_bytes: Input image as bytes
            document_type: Optional document type for context-aware analysis

        Returns:
            PhotoHolmesResults: Comprehensive results from all methods
        """
        # Store document type context for potential future use
        self.current_document_type = document_type
        self.logger.info("Starting comprehensive PhotoHolmes analysis")

        results = PhotoHolmesResults()

        # OPTIMIZATION: Decode and preprocess image ONCE for all methods
        # Previously each method was decoding the same image independently (8x redundant)
        preprocess_start = time.time()
        shared_image_np = self._preprocess_image_shared(image_bytes)
        self.logger.info(f"Shared image preprocessing took {time.time() - preprocess_start:.3f}s")

        # Extract EXIF data once for all methods that need it
        shared_exif_data = self.exif_extractor.extract_exif_data(image_bytes)
        self.logger.info(f"EXIF extraction completed: has_exif={shared_exif_data['has_exif']}")

        # Define method execution order (fast methods first for quick feedback)
        method_order = [
            ('dq', 'DQ', 'dq_method'),
            ('adaptive', 'Adaptive', 'adaptive_method'),
            ('noisesniffer', 'NoiseSniffer', 'noisesniffer_method'),
            ('psccnet', 'PSCCNet', 'psccnet_method'),
            ('splicebuster', 'Splicebuster', 'splicebuster_method'),
            ('trufor', 'TruFor', 'trufor_method'),
            # Conditionally include ZERO method based on performance flag
            *([('zero', 'ZERO', 'zero_method')] if self.enable_zero_method else []),
            # EXIF and Focal methods enabled despite ml_dtypes compatibility issues
            # ('exif_as_language', 'EXIF As Language', 'exif_as_language_method'),  # Temporarily disabled due to transformers version conflicts
            ('focal', 'FOCAL', 'focal_method'),  # Enabled despite ml_dtypes compatibility issue
            # IMDLBenCo CAT-Net disabled per user request due to performance issues
            # ('catnet', 'IMDLBenCo CAT-Net', 'imdlbenco_catnet_method'),
        ]

        successful_methods = 0
        methods_with_detections = 0
        all_scores = []

        # Run DQMethod separately (sequential) to prevent hanging
        # Note: DQ method needs raw bytes for JPEG DCT extraction, not the shared image
        dq_result = None
        if 'dq' in self.methods:
            try:
                self.logger.info("Running DQMethod sequentially to prevent hanging...")
                dq_method = self.methods['dq']
                dq_result = await self._run_single_method('dq', 'DQ', dq_method, 'dq_method', image_bytes, shared_exif_data, shared_image_np)
                self.logger.info("DQMethod completed successfully")
            except Exception as e:
                self.logger.error(f"DQMethod failed: {e}")

        # Run remaining methods concurrently
        tasks = []
        method_names = []

        for key, display_name, method_name in method_order:
            # Skip DQMethod - already handled sequentially
            if key == 'dq':
                continue

            try:
                # Check if method exists in the initialized methods
                if key not in self.methods:
                    self.logger.warning(f"Method {display_name} not available (insufficient device capabilities)")
                    continue

                method = self.methods[key]
                if hasattr(method, method_name):
                    task = asyncio.create_task(
                        self._run_single_method(key, display_name, method, method_name, image_bytes, shared_exif_data, shared_image_np)
                    )
                    tasks.append(task)
                    method_names.append((key, display_name))
                    self.logger.debug(f"Queued {display_name} for execution")
                else:
                    self.logger.warning(f"Method {display_name} does not have {method_name} method")

            except Exception as e:
                self.logger.error(f"Failed to prepare {display_name}: {e}")

        # Execute concurrent methods with framework-aware batching
        if tasks:
            successful_methods, methods_with_detections = await self._execute_framework_aware_batches(tasks, method_names, results, all_scores, methods_with_detections, successful_methods)

        # Process DQMethod result (sequential execution)
        if dq_result is not None:
            if isinstance(dq_result, DQMethodData):
                results.dq = dq_result
                all_scores.append(dq_result.max_probability)
                successful_methods += 1
                self.logger.info("DQMethod result processed successfully")

                # Count DQMethod detections
                if dq_result.max_probability > 0.1:
                    methods_with_detections += 1
            else:
                self.logger.error(f"DQMethod returned unexpected result type: {type(dq_result)}")

        # Calculate overall statistics
        results.total_methods_run = successful_methods
        results.methods_with_detections = methods_with_detections
        results.overall_forgery_probability = np.mean(all_scores) if all_scores else 0.0

        self.logger.info(
            f"PhotoHolmes analysis completed: {successful_methods}/{len(method_order)} methods successful, "
            f"{methods_with_detections} methods detected forgery, "
            f"overall probability: {results.overall_forgery_probability:.4f}"
        )

        # Final GPU memory cleanup to free resources for other services
        self._cleanup_gpu_memory(aggressive=True)

        # Log final memory state
        final_memory = self._get_gpu_memory_info()
        if torch.cuda.is_available():
            self.logger.info(f"Final GPU memory state: {final_memory['allocated']:.2f}GB allocated, {final_memory['cached']:.2f}GB cached")

        return results

    def extract_shared_exif_data(self, image_bytes: bytes) -> Dict[str, Any]:
        """Extract shared EXIF data that can be used by other services."""
        return self.exif_extractor.extract_exif_data(image_bytes)

    async def _execute_framework_aware_batches(self, tasks: list, method_names: list, results: PhotoHolmesResults, all_scores: list, methods_with_detections: int, successful_methods: int):
        """
        Execute tasks in framework-aware batches to optimize concurrent processing.

        Methods using different frameworks (PyTorch vs ONNX) can run concurrently,
        while methods using the same framework run sequentially to avoid memory conflicts.
        """
        # Group tasks by framework
        framework_tasks = {}
        for task, (key, display_name) in zip(tasks, method_names):
            framework = self.method_frameworks.get(key, 'general')
            if framework not in framework_tasks:
                framework_tasks[framework] = []
            framework_tasks[framework].append((task, key, display_name))

        self.logger.info(f"Executing {len(tasks)} methods across {len(framework_tasks)} frameworks: {list(framework_tasks.keys())}")

        # Execute framework batches concurrently
        framework_execution_tasks = []
        for framework, fw_tasks in framework_tasks.items():
            framework_task = asyncio.create_task(
                self._execute_framework_batch(framework, fw_tasks, results, all_scores)
            )
            framework_execution_tasks.append(framework_task)
            self.logger.info(f"Queued {len(fw_tasks)} {framework} methods for batch execution")

        # Wait for all framework batches to complete
        batch_results = await asyncio.gather(*framework_execution_tasks, return_exceptions=True)

        # Update counters from batch results
        for i, batch_result in enumerate(batch_results):
            if isinstance(batch_result, Exception):
                framework = list(framework_tasks.keys())[i]
                self.logger.error(f"Framework batch {framework} failed: {batch_result}")
            else:
                successful_methods += batch_result['successful']
                methods_with_detections += batch_result['with_detections']
                self.logger.info(f"Framework batch {i} completed: {batch_result['successful']} successful, {batch_result['with_detections']} with detections")

        return successful_methods, methods_with_detections

    async def _execute_framework_batch(self, framework: str, fw_tasks: list, results: PhotoHolmesResults, all_scores: list) -> Dict[str, int]:
        """Execute a batch of tasks that use the same framework sequentially to avoid memory conflicts."""
        successful = 0
        with_detections = 0

        self.logger.info(f"Executing {len(fw_tasks)} {framework} methods sequentially within framework batch...")

        for task, key, display_name in fw_tasks:
            try:
                # Get timeout for this specific method
                timeout = self._get_method_timeout(key)
                if timeout is None:
                    self.logger.info(f"{display_name} disabled (timeout set to 0)")
                    continue
                # Note: start logging is done in _run_single_method (includes framework info)

                # Execute task with timeout protection
                import time
                method_start = time.time()
                result = await asyncio.wait_for(task, timeout=timeout)
                method_duration = time.time() - method_start

                # Note: completion logging is done in _run_single_method (includes score)
                successful += 1

                # Store result in the appropriate field
                if key == 'adaptive' and isinstance(result, AdaptiveMethodData):
                    results.adaptive = result
                    all_scores.append(result.tampered_ratio)

                elif key == 'noisesniffer' and isinstance(result, NoiseSnifferData):
                    results.noisesniffer = result
                    all_scores.append(result.noise_confidence_score)

                elif key == 'psccnet' and isinstance(result, PsccnetMethodData):
                    results.psccnet = result
                    all_scores.append(result.psccnet_confidence_score)

                # elif key == 'exif_as_language' and isinstance(result, ExifAsLanguageData):
                #     results.exif_as_language = result
                #     all_scores.append(result.exif_confidence_score)

                elif key == 'focal' and isinstance(result, FocalMethodData):
                    results.focal = result
                    all_scores.append(result.focal_confidence_score)

                elif key == 'splicebuster' and isinstance(result, SplicebusterMethodData):
                    results.splicebuster = result
                    all_scores.append(result.splicebuster_confidence_score)

                elif key == 'trufor' and isinstance(result, TruforMethodData):
                    results.trufor = result
                    all_scores.append(result.trufor_confidence_score)

                elif key == 'zero' and isinstance(result, ZeroMethodData):
                    results.zero = result
                    all_scores.append(result.zero_confidence_score)

                # Count methods with significant detections (above 0.1)
                if all_scores and all_scores[-1] > 0.1:
                    with_detections += 1

                self.logger.debug(f"{display_name} ({framework}) completed successfully")

            except asyncio.TimeoutError:
                self.logger.warning(f"{display_name} ({framework}) timed out after {self._get_method_timeout(key)}s - NOT counted towards forgery threshold")
                # Timed out tests do NOT count towards the forgery detection threshold
                # We simply skip this method - no score added, no detection counted

            except Exception as e:
                self.logger.error(f"{display_name} ({framework}) failed: {e} - NOT counted towards forgery threshold")
                # Failed tests (like timeouts) do NOT count towards the forgery detection threshold
                # We simply skip this method - no score added, no detection counted

        return {
            'successful': successful,
            'with_detections': with_detections
        }

    def _get_method_timeout(self, key: str) -> Optional[int]:
        """Get timeout for a specific method in seconds. Returns None if method is disabled."""
        timeout_map = {
            'dq': photoholmes_settings.dq_method_timeout,
            'adaptive': photoholmes_settings.adaptive_method_timeout,
            'noisesniffer': photoholmes_settings.noisesniffer_method_timeout,
            'psccnet': photoholmes_settings.psccnet_method_timeout,
            'splicebuster': photoholmes_settings.splicebuster_method_timeout,
            'trufor': photoholmes_settings.trufor_method_timeout,
            'focal': photoholmes_settings.focal_method_timeout,
            'zero': photoholmes_settings.zero_method_timeout,
        }
        timeout = timeout_map.get(key, 10)  # Default 10s for unknown methods
        return None if timeout == 0 else timeout

    async def _run_single_method(self, key: str, display_name: str, method: Any, method_name: str, image_bytes: bytes, shared_exif_data: Optional[Dict[str, Any]] = None, shared_image_np: Optional[np.ndarray] = None) -> Any:
        """Run a single PhotoHolmes method with error handling, timeout protection, and GPU memory management.

        Args:
            key: Method key identifier
            display_name: Human-readable method name
            method: Method instance
            method_name: Name of the method to call
            image_bytes: Raw image bytes (needed for DQ/EXIF methods)
            shared_exif_data: Pre-extracted EXIF data
            shared_image_np: Pre-decoded and resized image (512x512 RGB numpy array)
        """
        import time
        start_time = time.time()

        try:
            # Get the framework for this method
            framework = self.method_frameworks.get(key, 'general')

            # Use framework-specific lock for memory management
            async with self._framework_lock(framework, display_name):
                self.logger.info(f"Starting {display_name} execution on {framework} framework")
                method_func = getattr(method, method_name)

                # Get baseline memory before method execution
                before_memory = self._get_gpu_memory_info()

                # Determine if this is a memory-intensive method
                is_memory_intensive = key in ['psccnet', 'focal']
                is_moderate_intensity = key in ['adaptive', 'trufor', 'zero']

                # Cleanup memory before execution for memory-intensive methods
                if is_memory_intensive:
                    self._cleanup_gpu_memory(aggressive=True)
                elif is_moderate_intensity:
                    self._cleanup_gpu_memory(aggressive=False)

                # Use consistent 10 second timeout for all methods
                timeout = 10  # Timed out tests don't count towards forgery threshold

                # Create method execution with arguments
                # DQ and EXIF methods need raw bytes (for DCT/EXIF extraction)
                # Other methods can use the pre-decoded shared image for efficiency
                if key == 'exif_as_language' and shared_exif_data:
                    method_execution = method_func(image_bytes, shared_exif_data)
                elif key == 'dq':
                    # DQ needs raw JPEG bytes for DCT coefficient extraction
                    method_execution = method_func(image_bytes)
                elif shared_image_np is not None:
                    # OPTIMIZATION: Pass pre-decoded image to avoid redundant decoding
                    method_execution = method_func(image_bytes, shared_image_np)
                else:
                    method_execution = method_func(image_bytes)

                # Direct execution - all methods are now async through BaseForgeryMethod
                # No need for batch processing since we're executing sequentially within framework batches
                result = await method_execution

                # Get memory after execution and monitor usage
                after_memory = self._get_gpu_memory_info()
                self._monitor_memory_usage(display_name, before_memory, after_memory)

                # Perform cleanup for memory-intensive methods to free resources for next method
                if is_memory_intensive:
                    self._cleanup_gpu_memory(aggressive=True)
                    self.logger.debug(f"Performed post-execution cleanup for {display_name}")

                # Log the score
                if hasattr(result, 'max_probability'):
                    score = result.max_probability
                elif hasattr(result, 'tampered_ratio'):
                    score = result.tampered_ratio
                elif hasattr(result, 'noise_confidence_score'):
                    score = result.noise_confidence_score
                elif hasattr(result, 'psccnet_confidence_score'):
                    score = result.psccnet_confidence_score
                elif hasattr(result, 'catnet_confidence_score'):
                    score = result.catnet_confidence_score
                elif hasattr(result, 'exif_confidence_score'):
                    score = result.exif_confidence_score
                elif hasattr(result, 'focal_confidence_score'):
                    score = result.focal_confidence_score
                elif hasattr(result, 'splicebuster_confidence_score'):
                    score = result.splicebuster_confidence_score
                elif hasattr(result, 'trufor_confidence_score'):
                    score = result.trufor_confidence_score
                elif hasattr(result, 'zero_confidence_score'):
                    score = result.zero_confidence_score
                else:
                    score = 0.0

                import time
                duration = time.time() - start_time if 'start_time' in locals() else 0
                self.logger.info(f"{display_name} completed in {duration:.2f}s - score = {score:.4f}")
                return result

        except asyncio.TimeoutError:
            self.logger.error(f"{display_name} timed out after {timeout} seconds")
            # Return default result instead of raising exception to prevent hanging
            self.logger.warning(f"Returning default result for {display_name} due to timeout")
            return self._get_default_method_result(key)
        except Exception as e:
            self.logger.error(f"{display_name} failed: {str(e)}")
            # Check for specific Apple Silicon/ONNX errors
            if "onnxruntime" in str(e).lower() or "mps" in str(e).lower():
                self.logger.error(f"{display_name} failed with Apple Silicon/ONNX error: {e}")
            # Return default result instead of raising exception to allow other methods to continue
            self.logger.warning(f"Returning default result for {display_name} due to error")
            return self._get_default_method_result(key)

    def get_method_summary(self, results: PhotoHolmesResults) -> Dict[str, Any]:
        """
        Generate a summary of all method results for logging and analysis.
        """
        summary = {
            'total_methods': results.total_methods_run,
            'methods_with_detections': results.methods_with_detections,
            'overall_probability': results.overall_forgery_probability,
            'method_scores': {}
        }

        # Collect all individual scores
        if results.dq:
            summary['method_scores']['DQ'] = results.dq.max_probability
        if results.adaptive:
            summary['method_scores']['Adaptive'] = results.adaptive.tampered_ratio
        if results.noisesniffer:
            summary['method_scores']['NoiseSniffer'] = results.noisesniffer.noise_confidence_score
        if results.psccnet:
            summary['method_scores']['PSCCNet'] = results.psccnet.psccnet_confidence_score
        # if results.exif_as_language:
        #     summary['method_scores']['EXIF'] = results.exif_as_language.exif_confidence_score
        if results.focal:
            summary['method_scores']['FOCAL'] = results.focal.focal_confidence_score
        if results.splicebuster:
            summary['method_scores']['Splicebuster'] = results.splicebuster.splicebuster_confidence_score
        if results.trufor:
            summary['method_scores']['TruFor'] = results.trufor.trufor_confidence_score
        if results.zero:
            summary['method_scores']['ZERO'] = results.zero.zero_confidence_score
        # IMDLBenCo CAT-Net disabled per user request
        # if results.catnet:
        #     summary['method_scores']['IMDLBenCo CAT-Net'] = results.catnet.catnet_confidence_score

        return summary

    def _get_default_method_result(self, key: str):
        """Get a default result for a method when it fails or times out."""
        # Import the appropriate data class based on method type
        if key == 'dq':
            return DQMethodData(max_probability=0.0)
        elif key == 'adaptive':
            return AdaptiveMethodData(tampered_ratio=0.0)
        elif key == 'noisesniffer':
            return NoiseSnifferData(noise_confidence_score=0.0)
        elif key == 'psccnet':
            return PsccnetMethodData(psccnet_confidence_score=0.0)
        elif key == 'focal':
            return FocalMethodData(focal_confidence_score=0.0)
        elif key == 'splicebuster':
            return SplicebusterMethodData(splicebuster_confidence_score=0.0)
        elif key == 'trufor':
            return TruforMethodData(trufor_confidence_score=0.0)
        elif key == 'zero':
            return ZeroMethodData(zero_confidence_score=0.0)
        else:
            # Return a generic result with 0.0 score for unknown methods
            return type('DefaultResult', (), {'max_probability': 0.0})()