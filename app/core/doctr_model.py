import os
import logging
from typing import Optional, Dict, List, Any, Tuple, TYPE_CHECKING
# Lazy imports to avoid dependency issues
def get_ocr_predictor():
    from doctr.models import ocr_predictor
    return ocr_predictor

def get_ocr_predictor_class():
    from doctr.models.predictor import OCRPredictor
    return OCRPredictor

# Import OCRPredictor for type hints when type checking
if TYPE_CHECKING:
    from doctr.models.predictor import OCRPredictor
from app.core.logger import get_logger
from app.core.key_injection import key_injection_manager, DocumentType, key_config
from app.core.gpu_manager import ModelType, get_gpu_manager
from app.core.framework_coordinator import get_framework_coordinator, FrameworkType
import warnings

logger = logging.getLogger(__name__)


class DoctrModel:
    """
    DocTR OCR model wrapper integrated with GPU resource manager.

    This class manages the DocTR OCR model with centralized
    GPU resource allocation and memory management.
    """

    _instance: Optional['DoctrModel'] = None
    _initialized: bool = False

    # Default model architectures
    DET_ARCH = "db_resnet50"
    RECO_ARCH = "crnn_vgg16_bn"

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DoctrModel, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self.gpu_manager = get_gpu_manager()
            self.framework_coordinator = get_framework_coordinator()
            self.logger = get_logger()
            self._initialized = True
            self._setup_environment()
            logger.info("DoctrModel initialized with unified framework coordinator")

    def _setup_environment(self):
        """Setup ONNX Runtime environment and configure GPU acceleration."""
        cache_dir = os.path.expanduser("~/.cache/onnxruntime")
        os.makedirs(cache_dir, exist_ok=True)
        os.environ['ONNXRUNTIME_CACHE_DIR'] = cache_dir

        # Configure PyTorch hub cache directory for DocTR weights
        torch_home = os.getenv('TORCH_HOME', os.path.join(os.path.dirname(__file__), '..', 'models'))
        torch_home = os.path.abspath(torch_home)
        os.makedirs(torch_home, exist_ok=True)
        os.environ['TORCH_HOME'] = torch_home
        self.logger.info(f"PyTorch hub cache directory set to: {torch_home}")

        # Configure ONNX Runtime for GPU acceleration
        self._configure_onnx_runtime()

    def _configure_onnx_runtime(self):
        """
        Skip ONNX Runtime configuration for DocTR (PyTorch version).
        DocTR with PyTorch backend doesn't need ONNX Runtime.
        """
        self.logger.info("DocTR uses PyTorch backend - skipping ONNX Runtime configuration")
        # DocTR will use PyTorch's CUDA/ROCm backend directly
        return

    def get_model(self, det_arch: Optional[str] = None, reco_arch: Optional[str] = None) -> 'OCRPredictor':
        """
        Get OCR model with GPU resource management (synchronous version).
        Creates model without async GPU manager for module-level initialization.

        Args:
            det_arch: Detection architecture
            reco_arch: Recognition architecture

        Returns:
            OCRPredictor model instance
        """
        try:
            # Direct model creation for module initialization
            # GPU resources will be managed on first use
            model = self._create_model(
                det_arch=det_arch or self.DET_ARCH,
                reco_arch=reco_arch or self.RECO_ARCH
            )

            self.logger.debug("DocTR OCR model created (GPU resources allocated on first use)")
            return model

        except Exception as e:
            self.logger.error(f"Failed to get DocTR OCR model: {e}")
            raise RuntimeError(f"DocTR OCR model initialization failed: {str(e)}")

    async def get_model_with_gpu(self, det_arch: Optional[str] = None, reco_arch: Optional[str] = None) -> 'OCRPredictor':
        """
        Get OCR model with async GPU resource management.

        Args:
            det_arch: Detection architecture
            reco_arch: Recognition architecture

        Returns:
            OCRPredictor model instance with GPU resources allocated
        """
        try:
            # Get model with GPU resource management
            model = await self.gpu_manager.get_model_with_gpu(
                model_type=ModelType.DOCTR_OCR,
                model_class=self._create_model,
                det_arch=det_arch or self.DET_ARCH,
                reco_arch=reco_arch or self.RECO_ARCH
            )

            self.logger.debug("DocTR OCR model retrieved from GPU manager")
            return model

        except Exception as e:
            self.logger.error(f"Failed to get DocTR OCR model: {e}")
            raise RuntimeError(f"DocTR OCR model initialization failed: {str(e)}")

    def _create_model(self, det_arch: str, reco_arch: str) -> 'OCRPredictor':
        """
        Create DocTR OCR model instance.

        Args:
            det_arch: Detection architecture
            reco_arch: Recognition architecture

        Returns:
            OCRPredictor model instance
        """
        try:
            # Check PyTorch GPU availability
            import torch
            self.logger.info(f"PyTorch version: {torch.__version__}")
            self.logger.info(f"PyTorch CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                self.logger.info(f"PyTorch device count: {torch.cuda.device_count()}")
                self.logger.info(f"PyTorch current device: {torch.cuda.current_device()}")
                self.logger.info(f"PyTorch device name: {torch.cuda.get_device_name()}")

            self.logger.info(
                f"Initializing OCR model with det_arch={det_arch}, "
                f"reco_arch={reco_arch}, export_as_straight_boxes=True"
            )

            # Suppress TensorFlow warnings from onnxtr
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning, module=".*tensorflow.*")
                warnings.filterwarnings("ignore", category=FutureWarning, module=".*tensorflow.*")

                ocr_predictor = get_ocr_predictor()
                model = ocr_predictor(
                    det_arch=det_arch,
                    reco_arch=reco_arch,
                    pretrained=True,  # Load pretrained weights for accurate OCR
                    assume_straight_pages=True,  # Optimize for horizontal text
                    export_as_straight_boxes=True,  # Handle rotated text
                    paragraph_break=0.035,  # Default line grouping
                    resolve_lines=True,  # Resolve lines for better text structure
                    resolve_blocks=False,  # Don't group into blocks - keep lines separate
                )

            self.logger.info("DocTR OCR model created successfully")
            return model

        except Exception as e:
            self.logger.error(f"Failed to create Doctr model: {str(e)}")
            raise

    async def release_model(self) -> bool:
        """
        Release DocTR model resources.

        Returns:
            True if model was released successfully
        """
        try:
            released = await self.gpu_manager.release_model(ModelType.DOCTR_OCR)
            if released:
                self.logger.info("DocTR OCR model resources released")
            return released
        except Exception as e:
            self.logger.error(f"Failed to release DocTR OCR model: {e}")
            return False

    def get_model_sync(self, det_arch: Optional[str] = None, reco_arch: Optional[str] = None) -> 'OCRPredictor':
        """
        Synchronous wrapper for get_model - use with caution in async contexts.

        Args:
            det_arch: Detection architecture
            reco_arch: Recognition architecture

        Returns:
            OCRPredictor model instance
        """
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self.get_model(det_arch, reco_arch))
        except RuntimeError:
            # No event loop running, create one
            return asyncio.run(self.get_model(det_arch, reco_arch))

    def _get_onnx_providers(self) -> List[str]:
        """
        Get ONNX Runtime execution providers prioritized for GPU acceleration

        Returns:
            List of execution providers in order of preference
        """
        try:
            import onnxruntime as ort
            available_providers = ort.get_available_providers()

            # Prioritize providers based on platform and availability
            preferred_providers = []

            # Platform-specific provider prioritization
            import platform
            system = platform.system()

            if system == "Darwin":  # macOS
                # Prioritize Apple-specific providers
                if 'CoreMLExecutionProvider' in available_providers:
                    preferred_providers.append('CoreMLExecutionProvider')
                    self.logger.info("CoreML Execution Provider configured for OCR (Apple Silicon)")
            else:  # Linux and other systems
                # Prioritize GPU providers
                if 'CUDAExecutionProvider' in available_providers:
                    preferred_providers.append('CUDAExecutionProvider')
                    self.logger.info("CUDA Execution Provider configured for OCR")
                elif 'ROCmExecutionProvider' in available_providers:
                    preferred_providers.append('ROCmExecutionProvider')
                    self.logger.info("ROCm Execution Provider configured for OCR (AMD)")

            # Add CPU provider as fallback if not already included
            if 'CPUExecutionProvider' in available_providers and 'CPUExecutionProvider' not in preferred_providers:
                preferred_providers.append('CPUExecutionProvider')

            # Add any remaining available providers
            for provider in available_providers:
                if provider not in preferred_providers:
                    preferred_providers.append(provider)

            return preferred_providers

        except ImportError:
            self.logger.warning("ONNX Runtime not available - using default providers")
            return ['CPUExecutionProvider']
        except Exception as e:
            self.logger.error(f"Error configuring ONNX providers: {e}")
            return ['CPUExecutionProvider']

    async def predict_with_key_injection(self,
                                 doc,
                                 document_type: DocumentType,
                                 cross_document_data: Optional[Dict[str, List[str]]] = None,
                                 det_arch: Optional[str] = None,
                                 reco_arch: Optional[str] = None) -> Dict[str, Any]:
        """
        Run OCR prediction with key injection enhancement.

        Args:
            doc: Document object from DocumentFile
            document_type: Type of document being processed
            cross_document_data: Values from other documents for cross-validation
            det_arch: Detection architecture
            reco_arch: Recognition architecture

        Returns:
            Dict containing OCR results and key injection analysis
        """
        try:
            self.logger.info(f"Running OCR prediction with key injection for {document_type.value}")

            # Get the base OCR model with GPU resource management
            model = await self.get_model(det_arch, reco_arch)

            # Run OCR prediction
            ocr_result = model(doc)

            # Convert OCR results to geometry format for key injection
            geometry_data = self._convert_ocr_to_geometry(ocr_result)

            # Process with key injection
            key_injection_result = key_injection_manager.process_document_with_key_injection(
                ocr_geometry_data=geometry_data,
                document_type=document_type,
                cross_document_data=cross_document_data
            )

            return {
                'ocr_result': ocr_result,
                'geometry_data': geometry_data,
                'key_injection_result': key_injection_result,
                'success': key_injection_result.success,
                'error': key_injection_result.error_message if not key_injection_result.success else None
            }

        except Exception as e:
            self.logger.error(f"OCR prediction with key injection failed: {str(e)}")
            return {
                'ocr_result': None,
                'geometry_data': [],
                'key_injection_result': None,
                'success': False,
                'error': str(e)
            }

    def _convert_ocr_to_geometry(self, ocr_result) -> List[Dict[str, Any]]:
        """
        Convert OCR results to geometry format compatible with key injection.

        Args:
            ocr_result: Raw OCR result from doctr model

        Returns:
            List of geometry data dictionaries
        """
        geometry_data = []

        for page_idx, page in enumerate(ocr_result.pages):
            for block_idx, block in enumerate(page.blocks):
                for line_idx, line in enumerate(block.lines):
                    # Get line text and geometry
                    line_words = [word.value for word in line.words]
                    if line_words:
                        line_text = " ".join(line_words)
                        x1, y1 = line.geometry[0]
                        x2, y2 = line.geometry[1]

                        # Calculate average confidence for the line
                        total_confidence = 0
                        word_count = 0
                        word_details = []

                        for word in line.words:
                            word_conf = getattr(word, 'confidence', 0.9)
                            total_confidence += word_conf
                            word_count += 1

                            # Word geometry
                            word_x1, word_y1 = word.geometry[0]
                            word_x2, word_y2 = word.geometry[1]

                            word_details.append({
                                'text': word.value,
                                'confidence': word_conf,
                                'x1': word_x1,
                                'y1': word_y1,
                                'x2': word_x2,
                                'y2': word_y2
                            })

                        avg_confidence = total_confidence / word_count if word_count > 0 else 0.9

                        geometry_data.append({
                            'text': line_text,
                            'x1': x1,
                            'y1': y1,
                            'x2': x2,
                            'y2': y2,
                            'confidence': avg_confidence,
                            'word_details': word_details,
                            'page': page_idx,
                            'block': block_idx,
                            'line': line_idx
                        })

        return geometry_data

    async def extract_key_values_only(self,
                              doc,
                              document_type: DocumentType,
                              min_confidence: float = 0.8,
                              det_arch: Optional[str] = None,
                              reco_arch: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Extract only key-value pairs with high confidence.

        Args:
            doc: Document object from DocumentFile
            document_type: Type of document being processed
            min_confidence: Minimum confidence threshold for extraction
            det_arch: Detection architecture
            reco_arch: Recognition architecture

        Returns:
            List of high-confidence key-value pairs
        """
        try:
            # Run OCR with key injection
            result = await self.predict_with_key_injection(
                doc, document_type, det_arch=det_arch, reco_arch=reco_arch
            )

            if not result['success'] or not result['key_injection_result']:
                return []

            # Get high-confidence extractions
            high_confidence = key_injection_manager.get_high_confidence_extractions(
                result['key_injection_result'], min_confidence
            )

            # Convert to simple format
            key_value_pairs = []
            for detected_key, enhanced_confidence in high_confidence:
                key_value_pairs.append({
                    'key': detected_key.key_name,
                    'value': detected_key.value_candidate,
                    'confidence': enhanced_confidence.overall_confidence,
                    'key_confidence': detected_key.confidence,
                    'spatial_confidence': detected_key.value_confidence,
                    'geometry': detected_key.geometry,
                    'explanation': enhanced_confidence.explanation
                })

            return key_value_pairs

        except Exception as e:
            self.logger.error(f"Key-value extraction failed: {str(e)}")
            return []

    async def validate_document_completeness(self,
                                     doc,
                                     document_type: DocumentType,
                                     det_arch: Optional[str] = None,
                                     reco_arch: Optional[str] = None) -> Dict[str, Any]:
        """
        Validate if all required keys are extracted with sufficient confidence.

        Args:
            doc: Document object from DocumentFile
            document_type: Type of document being processed
            det_arch: Detection architecture
            reco_arch: Recognition architecture

        Returns:
            Dict with validation results
        """
        try:
            # Run OCR with key injection
            result = await self.predict_with_key_injection(
                doc, document_type, det_arch=det_arch, reco_arch=reco_arch
            )

            if not result['success'] or not result['key_injection_result']:
                return {
                    'is_complete': False,
                    'missing_keys': [],
                    'low_confidence_keys': [],
                    'issues': [result['error']] if result['error'] else ['Key injection failed'],
                    'summary': None
                }

            # Validate completeness
            is_complete, issues = key_injection_manager.validate_document_completeness(
                result['key_injection_result']
            )

            # Get missing required keys
            missing_keys = key_injection_manager.get_missing_required_keys(
                result['key_injection_result']
            )

            # Get low confidence required keys
            low_confidence_keys = []
            for key in result['key_injection_result'].detected_keys:
                if key_config.is_required_key(document_type, key.key_name):
                    confidence = result['key_injection_result'].enhanced_confidences.get(key.key_name)
                    if confidence and confidence.overall_confidence < 0.7:
                        low_confidence_keys.append(key.key_name)

            return {
                'is_complete': is_complete,
                'missing_keys': missing_keys,
                'low_confidence_keys': low_confidence_keys,
                'issues': issues,
                'summary': result['key_injection_result'].processing_summary
            }

        except Exception as e:
            self.logger.error(f"Document completeness validation failed: {str(e)}")
            return {
                'is_complete': False,
                'missing_keys': [],
                'low_confidence_keys': [],
                'issues': [str(e)],
                'summary': None
            }
