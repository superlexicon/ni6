"""
Centralized device management for GPU/CPU detection and configuration
"""

import os
import platform
import subprocess
from enum import Enum
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass
from app.core.logger import get_logger


class DeviceType(Enum):
    """Available device types"""
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"  # Apple Metal Performance Shaders
    ROCM = "rocm"  # AMD ROCm


@dataclass
class DeviceInfo:
    """Device information structure"""
    device_type: DeviceType
    device_id: int
    device_name: str
    memory_total: Optional[int] = None  # MB
    memory_available: Optional[int] = None  # MB
    compute_capability: Optional[str] = None
    driver_version: Optional[str] = None
    is_available: bool = True
    error_message: Optional[str] = None


@dataclass
class ValidationResult:
    """GPU validation result"""
    is_valid: bool
    device_info: Optional[DeviceInfo]
    issues: List[str]
    warnings: List[str]
    recommendations: List[str]


class DeviceManager:
    """
    Centralized device management for GPU/CPU detection and configuration
    """
    _instance: Optional['DeviceManager'] = None
    _initialized: bool = False
    _cached_devices: Optional[Dict[DeviceType, List[DeviceInfo]]] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DeviceManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self._initialized = True
            self.logger = get_logger()

    def detect_available_devices(self) -> Dict[DeviceType, List[DeviceInfo]]:
        """
        Detect all available devices (GPU and CPU)

        Returns:
            Dictionary mapping device types to available devices
        """
        if self._cached_devices is None:
            self._cached_devices = {}

            # Detect NVIDIA GPUs
            self._cached_devices[DeviceType.CUDA] = self._detect_cuda_devices()

            # Detect Apple Silicon GPUs
            if platform.system() == "Darwin":
                self._cached_devices[DeviceType.MPS] = self._detect_mps_devices()

            # Detect AMD GPUs (ROCm)
            self._cached_devices[DeviceType.ROCM] = self._detect_rocm_devices()

            # CPU is always available
            self._cached_devices[DeviceType.CPU] = [self._get_cpu_device()]

        return self._cached_devices

    def get_recommended_device(self, prefer_gpu: bool = True) -> DeviceInfo:
        """
        Get the best available device for computation

        Args:
            prefer_gpu: Whether to prefer GPU over CPU

        Returns:
            DeviceInfo for the recommended device
        """
        devices = self.detect_available_devices()

        if prefer_gpu:
            # Priority: CUDA > MPS > ROCm > CPU
            gpu_priority = [
                DeviceType.CUDA,
                DeviceType.MPS,
                DeviceType.ROCM
            ]

            for device_type in gpu_priority:
                if device_type in devices and devices[device_type]:
                    return devices[device_type][0]

        # Fallback to CPU
        return devices[DeviceType.CPU][0]

    def validate_gpu_device(self, device_type: DeviceType, device_id: int = 0) -> ValidationResult:
        """
        Validate a specific GPU device for compatibility

        Args:
            device_type: Type of GPU to validate
            device_id: GPU device ID

        Returns:
            ValidationResult with validation details
        """
        issues = []
        warnings = []
        recommendations = []

        try:
            devices = self.detect_available_devices()

            if device_type not in devices:
                issues.append(f"Device type {device_type.value} not available")
                return ValidationResult(False, None, issues, warnings, recommendations)

            if device_id >= len(devices[device_type]):
                issues.append(f"Device ID {device_id} not available for {device_type.value}")
                return ValidationResult(False, None, issues, warnings, recommendations)

            device_info = devices[device_type][device_id]

            if not device_info.is_available:
                issues.append(f"Device not available: {device_info.error_message}")
                return ValidationResult(False, device_info, issues, warnings, recommendations)

            # Device-specific validation
            if device_type == DeviceType.CUDA:
                self._validate_cuda_device(device_info, issues, warnings, recommendations)
            elif device_type == DeviceType.MPS:
                self._validate_mps_device(device_info, issues, warnings, recommendations)
            elif device_type == DeviceType.ROCM:
                self._validate_rocm_device(device_info, issues, warnings, recommendations)

            # Common GPU validation
            self._validate_gpu_common(device_info, issues, warnings, recommendations)

            return ValidationResult(
                len(issues) == 0,
                device_info,
                issues,
                warnings,
                recommendations
            )

        except Exception as e:
            self.logger.error(f"GPU validation failed: {e}")
            issues.append(f"Validation error: {str(e)}")
            return ValidationResult(False, None, issues, warnings, recommendations)

    def configure_tensorflow_device(self, device_info: DeviceInfo) -> Dict[str, Any]:
        """
        Configure TensorFlow device settings

        Args:
            device_info: Target device information

        Returns:
            Dictionary with TensorFlow configuration
        """
        config = {}

        if device_info.device_type == DeviceType.CUDA:
            config = {
                'device_count': {'GPU': 1},
                'allow_growth': True,
                'memory_limit': None,  # Let TensorFlow manage memory
                'log_device_placement': False,
            }
        elif device_info.device_type == DeviceType.MPS:
            config = {
                'device_count': {'GPU': 0, 'MPS': 1},
                'allow_growth': True,
                'memory_limit': None,
                'log_device_placement': False,
            }
        else:  # CPU
            config = {
                'device_count': {'GPU': 0},
                'allow_growth': True,
                'memory_limit': None,
                'log_device_placement': False,
            }

        return config

    def configure_pytorch_device(self, device_info: DeviceInfo) -> str:
        """
        Configure PyTorch device string

        Args:
            device_info: Target device information

        Returns:
            PyTorch device string
        """
        if device_info.device_type == DeviceType.CUDA:
            return f"cuda:{device_info.device_id}"
        elif device_info.device_type == DeviceType.ROCM:
            # ROCm uses CUDA interface in PyTorch
            return f"cuda:{device_info.device_id}"
        elif device_info.device_type == DeviceType.MPS:
            return "mps"
        else:
            return "cpu"

    def get_device_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all available devices

        Returns:
            Dictionary with device summary
        """
        devices = self.detect_available_devices()
        summary = {
            'platform': platform.system(),
            'python_version': platform.python_version(),
            'devices': {}
        }

        for device_type, device_list in devices.items():
            summary['devices'][device_type.value] = [
                {
                    'device_id': d.device_id,
                    'name': d.device_name,
                    'memory_total': d.memory_total,
                    'is_available': d.is_available,
                    'error': d.error_message
                }
                for d in device_list
            ]

        recommended = self.get_recommended_device()
        summary['recommended_device'] = {
            'type': recommended.device_type.value,
            'id': recommended.device_id,
            'name': recommended.device_name,
            'memory_gb': self._get_memory_in_gb(recommended.memory_total) if recommended.memory_total else None
        }

        return summary

    def has_sufficient_memory_for_sida(self, device_info: DeviceInfo, required_gb: int = 16) -> bool:
        """
        Check if device has sufficient memory for SIDA operations

        Args:
            device_info: Device information
            required_gb: Required memory in GB (default 16GB for SIDA)

        Returns:
            True if device has sufficient memory
        """
        if device_info.device_type == DeviceType.CPU:
            return True  # CPU is always sufficient for fallback

        if device_info.memory_total is None:
            # Try to get memory info if not available
            device_info.memory_total = self._get_device_memory(device_info)

        memory_gb = self._get_memory_in_gb(device_info.memory_total)

        self.logger.info(f"Device {device_info.device_name}: {memory_gb}GB available, required: {required_gb}GB")

        return memory_gb >= required_gb

    def get_workflow_recommendations(self, device_info: DeviceInfo) -> Dict[str, Any]:
        """
        Get workflow recommendations based on device capabilities

        Args:
            device_info: Device information

        Returns:
            Dictionary with workflow recommendations
        """
        memory_gb = self._get_memory_in_gb(device_info.memory_total)

        recommendations = {
            'device_type': device_info.device_type.value,
            'device_name': device_info.device_name,
            'memory_gb': memory_gb,
            'can_run_sid_a': False,
            'can_run_doctr': False,
            'can_run_photoholmes': False,
            'recommended_approach': 'full',
            'memory_warnings': [],
            'performance_tips': []
        }

        # PhotoHolmes requirements (modest)
        if memory_gb >= 4:
            recommendations['can_run_photoholmes'] = True
            recommendations['performance_tips'].append("PhotoHolmes will run efficiently")
        else:
            recommendations['memory_warnings'].append("PhotoHolmes may be slow with <4GB GPU memory")

        # DocTR requirements (moderate)
        if memory_gb >= 6:
            recommendations['can_run_doctr'] = True
            recommendations['performance_tips'].append("DocTR will run efficiently")
        elif memory_gb >= 4:
            recommendations['can_run_doctr'] = True
            recommendations['memory_warnings'].append("DocTR may be slow with <6GB GPU memory")
        else:
            recommendations['memory_warnings'].append("DocTR may have issues with <4GB GPU memory")

        # SIDA requirements (high)
        if memory_gb >= 16:
            recommendations['can_run_sid_a'] = True
            recommendations['recommended_approach'] = 'full'
            recommendations['performance_tips'].append("SIDA will run efficiently with full capabilities")
        elif memory_gb >= 12:
            recommendations['can_run_sid_a'] = True
            recommendations['recommended_approach'] = 'limited'
            recommendations['memory_warnings'].append("SIDA may have reduced performance with <16GB GPU memory")
            recommendations['performance_tips'].append("Consider using smaller SIDA models or batch processing")
        elif memory_gb >= 8:
            recommendations['can_run_sid_a'] = False
            recommendations['recommended_approach'] = 'photoholmes_only'
            recommendations['memory_warnings'].append("Insufficient GPU memory for SIDA (<8GB)")
            recommendations['performance_tips'].append("Use PhotoHolmes-only workflow for cost optimization")
        else:
            recommendations['can_run_sid_a'] = False
            recommendations['recommended_approach'] = 'photoholmes_only'
            recommendations['memory_warnings'].append("Insufficient GPU memory for advanced models")
            recommendations['performance_tips'].append("Consider CPU-only processing or cloud deployment")

        # CPU fallback recommendations
        if device_info.device_type == DeviceType.CPU:
            recommendations['recommended_approach'] = 'photoholmes_cpu'
            recommendations['performance_tips'].append("CPU-only processing is available but slower")
            recommendations['performance_tips'].append("Consider cloud GPU deployment for better performance")

        return recommendations

    def _get_memory_in_gb(self, memory_mb: Optional[int]) -> float:
        """Convert memory from MB to GB"""
        return (memory_mb or 0) / 1024.0

    def _get_device_memory(self, device_info: DeviceInfo) -> Optional[int]:
        """Try to get device memory if not available"""
        try:
            if device_info.device_type == DeviceType.CUDA:
                # Try nvidia-smi for more accurate memory info
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=memory.total', '--id={}'.format(device_info.device_id),
                     '--format=csv,noheader,nounits'],
                    capture_output=True, text=True, timeout=10
                )
                if result.returncode == 0:
                    memory_str = result.stdout.strip()
                    if memory_str.isdigit():
                        return int(memory_str)

            elif device_info.device_type == DeviceType.MPS:
                # MPS doesn't have traditional GPU memory, estimate based on system RAM
                import psutil
                total_memory = psutil.virtual_memory().total // (1024 * 1024 * 1024)  # GB
                return total_memory // 4  # Estimate 25% of system RAM for GPU

            return device_info.memory_total
        except Exception:
            return device_info.memory_total

    def _detect_cuda_devices(self) -> List[DeviceInfo]:
        """Detect NVIDIA CUDA devices"""
        devices = []

        try:
            # Check for nvidia-smi
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name,memory.total,driver_version',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=10
            )

            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                for i, line in enumerate(lines):
                    parts = line.split(', ')
                    if len(parts) >= 3:
                        device = DeviceInfo(
                            device_type=DeviceType.CUDA,
                            device_id=i,
                            device_name=parts[0].strip(),
                            memory_total=int(parts[1].strip()) if parts[1].strip().isdigit() else None,
                            driver_version=parts[2].strip()
                        )
                        devices.append(device)
                self.logger.info(f"Detected {len(devices)} CUDA devices")
            else:
                self.logger.warning("nvidia-smi command failed - no CUDA devices detected")

        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError, PermissionError, OSError):
            self.logger.info("NVIDIA drivers not found or nvidia-smi unavailable")

        # Fallback: Try to detect via PyTorch
        if not devices:
            try:
                import torch
                if torch.cuda.is_available():
                    for i in range(torch.cuda.device_count()):
                        props = torch.cuda.get_device_properties(i)
                        device = DeviceInfo(
                            device_type=DeviceType.CUDA,
                            device_id=i,
                            device_name=props.name,
                            memory_total=props.total_memory // (1024*1024),  # Convert to MB
                            compute_capability=f"{props.major}.{props.minor}"
                        )
                        devices.append(device)
                    self.logger.info(f"Detected {len(devices)} CUDA devices via PyTorch")
            except ImportError:
                self.logger.info("PyTorch not available for CUDA detection")

        return devices

    def _detect_mps_devices(self) -> List[DeviceInfo]:
        """Detect Apple Metal Performance Shaders devices"""
        devices = []

        try:
            import torch
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = DeviceInfo(
                    device_type=DeviceType.MPS,
                    device_id=0,
                    device_name="Apple Metal Performance Shaders"
                )
                devices.append(device)
                self.logger.info("Detected MPS device (Apple Silicon)")
        except ImportError:
            self.logger.info("PyTorch not available for MPS detection")

        return devices

    def _detect_rocm_devices(self) -> List[DeviceInfo]:
        """Detect AMD ROCm devices"""
        devices = []

        # Method 1: Check for rocm-smi command
        try:
            result = subprocess.run(
                ['rocm-smi', '--showproductname'],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                # Parse GPU information
                gpu_info = {}
                try:
                    # Get detailed GPU info
                    result = subprocess.run(
                        ['rocm-smi', '--showtemp', '--showmeminfo', 'vram', '--json'],
                        capture_output=True, text=True, timeout=10
                    )
                    if result.returncode == 0:
                        import json
                        gpu_info = json.loads(result.stdout)
                except:
                    pass

                device_name = "AMD ROCm GPU"
                memory_mb = None

                # Extract GPU name and memory if available
                if isinstance(gpu_info, dict) and 'card' in gpu_info:
                    card_info = list(gpu_info['card'].values())[0]
                    if 'VRAM Total Memory' in card_info:
                        memory_mb = int(card_info['VRAM Total Memory'].split()[0])
                    # GPU name might be in other fields

                device = DeviceInfo(
                    device_type=DeviceType.ROCM,
                    device_id=0,
                    device_name=device_name,
                    memory_total=memory_mb,
                    driver_version=self._get_rocm_version()
                )
                devices.append(device)
                self.logger.info(f"Detected ROCm device via rocm-smi: {device_name}")
                return devices
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
            self.logger.debug("rocm-smi not available or failed")

        # Method 2: Check ROCm environment variables
        rocm_path = os.environ.get('ROCM_PATH', '/opt/rocm')
        if os.path.exists(f"{rocm_path}/bin/rocm-smi"):
            device = DeviceInfo(
                device_type=DeviceType.ROCM,
                device_id=0,
                device_name="AMD ROCm (env detected)",
                memory_total=None  # Will be detected later
            )
            devices.append(device)
            self.logger.info("Detected ROCm device via environment")
            return devices

        # Method 3: Check HSA environment variables
        if os.environ.get('HSA_OVERRIDE_GFX_VERSION') or os.path.exists('/dev/kfd'):
            device = DeviceInfo(
                device_type=DeviceType.ROCM,
                device_id=0,
                device_name="AMD ROCm (HSA detected)",
                memory_total=None
            )
            devices.append(device)
            self.logger.info("Detected ROCm device via HSA")
            return devices

        # Method 4: Check PyTorch ROCm support
        try:
            import torch
            if hasattr(torch.version, 'hip') and torch.version.hip:
                # Try to get device properties
                device_name = "AMD ROCm (PyTorch)"
                memory_mb = None
                try:
                    if torch.cuda.is_available():  # For ROCm, this checks HIP
                        props = torch.cuda.get_device_properties(0)
                        device_name = props.name
                        memory_mb = props.total_memory // (1024 * 1024)
                except:
                    pass

                device = DeviceInfo(
                    device_type=DeviceType.ROCM,
                    device_id=0,
                    device_name=device_name,
                    memory_total=memory_mb
                )
                devices.append(device)
                self.logger.info(f"Detected ROCm device via PyTorch: {device_name}")
                return devices
        except ImportError:
            self.logger.debug("PyTorch not available for ROCm detection")

        self.logger.info("No ROCm devices detected")
        return devices

    def _get_rocm_version(self) -> Optional[str]:
        """Get ROCm version"""
        try:
            result = subprocess.run(
                ['rocm-smi', '--showdriverversion'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    if 'Driver version' in line:
                        return line.split(':')[1].strip()
        except:
            pass

        # Try to get version from ROCm installation
        rocm_path = os.environ.get('ROCM_PATH', '/opt/rocm')
        version_file = f"{rocm_path}/.info/version"
        if os.path.exists(version_file):
            try:
                with open(version_file, 'r') as f:
                    return f.read().strip()
            except:
                pass

        return None

    def _get_cpu_device(self) -> DeviceInfo:
        """Get CPU device information"""
        return DeviceInfo(
            device_type=DeviceType.CPU,
            device_id=0,
            device_name=platform.processor() or "CPU"
        )

    def _validate_cuda_device(self, device: DeviceInfo, issues: List[str],
                            warnings: List[str], recommendations: List[str]):
        """Validate CUDA-specific requirements"""

        # Check compute capability
        if device.compute_capability:
            major_version = int(device.compute_capability.split('.')[0])
            if major_version < 6:
                issues.append(f"Compute capability {device.compute_capability} is too old (minimum 6.0)")

        # Check memory
        if device.memory_total and device.memory_total < 4000:  # 4GB minimum
            warnings.append(f"Low GPU memory ({device.memory_total}MB) - may cause OOM errors")

        # Check driver version
        if device.driver_version:
            version_parts = device.driver_version.split('.')
            if len(version_parts) >= 2:
                major_minor = f"{version_parts[0]}.{version_parts[1]}"
                try:
                    if float(major_minor) < 460.0:  # Minimum CUDA driver version
                        warnings.append("Consider updating NVIDIA drivers for better performance")
                except ValueError:
                    pass

    def _validate_mps_device(self, device: DeviceInfo, issues: List[str],
                           warnings: List[str], recommendations: List[str]):
        """Validate MPS-specific requirements"""

        # Check if running on macOS
        if platform.system() != "Darwin":
            issues.append("MPS is only available on macOS")

        # Check if Apple Silicon
        if not platform.machine().startswith(('arm64', 'arm')):
            issues.append("MPS requires Apple Silicon (M1/M2/M3)")

    def _validate_rocm_device(self, device: DeviceInfo, issues: List[str],
                            warnings: List[str], recommendations: List[str]):
        """Validate ROCm-specific requirements"""

        # Check if running on Linux
        if platform.system() != "Linux":
            issues.append("ROCm is only available on Linux")

    def _validate_gpu_common(self, device: DeviceInfo, issues: List[str],
                           warnings: List[str], recommendations: List[str]):
        """Common GPU validation checks"""

        # Memory checks
        if device.memory_total:
            if device.memory_total < 2000:
                recommendations.append("Consider using CPU for better performance with low GPU memory")
            elif device.memory_total < 8000:
                recommendations.append("Monitor GPU memory usage for large models")

        # Environment variables
        cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
        if device.device_type == DeviceType.CUDA and cuda_visible_devices:
            try:
                visible_ids = [int(x.strip()) for x in cuda_visible_devices.split(',')]
                if device.device_id not in visible_ids:
                    issues.append(f"Device {device.device_id} not in CUDA_VISIBLE_DEVICES={cuda_visible_devices}")
            except ValueError:
                warnings.append(f"Invalid CUDA_VISIBLE_DEVICES format: {cuda_visible_devices}")


# Global device manager instance
device_manager = DeviceManager()

# Convenience function for getting recommended device
def get_recommended_device(prefer_gpu: bool = True) -> DeviceInfo:
    """
    Get the recommended device for computation.

    Args:
        prefer_gpu: Whether to prefer GPU over CPU when both are available

    Returns:
        Recommended device information
    """
    return device_manager.get_recommended_device(prefer_gpu)