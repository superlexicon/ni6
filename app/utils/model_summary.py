"""
PhotoHolmes Model Summary Utility

Provides comprehensive overview of all available forgery detection methods,
their weights, status, and system capabilities.
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

class ModelSummary:
    """Comprehensive PhotoHolmes methods summary and status reporting."""

    def __init__(self):
        self.weights_dir = Path("app/photoholmes/weights")
        self.methods_data = self._get_methods_data()

    def _get_methods_data(self) -> List[Dict]:
        """Get comprehensive data for all PhotoHolmes methods."""
        return [
            {
                "name": "DQ",
                "category": "Forgery Detection",
                "framework": "PyTorch",
                "description": "Deep learning-free method based on quantification tables",
                "weights_files": [],
                "memory_tier": "Lightweight",
                "weights_available": False,
                "requires_weights": False,
                "license": "Open Source"
            },
            {
                "name": "Adaptive CFA Net",
                "category": "Forgery Detection",
                "framework": "PyTorch",
                "description": "Color filter array forgery detection",
                "weights_files": ["adaptive_cfa_net/weights.pth"],
                "memory_tier": "Moderate",
                "weights_available": self._check_weights_exist(["adaptive_cfa_net/weights.pth"]),
                "requires_weights": True,
                "license": "Research"
            },
            {
                "name": "NoiseSniffer",
                "category": "Photoshop Detection",
                "framework": "PyTorch",
                "description": "Detects inconsistencies in image noise patterns",
                "weights_files": [],
                "memory_tier": "Lightweight",
                "weights_available": False,
                "requires_weights": False,
                "license": "Open Source"
            },
            {
                "name": "PSCCNet",
                "category": "Photoshop Detection",
                "framework": "PyTorch",
                "description": "Photo consistency classification network",
                "weights_files": [
                    "psccnet/FENet.pth",
                    "psccnet/SegNet.pth",
                    "psccnet/ClsNet.pth"
                ],
                "memory_tier": "Memory Intensive",
                "weights_available": self._check_weights_exist([
                    "psccnet/FENet.pth",
                    "psccnet/SegNet.pth",
                    "psccnet/ClsNet.pth"
                ]),
                "requires_weights": True,
                "license": "Open Source"
            },
            {
                "name": "Focal",
                "category": "Forgery Detection",
                "framework": "PyTorch",
                "description": "Multi-scale focal forgery detection",
                "weights_files": [
                    "focal/HRNet_weights.pth",
                    "focal/VIT_weights.pth"
                ],
                "memory_tier": "Memory Intensive",
                "weights_available": self._check_weights_exist([
                    "focal/HRNet_weights.pth",
                    "focal/VIT_weights.pth"
                ]),
                "requires_weights": True,
                "license": "Open Source"
            },
            {
                "name": "Splicebuster",
                "category": "Forgery Detection",
                "framework": "ONNX Runtime",
                "description": "CNN-based forgery detection",
                "weights_files": [],
                "memory_tier": "Moderate",
                "weights_available": False,
                "requires_weights": False,
                "license": "Research Only"
            },
            {
                "name": "TruFor",
                "category": "Forgery Detection",
                "framework": "PyTorch",
                "description": "Transformer-based with noise-sensitive fingerprint",
                "weights_files": [],
                "memory_tier": "Moderate",
                "weights_available": False,
                "requires_weights": False,  # Research-only, requires external weights
                "license": "Research Only"
            },
            {
                "name": "ZERO",
                "category": "Forgery Detection",
                "framework": "ONNX Runtime",
                "description": "Zero-shot forgery detection",
                "weights_files": [],
                "memory_tier": "Memory Intensive",
                "weights_available": False,
                "requires_weights": False,
                "license": "Research Only"
            },
            {
                "name": "EXIF As Language",
                "category": "Forgery Detection",
                "framework": "PyTorch",
                "description": "EXIF metadata analysis for forgery detection",
                "weights_files": ["exif_as_language/weights.pth"],
                "memory_tier": "Memory Intensive",
                "weights_available": self._check_weights_exist(["exif_as_language/weights.pth"]),
                "requires_weights": True,
                "license": "Open Source"
            }
        ]

    def _check_weights_exist(self, weight_files: List[str]) -> bool:
        """Check if weight files exist."""
        return all((self.weights_dir / wf).exists() for wf in weight_files)

    def _get_weights_size(self, weight_files: List[str]) -> float:
        """Get total size of weight files in MB."""
        total_size = 0
        for wf in weight_files:
            file_path = self.weights_dir / wf
            if file_path.exists():
                total_size += file_path.stat().st_size / (1024 * 1024)  # Convert to MB
        return total_size

    def _calculate_total_weights_size(self) -> float:
        """Calculate total size of all available weights."""
        total_size = 0
        for method in self.methods_data:
            if method["weights_available"]:
                total_size += self._get_weights_size(method["weights_files"])
        return total_size

    def print_startup_summary(self, loaded_methods: List[str], failed_methods: List[str],
                            skipped_methods: List[str], device_caps: Dict, recommendations: Dict):
        """Print comprehensive startup summary."""
        print("\n" + "="*80)
        print("🔬 PHOTOHOLMES FORGERY DETECTION SYSTEM")
        print("="*80)

        # System Overview
        print(f"\n🖥️  SYSTEM OVERVIEW")
        print(f"   Device: {device_caps.get('device', 'Unknown').upper()}")
        print(f"   GPU Memory: {device_caps.get('gpu_memory_gb', 0):.1f} GB")
        print(f"   System Memory: {device_caps.get('system_memory_gb', 0):.1f} GB")
        print(f"   CPU Cores: {device_caps.get('cpu_cores', 0)}")
        print(f"   Available Frameworks: {'PyTorch' if device_caps.get('has_mps') else 'CUDA' if device_caps.get('has_cuda') else 'CPU'}")

        # Weights Overview
        total_weights_size = self._calculate_total_weights_size()
        print(f"\n💾 AVAILABLE MODEL WEIGHTS")
        print(f"   Total Methods: {len(self.methods_data)}")
        print(f"   Weights Available: {sum(1 for m in self.methods_data if m['weights_available'])}/{len(self.methods_data)}")
        print(f"   Total Weights Size: {total_weights_size:.1f} MB")
        print(f"   Weights Directory: {self.weights_dir}")

        # Method Categories
        forgery_methods = [m for m in self.methods_data if m['category'] == 'Forgery Detection']
        photoshop_methods = [m for m in self.methods_data if m['category'] == 'Photoshop Detection']

        print(f"\n📊 METHOD DISTRIBUTION")
        print(f"   Forgery Detection: {len(forgery_methods)} methods")
        print(f"   Photoshop Detection: {len(photoshop_methods)} methods")
        print(f"   PyTorch-based: {len([m for m in self.methods_data if m['framework'] == 'PyTorch'])}")
        print(f"   ONNX-based: {len([m for m in self.methods_data if m['framework'] == 'ONNX Runtime'])}")

        # Detailed Method Table
        print(f"\n🔍 METHOD STATUS MATRIX")
        print(f"{'Method':<20} {'Category':<18} {'Framework':<12} {'Weights':<8} {'Status':<10}")
        print("-" * 80)

        for method in self.methods_data:
            name = method['name'][:18] + ".." if len(method['name']) > 20 else method['name']
            category = method['category'][:16] + ".." if len(method['category']) > 18 else method['category']
            framework = method['framework'][:10] + ".." if len(method['framework']) > 12 else method['framework']

            if method['requires_weights']:
                weights_status = "✅ Available" if method['weights_available'] else "❌ Missing"
            else:
                weights_status = "🔧 Algorithmic"

            # Determine current status
            if method['name'] in loaded_methods:
                status = "✅ Loaded"
            elif method['name'] in failed_methods:
                status = "❌ Failed"
            elif method['name'] in skipped_methods:
                status = "⏸️ Skipped"
            elif not method['weights_available'] and method['requires_weights']:
                status = "⚠️ Skipped"
            elif method['name'] == 'EXIF As Language':
                status = "⚪ Disabled"
            elif method['name'] == 'ZERO' and not recommendations.get('zero', False):
                status = "⚪ Disabled"
            else:
                status = "⚪ Available"

            print(f"{name:<20} {category:<18} {framework:<12} {weights_status:<8} {status:<10}")

        # Memory Analysis
        memory_tiers = {"Memory Intensive": 0, "Moderate": 0, "Lightweight": 0}
        for method in self.methods_data:
            memory_tiers[method['memory_tier']] += 1

        print(f"\n💾 MEMORY REQUIREMENTS")
        print(f"   Memory Intensive (>1GB): {memory_tiers['Memory Intensive']} methods")
        print(f"   Moderate (500MB-1GB): {memory_tiers['Moderate']} methods")
        print(f"   Lightweight (<500MB): {memory_tiers['Lightweight']} methods")

        # Current Session Status
        print(f"\n🚀 CURRENT SESSION STATUS")
        print(f"   Successfully Loaded: {len(loaded_methods)} methods")
        print(f"   Failed to Load: {len(failed_methods)} methods")
        print(f"   Skipped: {len(skipped_methods)} methods")
        print(f"   Success Rate: {(len(loaded_methods) / len(self.methods_data) * 100):.1f}%")

        if loaded_methods:
            print(f"   Active Methods: {', '.join(loaded_methods)}")

        if failed_methods:
            print(f"   Failed Methods: {', '.join(failed_methods)}")

        if skipped_methods:
            print(f"   Skipped Methods: {', '.join(skipped_methods)}")

        # License Information
        research_methods = [m for m in self.methods_data if m['license'] == 'Research Only']
        if research_methods:
            print(f"\n⚠️  RESEARCH LICENSES")
            print(f"   {len(research_methods)} methods require research-only agreements:")
            print(f"   {', '.join(m['name'] for m in research_methods)}")

        # Performance Expectations
        print(f"\n📈 PERFORMANCE EXPECTATIONS")
        if device_caps.get('has_cuda'):
            print(f"   🎯 CUDA Device Detected: All methods will be GPU-accelerated")
            print(f"   ⚡ Expected Performance: 5-7x faster than CPU")
        elif device_caps.get('has_mps'):
            print(f"   🍎 MPS Device Detected: Lightweight methods GPU-accelerated")
            print(f"   ⚡ Expected Performance: 2-3x faster than CPU")
        else:
            print(f"   💻 CPU Only: Methods will run on CPU")
            print(f"   ⏱️  Expected Performance: Baseline processing speed")

        print("="*80)
        print(f"🎉 PhotoHolmes System Ready - {len(loaded_methods)}/{len(self.methods_data)} methods active")
        print("="*80 + "\n")