"""
Integrated PhotoHolmes forgery detection methods.

This module contains forgery detection algorithms from PhotoHolmes,
integrated with the IM-OSINT application architecture and GPU resource management.

Available Methods:
- PSCCNet: Primary forgery detection method
- DQ: Additional detection method
- AdaptiveCFANet: Adaptive detection method
- Noisesniffer: Noise-based detection method
"""

from .methods.psccnet import PSCCNet
from .methods.dq import DQ
from .methods.adaptive_cfa_net import AdaptiveCFANet
from .methods.noisesniffer import Noisesniffer
from .methods.trufor import TruFor

__all__ = [
    'PSCCNet',
    'DQ',
    'AdaptiveCFANet',
    'Noisesniffer',
    'TruFor'
]