"""
Forgery Detection Thresholds Configuration

This file defines thresholds for each forgery detector based on document type.
Raw detector scores ABOVE these thresholds indicate potential problems.

Score Interpretation:
- DQ: Probability image is AI-generated (0-1, higher = more likely AI)
- Adaptive: Ratio of tampered pixels (0-1, higher = more tampering)
- NoiseSniffer: Manipulation confidence (0-1, higher = more manipulation)
- PSCCNET: Splicing/copy-paste confidence (0-1, higher = more splicing)
"""

FORGERY_THRESHOLDS = {
    'passport': {
        'dq': {
            'problem': 0.60,  # 60% AI-generated probability
            'warning': 0.50,  # 50% AI-generated probability
            'description': 'AI generation probability'
        },
        'adaptive': {
            'problem': 0.60,  # 60% tampered pixels
            'warning': 0.50,  # 50% tampered pixels
            'description': 'Tampered pixel ratio'
        },
        'noisesniffer': {
            'problem': 0.45,  # 45% manipulation confidence
            'warning': 0.35,  # 35% manipulation confidence
            'description': 'Manipulation confidence'
        },
        'psccnet': {
            'problem': 0.20,  # 20% splicing confidence
            'warning': 0.15,  # 15% splicing confidence
            'description': 'Splicing/copy-paste confidence'
        }
    },

    'selfie': {
        'dq': {
            'problem': 0.65,
            'warning': 0.55,
            'description': 'AI generation probability'
        },
        'adaptive': {
            'problem': 0.65,
            'warning': 0.55,
            'description': 'Tampered pixel ratio'
        },
        'noisesniffer': {
            'problem': 0.55,
            'warning': 0.45,
            'description': 'Manipulation confidence'
        },
        'psccnet': {
            'problem': 0.35,
            'warning': 0.25,
            'description': 'Splicing/copy-paste confidence'
        }
    },

    'bank_statement': {
        'dq': {
            'problem': 0.55,
            'warning': 0.45,
            'description': 'AI generation probability'
        },
        'adaptive': {
            'problem': 0.55,
            'warning': 0.45,
            'description': 'Tampered pixel ratio'
        },
        'noisesniffer': {
            'problem': 0.60,
            'warning': 0.50,
            'description': 'Manipulation confidence'
        },
        'psccnet': {
            'problem': 0.35,
            'warning': 0.25,
            'description': 'Splicing/copy-paste confidence'
        }
    },

    'tax_statement': {
        'dq': {
            'problem': 0.55,
            'warning': 0.45,
            'description': 'AI generation probability'
        },
        'adaptive': {
            'problem': 0.55,
            'warning': 0.45,
            'description': 'Tampered pixel ratio'
        },
        'noisesniffer': {
            'problem': 0.60,
            'warning': 0.50,
            'description': 'Manipulation confidence'
        },
        'psccnet': {
            'problem': 0.35,
            'warning': 0.25,
            'description': 'Splicing/copy-paste confidence'
        }
    },

    'wealth_declaration': {
        'dq': {
            'problem': 0.55,
            'warning': 0.45,
            'description': 'AI generation probability'
        },
        'adaptive': {
            'problem': 0.55,
            'warning': 0.45,
            'description': 'Tampered pixel ratio'
        },
        'noisesniffer': {
            'problem': 0.60,
            'warning': 0.50,
            'description': 'Manipulation confidence'
        },
        'psccnet': {
            'problem': 0.35,
            'warning': 0.25,
            'description': 'Splicing/copy-paste confidence'
        }
    },

    # Default thresholds for unknown document types
    'default': {
        'dq': {
            'problem': 0.60,
            'warning': 0.50,
            'description': 'AI generation probability'
        },
        'adaptive': {
            'problem': 0.60,
            'warning': 0.50,
            'description': 'Tampered pixel ratio'
        },
        'noisesniffer': {
            'problem': 0.50,
            'warning': 0.40,
            'description': 'Manipulation confidence'
        },
        'psccnet': {
            'problem': 0.30,
            'warning': 0.20,
            'description': 'Splicing/copy-paste confidence'
        }
    }
}


def get_thresholds(document_type: str) -> dict:
    """
    Get forgery detection thresholds for a specific document type.

    Args:
        document_type: Type of document (e.g., 'passport', 'selfie', 'bank_statement')

    Returns:
        Dictionary of thresholds for each detector
    """
    return FORGERY_THRESHOLDS.get(document_type, FORGERY_THRESHOLDS['default'])
