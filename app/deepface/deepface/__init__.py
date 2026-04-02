# Add the parent directory to sys.path so 'from deepface.xxx' imports work
# This allows the forked deepface to use its original absolute imports
import sys
import os
_deepface_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _deepface_parent not in sys.path:
    sys.path.insert(0, _deepface_parent)

__version__ = "0.0.97"
