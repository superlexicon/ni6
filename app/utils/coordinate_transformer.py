"""
Coordinate transformer for handling landscape orientation.
Transforms coordinates from landscape to portrait orientation.
"""

from typing import Dict, Any, List, Tuple
from dataclasses import dataclass


@dataclass
class OrientationInfo:
    """Information about document orientation."""
    is_landscape: bool
    rotation: int
    width: float
    height: float
    aspect_ratio: float


class CoordinateTransformer:
    """Transforms coordinates for landscape documents."""

    @staticmethod
    def transform_coordinate(
        value: float,
        axis: str,
        orientation: OrientationInfo
    ) -> float:
        """
        Transform a single coordinate value.

        For landscape rotated 90° clockwise:
        - x becomes y (inverted)
        - y becomes x

        Args:
            value: Coordinate value (0-1 normalized)
            axis: 'x' or 'y'
            orientation: Document orientation info

        Returns:
            Transformed coordinate value
        """
        if not orientation.is_landscape:
            return value

        # For 90° clockwise rotation (most common landscape)
        if orientation.rotation == 90 or (orientation.is_landscape and orientation.rotation == 0):
            if axis == 'x':
                # New x = 1 - old y
                return 1.0 - value
            else:  # axis == 'y'
                # New y = old x
                return value

        # For 270° rotation (landscape 90° counter-clockwise)
        if orientation.rotation == 270:
            if axis == 'x':
                # New x = old y
                return value
            else:  # axis == 'y'
                # New y = 1 - old x
                return 1.0 - value

        # For 180° rotation (upside down portrait)
        if orientation.rotation == 180:
            # Both axes inverted
            return 1.0 - value

        return value

    @staticmethod
    def transform_bbox(
        x1: float, y1: float, x2: float, y2: float,
        orientation: OrientationInfo
    ) -> Tuple[float, float, float, float]:
        """
        Transform bounding box coordinates.

        Args:
            x1, y1, x2, y2: Bounding box coordinates (0-1 normalized)
            orientation: Document orientation info

        Returns:
            Transformed (x1, y1, x2, y2)
        """
        if not orientation.is_landscape:
            return x1, y1, x2, y2

        # Transform all four corners
        new_x1 = CoordinateTransformer.transform_coordinate(x1, 'x', orientation)
        new_y1 = CoordinateTransformer.transform_coordinate(y1, 'y', orientation)
        new_x2 = CoordinateTransformer.transform_coordinate(x2, 'x', orientation)
        new_y2 = CoordinateTransformer.transform_coordinate(y2, 'y', orientation)

        # Ensure x1 < x2 and y1 < y2 (swap if needed)
        if new_x1 > new_x2:
            new_x1, new_x2 = new_x2, new_x1
        if new_y1 > new_y2:
            new_y1, new_y2 = new_y2, new_y1

        return new_x1, new_y1, new_x2, new_y2

    @staticmethod
    def transform_geometry_dict(
        geometry: Dict[str, float],
        orientation: OrientationInfo
    ) -> Dict[str, float]:
        """
        Transform a geometry dictionary.

        Args:
            geometry: Dict with 'x1', 'y1', 'x2', 'y2', 'x', 'y', 'width', 'height' keys
            orientation: Document orientation info

        Returns:
            Transformed geometry dictionary
        """
        if not orientation.is_landscape:
            return geometry

        result = geometry.copy()

        # Transform bounding box
        x1, y1, x2, y2 = CoordinateTransformer.transform_bbox(
            geometry.get('x1', 0),
            geometry.get('y1', 0),
            geometry.get('x2', 0),
            geometry.get('y2', 0),
            orientation
        )

        result['x1'] = x1
        result['y1'] = y1
        result['x2'] = x2
        result['y2'] = y2

        # Transform point coordinates if present
        if 'x' in geometry:
            result['x'] = CoordinateTransformer.transform_coordinate(
                geometry['x'], 'x', orientation
            )
        if 'y' in geometry:
            result['y'] = CoordinateTransformer.transform_coordinate(
                geometry['y'], 'y', orientation
            )

        # Swap width/height for landscape
        if 'width' in geometry and 'height' in geometry:
            result['width'] = geometry.get('height', 0)
            result['height'] = geometry.get('width', 0)

        return result

    @staticmethod
    def transform_text_blocks(
        text_blocks: List[Dict[str, Any]],
        orientation: OrientationInfo
    ) -> List[Dict[str, Any]]:
        """
        Transform all text blocks for landscape orientation.

        Args:
            text_blocks: List of dicts with geometry data
            orientation: Document orientation info

        Returns:
            List of text blocks with transformed coordinates
        """
        if not orientation.is_landscape:
            return text_blocks

        transformed = []
        for block in text_blocks:
            new_block = block.copy()
            new_block.update(
                CoordinateTransformer.transform_geometry_dict(
                    {k: v for k, v in block.items() if k in ['x1', 'y1', 'x2', 'y2', 'x', 'y', 'width', 'height']},
                    orientation
                )
            )
            # Also transform word_details if present
            if 'word_details' in block:
                transformed_words = []
                for word in block['word_details']:
                    transformed_word = word.copy()
                    transformed_word.update(
                        CoordinateTransformer.transform_geometry_dict(
                            {k: v for k, v in word.items() if k in ['x1', 'y1', 'x2', 'y2', 'x', 'y', 'width', 'height']},
                            orientation
                        )
                    )
                    transformed_words.append(transformed_word)
                new_block['word_details'] = transformed_words
            transformed.append(new_block)

        return transformed
