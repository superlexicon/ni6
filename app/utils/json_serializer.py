"""
JSON Serializer with datetime support.

Provides a custom JSON encoder that handles datetime, date, and time objects
by converting them to ISO format strings.
"""

import json
from datetime import datetime, date, time
from typing import Any


class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder that converts datetime objects to ISO format strings."""

    def default(self, obj: Any) -> Any:
        """Convert datetime/date/time objects to ISO format strings."""
        if isinstance(obj, (datetime, date, time)):
            return obj.isoformat()
        return super().default(obj)


def dumps_datetime(obj: Any, **kwargs) -> str:
    """
    Serialize object to JSON string, converting datetime objects to ISO format.

    Args:
        obj: Object to serialize
        **kwargs: Additional arguments to pass to json.dumps()

    Returns:
        JSON string with datetime objects converted to ISO format
    """
    kwargs.setdefault('cls', DateTimeEncoder)
    return json.dumps(obj, **kwargs)


def serialize_datetime_dict(data: dict) -> dict:
    """
    Recursively convert datetime objects in a dictionary to ISO format strings.

    This is useful when you want to pre-process a dict before JSON serialization,
    or when using a JSON library that doesn't support custom encoders.

    Args:
        data: Dictionary that may contain datetime objects

    Returns:
        Dictionary with datetime objects converted to ISO format strings
    """
    if not isinstance(data, dict):
        return data

    result = {}
    for key, value in data.items():
        if isinstance(value, (datetime, date, time)):
            result[key] = value.isoformat()
        elif isinstance(value, dict):
            result[key] = serialize_datetime_dict(value)
        elif isinstance(value, list):
            result[key] = [
                serialize_datetime_dict(item) if isinstance(item, dict)
                else item.isoformat() if isinstance(item, (datetime, date, time))
                else item
                for item in value
            ]
        else:
            result[key] = value

    return result
