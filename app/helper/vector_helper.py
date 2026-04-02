"""Helper for MariaDB VECTOR type operations."""
from typing import List
import struct


class VectorHelper:
    """Convert embeddings to/from MariaDB VECTOR binary format."""

    @staticmethod
    def to_binary(embedding: List[float]) -> bytes:
        """
        Convert embedding list to binary format for MariaDB VECTOR.
        MariaDB VECTOR uses 32-bit IEEE 754 floats.
        """
        return struct.pack(f'{len(embedding)}f', *embedding)

    @staticmethod
    def to_hex(embedding: List[float]) -> str:
        """
        Convert embedding to hex string for SQL queries.
        Usage in SQL: x'<hex_string>'
        """
        binary = VectorHelper.to_binary(embedding)
        return binary.hex()

    @staticmethod
    def from_binary(data: bytes) -> List[float]:
        """
        Convert MariaDB VECTOR binary back to list of floats.
        """
        count = len(data) // 4  # 4 bytes per float
        return list(struct.unpack(f'{count}f', data))
