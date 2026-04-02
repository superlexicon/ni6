#!/usr/bin/env python3
"""
Compute public keys from seeds for internal API authentication.

This script outputs bash variable assignments for:
- GENERATED_SEEDS: Array of base64-encoded seeds
- GENERATED_PUBLIC_KEYS: Array of hex-encoded public keys

Usage:
    eval $(poetry run python3 deploy/compute_keys.py)
"""

import base64
import secrets
from ecdsa import SECP256k1, SigningKey
from ecdsa.util import randrange_from_seed__trytryagain


def generate_random_seed_and_public_key() -> tuple[str, str]:
    """
    Generate a random seed and compute its public key.

    Returns:
        Tuple of (base64_seed, hex_public_key)
    """
    # Generate 32 cryptographically secure random bytes
    seed_bytes = secrets.token_bytes(32)
    seed_b64 = base64.b64encode(seed_bytes).decode('ascii')

    # Compute public key from seed (same method as KeyPair.generate_secp256k1_keys)
    secret_exponent = randrange_from_seed__trytryagain(seed_bytes, SECP256k1.order)
    private_key = SigningKey.from_secret_exponent(secret_exponent, curve=SECP256k1)
    public_key = private_key.get_verifying_key()

    # Use uncompressed format (64 bytes = 128 hex chars) to match existing implementation
    public_key_hex = public_key.to_string().hex()

    return seed_b64, public_key_hex


def main():
    """Generate seeds and keys for 3 instances and output bash variables."""
    seeds = []
    public_keys = []

    for _ in range(3):
        seed, pub_key = generate_random_seed_and_public_key()
        seeds.append(seed)
        public_keys.append(pub_key)

    # Output bash variable assignments (simple format that works with eval)
    # Use space-separated strings that can be split into arrays
    print(f'GENERATED_SEEDS="{" ".join(seeds)}"')
    print(f'GENERATED_PUBLIC_KEYS="{" ".join(public_keys)}"')


if __name__ == "__main__":
    main()
