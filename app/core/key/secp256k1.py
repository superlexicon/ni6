import base64
import time
from ecdsa import SigningKey, SECP256k1
from app.dto import ServerKeyPair


class KeyPair:
    @staticmethod
    def generate_secp256k1_keys() -> ServerKeyPair:
        # Get seed from environment or generate random
        from app.core.key.seed_generator import GenerateSeed
        seed_value = GenerateSeed.get_seed()
        seed = base64.b64decode(seed_value.strip()) if isinstance(seed_value, str) else seed_value

        # Derive private key from seed
        from ecdsa.util import randrange_from_seed__trytryagain
        secret_exponent = randrange_from_seed__trytryagain(
            seed, SECP256k1.order)
        private_key = SigningKey.from_secret_exponent(
            secret_exponent, curve=SECP256k1)
        if private_key is None:
            raise ValueError(
                "Private key generation failed: private_key is None.")

        # Generate public key
        public_key = private_key.get_verifying_key()
        if public_key is None:
            raise ValueError(
                "Public key generation failed: public_key is None.")

        # Convert keys to hex
        private_key_hex = private_key.to_string().hex()
        public_key_hex = public_key.to_string().hex()
        return ServerKeyPair(
            public_key=public_key_hex,
            private_key=private_key_hex,
            seed=seed.hex()
        )
