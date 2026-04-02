from ecdsa import VerifyingKey, SECP256k1, util
from hashlib import sha256
from typing import Optional


class ECDSARecovery:
    """ECDSA signature verification with public key recovery"""

    @staticmethod
    def recover_public_key_from_signature(
        message: str,
        r: str,
        s: str
    ) -> Optional[str]:
        """
        Recover public key from ECDSA signature using public key recovery algorithm.

        Args:
            message: The message that was signed (new_public_key hex string)
            r: Signature r component (hex string)
            s: Signature s component (hex string)

        Returns:
            Recovered public key as hex string, or None if recovery fails
        """
        try:
            # DEBUG: Log inputs
            print(f"🔍 [ECDSA Recovery] Input message: {message}")
            print(f"🔍 [ECDSA Recovery] r: {r[:32]}... (len={len(r)})")
            print(f"🔍 [ECDSA Recovery] s: {s[:32]}... (len={len(s)})")

            # Convert hex to bytes
            r_bytes = bytes.fromhex(r)
            s_bytes = bytes.fromhex(s)

            print(f"🔍 [ECDSA Recovery] r_bytes len: {len(r_bytes)}")
            print(f"🔍 [ECDSA Recovery] s_bytes len: {len(s_bytes)}")

            # Compute message hash (message is UTF-8 text, not hex)
            message_bytes = message.encode('utf-8')
            message_hash = sha256(message_bytes).digest()

            print(f"🔍 [ECDSA Recovery] message_hash: {message_hash.hex()}")

            # Convert r and s to integers
            r_int = int.from_bytes(r_bytes, byteorder='big')
            s_int = int.from_bytes(s_bytes, byteorder='big')

            # DER-encode the signature (required by ecdsa library)
            sig_bytes_der = util.sigencode_der_canonize(r_int, s_int, SECP256k1.order)

            print(f"🔍 [ECDSA Recovery] DER signature (hex): {sig_bytes_der.hex()[:64]}... (len={len(sig_bytes_der)})")

            # Use from_public_key_recovery_with_digest which returns candidate keys
            # The library tries all recovery IDs internally
            candidate_keys = VerifyingKey.from_public_key_recovery_with_digest(
                sig_bytes_der,
                message_hash,
                SECP256k1,
                sigdecode=util.sigdecode_der
            )

            print(f"🔍 [ECDSA Recovery] Found {len(candidate_keys)} candidate keys")

            # Try each candidate key to find the one that verifies
            for i, vk in enumerate(candidate_keys):
                try:
                    # Verify the signature with this recovered key
                    if vk.verify_digest(sig_bytes_der, message_hash, sigdecode=util.sigdecode_der):
                        recovered = vk.to_string().hex()
                        print(f"✅ [ECDSA Recovery] SUCCESS with candidate {i}")
                        print(f"✅ [ECDSA Recovery] Recovered key: {recovered[:32]}...")
                        return recovered
                except Exception as e:
                    print(f"⚠️ [ECDSA Recovery] Candidate {i} failed verification: {str(e)[:80]}")
                    continue

            print(f"❌ [ECDSA Recovery] All {len(candidate_keys)} candidates failed verification")
            return None

        except Exception as e:
            print(f"❌ [ECDSA Recovery] Exception: {str(e)}")
            raise ValueError(f"Public key recovery failed: {str(e)}")

    @staticmethod
    def verify_signature(
        message: str,
        r: str,
        s: str,
        public_key: str
    ) -> bool:
        """
        Verify ECDSA signature against a known public key.

        Directly verifies the signature without trying to recover the public key.
        This avoids the ambiguity where multiple valid public keys can be recovered
        from the same signature.

        Args:
            message: The message that was signed
            r: Signature r component (hex)
            s: Signature s component (hex)
            public_key: Expected public key (hex)

        Returns:
            True if signature is valid for the public key
        """
        try:
            from ecdsa import VerifyingKey, SECP256k1, util

            print(f"🔍 [ECDSA Verify] Verifying signature for public key: {public_key[:32]}...")
            print(f"🔍 [ECDSA Verify] Message: {message}")

            # Convert hex to bytes
            r_bytes = bytes.fromhex(r)
            s_bytes = bytes.fromhex(s)

            # Compute message hash
            message_bytes = message.encode('utf-8')
            message_hash = sha256(message_bytes).digest()

            print(f"🔍 [ECDSA Verify] Message hash: {message_hash.hex()}")

            # Convert r and s to integers
            r_int = int.from_bytes(r_bytes, byteorder='big')
            s_int = int.from_bytes(s_bytes, byteorder='big')

            # DER-encode the signature
            sig_bytes_der = util.sigencode_der_canonize(r_int, s_int, SECP256k1.order)

            # Create VerifyingKey from the provided public key
            # Public key is 64 bytes (32 bytes for X, 32 bytes for Y) without prefix
            public_key_bytes = bytes.fromhex(public_key)
            vk = VerifyingKey.from_string(
                public_key_bytes,
                curve=SECP256k1,
                hashfunc=sha256
            )

            # Verify the signature directly
            vk.verify_digest(
                sig_bytes_der,
                message_hash,
                sigdecode=util.sigdecode_der
            )

            print(f"✅ [ECDSA Verify] Signature is VALID")
            return True

        except Exception as e:
            print(f"❌ [ECDSA Verify] Exception: {str(e)}")
            import traceback
            print(f"❌ [ECDSA Verify] Traceback: {traceback.format_exc()}")
            return False
