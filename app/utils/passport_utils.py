import hashlib


def hash_passport(passport_country: str, passport_number: str) -> str:
    """
    [DEPRECATED] Generate SHA-256 hash of passport_country + passport_number.

    DEPRECATED: passport_hash column removed from user_identity_index table.
    Face biometrics is now used as the primary identity uniqueness constraint
    via the trg_face_biometrics_cross_identity_check trigger.

    This function is kept for backward compatibility only and should not be used.

    Args:
        passport_country: Country code (ISO 3166-1 alpha-3)
        passport_number: Passport number

    Returns:
        SHA-256 hash of the combined passport data
    """
    combined = f"{passport_country}:{passport_number}"
    return hashlib.sha256(combined.encode()).hexdigest()
