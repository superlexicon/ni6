import os

from dotenv import load_dotenv

load_dotenv(override=False)  # Don't override command-line env vars


class GenerateSeed:
    @staticmethod
    def get_seed():
        seed = os.getenv("SEED")
        if not seed or not seed.strip():
            raise ValueError(
                "SEED environment variable is not set. "
                "Please set SEED in your .env file for deterministic key generation. "
                "Job routing requires consistent keys across server restarts."
            )
        return seed.strip()
