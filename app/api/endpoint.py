from enum import Enum
from typing import Literal


class Endpoint(str, Enum):
    API_PREFIX = "/api"

    HEALTH = f"{API_PREFIX}/health"

    DETECT_FORGERY = f"{API_PREFIX}/forgery/detect"
    ANALYZE_RESUME = f"{API_PREFIX}/forgery/analyze"
    EXTRACT_DATA = f"{API_PREFIX}/forgery/extract"
    GET_KEY = f"{API_PREFIX}/key/public-key"
    CREATE_KEY = f"{API_PREFIX}/key/create"
    KEY_RECOVERY = f"{API_PREFIX}/key/recovery"
    DELETE_KEY_RECOVERY = f"{API_PREFIX}/key/recovery"
    OTP_NUMBER = f"{API_PREFIX}/otp/random-number/{{length}}"
    GET_ALL_VERIFICATION = f"{API_PREFIX}/verification"
    VERIFY = f"{API_PREFIX}/verification"
    UPDATE_MANUAL_CHECK = f"{API_PREFIX}/verification/{{email}}"
    DELETE_USER_PRESENCE = f"{API_PREFIX}/user/delete-presence"

    @classmethod
    def get_url(cls, endpoint: "Endpoint", base_url: str = "") -> str:
        return f"{base_url.rstrip('/')}{endpoint.value}"


EndpointType = Literal[
    "HEALTH",
    "DETECT_FORGERY",
    "ANALYZE_RESUME",
    "GET_KEY",
    "CREATE_KEY",
    "OTP_NUMBER",
    "GET_ALL_VERIFICATION",
    "VERIFY",
    "UPDATE_MANUAL_CHECK",
    "EXTRACT_DATA",
    "DELETE_USER_PRESENCE"
]
