from fastapi import APIRouter, Path, status, HTTPException, Request

from app.api.endpoint import Endpoint
from app.dto import (
    DataResponse,
    PublicKeyResponse,
    UserShareKeyRequest
)
from app.dto.key import UserShareKeyResponse
from app.dto.add_device_key import (
    AddDeviceKeyRequest,
    AddDeviceKeyResponse
)
from app.services import key_service
from app.services.device_key_service import DeviceKeyService
from app.utils import http_exception_helper
from app.core.logger import get_logger

logger = get_logger()

router = APIRouter(tags=["KEY"])


@router.get(Endpoint.GET_KEY,
            response_model=DataResponse[PublicKeyResponse],
            status_code=status.HTTP_200_OK)
async def get_server_public_key():
    try:
        result = await key_service.get_server_public_key()
        return DataResponse[PublicKeyResponse](data=result)
    except Exception as exe:
        http_exception_helper.raise_for_exception(exe)

@router.post(Endpoint.CREATE_KEY,
             response_model=DataResponse[UserShareKeyResponse],
             status_code=status.HTTP_201_CREATED)
async def create_key(request: UserShareKeyRequest, http_request: Request = None):
    """
    Create a new user key with rate limiting.

    Rate Limiting:
    - 10 key creation requests per IP per hour
    - Prevents user enumeration and database pollution
    """
    try:
        # Apply rate limiting per IP
        limiter = http_request.app.state.limiter if http_request else None
        if limiter:
            limiter._check_request_limit(http_request, None, "10/hour")

        result = await key_service.create_key(request)
        return DataResponse[UserShareKeyResponse](data=result)
    except Exception as exe:
        http_exception_helper.raise_for_exception(exe)


@router.post(
    "/device-key/add",
    status_code=status.HTTP_201_CREATED,
    response_model=DataResponse[AddDeviceKeyResponse],
    summary="Add additional device key for multi-device support",
    description="""
    Add a new device key and secret share for a verified user.

    **Requirements:**
    - User must have completed all verification steps (selfie, passport, bank statement)
    - Request must include valid ECDSA signature of new public key
    - Signature must be from an existing registered device's private key
    - Public key must be unique (enforced by database index)

    **Authentication:**
    - Public key recovered from signature (r, s) and message hash
    - Recovered key matched against user_keys table
    - No explicit user identifier needed in request

    **Payload Encryption:**
    - Entire request payload must be encrypted with server's public key
    - Same pattern as other encrypted endpoints

    **Use Case:**
    User has completed verification on Device A and wants to add Device B.
    They sign the new public key (from Device B) with Device A's private key.
    """
)
async def add_device_key(
    request: AddDeviceKeyRequest,
    http_request: Request = None
) -> DataResponse[AddDeviceKeyResponse]:
    """
    Add a new device key for a verified user.

    This endpoint enables multi-device support by allowing users to register
    additional devices after completing the initial verification process.

    Rate Limiting:
    - 20 device key addition requests per IP per hour
    - Prevents device spam and account takeover attempts
    """
    try:
        # Apply rate limiting per IP
        limiter = http_request.app.state.limiter if http_request else None
        if limiter:
            limiter._check_request_limit(http_request, None, "20/hour")

        # Initialize services
        from app.repositories.user_key_repository import UserKeyRepository
        from app.services.verification_state_service import VerificationStateService

        user_key_repository = UserKeyRepository()
        verification_state_service = VerificationStateService()
        device_key_service = DeviceKeyService(
            user_key_repository,
            verification_state_service
        )

        # Add device key
        result = await device_key_service.add_device_key(
            new_public_key=request.new_public_key,
            secret_share=request.secret_share,
            signature_r=request.signature.r,
            signature_s=request.signature.s
        )

        # Build response
        response = AddDeviceKeyResponse(
            success=result['success'],
            public_key=result['public_key'],
            created_at=str(result['created_at']),
            message=result['message']
        )

        return DataResponse[AddDeviceKeyResponse](data=response)

    except ValueError as e:
        # Business logic errors (invalid signature, not verified, etc.)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        # Unexpected errors
        logger.error(f"Unexpected error in add_device_key: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to add device key: {str(e)}"
        )
