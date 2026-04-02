from typing import Optional
from fastapi import APIRouter, status, Path, Body, Query, Request, HTTPException
from pydantic import BaseModel, Field, field_validator
from app.api.endpoint import Endpoint
from app.dto import DataResponse, OTPResponse
from app.dto.otp import SignedOTPRequest, EncryptedOTPResponse
from app.utils import HTTPExceptionHelper
from app.services import otp_service
from app.services.signed_otp_service import SignedOTPService
from app.core.logger import get_logger

router = APIRouter(tags=["OTP"])
logger = get_logger()


class OTPRequest(BaseModel):
    """Request body for OTP generation"""
    mobile_number: str = Field(..., description="Mobile number WITH country code/prefix (e.g., +15551234567)")
    public_key: str = Field(..., description="Client public key for OTP binding (security)")

    @field_validator('mobile_number')
    @classmethod
    def normalize_mobile_number(cls, v: str) -> str:
        """Normalize mobile number by removing spaces."""
        return v.strip().replace(' ', '')


@router.post(
    Endpoint.OTP_NUMBER,
    response_model=DataResponse[OTPResponse],
    status_code=status.HTTP_200_OK,
)
async def generate_and_send_otp(
    length: int = Path(...),
    request: OTPRequest = Body(...),
    http_request: Request = None,
):
    """
    Generate an OTP code and send it via AWS SMS to the provided mobile number.

    Args:
        length: Length of the OTP code to generate
        request: OTP request body containing:
            - mobile_number: Mobile number with country code (e.g., +6512345678)
            - public_key: Client's public key for security binding

    Returns:
        OTPResponse with confirmation (actual OTP not returned for security)

    Note: OTP verification will be handled via selfie document processing where
    the OTP is extracted from the image using OCR and validated against both
    mobile_number and public_key for enhanced security.

    Rate Limiting:
    - 3 OTP requests per mobile number per 10 minutes
    - Prevents SMS bombing attacks

    **DEPRECATED:** Use /api/otp/ with SignedOTPRequest instead.
    """
    try:
        # Apply rate limiting per mobile number
        limiter = http_request.app.state.limiter if http_request else None
        if limiter:
            # Store mobile_number in request state for rate limiter key function
            http_request.state.mobile_number = request.mobile_number
            # Use the mobile number based rate limiting
            limiter._check_request_limit(http_request, lambda r: f"mobile:{request.mobile_number}", "3/10minutes")

        result = await otp_service.generate_and_send_otp_via_sms(
            length=length,
            mobile_number=request.mobile_number,
            client_public_key=request.public_key
        )
        return DataResponse[OTPResponse](data=result)
    except Exception as e:
        HTTPExceptionHelper.raise_for_exception(e)


@router.post(
    "/api/otp/",
    response_model=EncryptedOTPResponse,
    status_code=status.HTTP_200_OK,
)
async def request_signed_otp(
    request: SignedOTPRequest = Body(...),
    http_request: Request = None,
):
    """
    Request OTP with signature verification and encrypted response.

    **SECURE ENDPOINT** - Creates user_identity and user_keys records.

    **Security Features:**
    - ECDSA signature verification (proves key ownership)
    - Creates user_identity_index and user_keys records
    - Returns encrypted OTP (hybrid encryption)

    **Rate Limiting:** 3 requests per 10 minutes per mobile number

    **Request Body:**
    - client_public_key: Client's public key (hex)
    - mobile_number: Mobile number WITHOUT country code/prefix (e.g., '5551234567')
    - country_code: ISO country code (e.g., 'US', 'SG', 'GB'). Converted to phone prefix like '+1', '+65'
    - secret_share: Shamir secret share (format: '{index}:{base64}')
    - otp_length: OTP length (default 6)
    - timestamp: Unix timestamp for replay protection
    - signature: ECDSA signature of "otp:{timestamp}"
    - target_server_public_key: Server public key for response encryption
    - generate_otp: Whether to generate OTP (default: true). Set to false for instances that should only store secret shares.

    **Response (200 OK):**
    Encrypted envelope containing:
    - otp: The generated OTP code
    - otp_id: Unique identifier for this OTP
    - expires_at: Expiration timestamp
    - sent_at: When OTP was sent
    - user_identity_id: Created user identity ID

    **Authentication:**
    - Sign the message "otp:{timestamp}" with your private key
    - Timestamp must be within 5 minutes (replay protection)

    **Decryption:**
    - Client must decrypt response using their private key
    - Use hybrid crypto: ECDH for key exchange, AES-256-GCM for payload
    """
    try:
        # Apply rate limiting per mobile number
        limiter = http_request.app.state.limiter if http_request else None
        if limiter:
            # Store mobile_number in request state for rate limiter key function
            http_request.state.mobile_number = request.mobile_number
            # Use the mobile number based rate limiting
            limiter._check_request_limit(http_request, lambda r: f"mobile:{request.mobile_number}", "3/10minutes")

        # Create signed OTP service
        signed_otp_service = SignedOTPService()

        # Process signed OTP request
        result = await signed_otp_service.process_signed_request(
            client_public_key=request.client_public_key,
            mobile_number=request.mobile_number,
            country_code=request.country_code,
            secret_share=request.secret_share,
            secret_share_encrypted=request.secret_share_encrypted,
            timestamp=request.timestamp,
            signature_r=request.signature.r,
            signature_s=request.signature.s,
            target_server_public_key=request.target_server_public_key,
            otp_length=request.otp_length,
            generate_otp=request.generate_otp,
            gesture_mode=request.gesture_mode,
            device_id=request.device_id,
            api_url=request.api_url
        )

        if result['success']:
            logger.info(
                f"Signed OTP request processed for {request.mobile_number}, "
                f"user_identity_id: {result.get('user_identity_id')}"
            )
            return EncryptedOTPResponse(**result['encrypted_response'])
        else:
            raise HTTPException(
                status_code=400,
                detail=result['error']
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing signed OTP request: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process OTP request: {str(e)}"
        )


@router.post(
    "/api/otp/recovery",
    response_model=EncryptedOTPResponse,
    status_code=status.HTTP_200_OK,
)
async def request_recovery_otp(
    request: SignedOTPRequest = Body(...),
    http_request: Request = None,
):
    """
    Generate and send OTP for recovery flow (temporary key).

    This endpoint does NOT insert into user_keys table. It's used for
    secret share recovery where a temporary key is used for OTP verification.

    Differences from /api/otp/:
    - No user_identity creation
    - No user_keys insertion
    - No user_identity_id in response
    - No secret_share needed (can be omitted from request)

    Request model: SignedOTPRequest
    Response model: EncryptedOTPResponse
    """
    try:
        # Apply rate limiting per mobile number
        limiter = http_request.app.state.limiter if http_request else None
        if limiter:
            # Store mobile_number in request state for rate limiter key function
            http_request.state.mobile_number = request.mobile_number
            # Use the mobile number based rate limiting
            limiter._check_request_limit(http_request, lambda r: f"mobile:{request.mobile_number}", "3/10minutes")

        # Create signed OTP service
        signed_otp_service = SignedOTPService()

        # Process recovery OTP request (no user_keys insert)
        result = await signed_otp_service.process_recovery_request(
            client_public_key=request.client_public_key,
            mobile_number=request.mobile_number,
            country_code=request.country_code,
            timestamp=request.timestamp,
            signature_r=request.signature.r,
            signature_s=request.signature.s,
            target_server_public_key=request.target_server_public_key,
            otp_length=request.otp_length,
            generate_otp=request.generate_otp,
            gesture_mode=request.gesture_mode
        )

        if result['success']:
            logger.info(
                f"Recovery OTP request processed for {request.mobile_number}"
            )
            return EncryptedOTPResponse(**result['encrypted_response'])
        else:
            raise HTTPException(
                status_code=400,
                detail=result['error']
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing recovery OTP request: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process recovery OTP request: {str(e)}"
        )


@router.get(
    "/api/otp/legacy/{length}",
    response_model=DataResponse[OTPResponse],
    status_code=status.HTTP_200_OK,
)
async def generate_legacy_otp(
    length: int = Path(...),
    email: str = Query(..., description="Email address (legacy mode)")
):
    """
    Legacy OTP generation endpoint (for backward compatibility).

    Args:
        length: Length of the OTP code to generate
        email: Email address for legacy OTP generation

    Returns:
        OTPResponse with generated code (legacy mode - not sent via SMS)
    """
    try:
        result = await otp_service.get_random_number(length, email)
        return DataResponse[OTPResponse](data=result)
    except Exception as e:
        HTTPExceptionHelper.raise_for_exception(e)
