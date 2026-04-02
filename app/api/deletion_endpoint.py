"""
API endpoint for user presence deletion.
"""
from fastapi import APIRouter, HTTPException, status

from app.dto.delete_user_presence import (
    DeletePresenceRequest,
    DeletePresenceResponse
)
from app.services.user_deletion_service import UserDeletionService
from app.core import get_db_connection
from app.core.logger import get_logger


logger = get_logger()

router = APIRouter(tags=["User Management"])


@router.post(
    "/user/delete-presence",
    response_model=DeletePresenceResponse,
    status_code=status.HTTP_200_OK,
    summary="Delete all user data with signature verification",
    description="""
    Permanently delete all user data from the database.

    **Authentication:**
    - User signs a message with format "delete:{unix_timestamp}"
    - Public key is recovered from the ECDSA signature (r, s)
    - Recovered key must match an existing user_keys entry

    **Replay Protection:**
    - Message timestamp must be within 5 minutes of server time
    - Prevents replay attacks using old signatures

    **Deletion Scope:**
    - document_analysis_jobs (all jobs for this public key)
    - document_submissions (all submissions for this identity)
    - face_biometrics (all face embeddings for this identity)
    - otp (all OTP records for this public key)
    - user_keys (all device keys for this identity)
    - user_identity_index (the identity record itself)

    **Warning:** This action is irreversible. All user data will be permanently deleted.
    """
)
async def delete_user_presence(
    request: DeletePresenceRequest
) -> DeletePresenceResponse:
    """
    Delete all user presence from the database.

    Requires ECDSA signature verification to authenticate the request.
    """
    try:
        service = UserDeletionService()

        deleted = service.delete_user_presence(
            message=request.message,
            signature_r=request.signature.r,
            signature_s=request.signature.s,
            public_key=request.public_key
        )

        return DeletePresenceResponse(
            success=True,
            message="User presence deleted successfully",
            deleted_records=deleted
        )

    except ValueError as e:
        logger.error(f"User deletion failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Unexpected error during user deletion: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during deletion"
        )
