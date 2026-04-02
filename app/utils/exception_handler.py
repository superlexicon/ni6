from fastapi import HTTPException, status
from app.dto import ErrorResponse
from app.core.logger import get_logger

logger = get_logger()


class HTTPExceptionHelper:
    ERROR_STATUS_MAP = {
        "not found": status.HTTP_404_NOT_FOUND,
        "not convert": status.HTTP_400_BAD_REQUEST,
        "not allowed": status.HTTP_400_BAD_REQUEST,
        "already exists": status.HTTP_400_BAD_REQUEST,
        "Unsupported": status.HTTP_400_BAD_REQUEST,
        "Invalid image": status.HTTP_400_BAD_REQUEST,
        "verification required": status.HTTP_403_FORBIDDEN,
        "unauthorized": status.HTTP_401_UNAUTHORIZED,
        "unauthorized recovery": status.HTTP_403_FORBIDDEN,
        "forbidden": status.HTTP_403_FORBIDDEN,
        "duplicate passport": status.HTTP_409_CONFLICT,
        "duplicate submission": status.HTTP_409_CONFLICT,
        "timeout": status.HTTP_408_REQUEST_TIMEOUT,
        "too many requests": status.HTTP_429_TOO_MANY_REQUESTS,
    }

    @staticmethod
    def bad_request(message: str, details: str = None) -> HTTPException:
        """Create a 400 Bad Request HTTP exception"""
        detail = {"message": message}
        if details:
            detail["details"] = details
        return HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=detail
        )

    @staticmethod
    def not_found(message: str, details: str = None) -> HTTPException:
        """Create a 404 Not Found HTTP exception"""
        detail = {"message": message}
        if details:
            detail["details"] = details
        return HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=detail
        )

    @staticmethod
    def internal_server_error(message: str, details: str = None) -> HTTPException:
        """Create a 500 Internal Server Error HTTP exception"""
        detail = {"message": message}
        if details:
            detail["details"] = details
        return HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=detail
        )

    @staticmethod
    def raise_for_exception(exc: Exception) -> None:
        msg = str(exc)

        # Log the original exception for debugging
        logger.error(f"Exception occurred: {exc}")

        for keyword, http_status in HTTPExceptionHelper.ERROR_STATUS_MAP.items():
            if keyword.lower() in msg.lower():
                raise HTTPException(
                    status_code=http_status,
                    detail=ErrorResponse(message=msg).model_dump(),
                )
        else:
            # Default to 500 for unmapped exceptions
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=ErrorResponse(message="Internal server error").model_dump(),
            )
