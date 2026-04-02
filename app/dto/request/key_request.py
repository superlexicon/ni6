from pydantic import BaseModel


class UserShareKeyRequest(BaseModel):
    user_public_key: str
    share_encrypt_key: str
    email: str
