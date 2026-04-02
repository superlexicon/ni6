from pydantic import BaseModel


class PublicKeyResponse(BaseModel):
    public_key: str


class UserShareKeyResponse(BaseModel):
    user_public_key: str
    share_encrypt_key: str


class ServerKeyPair(BaseModel):
    public_key: str
    private_key: str
    seed: str


class EncryptedMessageData(BaseModel):
    enc: str
    iv: str


class DecryptedMessageData(BaseModel):
    plain_text: str
