import base64


class DecodeBase64:
    @staticmethod
    def decode_base64(base64_str: str) -> bytes:
        try:
            if ";base64," in base64_str:
                base64_str = base64_str.split(";base64,")[1]
            content = base64.b64decode(base64_str)
            return content
        except Exception as e:
            raise ValueError(f"Failed to decode or process base64 string: {e}")
