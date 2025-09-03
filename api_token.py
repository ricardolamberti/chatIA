import base64
import os
import time

import jwt


def _decode_secret() -> bytes:
    secret_b64 = os.environ.get("JWT_SECRET_B64")
    if not secret_b64:
        raise RuntimeError("JWT_SECRET_B64 environment variable not set")
    secret_b64 += "=" * (-len(secret_b64) % 4)
    return base64.b64decode(secret_b64)


def generate_token(user: str, company: str, ttl_seconds: int = 3600) -> str:
    secret = _decode_secret()
    now = int(time.time())
    payload = {"sub": user, "company": company, "iat": now, "exp": now + ttl_seconds}
    token = jwt.encode(payload, secret, algorithm="HS256")
    if isinstance(token, bytes):
        token = token.decode("utf-8")
    return token


def auth_headers(user: str, company: str, ttl_seconds: int = 3600) -> dict:
    token = generate_token(user, company, ttl_seconds=ttl_seconds)
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
