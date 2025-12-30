"""Authentication utilities for the e-commerce analytics backend."""

import os
from typing import Optional

from itsdangerous import BadSignature, BadTimeSignature, URLSafeTimedSerializer
from passlib.context import CryptContext
from sqlalchemy.orm import Session

from models import User

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
SESSION_SECRET = os.getenv("SESSION_SECRET", "change-me")
SESSION_MAX_AGE = int(os.getenv("SESSION_MAX_AGE", "86400"))  # 24 hours


def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def _get_serializer() -> URLSafeTimedSerializer:
    return URLSafeTimedSerializer(SESSION_SECRET)


def create_session_token(data: dict) -> str:
    serializer = _get_serializer()
    return serializer.dumps(data)


def verify_session_token(token: str, max_age: Optional[int] = None) -> Optional[dict]:
    serializer = _get_serializer()
    lifetime = max_age or SESSION_MAX_AGE
    try:
        return serializer.loads(token, max_age=lifetime)
    except (BadSignature, BadTimeSignature):
        return None


def ensure_default_admin(db: Session, email: str, password: str, name: str) -> None:
    """Create a default admin user if none exists for the configured email."""
    existing = db.query(User).filter(User.email == email).first()
    if existing:
        return

    safe_password = password[:72]

    admin = User(
        email=email,
        name=name,
        password_hash=get_password_hash(safe_password),
        role="admin",
    )
    db.add(admin)
    db.commit()
