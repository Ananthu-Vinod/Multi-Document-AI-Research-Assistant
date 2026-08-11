"""
Authentication routes for user registration, login, and token verification.
"""

from pydantic import BaseModel, EmailStr
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from auth import create_access_token, hash_password, require_current_user, verify_password
from database import get_db
from models import User

router = APIRouter(prefix="/auth", tags=["auth"])


class UserRegisterSchema(BaseModel):
    email: str
    password: str
    full_name: str | None = None


class TokenResponseSchema(BaseModel):
    access_token: str
    token_type: str = "bearer"
    email: str
    full_name: str | None = None


class UserProfileSchema(BaseModel):
    id: int
    email: str
    full_name: str | None = None
    created_at: str

    class Config:
        from_attributes = True


@router.post("/register", response_model=TokenResponseSchema, status_code=status.HTTP_201_CREATED)
def register(payload: UserRegisterSchema, db: Session = Depends(get_db)):
    """Register a new user account and return a JWT access token."""
    email = payload.email.strip().lower()
    if not email or "@" not in email:
        raise HTTPException(status_code=400, detail="Invalid email address")
    if len(payload.password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters")

    existing = db.query(User).filter(User.email == email).first()
    if existing:
        raise HTTPException(status_code=400, detail="User with this email already exists")

    hashed_pw = hash_password(payload.password)
    user = User(email=email, hashed_password=hashed_pw, full_name=payload.full_name)
    db.add(user)
    db.commit()
    db.refresh(user)

    token = create_access_token({"sub": user.email, "id": user.id})
    return TokenResponseSchema(
        access_token=token,
        email=user.email,
        full_name=user.full_name,
    )


@router.post("/login", response_model=TokenResponseSchema)
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    """Login with email (username) and password to get a JWT access token."""
    email = form_data.username.strip().lower()
    user = db.query(User).filter(User.email == email).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = create_access_token({"sub": user.email, "id": user.id})
    return TokenResponseSchema(
        access_token=token,
        email=user.email,
        full_name=user.full_name,
    )


@router.get("/me", response_model=UserProfileSchema)
def get_me(user: User = Depends(require_current_user)):
    """Get details of the currently authenticated user."""
    return UserProfileSchema(
        id=user.id,
        email=user.email,
        full_name=user.full_name,
        created_at=user.created_at.isoformat() if user.created_at else "",
    )
