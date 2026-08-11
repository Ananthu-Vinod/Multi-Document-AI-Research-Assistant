"""
SQLAlchemy models for persistent storage: Users, Documents, and Chat Messages.
"""

from datetime import datetime
from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from database import Base


class User(Base):
    """User account model for JWT authentication."""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Integer, default=1)

    documents = relationship("DocumentModel", back_populates="user", cascade="all, delete-orphan")
    messages = relationship("ChatMessageModel", back_populates="user", cascade="all, delete-orphan")


class DocumentModel(Base):
    """Document upload metadata model (replaces document_registry.json)."""
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)
    session_id = Column(String(255), index=True, nullable=False)
    filename = Column(String(255), nullable=False)
    file_hash = Column(String(64), index=True, nullable=False)
    file_path = Column(Text, nullable=True)
    chunk_count = Column(Integer, default=0)
    uploaded_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="documents")


class ChatMessageModel(Base):
    """Chat message history model for persistent chat storage."""
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)
    session_id = Column(String(255), index=True, nullable=False)
    role = Column(String(50), nullable=False)  # 'user' or 'assistant'
    content = Column(Text, nullable=False)
    citations = Column(Text, nullable=True)  # JSON string of citations
    search_mode = Column(String(50), nullable=True)
    latency_ms = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="messages")
