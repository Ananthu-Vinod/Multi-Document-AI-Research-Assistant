"""
Rate limiting middleware using slowapi.
Protects heavy AI generation and document upload routes.
"""

from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["120 per minute"],
    storage_uri="memory://",
)
