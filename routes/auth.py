"""
Auth Routes — Lightweight JWT-based auth for Python backend.
GET  /api/auth/user — returns current user from X-User-Id header
POST /api/auth/demo — creates demo session token
"""

import os
import logging
from fastapi import APIRouter, Header
from typing import Optional

logger = logging.getLogger("dermai.routes.auth")
router = APIRouter()


@router.get("/auth/user")
async def get_current_user(x_user_id: Optional[str] = Header(None)):
    """Return user info from session header."""
    if not x_user_id:
        return {"user": None, "authenticated": False}

    return {
        "user": {
            "id": x_user_id,
            "email": None,
            "firstName": "Demo",
            "lastName": "User",
        },
        "authenticated": True
    }


@router.post("/auth/demo")
async def create_demo_session():
    """Create a demo user session for testing."""
    import uuid
    demo_id = f"demo-{uuid.uuid4().hex[:8]}"
    return {
        "userId": demo_id,
        "token": demo_id,
        "message": "Demo session created. Use X-User-Id header for subsequent requests."
    }
