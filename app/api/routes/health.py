"""
app/api/routes/health.py
"""
import platform
import sys
from datetime import datetime, timezone

from fastapi import APIRouter

from app.core.constants import TAG_HEALTH

router = APIRouter(tags=[TAG_HEALTH])


@router.get("/health", summary="Service health check")
async def health_check() -> dict:
    return {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
    }
