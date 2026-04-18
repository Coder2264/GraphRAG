"""
API v1 router — aggregates ingest and query sub-routers.
"""

from fastapi import APIRouter

from app.api.v1.cleanup import router as cleanup_router
from app.api.v1.ingest import router as ingest_router
from app.api.v1.query import router as query_router

router = APIRouter(prefix="/api/v1")

router.include_router(ingest_router)
router.include_router(query_router)
router.include_router(cleanup_router)
