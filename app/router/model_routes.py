from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query

from app.core.security import SecurityService
from app.log.logger import get_routes_logger
from app.service.key.key_manager import KeyManager, get_key_manager_instance
from app.service.model.model_service import ModelService

router = APIRouter(prefix="/api/models", tags=["models"])
logger = get_routes_logger()

security_service = SecurityService()
model_service = ModelService()


async def get_key_manager():
    return await get_key_manager_instance()


@router.get("/upstream")
async def list_upstream_models(
    source: Literal["gemini", "vertex", "openai"] = Query(...),
    allowed_token: str = Depends(security_service.verify_authorization),
    key_manager: KeyManager = Depends(get_key_manager),
):
    if source == "vertex":
        api_key = await key_manager.get_next_working_vertex_key()
    else:
        api_key = await key_manager.get_random_valid_key()

    if not api_key:
        detail = "No valid API keys available to fetch upstream models."
        if source == "vertex":
            detail = "No valid Vertex API keys available to fetch upstream models."
        raise HTTPException(status_code=503, detail=detail)

    logger.info(f"Handling upstream models request for source: {source}")
    logger.info(f"Using allowed token: {allowed_token}")

    models = await model_service.get_upstream_models(source, api_key)
    if models is None:
        raise HTTPException(status_code=500, detail="Failed to fetch upstream models.")

    return {"source": source, "models": models}
