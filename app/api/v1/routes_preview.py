"""Route for the dedicated ``/preview`` endpoint.

Builds the prompt via local Ollama (Qwen 3), optimizes it, and returns
suggestions without persisting anything.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.api.v1.request_logging import log_request
from app.core.dependencies import OllamaPreviewDep
from app.core.logging import get_logger
from app.models.preview import PreviewRequest, PreviewResponse
from app.services.ollama_client import OllamaError

logger = get_logger(__name__)
router = APIRouter(prefix="/preview", tags=["preview"], dependencies=[Depends(log_request(logger))])


@router.post("", response_model=PreviewResponse, summary="Preview a prompt")
async def preview_prompt(
    payload: PreviewRequest, ollama_preview: OllamaPreviewDep
) -> PreviewResponse:
    """Build a non-persisted prompt preview via Ollama.

    Args:
        payload: The preview request (use case, inputs, models, estimation mode).
        ollama_preview: Ollama-backed preview service.

    Returns:
        One preview per requested model with optimized prompt text, token
        estimates, and LLM-generated suggestions.
    """
    try:
        return await ollama_preview.preview(
            payload.use_case_key,
            payload.structured_inputs,
            models=payload.models,
            token_estimation_mode=payload.token_estimation_mode,
        )
    except OllamaError as exc:
        logger.warning("Preview via Ollama failed: %s", exc)
        raise HTTPException(status_code=502, detail=str(exc)) from exc
