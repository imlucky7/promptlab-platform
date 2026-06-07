"""Route for the dedicated ``/preview`` endpoint.

Builds the prompt, estimates tokens and generates suggestions without persisting
anything (PRD FR-03/FR-04).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.v1.request_logging import log_request
from app.core.dependencies import PromptBuilderDep
from app.core.logging import get_logger
from app.models.preview import PreviewRequest, PreviewResponse

logger = get_logger(__name__)
router = APIRouter(prefix="/preview", tags=["preview"], dependencies=[Depends(log_request(logger))])


@router.post("", response_model=PreviewResponse, summary="Preview a prompt")
async def preview_prompt(
    payload: PreviewRequest, prompt_builder: PromptBuilderDep
) -> PreviewResponse:
    """Build non-persisted prompt previews for one or more template variants.

    Args:
        payload: The preview request (use case, inputs, models, estimation mode).
        prompt_builder: Prompt builder service.

    Returns:
        One preview per requested (or available) template variant, each with its
        prompt text, token estimates and suggestions.
    """
    return await prompt_builder.preview(
        payload.use_case_key,
        payload.structured_inputs,
        models=payload.models,
        token_estimation_mode=payload.token_estimation_mode,
    )
