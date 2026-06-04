"""Route for the dedicated ``/preview`` endpoint.

Builds the prompt, estimates tokens and generates suggestions without persisting
anything (PRD FR-03/FR-04).
"""

from __future__ import annotations

from fastapi import APIRouter

from app.core.dependencies import PromptBuilderDep
from app.models.preview import PreviewRequest, PreviewResponse

router = APIRouter(prefix="/preview", tags=["preview"])


@router.post("", response_model=PreviewResponse, summary="Preview a prompt")
async def preview_prompt(
    payload: PreviewRequest, prompt_builder: PromptBuilderDep
) -> PreviewResponse:
    """Build a non-persisted prompt preview with token estimates and suggestions.

    Args:
        payload: The preview request (use case, inputs, estimation mode).
        prompt_builder: Prompt builder service.

    Returns:
        The assembled prompt text, token estimates and suggestions.
    """
    return await prompt_builder.preview(
        payload.use_case_key,
        payload.structured_inputs,
        token_estimation_mode=payload.token_estimation_mode,
    )
