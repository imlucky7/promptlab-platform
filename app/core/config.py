"""Typed application configuration.

Configuration is loaded from environment variables (and an optional ``.env``
file) using ``pydantic-settings``. Keeping all configuration in a single typed
``Settings`` object makes the rest of the codebase free of ``os.environ`` access
and trivially testable.
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Annotated, Any, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

# Type aliases for the small, well-defined enumerations used in configuration.
TokenEstimationMode = Literal["gateway", "local"]
EnvironmentName = Literal["local", "staging", "production"]


class ModelConfig(BaseSettings):
    """Catalogue entry describing how to reach a single logical model.

    Attributes:
        provider: Optional provider key (e.g. ``"openai"``, ``"anthropic"``,
            ``"perplexity"``). When set and a matching API key is configured,
            the gateway client calls that provider's OpenAI-compatible endpoint
            directly instead of routing through the unified gateway.
        base_url: Optional per-model base URL override for direct calls. When
            unset the provider's default base URL (from settings) is used.
        provider_model_name: Human/provider facing model name (e.g.
            ``"claude-3-5-sonnet-20240620"``). This is the model id sent when
            calling the provider directly.
        gateway_model_identifier: Identifier understood by the unified LLM
            gateway (e.g. ``"anthropic/claude-3-5-sonnet-20240620"``).
        input_cost_per_1k: Optional input price per 1K tokens, used for cost
            estimation in the metrics engine.
        output_cost_per_1k: Optional output price per 1K tokens.
    """

    provider: str | None = Field(default=None, alias="provider")
    base_url: str | None = Field(default=None, alias="baseUrl")
    provider_model_name: str = Field(alias="providerModelName")
    gateway_model_identifier: str = Field(alias="gatewayModelIdentifier")
    input_cost_per_1k: float | None = Field(default=None, alias="inputCostPer1k")
    output_cost_per_1k: float | None = Field(default=None, alias="outputCostPer1k")

    model_config = SettingsConfigDict(populate_by_name=True, extra="ignore")


class Settings(BaseSettings):
    """Central application settings.

    All fields are populated from environment variables. See ``.env.example``
    for the full list and sample values.
    """

    # ---- Application ---------------------------------------------------------
    app_name: str = Field(default="Prompt Lab - Travel Planner API", alias="APP_NAME")
    environment: EnvironmentName = Field(default="local", alias="ENVIRONMENT")
    debug: bool = Field(default=True, alias="DEBUG")
    api_v1_prefix: str = Field(default="/api/v1", alias="API_V1_PREFIX")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    # ``NoDecode`` stops pydantic-settings from JSON-decoding the raw env value
    # so values like ``*`` or comma-separated lists reach ``_split_cors_origins``.
    cors_allow_origins: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["*"], alias="CORS_ALLOW_ORIGINS"
    )

    # ---- MongoDB ------------------------------------------------------------
    mongodb_uri: str = Field(default="mongodb://localhost:27017", alias="MONGODB_URI")
    mongodb_db: str = Field(default="prompt_lab", alias="MONGODB_DB")

    # ---- LLM gateway --------------------------------------------------------
    llm_gateway_stub_mode: bool = Field(default=True, alias="LLM_GATEWAY_STUB_MODE")
    llm_gateway_base_url: str | None = Field(default=None, alias="LLM_GATEWAY_BASE_URL")
    llm_gateway_api_key: str | None = Field(default=None, alias="LLM_GATEWAY_API_KEY")
    llm_gateway_timeout_seconds: float = Field(default=60.0, alias="LLM_GATEWAY_TIMEOUT_SECONDS")
    llm_gateway_max_retries: int = Field(default=2, alias="LLM_GATEWAY_MAX_RETRIES")

    # ---- Direct provider credentials ----------------------------------------
    # Used when a model in the catalogue declares a ``provider`` and you want to
    # call that provider's OpenAI-compatible API directly (no unified gateway).
    # Set the relevant key(s) and ``LLM_GATEWAY_STUB_MODE=false`` to go live.
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    perplexity_api_key: str | None = Field(default=None, alias="PERPLEXITY_API_KEY")
    openai_base_url: str = Field(default="https://api.openai.com/v1", alias="OPENAI_BASE_URL")
    anthropic_base_url: str = Field(
        default="https://api.anthropic.com/v1", alias="ANTHROPIC_BASE_URL"
    )
    perplexity_base_url: str = Field(default="https://api.perplexity.ai", alias="PERPLEXITY_BASE_URL")

    # ---- Ollama (local preview LLM) -----------------------------------------
    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")
    ollama_model: str = Field(default="qwen3:8b", alias="OLLAMA_MODEL")
    ollama_timeout_seconds: float = Field(default=120.0, alias="OLLAMA_TIMEOUT_SECONDS")
    ollama_preview_stub_mode: bool = Field(default=False, alias="OLLAMA_PREVIEW_STUB_MODE")

    # ---- Token estimation ---------------------------------------------------
    token_estimation_mode: TokenEstimationMode = Field(
        default="local", alias="TOKEN_ESTIMATION_MODE"
    )
    token_estimation_model: str = Field(default="gpt-4o-mini", alias="TOKEN_ESTIMATION_MODEL")
    local_tokenizer_model: str = Field(default="gpt-4o-mini", alias="LOCAL_TOKENIZER_MODEL")

    # ---- Model catalogue ----------------------------------------------------
    # ``NoDecode`` defers JSON parsing to ``_parse_model_catalog`` (which also
    # supplies the default catalogue when unset). ``validate_default`` ensures
    # that validator runs even when ``MODEL_CATALOG`` is not provided.
    model_catalog: Annotated[dict[str, ModelConfig], NoDecode] = Field(
        default_factory=dict, validate_default=True, alias="MODEL_CATALOG"
    )

    # ---- Seeding ------------------------------------------------------------
    seed_on_startup: bool = Field(default=True, alias="SEED_ON_STARTUP")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        populate_by_name=True,
        extra="ignore",
    )

    @field_validator("cors_allow_origins", mode="before")
    @classmethod
    def _split_cors_origins(cls, value: Any) -> Any:
        """Allow CORS origins to be provided as a comma separated string.

        Args:
            value: Raw value from the environment (string or list).

        Returns:
            A list of origin strings.
        """
        if isinstance(value, str):
            # Support both JSON arrays and simple comma separated values.
            stripped = value.strip()
            if stripped.startswith("["):
                return json.loads(stripped)
            return [item.strip() for item in stripped.split(",") if item.strip()]
        return value

    @field_validator("model_catalog", mode="before")
    @classmethod
    def _parse_model_catalog(cls, value: Any) -> Any:
        """Parse the model catalogue from a JSON string when needed.

        The catalogue is supplied as a JSON object mapping model keys to their
        provider/gateway descriptors. Falls back to a sensible default
        (Anthropic + Perplexity) when unset.

        Args:
            value: Raw value from the environment (JSON string or mapping).

        Returns:
            A mapping suitable for building ``ModelConfig`` instances.
        """
        if value in (None, "", {}):
            return _default_model_catalog()
        if isinstance(value, str):
            return json.loads(value)
        return value

    def get_model_config(self, model_key: str) -> ModelConfig | None:
        """Return the catalogue entry for ``model_key`` if it exists.

        Args:
            model_key: Logical model key (e.g. ``"anthropic"``).

        Returns:
            The matching :class:`ModelConfig`, or ``None`` when unknown.
        """
        return self.model_catalog.get(model_key)

    def resolve_provider_target(self, model_cfg: ModelConfig | None) -> tuple[str, str] | None:
        """Resolve the direct-provider ``(base_url, api_key)`` for a model.

        A direct call is possible only when the model declares a known
        ``provider`` *and* the corresponding API key is configured. The base URL
        falls back to the provider default when the model does not override it.

        Args:
            model_cfg: The catalogue entry for the target model, if any.

        Returns:
            A ``(base_url, api_key)`` tuple when a direct provider call can be
            made, otherwise ``None``.
        """
        if model_cfg is None or not model_cfg.provider:
            return None
        provider = model_cfg.provider.strip().lower()
        provider_defaults: dict[str, tuple[str, str | None]] = {
            "openai": (self.openai_base_url, self.openai_api_key),
            "anthropic": (self.anthropic_base_url, self.anthropic_api_key),
            "perplexity": (self.perplexity_base_url, self.perplexity_api_key),
        }
        default = provider_defaults.get(provider)
        if default is None:
            return None
        base_url, api_key = default
        if model_cfg.base_url:
            base_url = model_cfg.base_url
        if not api_key:
            return None
        return base_url, api_key

    def is_model_stubbed(self, model_key: str) -> bool:
        """Whether a specific model should be served by the deterministic stub.

        Stub mode wins when explicitly enabled. Otherwise a model is *live* when
        either a direct provider credential is configured for it, or the unified
        gateway (base URL + key) is configured. When neither is available the
        model falls back to the stub so the app stays runnable offline.

        Args:
            model_key: Logical model key (e.g. ``"chatgpt"``).

        Returns:
            ``True`` when the model should return stub responses.
        """
        if self.llm_gateway_stub_mode:
            return True
        if self.resolve_provider_target(self.get_model_config(model_key)) is not None:
            return False
        return not (self.llm_gateway_base_url and self.llm_gateway_api_key)

    @property
    def is_gateway_stubbed(self) -> bool:
        """Whether the unified gateway should operate in deterministic stub mode.

        Stub mode is active when explicitly enabled, or when the base URL/API
        key are missing (so the app remains runnable offline for local dev).
        This governs the unified-gateway code paths (e.g. token estimation);
        per-model completion routing uses :meth:`is_model_stubbed`.
        """
        if self.llm_gateway_stub_mode:
            return True
        return not (self.llm_gateway_base_url and self.llm_gateway_api_key)


def _default_model_catalog() -> dict[str, dict[str, Any]]:
    """Provide the default model catalogue.

    The keys mirror the template model keys used by the preview flow
    (``chatgpt`` and ``claude``) so a preview can be executed via ``/runs``
    without remapping, plus the original ``anthropic`` and ``perplexity``
    aliases. Each entry declares a ``provider`` so the gateway client can call
    the provider's OpenAI-compatible API directly when a key is configured.

    Returns:
        A mapping of model key to descriptor dictionaries.
    """
    return {
        "chatgpt": {
            "provider": "openai",
            "providerModelName": "gpt-4o",
            "gatewayModelIdentifier": "openai/gpt-4o",
            "inputCostPer1k": 0.005,
            "outputCostPer1k": 0.015,
        },
        "claude": {
            "provider": "anthropic",
            "providerModelName": "claude-3-5-sonnet-20240620",
            "gatewayModelIdentifier": "anthropic/claude-3-5-sonnet-20240620",
            "inputCostPer1k": 0.003,
            "outputCostPer1k": 0.015,
        },
        "anthropic": {
            "provider": "anthropic",
            "providerModelName": "claude-3-5-sonnet-20240620",
            "gatewayModelIdentifier": "anthropic/claude-3-5-sonnet-20240620",
            "inputCostPer1k": 0.003,
            "outputCostPer1k": 0.015,
        },
        "perplexity": {
            "provider": "perplexity",
            "providerModelName": "sonar-pro",
            "gatewayModelIdentifier": "perplexity/sonar-pro",
            "inputCostPer1k": 0.003,
            "outputCostPer1k": 0.015,
        },
    }


@lru_cache
def get_settings() -> Settings:
    """Return a cached :class:`Settings` instance.

    The ``lru_cache`` ensures configuration is parsed once per process, which is
    both efficient and convenient for dependency injection / test overrides.

    Returns:
        The process-wide settings object.
    """
    return Settings()
