"""Utilities to construct LangChain chat models based on runtime configuration."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from fastapi import HTTPException
from pydantic import BaseModel, Field, root_validator

try:  # Optional dependency used when OpenAI models are requested
    from langchain_openai import ChatOpenAI  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    ChatOpenAI = None  # type: ignore

try:  # Optional dependency used for AWS Bedrock models
    from langchain_aws import ChatBedrockConverse  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    ChatBedrockConverse = None  # type: ignore

from langchain_community.chat_models import ChatOllama
from langchain_core.language_models.chat_models import BaseChatModel


class LLMConfiguration(BaseModel):
    """Configuration payload describing how to instantiate a chat model."""

    provider: str = Field(
        ...,
        description=(
            "Identifier of the backing provider. Supported values: 'ollama', 'openai', 'bedrock'."
        ),
    )
    model: str = Field(..., description="Model identifier understood by the provider.")
    temperature: float = Field(0.2, ge=0.0, le=1.0)
    max_tokens: Optional[int] = Field(
        None, description="Optional upper bound for generated tokens when supported by the provider."
    )
    api_key: Optional[str] = Field(
        None, description="Direct API key value. Takes precedence over environment lookups."
    )
    api_key_env: Optional[str] = Field(
        None,
        description="Environment variable that stores the API key. Used when `api_key` is omitted.",
    )
    base_url: Optional[str] = Field(
        None,
        description="Custom endpoint for self-hosted deployments (e.g. Ollama running on another host).",
    )
    region: Optional[str] = Field(
        None, description="AWS region when using Bedrock. Defaults to provider SDK fallback."
    )
    profile: Optional[str] = Field(
        None,
        description="Named AWS credential profile to use for Bedrock. Defaults to boto3 behaviour when omitted.",
    )
    additional_parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Catch-all dictionary for provider specific overrides (e.g. timeout settings).",
    )

    @root_validator(pre=False)
    def _normalise_provider(cls, values: Dict[str, Any]) -> Dict[str, Any]:  # pragma: no cover - simple normalisation
        values["provider"] = values["provider"].lower()
        return values


def _resolve_api_key(config: LLMConfiguration) -> Optional[str]:
    if config.api_key:
        return config.api_key
    if config.api_key_env:
        return os.getenv(config.api_key_env)
    return None


def create_chat_model(config: LLMConfiguration) -> BaseChatModel:
    """Instantiate a LangChain chat model based on :class:`LLMConfiguration`."""

    provider = config.provider

    if provider == "ollama":
        return ChatOllama(
            model=config.model,
            temperature=config.temperature,
            base_url=config.base_url,
            **config.additional_parameters,
        )

    if provider == "openai":
        if ChatOpenAI is None:  # pragma: no cover - depends on optional dependency
            raise HTTPException(
                status_code=500,
                detail=(
                    "The 'langchain-openai' package is required to use OpenAI models. "
                    "Install it or choose a different provider."
                ),
            )
        api_key = _resolve_api_key(config) or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise HTTPException(
                status_code=400,
                detail=(
                    "OpenAI provider requested but no API key was supplied. "
                    "Provide `api_key` or set the `OPENAI_API_KEY` environment variable."
                ),
            )
        return ChatOpenAI(
            model=config.model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            openai_api_key=api_key,
            base_url=config.base_url,
            **config.additional_parameters,
        )

    if provider == "bedrock":
        if ChatBedrockConverse is None:  # pragma: no cover - depends on optional dependency
            raise HTTPException(
                status_code=500,
                detail=(
                    "The 'langchain-aws' package is required to use AWS Bedrock models. "
                    "Install it or choose a different provider."
                ),
            )
        init_kwargs: Dict[str, Any] = {
            "model": config.model,
            "temperature": config.temperature,
        }
        if config.max_tokens is not None:
            init_kwargs["max_tokens"] = config.max_tokens
        if config.region:
            init_kwargs["region_name"] = config.region
        if config.profile:
            init_kwargs["profile_name"] = config.profile
        init_kwargs.update(config.additional_parameters)
        return ChatBedrockConverse(**init_kwargs)

    raise HTTPException(
        status_code=400,
        detail=f"Unsupported LLM provider '{config.provider}'. Choose 'ollama', 'openai', or 'bedrock'.",
    )


__all__ = ["LLMConfiguration", "create_chat_model"]
