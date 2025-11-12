"""Centralized configuration for API endpoints and access tokens."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional
import os


@dataclass(frozen=True)
class EndpointConfig:
    """Configuration describing how to resolve an API endpoint."""

    env_var: str
    default: str


@dataclass(frozen=True)
class AccessTokenConfig:
    """Configuration describing how to resolve an API access token."""

    env_var: str
    placeholder: str


API_ENDPOINTS: Dict[str, EndpointConfig] = {
    "mistral": EndpointConfig(
        env_var="MISTRAL_ENDPOINT",
        default="https://api.mistral.ai/v1/chat/completions",
    ),
    "ollama": EndpointConfig(
        env_var="OLLAMA_API_URL",
        default="http://localhost:11434",
    ),
    "lm_studio": EndpointConfig(
        env_var="LM_STUDIO_API_URL",
        default="http://localhost:1234",
    ),
}


ACCESS_TOKENS: Dict[str, AccessTokenConfig] = {
    "github": AccessTokenConfig(
        env_var="GITHUB_API_KEY",
        placeholder="your_github_api_key_here",
    ),
    "openai": AccessTokenConfig(
        env_var="OPENAI_API_KEY",
        placeholder="your_openai_api_key_here",
    ),
    "azure": AccessTokenConfig(
        env_var="AZURE_API_KEY",
        placeholder="your_azure_api_key_here",
    ),
    "azure_deployment": AccessTokenConfig(
        env_var="AZURE_DEPLOYMENT_NAME",
        placeholder="your_azure_deployment_name_here",
    ),
    "google": AccessTokenConfig(
        env_var="GOOGLE_API_KEY",
        placeholder="your_google_api_key_here",
    ),
    "mistral": AccessTokenConfig(
        env_var="MISTRAL_API_KEY",
        placeholder="your_mistral_api_key_here",
    ),
    "mistral_access_token": AccessTokenConfig(
        env_var="MISTRAL_ACCESS_TOKEN",
        placeholder="your_mistral_access_token_here",
    ),
    "groq": AccessTokenConfig(
        env_var="GROQ_API_KEY",
        placeholder="your_groq_api_key_here",
    ),
}


def get_endpoint(name: str) -> str:
    """Resolve an API endpoint using environment variables or defaults."""

    config = API_ENDPOINTS[name]
    return os.getenv(config.env_var, config.default)


def get_access_token(name: str) -> Optional[str]:
    """Resolve an API access token from the environment, if available."""

    config = ACCESS_TOKENS[name]
    return os.getenv(config.env_var)


def get_access_token_placeholder(name: str) -> str:
    """Return a human-friendly placeholder for an API token."""

    config = ACCESS_TOKENS[name]
    return config.placeholder
