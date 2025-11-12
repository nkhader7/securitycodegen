"""Streamlit application for secure code generation recommendations using Mistral."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import List, Optional

import requests
import streamlit as st

from api_settings import API_ENDPOINTS, get_endpoint

MISTRAL_ENDPOINT_CONFIG = API_ENDPOINTS["mistral"]
DEFAULT_MISTRAL_ENDPOINT = MISTRAL_ENDPOINT_CONFIG.default
MISTRAL_ENDPOINT_ENV_VAR = MISTRAL_ENDPOINT_CONFIG.env_var

OLLAMA_ENDPOINT_CONFIG = API_ENDPOINTS["ollama"]
DEFAULT_OLLAMA_URL = OLLAMA_ENDPOINT_CONFIG.default


def _get_secret(name: str) -> Optional[str]:
    """Safely retrieve a Streamlit secret when configured."""

    secrets = getattr(st, "secrets", None)
    if secrets is None:
        return None

    getter = getattr(secrets, "get", None)
    if callable(getter):
        return getter(name)

    try:
        return secrets[name]  # type: ignore[index]
    except Exception:  # pragma: no cover - defensive for unexpected secrets objects
        return None


@dataclass
class ScanOption:
    """Represents an optional security scan toggle."""

    label: str
    description: str
    prompt_snippet: str


@dataclass
class LLMSettings:
    """Configuration values required to contact the selected LLM backend."""

    provider: str
    model: str
    temperature: float
    api_key: Optional[str] = None
    access_token: Optional[str] = None
    endpoint: str = DEFAULT_MISTRAL_ENDPOINT
    ollama_url: str = DEFAULT_OLLAMA_URL


SCAN_OPTIONS: List[ScanOption] = [
    ScanOption(
        label="Dependency Vulnerability Scan",
        description="Analyze third-party dependencies for known CVEs and insecure versions.",
        prompt_snippet="Perform a dependency vulnerability review and highlight outdated packages.",
    ),
    ScanOption(
        label="Static Code Analysis",
        description="Identify insecure coding patterns, injection vectors, and unsafe APIs.",
        prompt_snippet="Run a static code analysis pass to identify insecure patterns.",
    ),
    ScanOption(
        label="Secrets Detection",
        description="Inspect the content for hard-coded secrets, tokens, or credentials.",
        prompt_snippet="Detect potential credential or secret exposures in the supplied material.",
    ),
]


def _read_uploaded_file(upload) -> Optional[str]:
    """Safely decode an uploaded file as UTF-8 text."""

    if not upload:
        return None

    try:
        # ``UploadedFile.read`` exhausts the underlying buffer. When Streamlit reruns the
        # script (e.g., due to widget interaction) the same ``UploadedFile`` instance is
        # reused with its pointer already positioned at EOF, yielding empty content.
        # ``getvalue`` returns the cached payload without mutating the pointer so the
        # uploaded data persists across reruns.
        data = upload.getvalue()
        if isinstance(data, str):
            return data
        return data.decode("utf-8")
    except Exception:  # pragma: no cover - defensive fallback for unexpected encodings
        return None


def _pretty_format_content(name: str, content: Optional[str]) -> str:
    """Return a formatted block summarizing uploaded content."""

    if not content:
        return f"No {name.lower()} provided."

    truncated = content.strip()
    if len(truncated) > 2000:
        truncated = truncated[:2000] + "\n... (truncated)"
    return truncated


@st.cache_data(show_spinner=False)
def _build_prompt(
    instructions: str,
    manifest_content: Optional[str],
    scan_results: Optional[str],
    selected_options: List[ScanOption],
) -> str:
    """Create the composite prompt for the LLM."""

    prompt_parts = [
        "You are a secure code generation assistant.",
        "Provide actionable, high-quality recommendations that follow security best practices.",
        "Always explain the reasoning behind each recommendation.",
        "",
        "User Instructions:",
        instructions.strip() or "(No additional instructions provided.)",
    ]

    if manifest_content:
        prompt_parts.extend(
            [
                "",
                "Manifest File Content:",
                manifest_content.strip(),
            ]
        )

    if scan_results:
        prompt_parts.extend(
            [
                "",
                "Security Scan Results:",
                scan_results.strip(),
            ]
        )

    if selected_options:
        prompt_parts.append("")
        prompt_parts.append("Focus Areas:")
        prompt_parts.extend(option.prompt_snippet for option in selected_options)

    prompt_parts.append("")
    prompt_parts.append(
        "Produce the updated secure code that addresses the identified issues, including inline comments "
        "that explain the security hardening choices."
    )
    prompt_parts.append(
        "Wrap code snippets in fenced Markdown blocks, and precede them with short explanations of the "
        "changes so the user understands the mitigation steps."
    )

    return "\n".join(prompt_parts)


def _resolve_mistral_configuration() -> tuple[str, Optional[str]]:
    """Derive the API endpoint and key from secrets, env vars, or user input."""

    endpoint_candidates = [
        st.session_state.get("mistral_endpoint"),
        _get_secret(MISTRAL_ENDPOINT_ENV_VAR),
        get_endpoint("mistral"),
    ]
    endpoint = next((value for value in endpoint_candidates if value), DEFAULT_MISTRAL_ENDPOINT)

    api_key_candidates = [
        st.session_state.get("mistral_api_key"),
        _get_secret("MISTRAL_API_KEY"),
        os.getenv("MISTRAL_API_KEY"),
    ]
    api_key = next((value for value in api_key_candidates if value), None)

    return endpoint, api_key


def _call_mistral(
    prompt: str,
    model: str,
    temperature: float,
) -> str:
    """Call the Mistral chat completion API."""

    endpoint, api_key = _resolve_mistral_configuration()
    if not api_key:
        return (
            "A Mistral API key is required. Provide it via `.streamlit/secrets.toml`, the "
            "MISTRAL_API_KEY environment variable, or the sidebar configuration panel."
        )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": "You are a helpful secure coding assistant."},
            {"role": "user", "content": prompt},
        ],
    }

    try:
        response = requests.post(endpoint, headers=headers, data=json.dumps(payload), timeout=60)
        response.raise_for_status()
        body = response.json()
        choices = body.get("choices", [])
        if not choices:
            return "No completion returned by the Mistral API."
        return choices[0].get("message", {}).get("content", "No content in completion message.")
    except requests.RequestException as exc:
        return f"Mistral API request failed: {exc}"
    except (json.JSONDecodeError, KeyError) as exc:
        return f"Unexpected response format from Mistral API: {exc}"


def _render_sidebar() -> tuple[str, float]:
    st.sidebar.header("Mistral Settings")

    st.session_state.setdefault(
        "mistral_endpoint",
        _get_secret(MISTRAL_ENDPOINT_ENV_VAR)
        or get_endpoint("mistral"),
    )
    st.session_state.setdefault(
        "mistral_api_key",
        _get_secret("MISTRAL_API_KEY")
        or os.getenv("MISTRAL_API_KEY")
        or "",
    )
    model = st.sidebar.selectbox(
        "Model",
        options=[
            "mistral-small-latest",
            "mistral-medium-latest",
            "mistral-large-latest",
        ],
        index=0,
        help="Choose the hosted Mistral model to generate secure code recommendations.",
    )

    temperature = st.sidebar.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.2,
        step=0.1,
        help="Lower values make the model more deterministic; higher values increase creativity.",
    )

    if provider in {"hosted", "custom"}:
        default_endpoint = DEFAULT_MISTRAL_ENDPOINT
        if provider == "custom":
            default_endpoint = os.getenv("MISTRAL_API_ENDPOINT", DEFAULT_MISTRAL_ENDPOINT)

        endpoint = st.sidebar.text_input(
            "Mistral API Endpoint",
            value=default_endpoint,
            help="Override the Mistral API endpoint when using a self-hosted deployment.",
        )

        api_key = st.sidebar.text_input(
            "Mistral API Key",
            value=os.getenv("MISTRAL_API_KEY", ""),
            type="password",
            help="API key for authenticating with the Mistral service.",
        )
        access_token = st.sidebar.text_input(
            "Access Token (optional)",
            value=os.getenv("MISTRAL_ACCESS_TOKEN", ""),
            type="password",
            help="Optional secondary access token for custom deployments.",
        )

        model = st.sidebar.selectbox(
            "Model",
            options=[
                "mistral-small-latest",
                "mistral-medium-latest",
                "mistral-large-latest",
            ],
            index=0,
            help="Choose the hosted or custom Mistral model used to generate secure code.",
        )

        settings = LLMSettings(
            provider=provider,
            model=model,
            temperature=temperature,
            api_key=api_key or None,
            access_token=access_token or None,
            endpoint=endpoint or DEFAULT_MISTRAL_ENDPOINT,
        )
    else:
        ollama_url = st.sidebar.text_input(
            "Ollama API URL",
            value=get_endpoint("ollama"),
            help="Base URL where the Ollama service is reachable.",
        )
        ollama_model = st.sidebar.text_input(
            "Ollama Model",
            value=os.getenv("OLLAMA_MODEL", "mistral"),
            help="Name of the Mistral model available within Ollama.",
        )

        settings = LLMSettings(
            provider=provider,
            model=ollama_model or "mistral",
            temperature=temperature,
            ollama_url=ollama_url or DEFAULT_OLLAMA_URL,
        )

    st.sidebar.markdown(
        """
        **Usage Tips**
        - Provide as much context as possible to receive detailed secure coding guidance.
        - Upload manifest files (e.g., `package.json`, `requirements.txt`) to enrich the analysis.
        - Include scan reports to get tailored remediation steps.
        """
    )

    with st.sidebar.expander("API Configuration", expanded=False):
        st.text_input(
            "Mistral API endpoint",
            key="mistral_endpoint",
            help=(
                "Override the API endpoint if you are routing requests through a proxy or "
                "self-hosted deployment."
            ),
        )
        st.text_input(
            "Mistral API key",
            key="mistral_api_key",
            type="password",
            help=(
                "Stored only in the current Streamlit session. You can also configure this via "
                "MISTRAL_API_KEY or `.streamlit/secrets.toml`."
            ),
        )

    return model, temperature


def main() -> None:
    st.set_page_config(page_title="Secure Code Generation", page_icon="🛡️", layout="wide")
    st.title("🛡️ Secure Code Generation with Mistral")
    st.write(
        "Leverage Mistral's language models to review manifests, scan reports, and project context "
        "for actionable secure coding recommendations."
    )

    llm_settings = _render_sidebar()

    with st.expander("Input Guidance", expanded=False):
        st.markdown(
            """
            ### Recommended Workflow
            1. Provide a high-level description of the code you would like to generate or assess.
            2. Upload relevant manifest files (such as dependency descriptors) for analysis.
            3. Upload security scan reports (e.g., SAST, DAST, or dependency scans).
            4. Select additional focus areas and generate the report.
            """
        )

    instructions = st.text_area(
        "Describe the secure coding task or code generation goals",
        height=200,
        placeholder="e.g., Generate a secure FastAPI endpoint with JWT authentication and rate limiting.",
    )

    manifest_upload = st.file_uploader(
        "Upload Manifest File",
        type=["txt", "json", "yaml", "yml", "toml", "lock"],
        help="Include dependency manifests such as package.json, requirements.txt, or pyproject.toml.",
    )
    manifest_content = _read_uploaded_file(manifest_upload)

    scan_upload = st.file_uploader(
        "Upload Security Scan Results",
        type=["txt", "json", "md", "xml"],
        help="Attach scan reports from security tooling to tailor remediation steps.",
    )
    scan_content = _read_uploaded_file(scan_upload)

    st.markdown("### Optional Security Focus Areas")
    selected_options = []
    cols = st.columns(3)
    for idx, option in enumerate(SCAN_OPTIONS):
        with cols[idx % 3]:
            if st.checkbox(option.label, help=option.description):
                selected_options.append(option)

    st.markdown("---")

    if st.button("Generate Secure Code", type="primary"):
        if not instructions.strip() and not manifest_content and not scan_content:
            st.warning("Please provide instructions, a manifest file, or scan results to generate recommendations.")
            return

        with st.spinner("Contacting the configured Mistral model and generating secure code..."):
            prompt = _build_prompt(instructions, manifest_content, scan_content, selected_options)
            response = _call_mistral(prompt, settings=llm_settings)

        st.markdown("## Security-Hardened Code Suggestions")
        st.markdown(response)

        st.download_button(
            "Download Recommendations",
            data=response,
            file_name="secure_code_recommendations.md",
            mime="text/markdown",
        )

    with st.expander("Review Uploaded Content", expanded=False):
        st.markdown("#### Manifest Preview")
        st.code(_pretty_format_content("Manifest", manifest_content), language="text")

        st.markdown("#### Scan Results Preview")
        st.code(_pretty_format_content("Scan results", scan_content), language="text")

    st.markdown("---")
    st.caption(
        "Ensure that no sensitive information is uploaded to this tool. Review the generated "
        "content before applying it to production systems."
    )


if __name__ == "__main__":
    main()
