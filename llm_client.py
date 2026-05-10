import os
from pathlib import Path
from typing import Any, Optional

from openai import OpenAI


_ENV_LOADED = False


def _load_dotenv_if_exists() -> None:
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    _ENV_LOADED = True

    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.exists():
        return

    try:
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
    except Exception:
        # Keep runtime resilient even if .env has malformed lines.
        pass


def _read_env(name: str, default: Optional[str] = None) -> Optional[str]:
    _load_dotenv_if_exists()
    value = os.getenv(name, default)
    if value is None:
        return None
    return str(value).strip()


def get_llm_provider(default: str = "zhipu") -> str:
    provider = _read_env("LLM_PROVIDER", default) or default
    return provider.strip().lower()


def get_llm_base_url(default: Optional[str] = None) -> str:
    custom = _read_env("LLM_BASE_URL")
    if custom:
        return custom

    provider = get_llm_provider()
    provider_defaults = {
        # Project requirement: when provider is zhipu, call SiliconFlow OpenAI-compatible API.
        "zhipu": "https://api.siliconflow.cn/v1",
        "siliconflow": "https://api.siliconflow.cn/v1",
        "openai": "https://api.openai.com/v1",
    }
    if provider in provider_defaults:
        return provider_defaults[provider]

    if default:
        return default
    return "https://api.siliconflow.cn/v1"


def get_llm_api_key(default: Optional[str] = None) -> str:
    key = _read_env("LLM_API_KEY") or _read_env("OPENAI_API_KEY") or default
    if key:
        return key
    raise RuntimeError("Missing LLM API key. Please set LLM_API_KEY in environment or .env.")


def get_llm_model(default: str = "Pro/zai-org/GLM-4.7") -> str:
    return _read_env("LLM_MODEL", default) or default


def create_openai_client(base_url: Optional[str] = None, api_key: Optional[str] = None, chat_model_alias: Optional[str] = None) -> OpenAI:
    try:
        timeout = float(_read_env("LLM_TIMEOUT", "30") or 30)
    except Exception:
        timeout = 30.0
    client = OpenAI(
        api_key=api_key or get_llm_api_key(),
        base_url=base_url or get_llm_base_url(),
        timeout=timeout,
        max_retries=1,
    )
    # Keep model alias accessible for debugging without changing OpenAI client behavior.
    if chat_model_alias:
        setattr(client, "_chat_model_alias", chat_model_alias)
    return client


def extract_message_text(message: Any) -> str:
    if message is None:
        return ""

    # OpenAI ChatCompletionMessage
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text_part = item.get("text")
                if isinstance(text_part, str):
                    parts.append(text_part)
        if parts:
            return "\n".join(parts)

    if isinstance(message, dict):
        data_content = message.get("content")
        if isinstance(data_content, str):
            return data_content
        if isinstance(data_content, list):
            parts = []
            for item in data_content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    text_part = item.get("text")
                    if isinstance(text_part, str):
                        parts.append(text_part)
            if parts:
                return "\n".join(parts)

    return str(message)
