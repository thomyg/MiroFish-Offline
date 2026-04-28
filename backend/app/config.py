"""
Configuration Management
Loads configuration from .env file in project root directory.

Provider-related env vars (see docs/provider-config.md):

  LLM_PROVIDER          openai_compatible | azure_openai | anthropic_style
  LLM_API_KEY           Provider API key (any non-empty value works for Ollama)
  LLM_BASE_URL          Endpoint base URL
  LLM_MODEL_NAME        Model identifier
  LLM_DEPLOYMENT_NAME   Azure deployment name (Azure only)
  LLM_API_VERSION       Azure API version (Azure only)
  LLM_MAX_TOKENS        Default max tokens (provider-level default)
  LLM_TEMPERATURE       Default temperature

  EMBEDDING_PROVIDER    ollama | openai_compatible | azure_openai
  EMBEDDING_API_KEY     Provider API key (Ollama may leave blank)
  EMBEDDING_BASE_URL    Endpoint base URL
  EMBEDDING_MODEL       Embedding model identifier
  EMBEDDING_DEPLOYMENT_NAME  Azure deployment name (Azure only)
  EMBEDDING_API_VERSION      Azure API version (Azure only)
  EMBEDDING_DIMENSIONS  Vector dimensions (must match Neo4j vector index)

Defaults preserve the original Ollama-only setup.
"""

import os
from dotenv import load_dotenv

# Load .env file from project root
# Path: MiroFish/.env (relative to backend/app/config.py)
project_root_env = os.path.join(os.path.dirname(__file__), '../../.env')

if os.path.exists(project_root_env):
    load_dotenv(project_root_env, override=True)
else:
    # If no .env in root, try to load environment variables (for production)
    load_dotenv(override=True)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_str(name: str, default: str = "") -> str:
    val = os.environ.get(name)
    return val if val is not None else default


class Config:
    """Flask configuration class"""

    # Flask configuration
    SECRET_KEY = os.environ.get('SECRET_KEY', 'mirofish-secret-key')
    DEBUG = os.environ.get('FLASK_DEBUG', 'True').lower() == 'true'

    # JSON configuration - disable ASCII escaping to display Chinese directly (not as \uXXXX)
    JSON_AS_ASCII = False

    # ------------------------------------------------------------------
    # LLM provider configuration
    # ------------------------------------------------------------------
    LLM_PROVIDER = _env_str('LLM_PROVIDER', 'openai_compatible').strip().lower()
    LLM_API_KEY = _env_str('LLM_API_KEY')
    LLM_BASE_URL = _env_str('LLM_BASE_URL', 'http://localhost:11434/v1')
    LLM_MODEL_NAME = _env_str('LLM_MODEL_NAME', 'qwen2.5:32b')
    LLM_DEPLOYMENT_NAME = _env_str('LLM_DEPLOYMENT_NAME')
    LLM_API_VERSION = _env_str('LLM_API_VERSION', '2024-02-01')
    LLM_MAX_TOKENS = _env_int('LLM_MAX_TOKENS', 4096)
    LLM_TEMPERATURE = _env_float('LLM_TEMPERATURE', 0.7)
    # Anthropic-style providers (e.g. real Anthropic vs MiniMax) sometimes
    # disagree on the version header; allow override.
    LLM_ANTHROPIC_VERSION = _env_str('LLM_ANTHROPIC_VERSION', '2023-06-01')

    # ------------------------------------------------------------------
    # Embedding provider configuration
    # ------------------------------------------------------------------
    EMBEDDING_PROVIDER = _env_str('EMBEDDING_PROVIDER', 'ollama').strip().lower()
    EMBEDDING_API_KEY = _env_str('EMBEDDING_API_KEY')
    EMBEDDING_BASE_URL = _env_str('EMBEDDING_BASE_URL', 'http://localhost:11434')
    # EMBEDDING_MODEL kept for backwards compat; EMBEDDING_MODEL_NAME is the
    # canonical key used by the new provider config.
    EMBEDDING_MODEL = _env_str('EMBEDDING_MODEL_NAME') or _env_str('EMBEDDING_MODEL', 'nomic-embed-text')
    EMBEDDING_DEPLOYMENT_NAME = _env_str('EMBEDDING_DEPLOYMENT_NAME')
    EMBEDDING_API_VERSION = _env_str('EMBEDDING_API_VERSION', '2024-02-01')
    EMBEDDING_DIMENSIONS = _env_int('EMBEDDING_DIMENSIONS', 768)

    # ------------------------------------------------------------------
    # Neo4j configuration
    # ------------------------------------------------------------------
    NEO4J_URI = _env_str('NEO4J_URI', 'bolt://localhost:7687')
    NEO4J_USER = _env_str('NEO4J_USER', 'neo4j')
    NEO4J_PASSWORD = _env_str('NEO4J_PASSWORD', 'mirofish')

    # File upload configuration
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB
    UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), '../uploads')
    ALLOWED_EXTENSIONS = {'pdf', 'md', 'txt', 'markdown'}

    # Text processing configuration
    DEFAULT_CHUNK_SIZE = 500
    DEFAULT_CHUNK_OVERLAP = 50

    # OASIS simulation configuration
    OASIS_DEFAULT_MAX_ROUNDS = _env_int('OASIS_DEFAULT_MAX_ROUNDS', 10)
    OASIS_SIMULATION_DATA_DIR = os.path.join(os.path.dirname(__file__), '../uploads/simulations')

    # OASIS platform available actions configuration
    OASIS_TWITTER_ACTIONS = [
        'CREATE_POST', 'LIKE_POST', 'REPOST', 'FOLLOW', 'DO_NOTHING', 'QUOTE_POST'
    ]
    OASIS_REDDIT_ACTIONS = [
        'LIKE_POST', 'DISLIKE_POST', 'CREATE_POST', 'CREATE_COMMENT',
        'LIKE_COMMENT', 'DISLIKE_COMMENT', 'SEARCH_POSTS', 'SEARCH_USER',
        'TREND', 'REFRESH', 'DO_NOTHING', 'FOLLOW', 'MUTE'
    ]

    # Report Agent configuration
    REPORT_AGENT_MAX_TOOL_CALLS = _env_int('REPORT_AGENT_MAX_TOOL_CALLS', 5)
    REPORT_AGENT_MAX_REFLECTION_ROUNDS = _env_int('REPORT_AGENT_MAX_REFLECTION_ROUNDS', 2)
    REPORT_AGENT_TEMPERATURE = _env_float('REPORT_AGENT_TEMPERATURE', 0.5)

    @classmethod
    def validate(cls):
        """Validate required configuration. Returns a list of error strings (empty = OK)."""
        errors: list[str] = []

        # LLM
        if cls.LLM_PROVIDER not in {'openai_compatible', 'azure_openai', 'anthropic_style'}:
            errors.append(
                f"LLM_PROVIDER='{cls.LLM_PROVIDER}' is not supported "
                "(expected: openai_compatible | azure_openai | anthropic_style)"
            )
        if not cls.LLM_API_KEY:
            errors.append(
                "LLM_API_KEY not configured (set to any non-empty value, e.g. 'ollama')"
            )
        if not cls.LLM_BASE_URL:
            errors.append("LLM_BASE_URL not configured")
        if cls.LLM_PROVIDER == 'azure_openai' and not cls.LLM_DEPLOYMENT_NAME:
            errors.append("LLM_DEPLOYMENT_NAME is required when LLM_PROVIDER=azure_openai")

        # Embedding
        valid_emb = {'ollama', 'openai_compatible', 'azure_openai', 'minimax'}
        if cls.EMBEDDING_PROVIDER not in valid_emb:
            errors.append(
                f"EMBEDDING_PROVIDER='{cls.EMBEDDING_PROVIDER}' is not supported "
                f"(expected: {' | '.join(sorted(valid_emb))})"
            )
        if cls.EMBEDDING_PROVIDER == 'minimax' and not cls.EMBEDDING_API_KEY and not cls.LLM_API_KEY:
            errors.append("EMBEDDING_API_KEY (or LLM_API_KEY fallback) required when EMBEDDING_PROVIDER=minimax")
        if not cls.EMBEDDING_BASE_URL:
            errors.append("EMBEDDING_BASE_URL not configured")
        if cls.EMBEDDING_PROVIDER == 'openai_compatible' and not cls.EMBEDDING_API_KEY:
            # Allow blank for ollama, but require it for hosted OpenAI-compatible APIs.
            # Accept the LLM key as a fallback if the user is reusing one provider.
            if not cls.LLM_API_KEY:
                errors.append("EMBEDDING_API_KEY required when EMBEDDING_PROVIDER=openai_compatible")
        if cls.EMBEDDING_PROVIDER == 'azure_openai':
            if not cls.EMBEDDING_API_KEY:
                errors.append("EMBEDDING_API_KEY required when EMBEDDING_PROVIDER=azure_openai")
            if not cls.EMBEDDING_DEPLOYMENT_NAME:
                errors.append(
                    "EMBEDDING_DEPLOYMENT_NAME required when EMBEDDING_PROVIDER=azure_openai"
                )
        if cls.EMBEDDING_DIMENSIONS <= 0:
            errors.append(
                f"EMBEDDING_DIMENSIONS must be > 0 (got {cls.EMBEDDING_DIMENSIONS})"
            )

        # Neo4j
        if not cls.NEO4J_URI:
            errors.append("NEO4J_URI not configured")
        if not cls.NEO4J_PASSWORD:
            errors.append("NEO4J_PASSWORD not configured")

        return errors
