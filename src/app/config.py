"""Application configuration, loaded from environment only (no hardcoded secrets)."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# repo root = .../ai-engineer-test-materials
REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / "src" / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_env: str = "local"
    log_level: str = "info"
    api_port: int = 8000

    # --- LLM gateway (OpenAI-compatible) ---
    llm_api_key: str
    llm_api_base_url: str = "https://api.ntq.ai"
    llm_model_name: str = "nxchat"
    text_embedding_model_name: str = "nx-text-embedding"

    # Verified against the gateway during Phase 0 capability probe.
    embedding_dim: int = 1024

    # --- Retrieval ---
    knowledge_base_dir: Path = REPO_ROOT / "knowledge-base"
    workout_history_path: Path = REPO_ROOT / "sample-data" / "workout-history.json"
    chroma_dir: Path = REPO_ROOT / "data" / "chroma"
    chroma_collection: str = "fitness_kb"
    retrieval_top_k: int = 5
    # Below this best-match cosine similarity, treat the question as out of scope
    # rather than answering from weak context. Tuned in Phase 5 against the eval set.
    relevance_threshold: float = 0.35

    # --- Guardrails ---
    # Safety classifier runs concurrently with embedding, so it adds no latency
    # on the happy path. Toggle for the eval pipeline's guardrail-off baseline.
    guardrail_enabled: bool = True

    # --- Generation ---
    request_timeout_s: float = 60.0
    max_retries: int = 2
    agent_max_iterations: int = 5

    # --- Cost accounting (README deliverable) ---
    # Assumes OpenAI list pricing; the gateway publishes no rates, so these are
    # declared here and applied to *measured* token counts rather than guessed.
    usd_per_1m_input: float = Field(default=0.15)
    usd_per_1m_output: float = Field(default=0.60)
    usd_per_1m_embedding: float = Field(default=0.02)

    @property
    def base_url(self) -> str:
        """Normalised OpenAI-style base URL (always ends in /v1)."""
        root = self.llm_api_base_url.rstrip("/")
        return root if root.endswith("/v1") else f"{root}/v1"


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
