import os
from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict

class AgentSettings(BaseSettings):
    """Agent configuration from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8", 
        case_sensitive=False,
        extra="ignore"
    )

    # Server settings
    agent_host: str = "0.0.0.0"
    agent_port: int = 8000  # Databricks Apps default port

    # Agent metadata
    agent_name: str = "SOP Assistant"
    agent_description: str = "AI-powered assistant for SOPs"
    agent_version: str = "1.0.0"
    agent_url: str = os.getenv("DATABRICKS_APP_URL", "http://localhost:8000")

    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # --- Databricks Configuration ---
    # Default Vector Search Config (Fallbacks if config.json is missing)
    default_vs_endpoint_name: str = os.getenv("VECTOR_SEARCH_ENDPOINT_NAME", "sop-policy-vectors")
    default_vs_index_name: str = os.getenv("VECTOR_SEARCH_INDEX_NAME", "agent_brick.default.sop_policy_index")
    
    # Persistence (Unity Catalog Volume Path)
    checkpoint_path: str = os.getenv("CHECKPOINT_PATH", "/Volumes/agent_brick/default/checkpoints/sop_agent.sqlite")

settings = AgentSettings()