"""Configuration management for SmartFork."""

from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class SmartForkConfig(BaseSettings):
    """Configuration for SmartFork."""
    
    model_config = SettingsConfigDict(
        env_prefix="SMARTFORK_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )
    
    # Kilo Code paths
    kilo_code_tasks_path: Path = Field(
        default=Path.home() / "AppData/Roaming/Cursor/User/globalStorage/kilocode.kilo-code/tasks",
        description="Path to Kilo Code task storage"
    )
    
    # Database paths
    chroma_db_path: Path = Field(
        default=Path.home() / ".smartfork/chroma_db",
        description="Path to ChromaDB storage"
    )
    
    # Indexing settings
    chunk_size: int = Field(default=512, description="Size of chunks in words")
    chunk_overlap: int = Field(default=128, description="Overlap between chunks")
    
    # Search settings
    default_search_results: int = Field(default=10, description="Default number of search results")
    
    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
    log_file: Optional[Path] = Field(default=None, description="Log file path")


# Global config instance
_config: Optional[SmartForkConfig] = None


def get_config() -> SmartForkConfig:
    """Get the global configuration instance.
    
    Returns:
        SmartForkConfig instance
    """
    global _config
    if _config is None:
        _config = SmartForkConfig()
    return _config


def reload_config() -> SmartForkConfig:
    """Reload configuration from environment.
    
    Returns:
        SmartForkConfig instance
    """
    global _config
    _config = SmartForkConfig()
    return _config
