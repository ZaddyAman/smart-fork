"""Configuration management for SmartFork."""

import json
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, validator


# Config file path
CONFIG_DIR = Path.home() / ".smartfork"
CONFIG_FILE = CONFIG_DIR / "config.json"


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
    
    cache_dir: Path = Field(
        default=Path.home() / ".smartfork/cache",
        description="Path to cache directory"
    )
    
    # Indexing settings
    chunk_size: int = Field(default=512, description="Size of chunks in words")
    chunk_overlap: int = Field(default=128, description="Overlap between chunks")
    
    # Search settings
    default_search_results: int = Field(default=10, description="Default number of search results")
    
    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
    log_file: Optional[Path] = Field(default=None, description="Log file path")

    # Theme
    theme: str = Field(default="obsidian", description="UI theme (phosphor, obsidian, ember, arctic, iron, tungsten)")

    @validator("theme")
    def validate_theme(cls, v):
        valid = {"phosphor", "obsidian", "ember", "arctic", "iron", "tungsten"}
        v = v.lower()
        if v not in valid:
            raise ValueError(f"Unknown theme '{v}'. Valid: {', '.join(sorted(valid))}")
        return v

    def save(self):
        """Save configuration to disk."""
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        config_data = {
            "theme": self.theme,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "default_search_results": self.default_search_results,
            "log_level": self.log_level,
            "log_file": str(self.log_file) if self.log_file else None,
        }
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=2)

    @classmethod
    def load(cls) -> "SmartForkConfig":
        """Load configuration from disk or create default."""
        # Create instance with defaults first
        instance = cls()
        
        # Override with saved config if exists
        if CONFIG_FILE.exists():
            try:
                with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # Update fields from saved data
                if "theme" in data:
                    theme_val = data["theme"].lower()
                    valid_themes = {"phosphor", "obsidian", "ember", "arctic", "iron", "tungsten"}
                    if theme_val in valid_themes:
                        instance.theme = theme_val
                if "chunk_size" in data:
                    instance.chunk_size = data["chunk_size"]
                if "chunk_overlap" in data:
                    instance.chunk_overlap = data["chunk_overlap"]
                if "default_search_results" in data:
                    instance.default_search_results = data["default_search_results"]
                if "log_level" in data:
                    instance.log_level = data["log_level"]
                if "log_file" in data and data["log_file"]:
                    instance.log_file = Path(data["log_file"])
            except (json.JSONDecodeError, ValueError, TypeError):
                pass
        
        return instance


# Global config instance
_config: Optional[SmartForkConfig] = None


def get_config() -> SmartForkConfig:
    """Get the global configuration instance.
    
    Returns:
        SmartForkConfig instance
    """
    global _config
    if _config is None:
        _config = SmartForkConfig.load()
    return _config


def reload_config() -> SmartForkConfig:
    """Reload configuration from disk.
    
    Returns:
        SmartForkConfig instance
    """
    global _config
    _config = SmartForkConfig.load()
    return _config
