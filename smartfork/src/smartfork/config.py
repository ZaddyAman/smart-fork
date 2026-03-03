"""Configuration management for SmartFork."""

import json
import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, validator


# Performance constants
DEFAULT_ANIMATION_FPS = 10  # Reduced from 18 for lower CPU usage
LITE_MODE_ANIMATION_FPS = 5  # Minimal animation for lite mode
DEFAULT_BATCH_SIZE = 100  # Batch size for ChromaDB operations
MAX_CACHE_SIZE = 128  # Maximum LRU cache size for search results
DEFAULT_CACHE_TTL = 300  # Cache TTL in seconds (5 minutes)


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
    
    # Performance settings
    lite_mode: bool = Field(
        default=False, 
        description="Enable lite mode for minimal resource usage"
    )
    animation_fps: int = Field(
        default=DEFAULT_ANIMATION_FPS,
        ge=1,
        le=30,
        description="Animation frames per second (lower = less CPU)"
    )
    disable_animations: bool = Field(
        default=False,
        description="Disable all animations for minimal CPU usage"
    )
    batch_size: int = Field(
        default=DEFAULT_BATCH_SIZE,
        ge=10,
        le=1000,
        description="Batch size for database operations"
    )
    enable_search_cache: bool = Field(
        default=True,
        description="Enable LRU cache for search results"
    )
    search_cache_size: int = Field(
        default=MAX_CACHE_SIZE,
        ge=16,
        le=1024,
        description="Maximum number of cached search results"
    )
    search_cache_ttl: int = Field(
        default=DEFAULT_CACHE_TTL,
        ge=60,
        le=3600,
        description="Search cache TTL in seconds"
    )
    adaptive_fps: bool = Field(
        default=True,
        description="Automatically reduce FPS when user is inactive"
    )

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
            # Performance settings
            "lite_mode": self.lite_mode,
            "animation_fps": self.animation_fps,
            "disable_animations": self.disable_animations,
            "batch_size": self.batch_size,
            "enable_search_cache": self.enable_search_cache,
            "search_cache_size": self.search_cache_size,
            "search_cache_ttl": self.search_cache_ttl,
            "adaptive_fps": self.adaptive_fps,
        }
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=2)

    @classmethod
    def load(cls) -> "SmartForkConfig":
        """Load configuration from disk or create default."""
        # Create instance with defaults first
        instance = cls()
        
        # Check for environment variable lite mode
        if os.environ.get("SMARTFORK_LITE_MODE", "false").lower() == "true":
            instance.lite_mode = True
            instance.disable_animations = True
            instance.animation_fps = LITE_MODE_ANIMATION_FPS
        
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
                # Performance settings
                if "lite_mode" in data:
                    instance.lite_mode = data["lite_mode"]
                if "animation_fps" in data:
                    instance.animation_fps = data["animation_fps"]
                if "disable_animations" in data:
                    instance.disable_animations = data["disable_animations"]
                if "batch_size" in data:
                    instance.batch_size = data["batch_size"]
                if "enable_search_cache" in data:
                    instance.enable_search_cache = data["enable_search_cache"]
                if "search_cache_size" in data:
                    instance.search_cache_size = data["search_cache_size"]
                if "search_cache_ttl" in data:
                    instance.search_cache_ttl = data["search_cache_ttl"]
                if "adaptive_fps" in data:
                    instance.adaptive_fps = data["adaptive_fps"]
            except (json.JSONDecodeError, ValueError, TypeError):
                pass
        
        # Apply lite mode overrides if enabled
        if instance.lite_mode:
            instance.disable_animations = True
            instance.animation_fps = LITE_MODE_ANIMATION_FPS
            instance.enable_search_cache = True  # Cache helps reduce CPU
        
        return instance
    
    def get_effective_fps(self) -> int:
        """Get effective animation FPS based on settings."""
        if self.lite_mode or self.disable_animations:
            return LITE_MODE_ANIMATION_FPS
        return self.animation_fps
    
    def is_animation_enabled(self) -> bool:
        """Check if animations should be enabled."""
        return not (self.lite_mode or self.disable_animations)


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
