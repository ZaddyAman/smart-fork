"""UI components for SmartFork."""

from .progress import (
    SmartForkProgress,
    IndexingStats,
    display_discovery_phase,
    display_completion_summary,
    THEMES,
    DEFAULT_THEME,
    get_theme_colors,
    get_semantic_color,
    # Backward compatibility
    AnimatedProgressDisplay,
    IndexingProgressDisplay,
    index_with_progress,
)
from .contextual_help import (
    ContextualHelpManager,
    UserAction,
    UserState,
    get_help_manager,
)
from .interactive import (
    SmartForkShell,
    start_interactive_shell,
)

__all__ = [
    # New progress system
    "SmartForkProgress",
    "IndexingStats",
    "display_discovery_phase",
    "display_completion_summary",
    "THEMES",
    "DEFAULT_THEME",
    "get_theme_colors",
    "get_semantic_color",
    # Backward compatibility
    "AnimatedProgressDisplay",
    "IndexingProgressDisplay",
    "index_with_progress",
    # Contextual help
    "ContextualHelpManager",
    "UserAction",
    "UserState",
    "get_help_manager",
    # Interactive shell
    "SmartForkShell",
    "start_interactive_shell",
]
