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
from .result_card import (
    build_result_card,
    render_result_cards,
)
from .syntax_highlight import (
    highlight_code,
    highlight_snippet,
    detect_language,
    detect_language_from_content,
)
from .markdown_render import (
    render_markdown,
    render_markdown_panel,
    render_markdown_snippet,
    ThemeAwareMarkdown,
)

__all__ = [
    # Progress
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
    # Result cards
    "build_result_card",
    "render_result_cards",
    # Syntax highlighting
    "highlight_code",
    "highlight_snippet",
    "detect_language",
    "detect_language_from_content",
    # Markdown rendering
    "render_markdown",
    "render_markdown_panel",
    "render_markdown_snippet",
    "ThemeAwareMarkdown",
]
