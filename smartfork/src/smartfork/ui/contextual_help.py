"""Contextual Help system for SmartFork CLI.

Provides smart suggestions based on user state to reduce onboarding friction
and guide users through the workflow.
"""

from pathlib import Path
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich import box

from ..ui.progress import DEFAULT_THEME, get_theme_colors, get_semantic_color


class UserAction(Enum):
    """Trackable user actions."""
    INDEX = "index"
    SEARCH = "search"
    DETECT_FORK = "detect_fork"
    FORK = "fork"
    STATUS = "status"
    WATCH = "watch"
    COMPACTION_CHECK = "compaction_check"
    CLUSTER_ANALYSIS = "cluster_analysis"
    TREE_BUILD = "tree_build"
    VAULT_ADD = "vault_add"
    CONFIG_SHOW = "config_show"
    HELP = "help"
    FIRST_INSTALL = "first_install"


class UserState(Enum):
    """User states for contextual help."""
    FRESH_INSTALL = "fresh_install"
    NO_INDEX = "no_index"
    HAS_INDEX = "has_index"
    NO_SEARCH_RESULTS = "no_search_results"
    HAS_SEARCH_RESULTS = "has_search_results"
    FIRST_FORK = "first_fork"
    ACTIVE_USER = "active_user"


@dataclass
class ActionHistory:
    """Record of a user action."""
    action: UserAction
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ContextualTip:
    """A contextual tip to show the user."""
    icon: str
    title: str
    message: str
    command: Optional[str] = None
    priority: int = 0  # Higher = more important
    show_once: bool = False  # If True, only show once


class ContextualHelpManager:
    """Manages contextual help and user state tracking.
    
    This class tracks user actions, detects their current state,
    and provides relevant tips and suggestions.
    """
    
    # ASCII-only icons for Windows compatibility
    ICONS = {
        "rocket": ">>",
        "search": "?", 
        "lightbulb": "*",
        "info": "i",
        "warning": "!",
        "success": "OK",
        "fork": "Y",
        "book": "=",
        "star": "*",
        "arrow": "->",
        "check": "[OK]",
        "database": "[#]",
        "clock": "[t]",
    }
    
    def __init__(self, console: Optional[Console] = None, theme_name: Optional[str] = None):
        """Initialize the contextual help manager.
        
        Args:
            console: Optional Rich console instance
            theme_name: Optional theme name override
        """
        self.console = console or Console()
        self.history: List[ActionHistory] = []
        self.state_file = Path.home() / ".smartfork" / "user_state.json"
        self.shown_tips: set = self._load_shown_tips()
        self._state_cache: Optional[Dict[UserState, bool]] = None
        self._theme_name = theme_name or DEFAULT_THEME
        self._theme = get_theme_colors(self._theme_name)
    
    def _get_theme_colors(self):
        """Get current theme colors."""
        from ..config import get_config
        try:
            config = get_config()
            theme_name = getattr(config, "theme", self._theme_name)
            return get_theme_colors(theme_name)
        except Exception:
            return self._theme
    
    def _get_semantic_colors(self):
        """Get semantic colors from current theme."""
        theme = self._get_theme_colors()
        semantic = theme.get("semantic", {})
        return {
            "info": semantic.get("info", theme["text_primary"]),
            "success": semantic.get("success", theme["done_color"]),
            "warning": semantic.get("warning", "#F59E0B"),
            "error": semantic.get("error", "#EF4444"),
            "accent": semantic.get("accent", theme["text_primary"]),
            "text_primary": theme["text_primary"],
            "text_muted": theme["text_muted"],
            "panel_border": theme["panel_border"],
        }
    
    def _load_shown_tips(self) -> set:
        """Load the set of tips already shown to user."""
        if self.state_file.exists():
            try:
                data = json.loads(self.state_file.read_text())
                return set(data.get("shown_tips", []))
            except (json.JSONDecodeError, IOError):
                pass
        return set()
    
    def _save_shown_tips(self):
        """Save the set of shown tips."""
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "shown_tips": list(self.shown_tips),
                "last_updated": datetime.now().isoformat()
            }
            self.state_file.write_text(json.dumps(data, indent=2))
        except IOError:
            pass  # Silently fail if we can't write
    
    def record_action(self, action: UserAction, **metadata):
        """Record a user action.
        
        Args:
            action: The action performed
            **metadata: Additional context about the action
        """
        history_entry = ActionHistory(
            action=action,
            timestamp=datetime.now(),
            metadata=metadata
        )
        self.history.append(history_entry)
        # Invalidate state cache
        self._state_cache = None
    
    def get_recent_actions(self, since_hours: int = 24) -> List[ActionHistory]:
        """Get actions from the last N hours.
        
        Args:
            since_hours: Number of hours to look back
            
        Returns:
            List of recent actions
        """
        cutoff = datetime.now() - timedelta(hours=since_hours)
        return [h for h in self.history if h.timestamp > cutoff]
    
    def has_action(self, action: UserAction, since_hours: int = 24) -> bool:
        """Check if user has performed an action recently.
        
        Args:
            action: Action to check for
            since_hours: Time window to check
            
        Returns:
            True if action was performed
        """
        recent = self.get_recent_actions(since_hours)
        return any(h.action == action for h in recent)
    
    def detect_state(self, db_session_count: int = 0) -> Dict[UserState, bool]:
        """Detect the current user state.
        
        Args:
            db_session_count: Number of sessions in database
            
        Returns:
            Dictionary mapping states to boolean values
        """
        if self._state_cache is not None:
            return self._state_cache
        
        state = {}
        
        # Check if state file exists (indicates not first install)
        state[UserState.FRESH_INSTALL] = not self.state_file.exists()
        
        # Check if database has any sessions
        state[UserState.NO_INDEX] = db_session_count == 0
        state[UserState.HAS_INDEX] = db_session_count > 0
        
        # Check recent search results
        recent_searches = [
            h for h in self.get_recent_actions(since_hours=1)
            if h.action == UserAction.SEARCH
        ]
        if recent_searches:
            last_search = recent_searches[-1]
            result_count = last_search.metadata.get("result_count", 0)
            state[UserState.NO_SEARCH_RESULTS] = result_count == 0
            state[UserState.HAS_SEARCH_RESULTS] = result_count > 0
        else:
            state[UserState.NO_SEARCH_RESULTS] = False
            state[UserState.HAS_SEARCH_RESULTS] = False
        
        # Check if first fork
        state[UserState.FIRST_FORK] = not self.has_action(
            UserAction.FORK, since_hours=24*30
        ) and self.has_action(UserAction.FORK, since_hours=1)
        
        # Active user has indexed and searched
        state[UserState.ACTIVE_USER] = (
            self.has_action(UserAction.INDEX, since_hours=24*7) and
            self.has_action(UserAction.SEARCH, since_hours=24*7)
        )
        
        self._state_cache = state
        return state
    
    def should_show_welcome(self) -> bool:
        """Check if welcome message should be shown.
        
        Returns:
            True if this appears to be first run
        """
        return not self.state_file.exists()
    
    def mark_welcome_shown(self):
        """Mark that welcome has been shown."""
        self.shown_tips.add("welcome")
        self._save_shown_tips()
    
    def _was_tip_shown(self, tip_id: str) -> bool:
        """Check if a tip was already shown."""
        return tip_id in self.shown_tips
    
    def _mark_tip_shown(self, tip_id: str):
        """Mark a tip as shown."""
        self.shown_tips.add(tip_id)
        self._save_shown_tips()

    def get_welcome_message(self) -> Panel:
        """Get the welcome message panel for first-time users.
        
        Returns:
            Rich Panel with welcome message
        """
        colors = self._get_semantic_colors()
        
        content = Text()
        content.append("Welcome to SmartFork!\n\n", style=f"bold {colors['info']}")
        content.append("SmartFork helps you reuse context from your Kilo Code sessions.\n\n")
        
        content.append("Quick Start:\n", style=f"bold {colors['text_primary']}")
        steps = [
            ("1. Index your sessions", "smartfork index"),
            ("2. Search for relevant context", "smartfork search 'your task'"),
            ("3. Fork context when needed", "smartfork fork <session_id>"),
        ]
        for step, cmd in steps:
            content.append(f"  {step}\n", style=f"dim {colors['text_muted']}")
            content.append(f"     ", style=f"dim {colors['text_muted']}")
            content.append(f"$ {cmd}\n", style=colors['success'])
        
        content.append("\n")
        content.append("Tip: ", style=colors['warning'])
        content.append("Use --watch during indexing to auto-update.\n", style=f"dim {colors['text_muted']}")
        
        return Panel(
            content,
            title=f"[bold {colors['info']}]SmartFork - AI Session Intelligence[/bold {colors['info']}]",
            border_style=colors['panel_border'],
            box=box.ROUNDED
        )
    
    def get_post_index_tips(self) -> Optional[Panel]:
        """Get tips to show after indexing.
        
        Returns:
            Rich Panel with suggestions, or None if no tips
        """
        tips = []
        
        # Primary tip: Try searching
        if not self._was_tip_shown("index_to_search"):
            tips.append(ContextualTip(
                icon=self.ICONS["search"],
                title="Next Step: Search",
                message="Your sessions are now indexed! Try searching for relevant context.",
                command="smartfork search 'your task description'",
                priority=10,
                show_once=True
            ))
            self._mark_tip_shown("index_to_search")
        
        # Secondary tip: Enable watch mode next time
        if not self._was_tip_shown("watch_mode"):
            tips.append(ContextualTip(
                icon=self.ICONS["clock"],
                title="Pro Tip: Watch Mode",
                message="Next time, use --watch to auto-index new sessions as they're created.",
                command="smartfork index --watch",
                priority=5,
                show_once=True
            ))
        
        return self._build_tips_panel(tips, "What's Next?") if tips else None
    
    def get_no_results_tips(
        self, 
        query: str, 
        has_recent_index: bool = False
    ) -> Optional[Panel]:
        """Get tips when search returns no results.
        
        Args:
            query: The search query that returned no results
            has_recent_index: Whether there are recently indexed sessions
            
        Returns:
            Rich Panel with suggestions, or None
        """
        tips = []
        
        if has_recent_index:
            tips.append(ContextualTip(
                icon=self.ICONS["info"],
                title="No Results Found",
                message="Try a broader search query with fewer specific terms.",
                priority=10
            ))
            tips.append(ContextualTip(
                icon=self.ICONS["lightbulb"],
                title="Search Tips",
                message="Use general terms like 'react auth' instead of specific filenames.",
                priority=5
            ))
        else:
            tips.append(ContextualTip(
                icon=self.ICONS["warning"],
                title="No Results Found",
                message="Your sessions may need re-indexing if they were created recently.",
                command="smartfork index",
                priority=10
            ))
        
        return self._build_tips_panel(tips, "Search Suggestions")
    
    def get_post_fork_tips(self, session_id: str) -> Optional[Panel]:
        """Get tips after forking a session.
        
        Args:
            session_id: The forked session ID
            
        Returns:
            Rich Panel with suggestions, or None
        """
        tips = []
        
        if not self._was_tip_shown("fork_success"):
            tips.append(ContextualTip(
                icon=self.ICONS["success"],
                title="Fork Generated!",
                message="The fork.md file contains context from the selected session.",
                command=f"cat fork_{session_id[:8]}.md",
                priority=10,
                show_once=True
            ))
            self._mark_tip_shown("fork_success")
        
        # Suggest detect-fork for next time
        if not self._was_tip_shown("detect_fork_tip"):
            tips.append(ContextualTip(
                icon=self.ICONS["lightbulb"],
                title="Alternative: Detect-Fork",
                message="Use detect-fork to automatically find relevant sessions.",
                command="smartfork detect-fork 'your task'",
                priority=5,
                show_once=True
            ))
        
        return self._build_tips_panel(tips, "Fork Complete") if tips else None
    
    def get_status_tips(
        self, 
        total_tasks: int, 
        indexed_sessions: int
    ) -> Optional[Panel]:
        """Get tips based on status command output.
        
        Args:
            total_tasks: Total task directories found
            indexed_sessions: Number of indexed sessions
            
        Returns:
            Rich Panel with suggestions, or None
        """
        tips = []
        
        if total_tasks == 0:
            tips.append(ContextualTip(
                icon=self.ICONS["warning"],
                title="No Kilo Code Sessions Found",
                message="Make sure Kilo Code is installed and has created task directories.",
                priority=10
            ))
        elif indexed_sessions == 0:
            tips.append(ContextualTip(
                icon=self.ICONS["info"],
                title="Get Started",
                message="You have sessions but haven't indexed them yet.",
                command="smartfork index",
                priority=10
            ))
        elif indexed_sessions < total_tasks:
            missing = total_tasks - indexed_sessions
            tips.append(ContextualTip(
                icon=self.ICONS["info"],
                title="Incomplete Index",
                message=f"{missing} session(s) not indexed. Run index to update.",
                command="smartfork index",
                priority=8
            ))
        
        # Always show a random power tip for active users
        if indexed_sessions > 0 and not tips:
            power_tips = [
                ContextualTip(
                    icon=self.ICONS["star"],
                    title="Power Tip: Compaction Check",
                    message="Check for sessions at risk of compaction before data loss.",
                    command="smartfork compaction-check",
                    priority=3
                ),
                ContextualTip(
                    icon=self.ICONS["star"],
                    title="Power Tip: Cluster Analysis",
                    message="Find duplicate sessions and analyze your coding patterns.",
                    command="smartfork cluster-analysis",
                    priority=3
                ),
                ContextualTip(
                    icon=self.ICONS["star"],
                    title="Power Tip: Tree Visualization",
                    message="Visualize how your conversations branch and evolve.",
                    command="smartfork tree-visualize",
                    priority=3
                ),
            ]
            # Show a power tip occasionally
            import random
            if random.random() < 0.3:  # 30% chance
                tips.append(random.choice(power_tips))
        
        return self._build_tips_panel(tips, "Tips") if tips else None
    
    def _build_tips_panel(
        self, 
        tips: List[ContextualTip], 
        title: str
    ) -> Optional[Panel]:
        """Build a panel from a list of tips.
        
        Args:
            tips: List of tips to display
            title: Panel title
            
        Returns:
            Rich Panel or None if no tips
        """
        if not tips:
            return None
        
        colors = self._get_semantic_colors()
        
        # Sort by priority (highest first)
        tips.sort(key=lambda t: t.priority, reverse=True)
        
        content = Text()
        for i, tip in enumerate(tips):
            if i > 0:
                content.append("\n\n")
            
            # Icon and title
            content.append(f"{tip.icon} ", style=colors['info'])
            content.append(f"{tip.title}\n", style=f"bold {colors['text_primary']}")
            
            # Message
            content.append(f"  {tip.message}\n", style=f"dim {colors['text_muted']}")
            
            # Command if present
            if tip.command:
                content.append(f"  ", style=f"dim {colors['text_muted']}")
                content.append(f"$ {tip.command}", style=colors['success'])
        
        return Panel(
            content,
            title=f"[bold {colors['info']}]{title}[/bold {colors['info']}]",
            border_style=colors['panel_border'],
            box=box.ROUNDED
        )
    
    def show_tip(
        self, 
        title: str, 
        message: str, 
        icon: str = "info",
        command: Optional[str] = None
    ):
        """Show a simple tip panel.
        
        Args:
            title: Tip title
            message: Tip message
            icon: Icon key to use
            command: Optional command to show
        """
        colors = self._get_semantic_colors()
        
        content = Text()
        content.append(f"{self.ICONS.get(icon, self.ICONS['info'])} ", style=colors['info'])
        content.append(f"{title}\n", style=f"bold {colors['text_primary']}")
        content.append(f"  {message}\n", style=f"dim {colors['text_muted']}")
        if command:
            content.append(f"  ", style=f"dim {colors['text_muted']}")
            content.append(f"$ {command}", style=colors['success'])
        
        panel = Panel(
            content,
            border_style=colors['panel_border'],
            box=box.ROUNDED
        )
        self.console.print(panel)
    
    def show_after_command(
        self,
        action: UserAction,
        db_session_count: int = 0,
        **context
    ):
        """Show contextual help after a command completes.
        
        This is the main entry point for showing contextual tips.
        
        Args:
            action: The command that just completed
            db_session_count: Current number of indexed sessions
            **context: Additional context (query, result_count, etc.)
        """
        # Record the action
        self.record_action(action, **context)
        
        # Show welcome on first command
        if self.should_show_welcome():
            self.console.print(self.get_welcome_message())
            self.mark_welcome_shown()
            return
        
        # Get appropriate tips based on action
        panel = None
        
        if action == UserAction.INDEX:
            panel = self.get_post_index_tips()
        
        elif action == UserAction.SEARCH:
            result_count = context.get("result_count", 0)
            if result_count == 0:
                query = context.get("query", "")
                has_recent = self.has_action(UserAction.INDEX, since_hours=24)
                panel = self.get_no_results_tips(query, has_recent)
        
        elif action == UserAction.FORK:
            session_id = context.get("session_id", "")
            panel = self.get_post_fork_tips(session_id)
        
        elif action == UserAction.STATUS:
            total_tasks = context.get("total_tasks", 0)
            indexed = context.get("indexed_sessions", 0)
            panel = self.get_status_tips(total_tasks, indexed)
        
        if panel:
            self.console.print()
            self.console.print(panel)


# Global instance for easy access
_help_manager: Optional[ContextualHelpManager] = None


def get_help_manager(console: Optional[Console] = None, theme_name: Optional[str] = None) -> ContextualHelpManager:
    """Get the global help manager instance.
    
    Args:
        console: Optional Rich console
        theme_name: Optional theme name override
        
    Returns:
        ContextualHelpManager instance
    """
    global _help_manager
    if _help_manager is None:
        _help_manager = ContextualHelpManager(console, theme_name=theme_name)
    return _help_manager


def reset_help_manager():
    """Reset the global help manager (useful for testing)."""
    global _help_manager
    _help_manager = None
