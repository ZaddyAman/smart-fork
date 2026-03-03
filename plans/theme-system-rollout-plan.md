# SmartFork Theme System Rollout Plan

## Current State Analysis

### Existing Theme Structure (6 themes)
Each theme has:
- `panel_border`: Panel border color
- `bars[3]`: Progress bar colors (sessions, embedding, bm25)
- `text_primary`: Main text color
- `text_muted`: Secondary text color  
- `text_dim`: Very dim text
- `spinner_color`: Animation spinner color
- `done_color`: Success/completion color

### Hardcoded Colors Found
**CLI Commands using hardcoded colors:**
- `status`, `config_show`: cyan (property labels), green (values)
- `compaction_check`: cyan, yellow, blue, red
- `compaction_export`: cyan, green
- `cluster_analysis`: cyan, green, blue
- `vault_list`: cyan, green, blue
- `test`: cyan, red, dim
- `metrics`: cyan, green
- `ab_test_status`: cyan, green, blue, yellow, magenta

**UI Components using hardcoded colors:**
- `contextual_help.py`: cyan (headers), green (commands), yellow (tips), blue (borders)
- `interactive.py`: cyan (borders), green (values)

---

## Phase 1: Extend Theme System with Semantic Colors

### 1.1 Add Semantic Color Mappings to All Themes

Add these keys to each theme in `smartfork/ui/progress.py`:

```python
"semantic": {
    "success": "#22C55E",      # Completion, success messages
    "warning": "#F59E0B",      # Warnings, cautions
    "error": "#EF4444",        # Errors, failures
    "info": "#38BDF8",         # Info labels, IDs
    "accent": "#A78BFA",       # Highlights, special values
    "metric_good": "#4ADE80",  # Good metrics
    "metric_bad": "#F87171",   # Bad metrics
}
```

**Theme-specific semantic colors:**

| Theme | success | warning | error | info | accent |
|-------|---------|---------|-------|------|--------|
| phosphor | #4ADE80 | #FBBF24 | #F87171 | #4ADE80 | #86EFAC |
| obsidian | #94A3B8 | #FCD34D | #FCA5A5 | #94A3B8 | #CBD5E1 |
| ember | #F59E0B | #FCD34D | #EF4444 | #F59E0B | #FCD34D |
| arctic | #38BDF8 | #7DD3FC | #F87171 | #38BDF8 | #BAE6FD |
| iron | #9B8EC4 | #C4BAE8 | #F87171 | #9B8EC4 | #C4BAE8 |
| tungsten | #A3A3A3 | #D4D4D4 | #737373 | #A3A3A3 | #D4D4D4 |

---

## Phase 2: Create Theme Helper Infrastructure

### 2.1 Create `get_themed_console()` Helper

In `smartfork/ui/progress.py`, add:

```python
def get_theme_colors(theme_name: str = DEFAULT_THEME) -> dict:
    """Get full color palette for a theme."""
    theme = THEMES.get(theme_name, THEMES[DEFAULT_THEME])
    return theme

def get_semantic_color(theme_name: str, semantic_type: str) -> str:
    """Get semantic color from theme.
    
    Args:
        theme_name: Theme identifier
        semantic_type: One of: success, warning, error, info, accent
    """
    theme = THEMES.get(theme_name, THEMES[DEFAULT_THEME])
    semantic = theme.get("semantic", {})
    return semantic.get(semantic_type, theme["text_primary"])
```

### 2.2 Export Theme Utilities

Update `smartfork/ui/__init__.py` to export theme functions:

```python
from .progress import (
    THEMES, DEFAULT_THEME, get_theme_colors, get_semantic_color,
    SmartForkProgress, display_discovery_phase, display_completion_summary
)
```

---

## Phase 3: Update CLI Commands

### 3.1 Create CLI Theme Mixin Pattern

Add to `smartfork/cli.py`:

```python
class ThemeMixin:
    """Mixin to add theme support to CLI commands."""
    
    def __init__(self):
        self._theme_name = None
        self._theme = None
    
    @property
    def theme_name(self) -> str:
        if self._theme_name is None:
            config = get_config()
            self._theme_name = getattr(config, "theme", DEFAULT_THEME)
        return self._theme_name
    
    @property
    def theme(self) -> dict:
        if self._theme is None:
            from .ui.progress import THEMES, DEFAULT_THEME
            self._theme = THEMES.get(self.theme_name, THEMES[DEFAULT_THEME])
        return self._theme
    
    def style(self, semantic_type: str) -> str:
        """Get styled color for semantic type."""
        semantic = self.theme.get("semantic", {})
        return semantic.get(semantic_type, self.theme["text_primary"])
```

### 3.2 Update Each CLI Command

**Pattern for updating commands:**

```python
@app.command()
def status():
    """Show indexing status."""
    config = get_config()
    from .ui.progress import THEMES, DEFAULT_THEME
    theme = THEMES.get(getattr(config, "theme", DEFAULT_THEME), THEMES[DEFAULT_THEME])
    semantic = theme.get("semantic", {})
    
    # Use theme colors instead of hardcoded
    info_color = semantic.get("info", theme["text_primary"])
    success_color = semantic.get("success", theme["done_color"])
    
    table.add_column("Property", style=info_color)
    table.add_column("Value", style=success_color)
```

**Commands to update:**
1. `status` - Property labels (info), values (success)
2. `config_show` - Setting names (info), values (success)
3. `compaction_check` - Session (info), Messages (warning), Age (accent), Risk (error)
4. `compaction_export` - Session (info), Action (success)
5. `cluster_analysis` - Cluster (info), Sessions (success), Technologies (accent)
6. `vault_list` - Session (info), Vaulted At (success), Files (accent)
7. `test` - Test (info), Status (error), Error (dim)
8. `metrics` - Metric (info), Value (success)
9. `ab_test_status` - Test (info), Sessions (success), Control (accent), Treatment (warning), Result (accent)

---

## Phase 4: Update UI Components

### 4.1 Update `contextual_help.py`

Replace hardcoded colors with theme-aware styling:

```python
# Current:
content.append("Welcome to SmartFork!\n\n", style="bold cyan")
content.append(f"$ {cmd}\n", style="green")
content.append("Tip: ", style="yellow")

# New:
from ..ui.progress import get_semantic_color
primary = get_semantic_color(theme_name, "info")
success = get_semantic_color(theme_name, "success")
warning = get_semantic_color(theme_name, "warning")

content.append("Welcome to SmartFork!\n\n", style=f"bold {primary}")
content.append(f"$ {cmd}\n", style=success)
content.append("Tip: ", style=warning)
```

**Files to update:**
- Welcome panel borders
- Command styling
- Tip/warning styling
- Help panel borders

### 4.2 Update `interactive.py`

Replace hardcoded colors:

```python
# Current:
welcome_text.append("SmartFork Interactive Shell\n", style="bold cyan")
table.add_column("Property", style="cyan")

# New:
from .progress import get_semantic_color, get_theme_colors
theme = get_theme_colors(config.theme)
primary = theme["text_primary"]
info = theme["semantic"]["info"]

welcome_text.append("SmartFork Interactive Shell\n", style=f"bold {primary}")
table.add_column("Property", style=info)
```

---

## Phase 5: Error & Warning Color Consistency

### 5.1 Standardize Error Display

Replace all `style="red"` with theme-aware error color:

```python
# Current:
console.print(f"[red]Error: {message}[/red]")

# New:
from .ui.progress import get_semantic_color
error_color = get_semantic_color(theme_name, "error")
console.print(f"[{error_color}]Error: {message}[/{error_color}]")
```

### 5.2 Standardize Warning Display

Replace all `style="yellow"` with theme-aware warning color:

```python
# Current:
console.print(f"[yellow]Warning: {message}[/yellow]")

# New:
warning_color = get_semantic_color(theme_name, "warning")
console.print(f"[{warning_color}]Warning: {message}[/{warning_color}]")
```

---

## Phase 6: Border & Panel Consistency

### 6.1 Panel Border Theming

Update all `Panel` and `Panel.fit` calls to use theme borders:

```python
# Current:
console.print(Panel.fit(
    "[bold blue]SmartFork Status[/bold blue]",
    title="Status"
))

# New:
theme = get_theme_colors(config.theme)
console.print(Panel.fit(
    f"[bold {theme['text_primary']}]SmartFork Status[/bold {theme['text_primary']}]",
    title="Status",
    border_style=theme["panel_border"]
))
```

---

## Implementation Priority

### High Priority (Core UX)
1. ✅ `index` command - Already themed
2. `search` command - Uses results in themed panels
3. `detect_fork` command - Result panels
4. `fork` command - Output confirmation
5. `status` command - Status display

### Medium Priority (Secondary Commands)
6. `config_show` and `config-theme` - Configuration display
7. `contextual_help.py` - Help system
8. `interactive.py` - Interactive shell

### Lower Priority (Advanced Features)
9. `compaction_check/export` - Intelligence layer
10. `cluster_analysis` - Clustering
11. `tree_*` commands - Branching
12. `vault_*` commands - Privacy
13. `test`, `metrics`, `ab_test_status` - Testing

---

## Testing Checklist

- [ ] Each theme renders correctly in all commands
- [ ] Success messages use theme's success color
- [ ] Errors use theme's error color
- [ ] Warnings use theme's warning color
- [ ] Panel borders match theme aesthetic
- [ ] Interactive shell uses theme colors
- [ ] Help system uses theme colors
- [ ] Switching themes updates all outputs

---

## Files to Modify

1. `smartfork/ui/progress.py` - Add semantic colors
2. `smartfork/ui/__init__.py` - Export theme utilities
3. `smartfork/cli.py` - Update all commands
4. `smartfork/ui/contextual_help.py` - Theme-aware help
5. `smartfork/ui/interactive.py` - Theme-aware interactive shell

---

## Estimated Effort

- Phase 1 (Theme extension): ~30 minutes
- Phase 2 (Infrastructure): ~30 minutes
- Phase 3 (CLI commands): ~2 hours
- Phase 4 (UI components): ~1 hour
- Phase 5 (Error standardization): ~30 minutes
- Phase 6 (Border consistency): ~30 minutes
- Testing & refinement: ~1 hour

**Total: ~6 hours**
