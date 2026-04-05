"""Markdown rendering for terminal output using Rich.

Provides theme-aware markdown rendering with syntax-highlighted code blocks.
Supports headings, bold, italic, code, lists, blockquotes, tables, and links.
"""

from typing import Optional
from rich.console import Console, ConsoleOptions, RenderResult
from rich.markdown import Markdown, MarkdownContext
from rich.markdown import CodeBlock, Heading, Paragraph, ListElement, BlockQuote, Table
from rich.text import Text
from rich.panel import Panel
from rich.table import Table as RichTable
from rich.box import ROUNDED, SIMPLE

from .syntax_highlight import highlight_code, detect_language, detect_language_from_content
from .progress import DEFAULT_THEME, get_theme_colors


class ThemeAwareCodeBlock(CodeBlock):
    """Code block with Pygments syntax highlighting instead of Rich's default."""
    
    def __init__(self, *args, theme_name: str = DEFAULT_THEME, **kwargs):
        super().__init__(*args, **kwargs)
        self._theme_name = theme_name
    
    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        text = self.text
        
        lang = None
        if self.lexer_name:
            lang = self.lexer_name
        
        if not lang:
            lang = detect_language_from_content(text)
        
        theme = get_theme_colors(self._theme_name)
        highlighted = highlight_code(text, lang, theme_name=self._theme_name)
        
        title = f" {lang}" if lang else " code "
        yield Panel(
            highlighted,
            border_style=theme["text_muted"],
            title=title,
            title_align="left",
            padding=(0, 1),
            box=SIMPLE,
        )


class ThemeAwareHeading(Heading):
    """Heading with theme-aware colors."""
    
    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        text = Text()
        text.append(self.text, style=f"bold #{_get_theme_accent()}")
        text.append("\n")
        yield text


class ThemeAwareBlockQuote(BlockQuote):
    """Blockquote with theme-aware styling."""
    
    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        # Render children
        for child in self.children:
            for element in child:
                if isinstance(element, Text):
                    # Wrap in italic dim with quote prefix
                    lines = str(element).split("\n")
                    for line in lines:
                        t = Text()
                        t.append("  ", style="dim")
                        t.append("> ", style="bold dim")
                        t.append(line, style="italic dim")
                        t.append("\n")
                        yield t
                else:
                    yield element


class ThemeAwareMarkdown(Markdown):
    """Markdown renderer with theme-aware colors and Pygments syntax highlighting."""
    
    elements = {
        **Markdown.elements,
        "fence": ThemeAwareCodeBlock,
        "heading_open": ThemeAwareHeading,
        "blockquote_open": ThemeAwareBlockQuote,
    }
    
    def __init__(self, markdown: str, *, code_theme: Optional[str] = None,
                 theme_name: str = DEFAULT_THEME, **kwargs):
        super().__init__(markdown, **kwargs)
        self._theme_name = theme_name
    
    def __enter__(self):
        super().__enter__()
        # Inject theme_name into all ThemeAwareCodeBlock instances
        if hasattr(self, '_elements'):
            for elem in self._elements:
                if isinstance(elem, ThemeAwareCodeBlock):
                    elem._theme_name = self._theme_name
        return self


def _get_theme_accent() -> str:
    """Get the accent color from the current theme."""
    try:
        from ..config import get_config
        config = get_config()
        theme_name = getattr(config, "theme", DEFAULT_THEME)
        theme = get_theme_colors(theme_name)
        return theme["text_primary"]
    except Exception:
        return get_theme_colors(DEFAULT_THEME)["text_primary"]


def render_markdown(text: str, theme_name: str = DEFAULT_THEME, 
                    console: Optional[Console] = None) -> None:
    """Render markdown text to the terminal with theme-aware styling.
    
    Args:
        text: Markdown content to render
        theme_name: Theme name for color overrides
        console: Rich Console instance (creates default if None)
    """
    if console is None:
        console = Console()
    
    md = ThemeAwareMarkdown(text, theme_name=theme_name)
    console.print(md)


def render_markdown_panel(text: str, title: str = "", 
                          theme_name: str = DEFAULT_THEME,
                          border_style: Optional[str] = None,
                          console: Optional[Console] = None) -> None:
    """Render markdown text inside a themed panel.
    
    Args:
        text: Markdown content to render
        title: Panel title
        theme_name: Theme name for color overrides
        border_style: Override border color
        console: Rich Console instance (creates default if None)
    """
    if console is None:
        console = Console()
    
    theme = get_theme_colors(theme_name)
    border = border_style or theme["panel_border"]
    
    md = ThemeAwareMarkdown(text, theme_name=theme_name)
    
    panel = Panel(
        md,
        title=title,
        border_style=border,
        padding=(1, 2),
        box=ROUNDED,
    )
    
    console.print(panel)


def render_markdown_snippet(text: str, max_lines: int = 8,
                            theme_name: str = DEFAULT_THEME,
                            console: Optional[Console] = None) -> None:
    """Render a truncated markdown snippet (for search result previews).
    
    Args:
        text: Markdown content (will be truncated if too long)
        max_lines: Maximum number of lines to display
        theme_name: Theme name for color overrides
        console: Rich Console instance (creates default if None)
    """
    if console is None:
        console = Console()
    
    # Truncate text if too long
    lines = text.split("\n")
    if len(lines) > max_lines:
        text = "\n".join(lines[:max_lines]) + "\n..."
    
    md = ThemeAwareMarkdown(text, theme_name=theme_name)
    console.print(md)
