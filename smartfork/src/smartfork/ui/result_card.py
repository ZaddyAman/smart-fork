"""Rich result card renderer for CLI (v2).

Enhanced with syntax highlighting, visual score bars, named panel borders,
and theme-aware colors — inspired by claw-code's tool call visualization.

Format:
    ╭─ 📁 ProjectName — Task Title ──────────────────────╮
    │ ⚡ ████████████████████░░░░░░ 64% match            │
    │ 🕐 3 days ago (47 min) | 📁 auth.py, models.py     │
    │ "Decided to use JWT for stateless auth because..." │
    │ 🏷️ backend · auth · testing                        │
    │ → smartfork fork abc123                            │
    ╰────────────────────────────────────────────────────╯
"""

import time
from datetime import datetime
from typing import List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich import box

from ..database.models import ResultCard, SessionDocument, VectorResult
from .syntax_highlight import highlight_code, highlight_snippet, detect_language_from_content
from .progress import DEFAULT_THEME, get_theme_colors


def build_result_card(doc: SessionDocument, match_score: float,
                       snippet: str = "", why_matched: str = "",
                       matched_doc_type: str = "task_doc") -> ResultCard:
    """Build a ResultCard from a SessionDocument and search results.
    
    Args:
        doc: SessionDocument with session metadata
        match_score: Combined search score (0.0-1.0)
        snippet: Best matching text snippet
        why_matched: Rule-based explanation of why this matched
        matched_doc_type: Which document type had the best match
    
    Returns:
        ResultCard ready for rendering
    """
    relative_time = _format_relative_time(doc.session_start)
    
    task_short = doc.task_raw[:50].strip() if doc.task_raw else "untitled session"
    if len(doc.task_raw) > 50:
        task_short += "..."
    
    if snippet and len(snippet) > 120:
        snippet = snippet[:117] + "..."
    
    if not why_matched:
        why_parts = []
        if doc.reasoning_docs and snippet:
            why_parts.append("Contains reasoning/decisions")
        elif doc.summary_doc:
            why_parts.append("Summary matches")
        else:
            why_parts.append("Task description matches")
        
        if doc.domains:
            why_parts.append(f"Domains: {', '.join(doc.domains[:3])}")
        
        why_matched = " | ".join(why_parts) if why_parts else "Relevant match"
    
    files_changed = [f.split("/")[-1] for f in doc.files_edited[:3]]
    
    return ResultCard(
        session_id=doc.session_id,
        project_name=doc.project_name,
        task_short=task_short,
        relative_time=relative_time,
        duration_minutes=doc.duration_minutes,
        match_score=match_score,
        snippet=snippet or task_short,
        why_matched=why_matched,
        files_changed=files_changed,
        matched_doc_type=matched_doc_type,
    )


def render_result_cards(cards: List[ResultCard], console: Optional[Console] = None,
                        theme_name: str = DEFAULT_THEME,
                        command_prefix: str = "smartfork fork") -> None:
    """Render result cards to the terminal using Rich.
    
    Args:
        cards: List of ResultCard objects to display
        console: Optional Rich Console (creates default if none)
        theme_name: Theme name for color overrides
        command_prefix: Command shown on each card (default: "smartfork fork")
    """
    if console is None:
        console = Console()
    
    if not cards:
        console.print("[dim]No results found.[/dim]")
        return
    
    for i, card in enumerate(cards, start=1):
        _render_single_card(card, i, console, theme_name, command_prefix)
        console.print()


def _render_single_card(card: ResultCard, index: int, console: Console,
                        theme_name: str = DEFAULT_THEME,
                        command_prefix: str = "smartfork fork") -> None:
    """Render a single result card as an enhanced Rich Panel."""
    theme = get_theme_colors(theme_name)
    
    score_pct = int(card.match_score * 100)
    score_style = _score_style(score_pct, theme)
    border_color = _border_color(score_pct, theme)
    
    # Line 1: Project — Task (panel title)
    title_text = Text()
    title_text.append(f"[{index}] ", style=f"bold {theme['semantic']['accent']}")
    title_text.append(card.project_name, style=f"bold {theme['text_primary']}")
    title_text.append(" — ", style="dim")
    title_text.append(card.task_short, style="default")
    
    # Line 2: Visual score bar
    score_bar = _build_score_bar(score_pct, theme)
    
    # Line 3: Time + files (deduplicated)
    time_line = Text()
    time_line.append("🕐 ", style="")
    time_line.append(card.relative_time, style=f"dim {theme['text_muted']}")
    if card.duration_minutes > 0:
        time_line.append(f" ({int(card.duration_minutes)} min)", style=f"dim {theme['text_muted']}")
    if card.files_changed:
        unique_files = list(dict.fromkeys(card.files_changed))
        shown = unique_files[:4]
        truncated = ", ".join(shown)
        if len(unique_files) > 4:
            truncated += f" +{len(unique_files) - 4} more"
        time_line.append(" | 📁 ", style=f"dim {theme['text_muted']}")
        time_line.append(truncated, style=f"dim {theme['bars'][0]['color']}")
    
    # Line 4: Snippet
    snippet_line = Text()
    snippet_line.append('"', style="dim")
    snippet_text = highlight_snippet(card.snippet, max_len=150)
    snippet_line.append_text(snippet_text)
    snippet_line.append('"', style="dim")
    
    # Line 5: Tags (domains)
    tags_line = None
    why = card.why_matched
    if "Domains:" in why:
        domains_part = why.split("Domains:")[1].strip()
        tags_line = Text()
        tags_line.append("🏷️ ", style=f"dim {theme['text_muted']}")
        tags = [d.strip() for d in domains_part.split(",")]
        for i, tag in enumerate(tags):
            if i > 0:
                tags_line.append(" · ", style=f"dim {theme['text_muted']}")
            tags_line.append(tag, style=f"dim {theme['bars'][2]['color']}")
    
    # Line 6: Supersession status
    supersession_line = None
    if "🔄" in card.why_matched:
        supersession_line = Text()
        supersession_line.append("🔄 ", style=f"bold {theme['semantic']['success']}")
        supersession_line.append("This session fixes earlier attempts", style=theme['semantic']['success'])
    elif "⚠️" in card.why_matched or "superseded" in card.why_matched.lower():
        supersession_line = Text()
        supersession_line.append("⚠️ ", style=f"bold {theme['semantic']['warning']}")
        supersession_line.append("This session has been superseded", style=theme['semantic']['warning'])
    
    # Line 7: Fork command (short session ID)
    sid_short = card.session_id[:8]
    sid_line = Text()
    sid_line.append("→ ", style=f"bold {theme['bars'][0]['color']}")
    sid_line.append(f"{command_prefix} ", style=f"dim {theme['text_muted']}")
    sid_line.append(sid_short, style=f"bold {theme['bars'][0]['color']}")
    
    # Combine into panel content
    content = Text()
    content.append_text(score_bar)
    content.append("\n")
    content.append_text(time_line)
    content.append("\n")
    content.append_text(snippet_line)
    if tags_line:
        content.append("\n")
        content.append_text(tags_line)
    if supersession_line:
        content.append("\n")
        content.append_text(supersession_line)
    content.append("\n")
    content.append_text(sid_line)
    
    panel = Panel(
        content,
        title=title_text,
        border_style=border_color,
        padding=(0, 1),
        box=box.ROUNDED,
    )
    
    console.print(panel)


def _build_score_bar(score: int, theme: dict) -> Text:
    """Build a visual score bar with percentage.
    
    Example: ⚡ ████████████████████░░░░░░ 64% match
    """
    bar_width = 25
    filled = int(score * bar_width / 100)
    empty = bar_width - filled
    
    # Color based on score
    if score >= 80:
        fill_color = theme.get("semantic", {}).get("success", theme["text_primary"])
    elif score >= 50:
        fill_color = theme.get("semantic", {}).get("warning", theme["text_primary"])
    else:
        fill_color = theme.get("semantic", {}).get("error", theme["text_primary"])
    
    result = Text()
    result.append("⚡ ", style="")
    result.append("█" * filled, style=f"bold {fill_color}")
    result.append("░" * empty, style=f"dim {fill_color}")
    result.append(f"  {score}% match", style=f"bold {fill_color}")
    
    return result


def _score_style(score: int, theme: dict) -> str:
    """Get Rich style string for a score percentage."""
    if score >= 80:
        return f"bold {theme.get('semantic', {}).get('success', theme['text_primary'])}"
    elif score >= 50:
        return f"bold {theme.get('semantic', {}).get('warning', theme['text_primary'])}"
    else:
        return f"bold {theme.get('semantic', {}).get('error', theme['text_primary'])}"


def _border_color(score: int, theme: dict) -> str:
    """Get border color based on score."""
    if score >= 80:
        return theme.get("semantic", {}).get("success", theme["text_primary"])
    elif score >= 50:
        return theme.get("semantic", {}).get("warning", theme["text_primary"])
    else:
        return theme.get("semantic", {}).get("error", theme["text_primary"])


def _format_relative_time(timestamp_ms: int) -> str:
    """Format a timestamp as relative time (e.g., '3 days ago')."""
    if not timestamp_ms:
        return "unknown time"
    
    now = time.time()
    diff_seconds = now - (timestamp_ms / 1000)
    
    if diff_seconds < 0:
        return "just now"
    
    if diff_seconds < 60:
        return "just now"
    elif diff_seconds < 3600:
        minutes = int(diff_seconds / 60)
        return f"{minutes} min ago"
    elif diff_seconds < 86400:
        hours = int(diff_seconds / 3600)
        return f"{hours} hour{'s' if hours > 1 else ''} ago"
    elif diff_seconds < 604800:
        days = int(diff_seconds / 86400)
        return f"{days} day{'s' if days > 1 else ''} ago"
    elif diff_seconds < 2592000:
        weeks = int(diff_seconds / 604800)
        return f"{weeks} week{'s' if weeks > 1 else ''} ago"
    else:
        months = int(diff_seconds / 2592000)
        return f"{months} month{'s' if months > 1 else ''} ago"
