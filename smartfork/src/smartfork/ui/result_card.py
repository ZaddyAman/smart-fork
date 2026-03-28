"""Rich result card renderer for CLI (v2).

Displays search results as compact, informative 5-line cards.
Zero LLM calls — all data pulled from the existing index.

Format:
    📁 Project — Task Title
    🕐 3 days ago (47 min) | ⚡ 94% match
    "Snippet from best matching reasoning/summary chunk..."
    Why: Contains auth decision | Files: auth.py, models.py
    [1] Fork  [2] Preview  [3] Skip
"""

import time
from datetime import datetime
from typing import List

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table

from ..database.models import ResultCard, SessionDocument, VectorResult


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
    # Format relative time
    relative_time = _format_relative_time(doc.session_start)
    
    # Truncate task to 50 chars
    task_short = doc.task_raw[:50].strip() if doc.task_raw else "untitled session"
    if len(doc.task_raw) > 50:
        task_short += "..."
    
    # Truncate snippet to 120 chars
    if snippet and len(snippet) > 120:
        snippet = snippet[:117] + "..."
    
    # Build "why matched" explanation based on available document types
    if not why_matched:
        why_parts = []
        # If snippet came from reasoning, show reasoning match
        if doc.reasoning_docs and snippet:
            why_parts.append("Contains reasoning/decisions")
        # If no reasoning but has summary, show summary match
        elif doc.summary_doc:
            why_parts.append("Summary matches")
        # Otherwise show task match
        else:
            why_parts.append("Task description matches")
        
        if doc.domains:
            why_parts.append(f"Domains: {', '.join(doc.domains[:3])}")
        
        why_matched = " | ".join(why_parts) if why_parts else "Relevant match"
    
    # Get top 3 changed file basenames
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


def render_result_cards(cards: List[ResultCard], console: Console = None) -> None:
    """Render result cards to the terminal using Rich.
    
    Args:
        cards: List of ResultCard objects to display
        console: Optional Rich Console (creates default if none)
    """
    if console is None:
        console = Console()
    
    if not cards:
        console.print("[dim]No results found.[/dim]")
        return
    
    for i, card in enumerate(cards, start=1):
        _render_single_card(card, i, console)
        console.print()  # Spacer between cards


def _render_single_card(card: ResultCard, index: int, console: Console) -> None:
    """Render a single result card as a Rich Panel.
    
    Output format:
        📁 BharatLawAI — implement JWT authentication for...
        🕐 3 days ago (47 min) | ⚡ 94% match
        "Decided to use JWT for stateless auth because..."
        Why: Contains reasoning/decisions | Files: auth.py, models.py
        [1] Fork  [2] Preview  [3] Skip
    """
    # Line 1: Project — Task
    line1 = Text()
    line1.append("📁 ", style="bold")
    line1.append(card.project_name, style="bold cyan")
    line1.append(" — ", style="dim")
    line1.append(card.task_short, style="white")
    
    # Line 2: Time and score
    score_pct = int(card.match_score * 100)
    score_style = "bold green" if score_pct >= 80 else "yellow" if score_pct >= 50 else "red"
    
    line2 = Text()
    line2.append("🕐 ", style="")
    line2.append(card.relative_time, style="dim")
    if card.duration_minutes > 0:
        line2.append(f" ({int(card.duration_minutes)} min)", style="dim")
    line2.append(" | ", style="dim")
    line2.append("⚡ ", style="")
    line2.append(f"{score_pct}% match", style=score_style)
    
    # Line 3: Snippet
    line3 = Text()
    line3.append('"', style="dim")
    line3.append(card.snippet, style="italic")
    line3.append('"', style="dim")
    
    # Line 4: Why matched + files
    line4 = Text()
    line4.append("Why: ", style="bold dim")
    line4.append(card.why_matched, style="dim")
    if card.files_changed:
        line4.append(" | Files: ", style="bold dim")
        line4.append(", ".join(card.files_changed), style="dim cyan")
    
    # Line 5: Session ID + fork command
    line5 = Text()
    sid_short = card.session_id[:16] if len(card.session_id) > 16 else card.session_id
    line5.append(f"🔑 ", style="dim")
    line5.append(card.session_id, style="dim cyan")
    line5.append("  →  ", style="dim")
    line5.append(f"smartfork fork-v2 {sid_short}", style="bold green")
    
    # Combine into panel
    content = Text()
    content.append_text(line1)
    content.append("\n")
    content.append_text(line2)
    content.append("\n")
    content.append_text(line3)
    content.append("\n")
    content.append_text(line4)
    content.append("\n")
    content.append_text(line5)
    
    panel = Panel(
        content,
        border_style="dim",
        padding=(0, 1),
    )
    
    console.print(panel)


def _format_relative_time(timestamp_ms: int) -> str:
    """Format a timestamp as relative time (e.g., '3 days ago').
    
    Args:
        timestamp_ms: Unix timestamp in milliseconds
    
    Returns:
        Human-readable relative time string
    """
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
