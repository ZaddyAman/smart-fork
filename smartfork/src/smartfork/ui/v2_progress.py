"""SmartFork v2 — Clean Pipeline Progress UI.

Replaces the v1 multi-bar mesh display with a cleaner single-bar + step
checklist approach optimized for the v2 indexing pipeline.

Uses the existing theme system (obsidian, phosphor, ember, etc.) for colors.

Design principles:
- Single main progress bar (no confusion about what 3 bars mean)
- Per-session step checklist (user sees exactly what's happening)
- Human-readable ETA ("~3 min" not "7208s")
- Visible unfilled blocks on ANY terminal/theme
- Running stats footer (chunks, errors, speed)
"""

from __future__ import annotations
import math
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Optional, List

from rich import box
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .progress import (
    THEMES, DEFAULT_THEME, get_theme_colors,
    get_animation_fps, mark_activity,
    SPINNER_FRAMES, _UNICODE, _BOX,
)


# ═══════════════════════════════════════════════════════════════════════════════
# BAR CHARACTERS (visible on all terminals/themes)
# ═══════════════════════════════════════════════════════════════════════════════

# Use ━ for filled and ░ for empty — both always visible
BAR_FILLED = "━" if _UNICODE else "="
BAR_EMPTY = "░" if _UNICODE else "-"
BAR_HEAD = "╸" if _UNICODE else ">"
CHECK_DONE = "✓" if _UNICODE else "+"
CHECK_ACTIVE = "⠋" if _UNICODE else "*"
CHECK_PENDING = "○" if _UNICODE else "."
BULLET = "●" if _UNICODE else "*"


# ═══════════════════════════════════════════════════════════════════════════════
# V2 STATS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class V2IndexStats:
    """Statistics for v2 indexing pipeline."""
    total_sessions: int = 0
    indexed_sessions: int = 0
    total_chunks: int = 0
    errors: int = 0
    
    # Current session info
    current_session_id: str = ""
    current_project: str = ""
    current_step: str = ""  # parse, store, embed
    
    # Step statuses for current session
    step_parse: str = "pending"   # pending, active, done, error
    step_store: str = "pending"
    step_embed: str = "pending"
    
    # Parsed metadata for current session (shown in UI)
    current_files_count: int = 0
    current_domains: List[str] = field(default_factory=list)
    current_languages: List[str] = field(default_factory=list)
    
    # Timing
    start_time: float = field(default_factory=time.time)
    session_start_time: float = 0.0
    done: bool = False
    skip_embeddings: bool = False
    embedding_provider: str = ""
    
    @property
    def overall_progress(self) -> float:
        if self.total_sessions == 0:
            return 0.0
        return min(1.0, self.indexed_sessions / self.total_sessions)
    
    @property
    def elapsed(self) -> float:
        return time.time() - self.start_time
    
    @property
    def avg_session_time(self) -> float:
        if self.indexed_sessions == 0:
            return 0.0
        return self.elapsed / self.indexed_sessions
    
    @property
    def eta_seconds(self) -> float:
        if self.indexed_sessions == 0:
            return 0.0
        remaining = self.total_sessions - self.indexed_sessions
        return self.avg_session_time * remaining


# ═══════════════════════════════════════════════════════════════════════════════
# HUMAN-READABLE TIME
# ═══════════════════════════════════════════════════════════════════════════════

def _human_time(seconds: float) -> str:
    """Convert seconds to human-readable time."""
    if seconds <= 0:
        return ""
    if seconds < 60:
        return f"~{int(seconds)}s"
    minutes = seconds / 60
    if minutes < 60:
        return f"~{int(minutes)} min"
    hours = minutes / 60
    return f"~{hours:.1f}h"


# ═══════════════════════════════════════════════════════════════════════════════
# PROGRESS BAR RENDERER
# ═══════════════════════════════════════════════════════════════════════════════

def _render_bar(progress: float, width: int, bar_color: str, empty_color: str,
                frame: int = 0) -> Text:
    """Render a single clean progress bar.
    
    Example: ━━━━━━━━━━━━━━━╸░░░░░░░░░░░░░  48%
    """
    text = Text()
    filled = int(progress * width)
    
    # Filled portion (solid color to prevent glitching)
    for i in range(filled):
        text.append(BAR_FILLED, style=bar_color)
    
    # Head (animated pulse at the frontier)
    if filled < width and progress > 0:
        pulse = math.sin(frame * 0.3) * 0.5 + 0.5
        style = f"bold {bar_color}" if pulse > 0.5 else bar_color
        text.append(BAR_HEAD, style=style)
        filled += 1
    
    # Empty portion — always visible
    for i in range(filled, width):
        text.append(BAR_EMPTY, style=f"{empty_color}")
    
    return text


# ═══════════════════════════════════════════════════════════════════════════════
# STEP CHECKLIST RENDERER
# ═══════════════════════════════════════════════════════════════════════════════

def _step_icon(status: str, frame: int, colors: dict) -> Text:
    """Render step status icon with color."""
    text = Text()
    if status == "done":
        text.append(CHECK_DONE, style=f"bold {colors['success']}")
    elif status == "active":
        spin = SPINNER_FRAMES[frame % len(SPINNER_FRAMES)]
        text.append(spin, style=f"bold {colors['accent']}")
    elif status == "error":
        text.append("✗", style=f"bold {colors['error']}")
    else:  # pending
        text.append(CHECK_PENDING, style=f"dim {colors['muted']}")
    return text


def _step_label(label: str, detail: str, status: str, colors: dict) -> Text:
    """Render step label with detail text."""
    text = Text()
    if status == "active":
        text.append(f" {label}", style=f"bold {colors['primary']}")
        if detail:
            text.append(f"  {detail}", style=f"dim {colors['muted']}")
    elif status == "done":
        text.append(f" {label}", style=f"{colors['success']}")
        if detail:
            text.append(f"  {detail}", style=f"dim {colors['muted']}")
    elif status == "error":
        text.append(f" {label}", style=f"dim {colors['error']}")
        if detail:
            text.append(f"  {detail}", style=f"dim {colors['error']}")
    else:
        text.append(f" {label}", style=f"dim {colors['muted']}")
    return text


# ═══════════════════════════════════════════════════════════════════════════════
# FRAME BUILDER
# ═══════════════════════════════════════════════════════════════════════════════

def _build_v2_frame(stats: V2IndexStats, frame: int, theme: dict) -> Panel:
    """Build the complete v2 progress UI frame.
    
    Layout:
    ╭─────────────────────────────────────────────────────────╮
    │  ⚡ SmartFork v2 Indexer                    12/100      │
    │                                                         │
    │  ━━━━━━━━━━━━━━━╸░░░░░░░░░░░░░░░  12%   ~8 min left    │
    │                                                         │
    │  ┌─ 019cb2a3-8c6c  ─  BharatLawAI                      │
    │  │  ✓ Parsed       12 files · python · auth, backend    │
    │  │  ⠋ Storing      SQLite ↓                             │
    │  │  ○ Embedding    waiting...                           │
    │  └──────────────────────────────────────                │
    │                                                         │
    │  📊 152 chunks  ·  0 errors  ·  3.2s/session            │
    ╰─────────────────────────────────────────────────────────╯
    """
    primary = theme["text_primary"]
    muted = theme["text_muted"]
    bars = theme["bars"]
    semantic = theme.get("semantic", {})
    done_color = theme["done_color"]
    border = done_color if stats.done else theme["panel_border"]
    spin_color = theme["spinner_color"]
    
    colors = {
        "primary": primary,
        "muted": muted,
        "success": semantic.get("success", done_color),
        "warning": semantic.get("warning", "#FCD34D"),
        "error": semantic.get("error", "#F87171"),
        "accent": semantic.get("accent", primary),
        "bar": bars[0]["color"],
        "bar_glow": bars[0].get("glow_color", muted),
    }
    
    outer = Table.grid(expand=True, padding=(0, 0))
    outer.add_column()
    
    # ── HEADER ────────────────────────────────────────────────
    header = Table.grid(expand=True)
    header.add_column(ratio=2)
    header.add_column(ratio=1)
    
    left = Text()
    if stats.done:
        left.append("  ✓ ", style=f"bold {done_color}")
        left.append("Index Complete", style=f"bold {done_color}")
    else:
        spin = SPINNER_FRAMES[frame % len(SPINNER_FRAMES)]
        left.append(f"  {spin} ", style=f"bold {spin_color}")
        left.append("SmartFork v2 Indexer", style=f"bold {primary}")
    
    right = Text(justify="right")
    right.append(str(stats.indexed_sessions), style=f"bold {colors['bar']}")
    right.append(f"/{stats.total_sessions}  ", style=f"dim {muted}")
    
    header.add_row(left, right)
    outer.add_row(header)
    outer.add_row(Text(""))
    
    # ── MAIN PROGRESS BAR ────────────────────────────────────
    bar_row = Table.grid(expand=True)
    bar_row.add_column(width=2)
    bar_row.add_column(ratio=1)
    bar_row.add_column(width=22)
    
    bar = _render_bar(
        stats.overall_progress, width=40,
        bar_color=colors["bar"],
        empty_color=f"dim {muted}",
        frame=frame,
    )
    
    bar_info = Text(justify="right")
    pct = int(stats.overall_progress * 100)
    bar_info.append(f"  {pct:>3}%", style=f"bold {colors['bar']}")
    
    if stats.done:
        bar_info.append(f"   {stats.elapsed:.1f}s total", style=f"dim {muted}")
    elif stats.eta_seconds > 0:
        bar_info.append(f"   {_human_time(stats.eta_seconds)} left", style=f"dim {muted}")
    
    bar_row.add_row(Text("  "), bar, bar_info)
    outer.add_row(bar_row)
    outer.add_row(Text(""))
    
    # ── SESSION STEP CHECKLIST ────────────────────────────────
    if not stats.done and stats.current_session_id:
        # Session header line
        session_hdr = Text()
        session_hdr.append("  ┌─ ", style=f"dim {muted}")
        sid_short = stats.current_session_id[:16]
        session_hdr.append(sid_short, style=f"dim {muted}")
        if stats.current_project and stats.current_project != "unknown":
            session_hdr.append("  ─  ", style=f"dim {muted}")
            session_hdr.append(stats.current_project, style=f"bold {primary}")
        outer.add_row(session_hdr)
        
        # Build step details
        parse_detail = ""
        if stats.step_parse == "done":
            parts = []
            if stats.current_files_count > 0:
                parts.append(f"{stats.current_files_count} files")
            if stats.current_languages:
                parts.append(" · ".join(stats.current_languages[:2]))
            if stats.current_domains:
                parts.append(", ".join(stats.current_domains[:3]))
            parse_detail = " · ".join(parts) if parts else ""
        
        store_detail = ""
        if stats.step_store == "done":
            store_detail = "SQLite ✓"
        elif stats.step_store == "active":
            store_detail = "SQLite ↓"
        
        embed_detail = ""
        if stats.skip_embeddings:
            embed_detail = "skipped"
        elif stats.step_embed == "done":
            embed_detail = f"{stats.embedding_provider} ✓"
        elif stats.step_embed == "active":
            embed_detail = f"→ {stats.embedding_provider}..."
        elif stats.step_embed == "pending":
            embed_detail = "waiting..."
        
        # Render steps
        steps = [
            ("Parsed", parse_detail, stats.step_parse),
            ("Stored", store_detail, stats.step_store),
            ("Embed ", embed_detail, stats.step_embed),
        ]
        
        for label, detail, status in steps:
            step_row = Text()
            step_row.append("  │  ", style=f"dim {muted}")
            step_row.append_text(_step_icon(status, frame, colors))
            step_row.append_text(_step_label(label, detail, status, colors))
            outer.add_row(step_row)
        
        # Close bracket
        close = Text()
        close.append("  └", style=f"dim {muted}")
        close.append("─" * 38, style=f"dim {muted}")
        outer.add_row(close)
    
    outer.add_row(Text(""))
    
    # ── FOOTER STATS ──────────────────────────────────────────
    footer = Text()
    footer.append("  ")
    
    if stats.done:
        footer.append(f"{BULLET} ", style=f"bold {done_color}")
        footer.append(f"{stats.total_chunks:,} chunks", style=f"bold {colors['bar']}")
        footer.append("  ·  ", style=f"dim {muted}")
        footer.append(f"{stats.errors} errors", 
                      style=f"bold {colors['error']}" if stats.errors > 0 else f"dim {muted}")
        footer.append("  ·  ", style=f"dim {muted}")
        footer.append(f"{stats.elapsed:.1f}s total", style=f"dim {muted}")
    else:
        footer.append(f"{stats.total_chunks:,}", style=f"bold {bars[1]['color']}")
        footer.append(" chunks", style=f"dim {muted}")
        footer.append("  ·  ", style=f"dim {muted}")
        if stats.errors > 0:
            footer.append(f"{stats.errors} errors", style=f"bold {colors['error']}")
        else:
            footer.append("0 errors", style=f"dim {muted}")
        footer.append("  ·  ", style=f"dim {muted}")
        if stats.avg_session_time > 0:
            footer.append(f"{stats.avg_session_time:.1f}s/session", style=f"dim {muted}")
    
    outer.add_row(footer)
    
    return Panel(outer, border_style=border, padding=(1, 2), box=_BOX)


# ═══════════════════════════════════════════════════════════════════════════════
# V2 INDEX PROGRESS — MAIN CLASS
# ═══════════════════════════════════════════════════════════════════════════════


class V2IndexProgress:
    """Clean progress UI for v2 indexing pipeline.
    
    Usage:
        with V2IndexProgress(
            total_sessions=100,
            theme_name="obsidian",
            embedding_provider="qwen3-embedding:0.6b",
        ) as prog:
            for session_path in sessions:
                prog.start_session(session_path.name)
                
                # Parse
                prog.step_active("parse")
                doc = parser.parse(session_path)
                prog.step_done("parse", files=len(doc.files), 
                               domains=doc.domains, languages=doc.languages)
                prog.set_project(doc.project_name)
                
                # Store
                prog.step_active("store")
                store.upsert(doc)
                prog.step_done("store")
                
                # Embed
                prog.step_active("embed")
                vector_index.index(doc)
                prog.step_done("embed", chunks=count)
                
                prog.advance()
            prog.finish()
    """
    
    def __init__(self, total_sessions: int, theme_name: str = DEFAULT_THEME,
                 console: Optional[Console] = None,
                 embedding_provider: str = "",
                 skip_embeddings: bool = False,
                 animation_fps: Optional[int] = None,
                 disable_animation: bool = False):
        if theme_name not in THEMES:
            theme_name = DEFAULT_THEME
        self._theme = THEMES[theme_name]
        self._console = console or Console()
        self._stats = V2IndexStats(
            total_sessions=total_sessions,
            skip_embeddings=skip_embeddings,
            embedding_provider=embedding_provider,
        )
        self._frame = 0
        self._running = False
        self._live: Optional[Live] = None
        self._thread: Optional[threading.Thread] = None
        self._animation_fps = animation_fps or get_animation_fps()
        self._disable_animation = disable_animation
    
    def __enter__(self) -> "V2IndexProgress":
        refresh_rate = 2 if self._disable_animation else self._animation_fps
        self._live = Live(
            _build_v2_frame(self._stats, self._frame, self._theme),
            console=self._console,
            refresh_per_second=refresh_rate,
            transient=False,
        )
        self._live.__enter__()
        self._running = True
        if not self._disable_animation:
            self._thread = threading.Thread(
                target=self._loop, daemon=True, name="sf-v2-anim"
            )
            self._thread.start()
        return self
    
    def __exit__(self, *args):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._live:
            self._live.update(_build_v2_frame(self._stats, self._frame, self._theme))
            self._live.__exit__(*args)
        return False
    
    def _loop(self):
        """Animation loop."""
        while self._running:
            fps = get_animation_fps()
            interval = 1.0 / fps
            self._frame += 1
            if self._live:
                try:
                    self._live.update(
                        _build_v2_frame(self._stats, self._frame, self._theme)
                    )
                except Exception:
                    pass
            time.sleep(interval)
    
    # ── MUTATORS ──────────────────────────────────────────────
    
    def start_session(self, session_id: str) -> None:
        """Call at start of each session. Resets all steps."""
        mark_activity()
        self._stats.current_session_id = session_id
        self._stats.current_project = ""
        self._stats.step_parse = "pending"
        self._stats.step_store = "pending"
        self._stats.step_embed = "pending" if not self._stats.skip_embeddings else "pending"
        self._stats.current_files_count = 0
        self._stats.current_domains = []
        self._stats.current_languages = []
        self._stats.session_start_time = time.time()
    
    def set_project(self, name: str) -> None:
        """Set project name for current session."""
        if name and name != "unknown":
            self._stats.current_project = name
    
    def step_active(self, step: str) -> None:
        """Mark a step as active (spinner)."""
        mark_activity()
        if step == "parse":
            self._stats.step_parse = "active"
        elif step == "store":
            self._stats.step_store = "active"
        elif step == "embed":
            self._stats.step_embed = "active"
    
    def step_done(self, step: str, files: int = 0,
                  domains: list = None, languages: list = None,
                  chunks: int = 0) -> None:
        """Mark a step as done with optional metadata."""
        mark_activity()
        if step == "parse":
            self._stats.step_parse = "done"
            self._stats.current_files_count = files
            self._stats.current_domains = domains or []
            self._stats.current_languages = languages or []
        elif step == "store":
            self._stats.step_store = "done"
        elif step == "embed":
            self._stats.step_embed = "done"
            if chunks > 0:
                self._stats.total_chunks += chunks
    
    def step_error(self, step: str) -> None:
        """Mark a step as errored."""
        mark_activity()
        if step == "parse":
            self._stats.step_parse = "error"
        elif step == "store":
            self._stats.step_store = "error"
        elif step == "embed":
            self._stats.step_embed = "error"
    
    def step_skip(self, step: str) -> None:
        """Mark a step as skipped."""
        mark_activity()
        if step == "embed":
            self._stats.step_embed = "done"
    
    def advance(self) -> None:
        """Advance session counter by 1."""
        mark_activity()
        self._stats.indexed_sessions = min(
            self._stats.indexed_sessions + 1, self._stats.total_sessions
        )
    
    def add_error(self) -> None:
        """Increment error counter."""
        mark_activity()
        self._stats.errors += 1
    
    def add_chunks(self, n: int) -> None:
        """Add chunks to running total."""
        mark_activity()
        self._stats.total_chunks += n
    
    def finish(self) -> None:
        """Mark complete. Call inside the with block."""
        self._stats.indexed_sessions = self._stats.total_sessions
        self._stats.done = True
        time.sleep(0.5)
    
    @property
    def stats(self) -> V2IndexStats:
        """Access stats for summary display."""
        return self._stats
