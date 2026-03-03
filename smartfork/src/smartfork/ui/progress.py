"""SmartFork — Animated Mesh Progress Bar with Theme System."""
from __future__ import annotations
import math, sys, threading, time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from rich import box
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# Export animation control functions
__all__ = [
    'SmartForkProgress', 'IndexingStats',
    'display_discovery_phase', 'display_completion_summary',
    'THEMES', 'DEFAULT_THEME',
    'get_theme_colors', 'get_semantic_color',
    'set_animation_fps', 'get_animation_fps', 
    'set_adaptive_fps', 'mark_activity',
    'AnimatedProgressDisplay', 'IndexingProgressDisplay'
]


# ═══════════════════════════════════════════════════════════════════════════════
# THEME SYSTEM — Six carefully crafted color palettes
# ═══════════════════════════════════════════════════════════════════════════════

THEMES = {
    "phosphor": {
        "name": "Phosphor",
        "desc": "Classic CRT green — timeless hacker aesthetic",
        "panel_border": "#1A3A1A",
        "bars": [
            {"label": "Sessions",   "color": "#4ADE80", "glow_color": "#16A34A"},
            {"label": "Embedding",  "color": "#22C55E", "glow_color": "#15803D"},
            {"label": "BM25 Index", "color": "#86EFAC", "glow_color": "#4ADE80"},
        ],
        "text_primary": "#4ADE80",
        "text_muted":   "#1F4D1F",
        "text_dim":     "#0F2A0F",
        "spinner_color":"#4ADE80",
        "done_color":   "#86EFAC",
        "semantic": {
            "success": "#4ADE80",
            "warning": "#FBBF24",
            "error": "#F87171",
            "info": "#4ADE80",
            "accent": "#86EFAC",
        },
    },
    "obsidian": {
        "name": "Obsidian",
        "desc": "Cold steel blue-grey — serious infrastructure tooling",
        "panel_border": "#1E2D3D",
        "bars": [
            {"label": "Sessions",   "color": "#64748B", "glow_color": "#334155"},
            {"label": "Embedding",  "color": "#94A3B8", "glow_color": "#475569"},
            {"label": "BM25 Index", "color": "#CBD5E1", "glow_color": "#64748B"},
        ],
        "text_primary": "#CBD5E1",
        "text_muted":   "#334155",
        "text_dim":     "#1E2D3D",
        "spinner_color":"#94A3B8",
        "done_color":   "#CBD5E1",
        "semantic": {
            "success": "#94A3B8",
            "warning": "#FCD34D",
            "error": "#FCA5A5",
            "info": "#94A3B8",
            "accent": "#CBD5E1",
        },
    },
    "ember": {
        "name": "Ember",
        "desc": "Warm amber — focused, unusual, memorable",
        "panel_border": "#2A1F0E",
        "bars": [
            {"label": "Sessions",   "color": "#D97706", "glow_color": "#92400E"},
            {"label": "Embedding",  "color": "#F59E0B", "glow_color": "#B45309"},
            {"label": "BM25 Index", "color": "#FCD34D", "glow_color": "#D97706"},
        ],
        "text_primary": "#FCD34D",
        "text_muted":   "#78350F",
        "text_dim":     "#3C1A00",
        "spinner_color":"#F59E0B",
        "done_color":   "#FCD34D",
        "semantic": {
            "success": "#F59E0B",
            "warning": "#FCD34D",
            "error": "#EF4444",
            "info": "#F59E0B",
            "accent": "#FCD34D",
        },
    },
    "arctic": {
        "name": "Arctic",
        "desc": "Ice-cold blue — clinical precision",
        "panel_border": "#0F1D35",
        "bars": [
            {"label": "Sessions",   "color": "#38BDF8", "glow_color": "#0369A1"},
            {"label": "Embedding",  "color": "#7DD3FC", "glow_color": "#0284C7"},
            {"label": "BM25 Index", "color": "#BAE6FD", "glow_color": "#38BDF8"},
        ],
        "text_primary": "#BAE6FD",
        "text_muted":   "#0C2D4A",
        "text_dim":     "#071524",
        "spinner_color":"#38BDF8",
        "done_color":   "#BAE6FD",
        "semantic": {
            "success": "#38BDF8",
            "warning": "#7DD3FC",
            "error": "#F87171",
            "info": "#38BDF8",
            "accent": "#BAE6FD",
        },
    },
    "iron": {
        "name": "Iron",
        "desc": "Muted violet — high-end tooling depth",
        "panel_border": "#1C1830",
        "bars": [
            {"label": "Sessions",   "color": "#6D6494", "glow_color": "#3B3465"},
            {"label": "Embedding",  "color": "#9B8EC4", "glow_color": "#5B4D94"},
            {"label": "BM25 Index", "color": "#C4BAE8", "glow_color": "#7B6EB4"},
        ],
        "text_primary": "#C4BAE8",
        "text_muted":   "#2E2850",
        "text_dim":     "#18152A",
        "spinner_color":"#9B8EC4",
        "done_color":   "#C4BAE8",
        "semantic": {
            "success": "#9B8EC4",
            "warning": "#C4BAE8",
            "error": "#F87171",
            "info": "#9B8EC4",
            "accent": "#C4BAE8",
        },
    },
    "tungsten": {
        "name": "Tungsten",
        "desc": "Pure greyscale — maximum contrast, zero distraction",
        "panel_border": "#222222",
        "bars": [
            {"label": "Sessions",   "color": "#737373", "glow_color": "#404040"},
            {"label": "Embedding",  "color": "#A3A3A3", "glow_color": "#525252"},
            {"label": "BM25 Index", "color": "#D4D4D4", "glow_color": "#737373"},
        ],
        "text_primary": "#D4D4D4",
        "text_muted":   "#2A2A2A",
        "text_dim":     "#141414",
        "spinner_color":"#A3A3A3",
        "done_color":   "#D4D4D4",
        "semantic": {
            "success": "#A3A3A3",
            "warning": "#D4D4D4",
            "error": "#737373",
            "info": "#A3A3A3",
            "accent": "#D4D4D4",
        },
    },
}

DEFAULT_THEME = "obsidian"


def get_theme_colors(theme_name: str = DEFAULT_THEME) -> dict:
    """Get full color palette for a theme.
    
    Args:
        theme_name: Theme identifier (phosphor, obsidian, ember, arctic, iron, tungsten)
        
    Returns:
        Dictionary containing all theme colors
    """
    return THEMES.get(theme_name, THEMES[DEFAULT_THEME])


def get_semantic_color(theme_name: str, semantic_type: str) -> str:
    """Get semantic color from theme.
    
    Args:
        theme_name: Theme identifier
        semantic_type: One of: success, warning, error, info, accent
        
    Returns:
        Hex color code for the semantic type
    """
    theme = THEMES.get(theme_name, THEMES[DEFAULT_THEME])
    semantic = theme.get("semantic", {})
    return semantic.get(semantic_type, theme["text_primary"])


# ═══════════════════════════════════════════════════════════════════════════════
# ANIMATION CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

# Default FPS (can be overridden via config)
DEFAULT_ANIMATION_FPS = 10  # Reduced from 18 for lower CPU usage
MIN_ANIMATION_FPS = 5       # Minimum FPS for lite mode
ADAPTIVE_FPS_THRESHOLD = 30  # Seconds of inactivity before reducing FPS

ANIMATION_FPS        = DEFAULT_ANIMATION_FPS
MESH_ROWS            = 2
MESH_CELLS_PER_ROW   = 52
WAVE_SPEED_H         = 3.5
WAVE_SPEED_V         = 2.2
WAVE_WEIGHT_H        = 0.45
WAVE_WEIGHT_V        = 0.55
FRONTIER_BLINK_SPEED = 0.22

# Global animation settings (can be modified at runtime)
_animation_fps = DEFAULT_ANIMATION_FPS
_adaptive_fps_enabled = True
_last_activity_time = time.time()


def set_animation_fps(fps: int) -> None:
    """Set the global animation FPS."""
    global _animation_fps
    _animation_fps = max(MIN_ANIMATION_FPS, min(30, fps))


def get_animation_fps() -> int:
    """Get current animation FPS, considering adaptive settings."""
    global _animation_fps, _adaptive_fps_enabled, _last_activity_time
    
    if not _adaptive_fps_enabled:
        return _animation_fps
    
    # Check for inactivity - reduce FPS if user hasn't interacted recently
    inactive_seconds = time.time() - _last_activity_time
    if inactive_seconds > ADAPTIVE_FPS_THRESHOLD:
        # Gradually reduce FPS based on inactivity (max 50% reduction)
        reduction = min(0.5, inactive_seconds / 300)  # Max reduction after 5 minutes
        return max(MIN_ANIMATION_FPS, int(_animation_fps * (1 - reduction)))
    
    return _animation_fps


def mark_activity() -> None:
    """Mark that user activity has occurred (resets adaptive FPS)."""
    global _last_activity_time
    _last_activity_time = time.time()


def set_adaptive_fps(enabled: bool) -> None:
    """Enable or disable adaptive FPS."""
    global _adaptive_fps_enabled
    _adaptive_fps_enabled = enabled


# ═══════════════════════════════════════════════════════════════════════════════
# WINDOWS UNICODE FALLBACKS
# ═══════════════════════════════════════════════════════════════════════════════

def _unicode_ok() -> bool:
    """Check if terminal supports Unicode characters."""
    if sys.platform == "win32":
        try:
            sys.stdout.buffer.write("▪".encode("utf-8"))
            sys.stdout.buffer.flush()
            return True
        except Exception:
            return False
    return True

_UNICODE = _unicode_ok()
MESH_CELL_CHAR  = "▪" if _UNICODE else "#"
MESH_EMPTY_CHAR = "▫" if _UNICODE else "."
SPINNER_FRAMES  = (["⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧","⠇","⠏"]
                   if _UNICODE else ["|","/","-","\\"])
_BOX = box.ROUNDED if _UNICODE else box.ASCII


# ═══════════════════════════════════════════════════════════════════════════════
# INDEXING STATS DATACLASS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class IndexingStats:
    """Statistics for indexing operation."""
    total_sessions:       int   = 0
    indexed_sessions:     int   = 0
    total_chunks:         int   = 0
    errors:               int   = 0
    current_session_name: str   = ""
    current_session_title:str   = ""
    phase_label:          str   = "Parsing"
    phase_progress:       float = 0.0
    bm25_progress:        float = 0.0
    start_time:           float = field(default_factory=time.time)
    done:                 bool  = False

    @property
    def overall_progress(self) -> float:
        if self.total_sessions == 0:
            return 0.0
        return min(1.0, self.indexed_sessions / self.total_sessions)

    @property
    def elapsed(self) -> float:
        return time.time() - self.start_time

    @property
    def eta_seconds(self) -> float:
        p = self.overall_progress
        if p <= 0.01:
            return 0.0
        return (self.elapsed / p) * (1.0 - p)


# ═══════════════════════════════════════════════════════════════════════════════
# MESH RENDERER — THE CORE WAVE ANIMATION
# ═══════════════════════════════════════════════════════════════════════════════

def _render_mesh_bar(
    progress:   float,
    frame:      int,
    bar_color:  str,
    dim_color:  str,
    bar_offset: int = 0,
    rows:       int = MESH_ROWS,
    cols:       int = MESH_CELLS_PER_ROW,
) -> Text:
    """
    Renders one bar as a ROWS x COLS grid of small square characters.

    VISUAL EXAMPLE (2 rows, partial fill):
        ▪ ▪ ▪ ▪ ▪ ▪ ▪ ▪ ▫ ▫ ▫ ▫ ▫ ▫ ▫
        ▪ ▪ ▪ ▪ ▪ ▪ ▪ ▫ ▫ ▫ ▫ ▫ ▫ ▫ ▫
                      ↑ frontier cell

    FILL ORDER: column by column, all rows in a column before next column.

    WAVE: Two sine waves combine. Each filled cell gets unique brightness.
        w1 = sin( col_ratio * π*5  -  time*WAVE_SPEED_H  +  offset*1.2 )
        w2 = sin( row_ratio * π*3  +  time*WAVE_SPEED_V  +  col*0.18 )
        brightness = w1 * WAVE_WEIGHT_H + w2 * WAVE_WEIGHT_V

    BAR_OFFSET: Pass 0, 7, 14 for the three bars so waves are out of sync.
    """
    text   = Text()
    t      = frame / ANIMATION_FPS
    total  = cols * rows
    filled = int(progress * total)
    f_cols = filled // rows
    p_rows = filled % rows

    for row in range(rows):
        for col in range(cols):
            is_f     = (col < f_cols) or (col == f_cols and row < p_rows)
            is_front = (col == f_cols) and (row == p_rows) and (progress < 1.0)

            if is_f:
                w1 = math.sin((col/cols)*math.pi*5 - t*WAVE_SPEED_H + bar_offset*1.2)*0.5 + 0.5
                w2 = math.sin((row/rows)*math.pi*3 + t*WAVE_SPEED_V + col*0.18)*0.3 + 0.7
                b  = w1*WAVE_WEIGHT_H + w2*WAVE_WEIGHT_V
                if   b > 0.75:
                    style = f"bold {bar_color}"
                elif b > 0.50:
                    style = bar_color
                else:
                    style = f"dim {bar_color}"
                text.append(MESH_CELL_CHAR, style=style)

            elif is_front:
                pulse = math.sin(frame * FRONTIER_BLINK_SPEED) * 0.5 + 0.5
                style = f"bold {bar_color}" if pulse > 0.5 else bar_color
                text.append(MESH_CELL_CHAR, style=style)

            elif filled == 0:
                # Working animation at 0% - show pulsing cell at start
                pulse = math.sin(frame * FRONTIER_BLINK_SPEED) * 0.5 + 0.5
                if col == 0 and row == 0:
                    style = f"bold {bar_color}" if pulse > 0.5 else bar_color
                    text.append(MESH_CELL_CHAR, style=style)
                else:
                    text.append(MESH_EMPTY_CHAR, style=f"dim {dim_color}")

            else:
                text.append(MESH_EMPTY_CHAR, style=f"dim {dim_color}")

        if row < rows - 1:
            text.append("\n")

    return text


# ═══════════════════════════════════════════════════════════════════════════════
# FRAME BUILDER — ASSEMBLES COMPLETE UI PANEL
# ═══════════════════════════════════════════════════════════════════════════════

def _trunc(s: str, n: int) -> str:
    """Truncate string with ellipsis if too long."""
    return s if len(s) <= n else s[:n-1] + "…"


def _build_frame(stats: IndexingStats, frame: int, theme: dict) -> Panel:
    """
    Assembles the complete Rich Panel for one animation frame.

    LAYOUT:
        ╭──────────────────────────────────────────────╮
        │  ⠹ SmartFork Indexer   11/76  1,847ch  62s  │  header
        │  ────────────────────────────────────────────│  divider
        │  Sessions    ▪ ▪ ▪ ▪ ▪ · · · · · · ·  14%  │  bar1 (3 rows)
        │                                             │
        │  Embedding   ▪ ▪ ▪ ▪ ▪ ▪ ▪ · · · · ·  61%  │  bar2 (3 rows)
        │                                             │
        │  BM25 Index  ▪ ▪ ▪ ▪ · · · · · · · ·  39%  │  bar3 (3 rows)
        │  ────────────────────────────────────────── │  divider
        │  ▸ JWT Auth with FastAPI + SQLAlchemy        │  footer
        ╰──────────────────────────────────────────────╯
    """
    primary = theme["text_primary"]
    muted   = theme["text_muted"]
    dim_c   = theme["text_dim"]
    border  = theme["done_color"] if stats.done else theme["panel_border"]
    spin    = SPINNER_FRAMES[frame % len(SPINNER_FRAMES)]

    # Header
    hdr = Table.grid(expand=True)
    hdr.add_column(ratio=2)
    hdr.add_column(ratio=1)

    left = Text()
    if stats.done:
        left.append("✓ ", style=f"bold {theme['done_color']}")
        left.append("Index Complete", style=f"bold {theme['done_color']}")
    else:
        left.append(f"{spin} ", style=f"bold {theme['spinner_color']}")
        left.append("SmartFork Indexer", style=f"bold {primary}")

    right = Text(justify="right")
    right.append(str(stats.indexed_sessions), style=f"bold {theme['bars'][0]['color']}")
    right.append(f"/{stats.total_sessions}  ", style=f"dim {muted}")
    right.append(f"{stats.total_chunks:,}", style=f"bold {theme['bars'][1]['color']}")
    right.append(" ch  ", style=f"dim {muted}")
    if stats.done:
        right.append(f"{stats.elapsed:.1f}s", style=f"dim {muted}")
    elif stats.overall_progress > 0.01:
        right.append(f"ETA {int(stats.eta_seconds)}s", style=f"dim {muted}")
    hdr.add_row(left, right)

    # Three bars
    bar_table = Table.grid(expand=True, padding=(1, 0))
    bar_table.add_column(width=14)
    bar_table.add_column(ratio=1)
    bar_table.add_column(width=6)

    progresses  = [stats.overall_progress, stats.phase_progress, stats.bm25_progress]
    bar_offsets = [0, 7, 14]

    for i, bcfg in enumerate(theme["bars"]):
        mesh = _render_mesh_bar(
            progress=progresses[i], frame=frame,
            bar_color=bcfg["color"], dim_color=dim_c,
            bar_offset=bar_offsets[i],
        )
        lbl = Text()
        lbl.append(f"  {bcfg['label']:<10}", style=f"dim {muted}")
        pct = Text(justify="right")
        pct.append(f"  {int(progresses[i]*100):>3}%", style=f"bold {bcfg['color']}")
        bar_table.add_row(lbl, mesh, pct)

    # Footer
    footer = Text()
    if stats.done:
        footer.append("  ● ", style=f"bold {theme['done_color']}")
        footer.append("All sessions indexed — ready to search", style=f"dim {muted}")
    elif stats.errors > 0:
        footer.append(f"  ⚠ {stats.errors} error(s)  ", style="bold #F59E0B")
        if stats.current_session_name:
            footer.append(_trunc(stats.current_session_name, 40), style=f"dim {muted}")
    elif stats.current_session_name:
        footer.append("  ▸ ", style=f"bold {theme['bars'][1]['color']}")
        footer.append(_trunc(stats.current_session_title or stats.current_session_name, 55),
                      style=f"dim {muted}")

    outer = Table.grid(expand=True, padding=(0, 0))
    outer.add_column()
    outer.add_row(hdr)
    outer.add_row(Text(""))
    outer.add_row(bar_table)
    outer.add_row(Text(""))
    outer.add_row(footer)

    return Panel(outer, border_style=border, padding=(1, 2), box=_BOX)


# ═══════════════════════════════════════════════════════════════════════════════
# SMARTFORK PROGRESS — MAIN CLASS WITH CONTEXT MANAGER
# ═══════════════════════════════════════════════════════════════════════════════

class SmartForkProgress:
    """
    Animated mesh progress bar. Always use as context manager.

    CORRECT USAGE IN cli.py:

        with SmartForkProgress(total_sessions=N, theme_name=config.theme) as prog:
            for i, session_dir in enumerate(sessions):
                prog.set_session(session_dir.name)
                prog.set_phase("Parsing",   0.0)
                # ... parse work ...
                prog.set_phase("Parsing",   1.0)
                prog.set_phase("Embedding", 0.0)
                # ... embed work ...
                prog.set_phase("Embedding", 1.0)
                prog.add_chunks(chunk_count)
                prog.set_bm25((i+1)/N)
                prog.advance()
            prog.finish()   # MUST be inside the with block
    """

    def __init__(self, total_sessions: int, theme_name: str = DEFAULT_THEME,
                 console: Optional[Console] = None, animation_fps: Optional[int] = None,
                 disable_animation: bool = False):
        if theme_name not in THEMES:
            theme_name = DEFAULT_THEME
        self._theme   = THEMES[theme_name]
        self._console = console or Console()
        self._stats   = IndexingStats(total_sessions=total_sessions)
        self._frame   = 0
        self._running = False
        self._live:   Optional[Live]             = None
        self._thread: Optional[threading.Thread] = None
        self._animation_fps = animation_fps or get_animation_fps()
        self._disable_animation = disable_animation
        self._last_progress_hash = None  # For change detection
        self._frames_without_change = 0  # Skip frames when no change

    def __enter__(self) -> "SmartForkProgress":
        # Use lower refresh rate for lite mode or disabled animations
        refresh_rate = 2 if self._disable_animation else self._animation_fps
        
        self._live = Live(
            _build_frame(self._stats, self._frame, self._theme),
            console=self._console,
            refresh_per_second=refresh_rate,
            transient=False,
        )
        self._live.__enter__()
        self._running = True
        
        # Only start animation thread if not disabled
        if not self._disable_animation:
            self._thread = threading.Thread(
                target=self._loop, daemon=True, name="sf-anim"
            )
            self._thread.start()
        return self

    def __exit__(self, *args):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._live:
            self._live.update(_build_frame(self._stats, self._frame, self._theme))
            self._live.__exit__(*args)
        return False

    def _loop(self):
        """Animation loop with adaptive FPS and change detection."""
        while self._running:
            # Get current FPS (may be adaptively reduced)
            current_fps = get_animation_fps()
            interval = 1.0 / current_fps
            
            # Check if progress has actually changed
            current_hash = self._get_progress_hash()
            if current_hash == self._last_progress_hash:
                self._frames_without_change += 1
                # Skip rendering every other frame if no change (reduce CPU by 50%)
                if self._frames_without_change > 3:  # After 3 frames of no change
                    time.sleep(interval * 2)  # Double the interval
                    continue
            else:
                self._frames_without_change = 0
                self._last_progress_hash = current_hash
            
            self._frame += 1
            if self._live:
                try:
                    self._live.update(_build_frame(self._stats, self._frame, self._theme))
                except Exception:
                    pass
            time.sleep(interval)
    
    def _get_progress_hash(self) -> str:
        """Get a hash of current progress state for change detection."""
        return f"{self._stats.indexed_sessions}:{self._stats.phase_progress:.2f}:{self._stats.bm25_progress:.2f}:{self._stats.total_chunks}"

    # ── MUTATORS ──────────────────────────────────────────────────

    def set_session(self, name: str, title: str = "") -> None:
        """Call at start of each session. Resets phase bar to 0."""
        mark_activity()  # Mark user activity for adaptive FPS
        self._stats.current_session_name  = name
        self._stats.current_session_title = title
        self._stats.phase_label           = "Parsing"
        self._stats.phase_progress        = 0.0

    def set_phase(self, label: str, progress: float) -> None:
        """
        Update phase bar (Bar 2). Call multiple times per session.
        label: "Parsing" | "Embedding" | "Chunking" (max 12 chars)
        progress: 0.0 to 1.0
        """
        mark_activity()  # Mark user activity for adaptive FPS
        self._stats.phase_label    = label
        self._stats.phase_progress = max(0.0, min(1.0, progress))

    def set_bm25(self, progress: float) -> None:
        """Update BM25 bar (Bar 3). Call after each session: (i+1)/total."""
        mark_activity()  # Mark user activity for adaptive FPS
        self._stats.bm25_progress = max(0.0, min(1.0, progress))

    def advance(self, count: int = 1) -> None:
        """Advance sessions bar (Bar 1) by count. Call once per session."""
        mark_activity()  # Mark user activity for adaptive FPS
        self._stats.indexed_sessions = min(
            self._stats.indexed_sessions + count, self._stats.total_sessions
        )

    def add_chunks(self, n: int) -> None:
        """Add n to running chunk total in header."""
        mark_activity()  # Mark user activity for adaptive FPS
        self._stats.total_chunks += n

    def add_error(self) -> None:
        """Increment error counter shown in footer."""
        mark_activity()  # Mark user activity for adaptive FPS
        self._stats.errors += 1

    def finish(self) -> None:
        """
        Mark complete. Sets all bars to 100%. CALL INSIDE the with block.
        Includes 0.5s sleep so final frame renders before __exit__.
        """
        self._stats.indexed_sessions = self._stats.total_sessions
        self._stats.phase_progress   = 1.0
        self._stats.bm25_progress    = 1.0
        self._stats.done             = True
        time.sleep(0.5)


# ═══════════════════════════════════════════════════════════════════════════════
# DISCOVERY PHASE — SCANNING SESSIONS WITH SPINNER
# ═══════════════════════════════════════════════════════════════════════════════

def display_discovery_phase(
    tasks_path: Path, db_session_ids: set,
    console: Optional[Console] = None, theme_name: str = DEFAULT_THEME,
) -> tuple:
    """
    Scans tasks_path with live spinner. Returns (all_sessions, new_count, mod_count).
    VALID SESSION = directory containing api_conversation_history.json
    Spinner updates on EVERY directory scanned — no fake sleep.
    """
    if console is None:
        console = Console()
    theme   = THEMES.get(theme_name, THEMES[DEFAULT_THEME])
    found   = []
    new_c = 0
    mod_c = 0
    frame = 0

    with Live(refresh_per_second=8, console=console, transient=True) as live:  # Reduced from 15 for lower CPU
        try:
            items = list(tasks_path.iterdir())
        except PermissionError as e:
            console.print(f"[red]Cannot read tasks path: {e}[/red]")
            return [], 0, 0

        for item in items:
            frame += 1
            spin = SPINNER_FRAMES[frame % len(SPINNER_FRAMES)]
            t = Text()
            t.append(f"  {spin} ", style=f"bold {theme['spinner_color']}")
            t.append("Scanning sessions...  ", style=f"bold {theme['text_primary']}")
            t.append(f"{len(found)} found", style=f"dim {theme['text_muted']}")
            live.update(t)
            if item.is_dir() and (item / "api_conversation_history.json").exists():
                found.append(item)
                if item.name not in db_session_ids:
                    new_c += 1
                else:
                    mod_c += 1

    summary = Text()
    summary.append("  ✓ ", style=f"bold {theme['done_color']}")
    summary.append(f"Found {len(found)} sessions  ", style=theme["text_primary"])
    summary.append(f"({new_c} new, {mod_c} already indexed)",
                   style=f"dim {theme['text_muted']}")
    console.print(summary)
    console.print()
    return found, new_c, mod_c


# ═══════════════════════════════════════════════════════════════════════════════
# COMPLETION SUMMARY — FINAL METRICS DISPLAY
# ═══════════════════════════════════════════════════════════════════════════════

def display_completion_summary(
    stats: IndexingStats, total_db_sessions: int = 0,
    console: Optional[Console] = None, theme_name: str = DEFAULT_THEME,
) -> None:
    """
    3-column metric grid after indexing completes.

    Layout:
        ╭──────────────────────────────────────╮
        │         ✓  Index Complete            │
        │                                      │
        │   76         12,495       45.2s      │
        │ sessions      chunks    total time   │
        ╰──────────────────────────────────────╯
    """
    if console is None:
        console = Console()
    theme  = THEMES.get(theme_name, THEMES[DEFAULT_THEME])
    done_c = theme["done_color"]
    muted  = theme["text_muted"]
    bcolors = [b["color"] for b in theme["bars"]]

    grid = Table.grid(expand=True, padding=(0, 4))
    grid.add_column(ratio=1, justify="center")
    grid.add_column(ratio=1, justify="center")
    grid.add_column(ratio=1, justify="center")

    def stat(val, lbl, color):
        t = Text(justify="center")
        t.append(f"{val}\n", style=f"bold {color}")
        t.append(lbl, style=f"dim {muted}")
        return t

    grid.add_row(
        stat(str(stats.indexed_sessions), "sessions",  bcolors[0]),
        stat(f"{stats.total_chunks:,}",   "chunks",    bcolors[1]),
        stat(f"{stats.elapsed:.1f}s",     "total time", bcolors[2]),
    )
    if stats.errors > 0:
        grid.add_row(Text(""), Text(""), Text(""))
        et = Text(justify="center")
        et.append(f"⚠ {stats.errors} session(s) failed", style="bold #F59E0B")
        grid.add_row(et, Text(""), Text(""))

    console.print()
    console.print(Panel(grid, title=f"[bold {done_c}]  ✓  Index Complete[/bold {done_c}]",
                        border_style=done_c, box=_BOX, padding=(1, 2)))
    console.print(f"\n  [dim {muted}]Ready to search. Run [bold]smartfork search[/bold] to start.[/dim {muted}]\n")


# ═══════════════════════════════════════════════════════════════════════════════
# BACKWARD COMPATIBILITY ALIASES
# ═══════════════════════════════════════════════════════════════════════════════

class AnimatedProgressDisplay(SmartForkProgress):
    """Backward compatible alias for SmartForkProgress."""
    def __init__(self, console=None, total_sessions: int = 0, **kw):
        super().__init__(total_sessions=total_sessions, console=console)


class IndexingProgressDisplay(SmartForkProgress):
    """Backward compatible alias for SmartForkProgress."""
    def __init__(self, console=None, total_sessions: int = 0, **kw):
        super().__init__(total_sessions=total_sessions, console=console)


def index_with_progress(*args, **kwargs):
    """Legacy function - no longer used, kept for import compatibility."""
    pass
