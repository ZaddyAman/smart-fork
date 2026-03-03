# SmartFork — Complete Implementation Guide
## Animated Mesh Progress Bar + Multi-Theme System
### Written for LLM Implementors | March 2026

---

## OVERVIEW FOR THE IMPLEMENTING LLM

You are implementing two tightly coupled features for SmartFork,
an AI session intelligence CLI tool written in Python.

**Feature 1:** Replace the existing broken ASCII progress bar in
`src/smartfork/ui/progress.py` with an animated mesh progress bar
made of tiny 3-row block cells that wave like AGI particles.

**Feature 2:** Build a theme system that lets users permanently
set their preferred color palette via `smartfork config-theme <name>`,
stored in `~/.smartfork/config.toml`, and automatically applied
every time the indexer runs.

**Runtime:** Windows 11 + WSL, Python 3.10+, Rich library already installed.
Typer for CLI, Pydantic for config, Loguru for logging, ChromaDB as vector DB.

---

## PART 1 — FILE MAP

ONLY touch these files:

```
src/smartfork/
├── config.py                ← ADD: theme field to SmartForkConfig
├── ui/
│   ├── progress.py          ← FULL REWRITE
│   └── __init__.py          ← ADD: exports
└── cli.py                   ← MODIFY: index command + add config-theme command
```

---

## PART 2 — THE SIX THEMES

Paste this entire dict into progress.py as the THEMES constant.

```python
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
    },
}

DEFAULT_THEME = "obsidian"
```

---

## PART 3 — config.py CHANGES

### 3.1 Add theme field

Find SmartForkConfig (Pydantic BaseModel) and add ONE field:

```python
class SmartForkConfig(BaseModel):
    # ... all existing fields unchanged ...
    theme: str = "obsidian"
```

### 3.2 Add validator

```python
from pydantic import validator

@validator("theme")
def validate_theme(cls, v):
    valid = {"phosphor","obsidian","ember","arctic","iron","tungsten"}
    if v not in valid:
        raise ValueError(f"Unknown theme '{v}'. Valid: {', '.join(sorted(valid))}")
    return v
```

No other changes needed. Existing config save/load handles the new field automatically.

---

## PART 4 — progress.py FULL REWRITE

Delete entire file contents. Replace with the following sections in order.

### 4.1 Imports and constants

```python
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

# Paste THEMES dict and DEFAULT_THEME from Part 2 here

ANIMATION_FPS        = 18
SPINNER_FRAMES       = ["⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧","⠇","⠏"]
MESH_ROWS            = 3
MESH_CELL_CHAR       = "▪"
MESH_EMPTY_CHAR      = "·"
MESH_CELLS_PER_ROW   = 52
WAVE_SPEED_H         = 3.5
WAVE_SPEED_V         = 2.2
WAVE_WEIGHT_H        = 0.45
WAVE_WEIGHT_V        = 0.55
FRONTIER_BLINK_SPEED = 0.22
```

### 4.2 IndexingStats dataclass

```python
@dataclass
class IndexingStats:
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
        if self.total_sessions == 0: return 0.0
        return min(1.0, self.indexed_sessions / self.total_sessions)

    @property
    def elapsed(self) -> float:
        return time.time() - self.start_time

    @property
    def eta_seconds(self) -> float:
        p = self.overall_progress
        if p <= 0.01: return 0.0
        return (self.elapsed / p) * (1.0 - p)
```

### 4.3 Mesh renderer — THE CORE FUNCTION

```python
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
    Renders one bar as a ROWS x COLS grid of tiny block characters.

    VISUAL EXAMPLE (3 rows, partial fill):
        ▪ ▪ ▪ ▪ ▪ ▪ ▪ ▪ · · · · · · ·
        ▪ ▪ ▪ ▪ ▪ ▪ ▪ · · · · · · · ·
        ▪ ▪ ▪ ▪ ▪ ▪ ▪ ● · · · · · · ·
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
                if   b > 0.75: style = f"bold {bar_color}"
                elif b > 0.50: style = bar_color
                else:          style = f"dim {bar_color}"
                text.append(MESH_CELL_CHAR, style=style)

            elif is_front:
                pulse = math.sin(frame * FRONTIER_BLINK_SPEED) * 0.5 + 0.5
                style = f"bold {bar_color}" if pulse > 0.5 else bar_color
                text.append(MESH_CELL_CHAR, style=style)

            else:
                text.append(MESH_EMPTY_CHAR, style=f"dim {dim_color}")

            text.append(" ", style="")

        if row < rows - 1:
            text.append("\n")

    return text
```

### 4.4 Frame builder

```python
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
    bar_table = Table.grid(expand=True, padding=(0,0))
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
        if i < 2:
            bar_table.add_row(Text(""), Text(""), Text(""))

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

    outer = Table.grid(expand=True, padding=(0,0))
    outer.add_column()
    outer.add_row(hdr)
    outer.add_row(Text(""))
    outer.add_row(bar_table)
    outer.add_row(Text(""))
    outer.add_row(footer)

    return Panel(outer, border_style=border, padding=(1,2), box=box.ROUNDED)


def _trunc(s: str, n: int) -> str:
    return s if len(s) <= n else s[:n-1] + "…"
```

### 4.5 SmartForkProgress public class

```python
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
                 console: Optional[Console] = None):
        if theme_name not in THEMES:
            theme_name = DEFAULT_THEME
        self._theme   = THEMES[theme_name]
        self._console = console or Console()
        self._stats   = IndexingStats(total_sessions=total_sessions)
        self._frame   = 0
        self._running = False
        self._live:   Optional[Live]             = None
        self._thread: Optional[threading.Thread] = None

    def __enter__(self) -> "SmartForkProgress":
        self._live = Live(
            _build_frame(self._stats, self._frame, self._theme),
            console=self._console,
            refresh_per_second=0,
            transient=False,
        )
        self._live.__enter__()
        self._running = True
        self._thread  = threading.Thread(
            target=self._loop, daemon=True, name="sf-anim"
        )
        self._thread.start()
        return self

    def __exit__(self, *args):
        self._running = False
        if self._thread: self._thread.join(timeout=2.0)
        if self._live:
            self._live.update(_build_frame(self._stats, self._frame, self._theme))
            self._live.__exit__(*args)
        return False

    def _loop(self):
        interval = 1.0 / ANIMATION_FPS
        while self._running:
            self._frame += 1
            if self._live:
                try:
                    self._live.update(_build_frame(self._stats, self._frame, self._theme))
                except Exception:
                    pass
            time.sleep(interval)

    # ── MUTATORS ──────────────────────────────────────────────────

    def set_session(self, name: str, title: str = "") -> None:
        """Call at start of each session. Resets phase bar to 0."""
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
        self._stats.phase_label    = label
        self._stats.phase_progress = max(0.0, min(1.0, progress))

    def set_bm25(self, progress: float) -> None:
        """Update BM25 bar (Bar 3). Call after each session: (i+1)/total."""
        self._stats.bm25_progress = max(0.0, min(1.0, progress))

    def advance(self, count: int = 1) -> None:
        """Advance sessions bar (Bar 1) by count. Call once per session."""
        self._stats.indexed_sessions = min(
            self._stats.indexed_sessions + count, self._stats.total_sessions
        )

    def add_chunks(self, n: int) -> None:
        """Add n to running chunk total in header."""
        self._stats.total_chunks += n

    def add_error(self) -> None:
        """Increment error counter shown in footer."""
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
```

### 4.6 Discovery phase + completion summary

```python
def display_discovery_phase(
    tasks_path: Path, db_session_ids: set,
    console: Optional[Console] = None, theme_name: str = DEFAULT_THEME,
) -> tuple:
    """
    Scans tasks_path with live spinner. Returns (all_sessions, new_count, mod_count).
    VALID SESSION = directory containing api_conversation_history.json
    Spinner updates on EVERY directory scanned — no fake sleep.
    """
    if console is None: console = Console()
    theme   = THEMES.get(theme_name, THEMES[DEFAULT_THEME])
    found   = []; new_c = 0; mod_c = 0; frame = 0

    with Live(refresh_per_second=15, console=console, transient=True) as live:
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
                if item.name not in db_session_ids: new_c += 1
                else: mod_c += 1

    summary = Text()
    summary.append("  ✓ ", style=f"bold {theme['done_color']}")
    summary.append(f"Found {len(found)} sessions  ", style=theme["text_primary"])
    summary.append(f"({new_c} new, {mod_c} already indexed)",
                   style=f"dim {theme['text_muted']}")
    console.print(summary)
    console.print()
    return found, new_c, mod_c


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
    if console is None: console = Console()
    theme  = THEMES.get(theme_name, THEMES[DEFAULT_THEME])
    done_c = theme["done_color"]
    muted  = theme["text_muted"]
    bcolors = [b["color"] for b in theme["bars"]]

    grid = Table.grid(expand=True, padding=(0,4))
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
        stat(f"{stats.elapsed:.1f}s",     "total time",bcolors[2]),
    )
    if stats.errors > 0:
        grid.add_row(Text(""), Text(""), Text(""))
        et = Text(justify="center")
        et.append(f"⚠ {stats.errors} session(s) failed", style="bold #F59E0B")
        grid.add_row(et, Text(""), Text(""))

    console.print()
    console.print(Panel(grid, title=f"[bold {done_c}]  ✓  Index Complete[/bold {done_c}]",
                        border_style=done_c, box=box.ROUNDED, padding=(1,2)))
    console.print(f"\n  [dim {muted}]Ready to search. Run [bold]smartfork search[/bold] to start.[/dim {muted}]\n")


# Backward compat aliases
class AnimatedProgressDisplay(SmartForkProgress):
    def __init__(self, console=None, **kw):
        super().__init__(total_sessions=0, console=console)

class IndexingProgressDisplay(SmartForkProgress):
    def __init__(self, console=None, **kw):
        super().__init__(total_sessions=0, console=console)
```

---

## PART 5 — __init__.py EXPORTS

In `src/smartfork/ui/__init__.py`:

```python
from .progress import (
    SmartForkProgress,
    IndexingStats,
    display_discovery_phase,
    display_completion_summary,
    THEMES,
    DEFAULT_THEME,
)
__all__ = [
    "SmartForkProgress","IndexingStats",
    "display_discovery_phase","display_completion_summary",
    "THEMES","DEFAULT_THEME",
]
```

---

## PART 6 — cli.py CHANGES

### 6.1 Update imports at top of cli.py

```python
# REMOVE:
from .ui.progress import AnimatedProgressDisplay

# ADD:
from .ui.progress import (
    SmartForkProgress, display_discovery_phase,
    display_completion_summary, THEMES, DEFAULT_THEME,
)
from rich import box
import time
```

### 6.2 Replace entire index() function

```python
@app.command()
def index(
    force: bool = typer.Option(False, "--force", "-f", help="Force full re-index"),
    watch: bool = typer.Option(False, "--watch", "-w", help="Watch after indexing"),
):
    """Index all Kilo Code sessions."""
    config     = get_config()
    theme_name = getattr(config, "theme", DEFAULT_THEME)
    theme      = THEMES.get(theme_name, THEMES[DEFAULT_THEME])

    # Header panel
    path_str = str(config.kilo_code_tasks_path)
    if len(path_str) > 58: path_str = "…" + path_str[-55:]
    hdr = Text()
    hdr.append("⚡ SmartFork ", style=f"bold {theme['bars'][0]['color']}")
    hdr.append("Indexer\n",    style=f"bold {theme['text_primary']}")
    hdr.append(f"  {path_str}", style=f"dim {theme['text_muted']}")
    console.print(Panel(hdr, border_style=theme["panel_border"], box=box.ROUNDED, padding=(0,2)))
    console.print()

    if not config.kilo_code_tasks_path.exists():
        console.print(f"[red]Tasks path not found:[/red] {config.kilo_code_tasks_path}")
        raise typer.Exit(1)

    db = ChromaDatabase(config.chroma_db_path)
    if force:
        console.print(f"[{theme['text_muted']}]Resetting database...[/{theme['text_muted']}]")
        db.reset()

    db_session_ids = set()
    try: db_session_ids = set(db.get_unique_sessions())
    except Exception: pass

    # Discovery
    all_sessions, new_count, _ = display_discovery_phase(
        tasks_path=config.kilo_code_tasks_path,
        db_session_ids=db_session_ids,
        console=console, theme_name=theme_name,
    )

    if not all_sessions:
        console.print(f"[{theme['text_muted']}]No sessions found.[/{theme['text_muted']}]")
        raise typer.Exit(0)

    sessions_to_index = [s for s in all_sessions if s.name not in db_session_ids]

    if not sessions_to_index:
        console.print(f"[{theme['done_color']}]✓[/{theme['done_color']}] All sessions already indexed.\n")
        if watch: _start_watch_mode(config, db, console, theme)
        raise typer.Exit(0)

    console.print(
        f"  [{theme['bars'][1]['color']}]→[/{theme['bars'][1]['color']}] "
        f"Indexing {len(sessions_to_index)} sessions...\n"
    )

    indexer    = FullIndexer(db, chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap)
    final_stats = None

    with SmartForkProgress(
        total_sessions=len(sessions_to_index),
        theme_name=theme_name,
        console=console,
    ) as prog:

        for i, session_dir in enumerate(sessions_to_index):
            sid = session_dir.name
            prog.set_session(sid)

            try:
                prog.set_phase("Parsing", 0.3)
                prog.set_phase("Embedding", 0.0)
                chunks = indexer.index_session(session_dir)
                prog.set_phase("Embedding", 1.0)

                # Try to get LLM-generated title
                title = ""
                try:
                    sc = db.get_session_chunks(sid)
                    if sc: title = sc[0].metadata.session_title or ""
                except Exception: pass
                if title: prog.set_session(sid, title=title)

                prog.add_chunks(chunks)
                prog.advance()

            except Exception as e:
                logger.error(f"Failed to index {sid}: {e}")
                prog.add_error()
                prog.advance()

            prog.set_bm25((i+1) / len(sessions_to_index))

        prog.finish()
        final_stats = prog._stats

    # Completion
    total_db = 0
    try: total_db = len(db.get_unique_sessions())
    except Exception: pass

    if final_stats:
        display_completion_summary(
            stats=final_stats, total_db_sessions=total_db,
            console=console, theme_name=theme_name,
        )

    help_manager.show_after_command(
        UserAction.INDEX,
        db_session_count=total_db,
        processed=final_stats.indexed_sessions if final_stats else 0,
        failed=final_stats.errors if final_stats else 0,
    )

    if watch: _start_watch_mode(config, db, console, theme)


def _start_watch_mode(config, db, console, theme):
    console.print(
        f"\n  [{theme['bars'][0]['color']}]◉[/{theme['bars'][0]['color']}] "
        f"Watch mode. Ctrl+C to stop.\n"
    )
    incremental = IncrementalIndexer(db)
    watcher = TranscriptWatcher(config.kilo_code_tasks_path, incremental.on_session_changed)
    watcher.start()
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        console.print(f"\n  [{theme['text_muted']}]Watcher stopped.[/{theme['text_muted']}]")
        watcher.stop()
```

### 6.3 Add config-theme command

```python
@app.command("config-theme")
def config_theme(
    theme_name: Optional[str] = typer.Argument(None, help="Theme to set"),
    list_all:   bool = typer.Option(False, "--list", "-l", help="List all themes"),
):
    """
    Set or view the color theme.

    Usage:
        smartfork config-theme obsidian    # set theme
        smartfork config-theme --list      # list all themes
        smartfork config-theme             # show current
    """
    config  = get_config()
    current = getattr(config, "theme", DEFAULT_THEME)

    if list_all or theme_name is None:
        tbl = Table(show_header=True, box=box.SIMPLE)
        tbl.add_column("Theme",  style="bold", width=12)
        tbl.add_column("Description", style="dim", width=40)
        tbl.add_column("",  width=10)
        for tid, td in THEMES.items():
            c0,c1,c2 = [b["color"] for b in td["bars"]]
            swatch = f"[{c0}]▪[/{c0}][{c1}]▪[/{c1}][{c2}]▪[/{c2}] {td['name']}"
            status = "[green]● active[/green]" if tid == current else ""
            tbl.add_row(swatch, td["desc"], status)
        console.print(Panel(tbl, title="[bold]SmartFork Themes[/bold]", box=box.ROUNDED))
        if theme_name is None:
            console.print(f"\n  Current: [bold]{current}[/bold]")
            console.print(f"  Set with: [dim]smartfork config-theme <name>[/dim]\n")
        return

    if theme_name not in THEMES:
        console.print(f"[red]Unknown theme '{theme_name}'[/red]")
        console.print(f"[dim]Valid: {', '.join(THEMES.keys())}[/dim]")
        raise typer.Exit(1)

    config.theme = theme_name
    save_config(config)   # replace with your actual save function name

    td = THEMES[theme_name]
    c  = td["bars"][1]["color"]
    console.print(f"\n  [{c}]✓[/{c}] Theme → [bold]{td['name']}[/bold] — {td['desc']}")
    console.print(f"  [dim]Saved to config.toml[/dim]\n")
```

---

## PART 7 — WINDOWS COMPATIBILITY

Add to the top of cli.py before any Rich imports:

```python
import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
```

Add to progress.py after constants:

```python
def _unicode_ok() -> bool:
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
MESH_EMPTY_CHAR = "·" if _UNICODE else "."
SPINNER_FRAMES  = (["⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧","⠇","⠏"]
                   if _UNICODE else ["|","/","-","\\"])
_BOX = box.ROUNDED if _UNICODE else box.ASCII
```

Then replace `box.ROUNDED` with `_BOX` throughout progress.py (3 occurrences).

---

## PART 8 — TESTING CHECKLIST

```
□ smartfork index                    Mesh bars animate continuously at ~18fps
□ smartfork index                    Three bars, each different shade of theme color
□ smartfork index                    Wave ripples left-to-right through filled cells
□ smartfork index                    Frontier cell pulses brighter at leading edge
□ smartfork index                    Footer shows current session name updating
□ smartfork index                    Header shows live session + chunk count
□ smartfork index                    Spinner is braille ⠹ not / or |
□ smartfork index                    Completion panel shows 3-stat grid
□ smartfork config-theme --list      6 themes displayed with color swatches
□ smartfork config-theme ember       Confirmation message shown
□ smartfork index  (after above)     Ember amber colors applied throughout
□ cat ~/.smartfork/config.toml       theme = "ember" visible
□ smartfork config-theme bogus       Error message with valid theme list
□ Windows Terminal: block chars      ▪ and · render without garbling
□ Windows Terminal: braille spinner  ⠹ renders without garbling
```

---

## PART 9 — 10 CRITICAL MISTAKES TO AVOID

**1. Using callback pattern instead of context manager**
WRONG:  `progress.display_indexing_progress(sessions, callback)`
RIGHT:  `with SmartForkProgress(N) as prog:` + explicit `prog.set_phase()` calls
WHY:    Callbacks prevent cli.py from calling phase update methods mid-session.
        Bar 2 stays at 0% forever.

**2. Never calling set_phase()**
WRONG:  Only calling `prog.advance()` per session
RIGHT:  `prog.set_phase("Parsing", 0.3)` then `prog.set_phase("Embedding", 1.0)` etc.
WHY:    Bar 2 (Embedding/Parsing) shows 0% the entire run.

**3. Calling set_bm25() once after the loop**
WRONG:  `prog.set_bm25(1.0)` after all sessions finish
RIGHT:  `prog.set_bm25((i+1)/total)` after EACH session inside the loop
WHY:    Bar 3 snaps from 0% to 100% in one frame. Looks broken.

**4. Calling finish() outside the with block**
WRONG:  Code after `with SmartForkProgress() as prog:` block ends
RIGHT:  `prog.finish()` as LAST line INSIDE the with block
WHY:    __exit__ runs before finish() sets done=True.
        Final frame never shows the completed state.

**5. Removing time.sleep(0.5) from finish()**
The animation thread is asynchronous. Without the sleep,
__exit__ may fire before done=True propagates to the next render.
Keep it.

**6. Missing `import math`**
_render_mesh_bar uses math.sin and math.pi.
NameError on first index call.

**7. Missing `import threading`**
SmartForkProgress.__enter__ creates threading.Thread.
NameError on first index call.

**8. Missing `from rich import box`**
_build_frame uses box.ROUNDED.
NameError on first index call.

**9. Not falling back to DEFAULT_THEME on unknown theme**
If user has a typo in config.toml, SmartForkProgress.__init__
should silently fall back:
    if theme_name not in THEMES: theme_name = DEFAULT_THEME

**10. Using transient=True in Live()**
transient=True makes the progress bar disappear when done.
Use transient=False so the final completed state remains visible.

---

## PART 10 — QUICK REFERENCE

**Method call order per session:**
```
prog.set_session(name)           # 1. identify session, reset phase bar
prog.set_phase("Parsing",   0.3) # 2. start parsing (set to ~0.3 immediately)
prog.set_phase("Embedding", 0.0) # 3. start embedding
prog.set_phase("Embedding", 1.0) # 4. embedding done
prog.add_chunks(N)               # 5. record chunk count
prog.set_bm25((i+1)/total)       # 6. update bm25 bar
prog.advance()                   # 7. advance sessions bar
# ... repeat for each session ...
prog.finish()                    # 8. LAST LINE inside with block
```

**Bar → method → visual:**
```
Bar 1 (Sessions)   ← advance()     ← i/total loop counter
Bar 2 (Phase)      ← set_phase()   ← indexer phase reporting
Bar 3 (BM25 Index) ← set_bm25()    ← (i+1)/total formula
```

**Valid theme names:**
```
phosphor  obsidian  ember  arctic  iron  tungsten
```

**Total new code:**
```
progress.py   ~300 lines
config.py     ~8 lines
cli.py        ~110 lines
__init__.py   ~10 lines
```

---

*SmartFork Implementation Guide — End*
*Target: src/smartfork/ui/progress.py + config.py + cli.py*
