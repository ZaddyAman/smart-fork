"""Progress indicators for SmartFork indexing operations.

This module provides beautiful animated progress displays using only ASCII characters
for full Windows CMD compatibility (no Unicode emojis).

Features:
- Animated wave effects using ASCII characters
- Smooth progress bar with partial block simulation
- Real-time stats (sessions, chunks, speed, ETA)
- Threading for smooth animation while indexing
- Windows CMD compatible - pure ASCII only
"""

from pathlib import Path
from typing import Optional, List, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import time
import threading
import sys
import os

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeRemainingColumn,
    TaskProgressColumn,
)
from rich.style import Style
from rich.segment import Segment
from rich.console import ConsoleOptions, RenderResult
from rich.table import Table
from rich.text import Text
from rich.live import Live


# =============================================================================
# ASCII ANIMATION COMPONENTS
# =============================================================================

class ASCIIWaves:
    """ASCII wave animation patterns for terminal displays.
    
    Uses only ASCII-safe characters for Windows CMD compatibility.
    """
    
    # Wave patterns using ASCII characters
    WAVE_PATTERNS = [
        "~~~~~~",
        "~~~~~-",
        "~~~~--",
        "~~~---",
        "~~----",
        "~-----",
        "------",
        "-----~",
        "----~~",
        "---~~~",
        "--~~~~",
        "-~~~~~",
    ]
    
    # Scanning patterns
    SCAN_PATTERNS = [
        "[     ]",
        "[=    ]",
        "[==   ]",
        "[===  ]",
        "[==== ]",
        "[=====]",
        "[ ====]",
        "[  ===]",
        "[   ==]",
        "[    =]",
    ]
    
    # Spinner patterns (ASCII only)
    SPINNER_PATTERNS = [
        "|",
        "/",
        "-",
        "\\",
    ]
    
    # Pulse patterns for activity indication
    PULSE_PATTERNS = [
        "o......",
        ".o.....",
        "..o....",
        "...o...",
        "....o..",
        ".....o.",
        "......o",
        ".....o.",
        "....o..",
        "...o...",
        "..o....",
        ".o.....",
    ]
    
    def __init__(self):
        self._wave_idx = 0
        self._scan_idx = 0
        self._spinner_idx = 0
        self._pulse_idx = 0
    
    def next_wave(self) -> str:
        """Get next wave animation frame."""
        frame = self.WAVE_PATTERNS[self._wave_idx]
        self._wave_idx = (self._wave_idx + 1) % len(self.WAVE_PATTERNS)
        return frame
    
    def next_scan(self) -> str:
        """Get next scanning animation frame."""
        frame = self.SCAN_PATTERNS[self._scan_idx]
        self._scan_idx = (self._scan_idx + 1) % len(self.SCAN_PATTERNS)
        return frame
    
    def next_spinner(self) -> str:
        """Get next spinner frame."""
        frame = self.SPINNER_PATTERNS[self._spinner_idx]
        self._spinner_idx = (self._spinner_idx + 1) % len(self.SPINNER_PATTERNS)
        return frame
    
    def next_pulse(self) -> str:
        """Get next pulse animation frame."""
        frame = self.PULSE_PATTERNS[self._pulse_idx]
        self._pulse_idx = (self._pulse_idx + 1) % len(self.PULSE_PATTERNS)
        return frame


class SmoothProgressBar:
    """ASCII-only smooth progress bar with partial block simulation.
    
    Uses character density to simulate partial blocks without Unicode:
    # = full block
    O = 75% fill
    o = 50% fill
    . = 25% fill
    - = empty
    """
    
    # Partial fill characters (ASCII only, ordered by density)
    FILL_CHARS = ['#', 'O', 'o', '.', '-']
    
    def __init__(self, width: int = 40):
        self.width = width
    
    def render(self, percent: float, animated: bool = False) -> str:
        """Render progress bar at given percentage.
        
        Args:
            percent: Progress percentage (0-100)
            animated: Whether to add animated effect
            
        Returns:
            ASCII progress bar string
        """
        # Calculate filled width
        filled = (percent / 100.0) * self.width
        filled_int = int(filled)
        partial = filled - filled_int
        
        # Build bar
        bar = "["
        
        # Fully filled sections
        bar += "#" * filled_int
        
        # Partial section (if not complete)
        if filled_int < self.width:
            # Choose character based on partial fill amount
            if partial >= 0.75:
                bar += "O"
            elif partial >= 0.50:
                bar += "o"
            elif partial >= 0.25:
                bar += "."
            else:
                bar += "-"
            
            # Remaining empty space
            remaining = self.width - filled_int - 1
            bar += "-" * remaining
        
        bar += "]"
        
        # Add animated indicator if requested
        if animated and percent < 100:
            indicators = [">>", ">>"]
            idx = int(time.time() * 10) % len(indicators)
            bar = bar[:-1] + indicators[idx] + "]"
        
        return bar
    
    def render_with_percentage(self, percent: float, animated: bool = False) -> str:
        """Render progress bar with percentage."""
        bar = self.render(percent, animated)
        return f"{bar} {percent:5.1f}%"


# =============================================================================
# ANIMATED DISPLAY CLASSES
# =============================================================================

@dataclass
class IndexingStats:
    """Statistics for indexing operation."""
    sessions_found: int = 0
    sessions_new: int = 0
    sessions_modified: int = 0
    sessions_processed: int = 0
    sessions_failed: int = 0
    chunks_created: int = 0
    current_session: Optional[str] = None
    current_session_title: Optional[str] = None
    start_time: Optional[float] = None
    last_update_time: Optional[float] = None
    errors: List[str] = field(default_factory=list)
    
    @property
    def elapsed_time(self) -> float:
        """Get elapsed time since start."""
        if self.start_time is None:
            return 0.0
        return time.time() - self.start_time
    
    @property
    def sessions_per_second(self) -> float:
        """Calculate processing speed."""
        elapsed = self.elapsed_time
        if elapsed <= 0:
            return 0.0
        return self.sessions_processed / elapsed
    
    def eta_seconds(self, total_sessions: int) -> float:
        """Calculate estimated time to completion."""
        remaining = total_sessions - self.sessions_processed
        speed = self.sessions_per_second
        if speed <= 0:
            return 0.0
        return remaining / speed
    
    def format_eta(self, total_sessions: int) -> str:
        """Format ETA as human-readable string."""
        eta = self.eta_seconds(total_sessions)
        if eta <= 0:
            return "--:--"
        minutes = int(eta // 60)
        seconds = int(eta % 60)
        if minutes > 99:
            return f"{minutes//60}h{minutes%60:02d}m"
        return f"{minutes:02d}:{seconds:02d}"


class AnimatedProgressDisplay:
    """Enhanced animated progress display for indexing operations.
    
    Features:
    - Animated wave effects using ASCII-only characters
    - Smooth progress bar with partial block simulation
    - Real-time session title display with truncation
    - Live stats (sessions, chunks, speed, ETA)
    - Threading for smooth 60fps animation
    - Full Windows CMD compatibility (no Unicode)
    
    Animation phases:
    1. Scanning phase with wave animations
    2. Processing phase with smooth progress bar
    3. Completion summary with statistics
    """
    
    def __init__(self, console: Optional[Console] = None, refresh_rate: float = 15.0):
        """Initialize the animated progress display.
        
        Args:
            console: Optional Rich console instance
            refresh_rate: Animation refresh rate in Hz
        """
        self.console = console or Console()
        self.stats = IndexingStats()
        self.waves = ASCIIWaves()
        self.progress_bar = SmoothProgressBar(width=40)
        self.refresh_rate = refresh_rate
        self.refresh_interval = 1.0 / refresh_rate
        
        # Threading components
        self._animation_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._current_phase = "idle"  # idle, scanning, processing, complete
        self._total_sessions = 0
        
        # Display state
        self._last_display_lines = 0
    
    def _clear_lines(self, num_lines: int):
        """Clear specified number of lines from terminal."""
        # Use Rich's console for better compatibility
        if num_lines > 0:
            self.console.print("\r" + " " * 80 + "\r", end="")
    
    def _move_cursor_up(self, lines: int):
        """Move cursor up without clearing."""
        pass  # Not needed with Live display
    
    def _render_scanning_frame(self) -> List[str]:
        """Render a frame of the scanning animation."""
        wave = self.waves.next_wave()
        scan = self.waves.next_scan()
        spinner = self.waves.next_spinner()
        
        lines = [
            "",
            f"  {spinner} Scanning Kilo Code sessions directory... {scan}",
            f"  {wave}",
            "",
        ]
        
        if self.stats.sessions_found > 0:
            lines.append(f"  Found: {self.stats.sessions_found} sessions")
        
        return lines
    
    def _render_processing_frame(self) -> List[str]:
        """Render a frame of the processing animation."""
        lines = []
        
        with self._lock:
            total = self._total_sessions
            processed = self.stats.sessions_processed
            failed = self.stats.sessions_failed
            chunks = self.stats.chunks_created
            
            # Calculate progress
            if total > 0:
                percent = (processed / total) * 100
            else:
                percent = 0
            
            # Progress bar
            bar = self.progress_bar.render_with_percentage(percent, animated=True)
            
            # Current session info
            session_info = ""
            if self.stats.current_session:
                short_id = self.stats.current_session[:12]
                session_info = f"[{short_id}]"
                
                if self.stats.current_session_title:
                    title = self._truncate_title(self.stats.current_session_title, 35)
                    session_info += f" {title}"
            
            # Stats
            speed = self.stats.sessions_per_second
            eta = self.stats.format_eta(total)
            elapsed = self.stats.elapsed_time
            
            # Build display lines
            spinner = self.waves.next_spinner()
            lines.append("")
            lines.append(f"  {spinner} Indexing Sessions")
            lines.append(f"  {bar}")
            lines.append("")
            
            if session_info:
                lines.append(f"  Processing: {session_info}")
            
            lines.append(f"  Sessions: {processed}/{total} | Chunks: {chunks}")
            lines.append(f"  Speed: {speed:.1f} sess/s | ETA: {eta} | Elapsed: {elapsed:.1f}s")
            
            if failed > 0:
                lines.append(f"  Failed: {failed}")
        
        return lines
    
    def _render_complete_frame(self) -> List[str]:
        """Render completion frame."""
        lines = []
        
        elapsed = self.stats.elapsed_time
        
        lines.append("")
        lines.append("  [==== INDEXING COMPLETE ====]")
        lines.append("")
        lines.append(f"  Sessions processed: {self.stats.sessions_processed}")
        lines.append(f"  Chunks created: {self.stats.chunks_created}")
        lines.append(f"  Time elapsed: {elapsed:.1f}s")
        
        if self.stats.sessions_failed > 0:
            lines.append(f"  Failed: {self.stats.sessions_failed}")
        
        lines.append("")
        lines.append("  Ready to search!")
        lines.append("")
        
        return lines
    
    def _animation_loop(self):
        """Main animation loop running in separate thread."""
        while not self._stop_event.is_set():
            # Render appropriate frame based on phase
            if self._current_phase == "scanning":
                lines = self._render_scanning_frame()
            elif self._current_phase == "processing":
                lines = self._render_processing_frame()
            elif self._current_phase == "complete":
                lines = self._render_complete_frame()
            else:
                lines = []
            
            if lines:
                # Clear previous lines and render new frame
                self._clear_lines(self._last_display_lines)
                
                for line in lines:
                    sys.stdout.write(line + '\n')
                
                sys.stdout.flush()
                self._last_display_lines = len(lines)
            
            # Wait for next frame
            self._stop_event.wait(self.refresh_interval)
    
    def start_animation(self, phase: str):
        """Start the animation thread.
        
        Args:
            phase: Animation phase (scanning, processing, complete)
        """
        self._current_phase = phase
        self._stop_event.clear()
        
        if self._animation_thread is None or not self._animation_thread.is_alive():
            self._animation_thread = threading.Thread(target=self._animation_loop)
            self._animation_thread.daemon = True
            self._animation_thread.start()
    
    def stop_animation(self, clear: bool = True):
        """Stop the animation thread.
        
        Args:
            clear: Whether to clear the animation lines
        """
        self._stop_event.set()
        
        if self._animation_thread and self._animation_thread.is_alive():
            self._animation_thread.join(timeout=0.5)
        
        if clear and self._last_display_lines > 0:
            self._clear_lines(self._last_display_lines)
            self._last_display_lines = 0
    
    def _truncate_title(self, title: str, max_length: int = 40) -> str:
        """Truncate a title to fit within display limits.
        
        Args:
            title: The title to truncate
            max_length: Maximum length before truncation
            
        Returns:
            Truncated title with ellipsis if needed
        """
        if not title:
            return ""
        if len(title) <= max_length:
            return title
        return title[:max_length - 3] + "..."
    
    def display_discovery_phase(
        self,
        session_dirs: List[Path],
        db_session_ids: Optional[set] = None
    ) -> Tuple[int, int]:
        """Display animated discovery phase.
        
        Args:
            session_dirs: List of session directories found
            db_session_ids: Set of session IDs already in database
            
        Returns:
            Tuple of (new_count, modified_count)
        """
        self.stats.start_time = time.time()
        self.stats.sessions_found = len(session_dirs)
        
        # Calculate new and modified sessions
        new_count = 0
        modified_count = 0
        
        if db_session_ids:
            for session_dir in session_dirs:
                session_id = session_dir.name
                if session_id not in db_session_ids:
                    new_count += 1
                else:
                    modified_count += 1
        else:
            new_count = len(session_dirs)
        
        self.stats.sessions_new = new_count
        self.stats.sessions_modified = modified_count
        
        # Start scanning animation
        self.start_animation("scanning")
        
        # Let animation run briefly
        time.sleep(0.5)
        
        # Stop animation and show results
        self.stop_animation(clear=True)
        
        # Display discovery results
        self.console.print()
        self.console.print("[bold cyan]Scanning Kilo Code sessions directory...[/bold cyan]")
        
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column(style="dim")
        table.add_column(style="cyan", justify="right")
        table.add_column(style="dim")
        
        table.add_row("Found", str(self.stats.sessions_found), "total sessions")
        if new_count > 0:
            table.add_row("  -", str(new_count), "new sessions")
        if modified_count > 0:
            table.add_row("  -", str(modified_count), "modified sessions")
        
        self.console.print(table)
        self.console.print()
        
        return new_count, modified_count
    
    def display_indexing_progress(
        self,
        sessions_to_process: List[Path],
        index_callback: Callable[[Path], Tuple[int, Optional[str]]],
    ) -> dict:
        """Display animated progress while indexing sessions.
        
        Args:
            sessions_to_process: List of session directories to index
            index_callback: Function that indexes a session and returns (chunks, title)
            
        Returns:
            Statistics dictionary with results
        """
        self._total_sessions = len(sessions_to_process)
        
        # Start processing animation
        self.start_animation("processing")
        
        try:
            for i, session_dir in enumerate(sessions_to_process):
                session_id = session_dir.name
                
                with self._lock:
                    self.stats.current_session = session_id
                
                # Index the session
                try:
                    chunks, title = index_callback(session_dir)
                    
                    with self._lock:
                        self.stats.chunks_created += chunks
                        self.stats.sessions_processed += 1
                        if title:
                            self.stats.current_session_title = title
                    
                except Exception as e:
                    with self._lock:
                        self.stats.sessions_failed += 1
                        error_msg = str(e)
                        self.stats.errors.append(f"{session_id}: {error_msg}")
                
                # Small delay to allow animation to be visible for fast operations
                time.sleep(0.01)
        
        finally:
            # Stop animation
            self.stop_animation(clear=True)
        
        # Display final progress
        self._display_progress_summary()
        
        return {
            "processed": self.stats.sessions_processed,
            "failed": self.stats.sessions_failed,
            "chunks": self.stats.chunks_created,
            "errors": self.stats.errors,
        }
    
    def _display_progress_summary(self):
        """Display final progress summary after animation stops."""
        elapsed = self.stats.elapsed_time
        percent = 100.0 if self._total_sessions == 0 else (self.stats.sessions_processed / self._total_sessions) * 100
        
        # Final progress bar
        bar = self.progress_bar.render_with_percentage(percent, animated=False)
        
        self.console.print()
        self.console.print(f"  {bar}")
        self.console.print()
        self.console.print(f"  Sessions: {self.stats.sessions_processed}/{self._total_sessions}")
        self.console.print(f"  Chunks: {self.stats.chunks_created}")
        self.console.print(f"  Speed: {self.stats.sessions_per_second:.1f} sess/s")
        self.console.print(f"  Time: {elapsed:.1f}s")
        
        if self.stats.sessions_failed > 0:
            self.console.print(f"  [red]Failed: {self.stats.sessions_failed}[/red]")
        
        self.console.print()
    
    def display_completion_summary(
        self,
        total_db_sessions: int = 0,
        index_size_mb: Optional[float] = None
    ):
        """Display the final completion summary panel.
        
        Args:
            total_db_sessions: Total sessions now in database
            index_size_mb: Size of index in MB (if available)
        """
        elapsed = self.stats.elapsed_time
        
        # Build summary text
        summary_lines = [
            "[bold green]Indexing Complete[/bold green]",
            "",
        ]
        
        # Success stats
        if self.stats.sessions_processed > 0:
            summary_lines.append(
                f"  [green]+[/green] {self.stats.sessions_processed} sessions indexed"
            )
        
        if self.stats.sessions_failed > 0:
            summary_lines.append(
                f"  [red]x[/red] {self.stats.sessions_failed} sessions failed"
            )
        
        summary_lines.append(
            f"  [cyan]*[/cyan] {self.stats.chunks_created:,} chunks created"
        )
        
        if total_db_sessions > 0:
            summary_lines.append(
                f"  [blue]*[/blue] {total_db_sessions} total sessions in index"
            )
        
        summary_lines.append(f"  [dim]*[/dim] Time: {elapsed:.1f}s")
        
        if index_size_mb:
            summary_lines.append(
                f"  [dim]*[/dim] Index size: {index_size_mb:.1f} MB"
            )
        
        summary_lines.append("")
        summary_lines.append("[dim]Ready to search[/dim]")
        
        summary_text = "\n".join(summary_lines)
        
        # Create panel
        panel = Panel(
            summary_text,
            border_style="green",
            title="[bold]SmartFork Index[/bold]",
            title_align="center",
        )
        
        self.console.print()
        self.console.print(panel)
        
        # Show error summary if there were failures
        if self.stats.errors and len(self.stats.errors) > 0:
            self.console.print()
            error_table = Table(
                title=f"[yellow]Failed Sessions ({len(self.stats.errors)})[/yellow]",
                show_header=True,
                header_style="bold red"
            )
            error_table.add_column("Session ID", style="cyan", no_wrap=True)
            error_table.add_column("Error", style="red")
            
            for error in self.stats.errors[:10]:  # Show first 10
                parts = error.split(": ", 1)
                if len(parts) == 2:
                    error_table.add_row(parts[0][:20], parts[1][:60])
                else:
                    error_table.add_row("unknown", error[:60])
            
            if len(self.stats.errors) > 10:
                error_table.add_row(
                    "", 
                    f"[dim]... and {len(self.stats.errors) - 10} more[/dim]"
                )
            
            self.console.print(error_table)


# =============================================================================
# LEGACY CLASSES (for backward compatibility)
# =============================================================================

class ThickBarColumn(BarColumn):
    """A progress bar using hash symbols for better visibility.
    
    Uses only ASCII characters for Windows CMD compatibility.
    Renders as: [#######---------] 40%
    """
    
    def __init__(
        self,
        bar_width: int = 40,
        style: str = "bar.back",
        complete_style: str = "bar.complete",
        finished_style: str = "bar.finished",
        pulse_style: str = "bar.pulse",
    ):
        super().__init__(bar_width, style, complete_style, finished_style, pulse_style)
    
    def render(self, task) -> Text:
        """Render the progress bar as single-line text."""
        if task.total is None:
            return Text("", style=self.style)
        
        completed = min(task.completed, task.total)
        total = task.total
        
        # Calculate width
        width = self.bar_width or 40
        
        # Calculate filled vs empty
        filled_len = int(width * completed / total) if total > 0 else 0
        empty_len = width - filled_len
        
        # Determine style
        if task.finished:
            bar_style = self.finished_style
        elif getattr(task, 'pulse', None):
            bar_style = self.pulse_style
        else:
            bar_style = self.complete_style
        
        # Create single-line bar using ASCII characters
        bar = "[" + "#" * filled_len + "-" * empty_len + "]"
        
        return Text(bar, style=bar_style)


class IndexingProgressDisplay:
    """Beautiful progress display for indexing operations (Legacy).
    
    Features:
    - Discovery phase with scanning spinner
    - Progress bar during session processing
    - Per-session updates with truncated titles
    - Beautiful completion summary panel
    - Windows-compatible (no Unicode arrows)
    
    Note: Use AnimatedProgressDisplay for enhanced animations.
    """
    
    def __init__(self, console: Optional[Console] = None):
        """Initialize the progress display.
        
        Args:
            console: Optional Rich console instance
        """
        self.console = console or Console()
        self.stats = IndexingStats()
    
    def _truncate_title(self, title: str, max_length: int = 40) -> str:
        """Truncate a title to fit within display limits."""
        if not title:
            return ""
        if len(title) <= max_length:
            return title
        return title[:max_length - 3] + "..."
    
    def create_progress_bar(self) -> Progress:
        """Create a Rich Progress instance with custom columns."""
        return Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            ThickBarColumn(
                bar_width=50,
                complete_style="green",
                finished_style="green"
            ),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=self.console,
            expand=True,
        )
    
    def display_discovery_phase(
        self,
        session_dirs: List[Path],
        db_session_ids: Optional[set] = None
    ) -> Tuple[int, int]:
        """Display the discovery phase with scanning animation."""
        self.stats.start_time = time.time()
        self.stats.sessions_found = len(session_dirs)
        
        new_count = 0
        modified_count = 0
        
        if db_session_ids:
            for session_dir in session_dirs:
                session_id = session_dir.name
                if session_id not in db_session_ids:
                    new_count += 1
                else:
                    modified_count += 1
        else:
            new_count = len(session_dirs)
        
        self.stats.sessions_new = new_count
        self.stats.sessions_modified = modified_count
        
        self.console.print()
        self.console.print(
            "[bold cyan]Scanning Kilo Code sessions directory...[/bold cyan]"
        )
        
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column(style="dim")
        table.add_column(style="cyan", justify="right")
        table.add_column(style="dim")
        
        table.add_row("Found", str(self.stats.sessions_found), "total sessions")
        if new_count > 0:
            table.add_row("  -", str(new_count), "new sessions")
        if modified_count > 0:
            table.add_row("  -", str(modified_count), "modified sessions")
        
        self.console.print(table)
        self.console.print()
        
        return new_count, modified_count
    
    def display_indexing_progress(
        self,
        sessions_to_process: List[Path],
        index_callback: Callable[[Path], Tuple[int, Optional[str]]],
    ) -> dict:
        """Display progress while indexing sessions."""
        total = len(sessions_to_process)
        
        with self.create_progress_bar() as progress:
            task = progress.add_task(
                "Processing sessions...",
                total=total,
            )
            
            for i, session_dir in enumerate(sessions_to_process):
                session_id = session_dir.name
                short_id = session_id[:16] if len(session_id) > 16 else session_id
                
                display_name = f"[{short_id}]"
                progress.update(task, description=f"Processing {display_name}...")
                self.stats.current_session = session_id
                
                try:
                    chunks, title = index_callback(session_dir)
                    self.stats.chunks_created += chunks
                    self.stats.sessions_processed += 1
                    
                    if title:
                        self.stats.current_session_title = title
                        truncated = self._truncate_title(title, 35)
                        progress.console.print(
                            f"  [green]+[/green] {display_name} {truncated} "
                            f"([dim]{chunks} chunks[/dim])",
                            style=""
                        )
                    else:
                        progress.console.print(
                            f"  [green]+[/green] {display_name} "
                            f"([dim]{chunks} chunks[/dim])"
                        )
                    
                    progress.update(task, advance=1)
                    
                except Exception as e:
                    self.stats.sessions_failed += 1
                    error_msg = str(e)
                    self.stats.errors.append(f"{session_id}: {error_msg}")
                    
                    progress.console.print(
                        f"  [red]x[/red] {display_name} failed: {error_msg[:50]}"
                    )
                    progress.update(task, advance=1)
        
        return {
            "processed": self.stats.sessions_processed,
            "failed": self.stats.sessions_failed,
            "chunks": self.stats.chunks_created,
            "errors": self.stats.errors,
        }
    
    def display_completion_summary(
        self,
        total_db_sessions: int = 0,
        index_size_mb: Optional[float] = None
    ):
        """Display the final completion summary panel."""
        elapsed = time.time() - (self.stats.start_time or time.time())
        
        summary_lines = [
            "[bold green]Indexing Complete[/bold green]",
            "",
        ]
        
        if self.stats.sessions_processed > 0:
            summary_lines.append(
                f"  [green]+[/green] {self.stats.sessions_processed} sessions indexed"
            )
        
        if self.stats.sessions_failed > 0:
            summary_lines.append(
                f"  [red]x[/red] {self.stats.sessions_failed} sessions failed"
            )
        
        summary_lines.append(
            f"  [cyan]*[/cyan] {self.stats.chunks_created:,} chunks created"
        )
        
        if total_db_sessions > 0:
            summary_lines.append(
                f"  [blue]*[/blue] {total_db_sessions} total sessions in index"
            )
        
        summary_lines.append(f"  [dim]*[/dim] Time: {elapsed:.1f}s")
        
        if index_size_mb:
            summary_lines.append(
                f"  [dim]*[/dim] Index size: {index_size_mb:.1f} MB"
            )
        
        summary_lines.append("")
        summary_lines.append("[dim]Ready to search[/dim]")
        
        summary_text = "\n".join(summary_lines)
        
        panel = Panel(
            summary_text,
            border_style="green",
            title="[bold]SmartFork Index[/bold]",
            title_align="center",
        )
        
        self.console.print()
        self.console.print(panel)
        
        if self.stats.errors and len(self.stats.errors) > 0:
            self.console.print()
            error_table = Table(
                title=f"[yellow]Failed Sessions ({len(self.stats.errors)})[/yellow]",
                show_header=True,
                header_style="bold red"
            )
            error_table.add_column("Session ID", style="cyan", no_wrap=True)
            error_table.add_column("Error", style="red")
            
            for error in self.stats.errors[:10]:
                parts = error.split(": ", 1)
                if len(parts) == 2:
                    error_table.add_row(parts[0][:20], parts[1][:60])
                else:
                    error_table.add_row("unknown", error[:60])
            
            if len(self.stats.errors) > 10:
                error_table.add_row(
                    "", 
                    f"[dim]... and {len(self.stats.errors) - 10} more[/dim]"
                )
            
            self.console.print(error_table)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def index_with_progress(
    indexer,
    sessions: List[Path],
    db,
    console: Optional[Console] = None,
    animated: bool = True
) -> dict:
    """Index sessions with beautiful progress display.
    
    This is a convenience function that orchestrates the full indexing
    workflow with progress indicators.
    
    Args:
        indexer: FullIndexer instance
        sessions: List of session directories
        db: ChromaDatabase instance
        console: Optional Rich console
        animated: Whether to use animated display (vs static)
        
    Returns:
        Statistics dictionary
    """
    if animated:
        display = AnimatedProgressDisplay(console)
    else:
        display = IndexingProgressDisplay(console)
    
    # Discovery phase - get existing session IDs from DB
    db_session_ids = set()
    try:
        db_session_ids = set(db.get_unique_sessions())
    except Exception:
        pass  # Database might be empty
    
    # Display discovery
    display.display_discovery_phase(sessions, db_session_ids)
    
    # Indexing phase
    def index_one(session_dir: Path) -> Tuple[int, Optional[str]]:
        """Index a single session and return chunks + title."""
        chunks = indexer.index_session(session_dir)
        
        title = None
        try:
            session_chunks = db.get_session_chunks(session_dir.name)
            if session_chunks and len(session_chunks) > 0:
                title = session_chunks[0].metadata.session_title
        except Exception:
            pass
        
        return chunks, title
    
    results = display.display_indexing_progress(sessions, index_one)
    
    # Summary phase
    total_db_sessions = 0
    try:
        total_db_sessions = len(db.get_unique_sessions())
    except Exception:
        pass
    
    display.display_completion_summary(total_db_sessions)
    
    return results


def create_animated_display(console: Optional[Console] = None) -> AnimatedProgressDisplay:
    """Create an animated progress display instance.
    
    Args:
        console: Optional Rich console
        
    Returns:
        AnimatedProgressDisplay instance
    """
    return AnimatedProgressDisplay(console)
