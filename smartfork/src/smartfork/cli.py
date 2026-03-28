"""CLI for SmartFork."""

import sys
# CRITICAL: Configure UTF-8 BEFORE any Rich imports on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import typer
from pathlib import Path
from typing import Optional, List
from datetime import datetime, timedelta
from rich import box
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.tree import Tree
from loguru import logger
import sys
import json
import time

from .config import get_config, CONFIG_FILE
from .database.chroma_db import ChromaDatabase
from .indexer.indexer import FullIndexer, IncrementalIndexer
from .indexer.watcher import TranscriptWatcher
from .search.hybrid import HybridSearchEngine
from .fork.generator import ForkMDGenerator
from .fork.smart_generator import SmartForkMDGenerator, ContextExtractionConfig
from .intelligence.pre_compaction import CompactionManager
from .intelligence.clustering import SessionClusterer
from .intelligence.branching import BranchingTree
from .intelligence.privacy import PrivacyVault
from .intelligence.titling import TitleManager, TitleGenerator
from .indexer.parser import KiloCodeParser
from .ui.progress import (
    SmartForkProgress, display_discovery_phase,
    display_completion_summary, THEMES, DEFAULT_THEME,
    get_theme_colors, get_semantic_color,
    set_animation_fps, set_adaptive_fps,
)
from .ui.contextual_help import ContextualHelpManager, UserAction, get_help_manager
from .ui.interactive import start_interactive_shell

app = typer.Typer(help="SmartFork - AI Session Intelligence for Kilo Code")
console = Console()

# Initialize contextual help manager
help_manager = get_help_manager(console)


def setup_logging(log_level: str, log_file: Optional[Path] = None):
    """Setup logging configuration."""
    logger.remove()
    
    # Console logging
    logger.add(
        sys.stderr,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    
    # File logging
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_file,
            level=log_level,
            rotation="10 MB",
            retention="1 week"
        )


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
    log_file: Optional[Path] = typer.Option(None, "--log-file", help="Log file path"),
    lite: bool = typer.Option(False, "--lite", "-l", help="Lite mode - minimal resource usage"),
):
    """SmartFork - AI Session Intelligence for Kilo Code."""
    config = get_config()
    
    # Apply lite mode settings if enabled
    if lite:
        config.lite_mode = True
        config.disable_animations = True
        config.animation_fps = 5
        console.print(f"[dim]Lite mode enabled - reduced CPU/animations[/dim]")
    
    # Configure animation settings
    set_animation_fps(config.get_effective_fps())
    set_adaptive_fps(config.adaptive_fps and not config.disable_animations)
    
    log_level = "DEBUG" if verbose else config.log_level
    setup_logging(log_level, log_file)
    
    # Launch interactive shell if no subcommand provided
    if ctx.invoked_subcommand is None:
        theme_name = getattr(config, "theme", DEFAULT_THEME)
        theme = get_theme_colors(theme_name)
        semantic = theme.get("semantic", {})
        info_color = semantic.get("info", theme["text_primary"])
        error_color = semantic.get("error", "#EF4444")
        
        console.print(Panel.fit(
            f"[bold {info_color}]SmartFork Interactive Mode[/bold {info_color}]\n"
            "Starting interactive shell...",
            title="SmartFork",
            border_style=theme["panel_border"]
        ))
        
        try:
            start_interactive_shell()
        except Exception as e:
            console.print(f"Error starting interactive shell: {str(e)}", style=error_color, markup=False)
            raise typer.Exit(1)


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
    if len(path_str) > 58:
        path_str = "…" + path_str[-55:]
    hdr = Text()
    hdr.append("⚡ SmartFork ", style=f"bold {theme['bars'][0]['color']}")
    hdr.append("Indexer\n",    style=f"bold {theme['text_primary']}")
    hdr.append(f"  {path_str}", style=f"dim {theme['text_muted']}")
    console.print(Panel(hdr, border_style=theme["panel_border"], box=box.ROUNDED, padding=(0, 2)))
    console.print()

    if not config.kilo_code_tasks_path.exists():
        console.print(f"[red]Tasks path not found:[/red] {config.kilo_code_tasks_path}")
        raise typer.Exit(1)

    db = ChromaDatabase(config.chroma_db_path)
    if force:
        console.print(f"[{theme['text_muted']}]Resetting database...[/{theme['text_muted']}]")
        db.reset()

    db_session_ids = set()
    try:
        db_session_ids = set(db.get_unique_sessions())
    except Exception:
        pass

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
        if watch:
            _start_watch_mode(config, db, console, theme)
        raise typer.Exit(0)

    console.print(
        f"  [{theme['bars'][1]['color']}]→[/{theme['bars'][1]['color']}] "
        f"Indexing {len(sessions_to_index)} sessions...\n"
    )

    indexer = FullIndexer(
        db, 
        chunk_size=config.chunk_size, 
        chunk_overlap=config.chunk_overlap,
        batch_size=config.batch_size
    )
    final_stats = None

    with SmartForkProgress(
        total_sessions=len(sessions_to_index),
        theme_name=theme_name,
        console=console,
        animation_fps=config.get_effective_fps(),
        disable_animation=config.disable_animations,
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
                    if sc:
                        title = sc[0].metadata.session_title or ""
                except Exception:
                    pass
                if title:
                    prog.set_session(sid, title=title)

                prog.add_chunks(chunks)
                prog.advance()

            except Exception as e:
                logger.error(f"Failed to index {sid}: {e}")
                prog.add_error()
                prog.advance()

            prog.set_bm25((i+1) / len(sessions_to_index))

        prog.finish()
        final_stats = prog._stats

    # CRITICAL: Finalize to flush any remaining pending chunks to database
    indexer.finalize()

    # Completion
    total_db = 0
    try:
        total_db = len(db.get_unique_sessions())
    except Exception:
        pass

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

    if watch:
        _start_watch_mode(config, db, console, theme)


def _start_watch_mode(config, db, console, theme):
    """Helper to start watch mode after indexing."""
    console.print(
        f"\n  [{theme['bars'][0]['color']}]◉[/{theme['bars'][0]['color']}] "
        f"Watch mode. Ctrl+C to stop.\n"
    )
    
    # Use longer poll interval in lite mode
    poll_interval = 10.0 if config.lite_mode else 5.0
    if config.lite_mode:
        console.print(f"  [dim]Lite mode: using {poll_interval}s poll interval[/dim]\n")
    
    incremental = IncrementalIndexer(db)
    watcher = TranscriptWatcher(
        config.kilo_code_tasks_path, 
        incremental.on_session_changed,
        poll_interval=poll_interval
    )
    watcher.start()
    try:
        while True:
            # Longer sleep in lite mode to reduce CPU
            sleep_interval = 2.0 if config.lite_mode else 1.0
            time.sleep(sleep_interval)
    except KeyboardInterrupt:
        console.print(f"\n  [{theme['text_muted']}]Watcher stopped.[/{theme['text_muted']}]")
        watcher.stop()


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    n_results: int = typer.Option(5, "--results", "-n", help="Number of results"),
    path: Optional[Path] = typer.Option(None, "--path", "-p", help="Current directory for path matching"),
    json_output: bool = typer.Option(False, "--json", "-j", help="Output as JSON"),
):
    """Search indexed sessions using hybrid search (semantic + BM25 + recency + path)."""
    config = get_config()
    theme_name = getattr(config, "theme", DEFAULT_THEME)
    theme = get_theme_colors(theme_name)
    semantic = theme.get("semantic", {})
    
    # Theme-aware colors
    info_color = semantic.get("info", theme["text_primary"])
    success_color = semantic.get("success", theme["done_color"])
    warning_color = semantic.get("warning", "#F59E0B")
    error_color = semantic.get("error", "#EF4444")
    accent_color = semantic.get("accent", theme["text_primary"])
    
    db = ChromaDatabase(config.chroma_db_path)
    engine = HybridSearchEngine(
        db,
        enable_cache=config.enable_search_cache,
        cache_size=config.search_cache_size,
        cache_ttl=config.search_cache_ttl
    )
    
    if db.get_session_count() == 0:
        console.print(f"[{warning_color}]No sessions indexed. Run 'smartfork index' first.[/{warning_color}]")
        raise typer.Exit(1)
    
    console.print(f"[bold {info_color}]Searching for:[/bold {info_color}] {query}\n")
    
    # Use hybrid search with path matching
    current_dir = str(path) if path else str(Path.cwd())
    results = engine.search(query, current_dir=current_dir, n_results=n_results)
    
    if not results:
        console.print(f"[{warning_color}]No results found.[/{warning_color}]")
        return
    
    if json_output:
        output = [r.to_dict() for r in results]
        console.print(json.dumps(output, indent=2, default=str))
    else:
        # Display results with breakdown
        console.print(f"[{theme['text_muted']}]Found {len(results)} results (using hybrid search)[/{theme['text_muted']}]\n")
        
        for i, r in enumerate(results, 1):
            score_pct = f"{r.score:.1%}"
            breakdown = r.breakdown
            
            # Build breakdown string
            breakdown_str = " | ".join([
                f"sem:{breakdown.get('semantic', 0):.2f}",
                f"bm25:{breakdown.get('bm25', 0):.2f}",
                f"rec:{breakdown.get('recency', 0):.2f}",
                f"path:{breakdown.get('path', 0):.2f}"
            ])
            
            # Get files in context
            files = r.metadata.get("files_in_context", [])
            files_str = "\n".join([f"  [{theme['text_muted']}]* {f}[/{theme['text_muted']}]" for f in files[:3]]) if files else ""
            
            # Get last active
            last_active = r.metadata.get("last_active", "Unknown")
            if last_active and last_active != "Unknown":
                try:
                    dt = datetime.fromisoformat(last_active)
                    last_active = dt.strftime("%Y-%m-%d")
                except:
                    pass
            
            # Theme-aware border colors based on score
            if r.score > 0.7:
                border = success_color
            elif r.score > 0.4:
                border = warning_color
            else:
                border = error_color
            
            panel_content = f"[bold {info_color}]Score:[/bold {info_color}] {score_pct}\n"
            panel_content += f"[{theme['text_muted']}]{breakdown_str}[/{theme['text_muted']}]\n"
            panel_content += f"[{theme['text_muted']}]Last active: {last_active}[/{theme['text_muted']}]\n"
            if files_str:
                panel_content += f"\n[bold {info_color}]Files:[/bold {info_color}]\n{files_str}"
            
            # Get session title if available
            session_title = r.metadata.get("session_title")
            if session_title:
                title_text = f"[{i}] {session_title}"
            else:
                title_text = f"[{i}] Session {r.session_id[:16]}..."
            
            console.print(Panel(
                panel_content,
                title=title_text,
                border_style=border
            ))
        
        # Show contextual help
        help_manager.show_after_command(
            UserAction.SEARCH,
            db_session_count=db.get_session_count(),
            query=query,
            result_count=len(results)
        )


@app.command()
def detect_fork(
    query: str = typer.Argument(..., help="Describe what you're working on"),
    n_results: int = typer.Option(5, "--results", "-n", help="Number of suggestions"),
    path: Optional[Path] = typer.Option(None, "--path", "-p", help="Current working directory for path matching"),
    days: int = typer.Option(90, "--days", "-d", help="Limit to sessions from last N days"),
    json_output: bool = typer.Option(False, "--json", "-j", help="Output as JSON"),
):
    """Find relevant past sessions to fork context from."""
    config = get_config()
    theme_name = getattr(config, "theme", DEFAULT_THEME)
    theme = get_theme_colors(theme_name)
    semantic = theme.get("semantic", {})
    
    # Theme-aware colors
    info_color = semantic.get("info", theme["text_primary"])
    success_color = semantic.get("success", theme["done_color"])
    warning_color = semantic.get("warning", "#F59E0B")
    error_color = semantic.get("error", "#EF4444")
    accent_color = semantic.get("accent", theme["text_primary"])
    
    db = ChromaDatabase(config.chroma_db_path)
    engine = HybridSearchEngine(
        db,
        enable_cache=config.enable_search_cache,
        cache_size=config.search_cache_size,
        cache_ttl=config.search_cache_ttl
    )
    
    if db.get_session_count() == 0:
        console.print(f"[{warning_color}]No sessions indexed. Run 'smartfork index' first.[/{warning_color}]")
        raise typer.Exit(1)
    
    console.print(Panel.fit(
        f"[bold {info_color}]Detecting fork for:[/bold {info_color}] {query}",
        title="SmartFork Detect-Fork",
        border_style=theme["panel_border"]
    ))
    
    # Use hybrid search
    current_dir = str(path) if path else str(Path.cwd())
    results = engine.search(query, current_dir=current_dir, n_results=n_results * 2)
    
    # Filter by recency if specified
    if days < 90:
        cutoff = datetime.now() - timedelta(days=days)
        filtered = []
        for r in results:
            last_active = r.metadata.get("last_active")
            if last_active:
                try:
                    dt = datetime.fromisoformat(last_active)
                    if dt > cutoff:
                        filtered.append(r)
                except:
                    filtered.append(r)
            else:
                filtered.append(r)
        results = filtered
    
    if not results:
        console.print(f"[{warning_color}]No relevant sessions found.[/{warning_color}]")
        return
    
    # Limit results
    results = results[:n_results]
    
    if json_output:
        output = [r.to_dict() for r in results]
        console.print(json.dumps(output, indent=2, default=str))
    else:
        console.print(f"\n[{theme['text_muted']}]Found {len(results)} relevant session(s):[/{theme['text_muted']}]\n")
        
        for i, r in enumerate(results, 1):
            score_pct = f"{r.score:.1%}"
            breakdown = r.breakdown
            
            # Get files
            files = r.metadata.get("files_in_context", [])
            files_preview = f"\n[{theme['text_muted']}]Files:[/{theme['text_muted']}] {', '.join(files[:3])}..." if len(files) > 3 else f"\n[{theme['text_muted']}]Files:[/{theme['text_muted']}] {', '.join(files)}" if files else ""
            
            # Last active
            last_active = r.metadata.get("last_active", "Unknown")
            if last_active and last_active != "Unknown":
                try:
                    dt = datetime.fromisoformat(last_active)
                    last_active = dt.strftime("%Y-%m-%d")
                except:
                    pass
            
            match_reasons = []
            if breakdown.get('semantic', 0) > 0.5:
                match_reasons.append("semantic")
            if breakdown.get('bm25', 0) > 0.5:
                match_reasons.append("keyword")
            if breakdown.get('recency', 0) > 0.8:
                match_reasons.append("recent")
            if breakdown.get('path', 0) > 0.3:
                match_reasons.append("same project")
            
            match_str = f" ({', '.join(match_reasons)})" if match_reasons else ""
            
            # Get session title if available
            session_title = r.metadata.get("session_title")
            if session_title:
                title_text = f"[{i}] {session_title}"
            else:
                title_text = f"[{i}] {r.session_id[:20]}..."
            
            # Also show session ID in dim text if title is shown
            id_text = f"\n[{theme['text_muted']}]ID: {r.session_id[:20]}...[/{theme['text_muted']}]" if session_title else ""
            
            console.print(Panel(
                f"[bold {success_color}]{score_pct}[/bold {success_color}] relevance{match_str}{files_preview}\n\n"
                f"[{theme['text_muted']}]Last active: {last_active}[/{theme['text_muted']}]{id_text}",
                title=title_text,
                border_style=theme["panel_border"]
            ))
        
        console.print(f"\n[{theme['text_muted']}]Use [bold]smartfork fork <session_id>[/bold] to generate context file[/{theme['text_muted']}]")
        
        # Show contextual help
        help_manager.show_after_command(
            UserAction.DETECT_FORK,
            db_session_count=db.get_session_count(),
            query=query,
            result_count=len(results)
        )


@app.command()
def fork(
    session_id: str = typer.Argument(..., help="Session ID to fork context from"),
    query: str = typer.Option("", "--query", "-q", help="Original search query for context"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file path"),
    path: Optional[Path] = typer.Option(None, "--path", "-p", help="Current working directory"),
    smart: bool = typer.Option(False, "--smart", help="Use query-aware smart fork generation"),
    max_tokens: int = typer.Option(2000, "--max-tokens", "-t", help="Maximum tokens in output (smart mode only)"),
):
    """Generate a fork.md context file from a session."""
    config = get_config()
    theme_name = getattr(config, "theme", DEFAULT_THEME)
    theme = get_theme_colors(theme_name)
    semantic = theme.get("semantic", {})

    # Theme-aware colors
    info_color = semantic.get("info", theme["text_primary"])
    success_color = semantic.get("success", theme["done_color"])
    warning_color = semantic.get("warning", "#F59E0B")
    error_color = semantic.get("error", "#EF4444")
    accent_color = semantic.get("accent", theme["text_primary"])

    db = ChromaDatabase(config.chroma_db_path)

    if db.get_session_count() == 0:
        console.print(f"[{warning_color}]No sessions indexed. Run 'smartfork index' first.[/{warning_color}]")
        raise typer.Exit(1)

    # Check if session exists
    chunks = db.get_session_chunks(session_id)
    if not chunks:
        console.print(f"[{error_color}]Session not found: {session_id}[/{error_color}]")
        raise typer.Exit(1)

    # Get session title if available
    session_title = chunks[0].metadata.session_title if chunks else None
    if session_title:
        console.print(f"[bold {info_color}]Generating fork.md for session:[/bold {info_color}] {session_title}")
        console.print(f"[{theme['text_muted']}]ID: {session_id[:20]}...[/{theme['text_muted']}]")
    else:
        console.print(f"[bold {info_color}]Generating fork.md for session:[/bold {info_color}] {session_id[:20]}...")

    # Use smart generator if --smart flag is set
    current_dir = str(path) if path else str(Path.cwd())

    if smart:
        if not query:
            console.print(f"[{error_color}]Error: --query is required when using --smart mode[/{error_color}]")
            console.print(f"[{theme['text_muted']}]Usage: smartfork fork <session_id> --smart --query <query>[/{theme['text_muted']}]")
            raise typer.Exit(1)

        generator = SmartForkMDGenerator(db)
        content = generator.generate(
            session_id=session_id,
            query=query,
            current_dir=current_dir,
            max_tokens=max_tokens
        )
        console.print(f"[{theme['text_muted']}]Using smart query-aware generation[/{theme['text_muted']}]")
    else:
        generator = ForkMDGenerator(db)
        content = generator.generate(session_id, query or "forked session", current_dir)

    # Save to file
    if not output:
        short_id = session_id[:8]
        output = Path(f"fork_{short_id}.md")

    output.write_text(content, encoding="utf-8")

    console.print(f"[{success_color}]OK Fork.md saved to:[/{success_color}] {output.absolute()}")
    console.print(f"\n[{theme['text_muted']}]Preview:[/{theme['text_muted']}]\n{content[:500]}...")

    # Show contextual help
    help_manager.show_after_command(
        UserAction.FORK,
        db_session_count=db.get_session_count(),
        session_id=session_id,
        output_file=str(output)
    )


@app.command()
def resume(
    session: str = typer.Option(..., "--session", "-s", help="Session ID to resume from"),
    query: str = typer.Option(..., "--query", "-q", help="Query describing what you want to retrieve"),
    max_tokens: int = typer.Option(2000, "--max-tokens", "-t", help="Maximum tokens in output"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file path (optional)"),
    path: Optional[Path] = typer.Option(None, "--path", "-p", help="Current working directory"),
):
    """Generate a smart context fork from a previous session for resuming work."""
    config = get_config()
    theme_name = getattr(config, "theme", DEFAULT_THEME)
    theme = get_theme_colors(theme_name)
    semantic = theme.get("semantic", {})

    # Theme-aware colors
    info_color = semantic.get("info", theme["text_primary"])
    success_color = semantic.get("success", theme["done_color"])
    warning_color = semantic.get("warning", "#F59E0B")
    error_color = semantic.get("error", "#EF4444")
    accent_color = semantic.get("accent", theme["text_primary"])

    db = ChromaDatabase(config.chroma_db_path)

    if db.get_session_count() == 0:
        console.print(f"[{warning_color}]No sessions indexed. Run 'smartfork index' first.[/{warning_color}]")
        raise typer.Exit(1)

    # Check if session exists
    chunks = db.get_session_chunks(session)
    if not chunks:
        console.print(f"[{error_color}]Session not found: {session}[/{error_color}]")
        raise typer.Exit(1)

    # Get session title if available
    session_title = chunks[0].metadata.session_title if chunks else None

    console.print(Panel.fit(
        f"[bold {info_color}]Generating smart context fork[/bold {info_color}]\n"
        f"Query: {query}",
        title=f"Resume: {session_title or session[:20]}",
        border_style=theme["panel_border"]
    ))

    # Use SmartForkMDGenerator for query-aware context extraction
    generator = SmartForkMDGenerator(db)
    current_dir = str(path) if path else str(Path.cwd())

    with console.status(f"[bold {info_color}]Extracting relevant context..."):
        content = generator.generate(
            session_id=session,
            query=query,
            current_dir=current_dir,
            max_tokens=max_tokens
        )

    # Calculate token savings info
    total_chunks = len(chunks)
    relevant_chunks = content.count("### Exchange")  # Count exchange sections

    # Save or display output
    if output:
        output_path = generator.save(
            session_id=session,
            query=query,
            output_path=output,
            current_dir=current_dir,
            max_tokens=max_tokens
        )
        console.print(f"[{success_color}]OK Smart context fork saved to:[/{success_color}] {output_path.absolute()}")
    else:
        # Generate default filename
        short_id = session[:8]
        output_path = Path(f"fork_{short_id}.md")
        output_path.write_text(content, encoding="utf-8")
        console.print(f"[{success_color}]OK Smart context fork saved to:[/{success_color}] {output_path.absolute()}")

    # Show token efficiency info
    console.print(f"\n[bold {info_color}]Context Summary:[/bold {info_color}]")
    console.print(f"  [{theme['text_muted']}]Total chunks in session:[/{theme['text_muted']}] {total_chunks}")
    console.print(f"  [{success_color}]Relevant chunks retrieved:[/{success_color}] ~{relevant_chunks}")
    console.print(f"  [{theme['text_muted']}]Token budget:[/{theme['text_muted']}] {max_tokens}")

    console.print(f"\n[{theme['text_muted']}]Preview:[/{theme['text_muted']}]")
    console.print(Panel(
        content[:800] + "..." if len(content) > 800 else content,
        border_style=theme["panel_border"]
    ))


@app.command()
def status():
    """Show indexing status."""
    config = get_config()
    theme_name = getattr(config, "theme", DEFAULT_THEME)
    theme = get_theme_colors(theme_name)
    semantic = theme.get("semantic", {})
    
    db = ChromaDatabase(config.chroma_db_path)
    
    # Get stats
    total_chunks = db.get_session_count()
    unique_sessions = len(db.get_unique_sessions())
    
    # Check tasks directory
    if config.kilo_code_tasks_path.exists():
        task_dirs = [d for d in config.kilo_code_tasks_path.iterdir() if d.is_dir()]
        total_tasks = len(task_dirs)
    else:
        total_tasks = 0
    
    # Theme-aware colors
    info_color = semantic.get("info", theme["text_primary"])
    success_color = semantic.get("success", theme["done_color"])
    warning_color = semantic.get("warning", "#F59E0B")
    
    # Display status
    console.print(Panel.fit(
        f"[bold {theme['text_primary']}]SmartFork Status[/bold {theme['text_primary']}]",
        title="Status",
        border_style=theme["panel_border"]
    ))
    
    table = Table(show_header=False)
    table.add_column("Property", style=info_color)
    table.add_column("Value", style=success_color)
    
    table.add_row("Kilo Code Tasks Path", str(config.kilo_code_tasks_path))
    table.add_row("Database Path", str(config.chroma_db_path))
    table.add_row("Total Task Directories", str(total_tasks))
    table.add_row("Indexed Sessions", str(unique_sessions))
    table.add_row("Total Chunks", str(total_chunks))
    table.add_row("Index Coverage", f"{unique_sessions}/{total_tasks} ({unique_sessions/max(1,total_tasks)*100:.1f}%)")
    
    console.print(table)
    
    if unique_sessions < total_tasks:
        console.print(f"\n[{warning_color}]Tip: Run 'smartfork index' to index remaining sessions[/{warning_color}]")
    
    # Show contextual help
    help_manager.show_after_command(
        UserAction.STATUS,
        db_session_count=unique_sessions,
        total_tasks=total_tasks,
        indexed_sessions=unique_sessions
    )


@app.command("session-list")
def session_list(
    limit: int = typer.Option(20, "--limit", "-n", help="Maximum number of sessions to show"),
    json_output: bool = typer.Option(False, "--json", "-j", help="Output as JSON"),
):
    """List all indexed sessions with IDs and titles."""
    config = get_config()
    theme_name = getattr(config, "theme", DEFAULT_THEME)
    theme = get_theme_colors(theme_name)
    semantic = theme.get("semantic", {})
    
    # Theme-aware colors
    info_color = semantic.get("info", theme["text_primary"])
    success_color = semantic.get("success", theme["done_color"])
    warning_color = semantic.get("warning", "#F59E0B")
    accent_color = semantic.get("accent", theme["text_primary"])
    
    db = ChromaDatabase(config.chroma_db_path)
    
    if db.get_session_count() == 0:
        console.print(f"[{warning_color}]No sessions indexed. Run 'smartfork index' first.[/{warning_color}]")
        raise typer.Exit(1)
    
    # Get all unique sessions
    session_ids = db.get_unique_sessions()
    
    if not session_ids:
        console.print(f"[{warning_color}]No sessions found in database.[/{warning_color}]")
        raise typer.Exit(0)
    
    # Gather session info with titles
    sessions_info = []
    for session_id in session_ids:
        try:
            chunks = db.get_session_chunks(session_id)
            if chunks:
                # Get title from first chunk's metadata
                title = chunks[0].metadata.session_title or "Untitled"
                # Get last active timestamp
                last_active = chunks[0].metadata.last_active or "Unknown"
                # Count chunks for this session
                chunk_count = len(chunks)
                sessions_info.append({
                    "id": session_id,
                    "title": title,
                    "last_active": last_active,
                    "chunks": chunk_count
                })
        except Exception as e:
            logger.warning(f"Failed to get info for session {session_id}: {e}")
            sessions_info.append({
                "id": session_id,
                "title": "Error loading",
                "last_active": "Unknown",
                "chunks": 0
            })
    
    # Sort by last active (most recent first)
    sessions_info.sort(key=lambda x: x["last_active"] if x["last_active"] != "Unknown" else "", reverse=True)
    
    if json_output:
        output = {
            "total": len(sessions_info),
            "sessions": sessions_info[:limit]
        }
        console.print(json.dumps(output, indent=2, default=str))
        return
    
    # Display header
    console.print(Panel.fit(
        f"[bold {info_color}]{len(sessions_info)} indexed sessions[/bold {info_color}]",
        title="Session List",
        border_style=theme["panel_border"]
    ))
    
    # Create table
    table = Table(show_header=True)
    table.add_column("#", style=accent_color, justify="right", width=4)
    table.add_column("Session ID", style=info_color, width=12)
    table.add_column("Title", style=success_color, min_width=30)
    table.add_column("Last Active", style=theme["text_muted"], width=12)
    table.add_column("Chunks", style=accent_color, justify="right", width=8)
    
    for i, session in enumerate(sessions_info[:limit], 1):
        # Format short ID (first 8 chars)
        short_id = session["id"][:8]
        full_id = session["id"]
        
        # Create clickable text using Rich hyperlink
        # Clicking copies the full ID (supported in modern terminals)
        clickable_id = f"[link=copy:{full_id}]{short_id}[/link]"
        
        # Format last active
        last_active = session["last_active"]
        if last_active and last_active != "Unknown":
            try:
                dt = datetime.fromisoformat(last_active)
                last_active = dt.strftime("%Y-%m-%d")
            except:
                pass
        
        table.add_row(
            str(i),
            clickable_id,
            session["title"],
            last_active,
            str(session["chunks"])
        )
    
    console.print(table)
    
    if len(sessions_info) > limit:
        console.print(f"\n[{theme['text_muted']}]... and {len(sessions_info) - limit} more sessions (use --limit to show more)[/{theme['text_muted']}]")
    
    console.print(f"\n[{theme['text_muted']}]Tip: Click the Session ID to copy full ID, or use 'smartfork fork <number>'[/{theme['text_muted']}]")


@app.command()
def config_show():
    """Show current configuration."""
    config = get_config()
    theme_name = getattr(config, "theme", DEFAULT_THEME)
    theme = get_theme_colors(theme_name)
    semantic = theme.get("semantic", {})
    
    info_color = semantic.get("info", theme["text_primary"])
    success_color = semantic.get("success", theme["done_color"])

    console.print(Panel.fit(
        f"[bold {theme['text_primary']}]SmartFork Configuration[/bold {theme['text_primary']}]",
        title="Config",
        border_style=theme["panel_border"]
    ))

    table = Table(show_header=False)
    table.add_column("Setting", style=info_color)
    table.add_column("Value", style=success_color)

    for key, value in config.model_dump().items():
        table.add_row(key, str(value))

    console.print(table)


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
    from .config import reload_config
    config  = get_config()
    current = getattr(config, "theme", DEFAULT_THEME)

    if list_all or theme_name is None:
        tbl = Table(show_header=True, box=box.SIMPLE)
        tbl.add_column("Theme",  style="bold", width=12)
        tbl.add_column("Description", style="dim", width=40)
        tbl.add_column("",  width=10)
        for tid, td in THEMES.items():
            c0, c1, c2 = [b["color"] for b in td["bars"]]
            swatch = f"[{c0}]▪[/{c0}][{c1}]▪[/{c1}][{c2}]▪[/{c2}] {td['name']}"
            status = "[green]● active[/green]" if tid == current else ""
            tbl.add_row(swatch, td["desc"], status)
        console.print(Panel(tbl, title="[bold]SmartFork Themes[/bold]", box=box.ROUNDED))
        if theme_name is None:
            console.print(f"\n  Current: [bold]{current}[/bold]")
            console.print(f"  Set with: [dim]smartfork config-theme <name>[/dim]\n")
        return

    theme_name = theme_name.lower()
    if theme_name not in THEMES:
        console.print(f"[red]Unknown theme '{theme_name}'[/red]")
        console.print(f"[dim]Valid: {', '.join(THEMES.keys())}[/dim]")
        raise typer.Exit(1)

    # Update and save config
    config.theme = theme_name
    config.save()

    td = THEMES[theme_name]
    c  = td["bars"][1]["color"]
    console.print(f"\n  [{c}]✓[/{c}] Theme → [bold]{td['name']}[/bold] — {td['desc']}")
    console.print(f"  [dim]Saved to {CONFIG_FILE}[/dim]\n")


@app.command()
def reset(
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation")
):
    """Reset the database (WARNING: deletes all indexed data)."""
    config = get_config()
    theme_name = getattr(config, "theme", DEFAULT_THEME)
    theme = get_theme_colors(theme_name)
    semantic = theme.get("semantic", {})
    
    # Theme-aware colors
    success_color = semantic.get("success", theme["done_color"])
    warning_color = semantic.get("warning", "#F59E0B")
    
    if not force:
        confirm = typer.confirm("Are you sure you want to delete all indexed data?")
        if not confirm:
            console.print(f"[{warning_color}]Aborted.[/{warning_color}]")
            raise typer.Exit(0)
    
    db = ChromaDatabase(config.chroma_db_path)
    
    db.reset()
    console.print(f"[{success_color}]Database reset complete.[/{success_color}]")


@app.command()
def watch():
    """Watch for session changes and index incrementally."""
    config = get_config()
    theme_name = getattr(config, "theme", DEFAULT_THEME)
    theme = get_theme_colors(theme_name)
    semantic = theme.get("semantic", {})
    
    # Theme-aware colors
    info_color = semantic.get("info", theme["text_primary"])
    success_color = semantic.get("success", theme["done_color"])
    warning_color = semantic.get("warning", "#F59E0B")
    error_color = semantic.get("error", "#EF4444")
    
    if not config.kilo_code_tasks_path.exists():
        console.print(f"[{error_color}]Error: Tasks path does not exist: {config.kilo_code_tasks_path}[/{error_color}]")
        raise typer.Exit(1)
    
    console.print(f"[bold {info_color}]Starting watcher... Press Ctrl+C to stop.[/{info_color}]\n")
    
    db = ChromaDatabase(config.chroma_db_path)
    incremental = IncrementalIndexer(db)
    watcher = TranscriptWatcher(
        config.kilo_code_tasks_path,
        incremental.on_session_changed
    )
    
    watcher.start()
    
    try:
        while True:
            import time
            time.sleep(1)
    except KeyboardInterrupt:
        console.print(f"\n[{warning_color}]Stopping watcher...[/{warning_color}]")
        watcher.stop()
        console.print(f"[{success_color}]Watcher stopped.[/{success_color}]")


# Phase 2: Intelligence Layer Commands

@app.command()
def compaction_check(
    threshold_messages: int = typer.Option(100, "--messages", "-m", help="Message count threshold"),
    threshold_days: int = typer.Option(7, "--days", "-d", help="Age threshold in days"),
):
    """Check for sessions at risk of compaction."""
    from .intelligence.pre_compaction import PreCompactionHook
    
    config = get_config()
    theme_name = getattr(config, "theme", DEFAULT_THEME)
    theme = get_theme_colors(theme_name)
    semantic = theme.get("semantic", {})
    
    # Theme-aware colors
    info_color = semantic.get("info", theme["text_primary"])
    success_color = semantic.get("success", theme["done_color"])
    warning_color = semantic.get("warning", "#F59E0B")
    error_color = semantic.get("error", "#EF4444")
    accent_color = semantic.get("accent", theme["text_primary"])
    
    hook = PreCompactionHook(threshold_messages, threshold_days)
    
    with console.status(f"[bold {info_color}]Checking sessions..."):
        at_risk = hook.check_sessions(config.kilo_code_tasks_path)
    
    if not at_risk:
        console.print(f"[{success_color}]No sessions at risk of compaction.[/{success_color}]")
        return
    
    console.print(Panel.fit(
        f"[bold {warning_color}]{len(at_risk)} sessions at risk[/bold {warning_color}]",
        title="Compaction Check",
        border_style=theme["panel_border"]
    ))
    
    table = Table(show_header=True)
    table.add_column("Session", style=info_color)
    table.add_column("Messages", style=warning_color, justify="right")
    table.add_column("Age (days)", style=accent_color, justify="right")
    table.add_column("Risk", style=error_color)
    
    for session in at_risk[:20]:  # Show top 20
        table.add_row(
            session["session_id"][:20],
            str(session["message_count"]),
            str(session["age_days"]),
            session["risk_level"]
        )
    
    console.print(table)
    
    if len(at_risk) > 20:
        console.print(f"\n[{theme['text_muted']}]... and {len(at_risk) - 20} more[/{theme['text_muted']}]")


@app.command()
def compaction_export(
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be exported"),
    auto: bool = typer.Option(False, "--auto", "-a", help="Export all at-risk sessions"),
):
    """Export sessions before compaction."""
    from .intelligence.pre_compaction import CompactionManager
    
    config = get_config()
    theme_name = getattr(config, "theme", DEFAULT_THEME)
    theme = get_theme_colors(theme_name)
    semantic = theme.get("semantic", {})
    
    # Theme-aware colors
    info_color = semantic.get("info", theme["text_primary"])
    success_color = semantic.get("success", theme["done_color"])
    warning_color = semantic.get("warning", "#F59E0B")
    error_color = semantic.get("error", "#EF4444")
    accent_color = semantic.get("accent", theme["text_primary"])
    
    manager = CompactionManager()
    
    with console.status(f"[bold {info_color}]Running compaction export..."):
        results = manager.run_auto_export(dry_run=dry_run)
    
    if dry_run:
        console.print(Panel.fit(
            f"[bold {info_color}]{results['at_risk']} sessions would be exported[/bold {info_color}]",
            title="Dry Run",
            border_style=theme["panel_border"]
        ))
    else:
        console.print(Panel.fit(
            f"[bold {success_color}]Exported {results['exported']} sessions[/bold {success_color}]\n"
            f"Failed: {results['failed']}",
            title="Compaction Export",
            border_style=theme["panel_border"]
        ))
    
    if results['sessions']:
        table = Table(show_header=True)
        table.add_column("Session", style=info_color)
        table.add_column("Action", style=success_color)
        
        for session in results['sessions'][:10]:
            table.add_row(
                session["session_id"][:20],
                session["action"]
            )
        
        console.print(table)


@app.command()
def cluster_analysis():
    """Analyze session clusters and find duplicates."""
    from .intelligence.clustering import SessionClusterer
    
    config = get_config()
    theme_name = getattr(config, "theme", DEFAULT_THEME)
    theme = get_theme_colors(theme_name)
    semantic = theme.get("semantic", {})
    
    # Theme-aware colors
    info_color = semantic.get("info", theme["text_primary"])
    success_color = semantic.get("success", theme["done_color"])
    warning_color = semantic.get("warning", "#F59E0B")
    error_color = semantic.get("error", "#EF4444")
    accent_color = semantic.get("accent", theme["text_primary"])
    
    clusterer = SessionClusterer()
    
    with console.status(f"[bold {info_color}]Analyzing clusters..."):
        analysis = clusterer.analyze_clusters()
    
    console.print(Panel.fit(
        f"[bold {info_color}]{analysis['total_clusters']} clusters found[/bold {info_color}]\n"
        f"{analysis['noise_sessions']} unclustered sessions\n"
        f"{analysis['potential_duplicates']} potential duplicates",
        title="Cluster Analysis",
        border_style=theme["panel_border"]
    ))
    
    if analysis['clusters']:
        table = Table(show_header=True)
        table.add_column("Cluster", style=info_color, justify="right")
        table.add_column("Sessions", style=success_color, justify="right")
        table.add_column("Top Topics", style=accent_color)
        
        for cluster in analysis['clusters'][:10]:
            topics = ", ".join(cluster.get('common_topics', [])[:5]) if cluster.get('common_topics') else "-"
            table.add_row(
                str(cluster['cluster_id']),
                str(cluster['session_count']),
                topics
            )
        
        console.print(f"\n[bold {info_color}]Top Clusters:[/bold {info_color}]")
        console.print(table)
    
    if analysis['duplicate_pairs']:
        console.print(f"\n[bold {warning_color}]Potential Duplicates:[/bold {warning_color}]")
        for a, b, sim in analysis['duplicate_pairs'][:5]:
            console.print(f"  • {a[:16]}... ↔ {b[:16]}... ({sim:.1%})")


@app.command()
def tree_build():
    """Build conversation branching tree."""
    from .intelligence.branching import BranchingTree
    
    config = get_config()
    theme_name = getattr(config, "theme", DEFAULT_THEME)
    theme = get_theme_colors(theme_name)
    semantic = theme.get("semantic", {})
    
    # Theme-aware colors
    info_color = semantic.get("info", theme["text_primary"])
    success_color = semantic.get("success", theme["done_color"])
    accent_color = semantic.get("accent", theme["text_primary"])
    
    tree = BranchingTree()
    
    with console.status(f"[bold {info_color}]Building tree..."):
        tree.auto_build_tree(config.kilo_code_tasks_path)
    
    stats = tree.get_stats()
    
    console.print(Panel.fit(
        f"[bold {info_color}]{stats['total_sessions']} sessions[/bold {info_color}] in tree\n"
        f"{stats['root_sessions']} roots, {stats['leaf_sessions']} leaves\n"
        f"Max depth: {stats['max_depth']}",
        title="Tree Built",
        border_style=theme["panel_border"]
    ))


@app.command()
def tree_visualize(
    session_id: Optional[str] = typer.Option(None, "--session", "-s", help="Root session to visualize"),
    expanded: bool = typer.Option(False, "--expanded", "-e", help="Show expanded view with more details"),
):
    """Visualize conversation branching tree."""
    from .intelligence.branching import BranchingTree
    
    config = get_config()
    theme_name = getattr(config, "theme", DEFAULT_THEME)
    theme = get_theme_colors(theme_name)
    semantic = theme.get("semantic", {})
    
    # Theme-aware colors
    info_color = semantic.get("info", theme["text_primary"])
    accent_color = semantic.get("accent", theme["text_primary"])
    
    tree = BranchingTree()
    stats = tree.get_stats()
    
    # Use compact mode by default for better CLI readability
    tree_text = tree.visualize_tree(session_id, compact=not expanded)
    
    console.print(Panel.fit(
        tree_text,
        title=f"Conversation Tree ({stats['total_sessions']} sessions, {stats['root_sessions']} roots)",
        border_style=theme["panel_border"]
    ))
    
    console.print(f"\n[{theme['text_muted']}]Stats: {stats['leaf_sessions']} leaves, max depth {stats['max_depth']}[/{theme['text_muted']}]")
    console.print(f"[{theme['text_muted']}]Tip: Use --expanded for more details, or 'smartfork tree-export' for interactive HTML[/{theme['text_muted']}]")


@app.command()
def tree_export(
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output HTML file path"),
    open_browser: bool = typer.Option(False, "--open", "-b", help="Open in browser after export"),
):
    """Export conversation tree as interactive HTML."""
    from .intelligence.branching import BranchingTree
    
    config = get_config()
    theme_name = getattr(config, "theme", DEFAULT_THEME)
    theme = get_theme_colors(theme_name)
    semantic = theme.get("semantic", {})
    
    # Theme-aware colors
    info_color = semantic.get("info", theme["text_primary"])
    success_color = semantic.get("success", theme["done_color"])
    accent_color = semantic.get("accent", theme["text_primary"])
    
    tree = BranchingTree()
    
    with console.status(f"[bold {info_color}]Generating HTML visualization..."):
        html_path = tree.export_html(output)
    
    console.print(f"[{success_color}]OK Tree exported to:[/{success_color}] {html_path}")
    
    stats = tree.get_stats()
    console.print(Panel.fit(
        f"[bold {info_color}]{stats['total_sessions']}[/bold {info_color}] sessions\n"
        f"[bold {info_color}]{stats['root_sessions']}[/bold {info_color}] root sessions\n"
        f"[bold {info_color}]{stats['leaf_sessions']}[/bold {info_color}] leaf sessions\n"
        f"Max depth: [bold {info_color}]{stats['max_depth']}[/bold {info_color}]",
        title="Tree Statistics",
        border_style=theme["panel_border"]
    ))
    
    if open_browser:
        import webbrowser
        webbrowser.open(f"file://{html_path.absolute()}")
        console.print(f"[{theme['text_muted']}]Opened in browser[/{theme['text_muted']}]")


@app.command()
def vault_add(
    session_id: str = typer.Argument(..., help="Session ID to add to vault"),
    password: Optional[str] = typer.Option(None, "--password", "-p", help="Encryption password"),
):
    """Add a session to the privacy vault."""
    from .intelligence.privacy import PrivacyVault
    
    config = get_config()
    theme_name = getattr(config, "theme", DEFAULT_THEME)
    theme = get_theme_colors(theme_name)
    semantic = theme.get("semantic", {})
    
    # Theme-aware colors
    info_color = semantic.get("info", theme["text_primary"])
    success_color = semantic.get("success", theme["done_color"])
    error_color = semantic.get("error", "#EF4444")
    
    if not password:
        password = typer.prompt("Enter vault password", hide_input=True)
    
    vault = PrivacyVault(password)
    
    # Find session directory
    task_dir = config.kilo_code_tasks_path / session_id
    if not task_dir.exists():
        console.print(f"[{error_color}]Session not found: {session_id}[/{error_color}]")
        raise typer.Exit(1)
    
    with console.status(f"[bold {info_color}]Adding to vault..."):
        success = vault.add_to_vault(session_id, task_dir)
    
    if success:
        console.print(f"[{success_color}]OK Session {session_id[:16]}... added to vault[/{success_color}]")
    else:
        console.print(f"[{error_color}]Failed to add to vault[/{error_color}]")


@app.command()
def vault_list():
    """List vaulted sessions."""
    from .intelligence.privacy import PrivacyVault
    
    config = get_config()
    theme_name = getattr(config, "theme", DEFAULT_THEME)
    theme = get_theme_colors(theme_name)
    semantic = theme.get("semantic", {})
    
    # Theme-aware colors
    info_color = semantic.get("info", theme["text_primary"])
    success_color = semantic.get("success", theme["done_color"])
    accent_color = semantic.get("accent", theme["text_primary"])
    
    vault = PrivacyVault()
    sessions = vault.list_vaulted_sessions()
    
    if not sessions:
        console.print(f"[{theme['text_muted']}]No sessions in vault[/{theme['text_muted']}]")
        return
    
    console.print(Panel.fit(
        f"[bold {info_color}]{len(sessions)} vaulted sessions[/bold {info_color}]",
        title="Privacy Vault",
        border_style=theme["panel_border"]
    ))
    
    table = Table(show_header=True)
    table.add_column("Session", style=info_color)
    table.add_column("Vaulted At", style=success_color)
    table.add_column("Files", style=accent_color, justify="right")
    
    for session in sessions:
        table.add_row(
            session["session_id"][:20],
            session.get("vaulted_at", "unknown"),
            str(session.get("file_count", 0))
        )
    
    console.print(table)


@app.command()
def vault_restore(
    session_id: str = typer.Argument(..., help="Session ID to restore"),
    password: Optional[str] = typer.Option(None, "--password", "-p", help="Decryption password"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output directory"),
):
    """Restore a session from the vault."""
    from .intelligence.privacy import PrivacyVault
    
    config = get_config()
    theme_name = getattr(config, "theme", DEFAULT_THEME)
    theme = get_theme_colors(theme_name)
    semantic = theme.get("semantic", {})
    
    # Theme-aware colors
    info_color = semantic.get("info", theme["text_primary"])
    success_color = semantic.get("success", theme["done_color"])
    error_color = semantic.get("error", "#EF4444")
    
    if not password:
        password = typer.prompt("Enter vault password", hide_input=True)
    
    vault = PrivacyVault(password)
    
    with console.status(f"[bold {info_color}]Restoring from vault..."):
        result = vault.restore_from_vault(session_id, output)
    
    if result:
        console.print(f"[{success_color}]OK Session restored to: {result}[/{success_color}]")
    else:
        console.print(f"[{error_color}]Failed to restore session[/{error_color}]")


@app.command()
def vault_search(
    query: str = typer.Argument(..., help="Search query"),
    password: Optional[str] = typer.Option(None, "--password", "-p", help="Decryption password"),
):
    """Search within vaulted sessions."""
    from .intelligence.privacy import PrivacyVault
    
    config = get_config()
    theme_name = getattr(config, "theme", DEFAULT_THEME)
    theme = get_theme_colors(theme_name)
    semantic = theme.get("semantic", {})
    
    # Theme-aware colors
    info_color = semantic.get("info", theme["text_primary"])
    accent_color = semantic.get("accent", theme["text_primary"])
    
    if not password:
        password = typer.prompt("Enter vault password", hide_input=True)
    
    vault = PrivacyVault(password)
    
    with console.status(f"[bold {info_color}]Searching vault..."):
        results = vault.search_vault(query)
    
    console.print(Panel.fit(
        f"[bold {info_color}]{len(results)} results[/bold {info_color}]",
        title="Vault Search",
        border_style=theme["panel_border"]
    ))
    
    for r in results[:10]:
        console.print(Panel(
            f"[{theme['text_muted']}]{r['preview']}[/{theme['text_muted']}]",
            title=f"{r['session_id'][:16]}... / {r['file']}",
            border_style=theme["panel_border"]
        ))


# Phase 3: Testing and Metrics Commands

@app.command()
def test(
    suite: Optional[str] = typer.Option(None, "--suite", "-s", help="Test suite to run (indexer, search, database, fork)"),
):
    """Run SmartFork tests."""
    from .testing.test_runner import create_default_test_runner
    
    config = get_config()
    theme_name = getattr(config, "theme", DEFAULT_THEME)
    theme = get_theme_colors(theme_name)
    semantic = theme.get("semantic", {})
    
    # Theme-aware colors
    info_color = semantic.get("info", theme["text_primary"])
    success_color = semantic.get("success", theme["done_color"])
    error_color = semantic.get("error", "#EF4444")
    accent_color = semantic.get("accent", theme["text_primary"])
    
    runner = create_default_test_runner()
    
    if suite:
        with console.status(f"[bold {info_color}]Running {suite} tests..."):
            result = runner.run_suite(suite)
        suites = [result]
    else:
        with console.status(f"[bold {info_color}]Running all tests..."):
            suites = runner.run_all()
    
    # Display results
    for suite in suites:
        suite_color = success_color if suite.failed_count == 0 else error_color
        console.print(Panel.fit(
            f"[bold {suite_color}]{suite.passed_count}/{len(suite.tests)} passed[/bold {suite_color}]\n"
            f"Duration: {suite.total_duration_ms:.0f}ms",
            title=f"Test Suite: {suite.name}",
            border_style=theme["panel_border"]
        ))
        
        if suite.failed_count > 0:
            table = Table(show_header=True)
            table.add_column("Test", style=info_color)
            table.add_column("Status", style=error_color)
            table.add_column("Error", style=theme["text_muted"])
            
            for test in suite.tests:
                if not test.passed:
                    table.add_row(
                        test.name,
                        "FAILED",
                        test.error_message[:50] + "..." if test.error_message and len(test.error_message) > 50 else (test.error_message or "")
                    )
            
            console.print(table)
    
    summary = runner.get_summary()
    console.print(f"\n[{theme['text_muted']}]Total: {summary['passed']}/{summary['total_tests']} passed "
                  f"({summary['pass_rate']:.1%})[/{theme['text_muted']}]")


@app.command()
def metrics(
    days: int = typer.Option(7, "--days", "-d", help="Number of days to show"),
):
    """Show success metrics dashboard."""
    from .testing.metrics_tracker import MetricsTracker
    
    config = get_config()
    theme_name = getattr(config, "theme", DEFAULT_THEME)
    theme = get_theme_colors(theme_name)
    semantic = theme.get("semantic", {})
    
    # Theme-aware colors
    info_color = semantic.get("info", theme["text_primary"])
    success_color = semantic.get("success", theme["done_color"])
    warning_color = semantic.get("warning", "#F59E0B")
    error_color = semantic.get("error", "#EF4444")
    accent_color = semantic.get("accent", theme["text_primary"])
    
    tracker = MetricsTracker()
    data = tracker.get_dashboard_data(days)
    
    console.print(Panel.fit(
        f"[bold {info_color}]Success Metrics[/bold {info_color}] (last {data['period_days']} days)",
        title="Metrics Dashboard",
        border_style=theme["panel_border"]
    ))
    
    # Key metrics
    table = Table(show_header=True)
    table.add_column("Metric", style=info_color)
    table.add_column("Value", style=success_color)
    
    km = data['key_metrics']
    table.add_row("Unique Sessions", str(data['unique_sessions']))
    table.add_row("Avg Fork Gen Time", f"{km['avg_fork_generation_time_ms']:.0f}ms")
    table.add_row("Context Recovered", f"{km['total_context_recovered_mb']:.1f}MB")
    table.add_row("Sessions/Day", f"{km['sessions_per_day']:.1f}")
    
    console.print(table)
    
    # Metric summaries
    if data['metric_summaries']:
        console.print(f"\n[bold {info_color}]Metric Trends:[/bold {info_color}]")
        for name, summary in data['metric_summaries'].items():
            trend_color_map = {
                'improving': success_color,
                'stable': warning_color,
                'degrading': error_color,
                'insufficient_data': theme["text_muted"]
            }
            trend_color = trend_color_map.get(summary['trend'], theme["text_primary"])
            
            console.print(f"  {name}: {summary['mean']:.2f} "
                          f"([{trend_color}]{summary['trend']}[/{trend_color}])")


@app.command()
def ab_test_status():
    """Show A/B test status."""
    from .testing.ab_testing import ABTestManager
    
    config = get_config()
    theme_name = getattr(config, "theme", DEFAULT_THEME)
    theme = get_theme_colors(theme_name)
    semantic = theme.get("semantic", {})
    
    # Theme-aware colors
    info_color = semantic.get("info", theme["text_primary"])
    success_color = semantic.get("success", theme["done_color"])
    warning_color = semantic.get("warning", "#F59E0B")
    accent_color = semantic.get("accent", theme["text_primary"])
    
    manager = ABTestManager()
    summary = manager.get_test_summary()
    
    console.print(Panel.fit(
        f"[bold {info_color}]{summary['total_tests']}[/bold {info_color}] active tests\n"
        f"[bold {info_color}]{summary['total_sessions']}[/bold {info_color}] test sessions",
        title="A/B Testing",
        border_style=theme["panel_border"]
    ))
    
    if summary['active_tests']:
        table = Table(show_header=True)
        table.add_column("Test", style=info_color)
        table.add_column("Sessions", style=success_color, justify="right")
        table.add_column("Control", style=accent_color, justify="right")
        table.add_column("Treatment", style=warning_color, justify="right")
        table.add_column("Result", style=info_color)
        
        for test in summary['active_tests']:
            result_str = "No data"
            if test['result']:
                sig = "sig" if test['result']['significant'] else "not sig"
                result_str = f"{test['result']['improvement_pct']:+.1f}% ({sig})"
            
            table.add_row(
                test['name'],
                str(test['total_sessions']),
                str(test['control']),
                str(test['treatment']),
                result_str
            )
        
        console.print(table)


@app.command()
def update_titles(
    force: bool = typer.Option(False, "--force", "-f", help="Force regeneration of all titles"),
    dry_run: bool = typer.Option(False, "--dry-run", "-d", help="Show what would be generated without updating"),
):
    """Generate or update session titles for all indexed sessions."""
    config = get_config()
    theme_name = getattr(config, "theme", DEFAULT_THEME)
    theme = get_theme_colors(theme_name)
    semantic = theme.get("semantic", {})
    
    # Theme-aware colors
    info_color = semantic.get("info", theme["text_primary"])
    success_color = semantic.get("success", theme["done_color"])
    warning_color = semantic.get("warning", "#F59E0B")
    error_color = semantic.get("error", "#EF4444")
    accent_color = semantic.get("accent", theme["text_primary"])
    
    db = ChromaDatabase(config.chroma_db_path)
    
    if db.get_session_count() == 0:
        console.print(f"[{warning_color}]No sessions indexed. Run 'smartfork index' first.[/{warning_color}]")
        raise typer.Exit(1)
    
    # Get all unique sessions
    session_ids = db.get_unique_sessions()
    
    if not session_ids:
        console.print(f"[{warning_color}]No sessions found in database.[/{warning_color}]")
        raise typer.Exit(0)
    
    console.print(Panel.fit(
        f"[bold {info_color}]Update Session Titles[/bold {info_color}]\n"
        f"Found {len(session_ids)} sessions to process",
        title="SmartFork",
        border_style=theme["panel_border"]
    ))
    
    # Initialize title generator
    title_gen = TitleGenerator()
    title_manager = TitleManager(db, title_gen)
    parser = KiloCodeParser()
    
    # Track results
    updated = 0
    skipped = 0
    failed = 0
    
    # Process each session
    with console.status(f"[bold {info_color}]Generating titles...") as status:
        for i, session_id in enumerate(session_ids):
            try:
                # Check if session already has a title (unless force)
                if not force:
                    chunks = db.get_session_chunks(session_id)
                    if chunks and chunks[0].metadata.session_title:
                        skipped += 1
                        continue
                
                # Parse the session to get full content
                task_dir = config.kilo_code_tasks_path / session_id
                if not task_dir.exists():
                    failed += 1
                    logger.warning(f"Session directory not found: {task_dir}")
                    continue
                
                session = parser.parse_task_directory(task_dir)
                if not session:
                    failed += 1
                    logger.warning(f"Could not parse session: {session_id}")
                    continue
                
                # Generate title
                title = title_manager.generate_and_store_title(session)
                
                if dry_run:
                    console.print(f"[{theme['text_muted']}]{session_id[:16]}...[/{theme['text_muted']}] -> {title}")
                else:
                    # Re-index the session to store the new title
                    # Note: This requires re-indexing since ChromaDB doesn't support metadata updates
                    indexer = FullIndexer(
                        db,
                        chunk_size=config.chunk_size,
                        chunk_overlap=config.chunk_overlap,
                        batch_size=config.batch_size
                    )
                    indexer.index_session(task_dir)
                    # CRITICAL: Finalize to flush pending chunks immediately
                    indexer.finalize()
                    
                updated += 1
                
                if (i + 1) % 10 == 0:
                    status.update(f"[bold {info_color}]Processed {i + 1}/{len(session_ids)} sessions...")
                    
            except Exception as e:
                logger.error(f"Failed to update title for {session_id}: {e}")
                failed += 1
            
            # Small delay in lite mode to reduce CPU usage
            if config.lite_mode and i % 5 == 0:
                time.sleep(0.1)
    
    # Display results
    console.print(f"\n[bold {info_color}]Results:[/bold {info_color}]")
    console.print(f"  [{success_color}]Updated:[/{success_color}] {updated}")
    console.print(f"  [{warning_color}]Skipped:[/{warning_color}] {skipped}")
    if failed > 0:
        console.print(f"  [{error_color}]Failed:[/{error_color}] {failed}")
    
    if dry_run:
        console.print(f"\n[{theme['text_muted']}]This was a dry run. Use without --dry-run to apply changes.[/{theme['text_muted']}]")
    else:
        console.print(f"\n[{success_color}]Title update complete![/{success_color}]")


@app.command()
def interactive():
    """Start the interactive shell (REPL mode)."""
    config = get_config()
    theme_name = getattr(config, "theme", DEFAULT_THEME)
    theme = get_theme_colors(theme_name)
    semantic = theme.get("semantic", {})
    
    # Theme-aware colors
    info_color = semantic.get("info", theme["text_primary"])
    error_color = semantic.get("error", "#EF4444")
    
    console.print(Panel.fit(
        f"[bold {info_color}]SmartFork Interactive Mode[/bold {info_color}]\n"
        "Starting interactive shell...",
        title="SmartFork",
        border_style=theme["panel_border"]
    ))
    
    try:
        start_interactive_shell()
    except Exception as e:
        console.print(f"[{error_color}]Error starting interactive shell: {e}[/{error_color}]")
        raise typer.Exit(1)


# ═══════════════════════════════════════════════════════════════════════════════
# v2 COMMANDS — Structured Session Intelligence Pipeline
# ═══════════════════════════════════════════════════════════════════════════════


@app.command("index-v2")
def index_v2(
    force: bool = typer.Option(False, "--force", "-f", help="Force full re-index"),
    skip_embeddings: bool = typer.Option(False, "--skip-embeddings", help="Parse + SQLite only, skip vector embeddings"),
):
    """Index sessions using the v2 structured pipeline.
    
    Parses all 3 Kilo Code files per session, extracts structured signals,
    stores metadata in SQLite, and embeds documents into ChromaDB.
    """
    config = get_config()
    theme_name = getattr(config, "theme", DEFAULT_THEME)
    theme = get_theme_colors(theme_name)
    semantic = theme.get("semantic", {})
    info_color = semantic.get("info", theme["text_primary"])
    success_color = semantic.get("success", theme["done_color"])
    warning_color = semantic.get("warning", "#F59E0B")
    error_color = semantic.get("error", "#EF4444")

    from .indexer.session_parser import SessionParser
    from .indexer.session_scanner import SessionScanner
    from .database.metadata_store import MetadataStore

    if not config.kilo_code_tasks_path.exists():
        console.print(f"[{error_color}]Tasks path not found:[/{error_color}] {config.kilo_code_tasks_path}")
        raise typer.Exit(1)

    # Initialize v2 components
    store = MetadataStore(config.sqlite_db_path)
    parser = SessionParser()
    scanner = SessionScanner(config.kilo_code_tasks_path, store)

    if force:
        console.print(f"[{warning_color}]Resetting v2 index...[/{warning_color}]")
        store.reset()

    console.print(Panel.fit(
        f"[bold {info_color}]SmartFork v2 Indexer[/bold {info_color}]\n"
        f"Source: {config.kilo_code_tasks_path}\n"
        f"SQLite: {config.sqlite_db_path}",
        title="Index v2",
        border_style=theme["panel_border"]
    ))

    # Scan for sessions
    scan_result = scanner.scan()
    sessions_to_index = scan_result.new_session_paths + scan_result.changed_session_paths

    if not sessions_to_index:
        console.print(f"\n[{success_color}]✓ All {scan_result.total_found} sessions up to date.[/{success_color}]")
        store.close()
        raise typer.Exit(0)

    console.print(
        f"\n  [{info_color}]Found:[/{info_color}] {scan_result.total_found} sessions "
        f"({scan_result.new_sessions} new, {scan_result.changed_sessions} changed, "
        f"{scan_result.unchanged_sessions} unchanged)"
    )

    # Initialize vector index if embeddings are enabled
    vector_index = None
    if not skip_embeddings:
        try:
            from .search.embedder import get_embedder, check_ollama_available
            from .database.vector_index import VectorIndex

            # Check Ollama availability
            ollama_status = check_ollama_available(config.embedding_model)
            if ollama_status["available"]:
                embedder = get_embedder("ollama", config.embedding_model, config.embedding_dimensions)
                console.print(f"  [{success_color}]✓ Ollama ready[/{success_color}] ({config.embedding_model})")
            else:
                console.print(f"  [{warning_color}]⚠ Ollama not available, using sentence-transformers fallback[/{warning_color}]")
                console.print(f"  [{theme['text_muted']}]{ollama_status['message']}[/{theme['text_muted']}]")
                embedder = get_embedder("sentence-transformers")

            vector_index = VectorIndex(config.chroma_db_path / "v2_index", embedder)
        except Exception as e:
            console.print(f"  [{warning_color}]⚠ Embedding init failed: {e}[/{warning_color}]")
            console.print(f"  [{theme['text_muted']}]Continuing with metadata-only indexing[/{theme['text_muted']}]")

    # Index sessions
    indexed = 0
    failed = 0
    total_vectors = {"task": 0, "summary": 0, "reasoning": 0}

    from .ui.v2_progress import V2IndexProgress

    embed_provider_name = ""
    if vector_index:
        embed_provider_name = config.embedding_model
    
    with V2IndexProgress(
        total_sessions=len(sessions_to_index),
        theme_name=theme_name,
        console=console,
        embedding_provider=embed_provider_name,
        skip_embeddings=skip_embeddings or (vector_index is None),
        animation_fps=config.get_effective_fps(),
        disable_animation=config.disable_animations,
    ) as prog:

        for i, session_path in enumerate(sessions_to_index):
            sid = session_path.name
            prog.start_session(sid)

            try:
                # Step 1: Parse
                prog.step_active("parse")
                doc = parser.parse_session(session_path)

                if doc is None:
                    prog.step_error("parse")
                    failed += 1
                    prog.add_error()
                    prog.advance()
                    continue

                prog.step_done("parse",
                               files=len(doc.files_edited),
                               domains=doc.domains,
                               languages=doc.languages)
                
                if doc.project_name and doc.project_name != "unknown":
                    prog.set_project(doc.project_name)

                # Step 2: SQLite store
                prog.step_active("store")
                store.upsert_session(doc)
                prog.step_done("store")

                # Step 3: Embed
                if vector_index:
                    prog.step_active("embed")
                    try:
                        counts = vector_index.index_session(doc)
                        for k, v in counts.items():
                            total_vectors[k] = total_vectors.get(k, 0) + v
                        total_chunks = sum(counts.values())
                        prog.step_done("embed", chunks=total_chunks)
                    except Exception as e:
                        logger.warning(f"Vector indexing failed for {sid}: {e}")
                        prog.step_error("embed")
                else:
                    prog.step_skip("embed")
                    prog.add_chunks(1)

                indexed += 1
                prog.advance()

            except Exception as e:
                logger.error(f"Failed to index {sid}: {e}")
                failed += 1
                prog.add_error()
                prog.advance()

        prog.finish()
        v2_stats = prog.stats


    # Build BM25 index
    try:
        from .search.bm25_index import BM25Index
        bm25 = BM25Index()
        bm25_count = bm25.build_from_metadata(store)
        console.print(f"  [{success_color}]✓ BM25 index built[/{success_color}] ({bm25_count} documents)")
    except Exception as e:
        console.print(f"  [{warning_color}]⚠ BM25 build failed: {e}[/{warning_color}]")

    # Completion summary
    console.print()
    summary_table = Table(show_header=False, box=box.SIMPLE, padding=(0, 2))
    summary_table.add_column("Metric", style=info_color)
    summary_table.add_column("Value", style=success_color)
    summary_table.add_row("Sessions indexed", str(v2_stats.indexed_sessions))
    summary_table.add_row("Chunks created", f"{v2_stats.total_chunks:,}")
    summary_table.add_row("Errors", str(v2_stats.errors))
    summary_table.add_row("Total time", f"{v2_stats.elapsed:.1f}s")
    summary_table.add_row("Total in SQLite", str(store.get_session_count()))
    if vector_index:
        stats = vector_index.get_stats()
        for k, v in stats.items():
            summary_table.add_row(f"Vector ({k})", f"{v} docs")
    console.print(Panel(summary_table, title=f"[bold {success_color}]✓ Index Complete[/bold {success_color}]",
                        border_style=success_color, box=box.ROUNDED))
    console.print(f"\n  [dim {theme['text_muted']}]Run [bold]smartfork search-v2[/bold] to start searching.[/dim {theme['text_muted']}]\n")


    store.close()


@app.command("summarize-v2")
def summarize_v2(
    force: bool = typer.Option(False, "--force", "-f", help="Regenerate all summaries"),
):
    """Generate LLM summaries for all indexed sessions.
    
    Uses Ollama (qwen3:0.6b) to create 3-sentence summaries for search ranking.
    Skips sessions that already have summaries unless --force is used.
    """
    config = get_config()
    theme_name = getattr(config, "theme", DEFAULT_THEME)
    theme = get_theme_colors(theme_name)
    semantic = theme.get("semantic", {})
    info_color = semantic.get("info", theme["text_primary"])
    success_color = semantic.get("success", theme["done_color"])
    warning_color = semantic.get("warning", "#F59E0B")

    from .database.metadata_store import MetadataStore
    from .search.embedder import check_ollama_available, get_embedder
    from .intelligence.llm_provider import get_llm
    from .intelligence.session_summarizer import SessionSummarizer
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn

    store = MetadataStore(config.sqlite_db_path)

    if store.get_session_count() == 0:
        console.print(f"[{warning_color}]No sessions indexed. Run 'smartfork index-v2' first.[/{warning_color}]")
        store.close()
        raise typer.Exit(1)

    # Check Ollama
    ollama_status = check_ollama_available(config.embedding_model)
    if not ollama_status["available"]:
        console.print(f"[{warning_color}]Ollama not available. Start Ollama and try again.[/{warning_color}]")
        console.print(f"[{theme['text_muted']}]{ollama_status['message']}[/{theme['text_muted']}]")
        store.close()
        raise typer.Exit(1)

    # Init LLM and summarizer
    llm = get_llm("ollama")
    summarizer = SessionSummarizer(llm=llm)

    # Get all sessions
    all_sessions = store.get_all_sessions()
    to_summarize = [s for s in all_sessions if force or not s.summary_doc]

    if not to_summarize:
        console.print(f"[{success_color}]All {len(all_sessions)} sessions already have summaries.[/{success_color}]")
        store.close()
        return

    console.print(f"[{info_color}]Generating summaries for {len(to_summarize)} sessions...[/{info_color}]")

    generated = 0
    errors = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=40),
        MofNCompleteColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Summarizing", total=len(to_summarize))

        for doc in to_summarize:
            progress.update(task, description=f"[dim]{doc.project_name}[/dim]")

            try:
                summary = summarizer.summarize(doc)
                if summary:
                    store.update_summary(doc.session_id, summary)
                    generated += 1
                else:
                    errors += 1
            except Exception as e:
                logger.warning(f"Summary failed for {doc.session_id}: {e}")
                errors += 1

            progress.advance(task)

    # Re-embed summaries into ChromaDB
    console.print(f"\n[{info_color}]Re-embedding summaries into vector index...[/{info_color}]")
    try:
        embedder = get_embedder("ollama", config.embedding_model, config.embedding_dimensions)
        from .database.vector_index import VectorIndex
        from .indexer.contextual_chunker import ContextualChunker

        vector_index = VectorIndex(config.chroma_db_path / "v2_index", embedder)
        chunker = ContextualChunker()

        embedded = 0
        for doc in store.get_all_sessions():
            if not doc.summary_doc:
                continue
            summary_text = chunker.build_summary_doc(doc)
            if summary_text:
                try:
                    embedding = embedder.embed(summary_text, "summary_doc")
                    doc_id = f"{doc.session_id}_summary_0"
                    vector_index.summary_collection.upsert(
                        ids=[doc_id],
                        embeddings=[embedding],
                        documents=[summary_text],
                        metadatas=[{
                            "session_id": doc.session_id,
                            "doc_type": "summary_doc",
                            "project_name": doc.project_name,
                            "chunk_index": 0,
                        }]
                    )
                    embedded += 1
                except Exception as e:
                    logger.warning(f"Failed to embed summary for {doc.session_id}: {e}")

        console.print(f"  [dim]Embedded {embedded} summaries into vector index[/dim]")
    except Exception as e:
        console.print(f"[{warning_color}]Vector embedding of summaries failed: {e}[/{warning_color}]")

    console.print(f"\n[{success_color}]✓ Summaries generated: {generated}, errors: {errors}[/{success_color}]")
    store.close()



@app.command("search-v2")
def search_v2(
    query: str = typer.Argument(..., help="Search query"),
    n_results: int = typer.Option(5, "--results", "-n", help="Number of results"),
    project: Optional[str] = typer.Option(None, "--project", "-p", help="Filter by project name"),
    json_output: bool = typer.Option(False, "--json", "-j", help="Output as JSON"),
):
    """Search sessions using v2 structured pipeline with result cards.
    
    Uses query decomposition → metadata filtering → BM25 + vector → RRF fusion.
    """
    config = get_config()
    theme_name = getattr(config, "theme", DEFAULT_THEME)
    theme = get_theme_colors(theme_name)
    semantic = theme.get("semantic", {})
    info_color = semantic.get("info", theme["text_primary"])
    success_color = semantic.get("success", theme["done_color"])
    warning_color = semantic.get("warning", "#F59E0B")

    from .database.metadata_store import MetadataStore
    from .search.query_decomposer import QueryDecomposer, get_vector_weights
    from .search.bm25_index import BM25Index
    from .search.rrf_fusion import rrf_fuse_alpha
    from .ui.result_card import build_result_card, render_result_cards
    from .search.embedder import check_ollama_available, get_embedder

    store = MetadataStore(config.sqlite_db_path)

    if store.get_session_count() == 0:
        console.print(f"[{warning_color}]No v2 sessions indexed. Run 'smartfork index-v2' first.[/{warning_color}]")
        store.close()
        raise typer.Exit(1)

    # Get known project names for fuzzy matching
    known_projects = [p["project_name"] for p in store.get_project_list()]

    # Step 1: Decompose query (try LLM, fallback to rule-based)
    llm = None
    try:
        from .intelligence.llm_provider import get_llm
        ollama_check = check_ollama_available(config.embedding_model)
        if ollama_check["available"]:
            # Use default generative model (qwen3:0.6b) for query understanding
            llm = get_llm("ollama")
    except Exception:
        pass

    decomposer = QueryDecomposer(llm=llm, known_projects=known_projects)
    decomposition = decomposer.decompose(query)

    decomp_method = "rule-based"  # Always rule-based now
    console.print(f"[bold {info_color}]Query:[/bold {info_color}] {query}")
    console.print(f"[{theme['text_muted']}]Intent: {decomposition.intent} | "
                  f"Topic: {decomposition.topic or 'N/A'} | "
                  f"Project: {decomposition.project or project or 'all'} | "
                  f"Decomposer: {decomp_method}[/{theme['text_muted']}]\n")

    # Step 2: Metadata filter (Signal A)
    filter_project = project or decomposition.project

    # Convert time hint to timestamp
    time_after = None
    if decomposition.time_hint:
        import time as time_module
        now = time_module.time() * 1000
        time_map = {
            "yesterday": now - 86400000,
            "today": now - 43200000,
            "last_week": now - 604800000,
            "this_week": now - 604800000,
            "3_days_ago": now - 259200000,
            "last_month": now - 2592000000,
            "this_month": now - 2592000000,
        }
        time_after = int(time_map.get(decomposition.time_hint, now - 604800000))

    candidates = store.filter_sessions(
        project=filter_project,
        file_hint=decomposition.file_hint,
        time_after=time_after,
        limit=50,
    )

    if not candidates:
        console.print(f"[{warning_color}]No matching sessions found.[/{warning_color}]")
        store.close()
        return

    # Step 3: BM25 search (Signal B)
    bm25 = BM25Index()
    bm25.build_from_metadata(store)
    bm25_terms = decomposition.tech_terms + ([decomposition.topic] if decomposition.topic else [])
    bm25_results = bm25.search(bm25_terms, candidate_ids=candidates, n_results=n_results * 3)

    # Step 4: Vector search (Signal C) — dimension-safe
    vector_results_ranked = []
    try:
        from .database.vector_index import VectorIndex

        ollama_status = check_ollama_available(config.embedding_model)
        if ollama_status["available"]:
            # Use Ollama (same model that built the index) — dimensions will match
            embedder = get_embedder("ollama", config.embedding_model, config.embedding_dimensions)
            vector_index = VectorIndex(config.chroma_db_path / "v2_index", embedder)
            query_embedding = embedder.embed_query(query)

            weights = get_vector_weights(decomposition.intent)
            vec_results = vector_index.search_all_collections(
                query_embedding, session_ids=candidates, n_results=n_results * 3,
                weights=weights,
            )

            seen = set()
            for vr in vec_results:
                if vr.session_id not in seen:
                    vector_results_ranked.append((vr.session_id, vr.score))
                    seen.add(vr.session_id)
        else:
            # Don't use a different embedder — dimensions won't match
            console.print(f"[{warning_color}]⚠ Ollama not running — vector search disabled (BM25 only)[/{warning_color}]")
            console.print(f"[{theme['text_muted']}]  Start Ollama for better results[/{theme['text_muted']}]\n")
    except Exception as e:
        logger.warning(f"Vector search failed: {e}")

    # Step 5: RRF Fusion (combine BM25 + vector rankings)
    if bm25_results or vector_results_ranked:
        fused = rrf_fuse_alpha(
            bm25_ranking=bm25_results,
            vector_ranking=vector_results_ranked,
            intent_type=decomposition.intent,
            top_n=n_results
        )
    else:
        # Fallback: use candidate order (most recent first)
        fused = [(sid, 0.5) for sid in candidates[:n_results]]

    # Step 6: Build result cards
    cards = []
    for session_id, score in fused:
        doc = store.get_session(session_id)
        if doc:
            # Find best snippet
            snippet = ""
            if doc.reasoning_docs:
                snippet = doc.reasoning_docs[0][:120]
            elif doc.summary_doc:
                snippet = doc.summary_doc[:120]
            elif doc.task_raw:
                snippet = doc.task_raw[:120]

            # Normalize score to 0-1 range (RRF scores are small fractions)
            normalized_score = min(score / (score + 0.01), 1.0) if score > 0 else 0.0
            card = build_result_card(doc, match_score=normalized_score, snippet=snippet)
            cards.append(card)

    if json_output:
        output = [{"session_id": c.session_id, "project": c.project_name,
                    "task": c.task_short, "score": c.match_score,
                    "files": c.files_changed} for c in cards]
        console.print(json.dumps(output, indent=2, default=str))
    else:
        render_result_cards(cards, console)
        console.print(f"\n[{theme['text_muted']}]Use [bold]smartfork fork-v2 <session_id> --intent continue|reference|debug[/bold][/{theme['text_muted']}]")

    store.close()


@app.command("fork-v2")
def fork_v2(
    session_id: str = typer.Argument(..., help="Session ID to fork context from"),
    intent: str = typer.Option("continue", "--intent", "-i",
                                help="Fork intent: continue, reference, or debug"),
    query: str = typer.Option("", "--query", "-q", help="Optional query to filter context using Vector Search"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file path"),
    clipboard: bool = typer.Option(False, "--clipboard", "-c", help="Copy to clipboard"),
):
    """Generate intent-classified fork context from a v2-indexed session."""
    config = get_config()
    theme_name = getattr(config, "theme", DEFAULT_THEME)
    theme = get_theme_colors(theme_name)
    semantic = theme.get("semantic", {})
    info_color = semantic.get("info", theme["text_primary"])
    success_color = semantic.get("success", theme["done_color"])
    error_color = semantic.get("error", "#EF4444")

    from .database.metadata_store import MetadataStore
    from .fork.fork_assembler import assemble_fork_context
    from .mcp.mcp_server import file_drop_context, clipboard_context
    from .search.embedder import check_ollama_available

    store = MetadataStore(config.sqlite_db_path)
    doc = store.get_session(session_id)

    if not doc:
        console.print(f"[{error_color}]Session not found in v2 index: {session_id}[/{error_color}]")
        console.print(f"[{theme['text_muted']}]Run 'smartfork index-v2' first.[/{theme['text_muted']}]")
        store.close()
        raise typer.Exit(1)

    # Validate intent
    valid_intents = {"continue", "reference", "debug"}
    if intent.lower() not in valid_intents:
        console.print(f"[{error_color}]Invalid intent: {intent}. Use: continue, reference, or debug[/{error_color}]")
        store.close()
        raise typer.Exit(1)

    console.print(Panel.fit(
        f"[bold {info_color}]Forking session[/bold {info_color}]\n"
        f"Project: {doc.project_name}\n"
        f"Task: {doc.task_raw[:60]}{'...' if len(doc.task_raw) > 60 else ''}\n"
        f"Intent: {intent}",
        title=f"Fork v2 — {intent.title()}",
        border_style=theme["panel_border"]
    ))

    # Initialize LLM and VectorIndex for intelligent fork assembly
    llm = None
    vector_index = None
    try:
        from .intelligence.llm_provider import get_llm
        from .search.embedder import get_embedder
        from .database.vector_index import VectorIndex
        
        ollama_check = check_ollama_available(config.embedding_model)
        if ollama_check["available"]:
            llm = get_llm("ollama")
            embedder = get_embedder("ollama", config.embedding_model, config.embedding_dimensions)
            vector_index = VectorIndex(config.chroma_db_path / "v2_index", embedder)
            console.print(f"[{theme['text_muted']}]Using LLM for context distillation...[/{theme['text_muted']}]")
        else:
            console.print(f"[{theme['text_muted']}]Ollama not running — using cleaned raw assembly[/{theme['text_muted']}]")
    except Exception as e:
        logger.debug(f"Failed to init LLM/VectorIndex: {e}")

    # Assemble context (LLM-powered if available, raw fallback otherwise)
    with console.status(f"[{info_color}]Assembling context...[/{info_color}]"):
        context = assemble_fork_context(
            doc, intent, query=query, llm=llm, 
            vector_index=vector_index, store=store
        )

    # Deliver
    if clipboard:
        if clipboard_context(context):
            console.print(f"[{success_color}]✓ Context copied to clipboard[/{success_color}]")
        else:
            console.print(f"[{theme['text_muted']}]Clipboard unavailable, saving to file...[/{theme['text_muted']}]")
            clipboard = False

    if output or not clipboard:
        if not output:
            short_id = session_id[:8]
            output = Path(f"fork_{short_id}_{intent}.md")
        output.write_text(context, encoding="utf-8")
        console.print(f"[{success_color}]✓ Fork saved to:[/{success_color}] {output.absolute()}")

    # Preview
    console.print(f"\n[{theme['text_muted']}]Preview:[/{theme['text_muted']}]")
    preview = context[:600] + ("..." if len(context) > 600 else "")
    console.print(Panel(preview, border_style=theme["panel_border"]))

    store.close()


@app.command("status-v2")
def status_v2():
    """Show v2 index statistics — sessions, projects, domains, vectors."""
    config = get_config()
    theme_name = getattr(config, "theme", DEFAULT_THEME)
    theme = get_theme_colors(theme_name)
    semantic = theme.get("semantic", {})
    info_color = semantic.get("info", theme["text_primary"])
    success_color = semantic.get("success", theme["done_color"])
    warning_color = semantic.get("warning", "#F59E0B")

    from .database.metadata_store import MetadataStore

    store = MetadataStore(config.sqlite_db_path)

    session_count = store.get_session_count()
    if session_count == 0:
        console.print(f"[{warning_color}]No v2 sessions indexed. Run 'smartfork index-v2' first.[/{warning_color}]")
        store.close()
        raise typer.Exit(0)

    projects = store.get_project_list()
    domains = store.get_domain_breakdown()

    console.print(Panel.fit(
        f"[bold {theme['text_primary']}]SmartFork v2 Status[/bold {theme['text_primary']}]",
        title="Status v2",
        border_style=theme["panel_border"]
    ))

    # Main stats
    table = Table(show_header=False, box=box.SIMPLE)
    table.add_column("Property", style=info_color)
    table.add_column("Value", style=success_color)
    table.add_row("SQLite Path", str(config.sqlite_db_path))
    table.add_row("Indexed Sessions", str(session_count))
    table.add_row("Embedding Provider", config.embedding_provider)
    table.add_row("Embedding Model", config.embedding_model)
    table.add_row("LLM Provider", config.llm_provider)
    table.add_row("Schema Version", str(config.schema_version))
    console.print(table)

    # Projects
    if projects:
        console.print(f"\n[bold {info_color}]Projects[/bold {info_color}]")
        for p in projects[:10]:
            console.print(f"  [{success_color}]{p['project_name']}[/{success_color}] — {p['session_count']} sessions")

    # Domains
    if domains:
        console.print(f"\n[bold {info_color}]Domain Breakdown[/bold {info_color}]")
        for domain, count in list(domains.items())[:10]:
            bar_len = min(count * 2, 30)
            bar = "█" * bar_len
            console.print(f"  {domain:15s} [{success_color}]{bar}[/{success_color}] {count}")

    store.close()


@app.command("migrate-v2")
def migrate_v2(
    dry_run: bool = typer.Option(False, "--dry-run", "-d", help="Show what would be migrated"),
):
    """Migrate v1 indexed sessions to v2 structured format.
    
    Re-parses all sessions from the original Kilo Code files using the v2
    structured parser, then stores them in the SQLite metadata store.
    Existing v1 data is NOT modified.
    """
    config = get_config()
    theme_name = getattr(config, "theme", DEFAULT_THEME)
    theme = get_theme_colors(theme_name)
    semantic = theme.get("semantic", {})
    info_color = semantic.get("info", theme["text_primary"])
    success_color = semantic.get("success", theme["done_color"])
    warning_color = semantic.get("warning", "#F59E0B")
    error_color = semantic.get("error", "#EF4444")

    from .indexer.session_parser import SessionParser
    from .indexer.session_scanner import SessionScanner
    from .database.metadata_store import MetadataStore

    if not config.kilo_code_tasks_path.exists():
        console.print(f"[{error_color}]Tasks path not found:[/{error_color}] {config.kilo_code_tasks_path}")
        raise typer.Exit(1)

    store = MetadataStore(config.sqlite_db_path)
    parser = SessionParser()
    scanner = SessionScanner(config.kilo_code_tasks_path, store)

    # Get ALL session directories
    all_sessions = scanner.get_all_session_paths()

    if not all_sessions:
        console.print(f"[{warning_color}]No sessions found in {config.kilo_code_tasks_path}[/{warning_color}]")
        store.close()
        raise typer.Exit(0)

    console.print(Panel.fit(
        f"[bold {info_color}]SmartFork v1 → v2 Migration[/bold {info_color}]\n"
        f"Source: {config.kilo_code_tasks_path}\n"
        f"Sessions found: {len(all_sessions)}\n"
        f"Already in v2: {store.get_session_count()}",
        title="Migrate v2",
        border_style=theme["panel_border"]
    ))

    if dry_run:
        console.print(f"\n[{warning_color}]DRY RUN — no changes will be made[/{warning_color}]")
        for i, path in enumerate(all_sessions[:20], 1):
            console.print(f"  {i}. {path.name}")
        if len(all_sessions) > 20:
            console.print(f"  ... and {len(all_sessions) - 20} more")
        store.close()
        return

    migrated = 0
    failed = 0

    with console.status(f"[bold {info_color}]Migrating {len(all_sessions)} sessions...") as status:
        for i, session_path in enumerate(all_sessions):
            sid = session_path.name
            status.update(f"[bold {info_color}]Migrating [{i+1}/{len(all_sessions)}] {sid}...")

            try:
                doc = parser.parse_session(session_path)
                if doc:
                    store.upsert_session(doc)
                    migrated += 1
                else:
                    failed += 1
            except Exception as e:
                logger.error(f"Migration failed for {sid}: {e}")
                failed += 1

    console.print(f"\n[bold {success_color}]Migration Complete[/bold {success_color}]")
    console.print(f"  [{success_color}]✓ Migrated:[/{success_color}] {migrated}")
    console.print(f"  [{warning_color if failed else theme['text_muted']}]✗ Failed:[/{warning_color if failed else theme['text_muted']}] {failed}")
    console.print(f"  [{info_color}]Total in v2 store:[/{info_color}] {store.get_session_count()}")

    store.close()


@app.command("mcp-server")
def mcp_server_cmd():
    """Start the SmartFork MCP server for Kilo Code integration.
    
    Exposes SmartFork tools (search, fork, detect-fork, status) via MCP
    for direct IDE integration.
    """
    config = get_config()
    theme_name = getattr(config, "theme", DEFAULT_THEME)
    theme = get_theme_colors(theme_name)
    semantic = theme.get("semantic", {})
    info_color = semantic.get("info", theme["text_primary"])
    success_color = semantic.get("success", theme["done_color"])
    warning_color = semantic.get("warning", "#F59E0B")

    from .database.metadata_store import MetadataStore
    from .mcp.mcp_server import SmartForkMCPServer

    store = MetadataStore(config.sqlite_db_path)

    if store.get_session_count() == 0:
        console.print(f"[{warning_color}]No v2 sessions indexed. Run 'smartfork index-v2' first.[/{warning_color}]")
        store.close()
        raise typer.Exit(1)

    server = SmartForkMCPServer(metadata_store=store)
    tools = server.get_tool_definitions()

    console.print(Panel.fit(
        f"[bold {info_color}]SmartFork MCP Server[/bold {info_color}]\n"
        f"Sessions: {store.get_session_count()}\n"
        f"Tools: {', '.join(t['name'] for t in tools)}",
        title="MCP Server",
        border_style=theme["panel_border"]
    ))

    console.print(f"\n[{theme['text_muted']}]MCP server is ready. Tools registered:[/{theme['text_muted']}]")
    for tool in tools:
        console.print(f"  [{success_color}]•[/{success_color}] {tool['name']} — {tool['description'][:60]}")

    console.print(f"\n[{theme['text_muted']}]To connect from Kilo Code, add SmartFork as an MCP server in your settings.[/{theme['text_muted']}]")
    console.print(f"[{theme['text_muted']}]Press Ctrl+C to stop.[/{theme['text_muted']}]")

    try:
        import time as time_module
        while True:
            time_module.sleep(1)
    except KeyboardInterrupt:
        console.print(f"\n[{theme['text_muted']}]MCP server stopped.[/{theme['text_muted']}]")
        store.close()


if __name__ == "__main__":
    app()
