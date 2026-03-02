"""CLI for SmartFork."""

import typer
from pathlib import Path
from typing import Optional, List
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from loguru import logger
import sys

from .config import get_config
from .database.chroma_db import ChromaDatabase
from .indexer.indexer import FullIndexer, IncrementalIndexer
from .indexer.watcher import TranscriptWatcher
from .search.semantic import SemanticSearchEngine

app = typer.Typer(help="SmartFork - AI Session Intelligence for Kilo Code")
console = Console()


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


@app.callback()
def main(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
    log_file: Optional[Path] = typer.Option(None, "--log-file", help="Log file path")
):
    """SmartFork - AI Session Intelligence for Kilo Code."""
    config = get_config()
    log_level = "DEBUG" if verbose else config.log_level
    setup_logging(log_level, log_file)


@app.command()
def index(
    force: bool = typer.Option(False, "--force", "-f", help="Force full re-index (clears existing data)"),
    watch: bool = typer.Option(False, "--watch", "-w", help="Watch for changes after indexing"),
):
    """Index all Kilo Code sessions."""
    config = get_config()
    
    console.print(Panel.fit(
        "[bold blue]SmartFork Indexer[/bold blue]\n"
        f"Tasks path: {config.kilo_code_tasks_path}\n"
        f"Database: {config.chroma_db_path}",
        title="SmartFork"
    ))
    
    # Check if tasks path exists
    if not config.kilo_code_tasks_path.exists():
        console.print(f"[red]Error: Tasks path does not exist: {config.kilo_code_tasks_path}[/red]")
        console.print("[yellow]Make sure Kilo Code extension is installed and has created task directories.[/yellow]")
        raise typer.Exit(1)
    
    # Initialize components
    db = ChromaDatabase(config.chroma_db_path)
    
    if force:
        console.print("[yellow]Resetting database...[/yellow]")
        db.reset()
    
    indexer = FullIndexer(db, chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap)
    
    # Perform indexing
    with console.status("[bold green]Indexing sessions..."):
        result = indexer.index_all_sessions(config.kilo_code_tasks_path)
    
    # Display results
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Count", style="green")
    
    table.add_row("Sessions Indexed", str(result.indexed))
    table.add_row("Chunks Created", str(result.chunks_created))
    table.add_row("Failed", str(result.failed))
    table.add_row("Total Chunks in DB", str(db.get_session_count()))
    
    console.print(table)
    
    if result.failed > 0:
        console.print(f"[yellow]Warning: {result.failed} sessions failed to index[/yellow]")
    
    # Watch mode
    if watch:
        console.print("\n[bold]Watch mode enabled. Press Ctrl+C to stop.[/bold]")
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
            console.print("\n[yellow]Stopping watcher...[/yellow]")
            watcher.stop()


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    n_results: int = typer.Option(5, "--results", "-n", help="Number of results"),
    technology: Optional[List[str]] = typer.Option(None, "--tech", "-t", help="Filter by technology"),
    file: Optional[List[str]] = typer.Option(None, "--file", "-f", help="Filter by file path"),
    json_output: bool = typer.Option(False, "--json", "-j", help="Output as JSON"),
):
    """Search indexed sessions."""
    config = get_config()
    
    db = ChromaDatabase(config.chroma_db_path)
    engine = SemanticSearchEngine(db)
    
    if db.get_session_count() == 0:
        console.print("[yellow]No sessions indexed. Run 'smartfork index' first.[/yellow]")
        raise typer.Exit(1)
    
    console.print(f"[bold]Searching for:[/bold] {query}\n")
    
    results = engine.search(
        query, 
        n_results=n_results,
        technologies=technology,
        files=file
    )
    
    if not results:
        console.print("[yellow]No results found.[/yellow]")
        return
    
    if json_output:
        import json
        output = [
            {
                "session_id": r.session_id,
                "score": r.score,
                "content": r.content[:500],
                "metadata": r.metadata
            }
            for r in results
        ]
        console.print(json.dumps(output, indent=2, default=str))
    else:
        # Display results
        console.print(f"[dim]Found {len(results)} results[/dim]\n")
        
        for i, r in enumerate(results, 1):
            score_pct = f"{r.score:.1%}"
            
            # Get technologies if available
            techs = r.metadata.get("technologies", [])
            tech_str = f" [dim]({', '.join(techs[:3])})[/dim]" if techs else ""
            
            # Get files in context
            files = r.metadata.get("files_in_context", [])
            files_str = "\n".join([f"  [dim]• {f}[/dim]" for f in files[:3]]) if files else ""
            
            # Content preview
            preview = r.content[:200] + "..." if len(r.content) > 200 else r.content
            preview = preview.replace("\n", " ")
            
            panel_content = f"[bold]Score:[/bold] {score_pct}{tech_str}\n\n"
            panel_content += f"[dim]{preview}[/dim]\n"
            if files_str:
                panel_content += f"\n[bold]Files:[/bold]\n{files_str}"
            
            console.print(Panel(
                panel_content,
                title=f"[{i}] Session {r.session_id[:16]}...",
                border_style="green" if r.score > 0.7 else "yellow"
            ))


@app.command()
def detect_fork(
    query: str = typer.Argument(..., help="Describe what you're working on"),
    n_results: int = typer.Option(5, "--results", "-n", help="Number of suggestions"),
):
    """Find relevant past sessions to fork context from."""
    config = get_config()
    
    db = ChromaDatabase(config.chroma_db_path)
    engine = SemanticSearchEngine(db)
    
    if db.get_session_count() == 0:
        console.print("[yellow]No sessions indexed. Run 'smartfork index' first.[/yellow]")
        raise typer.Exit(1)
    
    console.print(Panel.fit(
        f"[bold]Detecting fork for:[/bold] {query}",
        title="SmartFork Detect-Fork"
    ))
    
    results = engine.search(query, n_results=n_results)
    
    if not results:
        console.print("[yellow]No relevant sessions found.[/yellow]")
        return
    
    console.print(f"\n[dim]Found {len(results)} relevant session(s):[/dim]\n")
    
    for i, r in enumerate(results, 1):
        score_pct = f"{r.score:.1%}"
        
        # Get technologies
        techs = r.metadata.get("technologies", [])
        tech_str = f"\n[dim]Tech:[/dim] {', '.join(techs[:5])}" if techs else ""
        
        # Get files
        files = r.metadata.get("files_in_context", [])
        files_preview = f"\n[dim]Files:[/dim] {', '.join(files[:3])}..." if len(files) > 3 else f"\n[dim]Files:[/dim] {', '.join(files)}" if files else ""
        
        content_preview = r.content[:150] + "..." if len(r.content) > 150 else r.content
        content_preview = content_preview.replace("\n", " ")
        
        console.print(Panel(
            f"[bold green]{score_pct}[/bold green] relevance{tech_str}{files_preview}\n\n"
            f"[dim]{content_preview}[/dim]",
            title=f"[{i}] {r.session_id[:20]}..."
        ))
    
    console.print("\n[dim]Use [bold]smartfork fork <session_id>[/bold] to generate context file[/dim]")


@app.command()
def status():
    """Show indexing status."""
    config = get_config()
    
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
    
    # Display status
    console.print(Panel.fit(
        "[bold blue]SmartFork Status[/bold blue]",
        title="Status"
    ))
    
    table = Table(show_header=False)
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Kilo Code Tasks Path", str(config.kilo_code_tasks_path))
    table.add_row("Database Path", str(config.chroma_db_path))
    table.add_row("Total Task Directories", str(total_tasks))
    table.add_row("Indexed Sessions", str(unique_sessions))
    table.add_row("Total Chunks", str(total_chunks))
    table.add_row("Index Coverage", f"{unique_sessions}/{total_tasks} ({unique_sessions/max(1,total_tasks)*100:.1f}%)")
    
    console.print(table)
    
    if unique_sessions < total_tasks:
        console.print("\n[yellow]Tip: Run 'smartfork index' to index remaining sessions[/yellow]")


@app.command()
def config_show():
    """Show current configuration."""
    config = get_config()
    
    console.print(Panel.fit(
        "[bold blue]SmartFork Configuration[/bold blue]",
        title="Config"
    ))
    
    table = Table(show_header=False)
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")
    
    for key, value in config.model_dump().items():
        table.add_row(key, str(value))
    
    console.print(table)


@app.command()
def reset(
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation")
):
    """Reset the database (WARNING: deletes all indexed data)."""
    if not force:
        confirm = typer.confirm("Are you sure you want to delete all indexed data?")
        if not confirm:
            console.print("[yellow]Aborted.[/yellow]")
            raise typer.Exit(0)
    
    config = get_config()
    db = ChromaDatabase(config.chroma_db_path)
    
    db.reset()
    console.print("[green]Database reset complete.[/green]")


@app.command()
def watch():
    """Watch for session changes and index incrementally."""
    config = get_config()
    
    if not config.kilo_code_tasks_path.exists():
        console.print(f"[red]Error: Tasks path does not exist: {config.kilo_code_tasks_path}[/red]")
        raise typer.Exit(1)
    
    console.print("[bold]Starting watcher... Press Ctrl+C to stop.[/bold]\n")
    
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
        console.print("\n[yellow]Stopping watcher...[/yellow]")
        watcher.stop()
        console.print("[green]Watcher stopped.[/green]")


if __name__ == "__main__":
    app()

