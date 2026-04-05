"""Interactive shell for SmartFork CLI.

Provides a persistent REPL-like environment for running SmartFork commands
without typing the 'smartfork' prefix each time.

Fully migrated to v2 pipeline: QueryDecomposer, BM25, Vector, RRF fusion,
result cards, fork-assembler, V2IndexProgress, MetadataStore.
"""

import cmd
import sys
import shlex
from pathlib import Path
from typing import List, Optional, Any
from datetime import datetime
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

# Windows compatibility for readline
try:
    import readline
except ImportError:
    try:
        import pyreadline3 as readline
    except ImportError:
        readline = None

from ..config import get_config, CONFIG_FILE, SmartForkConfig
from ..database.chroma_db import ChromaDatabase
from ..search.hybrid import HybridSearchEngine
from ..fork.generator import ForkMDGenerator
from ..ui.progress import (
    DEFAULT_THEME, get_theme_colors, get_semantic_color,
    display_discovery_phase, SmartForkProgress,
)
from ..ui.result_card import build_result_card, render_result_cards


class SmartForkShell(cmd.Cmd):
    """Interactive shell for SmartFork commands.

    Features:
    - Full v2 search pipeline (decomposition, BM25, vector, RRF fusion, result cards)
    - Intent-based fork-v2 (continue/reference/debug)
    - V2 indexing with supersession detection
    - Command shortcuts (s, i, f, df, st, t, c, q)
    - Quick fork with number keys after search
    - Command history with up/down arrows
    - Tab completion for session IDs and commands
    - Persistent session state
    """

    intro = ""

    prompt = "SmartFork> "

    def __init__(self):
        if readline is None:
            self.use_rawinput = False

        super().__init__()
        self.console = Console()
        self.config = get_config()

        # v1 (legacy, kept for backward compat)
        self.db = None
        self.search_engine = None
        self.fork_generator = None

        # v2 (primary)
        self.metadata_store = None

        # Shared state
        self.last_results: List[Any] = []
        self.last_sessions: List[dict] = []
        self.last_cards: List[Any] = []

        # Theme support
        self.theme_name = getattr(self.config, "theme", DEFAULT_THEME)
        self.theme = get_theme_colors(self.theme_name)
        self.semantic = self.theme.get("semantic", {})
        self.info_color = self.semantic.get("info", self.theme["text_primary"])
        self.success_color = self.semantic.get("success", self.theme["done_color"])
        self.warning_color = self.semantic.get("warning", self.theme["text_primary"])
        self.error_color = self.semantic.get("error", self.theme["text_primary"])
        self.accent_color = self.semantic.get("accent", self.theme["text_primary"])

        self._init_database()

    def _refresh_theme(self):
        """Refresh theme colors from current config."""
        self.theme_name = getattr(self.config, "theme", DEFAULT_THEME)
        self.theme = get_theme_colors(self.theme_name)
        self.semantic = self.theme.get("semantic", {})
        self.info_color = self.semantic.get("info", self.theme["text_primary"])
        self.success_color = self.semantic.get("success", self.theme["done_color"])
        self.warning_color = self.semantic.get("warning", self.theme["text_primary"])
        self.error_color = self.semantic.get("error", self.theme["text_primary"])
        self.accent_color = self.semantic.get("accent", self.theme["text_primary"])

    def _init_database(self):
        """Initialize database connections (v1 + v2)."""
        try:
            self.db = ChromaDatabase(self.config.chroma_db_path)
            self.search_engine = HybridSearchEngine(self.db)
            self.fork_generator = ForkMDGenerator(self.db)
        except Exception:
            pass

        # v2 MetadataStore
        try:
            from ..database.metadata_store import MetadataStore
            self.metadata_store = MetadataStore(self.config.sqlite_db_path)
        except Exception:
            self.metadata_store = None

    def _ensure_db(self):
        """Ensure database is initialized."""
        if self.db is None:
            try:
                self.db = ChromaDatabase(self.config.chroma_db_path)
                self.search_engine = HybridSearchEngine(self.db)
                self.fork_generator = ForkMDGenerator(self.db)
            except Exception as e:
                self.console.print(f"[{self.error_color}]Database init failed: {e}[/{self.error_color}]")
                return False
        return True

    def _ensure_metadata_store(self):
        """Ensure v2 MetadataStore is initialized."""
        if self.metadata_store is None:
            try:
                from ..database.metadata_store import MetadataStore
                self.metadata_store = MetadataStore(self.config.sqlite_db_path)
            except Exception as e:
                self.console.print(f"[{self.error_color}]MetadataStore init failed: {e}[/{self.error_color}]")
                return False
        return True

    def _display_welcome(self):
        """Show themed welcome panel with status."""
        theme = self.theme
        welcome_text = Text()
        welcome_text.append(f"SmartFork Interactive Shell", style=f"bold {theme['text_primary']}")
        welcome_text.append("\n", style="")
        welcome_text.append(f"Theme: {self.theme_name}", style=f"dim {theme['text_muted']}")

        # Show index status
        if self.metadata_store:
            try:
                count = self.metadata_store.get_session_count()
                if count > 0:
                    welcome_text.append(f"  |  {count} sessions indexed", style=f"dim {theme['text_muted']}")
                    welcome_text.append(f"  |  Ready to search", style=f"dim {self.success_color}")
                else:
                    welcome_text.append(f"  |  No sessions indexed yet — run 'index' to start", style=f"dim {self.warning_color}")
            except Exception:
                welcome_text.append(f"  |  Checking index...", style=f"dim {theme['text_muted']}")
        elif self.db:
            try:
                count = len(self.db.get_unique_sessions())
                if count > 0:
                    welcome_text.append(f"  |  {count} sessions indexed", style=f"dim {theme['text_muted']}")
                else:
                    welcome_text.append(f"  |  No sessions indexed yet — run 'index' to start", style=f"dim {self.warning_color}")
            except Exception:
                pass

        welcome_text.append("\n", style="")
        welcome_text.append("Type 'help' for commands, 'exit' to quit", style=f"dim {theme['text_muted']}")
        welcome_text.append("\n", style="")
        welcome_text.append("Quick tip: After search, press [1-9] to fork that result", style=f"dim {theme['text_muted']}")

        self.console.print(Panel(
            welcome_text,
            box=box.DOUBLE,
            border_style=theme["panel_border"]
        ))

    def preloop(self):
        """Called before the command loop starts."""
        self._display_welcome()

    def default(self, line: str):
        """Handle unknown commands - check for quick fork numbers."""
        stripped = line.strip()

        if stripped.isdigit():
            num = int(stripped)
            if 1 <= num <= 9:
                self.do_fork(str(num))
                return
            else:
                self.console.print(f"[{self.error_color}]Quick fork only supports 1-9. Use 'fork <n>' for larger numbers.[/{self.error_color}]")
                return

        parts = shlex.split(line)
        if parts:
            self.console.print(f"[{self.error_color}]Unknown command: {parts[0]}[/{self.error_color}]")
            self.console.print(f"[{self.theme['text_muted']}]Type 'help' for available commands[/{self.theme['text_muted']}]")

    def emptyline(self):
        """Do nothing on empty line."""
        pass

    def do_exit(self, arg: str):
        """Exit the interactive shell."""
        self.console.print(f"[{self.success_color}]Goodbye![/{self.success_color}]")
        return True

    def do_quit(self, arg: str):
        """Alias for exit."""
        return self.do_exit(arg)

    def do_EOF(self, arg: str):
        """Handle Ctrl+D (EOF)."""
        print()
        return self.do_exit(arg)

    def do_clear(self, arg: str):
        """Clear the screen."""
        self.console.clear()

    # ─────────────────────────────────────────────────────────────────
    # STATUS (v2)
    # ─────────────────────────────────────────────────────────────────

    def do_status(self, arg: str):
        """Show v2 index statistics — sessions, projects, domains, vectors."""
        if not self._ensure_metadata_store():
            return

        store = self.metadata_store
        session_count = store.get_session_count()

        if session_count == 0:
            self.console.print(f"[{self.warning_color}]No v2 sessions indexed. Run 'index' first.[/{self.warning_color}]")
            return

        projects = store.get_project_list()
        domains = store.get_domain_breakdown()

        self.console.print(Panel.fit(
            f"[bold {self.theme['text_primary']}]SmartFork v2 Status[/bold {self.theme['text_primary']}]",
            title="Status v2",
            border_style=self.theme["panel_border"]
        ))

        # Main stats
        table = Table(show_header=False, box=box.SIMPLE)
        table.add_column("Property", style=self.info_color)
        table.add_column("Value", style=self.success_color)
        table.add_row("SQLite Path", str(self.config.sqlite_db_path))
        table.add_row("Indexed Sessions", str(session_count))
        table.add_row("Embedding Provider", self.config.embedding_provider)
        table.add_row("Embedding Model", self.config.embedding_model)
        table.add_row("LLM Provider", self.config.llm_provider)
        table.add_row("Schema Version", str(self.config.schema_version))
        self.console.print(table)

        # Projects
        if projects:
            self.console.print(f"\n[bold {self.info_color}]Projects[/bold {self.info_color}]")
            for p in projects[:10]:
                self.console.print(f"  [{self.success_color}]{p['project_name']}[/{self.success_color}] — {p['session_count']} sessions")

        # Domains
        if domains:
            self.console.print(f"\n[bold {self.info_color}]Domain Breakdown[/bold {self.info_color}]")
            for domain, count in list(domains.items())[:10]:
                bar_len = min(count * 2, 30)
                bar = "█" * bar_len
                self.console.print(f"  {domain:15s} [{self.success_color}]{bar}[/{self.success_color}] {count}")

    # ─────────────────────────────────────────────────────────────────
    # SEARCH (v2 pipeline)
    # ─────────────────────────────────────────────────────────────────

    def do_search(self, arg: str):
        """Search sessions using v2 structured pipeline with result cards.

        Usage: search <query> [--results N] [--project NAME]
        Alias: s

        Uses query decomposition → metadata filtering → BM25 + vector → RRF fusion.
        """
        if not self._ensure_metadata_store():
            return

        if not arg.strip():
            self.console.print(f"[{self.error_color}]Usage: search <query>[/{self.error_color}]")
            return

        parts = shlex.split(arg)
        query = parts[0]
        n_results = 5
        project_filter = None

        for i, part in enumerate(parts):
            if part in ("--results", "-n") and i + 1 < len(parts):
                try:
                    n_results = int(parts[i + 1])
                except ValueError:
                    pass
            elif part in ("--project", "-p") and i + 1 < len(parts):
                project_filter = parts[i + 1]

        store = self.metadata_store

        if store.get_session_count() == 0:
            self.console.print(f"[{self.warning_color}]No v2 sessions indexed. Run 'index' first.[/{self.warning_color}]")
            return

        # Decompose query
        known_projects = [p["project_name"] for p in store.get_project_list()]
        decomposition = None
        try:
            from ..search.query_decomposer import QueryDecomposer
            decomposer = QueryDecomposer(known_projects=known_projects)
            decomposition = decomposer.decompose(query)
        except Exception:
            pass

        if decomposition:
            self.console.print(f"[bold {self.info_color}]Query:[/bold {self.info_color}] {query}")
            self.console.print(f"[{self.theme['text_muted']}]Intent: {decomposition.intent} | "
                               f"Topic: {decomposition.topic or 'N/A'} | "
                               f"Project: {decomposition.project or project_filter or 'all'}[/{self.theme['text_muted']}]\n")

        # Metadata filter
        filter_project = project_filter
        if decomposition and decomposition.project:
            filter_project = decomposition.project

        time_after = None
        if decomposition and decomposition.time_hint:
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
            file_hint=decomposition.file_hint if decomposition else None,
            time_after=time_after,
            limit=50,
        )

        if not candidates:
            self.console.print(f"[{self.warning_color}]No matching sessions found.[/{self.warning_color}]")
            self.last_results = []
            self.last_cards = []
            return

        # BM25 search
        bm25_results = []
        try:
            from ..search.bm25_index import BM25Index
            bm25 = BM25Index()
            bm25.build_from_metadata(store)
            bm25_terms = []
            if decomposition:
                bm25_terms = decomposition.tech_terms + ([decomposition.topic] if decomposition.topic else [])
            bm25_results = bm25.search(bm25_terms, candidate_ids=candidates, n_results=n_results * 3)
        except Exception as e:
            self.console.print(f"[{self.theme['text_muted']}]BM25 search skipped: {e}[/{self.theme['text_muted']}]")

        # Vector search
        vector_results_ranked = []
        try:
            from ..search.embedder import check_ollama_available, get_embedder
            from ..database.vector_index import VectorIndex
            from ..search.query_decomposer import get_vector_weights

            ollama_status = check_ollama_available(self.config.embedding_model)
            if ollama_status["available"]:
                embedder = get_embedder("ollama", self.config.embedding_model, self.config.embedding_dimensions)
                vi_path = self.config.chroma_db_path / "v2_index"
                vector_index = VectorIndex(str(vi_path), embedder)
                query_embedding = embedder.embed_query(query)
                weights = get_vector_weights(decomposition.intent) if decomposition else None
                vec_results = vector_index.search_all_collections(
                    query_embedding, session_ids=candidates, n_results=n_results * 3, weights=weights,
                )
                seen = set()
                for vr in vec_results:
                    if vr.session_id not in seen:
                        vector_results_ranked.append((vr.session_id, vr.score))
                        seen.add(vr.session_id)
            else:
                if not bm25_results:
                    self.console.print(f"[{self.warning_color}]⚠ Ollama not running — vector search disabled[/{self.warning_color}]")
        except Exception as e:
            self.console.print(f"[{self.theme['text_muted']}]Vector search skipped: {e}[/{self.theme['text_muted']}]")

        # RRF Fusion
        if bm25_results or vector_results_ranked:
            try:
                from ..search.rrf_fusion import rrf_fuse_alpha
                intent_type = decomposition.intent if decomposition else "continue"
                fused = rrf_fuse_alpha(
                    bm25_ranking=bm25_results,
                    vector_ranking=vector_results_ranked,
                    intent_type=intent_type,
                    top_n=n_results
                )
            except Exception:
                fused = [(sid, 0.5) for sid in candidates[:n_results]]
        else:
            fused = [(sid, 0.5) for sid in candidates[:n_results]]

        # Build result cards
        cards = []
        for session_id, score in fused:
            doc = store.get_session(session_id)
            if doc:
                snippet = ""
                if doc.reasoning_docs:
                    snippet = doc.reasoning_docs[0][:120]
                elif doc.summary_doc:
                    snippet = doc.summary_doc[:120]
                elif doc.task_raw:
                    snippet = doc.task_raw[:120]

                normalized_score = min(score / (score + 0.01), 1.0) if score > 0 else 0.0
                card = build_result_card(doc, match_score=normalized_score, snippet=snippet)
                cards.append(card)

        # Supersession annotation
        try:
            from ..search.supersession_annotator import annotate_supersession
            cards = annotate_supersession(cards, store)
        except Exception:
            pass

        self.last_cards = cards
        self.last_results = cards  # For quick fork by number

        render_result_cards(cards, self.console, theme_name=self.theme_name, command_prefix="fork")

        self.console.print(f"\n[{self.theme['text_muted']}]Use [bold]fork <session_id> --intent continue|reference|debug[/{self.theme['text_muted']}]")

    do_s = do_search

    # ─────────────────────────────────────────────────────────────────
    # FORK (v2 fork-assembler with intent)
    # ─────────────────────────────────────────────────────────────────

    def do_fork(self, arg: str):
        """Generate a fork context file using v2 intent-based assembly.

        Usage: fork <session_id_or_number> [--intent continue|reference|debug] [--query Q] [--output path] [--clipboard]
        Alias: f

        If a number is provided (1-9), forks the corresponding result from last search.
        """
        if not self._ensure_metadata_store():
            return

        arg = arg.strip()
        if not arg:
            self.console.print(f"[{self.error_color}]Usage: fork <session_id_or_number> [--intent continue][/{self.error_color}]")
            return

        parts = shlex.split(arg)

        # Parse flags
        intent = "continue"
        query = ""
        output_path = None
        clipboard = False

        filtered_parts = []
        i = 0
        while i < len(parts):
            if parts[i] in ("--intent", "-i") and i + 1 < len(parts):
                intent = parts[i + 1].lower()
                i += 2
            elif parts[i] in ("--query", "-q") and i + 1 < len(parts):
                query = parts[i + 1]
                i += 2
            elif parts[i] in ("--output", "-o") and i + 1 < len(parts):
                output_path = Path(parts[i + 1])
                i += 2
            elif parts[i] in ("--clipboard", "-c"):
                clipboard = True
                i += 1
            else:
                filtered_parts.append(parts[i])
                i += 1

        session_arg = filtered_parts[0] if filtered_parts else ""

        # Resolve number to session ID
        session_id = session_arg
        if session_arg.isdigit():
            num = int(session_arg)
            if self.last_results:
                if num < 1 or num > len(self.last_results):
                    self.console.print(f"[{self.error_color}]Invalid result number. Last search has {len(self.last_results)} results.[/{self.error_color}]")
                    return
                # Handle both v2 cards and v1 result objects
                r = self.last_results[num - 1]
                session_id = getattr(r, 'session_id', r.get('session_id', ''))
                self.console.print(f"[{self.theme['text_muted']}]Selected result #{num}: {session_id[:20]}...[/{self.theme['text_muted']}]")
            elif self.last_sessions:
                if num < 1 or num > len(self.last_sessions):
                    self.console.print(f"[{self.error_color}]Invalid session number.[/{self.error_color}]")
                    return
                session_id = self.last_sessions[num - 1]["id"]
                self.console.print(f"[{self.theme['text_muted']}]Selected session #{num}: {session_id[:20]}...[/{self.theme['text_muted']}]")
            else:
                self.console.print(f"[{self.error_color}]No search results or sessions list. Run 'search' or 'sessions' first.[/{self.error_color}]")
                return

        if not session_id:
            self.console.print(f"[{self.error_color}]Session ID is required[/{self.error_color}]")
            return

        store = self.metadata_store

        # Resolve short ID (8 chars) to full session ID
        if len(session_id) <= 16:
            try:
                row = store.conn.execute(
                    "SELECT session_id FROM sessions WHERE session_id LIKE ?",
                    (session_id + "%",)
                ).fetchone()
                if row:
                    session_id = row[0]
                else:
                    self.console.print(f"[{self.error_color}]Session not found: {session_id}[/{self.error_color}]")
                    return
            except Exception:
                pass

        # Validate intent
        valid_intents = {"continue", "reference", "debug"}
        if intent not in valid_intents:
            self.console.print(f"[{self.error_color}]Invalid intent: {intent}. Use: continue, reference, or debug[/{self.error_color}]")
            return

        doc = store.get_session(session_id)

        if not doc:
            self.console.print(f"[{self.error_color}]Session not found in v2 index: {session_id}[/{self.error_color}]")
            return

        self.console.print(Panel.fit(
            f"[bold {self.info_color}]Forking session[/bold {self.info_color}]\n"
            f"Project: {doc.project_name}\n"
            f"Task: {doc.task_raw[:60]}{'...' if len(doc.task_raw) > 60 else ''}\n"
            f"Intent: {intent}",
            title=f"Fork v2 — {intent.title()}",
            border_style=self.theme["panel_border"]
        ))

        # Assemble context
        try:
            from ..fork.fork_assembler import assemble_fork_context
            from ..intelligence.llm_provider import get_llm
            from ..search.embedder import check_ollama_available, get_embedder
            from ..database.vector_index import VectorIndex

            llm = None
            vector_index = None
            ollama_check = check_ollama_available(self.config.embedding_model)
            if ollama_check["available"]:
                try:
                    llm = get_llm("ollama")
                    embedder = get_embedder("ollama", self.config.embedding_model, self.config.embedding_dimensions)
                    vi_path = self.config.chroma_db_path / "v2_index"
                    vector_index = VectorIndex(str(vi_path), embedder)
                    self.console.print(f"[{self.theme['text_muted']}]Using LLM for context distillation...[/{self.theme['text_muted']}]")
                except Exception:
                    pass

            if not llm:
                self.console.print(f"[{self.theme['text_muted']}]Ollama not running — using cleaned raw assembly[/{self.theme['text_muted']}]")

            context = assemble_fork_context(
                doc, intent, query=query, llm=llm,
                vector_index=vector_index, store=store
            )
        except Exception as e:
            self.console.print(f"[{self.error_color}]Fork assembly failed: {e}[/{self.error_color}]")
            return

        # Deliver
        if clipboard:
            try:
                from ..mcp.mcp_server import clipboard_context
                if clipboard_context(context):
                    self.console.print(f"[{self.success_color}]✓ Context copied to clipboard[/{self.success_color}]")
                else:
                    clipboard = False
            except Exception:
                clipboard = False

        if output_path or not clipboard:
            if not output_path:
                short_id = session_id[:8]
                output_path = Path(f"fork_{short_id}_{intent}.md")
            output_path.write_text(context, encoding="utf-8")
            self.console.print(f"[{self.success_color}]✓ Fork saved to:[/{self.success_color}] {output_path.absolute()}")

        # Preview
        self.console.print(f"\n[{self.theme['text_muted']}]Preview:[/{self.theme['text_muted']}]")
        preview = context[:600] + ("..." if len(context) > 600 else "")
        try:
            from ..ui.markdown_render import render_markdown_panel
            render_markdown_panel(preview, title="Fork Preview", theme_name=self.theme_name, console=self.console)
        except Exception:
            self.console.print(Panel(preview, border_style=self.theme["panel_border"]))

    do_f = do_fork

    # ─────────────────────────────────────────────────────────────────
    # FORK-V2 (explicit v2 command)
    # ─────────────────────────────────────────────────────────────────

    def do_fork_v2(self, arg: str):
        """Alias for fork command (v2 intent-based assembly).

        Usage: fork-v2 <session_id> [--intent continue|reference|debug] [--query Q]
        """
        self.do_fork(arg)

    # ─────────────────────────────────────────────────────────────────
    # RESUME (shortcut for fork with continue intent)
    # ─────────────────────────────────────────────────────────────────

    def do_resume(self, arg: str):
        """Quick fork with continue intent.

        Usage: resume <session_id_or_number> [query]
        """
        if not arg.strip():
            self.console.print(f"[{self.error_color}]Usage: resume <session_id_or_number> [query][/{self.error_color}]")
            return

        parts = shlex.split(arg)
        session_arg = parts[0]
        query = ' '.join(parts[1:]) if len(parts) > 1 else ""

        self.do_fork(f"{session_arg} --intent continue" + (f" --query {query}" if query else ""))

    # ─────────────────────────────────────────────────────────────────
    # DETECT-FORK (v2 pipeline)
    # ─────────────────────────────────────────────────────────────────

    def do_detect_fork(self, arg: str):
        """Find relevant past sessions to fork context from (v2 pipeline).

        Usage: detect-fork <query> [--results N] [--project NAME]
        Alias: df
        """
        # Delegate to search — same v2 pipeline, same result cards
        self.do_search(arg)

    do_df = do_detect_fork

    # ─────────────────────────────────────────────────────────────────
    # INDEX (v2 pipeline)
    # ─────────────────────────────────────────────────────────────────

    def do_index(self, arg: str):
        """Index sessions using the v2 structured pipeline.

        Usage: index [--force] [--skip-embeddings]
        Alias: i

        Parses all 3 Kilo Code files per session, extracts structured signals,
        stores metadata in SQLite, and embeds documents into ChromaDB.
        """
        if not self.config.kilo_code_tasks_path.exists():
            self.console.print(f"[{self.error_color}]Error: Tasks path does not exist: {self.config.kilo_code_tasks_path}[/{self.error_color}]")
            return

        parts = shlex.split(arg) if arg else []
        force = "--force" in parts or "-f" in parts
        skip_embeddings = "--skip-embeddings" in parts

        # Initialize v2 components
        from ..database.metadata_store import MetadataStore
        from ..indexer.session_parser import SessionParser
        from ..indexer.session_scanner import SessionScanner

        store = MetadataStore(self.config.sqlite_db_path)
        self.metadata_store = store

        parser = SessionParser()
        scanner = SessionScanner(self.config.kilo_code_tasks_path, store)

        if force:
            self.console.print(f"[{self.warning_color}]Resetting v2 index...[/{self.warning_color}]")
            store.reset()

        self.console.print(Panel.fit(
            f"[bold {self.info_color}]SmartFork v2 Indexer[/bold {self.info_color}]\n"
            f"Source: {self.config.kilo_code_tasks_path}\n"
            f"SQLite: {self.config.sqlite_db_path}",
            title="Index v2",
            border_style=self.theme["panel_border"]
        ))

        scan_result = scanner.scan()
        sessions_to_index = scan_result.new_session_paths + scan_result.changed_session_paths

        if not sessions_to_index:
            self.console.print(f"\n[{self.success_color}]✓ All {scan_result.total_found} sessions up to date.[/{self.success_color}]")
            store.close()
            return

        self.console.print(
            f"\n  [{self.info_color}]Found:[/{self.info_color}] {scan_result.total_found} sessions "
            f"({scan_result.new_sessions} new, {scan_result.changed_sessions} changed, "
            f"{scan_result.unchanged_sessions} unchanged)"
        )

        # Vector index
        vector_index = None
        if not skip_embeddings:
            try:
                from ..search.embedder import get_embedder, check_ollama_available
                from ..database.vector_index import VectorIndex

                ollama_status = check_ollama_available(self.config.embedding_model)
                if ollama_status["available"]:
                    embedder = get_embedder("ollama", self.config.embedding_model, self.config.embedding_dimensions)
                    self.console.print(f"  [{self.success_color}]✓ Ollama ready[/{self.success_color}] ({self.config.embedding_model})")
                else:
                    self.console.print(f"  [{self.warning_color}]⚠ Ollama not available, using sentence-transformers fallback[/{self.warning_color}]")
                    embedder = get_embedder("sentence-transformers")

                vi_path = self.config.chroma_db_path / "v2_index"
                vector_index = VectorIndex(str(vi_path), embedder)
            except Exception as e:
                self.console.print(f"  [{self.warning_color}]⚠ Embedding init failed: {e}[/{self.warning_color}]")
                self.console.print(f"  [{self.theme['text_muted']}]Continuing with metadata-only indexing[/{self.theme['text_muted']}]")

        indexed = 0
        failed = 0

        from ..ui.v2_progress import V2IndexProgress

        embed_provider = self.config.embedding_model if vector_index else ""

        with V2IndexProgress(
            total_sessions=len(sessions_to_index),
            theme_name=self.theme_name,
            console=self.console,
            embedding_provider=embed_provider,
            skip_embeddings=skip_embeddings,
        ) as progress:
            for session_path in sessions_to_index:
                progress.start_session(session_path.name)

                try:
                    doc = parser.parse(session_path)
                    progress.step_done("parse", files=len(doc.files),
                                       domains=doc.domains, languages=doc.languages)
                    progress.set_project(doc.project_name)

                    store.upsert(doc)
                    progress.step_done("store")

                    if vector_index and not skip_embeddings:
                        count = vector_index.index_document(doc)
                        progress.step_done("embed", chunks=count)
                    else:
                        progress.step_done("embed", chunks=0)

                    progress.advance()
                    indexed += 1

                except Exception as e:
                    progress.step_fail("parse", str(e))
                    failed += 1
                    self.console.print(f"[{self.error_color}]Error indexing {session_path.name}: {e}[/{self.error_color}]")

            progress.finish()

        # Supersession detection
        if indexed > 0 and vector_index:
            try:
                from ..indexer.supersession_detector import detect_supersession, detect_resolution_status, load_embeddings_from_chromadb
                from ..database.chroma_db import ChromaDatabase

                self.console.print(f"\n[{self.info_color}]Running supersession detection...[/{self.info_color}]")
                chroma_db = ChromaDatabase(self.config.chroma_db_path)
                existing = load_embeddings_from_chromadb(chroma_db)
                superseded = 0
                for session_path in sessions_to_index[:indexed]:
                    doc = parser.parse(session_path)
                    if detect_supersession(doc, existing):
                        superseded += 1
                if superseded > 0:
                    self.console.print(f"[{self.theme['text_muted']}]  Detected {superseded} supersession relationship(s)[/{self.theme['text_muted']}]")
            except Exception as e:
                self.console.print(f"[{self.theme['text_muted']}]  Supersession detection skipped: {e}[/{self.theme['text_muted']}]")

        total_db = store.get_session_count()
        self.console.print(f"\n[{self.success_color}]✓ Indexing complete! {total_db} sessions indexed.[/{self.success_color}]")

        # Refresh search engine
        if self.db:
            self.search_engine = HybridSearchEngine(self.db)
            self.fork_generator = ForkMDGenerator(self.db)

    do_i = do_index

    # ─────────────────────────────────────────────────────────────────
    # SESSIONS (v2)
    # ─────────────────────────────────────────────────────────────────

    def do_sessions(self, arg: str):
        """List all indexed sessions with IDs and titles (v2).

        Usage: sessions [--limit N]
        Alias: sl
        """
        if not self._ensure_metadata_store():
            return

        store = self.metadata_store
        if store.get_session_count() == 0:
            self.console.print(f"[{self.warning_color}]No sessions indexed. Run 'index' first.[/{self.warning_color}]")
            return

        parts = shlex.split(arg) if arg else []
        limit = 20
        for i, part in enumerate(parts):
            if part in ("--limit", "-n", "-l") and i + 1 < len(parts):
                try:
                    limit = int(parts[i + 1])
                except ValueError:
                    pass
                break

        try:
            rows = store.conn.execute(
                'SELECT session_id, task_raw, project_name, session_end, duration_minutes '
                'FROM sessions ORDER BY session_end DESC LIMIT ?',
                (limit,)
            ).fetchall()

            sessions_info = []
            for row in rows:
                sessions_info.append({
                    "id": row[0],
                    "title": (row[1] or "Untitled")[:80],
                    "project": row[2] or "Unknown",
                    "last_active": row[3] or "Unknown",
                    "duration": row[4] or 0,
                })

            self.last_sessions = sessions_info

            self.console.print(Panel.fit(
                f"[bold {self.info_color}]{len(sessions_info)} indexed sessions[/bold {self.info_color}]",
                title="Session List",
                border_style=self.theme["panel_border"]
            ))

            table = Table(show_header=True)
            table.add_column("#", style=self.accent_color, justify="right", width=4)
            table.add_column("Session ID", style=self.info_color, width=12)
            table.add_column("Title", style=self.success_color, min_width=30)
            table.add_column("Project", style=self.theme["text_muted"], width=15)
            table.add_column("Last Active", style=self.theme["text_muted"], width=12)

            for i, session in enumerate(sessions_info, 1):
                short_id = session["id"][:8]
                last_active = session["last_active"]
                if last_active and last_active != "Unknown":
                    try:
                        dt = datetime.fromisoformat(last_active)
                        last_active = dt.strftime("%Y-%m-%d")
                    except Exception:
                        pass

                table.add_row(
                    str(i),
                    short_id,
                    session["title"],
                    session["project"],
                    last_active,
                )

            self.console.print(table)
            self.console.print(f"\n[{self.theme['text_muted']}]Tip: Use 'fork <number>' to fork a session[/{self.theme['text_muted']}]")

        except Exception as e:
            self.console.print(f"[{self.error_color}]Error listing sessions: {e}[/{self.error_color}]")

    do_sl = do_sessions

    # ─────────────────────────────────────────────────────────────────
    # RESULTS (re-show last results as cards)
    # ─────────────────────────────────────────────────────────────────

    def do_results(self, arg: str):
        """Show last search results again.

        Usage: results
        """
        if not self.last_cards and not self.last_results:
            self.console.print(f"[{self.warning_color}]No previous search results.[/{self.warning_color}]")
            return

        if self.last_cards:
            render_result_cards(self.last_cards, self.console, theme_name=self.theme_name)
        else:
            # v1 results fallback
            self.console.print(f"[{self.theme['text_muted']}]Last search results ({len(self.last_results)} items):[/{self.theme['text_muted']}]\n")
            for i, r in enumerate(self.last_results, 1):
                score_pct = f"{r.score:.1%}"
                session_title = getattr(r, 'metadata', {}).get("session_title", "")
                title_text = f"[{i}] {session_title}" if session_title else f"[{i}] Session {getattr(r, 'session_id', '')[:16]}..."

                if r.score > 0.7:
                    border = self.success_color
                elif r.score > 0.4:
                    border = self.warning_color
                else:
                    border = self.error_color

                self.console.print(Panel(
                    f"[bold {self.info_color}]Score:[/bold {self.info_color}] {score_pct}",
                    title=title_text,
                    border_style=border
                ))

        self.console.print(f"\n[{self.theme['text_muted']}]Tip: Type a number [1-9] to fork that result[/{self.theme['text_muted']}]")

    # ─────────────────────────────────────────────────────────────────
    # CONFIG (full subcommand support)
    # ─────────────────────────────────────────────────────────────────

    def do_config(self, arg: str):
        """Configuration management.

        Usage: config [get|set|reset|validate|path] [args]
        Alias: c

        Subcommands:
            config                  Show all settings (categorized)
            config get <key>        Get a single value
            config set <key> <val>  Set a value
            config reset [key]      Reset to defaults
            config validate         Validate configuration
            config path             Show config file location
        """
        parts = shlex.split(arg) if arg else []

        if not parts:
            # Show categorized config (matching CLI output)
            sections = {
                "Embedding": ["embedding_provider", "embedding_model", "embedding_dimensions"],
                "LLM": ["llm_provider", "llm_model"],
                "UI": ["theme", "animation_fps", "disable_animations", "lite_mode", "adaptive_fps"],
                "Indexing": ["chunk_size", "chunk_overlap"],
                "Search": ["default_search_results", "enable_search_cache", "search_cache_size", "search_cache_ttl"],
                "Performance": ["batch_size"],
                "Logging": ["log_level", "log_file"],
                "Schema": ["schema_version"],
            }

            content = Text()
            for section, keys in sections.items():
                content.append(f"  {section}\n", style=f"bold {self.info_color}")
                for key in keys:
                    value = getattr(self.config, key, None)
                    default = getattr(SmartForkConfig(), key, None)
                    marker = "" if value == default else " [changed]"
                    content.append(f"    {key:<25} ", style=f"dim {self.theme['text_muted']}")
                    content.append(f"{value}", style=self.success_color)
                    content.append(f"{marker}\n", style="dim yellow" if marker else "dim")
                content.append("\n")

            content.append(f"  Config file: ", style=f"dim {self.theme['text_muted']}")
            content.append(str(CONFIG_FILE), style=f"dim {self.theme['bars'][2]['color']}")
            content.append("\n", style=f"dim {self.theme['text_muted']}")
            content.append(f"  Edit: ", style=f"dim {self.theme['text_muted']}")
            content.append("config set <key> <value>", style=f"bold {self.theme['bars'][0]['color']}")
            content.append("  |  ", style=f"dim {self.theme['text_muted']}")
            content.append("config reset [key]", style=f"bold {self.theme['bars'][0]['color']}")
            content.append("\n", style=f"dim {self.theme['text_muted']}")

            self.console.print(Panel(
                content,
                title="[bold]SmartFork Configuration[/bold]",
                border_style=self.theme["panel_border"],
                box=box.ROUNDED,
                padding=(1, 1),
            ))
            return

        subcommand = parts[0]

        if subcommand == "get":
            if len(parts) < 2:
                self.console.print(f"[{self.error_color}]Usage: config get <key>[/{self.error_color}]")
                return
            key = parts[1]
            if not hasattr(self.config, key):
                self.console.print(f"[{self.error_color}]Unknown config key: '{key}'[/{self.error_color}]")
                return
            value = getattr(self.config, key)
            self.console.print(f"[bold]{key}[/bold] = {value}")

        elif subcommand == "set":
            if len(parts) < 3:
                self.console.print(f"[{self.error_color}]Usage: config set <key> <value>[/{self.error_color}]")
                return
            key, value = parts[1], parts[2]
            success, message = self.config.set_value(key, value)
            if success:
                self._refresh_theme()
                self.console.print(f"[bold {self.success_color}]✓ {message}[/{self.success_color}]")
                self.console.print(f"[{self.theme['text_muted']}]Saved to {CONFIG_FILE}[/{self.theme['text_muted']}]")
            else:
                self.console.print(f"[{self.error_color}]✗ {message}[/{self.error_color}]")

        elif subcommand == "reset":
            if len(parts) >= 2:
                key = parts[1]
                try:
                    self.config.reset(key)
                    self._refresh_theme()
                    self.console.print(f"[bold {self.success_color}]✓ {key} reset to default[/{self.success_color}]")
                except ValueError as e:
                    self.console.print(f"[{self.error_color}]{e}[/{self.error_color}]")
            else:
                confirm = input("Reset ALL config values to defaults? (yes/no): ").strip().lower()
                if confirm in ("yes", "y"):
                    self.config.reset()
                    self._refresh_theme()
                    self.console.print(f"[bold {self.success_color}]✓ All config values reset[/{self.success_color}]")

        elif subcommand == "validate":
            errors = self.config.validate_all()
            if not errors:
                self.console.print(f"[bold {self.success_color}]✓ Configuration is valid[/{self.success_color}]")
            else:
                self.console.print(f"[bold {self.error_color}]✗ {len(errors)} validation error(s):[/{self.error_color}]")
                for err in errors:
                    self.console.print(f"  [{self.error_color}]• {err}[/{self.error_color}]")

        elif subcommand == "path":
            self.console.print(f"[bold]Config file:[/bold] {CONFIG_FILE}")
            if CONFIG_FILE.exists():
                size = CONFIG_FILE.stat().st_size
                self.console.print(f"[{self.theme['text_muted']}]Exists: yes ({size} bytes)[/{self.theme['text_muted']}]")
            else:
                self.console.print(f"[{self.warning_color}]Does not exist yet (using defaults)[/{self.warning_color}]")

        else:
            self.console.print(f"[{self.error_color}]Unknown subcommand: {subcommand}[/{self.error_color}]")
            self.console.print(f"[{self.theme['text_muted']}]Use: get, set, reset, validate, path[/{self.theme['text_muted']}]")

    do_c = do_config

    # ─────────────────────────────────────────────────────────────────
    # THEME
    # ─────────────────────────────────────────────────────────────────

    def do_theme(self, arg: str):
        """Set or view the color theme.

        Usage: theme [name] [--list]
        Alias: t
        """
        from ..ui.progress import THEMES, DEFAULT_THEME
        from ..config import reload_config

        parts = shlex.split(arg) if arg else []
        list_themes = "--list" in parts or "-l" in parts

        current = getattr(self.config, "theme", DEFAULT_THEME)

        theme_name = None
        for part in parts:
            if not part.startswith("-"):
                theme_name = part
                break

        if list_themes or theme_name is None:
            theme_table = Table(show_header=True, box=box.SIMPLE)
            theme_table.add_column("Theme", style="bold", width=12)
            theme_table.add_column("Description", style="dim", width=40)
            theme_table.add_column("", width=10)

            for tid, td in THEMES.items():
                c0, c1, c2 = [b["color"] for b in td["bars"]]
                swatch = f"[{c0}]▪[/{c0}][{c1}]▪[/{c1}][{c2}]▪[/{c2}] {td['name']}"
                status = f"[{self.success_color}]● active[/{self.success_color}]" if tid == current else ""
                theme_table.add_row(swatch, td["desc"], status)

            self.console.print(Panel(
                theme_table,
                title="[bold]SmartFork Themes[/bold]",
                box=box.ROUNDED,
                border_style=self.theme["panel_border"]
            ))

            if theme_name is None:
                self.console.print(f"\n  Current: [bold]{current}[/bold]")
                self.console.print(f"  Set with: [dim]theme <name>[/dim]\n")
            return

        theme_name = theme_name.lower()
        if theme_name not in THEMES:
            self.console.print(f"[{self.error_color}]Unknown theme '{theme_name}'[/{self.error_color}]")
            self.console.print(f"[{self.theme['text_muted']}]Valid: {', '.join(THEMES.keys())}[/{self.theme['text_muted']}]")
            return

        self.config.theme = theme_name
        self.config.save()

        self._refresh_theme()

        td = THEMES[theme_name]
        c = td["bars"][1]["color"]
        self.console.print(f"\n  [{c}]✓[/{c}] Theme → [bold]{td['name']}[/bold] — {td['desc']}")
        self.console.print(f"  [{self.theme['text_muted']}]Saved to config[/{self.theme['text_muted']}]")

    do_t = do_theme

    # ─────────────────────────────────────────────────────────────────
    # RESET
    # ─────────────────────────────────────────────────────────────────

    def do_reset(self, arg: str):
        """Reset the database (WARNING: deletes all indexed data).

        Usage: reset [--force]
        """
        if not self._ensure_db():
            return

        parts = shlex.split(arg) if arg else []
        force = "--force" in parts or "-f" in parts

        if not force:
            self.console.print(f"[{self.warning_color}]Warning: This will delete all indexed data![/{self.warning_color}]")
            confirm = input("Are you sure? (yes/no): ").strip().lower()
            if confirm not in ("yes", "y"):
                self.console.print(f"[{self.warning_color}]Aborted.[/{self.warning_color}]")
                return

        try:
            self.db.reset()
            self.console.print(f"[{self.success_color}]✓ Database reset complete.[/{self.success_color}]")
            self._init_database()
        except Exception as e:
            self.console.print(f"[{self.error_color}]Error resetting database: {e}[/{self.error_color}]")

    # ─────────────────────────────────────────────────────────────────
    # SUMMARIZE (NEW)
    # ─────────────────────────────────────────────────────────────────

    def do_summarize(self, arg: str):
        """Generate LLM summaries for all indexed sessions.

        Usage: summarize [--force]

        Uses Ollama (qwen3:0.6b) to create 3-sentence summaries for search ranking.
        Skips sessions that already have summaries unless --force is used.
        """
        if not self._ensure_metadata_store():
            return

        parts = shlex.split(arg) if arg else []
        force = "--force" in parts or "-f" in parts

        store = self.metadata_store

        if store.get_session_count() == 0:
            self.console.print(f"[{self.warning_color}]No v2 sessions indexed. Run 'index' first.[/{self.warning_color}]")
            return

        try:
            from ..intelligence.llm_provider import get_llm
            from ..search.embedder import check_ollama_available

            ollama_check = check_ollama_available(self.config.embedding_model)
            if not ollama_check["available"]:
                self.console.print(f"[{self.error_color}]Ollama is required for summarization.[/{self.error_color}]")
                self.console.print(f"[{self.theme['text_muted']}]Start Ollama and try again.[/{self.theme['text_muted']}]")
                return

            llm = get_llm("ollama")
        except Exception as e:
            self.console.print(f"[{self.error_color}]Failed to initialize LLM: {e}[/{self.error_color}]")
            return

        # Get sessions without summaries
        if force:
            rows = store.conn.execute('SELECT session_id, task_raw FROM sessions').fetchall()
        else:
            rows = store.conn.execute('SELECT session_id, task_raw FROM sessions WHERE summary_doc IS NULL OR summary_doc = ""').fetchall()

        if not rows:
            self.console.print(f"[{self.success_color}]✓ All sessions already have summaries. Use --force to regenerate.[/{self.success_color}]")
            return

        self.console.print(f"[{self.info_color}]Generating summaries for {len(rows)} sessions...[/{self.info_color}]")

        updated = 0
        failed = 0

        for session_id, task_raw in rows:
            try:
                prompt = f"Write a concise 3-sentence summary of this AI coding session:\n\n{task_raw[:2000]}"
                summary = llm.generate(prompt, max_tokens=150)
                if summary:
                    store.conn.execute(
                        'UPDATE sessions SET summary_doc = ? WHERE session_id = ?',
                        (summary.strip(), session_id)
                    )
                    store.conn.commit()
                    updated += 1
                else:
                    failed += 1
            except Exception:
                failed += 1

        self.console.print(f"\n[{self.success_color}]✓ Summaries generated: {updated}[/{self.success_color}]")
        if failed > 0:
            self.console.print(f"[{self.warning_color}]Failed: {failed}[/{self.warning_color}]")

    # ─────────────────────────────────────────────────────────────────
    # COMPACT
    # ─────────────────────────────────────────────────────────────────

    def do_compact(self, arg: str):
        """Check for sessions at risk of compaction or export them.

        Usage: compact [check|export] [options]

        Subcommands:
            check              Check for at-risk sessions
            export             Export at-risk sessions

        Options:
            --messages, -m N   Message count threshold (default: 100)
            --days, -d N       Age threshold in days (default: 7)
            --dry-run          Show what would be exported
            --auto, -a         Export all at-risk sessions automatically
        """
        from ..intelligence.pre_compaction import PreCompactionHook, CompactionManager

        parts = shlex.split(arg) if arg else []
        subcommand = parts[0] if parts else "check"

        threshold_messages = 100
        threshold_days = 7
        dry_run = "--dry-run" in parts
        auto = "--auto" in parts or "-a" in parts

        for i, part in enumerate(parts):
            if part in ("--messages", "-m") and i + 1 < len(parts):
                try:
                    threshold_messages = int(parts[i + 1])
                except ValueError:
                    pass
            elif part in ("--days", "-d") and i + 1 < len(parts):
                try:
                    threshold_days = int(parts[i + 1])
                except ValueError:
                    pass

        try:
            if subcommand == "check":
                hook = PreCompactionHook(threshold_messages, threshold_days)
                at_risk = hook.check_sessions(self.config.kilo_code_tasks_path)

                if not at_risk:
                    self.console.print(f"[{self.success_color}]✓ No sessions at risk of compaction.[/{self.success_color}]")
                    return

                self.console.print(Panel.fit(
                    f"[bold {self.warning_color}]{len(at_risk)} sessions at risk[/bold {self.warning_color}]",
                    title="Compaction Check",
                    border_style=self.theme["panel_border"]
                ))

                table = Table(show_header=True)
                table.add_column("Session", style=self.info_color)
                table.add_column("Messages", style=self.warning_color, justify="right")
                table.add_column("Age (days)", style=self.accent_color, justify="right")
                table.add_column("Risk", style=self.error_color)

                for session in at_risk[:20]:
                    table.add_row(
                        session["session_id"][:20],
                        str(session["message_count"]),
                        str(session["age_days"]),
                        session["risk_level"]
                    )

                self.console.print(table)

                if len(at_risk) > 20:
                    self.console.print(f"\n[{self.theme['text_muted']}]... and {len(at_risk) - 20} more[/{self.theme['text_muted']}]")

            elif subcommand == "export":
                manager = CompactionManager()
                results = manager.run_auto_export(dry_run=dry_run)

                if dry_run:
                    self.console.print(Panel.fit(
                        f"[bold {self.info_color}]{results['at_risk']} sessions would be exported[/bold {self.info_color}]",
                        title="Dry Run",
                        border_style=self.theme["panel_border"]
                    ))
                else:
                    self.console.print(Panel.fit(
                        f"[bold {self.success_color}]Exported {results['exported']} sessions[/bold {self.success_color}]\n"
                        f"Failed: {results['failed']}",
                        title="Compaction Export",
                        border_style=self.theme["panel_border"]
                    ))

                if results['sessions']:
                    table = Table(show_header=True)
                    table.add_column("Session", style=self.info_color)
                    table.add_column("Action", style=self.success_color)

                    for session in results['sessions'][:10]:
                        table.add_row(
                            session["session_id"][:20],
                            session["action"]
                        )

                    self.console.print(table)
            else:
                self.console.print(f"[{self.error_color}]Unknown subcommand: {subcommand}[/{self.error_color}]")
                self.console.print(f"[{self.theme['text_muted']}]Use 'check' or 'export'[/{self.theme['text_muted']}]")

        except Exception as e:
            self.console.print(f"[{self.error_color}]Error: {e}[/{self.error_color}]")

    # ─────────────────────────────────────────────────────────────────
    # TREE
    # ─────────────────────────────────────────────────────────────────

    def do_tree(self, arg: str):
        """Build, visualize, or export conversation branching tree.

        Usage: tree [build|visualize|export] [options]

        Subcommands:
            build              Build conversation tree
            visualize          Visualize tree (default)
            export             Export as interactive HTML

        Options:
            --session, -s ID   Root session to visualize
            --expanded, -e     Show expanded view
            --output, -o FILE  Output file for export
            --open, -b         Open in browser after export
        """
        from ..intelligence.branching import BranchingTree

        parts = shlex.split(arg) if arg else []
        subcommand = parts[0] if parts and not parts[0].startswith("-") else "visualize"

        session_id = None
        expanded = "--expanded" in parts or "-e" in parts
        output = None
        open_browser = "--open" in parts or "-b" in parts

        for i, part in enumerate(parts):
            if part in ("--session", "-s") and i + 1 < len(parts):
                session_id = parts[i + 1]
            elif part in ("--output", "-o") and i + 1 < len(parts):
                output = Path(parts[i + 1])

        try:
            tree = BranchingTree()

            if subcommand == "build":
                with self.console.status(f"[bold {self.info_color}]Building tree..."):
                    tree.auto_build_tree(self.config.kilo_code_tasks_path)

                stats = tree.get_stats()

                self.console.print(Panel.fit(
                    f"[bold {self.info_color}]{stats['total_sessions']} sessions[/bold {self.info_color}] in tree\n"
                    f"{stats['root_sessions']} roots, {stats['leaf_sessions']} leaves\n"
                    f"Max depth: {stats['max_depth']}",
                    title="Tree Built",
                    border_style=self.theme["panel_border"]
                ))

            elif subcommand == "visualize":
                stats = tree.get_stats()
                tree_text = tree.visualize_tree(session_id, compact=not expanded)

                self.console.print(Panel.fit(
                    tree_text,
                    title=f"Conversation Tree ({stats['total_sessions']} sessions, {stats['root_sessions']} roots)",
                    border_style=self.theme["panel_border"]
                ))

                self.console.print(f"\n[{self.theme['text_muted']}]Stats: {stats['leaf_sessions']} leaves, max depth {stats['max_depth']}[/{self.theme['text_muted']}]")

            elif subcommand == "export":
                with self.console.status(f"[bold {self.info_color}]Generating HTML..."):
                    html_path = tree.export_html(output)

                self.console.print(f"[{self.success_color}]✓ Tree exported to:[/{self.success_color}] {html_path}")

                stats = tree.get_stats()
                self.console.print(Panel.fit(
                    f"[bold {self.info_color}]{stats['total_sessions']}[/bold {self.info_color}] sessions\n"
                    f"[bold {self.info_color}]{stats['root_sessions']}[/bold {self.info_color}] root sessions\n"
                    f"[bold {self.info_color}]{stats['leaf_sessions']}[/bold {self.info_color}] leaf sessions\n"
                    f"Max depth: [bold {self.info_color}]{stats['max_depth']}[/bold {self.info_color}]",
                    title="Tree Statistics",
                    border_style=self.theme["panel_border"]
                ))

                if open_browser:
                    import webbrowser
                    webbrowser.open(f"file://{html_path.absolute()}")
                    self.console.print(f"[{self.theme['text_muted']}]Opened in browser[/{self.theme['text_muted']}]")
            else:
                self.console.print(f"[{self.error_color}]Unknown subcommand: {subcommand}[/{self.error_color}]")

        except Exception as e:
            self.console.print(f"[{self.error_color}]Error with tree: {e}[/{self.error_color}]")

    # ─────────────────────────────────────────────────────────────────
    # VISUALIZE (supersessions)
    # ─────────────────────────────────────────────────────────────────

    def do_visualize(self, arg: str):
        """Generate visualization of supersession relationships.

        Usage: visualize [--obsidian] [--vault-path PATH] [--project-folders]

        Options:
            --obsidian, -O       Generate Obsidian vault instead of HTML
            --vault-path, -v P   Path for Obsidian vault
            --project-folders    Organize sessions into project subfolders
        """
        from ..database.metadata_store import MetadataStore

        parts = shlex.split(arg) if arg else []
        obsidian = "--obsidian" in parts or "-O" in parts
        project_folders = "--project-folders" in parts

        vault_path = Path("./obsidian-vault")
        for i, part in enumerate(parts):
            if part in ("--vault-path", "-v") and i + 1 < len(parts):
                vault_path = Path(parts[i + 1])

        if not self._ensure_metadata_store():
            return

        store = self.metadata_store
        links = store.conn.execute('SELECT session_id, superseded_id, confidence FROM session_supersessions').fetchall()

        if not links:
            self.console.print(f"[{self.warning_color}]No supersession relationships found to visualize.[/{self.warning_color}]")
            return

        sessions = set()
        session_info = {}
        for s1, s2, conf in links:
            sessions.add(s1)
            sessions.add(s2)

        for sid in sessions:
            row = store.conn.execute('''SELECT task_raw, project_name, duration_minutes
                FROM sessions WHERE session_id = ?''', (sid,)).fetchone()
            if row:
                task_raw = row[0] or 'Unknown task'
                task = task_raw[:50]
                if len(task_raw) > 50:
                    task += '...'
                session_info[sid] = {
                    'task': task,
                    'task_full': task_raw,
                    'project': row[1] or 'Unknown',
                    'duration': row[2] or 0,
                }
            else:
                session_info[sid] = {
                    'task': f'Session {sid[:12]}',
                    'task_full': sid,
                    'project': 'Unknown',
                    'duration': 0,
                }

        if obsidian:
            self._generate_obsidian_vault(session_info, links, vault_path, project_folders, store)
            return

        # HTML visualization
        import json
        from datetime import datetime

        def get_project_color(project):
            colors = {
                'smartfork': '#06b6d4',
                'kilocode': '#8b5cf6',
                'kilo': '#ec4899',
                'default': '#64748b'
            }
            p_lower = project.lower()
            for key, color in colors.items():
                if key in p_lower:
                    return color
            return colors['default']

        nodes = []
        superseding_ids = {link[0] for link in links}
        superseded_ids = {link[1] for link in links}

        for sid, info in session_info.items():
            nodes.append({
                'id': sid,
                'task': info['task'],
                'task_full': info['task_full'],
                'project': info['project'],
                'project_color': get_project_color(info['project']),
                'duration': info['duration'],
                'is_superseding': sid in superseding_ids,
                'is_superseded': sid in superseded_ids
            })

        links_data = []
        for superseding, superseded, conf in links:
            links_data.append({
                'source': superseded,
                'target': superseding,
                'confidence': conf
            })

        data_json = json.dumps({"nodes": nodes, "links": links_data}, default=str)

        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>SmartFork Supersession Visualization</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        * {{ box-sizing: border-box; }}
        body {{
            margin: 0; padding: 0;
            font-family: 'Segoe UI', system-ui, sans-serif;
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            color: #e2e8f0; min-height: 100vh; overflow: hidden;
        }}
        #container {{ display: flex; height: 100vh; }}
        #graph-panel {{ flex: 1; position: relative; }}
        #sidebar {{
            width: 320px; background: rgba(15, 23, 42, 0.95);
            border-left: 1px solid #334155; padding: 20px;
            overflow-y: auto; backdrop-filter: blur(10px);
        }}
        h1 {{ margin: 0 0 20px 0; font-size: 18px; color: #f8fafc; }}
        .stat {{
            background: #1e293b; padding: 12px; border-radius: 8px;
            margin-bottom: 10px; border: 1px solid #334155;
        }}
        .stat-label {{ font-size: 11px; color: #94a3b8; text-transform: uppercase; }}
        .stat-value {{ font-size: 24px; font-weight: 600; color: #f8fafc; }}
        .legend {{
            background: #1e293b; padding: 12px; border-radius: 8px;
            margin-bottom: 10px; border: 1px solid #334155;
        }}
        .legend-title {{ font-size: 12px; font-weight: 600; color: #94a3b8; margin-bottom: 8px; }}
        .legend-item {{ display: flex; align-items: center; gap: 8px; margin-bottom: 4px; font-size: 12px; }}
        .legend-dot {{ width: 12px; height: 12px; border-radius: 50%; }}
        .legend-line {{ width: 30px; height: 4px; border-radius: 2px; }}
        .node {{ cursor: pointer; }}
        .node:hover {{ filter: brightness(1.3); }}
        .link {{ stroke-opacity: 0.6; }}
        .tooltip {{
            position: absolute; background: rgba(15, 23, 42, 0.95);
            border: 1px solid #475569; border-radius: 8px; padding: 12px;
            font-size: 12px; pointer-events: none; max-width: 300px;
            backdrop-filter: blur(10px); z-index: 100;
        }}
    </style>
</head>
<body>
<div id="container">
    <div id="graph-panel">
        <svg id="graph"></svg>
        <div class="tooltip" id="tooltip" style="display:none;"></div>
    </div>
    <div id="sidebar">
        <h1>Supersession Graph</h1>
        <div class="stat">
            <div class="stat-label">Sessions</div>
            <div class="stat-value">{len(nodes)}</div>
        </div>
        <div class="stat">
            <div class="stat-label">Supersession Links</div>
            <div class="stat-value">{len(links_data)}</div>
        </div>
        <div class="legend">
            <div class="legend-title">Node Types</div>
            <div class="legend-item">
                <div class="legend-dot" style="background: #10b981;"></div>
                <span>Superseding (newer)</span>
            </div>
            <div class="legend-item">
                <div class="legend-dot" style="background: #f59e0b;"></div>
                <span>Superseded (older)</span>
            </div>
            <div class="legend-item">
                <div class="legend-dot" style="background: #64748b;"></div>
                <span>Both</span>
            </div>
        </div>
    </div>
</div>
<script>
const data = {data_json};
const width = document.getElementById('graph-panel').clientWidth;
const height = document.getElementById('graph-panel').clientHeight;

const svg = d3.select('#graph').attr('width', width).attr('height', height);
const g = svg.append('g');

const zoom = d3.zoom().scaleExtent([0.1, 4]).on('zoom', (e) => g.attr('transform', e.transform));
svg.call(zoom);

const simulation = d3.forceSimulation(data.nodes)
    .force('link', d3.forceLink(data.links).id(d => d.id).distance(120))
    .force('charge', d3.forceManyBody().strength(-400))
    .force('center', d3.forceCenter(width / 2, height / 2))
    .force('collision', d3.forceCollide().radius(40));

const link = g.append('g').selectAll('line').data(data.links).join('line')
    .attr('class', 'link')
    .attr('stroke', d => `hsl(${{200 + d.confidence * 60}}, 70%, 60%)`)
    .attr('stroke-width', d => 1 + d.confidence * 3);

const node = g.append('g').selectAll('circle').data(data.nodes).join('circle')
    .attr('class', 'node')
    .attr('r', d => 8 + d.duration / 10)
    .attr('fill', d => {{
        if (d.is_superseding && d.is_superseded) return '#64748b';
        if (d.is_superseding) return '#10b981';
        return '#f59e0b';
    }})
    .call(d3.drag().on('start', (e, d) => {{ if (!e.active) simulation.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; }})
        .on('drag', (e, d) => {{ d.fx = e.x; d.fy = e.y; }})
        .on('end', (e, d) => {{ if (!e.active) simulation.alphaTarget(0); d.fx = null; d.fy = null; }}));

const tooltip = document.getElementById('tooltip');
node.on('mouseover', (e, d) => {{
    tooltip.style.display = 'block';
    tooltip.innerHTML = `<b>${{d.task}}</b><br>Project: ${{d.project}}<br>ID: ${{d.id.substring(0,16)}}...`;
}})
.on('mouseout', () => {{ tooltip.style.display = 'none'; }});

simulation.on('tick', () => {{
    link.attr('x1', d => d.source.x).attr('y1', d => d.source.y)
        .attr('x2', d => d.target.x).attr('y2', d => d.target.y);
    node.attr('cx', d => d.x).attr('cy', d => d.y);
}});
</script>
</body>
</html>"""

        output_file = "supersession_visualization.html"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)

        self.console.print(f"[{self.success_color}]Visualization saved to {output_file}. Opening in browser...[/{self.success_color}]")

        import webbrowser
        webbrowser.open(f"file://{Path(output_file).resolve()}")

    def _generate_obsidian_vault(self, session_info, links, vault_path, project_folders, store):
        """Generate Obsidian vault for supersession visualization."""
        vault_path.mkdir(parents=True, exist_ok=True)
        sessions_dir = vault_path / "Sessions"
        sessions_dir.mkdir(exist_ok=True)

        if project_folders:
            projects_dir = vault_path / "Projects"
            projects_dir.mkdir(exist_ok=True)

        for sid, info in session_info.items():
            superseded_by = [l[0] for l in links if l[1] == sid]
            supersedes = [l[1] for l in links if l[0] == sid]

            content = f"# {info['task']}\n\n"
            content += f"**Session ID:** `{sid}`\n"
            content += f"**Project:** {info['project']}\n"
            content += f"**Duration:** {info['duration']} min\n\n"

            if supersedes:
                content += "## Supersedes\n"
                for s in supersedes:
                    content += f"- [[{s[:16]}]]\n"
                content += "\n"

            if superseded_by:
                content += "## Superseded By\n"
                for s in superseded_by:
                    content += f"- [[{s[:16]}]]\n"
                content += "\n"

            content += f"## Full Session ID\n`{sid}`\n"

            note_file = sessions_dir / f"{sid[:16]}.md"
            note_file.write_text(content, encoding='utf-8')

        # MOC
        moc = "# Supersession Map of Contents\n\n```dataview\nTABLE duration, project\nFROM \"Sessions\"\nSORT file.name DESC\n```\n\n"
        for sid in session_info:
            moc += f"- [[Sessions/{sid[:16]}]]\n"
        (vault_path / "MOC.md").write_text(moc, encoding='utf-8')

        self.console.print(f"[{self.success_color}]Obsidian vault generated at:[/{self.success_color}] {vault_path}")

    # ─────────────────────────────────────────────────────────────────
    # TEST
    # ─────────────────────────────────────────────────────────────────

    def do_test(self, arg: str):
        """Run SmartFork tests.

        Usage: test [--suite SUITE]

        Options:
            --suite, -s SUITE    Test suite to run (indexer, search, database, fork)
        """
        from ..testing.test_runner import create_default_test_runner

        parts = shlex.split(arg) if arg else []
        suite = None

        for i, part in enumerate(parts):
            if part in ("--suite", "-s") and i + 1 < len(parts):
                suite = parts[i + 1]
                break

        try:
            runner = create_default_test_runner()

            if suite:
                with self.console.status(f"[bold {self.info_color}]Running {suite} tests..."):
                    result = runner.run_suite(suite)
                suites = [result]
            else:
                with self.console.status(f"[bold {self.info_color}]Running all tests..."):
                    suites = runner.run_all()

            for suite_result in suites:
                suite_color = self.success_color if suite_result.failed_count == 0 else self.error_color
                self.console.print(Panel.fit(
                    f"[bold {suite_color}]{suite_result.passed_count}/{len(suite_result.tests)} passed[/bold {suite_color}]\n"
                    f"Duration: {suite_result.total_duration_ms:.0f}ms",
                    title=f"Test Suite: {suite_result.name}",
                    border_style=self.theme["panel_border"]
                ))

                if suite_result.failed_count > 0:
                    table = Table(show_header=True)
                    table.add_column("Test", style=self.info_color)
                    table.add_column("Status", style=self.error_color)
                    table.add_column("Error", style=self.theme["text_muted"])

                    for test in suite_result.tests:
                        if not test.passed:
                            error_msg = test.error_message or ""
                            table.add_row(
                                test.name,
                                "FAILED",
                                error_msg[:50] + "..." if len(error_msg) > 50 else error_msg
                            )

                    self.console.print(table)

            summary = runner.get_summary()
            self.console.print(f"\n[{self.theme['text_muted']}]Total: {summary['passed']}/{summary['total_tests']} passed "
                               f"({summary['pass_rate']:.1%})[/{self.theme['text_muted']}]")

        except Exception as e:
            self.console.print(f"[{self.error_color}]Error running tests: {e}[/{self.error_color}]")

    # ─────────────────────────────────────────────────────────────────
    # METRICS
    # ─────────────────────────────────────────────────────────────────

    def do_metrics(self, arg: str):
        """Show success metrics dashboard.

        Usage: metrics [--days N]

        Options:
            --days, -d N    Number of days to show (default: 7)
        """
        from ..testing.metrics_tracker import MetricsTracker

        parts = shlex.split(arg) if arg else []
        days = 7

        for i, part in enumerate(parts):
            if part in ("--days", "-d") and i + 1 < len(parts):
                try:
                    days = int(parts[i + 1])
                except ValueError:
                    pass
                break

        try:
            tracker = MetricsTracker()
            data = tracker.get_dashboard_data(days)

            self.console.print(Panel.fit(
                f"[bold {self.info_color}]Success Metrics[/bold {self.info_color}] (last {data['period_days']} days)",
                title="Metrics Dashboard",
                border_style=self.theme["panel_border"]
            ))

            table = Table(show_header=True)
            table.add_column("Metric", style=self.info_color)
            table.add_column("Value", style=self.success_color)

            km = data['key_metrics']
            table.add_row("Unique Sessions", str(data['unique_sessions']))
            table.add_row("Avg Fork Gen Time", f"{km['avg_fork_generation_time_ms']:.0f}ms")
            table.add_row("Context Recovered", f"{km['total_context_recovered_mb']:.1f}MB")
            table.add_row("Sessions/Day", f"{km['sessions_per_day']:.1f}")

            self.console.print(table)

            if data['metric_summaries']:
                self.console.print(f"\n[bold {self.info_color}]Metric Trends:[/bold {self.info_color}]")
                for name, summary in data['metric_summaries'].items():
                    trend_color_map = {
                        'improving': self.success_color,
                        'stable': self.warning_color,
                        'degrading': self.error_color,
                        'insufficient_data': self.theme["text_muted"]
                    }
                    trend_color = trend_color_map.get(summary['trend'], self.theme["text_primary"])

                    self.console.print(f"  {name}: {summary['mean']:.2f} "
                                       f"([{trend_color}]{summary['trend']}[/{trend_color}])")

        except Exception as e:
            self.console.print(f"[{self.error_color}]Error getting metrics: {e}[/{self.error_color}]")

    # ─────────────────────────────────────────────────────────────────
    # TITLES
    # ─────────────────────────────────────────────────────────────────

    def do_titles(self, arg: str):
        """Generate or update session titles for all indexed sessions.

        Usage: titles [--force] [--dry-run]

        Options:
            --force, -f     Force regeneration of all titles
            --dry-run, -d   Show what would be generated without updating
        """
        from ..intelligence.titling import TitleManager, TitleGenerator
        from ..indexer.parser import KiloCodeParser
        from ..indexer.indexer import FullIndexer
        import time

        if not self._ensure_db():
            return

        parts = shlex.split(arg) if arg else []
        force = "--force" in parts or "-f" in parts
        dry_run = "--dry-run" in parts or "-d" in parts

        try:
            session_ids = self.db.get_unique_sessions()

            if not session_ids:
                self.console.print(f"[{self.warning_color}]No sessions found in database.[/{self.warning_color}]")
                return

            self.console.print(Panel.fit(
                f"[bold {self.info_color}]Update Session Titles[/bold {self.info_color}]\n"
                f"Found {len(session_ids)} sessions to process",
                title="SmartFork",
                border_style=self.theme["panel_border"]
            ))

            title_gen = TitleGenerator()
            title_manager = TitleManager(self.db, title_gen)
            parser = KiloCodeParser()

            updated = 0
            skipped = 0
            failed = 0

            with self.console.status(f"[bold {self.info_color}]Generating titles...") as status:
                for i, session_id in enumerate(session_ids):
                    try:
                        if not force:
                            chunks = self.db.get_session_chunks(session_id)
                            if chunks and chunks[0].metadata.session_title:
                                skipped += 1
                                continue

                        task_dir = self.config.kilo_code_tasks_path / session_id
                        if not task_dir.exists():
                            failed += 1
                            continue

                        session = parser.parse_task_directory(task_dir)
                        if not session:
                            failed += 1
                            continue

                        title = title_manager.generate_and_store_title(session)

                        if dry_run:
                            self.console.print(f"[{self.theme['text_muted']}]{session_id[:16]}...[/{self.theme['text_muted']}] -> {title}")
                        else:
                            indexer = FullIndexer(
                                self.db,
                                chunk_size=self.config.chunk_size,
                                chunk_overlap=self.config.chunk_overlap,
                                batch_size=self.config.batch_size
                            )
                            indexer.index_session(task_dir)
                            indexer.finalize()

                        updated += 1

                        if (i + 1) % 10 == 0:
                            status.update(f"[bold {self.info_color}]Processed {i + 1}/{len(session_ids)} sessions...")

                        if self.config.lite_mode and i % 5 == 0:
                            time.sleep(0.1)

                    except Exception as e:
                        failed += 1

            self.console.print(f"\n[bold {self.info_color}]Results:[/bold {self.info_color}]")
            self.console.print(f"  [{self.success_color}]Updated:[/{self.success_color}] {updated}")
            self.console.print(f"  [{self.warning_color}]Skipped:[/{self.warning_color}] {skipped}")
            if failed > 0:
                self.console.print(f"  [{self.error_color}]Failed:[/{self.error_color}] {failed}")

            if dry_run:
                self.console.print(f"\n[{self.theme['text_muted']}]This was a dry run. Use without --dry-run to apply changes.[/{self.theme['text_muted']}]")
            else:
                self.console.print(f"\n[{self.success_color}]✓ Title update complete![/{self.success_color}]")

        except Exception as e:
            self.console.print(f"[{self.error_color}]Error updating titles: {e}[/{self.error_color}]")

    # ─────────────────────────────────────────────────────────────────
    # WATCH
    # ─────────────────────────────────────────────────────────────────

    def do_watch(self, arg: str):
        """Watch for session changes and index incrementally.

        Usage: watch

        Press Ctrl+C to stop watching.
        """
        from ..indexer.watcher import TranscriptWatcher
        from ..indexer.indexer import IncrementalIndexer
        import time

        if not self.config.kilo_code_tasks_path.exists():
            self.console.print(f"[{self.error_color}]Error: Tasks path does not exist: {self.config.kilo_code_tasks_path}[/{self.error_color}]")
            return

        if self.db is None:
            self.db = ChromaDatabase(self.config.chroma_db_path)

        self.console.print(f"[bold {self.info_color}]Starting watcher... Press Ctrl+C to stop.[/{self.info_color}]\n")

        poll_interval = 10.0 if self.config.lite_mode else 5.0
        if self.config.lite_mode:
            self.console.print(f"  [dim]Lite mode: using {poll_interval}s poll interval[/dim]\n")

        incremental = IncrementalIndexer(self.db)
        watcher = TranscriptWatcher(
            self.config.kilo_code_tasks_path,
            incremental.on_session_changed,
            poll_interval=poll_interval
        )

        watcher.start()

        try:
            while True:
                sleep_interval = 2.0 if self.config.lite_mode else 1.0
                time.sleep(sleep_interval)
        except KeyboardInterrupt:
            self.console.print(f"\n[{self.warning_color}]Stopping watcher...[/{self.warning_color}]")
            watcher.stop()
            self.console.print(f"[{self.success_color}]✓ Watcher stopped.[/{self.success_color}]")

    # ─────────────────────────────────────────────────────────────────
    # DIAGNOSE (setup verification)
    # ─────────────────────────────────────────────────────────────────

    def do_diagnose(self, arg: str):
        """Verify SmartFork setup — check Ollama, embeddings, DB, paths.

        Usage: diagnose
        """
        from ..search.embedder import check_ollama_available

        checks = []

        # 1. Tasks path
        tasks_ok = self.config.kilo_code_tasks_path.exists()
        checks.append(("Tasks path", str(self.config.kilo_code_tasks_path), tasks_ok))

        # 2. SQLite DB
        sqlite_ok = self.config.sqlite_db_path.exists()
        checks.append(("SQLite DB", str(self.config.sqlite_db_path), sqlite_ok))

        # 3. ChromaDB
        chroma_ok = self.config.chroma_db_path.exists()
        checks.append(("ChromaDB", str(self.config.chroma_db_path), chroma_ok))

        # 4. Ollama
        ollama_check = check_ollama_available(self.config.embedding_model)
        ollama_ok = ollama_check["available"]
        checks.append(("Ollama", self.config.embedding_model, ollama_ok))

        # 5. LLM
        try:
            from ..intelligence.llm_provider import get_llm
            llm = get_llm("ollama")
            llm_ok = llm is not None
        except Exception:
            llm_ok = False
        checks.append(("LLM", self.config.llm_model, llm_ok))

        # 6. MetadataStore
        ms_ok = self._ensure_metadata_store()
        session_count = 0
        if ms_ok:
            session_count = self.metadata_store.get_session_count()
        checks.append(("MetadataStore", f"{session_count} sessions", ms_ok))

        # Display
        self.console.print(f"[bold {self.info_color}]SmartFork Diagnostics[/bold {self.info_color}]\n")

        for name, detail, ok in checks:
            icon = f"[bold {self.success_color}]✓[/bold {self.success_color}]" if ok else f"[bold {self.error_color}]✗[/bold {self.error_color}]"
            self.console.print(f"  {icon}  {name:15s} {detail}")

        all_ok = all(ok for _, _, ok in checks)
        self.console.print()
        if all_ok:
            self.console.print(f"[{self.success_color}]✓ All systems operational[/{self.success_color}]")
        else:
            self.console.print(f"[{self.warning_color}]⚠ Some checks failed — review above[/{self.warning_color}]")

    # ─────────────────────────────────────────────────────────────────
    # HISTORY
    # ─────────────────────────────────────────────────────────────────

    def do_history(self, arg: str):
        """Show command history."""
        self.console.print(f"[{self.theme['text_muted']}]Use Up/Down arrow keys to navigate command history[/{self.theme['text_muted']}]")

    # ─────────────────────────────────────────────────────────────────
    # TAB COMPLETION
    # ─────────────────────────────────────────────────────────────────

    def complete_search(self, text: str, line: str, begidx: int, endidx: int) -> List[str]:
        """Tab completion for search command."""
        return []

    def complete_fork(self, text: str, line: str, begidx: int, endidx: int) -> List[str]:
        """Tab completion for fork command - suggest session IDs."""
        if not self.db:
            return []
        try:
            sessions = self.db.get_unique_sessions()
            return [s for s in sessions if s.startswith(text)]
        except Exception:
            return []

    def complete_smartfork(self, text: str, line: str, begidx: int, endidx: int) -> List[str]:
        """Tab completion for smartfork command - suggest session IDs."""
        if not self.db:
            return []
        try:
            sessions = self.db.get_unique_sessions()
            return [s for s in sessions if s.startswith(text)]
        except Exception:
            return []

    def complete_resume(self, text: str, line: str, begidx: int, endidx: int) -> List[str]:
        """Tab completion for resume command - suggest session IDs."""
        if not self.db:
            return []
        try:
            sessions = self.db.get_unique_sessions()
            return [s for s in sessions if s.startswith(text)]
        except Exception:
            return []

    def completedefault(self, text: str, line: str, begidx: int, endidx: int) -> List[str]:
        """Default tab completion."""
        return []

    # ─────────────────────────────────────────────────────────────────
    # HELP
    # ─────────────────────────────────────────────────────────────────

    def do_help(self, arg: str):
        """Show help information."""
        if arg:
            super().do_help(arg)
            return

        # Auto-discover available commands from do_* methods
        commands = {}
        for name in dir(self):
            if name.startswith('do_') and name not in ('do_EOF',):
                cmd_name = name[3:]
                method = getattr(self, name)
                doc = method.__doc__ or ''
                first_line = doc.strip().split('\n')[0] if doc.strip() else ''
                commands[cmd_name] = first_line

        # Known aliases and their targets
        alias_map = {
            's': 'search', 'sl': 'sessions', 'f': 'fork',
            'df': 'detect-fork', 'st': 'status', 't': 'theme',
            'c': 'config', 'i': 'index',
        }
        main_commands = {k: v for k, v in commands.items() if k not in alias_map}

        # Group commands logically
        groups = {
            'Core Commands': ['search', 'sessions', 'fork', 'resume', 'index'],
            'Database & Configuration': ['status', 'config', 'theme', 'reset', 'summarize'],
            'Intelligence Features': ['compact', 'tree', 'titles', 'watch', 'visualize'],
            'Diagnostics': ['diagnose', 'test', 'metrics'],
            'Utility': ['results', 'history', 'clear', 'help', 'exit', 'quit'],
        }

        help_lines = []
        help_lines.append(f"[bold {self.info_color}]Available Commands:[/bold {self.info_color}]")

        for group_name, group_cmds in groups.items():
            help_lines.append(f"\n[bold {self.theme['text_primary']}]{group_name}:[/bold {self.theme['text_primary']}]")
            for cmd in group_cmds:
                if cmd in main_commands:
                    desc = main_commands[cmd]
                    # Find aliases for this command
                    aliases = [a for a, t in alias_map.items() if t == cmd]
                    alias_str = f" (alias: {', '.join(aliases)})" if aliases else ""
                    help_lines.append(f"  {cmd:22s} {desc}{alias_str}")

        help_lines.append(f"\n[bold {self.theme['text_primary']}]Quick Tips:[/bold {self.theme['text_primary']}]")
        help_lines.append("  \u2022 After search/detect-fork, type [1-9] to quickly fork that result")
        help_lines.append("  \u2022 Use Up/Down arrows for command history")
        help_lines.append("  \u2022 Press Tab for command and session ID completion")
        help_lines.append("  \u2022 Type 'help <command>' for detailed help on a specific command")
        help_lines.append("  \u2022 Use 'fork <id> --intent debug' for debugging sessions")
        help_lines.append("")

        help_text = "\n".join(help_lines)

        self.console.print(Panel(
            help_text,
            title="SmartFork Interactive Shell Help",
            border_style=self.theme["panel_border"]
        ))


def start_interactive_shell():
    """Start the interactive shell."""
    shell = SmartForkShell()
    try:
        shell.cmdloop()
    except KeyboardInterrupt:
        print("\n")
        shell.console.print(f"[{shell.success_color}]Goodbye![/{shell.success_color}]")
    except Exception as e:
        if "readline" in str(e).lower() or "backend" in str(e).lower():
            shell.console.print(f"[{shell.warning_color}]Note: Advanced editing features not available on this terminal.[/{shell.warning_color}]")
            shell.console.print(f"[{shell.theme['text_muted']}]Basic interactive mode starting...[/{shell.theme['text_muted']}]\n")
            shell.use_rawinput = False
            try:
                shell.cmdloop()
            except KeyboardInterrupt:
                print("\n")
                shell.console.print(f"[{shell.success_color}]Goodbye![/{shell.success_color}]")
        else:
            raise


if __name__ == "__main__":
    start_interactive_shell()
