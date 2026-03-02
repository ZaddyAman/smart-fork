# SmartFork

AI Session Intelligence for Kilo Code - Never lose context across coding sessions.

## Overview

SmartFork indexes your Kilo Code conversation history and enables intelligent search across all your past coding sessions. When you start a new task, SmartFork helps you discover and fork context from relevant previous sessions.

## Features

- **Semantic Search**: Find sessions by natural language queries
- **Technology Detection**: Automatically tags sessions with technologies discussed
- **Incremental Indexing**: Watches for new sessions and indexes them automatically
- **Offline-First**: All data stored locally, works without internet
- **CLI Interface**: Simple commands for indexing and searching

## Installation

```bash
pip install -e .
```

## Quick Start

### 1. Index Your Sessions

```bash
smartfork index
```

This will scan your Kilo Code tasks directory and index all conversations.

### 2. Search for Relevant Sessions

```bash
smartfork search "authentication JWT"
```

### 3. Detect Fork Context

```bash
smartfork detect-fork "implementing OAuth2 flow"
```

This finds the most relevant past sessions for your current task.

### 4. Watch for Changes

```bash
smartfork watch
```

Automatically indexes new sessions as they're created.

## Commands

- `smartfork index [-f]` - Index all sessions (use `-f` to force re-index)
- `smartfork search <query>` - Search indexed sessions
- `smartfork detect-fork <query>` - Find relevant sessions to fork from
- `smartfork status` - Show indexing status
- `smartfork watch` - Watch for new sessions
- `smartfork config-show` - Show current configuration
- `smartfork reset [-f]` - Reset the database

## Configuration

SmartFork uses sensible defaults but can be configured via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `SMARTFORK_KILO_CODE_TASKS_PATH` | Auto-detected | Path to Kilo Code tasks |
| `SMARTFORK_CHROMA_DB_PATH` | `~/.smartfork/chroma_db` | Vector database location |
| `SMARTFORK_CHUNK_SIZE` | 512 | Chunk size for indexing |
| `SMARTFORK_CHUNK_OVERLAP` | 128 | Overlap between chunks |
| `SMARTFORK_LOG_LEVEL` | INFO | Logging level |

## Project Structure

```
smartfork/
├── src/smartfork/
│   ├── cli.py              # CLI entry point
│   ├── config.py           # Configuration
│   ├── database/
│   │   ├── chroma_db.py    # ChromaDB integration
│   │   └── models.py       # Data models
│   ├── indexer/
│   │   ├── parser.py       # Kilo Code transcript parser
│   │   ├── watcher.py      # File system watcher
│   │   └── indexer.py      # Indexing logic
│   └── search/
│       └── semantic.py     # Semantic search
├── tests/                  # Test suite
└── data/                   # Local data storage
```

## Phase 0 Status

Phase 0 provides the foundation:
- ✅ Offline-first ChromaDB storage
- ✅ Kilo Code transcript parsing
- ✅ Basic semantic search
- ✅ File system watcher
- ✅ CLI interface
- ✅ Incremental indexing

## Development

### Running Tests

```bash
pip install -e ".[dev]"
pytest
```

### Project Roadmap

- **Phase 1**: Hybrid search (semantic + keyword + recency)
- **Phase 2**: Fork.md generation
- **Phase 3**: Pre-compaction hooks and deduplication
- **Phase 4**: Advanced features

## License

MIT License - See LICENSE file for details.
