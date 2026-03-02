# SmartFork Phase 1: Hybrid MVP Implementation Plan

## Overview

Phase 1 builds upon the Phase 0 foundation to deliver the core SmartFork experience: intelligent session discovery through hybrid search and context forking. This phase transforms the basic indexer into a powerful context retrieval system.

**Prerequisites**: Phase 0 complete (transcript watcher, ChromaDB, basic CLI)
**Timeline**: 3 weeks
**Success Criteria**: End-to-end fork in <10 seconds, 85%+ relevance accuracy

---

## 1. Hybrid Search Engine

### 1.1 Architecture

The hybrid search combines four signals with weighted scoring:

```python
class HybridSearchEngine:
    """
    Combines multiple search signals for optimal relevance.
    
    Final Score = (semantic * 0.50) + (bm25 * 0.25) + (recency * 0.15) + (path_match * 0.10)
    """
    
    WEIGHTS = {
        'semantic': 0.50,
        'keyword': 0.25,
        'recency': 0.15,
        'path': 0.10
    }
```

### 1.2 Semantic Search (50%)

**Implementation**:
```python
class SemanticSearch:
    def __init__(self, collection):
        self.collection = collection
        self.embedding_fn = self._load_nomic_embed()
    
    def search(self, query: str, n_results: int = 20) -> List[SearchResult]:
        # Embed query using nomic-embed-text-v1.5
        query_embedding = self.embedding_fn(query)
        
        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=['documents', 'metadatas', 'distances']
        )
        
        return self._normalize_scores(results)
```

**Configuration**:
- Model: `nomic-embed-text-v1.5` (8,192 token window)
- Distance metric: Cosine similarity
- Chunk size: 512 tokens with 128 overlap
- Store full conversation context in metadata

### 1.3 BM25 Keyword Search (25%)

**Implementation**:
```python
from rank_bm25 import BM25Okapi

class BM25Search:
    def __init__(self, tokenized_corpus: List[List[str]]):
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.corpus_ids = []  # Maps index to session_id
    
    def search(self, query: str, n_results: int = 20) -> List[SearchResult]:
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        
        # Normalize scores to 0-1 range
        max_score = max(scores) if max(scores) > 0 else 1
        normalized = [(idx, score/max_score) for idx, score in enumerate(scores)]
        
        # Return top N
        normalized.sort(key=lambda x: x[1], reverse=True)
        return [
            SearchResult(
                session_id=self.corpus_ids[idx],
                score=score,
                source='bm25'
            )
            for idx, score in normalized[:n_results]
        ]
```

**Text Preprocessing**:
- Tokenize on whitespace and punctuation
- Lowercase all terms
- Preserve code identifiers (camelCase, snake_case)
- Remove stop words (common English words)
- Boost importance of: error messages, function names, file paths

### 1.4 Recency Weighting (15%)

**Implementation**:
```python
class RecencyScorer:
    """
    Exponential decay based on session age.
    Sessions from last 7 days get maximum score.
    Decay rate: 50% per month after 7 days.
    """
    
    MAX_AGE_DAYS = 7
    HALF_LIFE_DAYS = 30
    
    def score(self, last_active: datetime) -> float:
        age_days = (datetime.now() - last_active).days
        
        if age_days <= self.MAX_AGE_DAYS:
            return 1.0
        
        # Exponential decay
        decay_days = age_days - self.MAX_AGE_DAYS
        return 0.5 ** (decay_days / self.HALF_LIFE_DAYS)
```

### 1.5 Project Path Matching (10%)

**Implementation**:
```python
class PathMatcher:
    """
    Boosts sessions that worked on files in the current directory tree.
    """
    
    def score(self, session_paths: List[str], current_dir: str) -> float:
        if not current_dir or not session_paths:
            return 0.0
        
        current_parts = Path(current_dir).parts
        matches = 0
        
        for session_path in session_paths:
            session_parts = Path(session_path).parts
            
            # Count common path components
            common = 0
            for a, b in zip(current_parts, session_parts):
                if a == b:
                    common += 1
                else:
                    break
            
            # Score based on overlap depth
            if common > 0:
                matches += common / len(current_parts)
        
        return min(matches / len(session_paths), 1.0) if session_paths else 0.0
```

### 1.6 Combined Scoring

```python
class HybridSearchEngine:
    def search(self, query: str, current_dir: str = None) -> List[HybridResult]:
        # Get results from each component
        semantic_results = self.semantic.search(query, n_results=50)
        bm25_results = self.bm25.search(query, n_results=50)
        
        # Combine into unified scoring
        all_session_ids = set()
        for r in semantic_results + bm25_results:
            all_session_ids.add(r.session_id)
        
        combined = []
        for session_id in all_session_ids:
            # Get individual scores
            sem_score = self._get_score(semantic_results, session_id)
            bm25_score = self._get_score(bm25_results, session_id)
            
            # Get session metadata
            metadata = self.db.get_session_metadata(session_id)
            
            # Calculate recency score
            rec_score = self.recency.score(metadata.last_active)
            
            # Calculate path match score
            path_score = 0.0
            if current_dir and metadata.files_in_context:
                path_score = self.path_matcher.score(
                    metadata.files_in_context,
                    current_dir
                )
            
            # Weighted combination
            final_score = (
                self.WEIGHTS['semantic'] * sem_score +
                self.WEIGHTS['keyword'] * bm25_score +
                self.WEIGHTS['recency'] * rec_score +
                self.WEIGHTS['path'] * path_score
            )
            
            combined.append(HybridResult(
                session_id=session_id,
                score=final_score,
                breakdown={
                    'semantic': sem_score,
                    'bm25': bm25_score,
                    'recency': rec_score,
                    'path': path_score
                },
                metadata=metadata
            ))
        
        # Sort by final score
        combined.sort(key=lambda x: x.score, reverse=True)
        return combined[:10]  # Return top 10
```

---

## 2. /detect-fork Command

### 2.1 CLI Interface

```bash
# Basic usage
smartfork detect-fork "implement JWT authentication"

# With filters
smartfork detect-fork "database migration" --path ./src --days 30

# Interactive mode
smartfork detect-fork --interactive

# Export results
smartfork detect-fork "error handling" --json > results.json
```

### 2.2 Implementation

```python
@app.command()
def detect_fork(
    query: str = typer.Argument(..., help="Search query describing your intent"),
    path: Optional[Path] = typer.Option(None, "--path", "-p", help="Current working directory for path matching"),
    days: int = typer.Option(90, "--days", "-d", help="Limit to sessions from last N days"),
    interactive: bool = typer.Option(False, "--interactive", "-i", help="Interactive selection mode"),
    json_output: bool = typer.Option(False, "--json", "-j", help="Output as JSON"),
    limit: int = typer.Option(5, "--limit", "-n", help="Number of results to show"),
):
    """Find relevant past sessions for context forking."""
    
    # Initialize search engine
    engine = HybridSearchEngine()
    
    # Get current directory if not provided
    if not path:
        path = Path.cwd()
    
    # Perform search
    console.print(f"[dim]Searching for: {query}...[/dim]")
    results = engine.search(query, current_dir=str(path))
    
    # Filter by recency if specified
    if days:
        cutoff = datetime.now() - timedelta(days=days)
        results = [r for r in results if r.metadata.last_active > cutoff]
    
    # Display results
    if json_output:
        print_json(data=[r.to_dict() for r in results[:limit]])
    else:
        display_results_table(results[:limit])
    
    # Interactive mode
    if interactive and results:
        selected = interactive_select(results[:limit])
        if selected:
            generate_fork_md(selected)
```

### 2.3 Output Format

```
┌─────────────────────────────────────────────────────────────────────┐
│ Found 5 relevant sessions (searched 127 total)                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│ [1] Score: 94%                                                      │
│     Title: JWT Authentication with Refresh Tokens                   │
│     Files: src/auth/jwt_handler.py, src/auth/middleware.py          │
│     Last Active: 3 days ago                                         │
│     Match: High semantic similarity + recent activity               │
│                                                                     │
│ [2] Score: 87%                                                      │
│     Title: OAuth2 Integration Implementation                        │
│     Files: src/oauth/provider.py, tests/test_oauth.py               │
│     Last Active: 12 days ago                                        │
│     Match: Strong keyword match (OAuth, JWT)                        │
│                                                                     │
│ [3] Score: 82%                                                      │
│     Title: API Security Hardening                                   │
│     Files: src/middleware/security.py                               │
│     Last Active: 5 days ago                                         │
│     Match: Same project directory + recent                          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

Run `smartfork fork <id>` to generate context file
```

---

## 3. /fork.md Generation

### 3.1 Context File Structure

```markdown
# Context Fork: Session {session_id}

## Session Overview
- **Original Title**: JWT Authentication Implementation
- **Date Range**: 2024-01-15 to 2024-01-16
- **Files Discussed**: 8 files
- **Key Technologies**: FastAPI, python-jose, bcrypt

## Context Summary
This session implemented JWT-based authentication with refresh tokens
for a FastAPI application. Key decisions included:
- Using python-jose for token signing
- bcrypt for password hashing (12 rounds)
- Access tokens: 15min expiry, Refresh: 7 days
- Middleware for automatic token refresh

## Key Implementation Details

### File: src/auth/jwt_handler.py
Key functions discussed:
- `create_access_token()` - JWT encoding with claims
- `verify_token()` - Signature validation and expiry check
- `refresh_access_token()` - Refresh token rotation

### File: src/auth/middleware.py
Key implementation:
- Custom middleware for token validation
- Automatic refresh on 401 responses
- Header injection pattern

## Relevant Code Snippets

```python
# JWT token creation with proper claims
def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm="HS256")
```

## Dependencies Added
- python-jose[cryptography]>=3.3.0
- bcrypt>=4.0.0
- fastapi>=0.100.0

## Next Steps from Original Session
- [ ] Implement rate limiting on auth endpoints
- [ ] Add audit logging for security events
- [ ] Create admin token revocation endpoint

## How This Relates to Your Current Work
Current query: "authentication JWT"
Overlap with current directory: 3 shared files
Relevance score: 94%
```

### 3.2 Implementation

```python
class ForkMDGenerator:
    def __init__(self, db: ChromaDB, session_analyzer: SessionAnalyzer):
        self.db = db
        self.analyzer = session_analyzer
    
    def generate(self, session_id: str, query: str) -> str:
        # Load session data
        session = self.db.get_session(session_id)
        chunks = self.db.get_session_chunks(session_id)
        
        # Analyze session
        analysis = self.analyzer.analyze(chunks)
        
        # Build context file
        sections = [
            self._generate_header(session),
            self._generate_summary(analysis),
            self._generate_file_details(session.files_in_context),
            self._generate_code_snippets(chunks, analysis.key_topics),
            self._generate_dependencies(analysis),
            self._generate_next_steps(session),
            self._generate_relevance_note(query, session)
        ]
        
        return '\n\n'.join(sections)
    
    def _generate_code_snippets(self, chunks: List[Chunk], topics: List[str]) -> str:
        """Extract most relevant code snippets based on topics."""
        snippets = []
        
        for topic in topics[:5]:  # Top 5 topics
            relevant = self._find_relevant_chunks(chunks, topic)
            if relevant:
                snippets.append(f"### {topic}\n```python\n{relevant[0].content}\n```")
        
        return "## Key Code Snippets\n\n" + '\n\n'.join(snippets)
```

---

## 4. Pre-Compaction Hooks

### 4.1 Concept

Kilo Code sessions can grow very large. Pre-compaction hooks capture the full transcript before the system truncates/summarizes old messages.

```python
class PreCompactionHook:
    """
    Captures full session content before Kilo Code compaction.
    """
    
    SIZE_THRESHOLD_MB = 5  # Trigger when session > 5MB
    
    def __init__(self, watcher: TranscriptWatcher, exporter: SessionExporter):
        self.watcher = watcher
        self.exporter = exporter
        self.monitored_sessions: Dict[str, float] = {}  # session_id -> size
    
    def on_session_update(self, session_id: str, transcript_path: Path):
        """Called when transcript watcher detects changes."""
        size_mb = transcript_path.stat().st_size / (1024 * 1024)
        
        # Check if approaching compaction threshold
        if size_mb > self.SIZE_THRESHOLD_MB:
            if session_id not in self.monitored_sessions:
                self._trigger_pre_compaction_export(session_id, transcript_path)
        
        self.monitored_sessions[session_id] = size_mb
    
    def _trigger_pre_compaction_export(self, session_id: str, path: Path):
        """Export full session before potential compaction."""
        console.print(f"[yellow]Session {session_id[:8]}... approaching size limit. Exporting...[/yellow]")
        
        # Export to archive
        archive_path = self.exporter.export_session(session_id, path)
        
        # Create searchable snapshot
        self.exporter.create_snapshot(session_id, path)
        
        console.print(f"[green]Exported to: {archive_path}[/green]")
```

### 4.2 Session Exporter

```python
class SessionExporter:
    """Exports sessions for archival and searchability."""
    
    def __init__(self, export_dir: Path):
        self.export_dir = export_dir
        self.export_dir.mkdir(parents=True, exist_ok=True)
    
    def export_session(self, session_id: str, transcript_path: Path) -> Path:
        """Export full session to archive."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_path = self.export_dir / f"{session_id}_{timestamp}.json"
        
        # Load and export
        with open(transcript_path) as f:
            data = json.load(f)
        
        export_data = {
            'session_id': session_id,
            'exported_at': timestamp,
            'source_file': str(transcript_path),
            'conversation_history': data
        }
        
        with open(export_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        return export_path
    
    def create_snapshot(self, session_id: str, transcript_path: Path):
        """Create searchable snapshot in vector DB."""
        # Parse and chunk
        chunks = self.parse_transcript(transcript_path)
        
        # Mark as pre-compaction snapshot
        for chunk in chunks:
            chunk.metadata['is_snapshot'] = True
            chunk.metadata['snapshot_time'] = datetime.now().isoformat()
        
        # Index
        self.indexer.index_chunks(chunks)
```

---

## 5. Semantic Deduplication

### 5.1 Problem

Multiple sessions may cover similar topics (e.g., 50 sessions about "FastAPI setup"). Deduplication clusters these and shows only the best representative.

### 5.2 Implementation

```python
from hdbscan import HDBSCAN
import numpy as np

class DeduplicationEngine:
    """
    Clusters similar sessions and returns representatives.
    """
    
    def __init__(self, embedding_fn):
        self.embedding_fn = embedding_fn
        self.min_cluster_size = 3
    
    def cluster_results(self, results: List[HybridResult]) -> List[Cluster]:
        """
        Cluster search results by semantic similarity.
        """
        if len(results) < self.min_cluster_size:
            # Not enough results for clustering
            return [Cluster(representative=r, members=[r]) for r in results]
        
        # Get embeddings for all results
        embeddings = []
        for result in results:
            # Create summary text from metadata
            summary = f"{result.metadata.title} {' '.join(result.metadata.files_in_context)}"
            embedding = self.embedding_fn(summary)
            embeddings.append(embedding)
        
        # Perform clustering
        clusterer = HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            metric='cosine'
        )
        labels = clusterer.fit_predict(np.array(embeddings))
        
        # Group by cluster
        clusters = defaultdict(list)
        for idx, label in enumerate(labels):
            clusters[label].append(results[idx])
        
        # Select representatives
        representatives = []
        for label, members in clusters.items():
            if label == -1:
                # Noise points - each is its own cluster
                for member in members:
                    representatives.append(Cluster(representative=member, members=[member]))
            else:
                # Select best representative (highest score)
                best = max(members, key=lambda x: x.score)
                representatives.append(Cluster(representative=best, members=members))
        
        return representatives
    
    def display_clusters(self, clusters: List[Cluster]):
        """Display clustered results to user."""
        for i, cluster in enumerate(clusters, 1):
            rep = cluster.representative
            
            if len(cluster.members) > 1:
                console.print(f"\n[bold][{i}] Score: {rep.score:.0%}[/bold] ([cyan]{len(cluster.members)} similar sessions[/cyan])")
            else:
                console.print(f"\n[bold][{i}] Score: {rep.score:.0%}[/bold]")
            
            console.print(f"    Title: {rep.metadata.title}")
            console.print(f"    Files: {', '.join(rep.metadata.files_in_context[:3])}")
```

---

## 6. Data Schema Updates

### 6.1 ChromaDB Collection Schema

```python
# Collection: sessions
session_schema = {
    'id': 'session_uuid',
    'embedding': Vector(768),  # nomic-embed-text-v1.5
    'document': 'chunk text content',
    'metadata': {
        'session_id': 'uuid',
        'chunk_index': 0,
        'chunk_type': 'user_message' | 'assistant_message' | 'tool_output',
        'timestamp': '2024-01-15T10:30:00Z',
        'files_in_context': ['src/auth.py', 'tests/test_auth.py'],
        'technologies': ['FastAPI', 'JWT', 'Python'],
        'is_snapshot': False,
        'snapshot_time': None,
        'session_title': 'JWT Implementation',
        'total_tokens': 512,
    }
}

# Collection: bm25_index (for keyword search)
bm25_schema = {
    'session_id': 'uuid',
    'tokenized_text': ['jwt', 'auth', 'token', 'fastapi'],
    'last_active': '2024-01-15T10:30:00Z',
}
```

### 6.2 Metadata Enrichment

```python
class MetadataEnricher:
    """Enriches session metadata with derived information."""
    
    TECH_PATTERNS = {
        'FastAPI': [r'fastapi', r'from fastapi'],
        'Django': [r'django', r'from django'],
        'Flask': [r'flask', r'from flask'],
        'React': [r'react', r'import.*react'],
        'PostgreSQL': [r'postgres', r'postgresql', r'psycopg'],
        'Redis': [r'redis'],
        'Docker': [r'docker', r'dockerfile', r'container'],
        'JWT': [r'jwt', r'json.?web.?token'],
    }
    
    def enrich(self, chunks: List[Chunk]) -> SessionMetadata:
        """Extract metadata from conversation chunks."""
        all_text = ' '.join([c.content for c in chunks])
        
        # Detect technologies
        technologies = []
        for tech, patterns in self.TECH_PATTERNS.items():
            if any(re.search(p, all_text, re.I) for p in patterns):
                technologies.append(tech)
        
        # Extract file paths
        file_pattern = r'[\w\-./]+\.(py|js|ts|jsx|tsx|java|go|rs|cpp|c|h|yaml|yml|json|md)'
        files = list(set(re.findall(file_pattern, all_text)))
        
        # Extract code blocks
        code_blocks = self._extract_code_blocks(chunks)
        
        return SessionMetadata(
            technologies=technologies,
            files_in_context=files,
            code_languages=self._detect_languages(code_blocks),
            estimated_tokens=self._estimate_tokens(all_text),
            has_errors='error' in all_text.lower() or 'exception' in all_text.lower(),
            has_tests='test' in all_text.lower() or 'spec' in all_text.lower(),
        )
```

---

## 7. Testing Strategy

### 7.1 Unit Tests

```python
class TestHybridSearchEngine:
    def test_weighted_scoring(self):
        engine = HybridSearchEngine()
        
        # Mock results
        result = HybridResult(
            session_id='test-123',
            score=0.0,
            breakdown={
                'semantic': 0.9,
                'bm25': 0.8,
                'recency': 0.7,
                'path': 0.6
            },
            metadata=None
        )
        
        expected = (0.5 * 0.9 + 0.25 * 0.8 + 0.15 * 0.7 + 0.10 * 0.6)
        
        final = engine._calculate_final_score(result.breakdown)
        assert abs(final - expected) < 0.001
    
    def test_path_matching(self):
        matcher = PathMatcher()
        
        score = matcher.score(
            session_paths=['/home/user/project/src/auth.py'],
            current_dir='/home/user/project'
        )
        
        assert score > 0.8  # High match for same project

class TestForkMDGenerator:
    def test_generates_valid_markdown(self):
        generator = ForkMDGenerator(mock_db, mock_analyzer)
        
        md = generator.generate('session-123', 'authentication')
        
        assert '# Context Fork' in md
        assert 'Session session-123' in md
        assert 'authentication' in md  # Query mentioned
```

### 7.2 Integration Tests

```python
class TestEndToEndFork:
    def test_full_fork_workflow(self, tmp_path):
        """Test complete detect-fork to fork.md workflow."""
        # Setup test data
        create_test_session(tmp_path, 'jwt-auth')
        
        # Index
        indexer = FullIndexer(db_path=tmp_path / 'db')
        indexer.index_all_sessions(tmp_path / 'tasks')
        
        # Search
        engine = HybridSearchEngine(db_path=tmp_path / 'db')
        results = engine.search('JWT authentication')
        
        assert len(results) > 0
        assert results[0].score > 0.5
        
        # Generate fork.md
        generator = ForkMDGenerator(db_path=tmp_path / 'db')
        fork_md = generator.generate(results[0].session_id, 'JWT authentication')
        
        assert 'JWT' in fork_md
        assert '```python' in fork_md  # Code blocks included
```

### 7.3 Performance Benchmarks

```python
class TestPerformance:
    def test_search_latency(self):
        """Search should complete in <2 seconds for 1000 sessions."""
        engine = create_engine_with_n_sessions(1000)
        
        start = time.time()
        results = engine.search('database connection')
        elapsed = time.time() - start
        
        assert elapsed < 2.0
    
    def test_relevance_accuracy(self):
        """Top result should be relevant >85% of time."""
        test_queries = load_test_queries()  # Manually labeled
        
        correct = 0
        for query, expected_session in test_queries:
            results = engine.search(query)
            if results and results[0].session_id == expected_session:
                correct += 1
        
        accuracy = correct / len(test_queries)
        assert accuracy > 0.85
```

---

## 8. Success Criteria

| Metric | Target | Measurement |
|--------|--------|-------------|
| End-to-end fork time | <10 seconds | Time from query to fork.md output |
| Search relevance | >85% | Manual evaluation on test queries |
| Top-3 hit rate | >70% | Ground truth session in top 3 |
| False positive rate | <10% | Irrelevant sessions in results |
| CLI responsiveness | <2 seconds | Time to show initial results |
| Fork.md completeness | >95% | Context coverage score |

---

## 9. Deliverables

1. **Hybrid Search Engine** - Four-signal weighted search
2. **`smartfork detect-fork` Command** - CLI with filters and output options
3. **Fork.md Generator** - Context file generation with structure
4. **Pre-Compaction System** - Automatic export before truncation
5. **Deduplication Engine** - HDBSCAN-based clustering
6. **Updated Tests** - Unit, integration, and performance tests
7. **Documentation** - Usage guide and API reference

---

## 10. Migration from Phase 0

```bash
# Phase 0 -> Phase 1 migration steps

# 1. Backup existing database
cp -r ~/.smartfork/db ~/.smartfork/db.backup

# 2. Update schema (adds new metadata fields)
smartfork migrate --to-phase-1

# 3. Re-index with new chunking strategy (optional but recommended)
smartfork reindex --strategy=multi-modal

# 4. Build BM25 index
smartfork build-bm25-index

# 5. Verify installation
smartfork doctor  # Checks all components
```
