"""Query decomposition for structured search (v2).

Takes a raw user query and produces a QueryDecomposition object via:
1. LLM decomposition (primary) — highest quality, understands natural language
2. Rule-based fallback — works offline, passes raw terms to BM25

The decomposition drives all downstream retrieval decisions:
- Which metadata filters to apply (Signal A)
- Which BM25 terms to search (Signal B)
- How to weight vector search collections (Signal C)
"""

import json
import re
from typing import Optional, List
from loguru import logger

from ..database.models import QueryDecomposition
from ..intelligence.llm_provider import LLMProvider


# ═══════════════════════════════════════════════════════════════════════════════
# LLM DECOMPOSITION PROMPT
# ═══════════════════════════════════════════════════════════════════════════════

DECOMPOSITION_PROMPT = """You are a query parser for a developer session search tool.
Extract structured information from the developer's search query.

Query: "{raw_query}"

{project_context}

Return ONLY valid JSON with these fields:
{{
  "intent": one of [decision_hunting, implementation_lookup, error_recall, file_lookup, temporal_lookup, pattern_hunting, vague_memory],
  "topic": main technical topic (extracted from query, e.g. "hybrid search system", "JWT authentication", "database schema") or null,
  "project": project name if mentioned or null,
  "file_hint": specific filename if mentioned or null,
  "time_hint": time reference like "last_week", "yesterday", "3_days_ago", "last_month" or null,
  "tech_terms": list of technical keywords for exact matching (e.g. ["hybrid", "search", "bm25", "vector"]),
  "is_temporal_only": true if query has no topic and only time reference
}}"""


# ═══════════════════════════════════════════════════════════════════════════════
# INTENT KEYWORD PATTERNS (for rule-based fallback)
# ═══════════════════════════════════════════════════════════════════════════════

INTENT_PATTERNS = {
    "decision_hunting": ["why", "decided", "chose", "chosen", "approach", "reason", "rationale",
                         "because", "trade-off", "tradeoff", "comparison"],
    "implementation_lookup": ["how did i", "code for", "implement", "build", "create", "wrote",
                              "write", "setup", "set up", "configure"],
    "error_recall": ["bug", "error", "fix", "fixed", "broke", "broken", "issue", "crash",
                     "exception", "traceback", "fail", "failed", "debug", "debugging"],
    "file_lookup": [],  # Detected via regex, not keywords
    "temporal_lookup": ["yesterday", "last week", "last month", "days ago", "today",
                        "this week", "this month", "recently", "recent"],
    "pattern_hunting": ["all sessions", "every time", "whenever", "all the", "always",
                        "each time", "all projects"],
}

TIME_HINT_MAP = {
    "yesterday": "yesterday",
    "today": "today",
    "last week": "last_week",
    "this week": "this_week",
    "last month": "last_month",
    "this month": "this_month",
    "3 days ago": "3_days_ago",
    "few days ago": "3_days_ago",
    "recently": "last_week",
    "recent": "last_week",
}

# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL MODELS (spaCy for time extraction only)
# ═══════════════════════════════════════════════════════════════════════════════

_spacy_nlp = None

def get_spacy_nlp():
    """Returns spaCy model for time extraction only."""
    global _spacy_nlp
    if _spacy_nlp is None:
        import spacy
        logger.info("Loading spaCy english model (en_core_web_sm)...")
        _spacy_nlp = spacy.load("en_core_web_sm")
        ruler = _spacy_nlp.add_pipe("entity_ruler")
        patterns = [{"label": "TIME_HINT", "pattern": p} for p in TIME_HINT_MAP.keys()]
        ruler.add_patterns(patterns)
    return _spacy_nlp


def extract_entities_from_query(query: str, known_projects: list[str]) -> dict:
    """Fast rule-based entity extraction without ML models.
    
    - Project: fuzzy match against known projects from SQLite
    - Topic: regex for CamelCase/CAPS terms (JWT, ChromaDB, FastAPI)
    - File: regex for known extensions
    """
    import re
    from difflib import get_close_matches
    
    query_lower = query.lower()
    
    # Step 1: Project extraction - match against known projects
    # Handles: exact match, partial match (bharatlaw -> bharatlaw-frontend)
    project = None
    for proj in known_projects:
        if proj and proj.lower() in query_lower:
            project = proj
            break
    
    # Try partial match: query contains "bharatlaw" -> matches "bharatlaw-frontend"
    if not project and known_projects:
        for proj in known_projects:
            if proj:
                # Split project name into parts: bharatlaw-frontend -> [bharatlaw, frontend]
                proj_parts = proj.lower().replace("-", " ").replace("_", " ").split()
                for part in proj_parts:
                    if len(part) >= 4 and part in query_lower:  # require at least 4 chars
                        project = proj
                        break
                if project:
                    break
    
    # Last resort: fuzzy match
    if not project and known_projects:
        words = query.split()
        for word in words:
            word_clean = word.lower().replace("-", " ").replace("_", " ")
            for proj in known_projects:
                if proj:
                    proj_clean = proj.lower().replace("-", " ").replace("_", " ")
                    if word_clean in proj_clean or proj_clean in word_clean:
                        project = proj
                        break
            if project:
                break
    
    # Step 2: Topic/technology extraction - regex for CamelCase/CAPS
    # But filter out any matches that are project names (or partial matches)
    tech_pattern = re.compile(r'\b[A-Z][a-zA-Z0-9]*(?:[A-Z][a-zA-Z0-9]*)+\b|\b[A-Z]{2,}\b')
    tech_matches = tech_pattern.findall(query)
    stopwords = {"I", "API", "UI", "URL", "SQL", "THE", "AND", "OR", "FOR", "NOT"}
    
    # Filter out project names and stopwords
    filtered_tech = []
    for t in tech_matches:
        t_lower = t.lower()
        # Skip if it's a stopword
        if t_lower in [s.lower() for s in stopwords]:
            continue
        # Skip if it matches any known project (check partial matches too)
        is_project = False
        for proj in known_projects:
            if proj:
                proj_clean = proj.lower().replace("-", " ").replace("_", " ")
                # Check both directions: term in project OR project in term
                if t_lower == proj.lower() or t_lower in proj_clean or proj_clean in t_lower:
                    is_project = True
                    break
        if is_project:
            continue
        filtered_tech.append(t)
    
    topic = filtered_tech[0].lower() if filtered_tech else None
    
    # Step 3: File extraction - known extensions
    file_pattern = re.compile(r'\b\w+\.(py|ts|tsx|js|go|java|css|md|json|yaml|yml|toml)\b')
    file_match = file_pattern.search(query)
    file_hint = file_match.group(0) if file_match else None
    
    return {
        "project": project,
        "topic": topic,
        "file_hint": file_hint,
    }


class QueryDecomposer:
    """Decomposes raw user queries into structured QueryDecomposition objects.
    
    Uses LLM when available (primary path), falls back to rule-based extraction.
    Supports fuzzy project matching against actual indexed project names.
    
    Usage:
        decomposer = QueryDecomposer(llm=get_llm("ollama"), known_projects=["bharatlaw-frontend"])
        result = decomposer.decompose("auth decisions in BharatLawAI last month")
        # → QueryDecomposition(intent="decision_hunting", project="bharatlaw-frontend", ...)
    """
    
    def __init__(self, llm: Optional[LLMProvider] = None,
                 known_projects: Optional[List[str]] = None):
        self.llm = llm
        self.known_projects = known_projects or []
        
        # Rule-based intent keywords (fast, no ML)
        self.INTENT_KEYWORDS = {
            "error_recall": [
                "bug", "error", "fix", "broke", "failed", "exception",
                "issue", "crash", "wrong", "broken", "debug", "traceback",
                "not working", "doesn't work", "problem"
            ],
            "decision_hunting": [
                "why", "decided", "chose", "reason", "approach", "instead",
                "rationale", "thought", "consideration", "trade-off", "versus",
                "decisions", "decision", "picked", "selected", "went with"
            ],
            "file_lookup": [
                ".py", ".ts", ".tsx", ".js", ".go", ".java", "file", "module",
                "component", "class", "function", "method"
            ],
            "temporal_lookup": [
                "yesterday", "last week", "last month", "ago", "recently",
                "today", "this week", "earlier", "before", "previous"
            ],
            "implementation_lookup": [
                "how", "implement", "built", "created", "code for", "setup",
                "configure", "install", "integrate", "add", "write", "structure"
            ],
            "pattern_hunting": [
                "all sessions", "every time", "whenever", "pattern", "approach",
                "usually", "always", "across", "multiple", "all my"
            ],
        }
    
    def _rule_based_intent(self, query: str) -> str:
        """Fast rule-based intent classification (<5ms)."""
        query_lower = query.lower()
        scores = {}
        
        for intent, keywords in self.INTENT_KEYWORDS.items():
            scores[intent] = sum(1 for kw in keywords if kw in query_lower)
        
        # Return the highest scorer - keyword lists are specific enough now
        best_intent = max(scores, key=scores.get)
        if scores[best_intent] == 0:
            return "vague_memory"
        
        return best_intent

    def decompose(self, query: str) -> QueryDecomposition:
        """Decompose a raw query into structured components.
        
        Uses rule-based extraction for all entities:
        - Intent: keyword matching
        - Project: fuzzy match against known projects
        - Topic: CamelCase/CAPS regex
        - File: extension regex
        - Time: spaCy NER
        """
        if not query or not query.strip():
            return QueryDecomposition(intent="vague_memory")
        
        # Get known projects (cached)
        known_projects = self.known_projects or []
        
        # Rule-based intent classification
        intent = self._rule_based_intent(query)
        
        # Rule-based entity extraction
        entities = extract_entities_from_query(query, known_projects)
        
        # spaCy for time hints only
        time_hint = None
        try:
            nlp = get_spacy_nlp()
            doc = nlp(query.lower())
            for ent in doc.ents:
                if ent.label_ == "TIME_HINT":
                    time_hint = TIME_HINT_MAP.get(ent.text, ent.text.replace(' ', '_'))
                    break
        except Exception:
            pass
        
        # Extract tech terms from query
        tech_terms = []
        import re
        tech_pattern = re.compile(r'\b[A-Z][a-zA-Z0-9]*(?:[A-Z][a-zA-Z0-9]*)+\b|\b[A-Z]{2,}\b')
        for match in tech_pattern.findall(query):
            if match.lower() not in [p.lower() for p in known_projects if p]:
                tech_terms.append(match.lower())
        
        return QueryDecomposition(
            intent=intent,
            topic=entities["topic"],
            project=entities["project"],
            file_hint=entities["file_hint"],
            time_hint=time_hint,
            tech_terms=list(set(tech_terms)),
            is_temporal_only=(time_hint is not None and not entities["topic"] and intent in ["temporal_lookup", "vague_memory"])
        )
    



    def _ml_decompose(self, query: str) -> Optional[QueryDecomposition]:
        """Fast ML-powered decomposition using GLiNER + spaCy + rule-based intent."""
        try:
            # 1. Load GLiNER and spaCy (entity extraction)
            gliner = get_gliner()
            nlp = get_spacy_nlp()
            
            # 2. spaCy Temporal Extraction
            doc = nlp(query.lower())
            time_hint = None
            for ent in doc.ents:
                if ent.label_ == "TIME_HINT":
                    time_hint = TIME_HINT_MAP.get(ent.text, ent.text.replace(' ', '_'))
                    break
            
            # 3. GLiNER Span Extraction - natural language labels for zero-shot
            GLINER_LABELS = [
                "software project name like BharatLawAI or SmartFork",
                "programming library or technology like JWT or ChromaDB",
                "source code file name with extension like auth.py",
                "software error or bug name like CORS or NullPointer",
            ]
            entities = gliner.predict_entities(query, GLINER_LABELS)
            
            project_name = None
            topic = None
            error_code = None
            tech_terms = []
            
            for ent in entities:
                label = ent["label"]
                text = ent["text"].strip()
                # Map GLiNER labels to our fields
                if label == "software project name" and not project_name:
                    project_name = text
                elif label == "programming technology or library" and not topic:
                    topic = text
                elif label == "source code file name":
                    if not file_hint:
                        file_hint = text
                elif label == "error or bug name" and not error_code:
                    error_code = text
            
            # 4. Rule-based Intent Classification (fast, no ML)
            intent_type = self._rule_based_intent(query)
            
            # 5. Rapidfuzz Project Resolution
            if project_name and self.known_projects:
                from rapidfuzz import process, fuzz
                best_match = process.extractOne(project_name, self.known_projects, scorer=fuzz.WRatio)
                if best_match and best_match[1] >= 80:
                    project_name = best_match[0]
            
            # 6. File tracking (legacy regex is robust enough)
            file_hint = self._extract_file_hint(query)
            
            if error_code:
                tech_terms.append(error_code)

            return QueryDecomposition(
                intent=intent_type,
                topic=topic,
                project=project_name,
                file_hint=file_hint,
                time_hint=time_hint,
                tech_terms=list(set(tech_terms)),
                is_temporal_only=(time_hint is not None and not topic and intent_type in ["temporal_lookup", "vague_memory"])
            )
            
        except Exception as e:
            logger.warning(f"ML decomposition failed, using rule-based fallback: {e}")
            return None
    
    # ── FUZZY PROJECT MATCHING ───────────────────────────────────
    
    def _fuzzy_match_project(self, project_hint: str) -> Optional[str]:
        """Match a project hint against actual indexed project names.
        
        Handles: BharatLawAI → bharatlaw-frontend, smartfork → smartfork, etc.
        """
        if not self.known_projects or not project_hint:
            return None
        
        hint_lower = project_hint.lower().replace("-", "").replace("_", "")
        
        # Exact match first
        for name in self.known_projects:
            if name.lower() == project_hint.lower():
                return name
        
        # Substring match (either direction)
        for name in self.known_projects:
            name_normalized = name.lower().replace("-", "").replace("_", "")
            if hint_lower in name_normalized or name_normalized in hint_lower:
                return name
        
        # Prefix match (at least 4 chars)
        if len(hint_lower) >= 4:
            for name in self.known_projects:
                name_normalized = name.lower().replace("-", "").replace("_", "")
                if name_normalized.startswith(hint_lower[:4]) or hint_lower.startswith(name_normalized[:4]):
                    return name
        
        return None
    
    def _fuzzy_match_project_from_query(self, query: str) -> Optional[str]:
        """Try to find a project name anywhere in the raw query text."""
        query_lower = query.lower()
        
        for name in self.known_projects:
            name_lower = name.lower()
            # Skip generic names
            if name_lower in ("unknown", "unknown_project", "plans"):
                continue
            # Check if project name appears in query (stripped of special chars)
            name_stripped = name_lower.replace("-", "").replace("_", "")
            if name_stripped in query_lower.replace(" ", "").replace("-", ""):
                return name
            # Also check each word
            if name_lower in query_lower:
                return name
        
        return None
    
    # ── RULE-BASED FALLBACK ──────────────────────────────────────
    
    def _rule_based_decompose(self, query: str) -> QueryDecomposition:
        """Rule-based decomposition fallback (offline mode).
        
        Instead of trying to be smart about topic extraction (which fails
        on unknown words), we pass the raw query to BM25 and let the
        scoring engine handle relevance.
        """
        query_lower = query.lower().strip()
        
        # Detect intent from keywords
        intent = self._detect_intent(query_lower)
        
        # Extract file hint via regex
        file_hint = self._extract_file_hint(query)
        if file_hint and intent == "vague_memory":
            intent = "file_lookup"
        
        # Extract time hint
        time_hint = self._extract_time_hint(query_lower)
        
        # Check if temporal only
        is_temporal_only = (time_hint is not None and intent in ["temporal_lookup", "vague_memory"]
                           and not file_hint)
        if is_temporal_only:
            intent = "temporal_lookup"
        
        # Fuzzy match project against known indexed names
        project = self._extract_project(query)
        if project:
            matched = self._fuzzy_match_project(project)
            if matched:
                project = matched
        
        if not project:
            project = self._fuzzy_match_project_from_query(query)
        
        # Topic: use the full query as-is for BM25 — don't try to be smart
        # BM25's TF-IDF scoring naturally ignores common words
        topic = query_lower
        
        # Tech terms: extract anything that looks technical
        tech_terms = self._extract_tech_terms(query_lower)
        
        return QueryDecomposition(
            intent=intent,
            topic=topic,
            project=project,
            file_hint=file_hint,
            time_hint=time_hint,
            tech_terms=tech_terms,
            is_temporal_only=is_temporal_only,
        )
    
    def _detect_intent(self, query_lower: str) -> str:
        """Detect query intent from keyword patterns."""
        scores = {}
        for intent, keywords in INTENT_PATTERNS.items():
            score = sum(1 for kw in keywords if kw in query_lower)
            if score > 0:
                scores[intent] = score
        
        if scores:
            return max(scores, key=scores.get)
        return "vague_memory"
    
    def _extract_file_hint(self, query: str) -> Optional[str]:
        """Extract a filename from the query."""
        match = re.search(r'[\w\-]+\.\w{1,5}', query)
        return match.group() if match else None
    
    def _extract_time_hint(self, query_lower: str) -> Optional[str]:
        """Extract a time reference from the query."""
        for phrase, hint in TIME_HINT_MAP.items():
            if phrase in query_lower:
                return hint
        
        # Check for "N days ago" pattern
        days_match = re.search(r'(\d+)\s*days?\s*ago', query_lower)
        if days_match:
            return f"{days_match.group(1)}_days_ago"
        
        return None
    
    def _extract_project(self, query: str) -> Optional[str]:
        """Extract a project name (PascalCase or quoted)."""
        # PascalCase: BharatLawAI, SmartFork
        pascal_match = re.search(r'\b([A-Z][a-z]+(?:[A-Z][a-zA-Z]*)+)\b', query)
        if pascal_match:
            return pascal_match.group(1)
        
        # Quoted strings
        quoted_match = re.search(r'["\'](.+?)["\']', query)
        if quoted_match:
            return quoted_match.group(1)
        
        return None
    
    def _extract_tech_terms(self, query_lower: str) -> list:
        """Extract technical keywords for BM25 matching.
        
        Instead of aggressive stop-word filtering, we extract terms
        and let BM25's TF-IDF scoring handle relevance naturally.
        """
        # Find words that look technical
        terms = re.findall(r'[a-z]+(?:_[a-z]+)+|[a-z]+\.[a-z]+|\b[a-z]{3,}\b', query_lower)
        
        # Only remove the most basic English stop words
        basic_stops = {"the", "and", "for", "was", "were", "that", "this",
                       "with", "from", "have", "what", "when", "where",
                       "how", "did", "our", "want", "all", "can", "you",
                       "will", "about", "which"}
        terms = [t for t in terms if t not in basic_stops and len(t) > 2]
        
        return list(set(terms))


# ═══════════════════════════════════════════════════════════════════════════════
# INTENT-AWARE VECTOR SEARCH WEIGHTS
# ═══════════════════════════════════════════════════════════════════════════════

INTENT_VECTOR_WEIGHTS = {
    "decision_hunting": {"task": 0.1, "summary": 0.3, "reasoning": 0.6},
    "implementation_lookup": {"task": 0.5, "summary": 0.4, "reasoning": 0.1},
    "error_recall": {"task": 0.1, "summary": 0.2, "reasoning": 0.7},
    "file_lookup": {"task": 0.3, "summary": 0.3, "reasoning": 0.4},
    "temporal_lookup": {"task": 0.3, "summary": 0.5, "reasoning": 0.2},
    "pattern_hunting": {"task": 0.3, "summary": 0.6, "reasoning": 0.1},
    "vague_memory": {"task": 0.3, "summary": 0.6, "reasoning": 0.1},
}


def get_vector_weights(intent: str) -> dict:
    """Get vector search weights based on query intent."""
    return INTENT_VECTOR_WEIGHTS.get(intent, INTENT_VECTOR_WEIGHTS["vague_memory"])
