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
# GLOBAL MODELS (Cold-start fix for persistent MCP process)
# ═══════════════════════════════════════════════════════════════════════════════

_gliner_model = None
_intent_classifier = None
_spacy_nlp = None

def get_gliner():
    global _gliner_model
    if _gliner_model is None:
        from gliner import GLiNER
        logger.info("Loading GLiNER model (gliner-medium-v2.1)...")
        _gliner_model = GLiNER.from_pretrained("urchade/gliner_medium-v2.1")
    return _gliner_model

def get_intent_classifier():
    global _intent_classifier
    if _intent_classifier is None:
        from transformers import pipeline
        logger.info("Loading DeBERTa Intent Classifier (nli-deberta-v3-xsmall)...")
        _intent_classifier = pipeline("zero-shot-classification", model="cross-encoder/nli-deberta-v3-xsmall")
    return _intent_classifier

def get_spacy_nlp():
    global _spacy_nlp
    if _spacy_nlp is None:
        import spacy
        logger.info("Loading spaCy english model (en_core_web_sm)...")
        _spacy_nlp = spacy.load("en_core_web_sm")
        ruler = _spacy_nlp.add_pipe("entity_ruler")
        patterns = [{"label": "TIME_HINT", "pattern": p} for p in TIME_HINT_MAP.keys()]
        ruler.add_patterns(patterns)
    return _spacy_nlp


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
    
    def decompose(self, query: str) -> QueryDecomposition:
        """Decompose a raw query into structured components.
        
        Tries ML (GLiNER + DeBERTa) first, falls back to rule-based extraction on failure.
        Always runs fuzzy project matching against known indexed projects.
        """
        if not query or not query.strip():
            return QueryDecomposition(intent="vague_memory")
        
        # Try zero-shot ML decomposition first (fast, primary path)
        result = self._ml_decompose(query)
        if result:
            # Post-process: fuzzy match the extracted project against indexed names
            if result.project:
                matched = self._fuzzy_match_project(result.project)
                if matched:
                    result.project = matched
            elif not result.project:
                # ML didn't extract a project, try fuzzy matching raw query
                matched = self._fuzzy_match_project_from_query(query)
                if matched:
                    result.project = matched
            return result
        
        # Fallback to rule-based
        return self._rule_based_decompose(query)
    



    def _ml_decompose(self, query: str) -> Optional[QueryDecomposition]:
        """Attempt fast ML-powered decomposition (GLiNER + DeBERTa)."""
        try:
            # 1. Load models (cached globally after first run)
            gliner = get_gliner()
            classifier = get_intent_classifier()
            nlp = get_spacy_nlp()
            
            # 2. spaCy Temporal Fallback
            doc = nlp(query.lower())
            time_hint = None
            for ent in doc.ents:
                if ent.label_ == "TIME_HINT":
                    time_hint = TIME_HINT_MAP.get(ent.text, ent.text.replace(' ', '_'))
                    break
            
            # 3. GLiNER Span Extraction
            labels = ["project_name", "topic", "time_hint", "error_code", "library"]
            entities = gliner.predict_entities(query, labels)
            
            project_name = None
            topic = None
            error_code = None
            tech_terms = []
            
            for ent in entities:
                label = ent["label"]
                text = ent["text"].strip()
                if label == "project_name" and not project_name:
                    project_name = text
                elif label == "topic" and not topic:
                    topic = text
                elif label == "error_code" and not error_code:
                    error_code = text
                elif label == "library":
                    tech_terms.append(text)
                elif label == "time_hint" and not time_hint:
                    time_hint_clean = text.lower().replace(' ', '_')
                    time_hint = TIME_HINT_MAP.get(text.lower(), time_hint_clean)
            
            # 4. DeBERTa Sequence Classification (Intent)
            candidate_labels = [
                "decision_hunting", "implementation_lookup", 
                "error_recall", "pattern_hunting", "temporal_lookup"
            ]
            
            if not topic and not project_name and not error_code and len(query.split()) < 3:
                intent_type = "temporal_lookup" if time_hint else "vague_memory"
            else:
                result = classifier(query, candidate_labels)
                intent_type = result["labels"][0]
            
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
