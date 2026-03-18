"""Intent-classified fork context assembler (v2).

Assembles context packages based on fork intent:
- CONTINUE: Pick up exactly where the last session left off
- REFERENCE: Reuse the approach/decisions in new work
- DEBUG: Hit the same problem again, need the fix

Two modes:
1. LLM mode (primary): LLM reads full session data and distills intent-specific context
2. Cleaned raw mode (fallback): Structured assembly from cleaned data when Ollama unavailable
"""

import re
from typing import Optional
from loguru import logger

from ..database.models import SessionDocument, ForkIntent
from ..intelligence.llm_provider import LLMProvider


# ═══════════════════════════════════════════════════════════════════════════════
# INTENT-SPECIFIC LLM PROMPTS
# ═══════════════════════════════════════════════════════════════════════════════

FORK_PROMPTS = {
    ForkIntent.CONTINUE: """You are extracting deep technical context from a coding session so a developer can resume work exactly where they left off.

Session info:
- Project: {project_name}
- Task: {task_raw}
- Files modified: {files_edited}
- Domains: {domains}
- Duration: {duration} minutes

Reasoning trail (developer's thought process during the session):
{reasoning_text}

Write a highly detailed, structured fork context with these sections:
1. **What was being built** (The overarching goal and specific features)
2. **Approach taken** (Architecture, design decisions, libraries used, and why)
3. **What was completed** (Be extremely specific: mention exact file paths, specific function names, and what logic was implemented)
4. **What remains / stopping point** (Exactly where work stopped, uncommitted changes, specific methods left pending, what's next)
5. **Open issues** (Blockers, bugs, terminal errors, or unresolved edge cases)

Provide as much detailed technical context as possible. Mention exact file paths and code entities.
Do NOT include raw code blocks of entire files or raw diff markers (`>>>>>> REPLACE`).
Keep the final output dense and concise, roughly between 400 to 800 words.
""",

    ForkIntent.REFERENCE: """You are extracting reusable technical decisions from a coding session so a developer can apply the same approach to new work.

Session info:
- Project: {project_name}
- Task: {task_raw}
- Files modified: {files_edited}
- Domains: {domains}

Reasoning trail:
{reasoning_text}

Write a highly detailed structured reference with these sections:
1. **Problem solved** (What was the challenge or requirement)
2. **Approach chosen** (The solution architecture, data structures, UI patterns, and why it was chosen)
3. **Alternatives considered** (What approaches were rejected and why)
4. **Key patterns** (Reusable implementation patterns, specific library usage, exact file paths to reference)
5. **Gotchas** (Things that went wrong, surprising edge cases, terminal errors overcome)

Focus heavily on decisions, rationale, and reusable patterns. Name specific files and functions.
Do NOT include raw code blocks of entire files or raw diff markers.
Keep the final output dense and concise, roughly between 300 to 600 words.
""",

    ForkIntent.DEBUG: """You are extracting deep debugging context from a coding session so a developer can understand and fix a similar issue.

Session info:
- Project: {project_name}
- Task: {task_raw}
- Files modified: {files_edited}

Reasoning trail:
{reasoning_text}

Write a highly detailed structured debug reference with these sections:
1. **Error encountered** (The exact error message, stack trace snippets, or symptom)
2. **Root cause** (The exact technical reason for the failure in specific files/lines)  
3. **What was tried** (All debugging steps taken, including failed attempts and dead ends)
4. **Fix applied** (The exact solution that worked, referencing specific file paths and changes)
5. **Prevention** (How to avoid this or test for it in the future)

Be extremely specific about exact error messages, exact file paths, and what code was changed.
Do NOT include raw code blocks of entire files or raw diff markers.
Keep the final output dense and concise, roughly between 300 to 600 words.
""",

    ForkIntent.SYNTHESIZE: """You are synthesizing a multi-session development Epic to help a developer onboard into a complex feature timeline.

Sessions in Epic: {project_name}
Task Overview: {task_raw}
Key Domains touched: {domains}

Chronological Reasoning Trail:
{reasoning_text}

Write a comprehensive chronological synthesis with these sections:
1. **Epic Overview** (What this string of sessions aimed to achieve at a high level)
2. **Timeline of Execution** (Chronological breakdown of key architectural phases/checkpoints)
3. **Core Architectural Shifts** (Major decisions made, pivots, or design changes over time)
4. **Current State** (Where the code currently stands and what is unresolved)

Focus on the major narrative and cross-session evolution. 
Do NOT include raw code. Keep it extremely dense and informative.
""",
}

# ═══════════════════════════════════════════════════════════════════════════════
# TOKEN BUDGETS FOR FALLBACK MODE
# ═══════════════════════════════════════════════════════════════════════════════

INTENT_LAYERS = {
    ForkIntent.CONTINUE: {
        "description": "Enough context to resume work exactly where you left off",
        "layers": ["A", "B", "D", "F"],
        "max_reasoning_blocks": 40,
    },
    ForkIntent.REFERENCE: {
        "description": "Decisions and approach without implementation noise",
        "layers": ["A", "B", "C"],
        "max_reasoning_blocks": 30,
    },
    ForkIntent.DEBUG: {
        "description": "Error context and the fix that resolved it",
        "layers": ["A", "E", "D"],
        "max_reasoning_blocks": 30,
    },
    ForkIntent.SYNTHESIZE: {
        "description": "Cross-session synthesis compiling an entire Epic timeline",
        "layers": ["A", "B", "C"],
        "max_reasoning_blocks": 50,
    },
}


class ForkAssembler:
    """Assembles fork context based on intent classification.
    
    Primary mode: LLM reads session data and generates focused, clean context.
    Fallback mode: Structured assembly from cleaned raw data (when Ollama unavailable).
    """
    
    def __init__(self, llm: Optional[LLMProvider] = None, 
                 vector_index = None, store = None):
        self.llm = llm
        self.vector_index = vector_index
        self.store = store
    
    def assemble(self, doc: SessionDocument, intent: ForkIntent,
                  custom_query: str = "") -> str:
        """Assemble context for forking based on intent.
        
        Primary: Uses LLM to distill intent-specific context from full session data.
        Fallback: Structured assembly from cleaned data layers.
        """
        # Try LLM-powered assembly first
        if self.llm:
            result = self._llm_assemble(doc, intent, custom_query)
            if result:
                header = self._build_header(doc, intent)
                return header + "\n\n" + result
        
        # Fallback to cleaned raw assembly
        logger.info("LLM unavailable, using cleaned raw assembly")
        return self._raw_assemble(doc, intent)
    
    # ── LLM-POWERED ASSEMBLY (PRIMARY) ──────────────────────────
    
    def _llm_assemble(self, doc: SessionDocument, intent: ForkIntent, custom_query: str = "") -> Optional[str]:
        """Use LLM to read session data and generate intent-specific context."""
        try:
            # Build reasoning text — take most relevant blocks, cap total length
            config = INTENT_LAYERS[intent]
            max_blocks = config["max_reasoning_blocks"]
            
            reasoning_blocks = []
            
            # Cross-session epic retrieval for SYNTHESIZE intent
            if intent == ForkIntent.SYNTHESIZE and doc.cluster_id and self.store:
                try:
                    logger.debug(f"Fork assembly: fetching Epic timeline for cluster {doc.cluster_id}")
                    epic_sessions = self.store.conn.execute(
                        "SELECT session_id, reasoning_docs FROM sessions WHERE cluster_id = ? ORDER BY session_start ASC",
                        (doc.cluster_id,)
                    ).fetchall()
                    
                    import json
                    for row in epic_sessions:
                        docs = json.loads(row['reasoning_docs'] or '[]')
                        # Take top 5 blocks from each session in the epic to prevent token limits
                        reasoning_blocks.extend(docs[:5])
                    reasoning_blocks = reasoning_blocks[:max_blocks]
                except Exception as e:
                    logger.warning(f"Failed to fetch SYNTHESIZE epic timeline: {e}")
                    reasoning_blocks = doc.reasoning_docs[:max_blocks] if doc.reasoning_docs else []
            # Parent-child semantic retrieval for queried RAG
            elif custom_query and self.vector_index and self.store:
                try:
                    logger.debug(f"Fork assembly: injecting parent chunks for query '{custom_query}'")
                    query_embedding = self.vector_index.embedder.embed(custom_query, "query")
                    
                    # Search reasoning collection just for this session
                    results = self.vector_index.search(
                        query_embedding, "reasoning_doc", 
                        session_ids=[doc.session_id], 
                        n_results=max_blocks * 2
                    )
                    
                    seen_parents = set()
                    for r in results:
                        if r.parent_id and r.parent_id not in seen_parents:
                            parent_text = self.store.get_parent_chunk(r.parent_id)
                            if parent_text:
                                reasoning_blocks.append(parent_text)
                                seen_parents.add(r.parent_id)
                        elif not r.parent_id:
                            # Fallback for legacy v2 sessions without parent_id
                            reasoning_blocks.append(r.content)
                            
                    reasoning_blocks = reasoning_blocks[:max_blocks]
                except Exception as e:
                    logger.warning(f"Parent-child retrieval failed, falling back to chronological: {e}")
                    reasoning_blocks = doc.reasoning_docs[:max_blocks] if doc.reasoning_docs else []
            else:
                # No query provided or missing dependencies: fall back to raw chronological
                reasoning_blocks = doc.reasoning_docs[:max_blocks] if doc.reasoning_docs else []
            
            # Cap each block to 1000 chars and total to ~24000 chars (fits in 32K context easily)
            reasoning_parts = []
            total_chars = 0
            for i, block in enumerate(reasoning_blocks):
                capped = block[:1000]
                if len(block) > 1000:
                    capped += "..."
                reasoning_parts.append(f"Step {i+1}: {capped}")
                total_chars += len(capped)
                if total_chars > 24000:
                    break
            
            reasoning_text = "\n\n".join(reasoning_parts) if reasoning_parts else "No reasoning blocks available."
            
            # Build prompt
            prompt_template = FORK_PROMPTS[intent]
            prompt = prompt_template.format(
                project_name=doc.project_name or "unknown",
                task_raw=(doc.task_raw or "no task description")[:1000],
                files_edited=", ".join(doc.files_edited[:20]) or "none",
                domains=", ".join(doc.domains[:10]) if doc.domains else "none",
                duration=round(doc.duration_minutes, 1) if doc.duration_minutes else "unknown",
                reasoning_text=reasoning_text,
            )
            
            response = self.llm.complete(prompt, max_tokens=4000)
            
            if not response or len(response.strip()) < 50:
                logger.warning("LLM fork assembly returned empty/too short response")
                return None
            
            # Clean LLM response
            result = response.strip()
            # Remove thinking tags if present
            result = re.sub(r'<think>.*?</think>', '', result, flags=re.DOTALL).strip()
            # Remove any preamble
            result = re.sub(r'^(?:Here (?:is|are) (?:the|a|your) ).*?:\s*\n', '', result, flags=re.IGNORECASE)
            
            return result
            
        except Exception as e:
            logger.warning(f"LLM fork assembly failed: {e}")
            return None
    
    # ── CLEANED RAW ASSEMBLY (FALLBACK) ─────────────────────────
    
    def _raw_assemble(self, doc: SessionDocument, intent: ForkIntent) -> str:
        """Fallback: structured assembly from cleaned data layers."""
        config = INTENT_LAYERS[intent]
        layers = config["layers"]
        sections = []
        
        header = self._build_header(doc, intent)
        sections.append(header)
        
        if "A" in layers:
            sections.append(self._build_layer_a(doc))
        if "B" in layers and doc.summary_doc:
            sections.append(self._build_layer_b(doc))
        if "C" in layers and doc.reasoning_docs:
            sections.append(self._build_layer_c(doc))
        if "D" in layers:
            layer_d = self._build_layer_d(doc)
            if layer_d:
                sections.append(layer_d)
        if "E" in layers and doc.reasoning_docs:
            sections.append(self._build_layer_e(doc))
        if "F" in layers and doc.reasoning_docs:
            sections.append(self._build_layer_f(doc))
        
        return "\n\n".join(s for s in sections if s)
    
    def _build_header(self, doc: SessionDocument, intent: ForkIntent) -> str:
        config = INTENT_LAYERS[intent]
        return (
            f"# SmartFork Context — {intent.value.title()}\n"
            f"**Project:** {doc.project_name}\n"
            f"**Session:** {doc.session_id}\n"
            f"**Intent:** {config['description']}"
        )
    
    def _build_layer_a(self, doc: SessionDocument) -> str:
        """Layer A: Task + project context."""
        parts = ["## Task"]
        # Clean the task text — truncate if it's a huge terminal dump
        task_text = doc.task_raw or "No task description"
        if len(task_text) > 500:
            task_text = task_text[:500] + "\n\n*(task text truncated)*"
        parts.append(task_text)
        
        if doc.domains:
            parts.append(f"**Domains:** {', '.join(doc.domains)}")
        if doc.languages:
            parts.append(f"**Languages:** {', '.join(doc.languages)}")
        
        return "\n".join(parts)
    
    def _build_layer_b(self, doc: SessionDocument) -> str:
        """Layer B: LLM-generated summary."""
        return f"## Summary\n{doc.summary_doc}"
    
    def _build_layer_c(self, doc: SessionDocument) -> str:
        """Layer C: Key decisions (filtered reasoning blocks)."""
        parts = ["## Key Decisions"]
        
        decision_keywords = ["decided", "chose", "because", "approach", "instead",
                            "rejected", "trade-off", "better", "opted", "alternative",
                            "compared", "over", "rather than"]
        
        decision_blocks = []
        for r in doc.reasoning_docs:
            if any(kw in r.lower() for kw in decision_keywords):
                # Cap each block to 300 chars
                block = r[:300]
                if len(r) > 300:
                    block += "..."
                decision_blocks.append(block)
        
        if not decision_blocks:
            decision_blocks = [r[:200] for r in doc.reasoning_docs[:3]]
        
        for block in decision_blocks[:5]:
            parts.append(f"- {block}")
        
        return "\n".join(parts)
    
    def _build_layer_d(self, doc: SessionDocument) -> str:
        """Layer D: Files touched."""
        parts = ["## Files Modified"]
        
        if doc.files_edited:
            for f in doc.files_edited[:10]:
                parts.append(f"- `{f}`")
        
        if doc.files_read:
            parts.append("\n**Files Read:**")
            for f in doc.files_read[:5]:
                parts.append(f"- `{f}`")
        
        return "\n".join(parts) if len(parts) > 1 else ""
    
    def _build_layer_e(self, doc: SessionDocument) -> str:
        """Layer E: Error context (for DEBUG intent)."""
        parts = ["## Error Context"]
        
        error_keywords = ["error", "bug", "fix", "broke", "exception", "traceback",
                         "fail", "crash", "issue", "debug"]
        
        error_blocks = []
        for r in doc.reasoning_docs:
            if any(kw in r.lower() for kw in error_keywords):
                block = r[:400]
                if len(r) > 400:
                    block += "..."
                error_blocks.append(block)
        
        if not error_blocks:
            parts.append("No specific error context found in this session.")
            return "\n".join(parts)
        
        for block in error_blocks[:5]:
            parts.append(block)
        
        return "\n\n".join(parts)
    
    def _build_layer_f(self, doc: SessionDocument) -> str:
        """Layer F: Reasoning trail (for CONTINUE intent)."""
        parts = ["## Reasoning Trail"]
        
        for i, block in enumerate(doc.reasoning_docs[:15]):
            # Cap each block
            capped = block[:400]
            if len(block) > 400:
                capped += "..."
            parts.append(f"**Step {i+1}:** {capped}")
        
        if len(doc.reasoning_docs) > 15:
            parts.append(f"\n*({len(doc.reasoning_docs) - 15} more reasoning blocks omitted)*")
        
        return "\n\n".join(parts)


def assemble_fork_context(doc: SessionDocument, intent: str = "continue",
                           query: str = "", llm: Optional[LLMProvider] = None,
                           vector_index = None, store = None) -> str:
    """Convenience function to assemble fork context.
    
    Args:
        doc: SessionDocument
        intent: "continue", "reference", or "debug"
        query: Optional search query for parent context retrieval
        llm: Optional LLMProvider for LLM-powered assembly
        vector_index: Optional VectorIndex
        store: Optional MetadataStore
    
    Returns:
        Formatted fork context string
    """
    intent_map = {
        "continue": ForkIntent.CONTINUE,
        "reference": ForkIntent.REFERENCE,
        "debug": ForkIntent.DEBUG,
    }
    fork_intent = intent_map.get(intent.lower(), ForkIntent.CONTINUE)
    
    assembler = ForkAssembler(llm=llm, vector_index=vector_index, store=store)
    return assembler.assemble(doc, fork_intent, custom_query=query)
