"""Proposition extraction — extracts atomic facts from session content (v2).

Two extraction modes:
1. Rule-based (task_raw): Handles ~70% of propositions without LLM
2. LLM-based (reasoning blocks): Optional, for free-form reasoning text

Propositions are independently searchable, self-describing factual statements.
They are the most powerful technique for "decision hunting" queries.
"""

import re
from typing import List, Optional
from loguru import logger

from ..database.models import SessionDocument
from .llm_provider import LLMProvider


PROPOSITION_PROMPT = """Extract atomic factual statements from this coding session text.
Each proposition should be a single, self-contained fact.

Session text:
{text}

Return one proposition per line. No numbering, no preamble. Just the facts.
Example format:
Developer chose JWT authentication for BharatLawAI
Session-based auth was rejected because ChromaDB lacks session support
BharatLawAI uses ChromaDB as its vector database"""


class PropositionExtractor:
    """Extracts atomic fact statements from session content.
    
    Usage:
        extractor = PropositionExtractor(llm=get_llm("ollama"))
        props = extractor.extract(session_doc)
        # → ["Developer implemented JWT for BharatLawAI", ...]
    """
    
    def __init__(self, llm: Optional[LLMProvider] = None):
        self.llm = llm
    
    def extract(self, doc: SessionDocument) -> List[str]:
        """Extract propositions from a session document.
        
        Uses rule-based extraction for task_raw and LLM for reasoning blocks.
        
        Args:
            doc: SessionDocument to extract from
        
        Returns:
            List of proposition strings
        """
        propositions = []
        
        # Rule-based extraction from task_raw (no LLM needed)
        task_props = self.extract_task_propositions(
            doc.task_raw, doc.project_name, doc.files_edited
        )
        propositions.extend(task_props)
        
        # LLM-based extraction from reasoning (if LLM available)
        if self.llm and doc.reasoning_docs:
            reasoning_props = self.extract_reasoning_propositions(doc.reasoning_docs)
            propositions.extend(reasoning_props)
        
        return propositions
    
    def extract_task_propositions(self, task_raw: str, project: str,
                                   files: List[str]) -> List[str]:
        """Rule-based proposition extraction from task description.
        
        Handles ~70% of propositions without any LLM call.
        
        Args:
            task_raw: Raw task description text
            project: Project name
            files: List of edited files
        
        Returns:
            List of proposition strings
        """
        if not task_raw:
            return []
        
        propositions = []
        project_prefix = f"for {project}" if project != "unknown_project" else ""
        
        # Main task proposition
        task_clean = task_raw.strip().rstrip(".")
        if task_clean:
            propositions.append(f"Developer task: {task_clean} {project_prefix}".strip())
        
        # File-based propositions
        if files:
            file_list = ", ".join(f.split("/")[-1] for f in files[:5])
            propositions.append(
                f"Files modified {project_prefix}: {file_list}".strip()
            )
        
        # Extract technology mentions from task
        tech_patterns = {
            "JWT": "JWT authentication",
            "React": "React framework",
            "FastAPI": "FastAPI framework",
            "ChromaDB": "ChromaDB vector database",
            "Docker": "Docker containerization",
            "PostgreSQL": "PostgreSQL database",
            "Redis": "Redis cache",
            "GraphQL": "GraphQL API",
            "REST": "REST API",
            "WebSocket": "WebSocket connection",
        }
        for tech, description in tech_patterns.items():
            if tech.lower() in task_raw.lower():
                propositions.append(f"{project} uses {description}" if project != "unknown_project"
                                   else f"Session involves {description}")
        
        return propositions
    
    def extract_reasoning_propositions(self, reasoning_docs: List[str]) -> List[str]:
        """LLM-based proposition extraction from reasoning blocks.
        
        Args:
            reasoning_docs: List of reasoning text blocks
        
        Returns:
            List of proposition strings
        """
        if not self.llm or not reasoning_docs:
            return []
        
        propositions = []
        
        # Only process first 3 reasoning blocks to limit LLM cost
        for block in reasoning_docs[:3]:
            if len(block) < 50:  # Skip very short blocks
                continue
            
            try:
                # Truncate to ~500 chars for the prompt
                text = block[:500] + ("..." if len(block) > 500 else "")
                prompt = PROPOSITION_PROMPT.format(text=text)
                
                response = self.llm.complete(prompt, max_tokens=300)
                
                if response:
                    # Parse one proposition per line
                    lines = response.strip().split('\n')
                    for line in lines:
                        line = line.strip().lstrip('- ').lstrip('• ').strip()
                        if line and len(line) > 10 and len(line) < 200:
                            propositions.append(line)
            
            except Exception as e:
                logger.warning(f"Proposition extraction failed: {e}")
                continue
        
        return propositions
