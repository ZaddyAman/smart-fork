"""Session summarizer — generates 3-sentence summaries at index time (v2).

One LLM call per session. Summaries serve TWO purposes:
1. Search ranking: BM25 + vector index uses summaries for better matching
2. Quick context: Users see a high-level overview before forking

Summaries answer 3 questions:
1. What the developer asked for (task)
2. What approach was taken or what decisions were made
3. What was changed or accomplished
"""

from typing import Optional
from loguru import logger

from ..database.models import SessionDocument
from .llm_provider import LLMProvider


SUMMARY_PROMPT = """You are summarizing a coding session for a developer search tool.
Write a concise 3-sentence summary answering:
1. What was the developer building or fixing?
2. What approach or key decisions were made?
3. What was the outcome (completed, partially done, or blocked)?

Session info:
- Project: {project_name}
- Task: {task_raw}
- Files edited: {files_edited}
- Domains: {domains}
- Duration: {duration} minutes
- Key reasoning excerpts:
{reasoning_excerpt}

Write ONLY the 3-sentence summary. No preamble, no bullet points, just 3 sentences."""


class SessionSummarizer:
    """Generates 3-sentence summaries for indexed sessions.
    
    Cost: Free with local Ollama (qwen3:0.6b).
    Speed: ~2-5s per session on GPU, ~15-30s on CPU.
    
    Usage:
        summarizer = SessionSummarizer(llm=get_llm("ollama"))
        summary = summarizer.summarize(session_doc)
    """
    
    def __init__(self, llm: LLMProvider):
        self.llm = llm
    
    def summarize(self, doc: SessionDocument) -> str:
        """Generate a 3-sentence summary for a session.
        
        Args:
            doc: SessionDocument with task_raw, files_edited, reasoning_docs
        
        Returns:
            3-sentence summary string, or empty string on failure
        """
        if not doc.task_raw and not doc.reasoning_docs:
            return ""
        
        # Build prompt inputs
        task_raw = (doc.task_raw or "no task description")[:500]
        files_edited = ", ".join(
            f.split("/")[-1] for f in doc.files_edited[:8]
        ) if doc.files_edited else "none"
        domains = ", ".join(doc.domains[:5]) if doc.domains else "none"
        duration = round(doc.duration_minutes, 1) if doc.duration_minutes else "unknown"
        
        # Generate 3-window (33% / 66% / 100%) Temporal Summary using sumy TextRank
        reasoning_excerpt = ""
        if doc.reasoning_docs:
            try:
                from sumy.parsers.plaintext import PlaintextParser
                from sumy.nlp.tokenizers import Tokenizer
                from sumy.summarizers.text_rank import TextRankSummarizer
                import nltk
                try:
                    nltk.data.find('tokenizers/punkt')
                except LookupError:
                    # Some versions require punkt_tab, some punkt. download both.
                    nltk.download('punkt', quiet=True)
                    nltk.download('punkt_tab', quiet=True)
                
                blocks = doc.reasoning_docs
                n = len(blocks)
                
                # Split independently: 0-33%, 33-66%, 66-100%
                w1_end = max(1, n // 3)
                w2_end = max(2, (2 * n) // 3)
                
                start_window = blocks[:w1_end]
                mid_window = blocks[w1_end:w2_end]
                end_window = blocks[w2_end:]
                
                def extract_text(window_blocks: list[str]) -> str:
                    text = "\n".join(window_blocks).strip()
                    if not text:
                        return ""
                    parser = PlaintextParser.from_string(text, Tokenizer("english"))
                    summarizer = TextRankSummarizer()
                    sentences = summarizer(parser.document, 2)
                    return " ".join(str(s) for s in sentences)
                
                excerpts = []
                for win in [start_window, mid_window, end_window]:
                    if win:
                        e = extract_text(win)
                        if e:
                            excerpts.append(e)
                reasoning_excerpt = "\n---\n".join(excerpts)
            except Exception as e:
                logger.warning(f"sumy TextRank generation failed, falling back to simple extraction: {e}")
                
            if not reasoning_excerpt:
                # Fallback if sumy failed or returned empty
                excerpts = []
                for r in doc.reasoning_docs[:5]:
                    excerpt = r[:500].strip()
                    if len(r) > 500:
                        excerpt += "..."
                    excerpts.append(excerpt)
                reasoning_excerpt = "\n---\n".join(excerpts)
                
        reasoning_excerpt = reasoning_excerpt or "no reasoning available"
        
        prompt = SUMMARY_PROMPT.format(
            project_name=doc.project_name or "unknown",
            task_raw=task_raw,
            files_edited=files_edited,
            domains=domains,
            duration=duration,
            reasoning_excerpt=reasoning_excerpt,
        )
        
        try:
            summary = self.llm.complete(prompt, max_tokens=250)
            summary = summary.strip()
            
            if not summary:
                return ""
            
            # Remove any thinking tags if model wraps output
            import re
            summary = re.sub(r'<think>.*?</think>', '', summary, flags=re.DOTALL).strip()
            
            # Validate: should be 1-3 sentences, not too long
            if len(summary) > 600:
                sentences = summary.split('. ')
                summary = '. '.join(sentences[:3])
                if not summary.endswith('.'):
                    summary += '.'
            
            # Remove any preamble like "Here is the summary:" etc
            preamble_patterns = [
                r'^(?:Here (?:is|are) (?:the|a) )?summary[:\s]*',
                r'^(?:The )?3-sentence summary[:\s]*',
                r'^Summary[:\s]*',
            ]
            for pattern in preamble_patterns:
                summary = re.sub(pattern, '', summary, flags=re.IGNORECASE).strip()
            
            logger.debug(f"Generated summary for {doc.session_id}: {summary[:80]}...")
            return summary
            
        except Exception as e:
            logger.warning(f"Summary generation failed for {doc.session_id}: {e}")
            return ""
    
    def summarize_batch(self, docs: list[SessionDocument],
                        skip_existing: bool = True) -> dict[str, str]:
        """Generate summaries for multiple sessions.
        
        Args:
            docs: List of SessionDocuments to summarize
            skip_existing: Skip sessions that already have summary_doc
        
        Returns:
            Dict mapping session_id to generated summary
        """
        results = {}
        
        for doc in docs:
            if skip_existing and doc.summary_doc:
                continue
            
            summary = self.summarize(doc)
            if summary:
                results[doc.session_id] = summary
        
        logger.info(f"Generated {len(results)} summaries out of {len(docs)} sessions")
        return results
