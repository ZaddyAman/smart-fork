"""Structured session parser for Kilo Code transcript files (v2).

Replaces the v1 parser with a full structured signal extractor that
respects the natural structure of Kilo Code's 3-file session format.

Key differences from v1:
- Extracts specific signals from specific locations (not concatenation)
- Categorizes files by record_source (edited/read/mentioned/user_edited)
- Extracts <task> tag text as the primary retrieval key
- Only extracts say:"reasoning" blocks from ui_messages.json
- Skips <file_content> tags, <environment_details>, and tool call turns
- Derives project name, domains, languages, patterns from file paths
"""

import json
import re
import time
from pathlib import Path
from typing import Optional, List
from loguru import logger

from ..database.models import (
    SessionDocument, TaskMetadataV2, ConversationDataV2
)
from .project_extractor import (
    derive_project_name, extract_domains, extract_languages,
    extract_layers, classify_session_pattern
)


# Tool call indicators — assistant turns containing these are structural, not semantic
TOOL_CALL_TAGS = [
    "<read_file>", "<write_to_file>", "<list_code_definitions>",
    "<search_files>", "<list_files>", "<execute_command>",
    "<ask_followup_question>", "<attempt_completion>",
    "<replace_in_file>", "<insert_code_block>",
    "<browser_action>", "<use_mcp_tool>",
    "<access_mcp_resource>", "<switch_mode>",
]


class SessionParser:
    """Parses all 3 Kilo Code files and produces a structured SessionDocument.
    
    This is the entry point for the v2 indexing pipeline:
        SessionParser.parse_session() → SessionDocument → MetadataStore + VectorIndex
    """
    
    def parse_session(self, session_path: Path) -> Optional[SessionDocument]:
        """Parse a Kilo Code session directory into a structured SessionDocument.
        
        Args:
            session_path: Path to the session folder (e.g., .../tasks/1773250127795/)
        
        Returns:
            SessionDocument with all structured fields populated, or None if parsing fails.
        """
        if not session_path.is_dir():
            logger.warning(f"Not a directory: {session_path}")
            return None
        
        session_id = session_path.name
        
        try:
            # Step 1: Parse task_metadata.json (structured file signals)
            metadata_path = session_path / "task_metadata.json"
            task_meta = self._parse_task_metadata(metadata_path) if metadata_path.exists() else TaskMetadataV2()
            
            # Step 2: Parse api_conversation_history.json (semantic content)
            history_path = session_path / "api_conversation_history.json"
            conv_data = self._parse_api_conversation(history_path) if history_path.exists() else ConversationDataV2()
            
            # Step 3: Parse ui_messages.json (reasoning blocks only)
            ui_path = session_path / "ui_messages.json"
            reasoning_blocks = self._parse_ui_messages(ui_path) if ui_path.exists() else []
            
            # Step 4: Combine all file paths for derivation
            all_file_paths = list(set(
                task_meta.files_edited + task_meta.files_read +
                task_meta.files_mentioned + task_meta.files_user_edited
            ))
            
            # Step 5: Derive project name (zero LLM)
            project_name = derive_project_name(conv_data.workspace_dir, all_file_paths)
            
            # Step 6: Derive domains, languages, layers (zero LLM)
            domains = extract_domains(all_file_paths)
            languages = extract_languages(all_file_paths)
            layers = extract_layers(all_file_paths)
            
            # Step 7: Classify session pattern
            session_pattern = classify_session_pattern(
                conv_data.tool_call_sequence,
                task_meta.user_edit_count,
                task_meta.edit_count
            )
            
            # Step 8: Merge reasoning from both sources
            # api_conversation reasoning turns + ui_messages reasoning blocks
            all_reasoning = conv_data.reasoning_turns + reasoning_blocks
            # Deduplicate while preserving order (some reasoning appears in both)
            seen = set()
            unique_reasoning = []
            for r in all_reasoning:
                # Use first 100 chars as dedup key (exact match is too strict)
                key = r[:100].strip() if r else ""
                if key and key not in seen:
                    seen.add(key)
                    unique_reasoning.append(r)
            
            # Step 9: Build the SessionDocument
            doc = SessionDocument(
                session_id=session_id,
                project_name=project_name,
                project_root=conv_data.workspace_dir,
                session_start=conv_data.session_start,
                session_end=conv_data.session_end,
                duration_minutes=conv_data.duration_minutes,
                model_used=conv_data.model_used,
                
                files_edited=task_meta.files_edited,
                files_read=task_meta.files_read,
                files_mentioned=task_meta.files_mentioned,
                edit_count=task_meta.edit_count,
                user_edit_count=task_meta.user_edit_count,
                final_files=task_meta.final_files,
                
                domains=domains,
                languages=languages,
                layers=layers,
                session_pattern=session_pattern,
                
                task_raw=conv_data.task_raw,
                reasoning_docs=unique_reasoning,
                
                indexed_at=int(time.time() * 1000),
                schema_version=2,
            )
            
            logger.debug(
                f"Parsed session {session_id}: project={project_name}, "
                f"task={conv_data.task_raw[:50]}..., "
                f"domains={domains}, "
                f"reasoning_blocks={len(unique_reasoning)}, "
                f"files_edited={len(task_meta.files_edited)}"
            )
            
            return doc
            
        except Exception as e:
            logger.error(f"Failed to parse session {session_id}: {e}")
            return None
    
    def _parse_task_metadata(self, path: Path) -> TaskMetadataV2:
        """Parse task_metadata.json into categorized file signals.
        
        Categorizes files by their record_source:
        - roo_edited → files_edited (AI modified these)
        - read_tool → files_read (AI read for context) 
        - file_mentioned → files_mentioned (referenced in task)
        - user_edited → files_user_edited (human changed these)
        
        Also tracks:
        - final_files: record_state="active" AND roo_edit_date is not null
        - edit_count / user_edit_count: count of each type
        """
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        files_edited = []
        files_read = []
        files_mentioned = []
        files_user_edited = []
        final_files = []
        
        for record in data.get('files_in_context', []):
            file_path = record.get('path', '')
            if not file_path:
                continue
            
            source = record.get('record_source', '')
            state = record.get('record_state', '')
            
            if source == 'roo_edited':
                files_edited.append(file_path)
            elif source == 'read_tool':
                files_read.append(file_path)
            elif source == 'file_mentioned':
                files_mentioned.append(file_path)
            elif source == 'user_edited':
                files_user_edited.append(file_path)
            
            # Final files = active state + has been edited by AI
            if state == 'active' and record.get('roo_edit_date') is not None:
                final_files.append(file_path)
        
        return TaskMetadataV2(
            files_edited=files_edited,
            files_read=files_read,
            files_mentioned=files_mentioned,
            files_user_edited=files_user_edited,
            edit_count=len(files_edited),
            user_edit_count=len(files_user_edited),
            final_files=final_files,
            raw_data=data,
        )
    
    def _parse_api_conversation(self, path: Path) -> ConversationDataV2:
        """Parse api_conversation_history.json for structured signals.
        
        Extracts:
        - task_raw: Text inside <task> tag in first user turn
        - workspace_dir: From <environment_details> → Current Workspace Directory
        - session timestamps and duration
        - model_used: From <environment_details> → Current Mode
        - open_tabs: From <environment_details> → VSCode Open Tabs
        - reasoning_turns: Assistant turns that are NOT tool calls
        - tool_call_sequence: Tool call names (for session pattern classification)
        
        SKIPS: <file_content> tags, full <environment_details> blocks
        """
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list) or len(data) == 0:
            return ConversationDataV2()
        
        task_raw = ""
        workspace_dir = ""
        model_used = None
        open_tabs = []
        reasoning_turns = []
        tool_call_sequence = []
        
        # Extract timestamps
        session_start = data[0].get('ts', 0) if data else 0
        session_end = data[-1].get('ts', 0) if data else 0
        duration_minutes = (session_end - session_start) / 60000.0 if session_end > session_start else 0.0
        
        for i, msg in enumerate(data):
            role = msg.get('role', '')
            content = self._extract_text_content(msg)
            
            if not content:
                continue
            
            if role == 'user':
                # Extract <task> tag from first user message
                if not task_raw:
                    task_match = re.search(r'<task>\s*(.*?)\s*</task>', content, re.DOTALL)
                    if task_match:
                        task_raw = task_match.group(1).strip()
                
                # Extract workspace directory from <environment_details>
                if not workspace_dir:
                    ws_match = re.search(
                        r'<environment_details>.*?'
                        r'(?:Current Workspace Directory|Current Working Directory)\s*\n\s*(.*?)\s*\n',
                        content, re.DOTALL
                    )
                    if ws_match:
                        workspace_dir = ws_match.group(1).strip()
                
                # Extract model/mode from <environment_details>
                if model_used is None:
                    mode_match = re.search(
                        r'<environment_details>.*?Current Mode:\s*(.*?)\s*\n',
                        content, re.DOTALL
                    )
                    if mode_match:
                        model_used = mode_match.group(1).strip()
                
                # Extract open tabs from <environment_details>
                if not open_tabs:
                    tabs_match = re.search(
                        r'# VSCode (?:Visible|Open) (?:Files|Tabs)\s*\n(.*?)(?:\n#|\n</environment_details>)',
                        content, re.DOTALL
                    )
                    if tabs_match:
                        tabs_text = tabs_match.group(1).strip()
                        open_tabs = [t.strip() for t in tabs_text.split('\n') if t.strip()]
            
            elif role == 'assistant':
                # Classify: is this a tool call turn or a reasoning turn?
                is_tool_call = self._is_tool_call(content)
                
                if is_tool_call:
                    # Extract tool call name for pattern classification
                    tool_name = self._extract_tool_name(content)
                    if tool_name:
                        tool_call_sequence.append(tool_name)
                else:
                    # This is a reasoning turn — strip noise, keep the reasoning
                    clean_reasoning = self._clean_reasoning_text(content)
                    if clean_reasoning and len(clean_reasoning) > 20:
                        reasoning_turns.append(clean_reasoning)
        
        return ConversationDataV2(
            task_raw=task_raw,
            workspace_dir=workspace_dir,
            session_start=session_start,
            session_end=session_end,
            duration_minutes=round(duration_minutes, 1),
            model_used=model_used,
            open_tabs=open_tabs,
            reasoning_turns=reasoning_turns,
            tool_call_sequence=tool_call_sequence,
        )
    
    def _parse_ui_messages(self, path: Path) -> List[str]:
        """Parse ui_messages.json — extract ONLY say:"reasoning" blocks.
        
        These are the AI's internal chain-of-thought blocks containing
        architectural reasoning, decision rationale, approach selection.
        
        Everything else in ui_messages.json is skipped (redundant with
        api_conversation_history.json).
        """
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            return []
        
        reasoning_blocks = []
        for msg in data:
            if msg.get('say') == 'reasoning':
                text = msg.get('text', '').strip()
                if text and len(text) > 20:
                    reasoning_blocks.append(text)
        
        return reasoning_blocks
    
    # ═══════════════════════════════════════════════════════════════════════════
    # HELPER METHODS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _extract_text_content(self, msg: dict) -> str:
        """Extract text content from a conversation message.
        
        Handles both string content and array-of-parts content format.
        """
        content = msg.get('content', '')
        
        if isinstance(content, str):
            return content
        
        if isinstance(content, list):
            texts = []
            for part in content:
                if isinstance(part, dict):
                    if 'text' in part:
                        text_val = part['text']
                        if isinstance(text_val, dict):
                            texts.append(text_val.get('value', ''))
                        else:
                            texts.append(str(text_val))
            return '\n'.join(texts)
        
        return ''
    
    def _is_tool_call(self, content: str) -> bool:
        """Check if an assistant turn is a tool call (structural, not semantic).
        
        Tool call turns contain XML tags like <read_file>, <write_to_file>, etc.
        These are structural operations with zero semantic retrieval value.
        """
        for tag in TOOL_CALL_TAGS:
            if tag in content:
                return True
        return False
    
    def _extract_tool_name(self, content: str) -> Optional[str]:
        """Extract the tool name from a tool call turn.
        
        Used to build tool_call_sequence for session pattern classification.
        """
        match = re.search(r'<(\w+)>', content)
        if match:
            return match.group(1)
        return None
    
    def _clean_reasoning_text(self, content: str) -> str:
        """Clean assistant reasoning text by removing noise.
        
        Removes:
        - <file_content> blocks (code dumps, too noisy)
        - <environment_details> blocks (already captured in metadata)
        - XML tags
        - SEARCH/REPLACE diff blocks (>>>>>> REPLACE, <<<<<<, =======)
        - Raw terminal error output (npm, pip, git errors)
        - Excessive file path listings
        - Repetitive whitespace
        """
        text = content
        
        # Remove <file_content>...</file_content> blocks
        text = re.sub(r'<file_content.*?>.*?</file_content>', '', text, flags=re.DOTALL)
        
        # Remove <environment_details>...</environment_details> blocks
        text = re.sub(r'<environment_details>.*?</environment_details>', '', text, flags=re.DOTALL)
        
        # Remove remaining XML-like tags but keep content between them
        text = re.sub(r'<[^>]+>', '', text)
        
        # ── DIFF / SEARCH-REPLACE BLOCK REMOVAL ──
        # Remove full SEARCH/REPLACE blocks: <<<<<<< SEARCH ... ======= ... >>>>>>> REPLACE
        text = re.sub(
            r'<{3,}<?<?<?\s*SEARCH.*?>{3,}>?>?>?\s*REPLACE',
            '', text, flags=re.DOTALL
        )
        # Remove standalone diff markers
        text = re.sub(r'^[<>=]{3,}.*$', '', text, flags=re.MULTILINE)
        # Remove >>>>>> REPLACE, <<<<<<< SEARCH etc standalone lines
        text = re.sub(r'^\s*>{3,}\s*(REPLACE|replace).*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*<{3,}\s*(SEARCH|search).*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*={3,}\s*$', '', text, flags=re.MULTILINE)
        
        # ── RAW TERMINAL OUTPUT REMOVAL ──
        # npm/pip/git error blocks (multi-line error dumps)
        text = re.sub(
            r'(?:npm\s+(?:error|warn|ERR!).*\n?){2,}',
            '[terminal output removed]', text
        )
        text = re.sub(
            r'(?:pip\s+(?:error|WARNING).*\n?){2,}',
            '[terminal output removed]', text
        )
        # Traceback blocks
        text = re.sub(
            r'Traceback \(most recent call last\):.*?(?=\n\n|\Z)',
            '[traceback removed]', text, flags=re.DOTALL
        )
        
        # ── CODE BLOCK CLEANUP ──
        # Remove large inline code blocks (>10 lines) — these are file dumps, not reasoning
        text = re.sub(
            r'```[\w]*\n(?:[^\n]*\n){10,}```',
            '[code block removed]', text
        )
        
        # ── FILE PATH LIST CLEANUP ──
        # Cap long lists of file paths (>5 consecutive path lines)
        def _cap_file_list(match):
            lines = match.group(0).strip().split('\n')
            if len(lines) > 5:
                return '\n'.join(lines[:5]) + f'\n... and {len(lines) - 5} more files'
            return match.group(0)
        text = re.sub(
            r'(?:[-•*]\s*`?[\w/\\]+\.[\w]+`?\s*\n){5,}',
            _cap_file_list, text
        )
        
        # ── FINAL CLEANUP ──
        # Clean up excessive whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        # Remove lines that are only whitespace or punctuation
        text = re.sub(r'^\s*[^\w\s]*\s*$', '', text, flags=re.MULTILINE)
        # Collapse multiple blank lines again after removals
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = text.strip()
        
        # Drop if it's too short after cleaning (just noise)
        if len(text) < 30:
            return ""
        
        return text

