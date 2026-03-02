"""Conversation branching tree for tracking session lineage."""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

from loguru import logger

from ..config import get_config


@dataclass
class SessionBranch:
    """Represents a branch in the conversation tree."""
    session_id: str
    parent_id: Optional[str] = None
    children: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    topic_hint: str = ""
    files_touched: Set[str] = field(default_factory=set)
    is_archived: bool = False
    
    def to_dict(self) -> Dict:
        return {
            "session_id": self.session_id,
            "parent_id": self.parent_id,
            "children": self.children,
            "created_at": self.created_at.isoformat(),
            "topic_hint": self.topic_hint,
            "files_touched": list(self.files_touched),
            "is_archived": self.is_archived
        }
        
    @classmethod
    def from_dict(cls, data: Dict) -> "SessionBranch":
        return cls(
            session_id=data["session_id"],
            parent_id=data.get("parent_id"),
            children=data.get("children", []),
            created_at=datetime.fromisoformat(data["created_at"]),
            topic_hint=data.get("topic_hint", ""),
            files_touched=set(data.get("files_touched", [])),
            is_archived=data.get("is_archived", False)
        )


class BranchingTree:
    """Manages conversation branching tree structure."""
    
    def __init__(self):
        self.branches: Dict[str, SessionBranch] = {}
        cfg = get_config()
        self.tree_file = cfg.cache_dir / "branching_tree.json"
        self._load_tree()
        
    def _load_tree(self):
        """Load tree from disk."""
        if self.tree_file.exists():
            try:
                with open(self.tree_file, 'r') as f:
                    data = json.load(f)
                    
                for session_id, branch_data in data.items():
                    self.branches[session_id] = SessionBranch.from_dict(branch_data)
                    
                logger.info(f"Loaded {len(self.branches)} branches from tree")
            except Exception as e:
                logger.warning(f"Failed to load branching tree: {e}")
                
    def _save_tree(self):
        """Save tree to disk."""
        data = {sid: branch.to_dict() for sid, branch in self.branches.items()}
        with open(self.tree_file, 'w') as f:
            json.dump(data, f, indent=2)
            
    def add_session(self, session_id: str, parent_id: Optional[str] = None,
                    topic_hint: str = "", files: Optional[List[str]] = None):
        """Add a new session to the tree."""
        if session_id in self.branches:
            return
            
        branch = SessionBranch(
            session_id=session_id,
            parent_id=parent_id,
            topic_hint=topic_hint,
            files_touched=set(files or [])
        )
        
        self.branches[session_id] = branch
        
        # Update parent's children
        if parent_id and parent_id in self.branches:
            if session_id not in self.branches[parent_id].children:
                self.branches[parent_id].children.append(session_id)
                
        self._save_tree()
        logger.debug(f"Added session {session_id} to tree")
        
    def get_lineage(self, session_id: str) -> List[str]:
        """Get ancestry lineage of a session."""
        lineage = []
        current = session_id
        
        while current and current in self.branches:
            lineage.append(current)
            current = self.branches[current].parent_id
            
        return list(reversed(lineage))
        
    def get_descendants(self, session_id: str) -> List[str]:
        """Get all descendants of a session."""
        if session_id not in self.branches:
            return []
            
        descendants = []
        queue = [session_id]
        
        while queue:
            current = queue.pop(0)
            if current != session_id:
                descendants.append(current)
                
            if current in self.branches:
                queue.extend(self.branches[current].children)
                
        return descendants
        
    def find_related_sessions(self, session_id: str, max_depth: int = 2) -> List[str]:
        """Find sessions related by topic or files."""
        if session_id not in self.branches:
            return []
            
        branch = self.branches[session_id]
        related = set()
        
        # Check siblings
        if branch.parent_id and branch.parent_id in self.branches:
            parent = self.branches[branch.parent_id]
            for child in parent.children:
                if child != session_id:
                    related.add(child)
                    
        # Check file overlap
        for sid, other_branch in self.branches.items():
            if sid != session_id:
                overlap = branch.files_touched & other_branch.files_touched
                if len(overlap) > 0:
                    related.add(sid)
                    
        return list(related)
        
    def visualize_tree(self, root_id: Optional[str] = None, compact: bool = True) -> str:
        """Generate ASCII tree visualization (Windows-compatible)."""
        lines = []
        
        def clean_topic(topic: str, max_len: int = 35) -> str:
            """Clean and truncate topic for display."""
            # Remove newlines and extra whitespace
            topic = " ".join(topic.split())
            # Remove XML tags
            import re
            topic = re.sub(r'<[^>]+>', '', topic)
            # Truncate
            if len(topic) > max_len:
                topic = topic[:max_len-3] + "..."
            return topic.strip()
        
        def render_branch(session_id: str, prefix: str = "", is_last: bool = True, depth: int = 0):
            if session_id not in self.branches:
                return
                
            branch = self.branches[session_id]
            # Use ASCII-only characters for Windows compatibility
            connector = "`-- " if is_last else "|-- "
            
            # Build display string
            sid_short = session_id[:8]
            
            if compact:
                # Compact format: ID + short topic
                topic = clean_topic(branch.topic_hint, 30)
                topic_str = f" -- {topic}" if topic else ""
                line = f"{prefix}{connector}{sid_short}{topic_str}"
                lines.append(line[:80])  # Limit line length
            else:
                # Expanded format with more details
                topic = clean_topic(branch.topic_hint, 50)
                topic_str = f"\n{prefix}    [T] {topic}" if topic else ""
                files_str = ""
                if branch.files_touched:
                    files_list = list(branch.files_touched)[:3]
                    files_str = f"\n{prefix}    [F] {', '.join(files_list)}"
                    if len(branch.files_touched) > 3:
                        files_str += f" (+{len(branch.files_touched)-3} more)"
                
                lines.append(f"{prefix}{connector}{sid_short}{topic_str}{files_str}")
            
            children = branch.children
            for i, child_id in enumerate(children):
                is_last_child = (i == len(children) - 1)
                new_prefix = prefix + ("    " if is_last else "|   ")
                render_branch(child_id, new_prefix, is_last_child, depth + 1)
                
        if root_id:
            render_branch(root_id)
        else:
            # Find roots (no parent)
            roots = [sid for sid, b in self.branches.items() if b.parent_id is None]
            for i, root in enumerate(roots):
                render_branch(root, is_last=(i == len(roots) - 1))
                
        return "\n".join(lines) if lines else "(empty tree)"
    
    def export_html(self, output_path: Optional[Path] = None) -> Path:
        """Export tree as interactive HTML visualization."""
        import html
        
        if output_path is None:
            cfg = get_config()
            output_path = cfg.cache_dir / "conversation_tree.html"
        
        # Build tree data for visualization
        tree_data = []
        for session_id, branch in self.branches.items():
            tree_data.append({
                "id": session_id,
                "parent": branch.parent_id if branch.parent_id else "#",
                "text": f"{session_id[:8]}: {branch.topic_hint[:50]}..." if len(branch.topic_hint) > 50 else f"{session_id[:8]}: {branch.topic_hint}",
                "topic": branch.topic_hint,
                "files": list(branch.files_touched),
                "created": branch.created_at.isoformat()
            })
        
        html_content = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>SmartFork Conversation Tree</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/jstree/3.3.15/jstree.min.js"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/jstree/3.3.15/themes/default/style.min.css">
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1400px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; margin-bottom: 10px; }}
        .stats {{ display: flex; gap: 20px; margin-bottom: 20px; flex-wrap: wrap; }}
        .stat {{ background: #f0f0f0; padding: 10px 20px; border-radius: 4px; }}
        .stat-value {{ font-size: 24px; font-weight: bold; color: #2196F3; }}
        .stat-label {{ font-size: 12px; color: #666; text-transform: uppercase; }}
        #tree {{ margin-top: 20px; }}
        .jstree-anchor {{ font-size: 14px; }}
        .topic-preview {{ color: #666; font-size: 12px; margin-left: 10px; }}
        .detail-panel {{
            position: fixed; right: 20px; top: 20px; width: 350px;
            background: white; padding: 20px; border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15); display: none;
            max-height: 80vh; overflow-y: auto;
        }}
        .detail-panel.active {{ display: block; }}
        .detail-panel h3 {{ margin-top: 0; color: #333; }}
        .detail-panel .files {{ margin-top: 10px; }}
        .detail-panel .file-tag {{
            display: inline-block; background: #e3f2fd; color: #1976d2;
            padding: 2px 8px; border-radius: 12px; font-size: 11px; margin: 2px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🌳 Conversation Branching Tree</h1>
        <div class="stats">
            <div class="stat">
                <div class="stat-value">{len(self.branches)}</div>
                <div class="stat-label">Total Sessions</div>
            </div>
            <div class="stat">
                <div class="stat-value">{len([b for b in self.branches.values() if b.parent_id is None])}</div>
                <div class="stat-label">Root Sessions</div>
            </div>
            <div class="stat">
                <div class="stat-value">{len([b for b in self.branches.values() if not b.children])}</div>
                <div class="stat-label">Leaf Sessions</div>
            </div>
            <div class="stat">
                <div class="stat-value">{max((len(self.get_lineage(sid)) for sid in self.branches), default=0)}</div>
                <div class="stat-label">Max Depth</div>
            </div>
        </div>
        <div id="tree"></div>
    </div>
    
    <div class="detail-panel" id="detailPanel">
        <h3>Session Details</h3>
        <div id="detailContent"></div>
    </div>
    
    <script>
        const treeData = {json.dumps(tree_data)};
        
        $(function() {{
            $('#tree').jstree({{
                'core': {{
                    'data': treeData,
                    'themes': {{
                        'responsive': true,
                        'variant': 'large'
                    }}
                }},
                'plugins': ['search', 'wholerow']
            }});
            
            $('#tree').on('select_node.jstree', function(e, data) {{
                const node = data.node;
                const sessionData = treeData.find(s => s.id === node.id);
                if (sessionData) {{
                    showDetails(sessionData);
                }}
            }});
        }});
        
        function showDetails(session) {{
            const panel = document.getElementById('detailPanel');
            const content = document.getElementById('detailContent');
            
            let filesHtml = '';
            if (session.files && session.files.length > 0) {{
                filesHtml = '<div class="files"><strong>Files:</strong><br>' +
                    session.files.map(f => `<span class="file-tag">${{f}}</span>`).join('') +
                    '</div>';
            }}
            
            content.innerHTML = `
                <p><strong>ID:</strong> ${{session.id}}</p>
                <p><strong>Created:</strong> ${{new Date(session.created).toLocaleString()}}</p>
                <div style="margin-top: 15px;">
                    <strong>Topic:</strong>
                    <p style="color: #666; font-size: 13px; line-height: 1.5;">${{session.topic || '(No topic)'}}</p>
                </div>
                ${{filesHtml}}
            `;
            
            panel.classList.add('active');
        }}
    </script>
</body>
</html>'''
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Exported HTML tree to {output_path}")
        return output_path
        
    def get_stats(self) -> Dict:
        """Get tree statistics."""
        total = len(self.branches)
        roots = len([b for b in self.branches.values() if b.parent_id is None])
        leaves = len([b for b in self.branches.values() if not b.children])
        max_depth = 0
        
        for session_id in self.branches:
            depth = len(self.get_lineage(session_id))
            max_depth = max(max_depth, depth)
            
        return {
            "total_sessions": total,
            "root_sessions": roots,
            "leaf_sessions": leaves,
            "max_depth": max_depth,
            "avg_children": sum(len(b.children) for b in self.branches.values()) / total if total > 0 else 0
        }
        
    def archive_branch(self, session_id: str):
        """Archive a branch and its descendants."""
        to_archive = [session_id] + self.get_descendants(session_id)
        
        for sid in to_archive:
            if sid in self.branches:
                self.branches[sid].is_archived = True
                
        self._save_tree()
        logger.info(f"Archived branch starting from {session_id}")
        
    def auto_build_tree(self, sessions_dir: Path):
        """Automatically build tree from existing sessions."""
        from ..indexer.parser import KiloCodeParser
        
        parser = KiloCodeParser()
        
        for task_dir in sessions_dir.iterdir():
            if not task_dir.is_dir():
                continue
                
            session_id = task_dir.name
            if session_id in self.branches:
                continue
                
            try:
                # Parse session for metadata
                session = parser.parse_task_directory(task_dir)
                if not session:
                    continue
                metadata = session.metadata
                api_history = session.conversation
                
                # Extract topic hint from first user message
                topic_hint = ""
                for msg in api_history:
                    if msg.role == "user":
                        content = msg.content
                        if isinstance(content, str):
                            topic_hint = content[:100]
                        break
                        
                # Get files
                files = metadata.files_in_context if metadata else []
                
                # Try to find parent by file overlap
                parent_id = None
                best_overlap = 0
                
                for sid, branch in self.branches.items():
                    overlap = len(set(files) & branch.files_touched)
                    if overlap > best_overlap and overlap >= 2:
                        best_overlap = overlap
                        parent_id = sid
                        
                self.add_session(session_id, parent_id, topic_hint, files)
                
            except Exception as e:
                logger.warning(f"Error building tree for {session_id}: {e}")
                continue