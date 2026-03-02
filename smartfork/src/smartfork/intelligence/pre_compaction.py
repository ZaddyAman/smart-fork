"""Pre-compaction hook system for exporting large sessions before summarization."""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from loguru import logger

from ..config import get_config


class PreCompactionHook:
    """Detects large sessions and exports them before Kilo Code compaction."""
    
    def __init__(self, threshold_messages: int = 100, threshold_age_days: int = 7):
        self.threshold_messages = threshold_messages
        self.threshold_age_days = threshold_age_days
        cfg = get_config()
        self.export_dir = cfg.cache_dir / "pre_compaction_exports"
        self.export_dir.mkdir(parents=True, exist_ok=True)
        
    def check_sessions(self, tasks_dir: Path) -> List[Dict]:
        """Check all sessions for compaction risk."""
        at_risk = []
        
        for task_dir in tasks_dir.iterdir():
            if not task_dir.is_dir():
                continue
                
            session_id = task_dir.name
            api_history_file = task_dir / "api_conversation_history.json"
            
            if not api_history_file.exists():
                continue
                
            try:
                with open(api_history_file, 'r', encoding='utf-8') as f:
                    api_history = json.load(f)
                    
                message_count = len(api_history)
                
                # Check modification time
                mtime = datetime.fromtimestamp(api_history_file.stat().st_mtime)
                age_days = (datetime.now() - mtime).days
                
                if message_count > self.threshold_messages or age_days > self.threshold_age_days:
                    at_risk.append({
                        "session_id": session_id,
                        "message_count": message_count,
                        "age_days": age_days,
                        "task_dir": task_dir,
                        "risk_level": "high" if message_count > 200 else "medium"
                    })
                    
            except Exception as e:
                logger.warning(f"Error checking session {session_id}: {e}")
                continue
                
        return sorted(at_risk, key=lambda x: x["message_count"], reverse=True)
        
    def export_session(self, session_info: Dict) -> Optional[Path]:
        """Export a session before compaction."""
        session_id = session_info["session_id"]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_path = self.export_dir / f"{session_id}_{timestamp}"
        
        try:
            # Copy entire task directory
            if export_path.exists():
                shutil.rmtree(export_path)
            shutil.copytree(session_info["task_dir"], export_path)
            
            # Create metadata file
            metadata = {
                "session_id": session_id,
                "exported_at": datetime.now().isoformat(),
                "message_count": session_info["message_count"],
                "age_days": session_info["age_days"],
                "risk_level": session_info["risk_level"],
                "original_path": str(session_info["task_dir"]),
            }
            
            with open(export_path / "_export_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
                
            logger.info(f"Exported session {session_id} to {export_path}")
            return export_path
            
        except Exception as e:
            logger.error(f"Failed to export session {session_id}: {e}")
            return None
            
    def get_exported_sessions(self) -> List[Dict]:
        """Get list of all exported sessions."""
        exports = []
        
        for export_dir in self.export_dir.iterdir():
            if not export_dir.is_dir():
                continue
                
            metadata_file = export_dir / "_export_metadata.json"
            if metadata_file.exists():
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    metadata["export_path"] = str(export_dir)
                    exports.append(metadata)
                except Exception:
                    continue
                    
        return sorted(exports, key=lambda x: x.get("exported_at", ""), reverse=True)
        
    def restore_session(self, session_id: str) -> Optional[Path]:
        """Restore a session from export."""
        for export in self.get_exported_sessions():
            if export["session_id"] == session_id:
                original_path = Path(export["original_path"])
                export_path = Path(export["export_path"])
                
                try:
                    # Restore from export
                    if original_path.exists():
                        backup_path = original_path.with_suffix(".backup")
                        shutil.move(original_path, backup_path)
                        
                    # Copy back (excluding metadata file)
                    shutil.copytree(
                        export_path, 
                        original_path,
                        ignore=lambda d, files: [f for f in files if f.startswith("_")]
                    )
                    
                    logger.info(f"Restored session {session_id} from {export_path}")
                    return original_path
                    
                except Exception as e:
                    logger.error(f"Failed to restore session {session_id}: {e}")
                    return None
                    
        return None


class CompactionManager:
    """Manages session compaction workflow."""
    
    def __init__(self):
        self.hook = PreCompactionHook()
        
    def run_auto_export(self, dry_run: bool = False) -> Dict:
        """Automatically export at-risk sessions."""
        cfg = get_config()
        tasks_dir = cfg.kilo_code_tasks_path
        at_risk = self.hook.check_sessions(tasks_dir)
        
        results = {
            "checked": 0,
            "at_risk": len(at_risk),
            "exported": 0,
            "failed": 0,
            "sessions": []
        }
        
        for session_info in at_risk:
            results["checked"] += 1
            
            if dry_run:
                results["sessions"].append({
                    "session_id": session_info["session_id"],
                    "action": "would_export",
                    "message_count": session_info["message_count"]
                })
            else:
                export_path = self.hook.export_session(session_info)
                if export_path:
                    results["exported"] += 1
                    results["sessions"].append({
                        "session_id": session_info["session_id"],
                        "action": "exported",
                        "path": str(export_path)
                    })
                else:
                    results["failed"] += 1
                    
        return results
        
    def get_storage_stats(self) -> Dict:
        """Get storage statistics for compaction planning."""
        cfg = get_config()
        tasks_dir = cfg.kilo_code_tasks_path
        
        total_sessions = 0
        total_messages = 0
        large_sessions = 0
        old_sessions = 0
        
        for task_dir in tasks_dir.iterdir():
            if not task_dir.is_dir():
                continue
                
            api_file = task_dir / "api_conversation_history.json"
            if api_file.exists():
                total_sessions += 1
                
                try:
                    with open(api_file, 'r') as f:
                        history = json.load(f)
                    message_count = len(history)
                    total_messages += message_count
                    
                    if message_count > 100:
                        large_sessions += 1
                        
                    mtime = datetime.fromtimestamp(api_file.stat().st_mtime)
                    age_days = (datetime.now() - mtime).days
                    if age_days > 7:
                        old_sessions += 1
                        
                except Exception:
                    continue
                    
        return {
            "total_sessions": total_sessions,
            "total_messages": total_messages,
            "avg_messages_per_session": total_messages / total_sessions if total_sessions > 0 else 0,
            "large_sessions": large_sessions,
            "old_sessions": old_sessions,
            "compaction_recommended": large_sessions > 10 or old_sessions > 20
        }