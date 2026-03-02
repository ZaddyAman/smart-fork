"""Privacy vault and end-to-end encryption for sensitive sessions."""

import base64
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Set

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from loguru import logger

from ..config import get_config


class E2EEncryption:
    """End-to-end encryption for session data."""
    
    def __init__(self, password: Optional[str] = None):
        self.password = password or os.environ.get("SMARTFORK_VAULT_PASSWORD")
        self._key = None
        
    def _derive_key(self, salt: bytes) -> bytes:
        """Derive encryption key from password."""
        if not self.password:
            raise ValueError("No password provided for encryption")
            
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(self.password.encode()))
        return key
        
    def encrypt(self, data: str) -> str:
        """Encrypt data with password-derived key."""
        salt = os.urandom(16)
        key = self._derive_key(salt)
        f = Fernet(key)
        encrypted = f.encrypt(data.encode())
        
        # Prepend salt to encrypted data
        result = base64.urlsafe_b64encode(salt + encrypted).decode()
        return result
        
    def decrypt(self, encrypted_data: str) -> str:
        """Decrypt data with password-derived key."""
        try:
            decoded = base64.urlsafe_b64decode(encrypted_data.encode())
            salt = decoded[:16]
            encrypted = decoded[16:]
            
            key = self._derive_key(salt)
            f = Fernet(key)
            decrypted = f.decrypt(encrypted)
            return decrypted.decode()
        except Exception as e:
            raise ValueError(f"Decryption failed: {e}")
            
    def encrypt_file(self, file_path: Path, output_path: Optional[Path] = None) -> Path:
        """Encrypt a file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        encrypted = self.encrypt(content)
        
        output = output_path or file_path.with_suffix(file_path.suffix + ".vault")
        with open(output, 'w') as f:
            f.write(encrypted)
            
        return output
        
    def decrypt_file(self, file_path: Path, output_path: Optional[Path] = None) -> Path:
        """Decrypt a file."""
        with open(file_path, 'r') as f:
            encrypted = f.read()
            
        decrypted = self.decrypt(encrypted)
        
        output = output_path or file_path.with_suffix('')
        with open(output, 'w', encoding='utf-8') as f:
            f.write(decrypted)
            
        return output


class PrivacyVault:
    """Privacy vault for sensitive sessions."""
    
    def __init__(self, password: Optional[str] = None):
        self.encryption = E2EEncryption(password)
        cfg = get_config()
        self.vault_dir = cfg.cache_dir / "vault"
        self.vault_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = self.vault_dir / "vault_index.json"
        self.vault_index = self._load_index()
        
    def _load_index(self) -> Dict:
        """Load vault index."""
        if self.index_file.exists():
            try:
                with open(self.index_file, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        return {"vaulted_sessions": [], "metadata": {}}
        
    def _save_index(self):
        """Save vault index."""
        with open(self.index_file, 'w') as f:
            json.dump(self.vault_index, f, indent=2)
            
    def add_to_vault(self, session_id: str, task_dir: Path) -> bool:
        """Add a session to the privacy vault."""
        try:
            vault_path = self.vault_dir / f"{session_id}.vault"
            
            # Read and encrypt session files
            vault_data = {
                "session_id": session_id,
                "vaulted_at": str(Path.cwd()),
                "files": {}
            }
            
            for file_name in ["api_conversation_history.json", "ui_messages.json", "task_metadata.json"]:
                file_path = task_dir / file_name
                if file_path.exists():
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    vault_data["files"][file_name] = content
                    
            # Encrypt and save
            encrypted = self.encryption.encrypt(json.dumps(vault_data))
            with open(vault_path, 'w') as f:
                f.write(encrypted)
                
            # Update index
            self.vault_index["vaulted_sessions"].append(session_id)
            self.vault_index["metadata"][session_id] = {
                "vaulted_at": str(Path.cwd()),
                "original_path": str(task_dir),
                "file_count": len(vault_data["files"])
            }
            self._save_index()
            
            logger.info(f"Session {session_id} added to vault")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add {session_id} to vault: {e}")
            return False
            
    def restore_from_vault(self, session_id: str, output_dir: Optional[Path] = None) -> Optional[Path]:
        """Restore a session from the vault."""
        try:
            vault_path = self.vault_dir / f"{session_id}.vault"
            
            if not vault_path.exists():
                logger.error(f"Session {session_id} not found in vault")
                return None
                
            # Decrypt
            with open(vault_path, 'r') as f:
                encrypted = f.read()
            decrypted = self.encryption.decrypt(encrypted)
            vault_data = json.loads(decrypted)
            
            # Determine output directory
            if output_dir:
                target_dir = output_dir / session_id
            else:
                # Use original path if available
                meta = self.vault_index["metadata"].get(session_id, {})
                original = meta.get("original_path")
                target_dir = Path(original) if original else (self.vault_dir / "restored" / session_id)
                
            target_dir.mkdir(parents=True, exist_ok=True)
            
            # Restore files
            for file_name, content in vault_data["files"].items():
                file_path = target_dir / file_name
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                    
            logger.info(f"Session {session_id} restored to {target_dir}")
            return target_dir
            
        except Exception as e:
            logger.error(f"Failed to restore {session_id}: {e}")
            return None
            
    def list_vaulted_sessions(self) -> List[Dict]:
        """List all vaulted sessions."""
        sessions = []
        for session_id in self.vault_index["vaulted_sessions"]:
            meta = self.vault_index["metadata"].get(session_id, {})
            sessions.append({
                "session_id": session_id,
                "vaulted_at": meta.get("vaulted_at", "unknown"),
                "file_count": meta.get("file_count", 0)
            })
        return sessions
        
    def remove_from_vault(self, session_id: str) -> bool:
        """Remove a session from the vault."""
        try:
            vault_path = self.vault_dir / f"{session_id}.vault"
            
            if vault_path.exists():
                vault_path.unlink()
                
            if session_id in self.vault_index["vaulted_sessions"]:
                self.vault_index["vaulted_sessions"].remove(session_id)
                
            if session_id in self.vault_index["metadata"]:
                del self.vault_index["metadata"][session_id]
                
            self._save_index()
            logger.info(f"Session {session_id} removed from vault")
            return True
            
        except Exception as e:
            logger.error(f"Failed to remove {session_id}: {e}")
            return False
            
    def search_vault(self, query: str) -> List[Dict]:
        """Search within vaulted sessions (requires decryption)."""
        results = []
        
        for session_id in self.vault_index["vaulted_sessions"]:
            try:
                vault_path = self.vault_dir / f"{session_id}.vault"
                
                with open(vault_path, 'r') as f:
                    encrypted = f.read()
                decrypted = self.encryption.decrypt(encrypted)
                vault_data = json.loads(decrypted)
                
                # Search in file contents
                for file_name, content in vault_data["files"].items():
                    if query.lower() in content.lower():
                        results.append({
                            "session_id": session_id,
                            "file": file_name,
                            "preview": content[:200] + "..." if len(content) > 200 else content
                        })
                        break
                        
            except Exception:
                continue
                
        return results