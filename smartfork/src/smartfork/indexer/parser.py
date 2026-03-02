"""Parser for Kilo Code transcript files."""

import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from loguru import logger

from ..database.models import TaskSession, TaskMetadata, ConversationMessage, UIMessage


class KiloCodeParser:
    """Parses Kilo Code transcript files from task directories."""
    
    # Technology detection patterns
    TECH_PATTERNS = {
        'FastAPI': [r'fastapi', r'from fastapi', r'import fastapi'],
        'Django': [r'django', r'from django', r'import django'],
        'Flask': [r'flask', r'from flask', r'import flask'],
        'React': [r'react', r'import.*react', r'from.*react'],
        'Vue': [r'vue', r'import.*vue', r'from.*vue'],
        'Angular': [r'angular', r'@angular'],
        'PostgreSQL': [r'postgres', r'postgresql', r'psycopg', r'asyncpg'],
        'MySQL': [r'mysql', r'pymysql', r'mysql-connector'],
        'MongoDB': [r'mongodb', r'pymongo', r'mongoose'],
        'Redis': [r'redis', r'aioredis'],
        'Docker': [r'docker', r'dockerfile', r'docker-compose'],
        'Kubernetes': [r'kubernetes', r'k8s', r'kubectl'],
        'JWT': [r'jwt', r'json.?web.?token', r'pyjwt', r'jose'],
        'OAuth': [r'oauth', r'oauth2'],
        'GraphQL': [r'graphql', r'graphene', r'apollo'],
        'REST': [r'rest.?api', r'restful'],
        'gRPC': [r'grpc'],
        'WebSocket': [r'websocket', r'websockets', r'socket\.io'],
        'Celery': [r'celery'],
        'RabbitMQ': [r'rabbitmq', r'pika'],
        'Kafka': [r'kafka', r'confluent-kafka'],
        'Elasticsearch': [r'elasticsearch', r'elasticsearch-dsl'],
        'Pandas': [r'pandas', r'pd\.'],
        'NumPy': [r'numpy', r'np\.'],
        'TensorFlow': [r'tensorflow', r'tf\.'],
        'PyTorch': [r'torch', r'pytorch'],
        'Scikit-learn': [r'sklearn', r'scikit-learn'],
        'OpenAI': [r'openai', r'chatgpt', r'gpt-'],
        'Anthropic': [r'anthropic', r'claude'],
        'LangChain': [r'langchain'],
        'HuggingFace': [r'huggingface', r'transformers'],
    }
    
    def parse_task_directory(self, task_path: Path) -> Optional[TaskSession]:
        """Parse a Kilo Code task directory.
        
        Args:
            task_path: Path to the task directory
            
        Returns:
            TaskSession object or None if parsing fails
        """
        if not task_path.is_dir():
            return None
        
        task_id = task_path.name
        
        try:
            # Load metadata
            metadata_path = task_path / "task_metadata.json"
            metadata = self._parse_metadata(metadata_path) if metadata_path.exists() else TaskMetadata()
            
            # Load conversation history
            history_path = task_path / "api_conversation_history.json"
            conversation = self._parse_conversation(history_path) if history_path.exists() else []
            
            # Load UI messages
            ui_path = task_path / "ui_messages.json"
            ui_messages = self._parse_ui_messages(ui_path) if ui_path.exists() else []
            
            return TaskSession(
                task_id=task_id,
                metadata=metadata,
                conversation=conversation,
                ui_messages=ui_messages
            )
        except Exception as e:
            logger.error(f"Failed to parse task directory {task_id}: {e}")
            return None
    
    def _parse_metadata(self, path: Path) -> TaskMetadata:
        """Parse task_metadata.json."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        files_in_context = []
        if 'files_in_context' in data:
            for file_record in data['files_in_context']:
                if file_record.get('record_state') == 'active':
                    files_in_context.append(file_record['path'])
        
        return TaskMetadata(
            files_in_context=files_in_context,
            raw_data=data
        )
    
    def _parse_conversation(self, path: Path) -> List[ConversationMessage]:
        """Parse api_conversation_history.json."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        messages = []
        for msg in data:
            try:
                messages.append(ConversationMessage(
                    role=msg.get('role', 'unknown'),
                    content=self._extract_content(msg),
                    timestamp=msg.get('ts'),
                    type=msg.get('type')
                ))
            except Exception as e:
                logger.warning(f"Failed to parse message: {e}")
                continue
        
        return messages
    
    def _parse_ui_messages(self, path: Path) -> List[UIMessage]:
        """Parse ui_messages.json."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        messages = []
        for msg in data:
            try:
                messages.append(UIMessage(
                    message_type=msg.get('type'),
                    say=msg.get('say'),
                    ask=msg.get('ask'),
                    text=msg.get('text'),
                    ts=msg.get('ts')
                ))
            except Exception as e:
                logger.warning(f"Failed to parse UI message: {e}")
                continue
        
        return messages
    
    def _extract_content(self, msg: Dict[str, Any]) -> str:
        """Extract text content from a message."""
        if 'content' in msg:
            content = msg['content']
            if isinstance(content, str):
                return content
            elif isinstance(content, list):
                # Handle array of content parts (OpenAI format)
                texts = []
                for part in content:
                    if isinstance(part, dict):
                        if 'text' in part:
                            texts.append(part['text'])
                        elif part.get('type') == 'text' and 'text' in part:
                            texts.append(part['text']['value'] if isinstance(part['text'], dict) else part['text'])
                return ' '.join(texts)
        
        # Try to extract from other fields
        if 'text' in msg:
            return msg['text']
        
        return ''
    
    def detect_technologies(self, text: str) -> List[str]:
        """Detect technologies mentioned in the text.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of detected technology names
        """
        text_lower = text.lower()
        detected = []
        
        for tech, patterns in self.TECH_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    detected.append(tech)
                    break
        
        return detected
    
    def extract_file_paths(self, text: str) -> List[str]:
        """Extract file paths mentioned in the text.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of file paths
        """
        # Pattern for common file paths
        pattern = r'[\w\-./]+\.(py|js|ts|jsx|tsx|java|go|rs|cpp|c|h|yaml|yml|json|md|txt|sql|html|css|scss|vue|php|rb|swift|kt|scala|r|m|mm)'
        matches = re.findall(pattern, text)
        
        return list(set(matches))
