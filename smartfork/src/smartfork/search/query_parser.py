"""Parse search queries to extract intent, files, and technologies."""

import re
from typing import List, Set, Dict
from dataclasses import dataclass


@dataclass
class QueryIntent:
    """
    Structured representation of search query.
    
    This dataclass captures the parsed intent and entities extracted from
    a natural language search query, enabling query-aware chunk retrieval.
    
    Attributes:
        original_query: The raw search query as entered by the user
        normalized_query: Normalized version (lowercase, cleaned spacing)
        file_references: List of file paths mentioned in the query
        technologies: List of technology keywords detected
        actions: List of action keywords (fix, debug, implement, etc.)
        intent_type: Classified intent category (debug, implement, explain, etc.)
        prefer_recent: Whether to prioritize recent chunks
        prefer_code: Whether to prioritize code blocks
    """
    original_query: str
    normalized_query: str
    file_references: List[str]
    technologies: List[str]
    actions: List[str]
    intent_type: str
    prefer_recent: bool = True
    prefer_code: bool = False


class QueryParser:
    """
    Parse natural language queries into structured intents.
    
    This parser extracts entities like file references, technologies, and actions
    from search queries to enable intelligent chunk ranking and retrieval.
    
    Example:
        >>> parser = QueryParser()
        >>> intent = parser.parse("How do I fix the bug in app.py?")
        >>> intent.file_references
        ['app.py']
        >>> intent.actions
        ['debug']
        >>> intent.intent_type
        'debug'
    """
    
    # File extension patterns for detecting file references
    FILE_PATTERNS: List[str] = [
        # Standard file paths with extensions
        r'([\w\-./]+\.(py|js|ts|jsx|tsx|java|go|rs|cpp|c|h|hpp|yaml|yml|json|md|txt|sql|html|css|scss|vue|php|rb|swift|kt|scala|r|m|mm|sh|bat|ps1|xml|toml|ini|cfg))',
        # Backtick file references: `filename.py`
        r'`([^`]+\.[\w]+)`',
    ]
    
    # Technology keywords for tech stack detection
    TECH_KEYWORDS: Set[str] = {
        # Frontend frameworks
        'react', 'vue', 'angular', 'svelte', 'nextjs', 'nuxt', 'remix', 'gatsby',
        # Backend frameworks
        'fastapi', 'flask', 'django', 'express', 'spring', 'nestjs', 'laravel', 'rails',
        # Databases
        'postgresql', 'mysql', 'mongodb', 'redis', 'elasticsearch', 'sqlite', 'dynamodb',
        'cassandra', 'neo4j', 'couchdb', 'firebase', 'supabase', 'prisma', 'sqlalchemy',
        # DevOps/Cloud
        'docker', 'kubernetes', 'aws', 'gcp', 'azure', 'terraform', 'ansible', 'jenkins',
        'github', 'gitlab', 'ci/cd', 'pipeline', 'github-actions',
        # Authentication/Security
        'jwt', 'oauth', 'oauth2', 'auth', 'authentication', 'authorization', 'sso',
        'ldap', 'saml', 'openid', 'cors', 'csrf', 'xss', 'encryption', 'hashing',
        # APIs
        'api', 'rest', 'graphql', 'grpc', 'websocket', 'soap', 'openapi', 'swagger',
        'fastapi', 'flask-restful', 'django-rest-framework',
        # Testing
        'test', 'pytest', 'jest', 'mocha', 'unittest', 'cypress', 'playwright', 'selenium',
        'integration-test', 'unit-test', 'e2e-test', 'mock', 'fixture',
        # Languages
        'python', 'javascript', 'typescript', 'java', 'golang', 'rust', 'cpp', 'c++',
        'ruby', 'php', 'swift', 'kotlin', 'scala', 'r', 'matlab',
        # Tools
        'git', 'npm', 'yarn', 'pip', 'poetry', 'conda', 'maven', 'gradle', 'webpack',
        'vite', 'rollup', 'babel', 'eslint', 'prettier', 'black', 'mypy',
        # Concepts
        'async', 'await', 'promise', 'callback', 'middleware', 'decorator', 'middleware',
        'orm', 'sql', 'nosql', 'microservice', 'serverless', 'lambda', 'function',
        'class', 'interface', 'type', 'enum', 'generic', 'inheritance', 'polymorphism',
    }
    
    # Action keywords mapped to intent categories
    ACTION_KEYWORDS: Dict[str, str] = {
        # Debug actions
        'fix': 'debug',
        'debug': 'debug',
        'solve': 'debug',
        'error': 'debug',
        'bug': 'debug',
        'issue': 'debug',
        'problem': 'debug',
        'broken': 'debug',
        'crash': 'debug',
        'exception': 'debug',
        'traceback': 'debug',
        'fail': 'debug',
        'failing': 'debug',
        # Implement actions
        'implement': 'implement',
        'create': 'implement',
        'build': 'implement',
        'add': 'implement',
        'write': 'implement',
        'generate': 'implement',
        'develop': 'implement',
        'setup': 'implement',
        'configure': 'implement',
        'install': 'implement',
        'integrate': 'implement',
        # Test actions
        'test': 'test',
        'unittest': 'test',
        'pytest': 'test',
        'jest': 'test',
        'validate': 'test',
        'verify': 'test',
        'check': 'test',
        # Explain actions
        'explain': 'explain',
        'how': 'explain',
        'what': 'explain',
        'why': 'explain',
        'describe': 'explain',
        'clarify': 'explain',
        'understand': 'explain',
        'meaning': 'explain',
        # Find actions
        'find': 'find_code',
        'locate': 'find_code',
        'where': 'find_code',
        'search': 'find_code',
        'lookup': 'find_code',
        'reference': 'find_code',
        # Refactor actions
        'refactor': 'refactor',
        'improve': 'refactor',
        'optimize': 'refactor',
        'clean': 'refactor',
        'restructure': 'refactor',
        'simplify': 'refactor',
        'performance': 'refactor',
        'efficient': 'refactor',
        # Update actions
        'update': 'update',
        'upgrade': 'update',
        'migrate': 'update',
        'change': 'update',
        'modify': 'update',
        'edit': 'update',
        'replace': 'update',
        'remove': 'update',
        'delete': 'update',
    }
    
    def parse(self, query: str) -> QueryIntent:
        """
        Parse a search query into structured intent.
        
        This method extracts files, technologies, actions, and classifies the
        overall intent from the search query.
        
        Args:
            query: The raw search query string
            
        Returns:
            QueryIntent with all extracted entities and classifications
            
        Example:
            >>> parser = QueryParser()
            >>> intent = parser.parse("Fix JWT auth in app.py")
            >>> intent.file_references
            ['app.py']
            >>> intent.technologies
            ['jwt', 'auth']
            >>> intent.intent_type
            'debug'
        """
        if not query or not isinstance(query, str):
            return QueryIntent(
                original_query=query or "",
                normalized_query="",
                file_references=[],
                technologies=[],
                actions=[],
                intent_type="general",
                prefer_recent=True,
                prefer_code=False
            )
        
        # Normalize the query
        normalized = self._normalize(query)
        
        # Extract entities
        files = self._extract_files(query)
        techs = self._extract_technologies(query)
        actions = self._extract_actions(query)
        
        # Classify intent
        intent_type = self._classify_intent(query, actions)
        
        # Determine preferences based on query content
        prefer_code = self._detect_code_preference(query)
        prefer_recent = self._detect_recency_preference(query)
        
        return QueryIntent(
            original_query=query,
            normalized_query=normalized,
            file_references=list(files),
            technologies=list(techs),
            actions=actions,
            intent_type=intent_type,
            prefer_recent=prefer_recent,
            prefer_code=prefer_code
        )
    
    def _normalize(self, text: str) -> str:
        """
        Normalize query text for consistent processing.
        
        Converts to lowercase, removes extra whitespace, and strips
        leading/trailing spaces.
        
        Args:
            text: Raw query text
            
        Returns:
            Normalized query string
        """
        # Lowercase and strip
        normalized = text.lower().strip()
        # Replace multiple spaces with single space
        normalized = re.sub(r'\s+', ' ', normalized)
        return normalized
    
    def _extract_files(self, text: str) -> Set[str]:
        """
        Extract file references from query text.
        
        Detects file paths with common extensions and backtick-wrapped
        file references.
        
        Args:
            text: Query text to analyze
            
        Returns:
            Set of detected file paths
            
        Example:
            >>> parser = QueryParser()
            >>> parser._extract_files("Check app.py and `utils.js`")
            {'app.py', 'utils.js'}
        """
        files: Set[str] = set()
        
        for pattern in self.FILE_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                # match might be a tuple if pattern has groups
                if isinstance(match, tuple):
                    # Take the first group (full match)
                    files.add(match[0])
                else:
                    files.add(match)
        
        return files
    
    def _extract_technologies(self, text: str) -> Set[str]:
        """
        Extract technology keywords from query text.
        
        Scans for known technology keywords like frameworks, databases,
        tools, and concepts.
        
        Args:
            text: Query text to analyze
            
        Returns:
            Set of detected technology keywords
            
        Example:
            >>> parser = QueryParser()
            >>> parser._extract_technologies("Using React with FastAPI")
            {'react', 'fastapi'}
        """
        query_lower = text.lower()
        found: Set[str] = set()
        
        for tech in self.TECH_KEYWORDS:
            # Use word boundary matching for more precise detection
            pattern = r'\b' + re.escape(tech) + r'\b'
            if re.search(pattern, query_lower):
                found.add(tech)
        
        return found
    
    def _extract_actions(self, text: str) -> List[str]:
        """
        Extract action keywords from query text.
        
        Identifies action verbs and maps them to intent categories.
        Preserves order of first occurrence and removes duplicates.
        
        Args:
            text: Query text to analyze
            
        Returns:
            List of detected action types in order of first occurrence
            
        Example:
            >>> parser = QueryParser()
            >>> parser._extract_actions("How to fix and test the bug")
            ['debug', 'test']
        """
        query_lower = text.lower()
        actions: List[str] = []
        seen: Set[str] = set()
        
        for keyword, action_type in self.ACTION_KEYWORDS.items():
            if keyword in query_lower and action_type not in seen:
                actions.append(action_type)
                seen.add(action_type)
        
        return actions
    
    def _classify_intent(self, query: str, actions: List[str]) -> str:
        """
        Classify the overall query intent.
        
        Uses extracted actions and query content to determine the
        primary intent type.
        
        Args:
            query: Original query string
            actions: List of detected actions
            
        Returns:
            Intent type classification string
        """
        # Primary action takes precedence
        if actions:
            return actions[0]
        
        query_lower = query.lower()
        
        # Check for question patterns
        if any(kw in query_lower for kw in ['how', 'what', 'why', 'explain', 'describe']):
            return 'explain'
        elif any(kw in query_lower for kw in ['where', 'find', 'locate', 'search']):
            return 'find_code'
        elif any(kw in query_lower for kw in ['error', 'exception', 'bug', 'issue', 'problem']):
            return 'debug'
        elif any(kw in query_lower for kw in ['create', 'build', 'make', 'generate']):
            return 'implement'
        else:
            return 'general'
    
    def _detect_code_preference(self, query: str) -> bool:
        """
        Detect if query prefers code blocks over text.
        
        Analyzes query for keywords indicating code-related intent.
        
        Args:
            query: Original query string
            
        Returns:
            True if code blocks should be preferred
        """
        query_lower = query.lower()
        code_keywords = [
            'code', 'function', 'class', 'method', 'implementation',
            'snippet', 'example', 'syntax', 'api', 'def ', 'const ',
            'import ', 'from ', 'return ', 'async ', 'await '
        ]
        return any(kw in query_lower for kw in code_keywords)
    
    def _detect_recency_preference(self, query: str) -> bool:
        """
        Detect if query prefers recent content.
        
        Analyzes query for time-related keywords indicating recency preference.
        
        Args:
            query: Original query string
            
        Returns:
            True if recent chunks should be preferred
        """
        query_lower = query.lower()
        recency_keywords = [
            'recent', 'last', 'latest', 'new', 'newest', 'current',
            'today', 'yesterday', 'this week', 'last week'
        ]
        return any(kw in query_lower for kw in recency_keywords)
