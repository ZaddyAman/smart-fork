"""Syntax highlighting for code snippets using Pygments.

Provides terminal-friendly syntax highlighting with theme-aware color mapping.
Falls back gracefully on Windows terminals without truecolor support.
"""

from typing import Optional, Dict
from pygments import lex
from pygments.lexers import get_lexer_by_name, get_lexer_for_filename, TextLexer
from pygments.token import Token
from pygments.util import ClassNotFound
from rich.text import Text
from rich.style import Style

from .progress import DEFAULT_THEME, get_theme_colors


def _build_token_styles(theme_name: str = DEFAULT_THEME) -> Dict:
    """Build Pygments token → Rich style mapping from a theme.
    
    Maps token types to theme colors so syntax highlighting adapts
    to the user's chosen theme instead of using a hardcoded palette.
    """
    theme = get_theme_colors(theme_name)
    bars = theme["bars"]
    sem = theme.get("semantic", {})
    
    return {
        # Keywords → bar[0] color (prominent, distinctive)
        Token.Keyword:          f"bold {bars[0]['color']}",
        Token.Keyword.Constant: f"bold {bars[0]['color']}",
        Token.Keyword.Declaration: f"bold {bars[0]['color']}",
        Token.Keyword.Namespace:   f"bold {bars[0]['color']}",
        Token.Keyword.Type:        f"bold {bars[2]['color']}",
        
        # Names → bar colors
        Token.Name.Builtin:        bars[1]['color'],
        Token.Name.Class:          f"bold {bars[2]['color']}",
        Token.Name.Function:       bars[1]['color'],
        Token.Name.Variable:       theme['text_primary'],
        Token.Name.Variable.Global: theme['text_primary'],
        Token.Name.Constant:       sem.get('warning', bars[2]['color']),
        Token.Name.Decorator:      bars[0]['color'],
        Token.Name.Exception:      f"bold {sem.get('error', theme['text_primary'])}",
        Token.Name.Attribute:      bars[2]['color'],
        Token.Name.Tag:            sem.get('error', theme['text_primary']),
        
        # Strings → success/bar[1] color
        Token.String:              sem.get('success', bars[1]['color']),
        Token.String.Doc:          f"italic {theme['text_muted']}",
        Token.String.Escape:       f"bold {bars[2]['color']}",
        Token.String.Interpol:     bars[2]['color'],
        Token.String.Regex:        bars[2]['color'],
        
        # Numbers → warning color
        Token.Number:              sem.get('warning', bars[2]['color']),
        
        # Operators → primary text
        Token.Operator:            theme['text_primary'],
        Token.Operator.Word:       f"bold {bars[0]['color']}",
        
        # Comments → muted/dim
        Token.Comment:             f"italic {theme['text_muted']}",
        Token.Comment.Single:      f"italic {theme['text_muted']}",
        Token.Comment.Multiline:   f"italic {theme['text_muted']}",
        Token.Comment.Preproc:     bars[0]['color'],
        Token.Comment.PreprocFile: theme['text_muted'],
        
        # Punctuation → primary
        Token.Punctuation:         theme['text_primary'],
        Token.Error:               sem.get('error', theme['text_primary']),
        
        # Generic
        Token.Generic.Heading:     f"bold {bars[1]['color']}",
        Token.Generic.Subheading:  f"bold {bars[1]['color']}",
        Token.Generic.Deleted:     sem.get('error', theme['text_primary']),
        Token.Generic.Inserted:    sem.get('success', bars[1]['color']),
        Token.Generic.Output:      theme['text_muted'],
        Token.Generic.Prompt:      f"bold {theme['text_muted']}",
        Token.Literal.Date:        sem.get('warning', bars[2]['color']),
    }


# Cache of token styles per theme
_token_style_cache: Dict[str, Dict] = {}


def _get_token_styles(theme_name: str = DEFAULT_THEME) -> Dict:
    """Get cached token styles for a theme."""
    if theme_name not in _token_style_cache:
        _token_style_cache[theme_name] = _build_token_styles(theme_name)
    return _token_style_cache[theme_name]


def clear_token_style_cache():
    """Clear the token style cache (call after theme change)."""
    _token_style_cache.clear()

# File extension → language name mapping for common cases
_EXTENSION_MAP = {
    ".py": "python",
    ".pyi": "python",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".js": "javascript",
    ".jsx": "javascript",
    ".rs": "rust",
    ".go": "go",
    ".rb": "ruby",
    ".java": "java",
    ".c": "c",
    ".cpp": "cpp",
    ".h": "c",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".sh": "bash",
    ".bash": "bash",
    ".zsh": "bash",
    ".fish": "fish",
    ".ps1": "powershell",
    ".sql": "sql",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".xml": "xml",
    ".html": "html",
    ".css": "css",
    ".scss": "scss",
    ".md": "markdown",
    ".diff": "diff",
    ".dockerfile": "docker",
    ".ini": "ini",
    ".cfg": "ini",
    ".env": "bash",
    ".lua": "lua",
    ".r": "r",
    ".R": "r",
    ".swift": "swift",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".scala": "scala",
    ".php": "php",
    ".dart": "dart",
    ".ex": "elixir",
    ".exs": "elixir",
    ".erl": "erlang",
    ".hs": "haskell",
    ".ml": "ocaml",
    ".zig": "zig",
    ".tf": "hcl",
    ".proto": "protobuf",
    ".graphql": "graphql",
    ".vim": "vim",
    ".bat": "batch",
    ".cmd": "batch",
    ".log": "text",
    ".txt": "text",
}

# Common code fence aliases → canonical lexer names
_FENCE_ALIASES = {
    "js": "javascript",
    "ts": "typescript",
    "py": "python",
    "rs": "rust",
    "rb": "ruby",
    "sh": "bash",
    "shell": "bash",
    "console": "bash",
    "terminal": "bash",
    "yml": "yaml",
    "md": "markdown",
    "bash": "bash",
    "zsh": "bash",
    "text": "text",
    "plain": "text",
    "plaintext": "text",
    "docker": "docker",
    "dockerfile": "docker",
    "csharp": "csharp",
    "cs": "csharp",
    "cpp": "cpp",
    "c++": "cpp",
    "hcl": "hcl",
    "terraform": "hcl",
    "protobuf": "protobuf",
}


def detect_language(filename: str) -> Optional[str]:
    """Detect language from filename extension.
    
    Args:
        filename: Filename or file path
        
    Returns:
        Language name suitable for get_lexer_by_name, or None
    """
    if not filename:
        return None
    
    # Get extension (lowercase)
    dot_idx = filename.rfind(".")
    if dot_idx == -1:
        # Check for known filenames without extension
        base = filename.lower()
        if base == "dockerfile":
            return "docker"
        if base == "makefile":
            return "make"
        return None
    
    ext = filename[dot_idx:].lower()
    return _EXTENSION_MAP.get(ext)


def _get_lexer(language: Optional[str]) -> object:
    """Get a Pygments lexer for the given language.
    
    Falls back to TextLexer (no highlighting) if language is unknown.
    """
    if not language:
        return TextLexer()
    
    # Resolve aliases
    lang = _FENCE_ALIASES.get(language.lower(), language)
    
    try:
        return get_lexer_by_name(lang, stripall=True)
    except ClassNotFound:
        return TextLexer()


def highlight_code(code: str, language: Optional[str] = None,
                   theme_name: str = DEFAULT_THEME) -> Text:
    """Highlight code string and return a Rich Text object.
    
    Args:
        code: Source code to highlight
        language: Language name (e.g., "python", "typescript"). 
                  Auto-detected if None and a filename hint is available.
        theme_name: Theme name for color mapping.
    
    Returns:
        Rich Text object with syntax highlighting applied
    """
    lexer = _get_lexer(language)
    
    if isinstance(lexer, TextLexer):
        return Text(code)
    
    result = Text()
    token_styles = _get_token_styles(theme_name)
    
    try:
        tokens = list(lex(code, lexer))
    except Exception:
        return Text(code)
    
    for token_type, value in tokens:
        style = token_styles.get(token_type)
        if style:
            result.append(value, style=style)
        else:
            parent = token_type.parent
            while parent and parent not in token_styles:
                parent = parent.parent
            if parent and parent in token_styles:
                result.append(value, style=token_styles[parent])
            else:
                result.append(value)
    
    return result


def highlight_snippet(text: str, max_len: int = 200) -> Text:
    """Highlight a text snippet, auto-detecting if it contains code.
    
    Heuristics: if the text looks like code (has keywords, operators, etc.),
    apply syntax highlighting. Otherwise return as plain styled text.
    
    Args:
        text: Text snippet (could be code or prose)
        max_len: Maximum length before truncation
        
    Returns:
        Rich Text object
    """
    if not text:
        return Text("")
    
    # Truncate if needed
    if len(text) > max_len:
        text = text[:max_len - 3] + "..."
    
    # Quick heuristic: does this look like code?
    code_indicators = [
        "def ", "fn ", "func ", "class ", "struct ", "impl ",
        "const ", "let ", "var ", "import ", "from ", "require",
        "async ", "await ", "return ", "if ", "else ", "for ",
        "while ", "match ", "switch ", "case ",
        "=>", "->", "::", "<<", ">>",
        "{", "}", "(", ")", "[", "]",
    ]
    
    # Check for code indicators
    code_score = sum(1 for ind in code_indicators if ind in text)
    
    # Also check for markdown code fences
    has_fences = "```" in text
    
    if has_fences:
        # Extract and highlight code blocks within markdown
        return _highlight_markdown_snippet(text)
    elif code_score >= 2:
        # Looks like code — try Python first (most common), then fall back
        highlighted = highlight_code(text, "python")
        # If highlighting didn't add any style, return as plain text
        if not any(s for _, s, _ in highlighted._spans):
            return Text(text, style="italic dim")
        return highlighted
    else:
        # Prose — return as italic dim (quote style)
        return Text(text, style="italic dim")


def _highlight_markdown_snippet(text: str) -> Text:
    """Highlight a text that may contain markdown code fences.
    
    Splits on ``` fences and highlights code blocks individually.
    """
    result = Text()
    parts = text.split("```")
    
    for i, part in enumerate(parts):
        if i % 2 == 0:
            # Outside code fence — plain text
            if part.strip():
                result.append(part, style="italic dim")
        else:
            # Inside code fence — extract language and highlight
            lines = part.split("\n", 1)
            lang_hint = lines[0].strip() if lines else ""
            code = lines[1] if len(lines) > 1 else lang_hint
            
            # Resolve language
            lang = _FENCE_ALIASES.get(lang_hint, lang_hint) if lang_hint else None
            if not lang:
                lang = detect_language(lang_hint) if lang_hint else None
            
            highlighted = highlight_code(code, lang)
            result.append_text(highlighted)
    
    return result


def detect_language_from_content(text: str) -> Optional[str]:
    """Attempt to detect language from code content using heuristics.
    
    Used when no filename or language hint is available.
    
    Args:
        text: Code content
        
    Returns:
        Detected language name or None
    """
    text_stripped = text.lstrip()
    
    # Python indicators
    if text_stripped.startswith(("def ", "class ", "import ", "from ")) or \
       ("self." in text and ": " in text) or \
       ("elif " in text or "print(" in text):
        return "python"
    
    # TypeScript/JavaScript indicators
    if ("const " in text and " => " in text) or \
       ("import " in text and "from " in text and "}" in text) or \
       ("interface " in text and "{" in text) or \
       ("async " in text and "await " in text):
        if "interface " in text or ": " in text and "=>" in text:
            return "typescript"
        return "javascript"
    
    # Rust indicators
    if ("fn " in text and " -> " in text) or \
       ("impl " in text and "{" in text) or \
       ("let mut " in text) or \
       ("pub " in text and "fn " in text):
        return "rust"
    
    # Go indicators
    if ("func " in text and "(" in text) or \
       ("package " in text_stripped[:50]) or \
       ("go " in text and "chan " in text):
        return "go"
    
    # SQL indicators
    if text_stripped.upper().startswith(("SELECT ", "INSERT ", "UPDATE ", "DELETE ", "CREATE ", "ALTER ")) or \
       ("FROM " in text.upper() and "WHERE " in text.upper()):
        return "sql"
    
    # JSON indicators
    if text_stripped.startswith("{") and '": ' in text:
        return "json"
    
    # YAML indicators
    if all(line.startswith("  ") or line.startswith("- ") or ":" in line 
           for line in text_stripped.split("\n")[:5] if line.strip()):
        if ": " in text and not text_stripped.startswith("{"):
            return "yaml"
    
    # Bash indicators
    if text_stripped.startswith("#!/") or \
       ("$ " in text and ("echo " in text or "cd " in text or "ls " in text)):
        return "bash"
    
    return None
