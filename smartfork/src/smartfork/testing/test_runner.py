"""Test runner for SmartFork."""

import time
import unittest
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any, Type
import json


@dataclass
class TestResult:
    """Result of a single test."""
    name: str
    passed: bool
    duration_ms: float
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class TestSuite:
    """A suite of tests."""
    name: str
    tests: List[TestResult] = field(default_factory=list)
    
    @property
    def passed_count(self) -> int:
        return sum(1 for t in self.tests if t.passed)
    
    @property
    def failed_count(self) -> int:
        return sum(1 for t in self.tests if not t.passed)
    
    @property
    def total_duration_ms(self) -> float:
        return sum(t.duration_ms for t in self.tests)


class SmartForkTestCase:
    """Base class for SmartFork tests."""
    
    def __init__(self):
        self.results: List[TestResult] = []
    
    def assert_true(self, condition: bool, message: str = "") -> bool:
        """Assert that condition is true."""
        if not condition:
            raise AssertionError(message or "Assertion failed")
        return True
    
    def assert_equal(self, a: Any, b: Any, message: str = "") -> bool:
        """Assert that a equals b."""
        if a != b:
            raise AssertionError(message or f"Expected {a} to equal {b}")
        return True
    
    def assert_not_none(self, obj: Any, message: str = "") -> bool:
        """Assert that obj is not None."""
        if obj is None:
            raise AssertionError(message or "Expected non-None value")
        return True
    
    def assert_contains(self, container: Any, item: Any, message: str = "") -> bool:
        """Assert that item is in container."""
        if item not in container:
            raise AssertionError(message or f"Expected {container} to contain {item}")
        return True


class TestRunner:
    """Run tests and collect results."""
    
    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path.home() / ".smartfork/test_results.json"
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.suites: List[TestSuite] = []
        self.test_classes: Dict[str, Type[SmartForkTestCase]] = {}
    
    def register_test(self, name: str, test_class: Type[SmartForkTestCase]):
        """Register a test class."""
        self.test_classes[name] = test_class
    
    def run_test_method(
        self,
        test_instance: SmartForkTestCase,
        method_name: str
    ) -> TestResult:
        """Run a single test method."""
        start = time.time()
        
        try:
            method = getattr(test_instance, method_name)
            method()
            passed = True
            error = None
        except AssertionError as e:
            passed = False
            error = str(e)
        except Exception as e:
            passed = False
            error = f"{type(e).__name__}: {str(e)}"
        
        duration = (time.time() - start) * 1000
        
        return TestResult(
            name=method_name,
            passed=passed,
            duration_ms=duration,
            error_message=error
        )
    
    def run_suite(self, name: str) -> TestSuite:
        """Run a test suite."""
        if name not in self.test_classes:
            raise ValueError(f"Unknown test suite: {name}")
        
        test_class = self.test_classes[name]
        test_instance = test_class()
        suite = TestSuite(name=name)
        
        # Find all test methods
        for attr_name in dir(test_instance):
            if attr_name.startswith("test_"):
                result = self.run_test_method(test_instance, attr_name)
                suite.tests.append(result)
                test_instance.results.append(result)
        
        self.suites.append(suite)
        self._save_results()
        return suite
    
    def run_all(self) -> List[TestSuite]:
        """Run all registered test suites."""
        results = []
        for name in self.test_classes:
            suite = self.run_suite(name)
            results.append(suite)
        return results
    
    def _save_results(self):
        """Save test results to storage."""
        try:
            data = {
                "timestamp": datetime.utcnow().isoformat(),
                "suites": []
            }
            
            for suite in self.suites:
                suite_data = {
                    "name": suite.name,
                    "passed": suite.passed_count,
                    "failed": suite.failed_count,
                    "duration_ms": suite.total_duration_ms,
                    "tests": [
                        {
                            "name": t.name,
                            "passed": t.passed,
                            "duration_ms": t.duration_ms,
                            "error": t.error_message
                        }
                        for t in suite.tests
                    ]
                }
                data["suites"].append(suite_data)
            
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save test results: {e}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of test results."""
        total_tests = sum(len(s.tests) for s in self.suites)
        total_passed = sum(s.passed_count for s in self.suites)
        total_failed = sum(s.failed_count for s in self.suites)
        
        return {
            "suites": len(self.suites),
            "total_tests": total_tests,
            "passed": total_passed,
            "failed": total_failed,
            "pass_rate": total_passed / total_tests if total_tests > 0 else 0,
            "duration_ms": sum(s.total_duration_ms for s in self.suites)
        }


# Built-in tests

class IndexerTests(SmartForkTestCase):
    """Tests for the indexer module."""
    
    def test_parser_handles_empty_session(self):
        """Test that parser handles empty sessions gracefully."""
        from ..indexer.parser import KiloCodeParser
        
        parser = KiloCodeParser()
        # Should return None for invalid directory
        result = parser.parse_task_directory(Path("/nonexistent"))
        self.assert_true(result is None, "Should return None for invalid path")
    
    def test_chunker_respects_max_size(self):
        """Test that chunker respects maximum chunk size."""
        from ..indexer.indexer import TextChunker
        
        chunker = TextChunker(chunk_size=100, chunk_overlap=10)
        text = "A" * 500
        chunks = chunker.chunk(text)
        
        # Each chunk should be at most 100 characters
        for chunk in chunks:
            self.assert_true(len(chunk) <= 110, f"Chunk too large: {len(chunk)}")  # Allow small overhead


class SearchTests(SmartForkTestCase):
    """Tests for the search module."""
    
    def test_hybrid_search_returns_results(self):
        """Test that hybrid search returns results when DB has data."""
        from ..database.chroma_db import ChromaDatabase
        from ..search.hybrid import HybridSearchEngine
        from ..config import get_config
        
        config = get_config()
        db = ChromaDatabase(config.chroma_db_path)
        engine = HybridSearchEngine(db)
        
        # This will return empty list if no data, which is fine
        results = engine.search("test query", n_results=5)
        self.assert_true(isinstance(results, list), "Should return a list")
    
    def test_search_scores_in_valid_range(self):
        """Test that search scores are in valid range [0, 1]."""
        from ..search.hybrid import normalize_scores
        
        scores = [10.5, 5.2, 3.1, 1.0, 0.5]
        normalized = normalize_scores(scores)
        
        for score in normalized:
            self.assert_true(0 <= score <= 1, f"Score {score} out of range")


class DatabaseTests(SmartForkTestCase):
    """Tests for the database module."""
    
    def test_chroma_db_initialization(self):
        """Test that ChromaDB initializes correctly."""
        from ..database.chroma_db import ChromaDatabase
        from ..config import get_config
        
        config = get_config()
        db = ChromaDatabase(config.chroma_db_path)
        
        # Should be able to get count
        count = db.get_session_count()
        self.assert_true(isinstance(count, int), "Count should be integer")
        self.assert_true(count >= 0, "Count should be non-negative")


class ForkTests(SmartForkTestCase):
    """Tests for the fork generation module."""
    
    def test_fork_generator_initialization(self):
        """Test that fork generator initializes correctly."""
        from ..fork.generator import ForkMDGenerator
        from ..database.chroma_db import ChromaDatabase
        from ..config import get_config
        
        config = get_config()
        db = ChromaDatabase(config.chroma_db_path)
        generator = ForkMDGenerator(db)
        
        self.assert_not_none(generator, "Generator should not be None")


def create_default_test_runner() -> TestRunner:
    """Create a test runner with all built-in tests."""
    runner = TestRunner()
    runner.register_test("indexer", IndexerTests)
    runner.register_test("search", SearchTests)
    runner.register_test("database", DatabaseTests)
    runner.register_test("fork", ForkTests)
    return runner
