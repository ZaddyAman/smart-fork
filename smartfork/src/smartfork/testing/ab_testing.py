"""A/B testing framework for SmartFork."""

import json
import random
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any
import statistics


class AlgorithmVariant(Enum):
    """Algorithm variants for A/B testing."""
    CONTROL = "control"
    TREATMENT = "treatment"


@dataclass
class ABTestResult:
    """Result of an A/B test comparison."""
    test_name: str
    control_mean: float
    treatment_mean: float
    improvement_pct: float
    p_value: float
    significant: bool
    control_samples: int
    treatment_samples: int
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TestSession:
    """A single test session measurement."""
    session_id: str
    test_name: str
    variant: AlgorithmVariant
    query: str
    results_shown: int
    result_selected: Optional[int] = None
    time_to_select_ms: Optional[float] = None
    satisfaction_score: Optional[int] = None  # 1-5
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metrics: Dict[str, float] = field(default_factory=dict)


class ABTestManager:
    """Manage A/B tests for SmartFork algorithms."""
    
    # Active tests configuration
    TESTS = {
        "search_ranking": {
            "description": "Test new hybrid search ranking algorithm",
            "control_weights": {"semantic": 0.5, "bm25": 0.25, "recency": 0.15, "path": 0.1},
            "treatment_weights": {"semantic": 0.6, "bm25": 0.2, "recency": 0.15, "path": 0.05},
        },
        "chunk_size": {
            "description": "Test different chunk sizes",
            "control_config": {"chunk_size": 500, "chunk_overlap": 50},
            "treatment_config": {"chunk_size": 800, "chunk_overlap": 100},
        },
        "fork_generation": {
            "description": "Test improved fork.md generation",
            "control_version": "v1",
            "treatment_version": "v2",
        }
    }
    
    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path.home() / ".smartfork/ab_tests.json"
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.sessions: List[TestSession] = []
        self.user_assignments: Dict[str, AlgorithmVariant] = {}
        self._load_data()
    
    def _load_data(self):
        """Load test data from storage."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path) as f:
                    data = json.load(f)
                    
                for item in data.get("sessions", []):
                    item['timestamp'] = datetime.fromisoformat(item['timestamp'])
                    item['variant'] = AlgorithmVariant(item['variant'])
                    self.sessions.append(TestSession(**item))
                    
                self.user_assignments = {
                    k: AlgorithmVariant(v) 
                    for k, v in data.get("assignments", {}).items()
                }
            except Exception as e:
                print(f"Warning: Could not load A/B test data: {e}")
    
    def _save_data(self):
        """Save test data to storage."""
        try:
            data = {
                "sessions": [],
                "assignments": {k: v.value for k, v in self.user_assignments.items()}
            }
            
            for session in self.sessions:
                item = asdict(session)
                item['timestamp'] = session.timestamp.isoformat()
                item['variant'] = session.variant.value
                data["sessions"].append(item)
            
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save A/B test data: {e}")
    
    def get_variant(self, user_id: str, test_name: str) -> AlgorithmVariant:
        """Get or assign variant for a user."""
        key = f"{user_id}:{test_name}"
        
        if key not in self.user_assignments:
            # Random assignment with 50/50 split
            self.user_assignments[key] = random.choice([
                AlgorithmVariant.CONTROL,
                AlgorithmVariant.TREATMENT
            ])
            self._save_data()
        
        return self.user_assignments[key]
    
    def record_session(
        self,
        session_id: str,
        test_name: str,
        variant: AlgorithmVariant,
        query: str,
        results_shown: int,
        result_selected: Optional[int] = None,
        time_to_select_ms: Optional[float] = None,
        satisfaction_score: Optional[int] = None,
        metrics: Optional[Dict[str, float]] = None
    ):
        """Record a test session."""
        session = TestSession(
            session_id=session_id,
            test_name=test_name,
            variant=variant,
            query=query,
            results_shown=results_shown,
            result_selected=result_selected,
            time_to_select_ms=time_to_select_ms,
            satisfaction_score=satisfaction_score,
            metrics=metrics or {}
        )
        
        self.sessions.append(session)
        self._save_data()
        return session
    
    def analyze_test(self, test_name: str, min_samples: int = 30) -> Optional[ABTestResult]:
        """Analyze results of an A/B test."""
        test_sessions = [s for s in self.sessions if s.test_name == test_name]
        
        if len(test_sessions) < min_samples:
            return None
        
        control = [s for s in test_sessions if s.variant == AlgorithmVariant.CONTROL]
        treatment = [s for s in test_sessions if s.variant == AlgorithmVariant.TREATMENT]
        
        if len(control) < min_samples // 2 or len(treatment) < min_samples // 2:
            return None
        
        # Calculate click-through rate
        control_ctr = sum(1 for s in control if s.result_selected is not None) / len(control)
        treatment_ctr = sum(1 for s in treatment if s.result_selected is not None) / len(treatment)
        
        # Calculate mean time to select
        control_times = [s.time_to_select_ms for s in control if s.time_to_select_ms]
        treatment_times = [s.time_to_select_ms for s in treatment if s.time_to_select_ms]
        
        control_mean_time = statistics.mean(control_times) if control_times else 0
        treatment_mean_time = statistics.mean(treatment_times) if treatment_times else 0
        
        # Calculate satisfaction
        control_sat = [s.satisfaction_score for s in control if s.satisfaction_score]
        treatment_sat = [s.satisfaction_score for s in treatment if s.satisfaction_score]
        
        control_mean_sat = statistics.mean(control_sat) if control_sat else 0
        treatment_mean_sat = statistics.mean(treatment_sat) if treatment_sat else 0
        
        # Simple statistical test (would use proper t-test in production)
        improvement_pct = ((treatment_ctr - control_ctr) / control_ctr * 100) if control_ctr > 0 else 0
        
        # Determine significance (simplified - would use proper p-value calculation)
        significant = abs(improvement_pct) > 10 and len(test_sessions) >= 100
        
        return ABTestResult(
            test_name=test_name,
            control_mean=control_ctr,
            treatment_mean=treatment_ctr,
            improvement_pct=improvement_pct,
            p_value=0.05 if significant else 0.5,  # Placeholder
            significant=significant,
            control_samples=len(control),
            treatment_samples=len(treatment),
            metadata={
                "control_mean_time_ms": control_mean_time,
                "treatment_mean_time_ms": treatment_mean_time,
                "control_mean_satisfaction": control_mean_sat,
                "treatment_mean_satisfaction": treatment_mean_sat
            }
        )
    
    def get_active_tests(self) -> List[Dict]:
        """Get list of active A/B tests."""
        tests = []
        for name, config in self.TESTS.items():
            result = self.analyze_test(name, min_samples=1)
            tests.append({
                "name": name,
                "description": config["description"],
                "total_sessions": len([s for s in self.sessions if s.test_name == name]),
                "has_result": result is not None
            })
        return tests
    
    def get_test_summary(self) -> Dict[str, Any]:
        """Get summary of all A/B tests."""
        summary = {
            "total_tests": len(self.TESTS),
            "total_sessions": len(self.sessions),
            "active_tests": []
        }
        
        for test_name in self.TESTS:
            test_sessions = [s for s in self.sessions if s.test_name == test_name]
            control = len([s for s in test_sessions if s.variant == AlgorithmVariant.CONTROL])
            treatment = len([s for s in test_sessions if s.variant == AlgorithmVariant.TREATMENT])
            
            result = self.analyze_test(test_name)
            
            summary["active_tests"].append({
                "name": test_name,
                "total_sessions": len(test_sessions),
                "control": control,
                "treatment": treatment,
                "result": {
                    "improvement_pct": result.improvement_pct if result else None,
                    "significant": result.significant if result else False
                } if result else None
            })
        
        return summary


class ExperimentRunner:
    """Run controlled experiments."""
    
    def __init__(self, ab_manager: ABTestManager):
        self.ab_manager = ab_manager
    
    def run_experiment(
        self,
        name: str,
        control_func: Callable,
        treatment_func: Callable,
        test_data: List[Any],
        metric_func: Callable[[Any], float]
    ) -> Dict[str, Any]:
        """Run a controlled experiment comparing two functions."""
        control_results = []
        treatment_results = []
        
        for item in test_data:
            # Run control
            start = time.time()
            control_output = control_func(item)
            control_time = (time.time() - start) * 1000
            control_metric = metric_func(control_output)
            control_results.append({"metric": control_metric, "time_ms": control_time})
            
            # Run treatment
            start = time.time()
            treatment_output = treatment_func(item)
            treatment_time = (time.time() - start) * 1000
            treatment_metric = metric_func(treatment_output)
            treatment_results.append({"metric": treatment_metric, "time_ms": treatment_time})
        
        # Calculate statistics
        control_metrics = [r["metric"] for r in control_results]
        treatment_metrics = [r["metric"] for r in treatment_results]
        
        return {
            "name": name,
            "samples": len(test_data),
            "control": {
                "mean": statistics.mean(control_metrics),
                "stdev": statistics.stdev(control_metrics) if len(control_metrics) > 1 else 0,
                "mean_time_ms": statistics.mean([r["time_ms"] for r in control_results])
            },
            "treatment": {
                "mean": statistics.mean(treatment_metrics),
                "stdev": statistics.stdev(treatment_metrics) if len(treatment_metrics) > 1 else 0,
                "mean_time_ms": statistics.mean([r["time_ms"] for r in treatment_results])
            },
            "improvement_pct": (
                (statistics.mean(treatment_metrics) - statistics.mean(control_metrics))
                / statistics.mean(control_metrics) * 100
                if statistics.mean(control_metrics) != 0 else 0
            )
        }
