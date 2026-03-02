"""Metrics tracking for success measurement."""

import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any
import statistics


class MetricType(Enum):
    """Types of metrics that can be tracked."""
    COUNTER = "counter"
    TIMER = "timer"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"


@dataclass
class SuccessMetric:
    """A single success metric measurement."""
    name: str
    value: float
    metric_type: MetricType
    timestamp: datetime = field(default_factory=datetime.utcnow)
    tags: Dict[str, str] = field(default_factory=dict)
    context: Optional[str] = None


@dataclass  
class MetricSummary:
    """Summary statistics for a metric."""
    name: str
    count: int
    mean: float
    median: float
    std_dev: float
    min_val: float
    max_val: float
    p95: float
    p99: float


class MetricsTracker:
    """Track and analyze success metrics for SmartFork."""
    
    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path.home() / ".smartfork/metrics.json"
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.metrics: List[SuccessMetric] = []
        self.session_start = time.time()
        self._load_metrics()
    
    def _load_metrics(self):
        """Load existing metrics from storage."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path) as f:
                    data = json.load(f)
                    for item in data:
                        item['timestamp'] = datetime.fromisoformat(item['timestamp'])
                        item['metric_type'] = MetricType(item['metric_type'])
                        self.metrics.append(SuccessMetric(**item))
            except Exception as e:
                print(f"Warning: Could not load metrics: {e}")
    
    def _save_metrics(self):
        """Save metrics to storage."""
        try:
            data = []
            for m in self.metrics:
                item = asdict(m)
                item['timestamp'] = m.timestamp.isoformat()
                item['metric_type'] = m.metric_type.value
                data.append(item)
            
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save metrics: {e}")
    
    def record(
        self,
        name: str,
        value: float,
        metric_type: MetricType = MetricType.GAUGE,
        tags: Optional[Dict[str, str]] = None,
        context: Optional[str] = None
    ) -> SuccessMetric:
        """Record a metric measurement."""
        metric = SuccessMetric(
            name=name,
            value=value,
            metric_type=metric_type,
            tags=tags or {},
            context=context
        )
        self.metrics.append(metric)
        self._save_metrics()
        return metric
    
    def time_operation(self, name: str, tags: Optional[Dict[str, str]] = None):
        """Context manager for timing operations."""
        return _TimedOperation(self, name, tags)
    
    def increment(self, name: str, value: float = 1.0, tags: Optional[Dict[str, str]] = None):
        """Increment a counter metric."""
        return self.record(name, value, MetricType.COUNTER, tags)
    
    def gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Record a gauge metric."""
        return self.record(name, value, MetricType.GAUGE, tags)
    
    def get_summary(self, name: str, days: int = 30) -> Optional[MetricSummary]:
        """Get summary statistics for a metric."""
        cutoff = datetime.utcnow().timestamp() - (days * 86400)
        values = [
            m.value for m in self.metrics
            if m.name == name and m.timestamp.timestamp() > cutoff
        ]
        
        if not values:
            return None
        
        values.sort()
        n = len(values)
        
        return MetricSummary(
            name=name,
            count=n,
            mean=statistics.mean(values),
            median=statistics.median(values),
            std_dev=statistics.stdev(values) if n > 1 else 0.0,
            min_val=min(values),
            max_val=max(values),
            p95=values[int(n * 0.95)],
            p99=values[int(n * 0.99)] if n >= 100 else values[-1]
        )
    
    def get_metrics_by_tag(self, tag_key: str, tag_value: str) -> List[SuccessMetric]:
        """Get all metrics matching a tag."""
        return [m for m in self.metrics if m.tags.get(tag_key) == tag_value]
    
    def get_dashboard_data(self, days: int = 7) -> Dict[str, Any]:
        """Get data for the success metrics dashboard."""
        cutoff = datetime.utcnow().timestamp() - (days * 86400)
        recent = [m for m in self.metrics if m.timestamp.timestamp() > cutoff]
        
        # Group by metric name
        by_name: Dict[str, List[SuccessMetric]] = {}
        for m in recent:
            by_name.setdefault(m.name, []).append(m)
        
        # Calculate summaries
        summaries = {}
        for name, metrics in by_name.items():
            values = [m.value for m in metrics]
            summaries[name] = {
                "count": len(values),
                "mean": statistics.mean(values),
                "trend": self._calculate_trend(values)
            }
        
        # Calculate key metrics
        total_sessions = len(set(
            m.tags.get("session_id", "") for m in recent
        ))
        
        avg_fork_time = summaries.get("fork_generation_time", {}).get("mean", 0)
        
        return {
            "period_days": days,
            "total_metrics_recorded": len(recent),
            "unique_sessions": total_sessions,
            "metric_summaries": summaries,
            "key_metrics": {
                "avg_fork_generation_time_ms": avg_fork_time,
                "sessions_per_day": total_sessions / days,
                "total_context_recovered_mb": sum(
                    m.value for m in recent
                    if m.name == "context_recovered_bytes"
                ) / (1024 * 1024)
            }
        }
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction from values."""
        if len(values) < 10:
            return "insufficient_data"
        
        # Split into halves
        mid = len(values) // 2
        first_half = statistics.mean(values[:mid])
        second_half = statistics.mean(values[mid:])
        
        diff_pct = ((second_half - first_half) / first_half * 100) if first_half != 0 else 0
        
        if diff_pct > 10:
            return "improving"  # Lower is better for most metrics
        elif diff_pct < -10:
            return "degrading"
        else:
            return "stable"
    
    def compare_periods(
        self,
        metric_name: str,
        period1_days: int = 7,
        period2_days: int = 7,
        period2_offset_days: int = 7
    ) -> Dict[str, Any]:
        """Compare metric between two time periods."""
        now = datetime.utcnow().timestamp()
        
        # Period 1: Most recent
        p1_start = now - (period1_days * 86400)
        p1_values = [
            m.value for m in self.metrics
            if m.name == metric_name and p1_start <= m.timestamp.timestamp() <= now
        ]
        
        # Period 2: Earlier period
        p2_end = now - (period2_offset_days * 86400)
        p2_start = p2_end - (period2_days * 86400)
        p2_values = [
            m.value for m in self.metrics
            if m.name == metric_name and p2_start <= m.timestamp.timestamp() <= p2_end
        ]
        
        if not p1_values or not p2_values:
            return {"error": "Insufficient data for comparison"}
        
        p1_mean = statistics.mean(p1_values)
        p2_mean = statistics.mean(p2_values)
        
        change_pct = ((p1_mean - p2_mean) / p2_mean * 100) if p2_mean != 0 else 0
        
        return {
            "metric": metric_name,
            "period1": {
                "days": period1_days,
                "count": len(p1_values),
                "mean": p1_mean
            },
            "period2": {
                "days": period2_days,
                "offset": period2_offset_days,
                "count": len(p2_values),
                "mean": p2_mean
            },
            "change_percent": change_pct,
            "change_direction": "increased" if change_pct > 0 else "decreased"
        }


class _TimedOperation:
    """Context manager for timing operations."""
    
    def __init__(self, tracker: MetricsTracker, name: str, tags: Optional[Dict[str, str]]):
        self.tracker = tracker
        self.name = name
        self.tags = tags or {}
        self.start_time: Optional[float] = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration_ms = (time.time() - self.start_time) * 1000
            self.tracker.record(
                self.name,
                duration_ms,
                MetricType.TIMER,
                self.tags
            )
