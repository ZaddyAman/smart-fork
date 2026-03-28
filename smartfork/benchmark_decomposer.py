"""Benchmark query decomposition latency."""
import time
import psutil
import os
import sys

# Change to smartfork directory for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Measure cold start
process = psutil.Process(os.getpid())
mem_before = process.memory_info().rss / 1024 / 1024

print("Loading decomposer...")
cold_start = time.time()
from smartfork.search.query_decomposer import QueryDecomposer
decomposer = QueryDecomposer(known_projects=["BharatLawAI", "SmartFork", "bharatlaw-frontend", "smartfork"])
# Trigger model loading during cold start
_decomposer = decomposer.decompose("test query")
cold_load_time = time.time() - cold_start
mem_after = process.memory_info().rss / 1024 / 1024

print(f"Cold start: {cold_load_time:.2f}s")
print(f"Memory used by models: {mem_after - mem_before:.0f} MB")

# Measure warm queries
test_queries = [
    "auth decisions in BharatLawAI",
    "that CORS bug I fixed last week", 
    "how did I structure FastAPI routes",
    "ChromaDB embedding setup",
    "what was I working on yesterday",
    "why did I choose JWT over sessions",
    "pagination logic",
    "fix the scrollbar issue",
    "kafka consumer error",
    "hybrid search implementation"
]

print("\nWarm query latencies:")
times = []
for query in test_queries:
    start = time.time()
    result = decomposer.decompose(query)
    elapsed = time.time() - start
    times.append(elapsed)
    print(f"  {elapsed:.3f}s | {query[:40]}")
    print(f"           -> intent={result.intent} project={result.project} topic={result.topic}")

print(f"\nAverage: {sum(times)/len(times):.3f}s")
print(f"Max:     {max(times):.3f}s")
print(f"Min:     {min(times):.3f}s")
