"""Intelligence layer for SmartFork - Phase 2 features."""

from .pre_compaction import PreCompactionHook, CompactionManager
from .clustering import SemanticClustering, SessionClusterer
from .branching import BranchingTree, SessionBranch
from .privacy import PrivacyVault, E2EEncryption

__all__ = [
    "PreCompactionHook",
    "CompactionManager", 
    "SemanticClustering",
    "SessionClusterer",
    "BranchingTree",
    "SessionBranch",
    "PrivacyVault",
    "E2EEncryption",
]