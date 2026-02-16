"""
HCP Merge Assistant - Intelligent Bulk Merge Solution for Reltio MDM

This solution enables Data Stewards to:
1. Upload files with HCP records to be analyzed/merged
2. Run intelligent match analysis against Reltio data
3. Review match results with AI-powered recommendations
4. Execute bulk merges with confirmation

Optimized for speed and efficiency with:
- Async concurrent API calls
- Connection pooling
- Batch operations
- Progress streaming
"""

__version__ = "1.0.0"
__author__ = "Reltio AI MDM Solution"

from src.core.merge_assistant import MergeAssistant
from src.core.reltio_client import ReltioClient
from src.core.match_analyzer import MatchAnalyzer

__all__ = ["MergeAssistant", "ReltioClient", "MatchAnalyzer"]
