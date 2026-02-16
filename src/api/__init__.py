"""
Python API for HCP Merge Assistant

Provides a clean programmatic interface for integrating
merge operations into custom workflows and pipelines.

Usage:
    from src.api import MergeAPI

    api = MergeAPI.from_env()
    session = api.analyze_file("hcp_records.csv")
    api.auto_select(session, min_score=90)
    result = api.execute_merges(session, dry_run=True)
"""

from src.api.merge_api import MergeAPI

__all__ = ["MergeAPI"]
