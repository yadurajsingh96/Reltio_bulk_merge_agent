"""
Python API for HCP Merge Assistant

High-level API for programmatic integration. Wraps MergeAssistant
with a simplified interface suitable for scripts, notebooks, and pipelines.

Usage:
    from src.api import MergeAPI

    # Initialize from environment variables
    api = MergeAPI.from_env()

    # Or with explicit config
    api = MergeAPI(
        client_id="...",
        client_secret="...",
        tenant_id="...",
        environment="dev"
    )

    # Full workflow
    session = api.analyze_file("hcp_records.csv", entity_type="HCP")
    summary = api.get_summary(session)

    # Review and select
    api.auto_select(session, min_score=90)
    # or manual: api.select(session, record_index=0, candidate_index=0)

    # Execute merges
    result = api.execute_merges(session, dry_run=True)

    # Export report
    api.export_json(session, "report.json")
    api.export_csv(session, "report.csv")
"""

import asyncio
import logging
import os
from typing import Dict, Any, Optional, List, Callable
from pathlib import Path

from src.core.merge_assistant import MergeAssistant, AssistantConfig, AnalysisSession
from src.core.match_analyzer import MatchResult, MatchConfidence
from src.core.merge_executor import MergeBatchResult

logger = logging.getLogger(__name__)


class MergeAPI:
    """
    Simplified Python API for HCP Merge Assistant.

    Provides a streamlined interface for:
    - File analysis with match scoring
    - Merge selection (auto or manual)
    - Bulk merge execution
    - Report generation

    Thread-safe for use in web frameworks and task queues.
    """

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        tenant_id: str,
        environment: str = "dev",
        llm_api_key: Optional[str] = None,
        llm_provider: str = "openai",
        use_llm: bool = True,
        max_concurrent: int = 10,
        auto_merge_threshold: float = 95.0,
        review_threshold: float = 70.0
    ):
        """
        Initialize the Merge API.

        Args:
            client_id: Reltio OAuth client ID
            client_secret: Reltio OAuth client secret
            tenant_id: Reltio tenant ID
            environment: Reltio environment (dev, test, prod)
            llm_api_key: OpenAI or Anthropic API key
            llm_provider: 'openai' or 'anthropic'
            use_llm: Whether to use LLM for analysis
            max_concurrent: Maximum concurrent API requests
            auto_merge_threshold: Score threshold for auto-merge
            review_threshold: Score threshold for review
        """
        self._config = AssistantConfig(
            reltio_client_id=client_id,
            reltio_client_secret=client_secret,
            reltio_tenant_id=tenant_id,
            reltio_environment=environment,
            llm_api_key=llm_api_key,
            llm_provider=llm_provider,
            use_llm=use_llm and bool(llm_api_key),
            max_concurrent_requests=max_concurrent,
            auto_merge_threshold=auto_merge_threshold,
            review_threshold=review_threshold
        )
        self._assistant = MergeAssistant(self._config)

    @classmethod
    def from_env(cls) -> "MergeAPI":
        """
        Create MergeAPI from environment variables.

        Required env vars:
            RELTIO_CLIENT_ID, RELTIO_CLIENT_SECRET, RELTIO_TENANT_ID

        Optional env vars:
            RELTIO_ENVIRONMENT (default: dev)
            OPENAI_API_KEY or ANTHROPIC_API_KEY
            MAX_CONCURRENT (default: 10)
            AUTO_MERGE_THRESHOLD (default: 95)
            REVIEW_THRESHOLD (default: 70)

        Returns:
            Configured MergeAPI instance
        """
        llm_key = os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
        llm_provider = "anthropic" if os.getenv("ANTHROPIC_API_KEY") and not os.getenv("OPENAI_API_KEY") else "openai"

        return cls(
            client_id=os.getenv("RELTIO_CLIENT_ID", ""),
            client_secret=os.getenv("RELTIO_CLIENT_SECRET", ""),
            tenant_id=os.getenv("RELTIO_TENANT_ID", ""),
            environment=os.getenv("RELTIO_ENVIRONMENT", "dev"),
            llm_api_key=llm_key,
            llm_provider=llm_provider,
            use_llm=bool(llm_key),
            max_concurrent=int(os.getenv("MAX_CONCURRENT", "10")),
            auto_merge_threshold=float(os.getenv("AUTO_MERGE_THRESHOLD", "95")),
            review_threshold=float(os.getenv("REVIEW_THRESHOLD", "70"))
        )

    # =========================================================================
    # ANALYSIS
    # =========================================================================

    def analyze_file(
        self,
        file_path: str,
        entity_type: str = "HCP",
        column_mapping: Optional[Dict[str, str]] = None,
        on_progress: Optional[Callable[[int, int], None]] = None
    ) -> AnalysisSession:
        """
        Analyze a file for potential merges.

        Parses the input file, searches Reltio for matches,
        and scores each record against potential merge candidates.

        Args:
            file_path: Path to CSV, Excel, or JSON file
            entity_type: Reltio entity type (default: HCP)
            column_mapping: Optional custom column name mapping
            on_progress: Optional callback(completed, total)

        Returns:
            AnalysisSession with match results

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is unsupported
        """
        if not Path(file_path).exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        self._config.entity_type = entity_type
        session = self._assistant.create_session(file_path, column_mapping)
        self._assistant.run_analysis(session, on_progress)

        return session

    async def analyze_file_async(
        self,
        file_path: str,
        entity_type: str = "HCP",
        column_mapping: Optional[Dict[str, str]] = None,
        on_progress: Optional[Callable[[int, int], None]] = None
    ) -> AnalysisSession:
        """
        Async version of analyze_file.

        Use this when running inside an existing event loop
        (e.g., FastAPI, async frameworks).
        """
        if not Path(file_path).exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        self._config.entity_type = entity_type
        session = self._assistant.create_session(file_path, column_mapping)
        await self._assistant.analyze_matches(session, on_progress)

        return session

    # =========================================================================
    # SELECTION
    # =========================================================================

    def auto_select(
        self,
        session: AnalysisSession,
        min_score: Optional[float] = None
    ) -> int:
        """
        Auto-select high-confidence merges.

        Args:
            session: Analysis session from analyze_file()
            min_score: Minimum match score (default: auto_merge_threshold)

        Returns:
            Number of merges selected
        """
        return self._assistant.auto_select_merges(session, min_score)

    def select(
        self,
        session: AnalysisSession,
        record_index: int,
        candidate_index: int = 0
    ) -> None:
        """
        Manually select a specific match for merge.

        Args:
            session: Analysis session
            record_index: Index of the input record (0-based)
            candidate_index: Index of the match candidate (0-based, default: best match)
        """
        self._assistant.select_merge(session, record_index, candidate_index)

    def deselect(
        self,
        session: AnalysisSession,
        record_index: int
    ) -> None:
        """
        Remove a merge selection.

        Args:
            session: Analysis session
            record_index: Index of the record to deselect
        """
        self._assistant.deselect_merge(session, record_index)

    def select_all(
        self,
        session: AnalysisSession,
        min_confidence: str = "high"
    ) -> int:
        """
        Select all records at or above a confidence level.

        Args:
            session: Analysis session
            min_confidence: Minimum confidence ('exact', 'high', 'medium', 'low')

        Returns:
            Number of merges selected
        """
        confidence_order = ["low", "medium", "high", "exact"]
        min_idx = confidence_order.index(min_confidence) if min_confidence in confidence_order else 0

        count = 0
        for i, result in enumerate(session.match_results):
            if result.best_match:
                conf_val = result.best_match.confidence.value
                if conf_val in confidence_order:
                    conf_idx = confidence_order.index(conf_val)
                    if conf_idx >= min_idx:
                        session.selected_merges[i] = 0
                        count += 1

        return count

    def clear_selections(self, session: AnalysisSession) -> None:
        """Clear all merge selections."""
        session.selected_merges.clear()

    # =========================================================================
    # MERGE EXECUTION
    # =========================================================================

    def execute_merges(
        self,
        session: AnalysisSession,
        dry_run: bool = False,
        on_progress: Optional[Callable] = None
    ) -> MergeBatchResult:
        """
        Execute selected merges.

        Args:
            session: Analysis session with selections
            dry_run: If True, validate without executing
            on_progress: Optional callback(completed, total, operation)

        Returns:
            MergeBatchResult with execution details
        """
        return self._assistant.run_merges(session, on_progress, dry_run)

    async def execute_merges_async(
        self,
        session: AnalysisSession,
        dry_run: bool = False,
        on_progress: Optional[Callable] = None
    ) -> MergeBatchResult:
        """Async version of execute_merges."""
        return await self._assistant.execute_merges(session, on_progress, dry_run)

    def merge_pairs(
        self,
        pairs: List[Dict[str, str]],
        dry_run: bool = False,
        on_progress: Optional[Callable] = None
    ) -> MergeBatchResult:
        """
        Directly merge entity pairs without file analysis.

        Args:
            pairs: List of {"winner_uri": "...", "loser_uri": "..."}
            dry_run: Validate only mode
            on_progress: Optional progress callback

        Returns:
            MergeBatchResult
        """
        return asyncio.run(
            self._assistant.merge_entity_pairs(pairs, on_progress, dry_run)
        )

    # =========================================================================
    # REPORTING
    # =========================================================================

    def get_summary(self, session: AnalysisSession) -> Dict[str, Any]:
        """
        Get analysis summary statistics.

        Returns dict with keys:
            total_records, with_matches, no_matches,
            selected_for_merge, by_confidence, by_recommendation
        """
        return self._assistant.get_merge_summary(session)

    def get_results(self, session: AnalysisSession) -> List[MatchResult]:
        """Get all match results from a session."""
        return session.match_results

    def get_result(self, session: AnalysisSession, index: int) -> Optional[MatchResult]:
        """Get a specific match result by index."""
        if 0 <= index < len(session.match_results):
            return session.match_results[index]
        return None

    def export_json(self, session: AnalysisSession, output_path: str) -> str:
        """
        Export analysis report to JSON.

        Args:
            session: Analysis session
            output_path: Output file path

        Returns:
            Path to exported file
        """
        self._assistant.export_report(session, output_path, format="json")
        return output_path

    def export_csv(self, session: AnalysisSession, output_path: str) -> str:
        """
        Export analysis report to CSV.

        Args:
            session: Analysis session
            output_path: Output file path

        Returns:
            Path to exported file
        """
        self._assistant.export_report(session, output_path, format="csv")
        return output_path

    def get_report(
        self,
        session: AnalysisSession,
        include_all_candidates: bool = False
    ) -> Dict[str, Any]:
        """
        Get full report as a dictionary.

        Args:
            session: Analysis session
            include_all_candidates: Include all match candidates

        Returns:
            Complete report dictionary
        """
        return self._assistant.generate_report(session, include_all_candidates)

    # =========================================================================
    # HEALTH
    # =========================================================================

    def health_check(self) -> Dict[str, Any]:
        """
        Check Reltio and LLM connectivity.

        Returns:
            Dict with 'reltio' and 'llm' status
        """
        return self._assistant.run_health_check()

    # =========================================================================
    # CONVENIENCE
    # =========================================================================

    def quick_merge(
        self,
        file_path: str,
        entity_type: str = "HCP",
        min_score: float = 95.0,
        dry_run: bool = True,
        on_progress: Optional[Callable[[int, int], None]] = None
    ) -> Dict[str, Any]:
        """
        One-call workflow: analyze, auto-select, merge, report.

        Convenience method that runs the full merge workflow in one call.
        Defaults to dry_run=True for safety.

        Args:
            file_path: Input file path
            entity_type: Reltio entity type
            min_score: Minimum score for auto-selection
            dry_run: If True, validates but doesn't merge (default: True)
            on_progress: Optional progress callback

        Returns:
            Dict with 'session', 'summary', 'merge_result', 'report'
        """
        # Analyze
        session = self.analyze_file(file_path, entity_type, on_progress=on_progress)

        # Auto-select
        selected = self.auto_select(session, min_score=min_score)

        # Get summary
        summary = self.get_summary(session)

        # Execute (dry_run by default)
        merge_result = None
        if selected > 0:
            merge_result = self.execute_merges(session, dry_run=dry_run)

        return {
            "session": session,
            "summary": summary,
            "selected_count": selected,
            "merge_result": merge_result,
            "report": self.get_report(session)
        }
