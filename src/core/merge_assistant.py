"""
HCP Merge Assistant - Main Orchestration Class

This is the primary interface for the merge assistant, combining:
- File parsing
- Match analysis
- Merge execution
- Progress tracking
- Reporting

Designed for both programmatic use and as backend for UI/CLI.
"""

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from src.core.match_analyzer import MatchAnalyzer, MatchConfidence, MatchResult
from src.core.merge_executor import MergeBatchResult, MergeExecutor, MergeOperation
from src.core.reltio_client import ReltioClient, ReltioConfig
from src.parsers.file_parser import FileParser, ParsedFile

logger = logging.getLogger(__name__)


@dataclass
class AssistantConfig:
    """Configuration for the Merge Assistant"""
    # Reltio connection
    reltio_client_id: str
    reltio_client_secret: str
    reltio_tenant_id: str
    reltio_environment: str = "dev"

    # LLM configuration (optional)
    llm_api_key: Optional[str] = None
    llm_provider: str = "openai"  # 'openai' or 'anthropic'
    use_llm: bool = True

    # Processing options
    entity_type: str = "HCP"
    max_concurrent_requests: int = 10
    batch_size: int = 100

    # Merge options
    auto_merge_threshold: float = 95.0  # Auto-approve merges above this score
    review_threshold: float = 70.0       # Require review for scores above this

    @classmethod
    def from_env(cls) -> "AssistantConfig":
        """Create config from environment variables"""
        return cls(
            reltio_client_id=os.getenv("RELTIO_CLIENT_ID", ""),
            reltio_client_secret=os.getenv("RELTIO_CLIENT_SECRET", ""),
            reltio_tenant_id=os.getenv("RELTIO_TENANT_ID", ""),
            reltio_environment=os.getenv("RELTIO_ENVIRONMENT", "dev"),
            llm_api_key=os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY"),
            llm_provider="anthropic" if os.getenv("ANTHROPIC_API_KEY") else "openai",
            use_llm=bool(os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")),
            entity_type=os.getenv("ENTITY_TYPE", "HCP"),
            max_concurrent_requests=int(os.getenv("MAX_CONCURRENT", "10"))
        )


@dataclass
class AnalysisSession:
    """Represents an active analysis session"""
    session_id: str
    file_path: str
    parsed_file: ParsedFile
    match_results: List[MatchResult] = field(default_factory=list)
    selected_merges: Dict[int, int] = field(default_factory=dict)  # result_idx -> candidate_idx
    merge_result: Optional[MergeBatchResult] = None
    created_at: datetime = field(default_factory=datetime.now)
    status: str = "initialized"  # initialized, analyzing, analyzed, merging, completed


class MergeAssistant:
    """
    Main HCP Merge Assistant class

    Provides a high-level interface for:
    1. Loading and parsing input files
    2. Running match analysis
    3. Reviewing and selecting matches
    4. Executing bulk merges
    5. Generating reports

    Can be used directly via Python API or as backend for Streamlit/CLI.
    """

    def __init__(self, config: AssistantConfig):
        """
        Initialize the Merge Assistant

        Args:
            config: Assistant configuration
        """
        self.config = config
        self.file_parser = FileParser()

        # Build Reltio config
        self.reltio_config = ReltioConfig(
            client_id=config.reltio_client_id,
            client_secret=config.reltio_client_secret,
            tenant_id=config.reltio_tenant_id,
            environment=config.reltio_environment,
            max_concurrent_requests=config.max_concurrent_requests
        )

        # Initialize components
        self.match_analyzer = MatchAnalyzer(
            reltio_config=self.reltio_config,
            llm_api_key=config.llm_api_key,
            llm_provider=config.llm_provider,
            use_llm=config.use_llm
        )

        self.merge_executor = MergeExecutor(
            max_concurrent=config.max_concurrent_requests
        )

        # Active sessions
        self._sessions: Dict[str, AnalysisSession] = {}

    # =========================================================================
    # FILE OPERATIONS
    # =========================================================================

    def parse_file(
        self,
        file_path: str,
        column_mapping: Optional[Dict[str, str]] = None
    ) -> ParsedFile:
        """
        Parse an input file (CSV, Excel, or JSON)

        Args:
            file_path: Path to the file
            column_mapping: Optional custom column mapping

        Returns:
            ParsedFile with all records
        """
        if column_mapping:
            self.file_parser = FileParser(column_mapping=column_mapping)

        parsed = self.file_parser.parse_file(file_path)

        logger.info(
            f"Parsed {parsed.filename}: {parsed.total_records} records, "
            f"{parsed.valid_records} valid, {parsed.invalid_records} invalid"
        )

        return parsed

    def create_session(
        self,
        file_path: str,
        column_mapping: Optional[Dict[str, str]] = None
    ) -> AnalysisSession:
        """
        Create a new analysis session from a file

        Args:
            file_path: Path to input file
            column_mapping: Optional column mapping

        Returns:
            New AnalysisSession
        """
        parsed_file = self.parse_file(file_path, column_mapping)

        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        session = AnalysisSession(
            session_id=session_id,
            file_path=file_path,
            parsed_file=parsed_file,
            status="initialized"
        )

        self._sessions[session_id] = session
        logger.info(f"Created session {session_id}")

        return session

    def get_session(self, session_id: str) -> Optional[AnalysisSession]:
        """Get an existing session by ID"""
        return self._sessions.get(session_id)

    # =========================================================================
    # MATCH ANALYSIS
    # =========================================================================

    async def analyze_matches(
        self,
        session: AnalysisSession,
        on_progress: Optional[Callable[[int, int], None]] = None
    ) -> List[MatchResult]:
        """
        Run match analysis on all records in a session

        Args:
            session: The analysis session
            on_progress: Optional callback(completed, total)

        Returns:
            List of MatchResults
        """
        session.status = "analyzing"
        records = session.parsed_file.records

        logger.info(f"Starting match analysis for {len(records)} records")

        async with ReltioClient(self.reltio_config) as client:
            results = await self.match_analyzer.analyze_batch(
                records=records,
                reltio_client=client,
                entity_type=self.config.entity_type,
                on_progress=on_progress
            )

        session.match_results = results
        session.status = "analyzed"

        # Log summary
        auto_merge = sum(
            1 for r in results if r.best_match and r.best_match.match_score >= self.config.auto_merge_threshold
        )
        review = sum(
            1 for r in results
            if r.best_match
            and self.config.review_threshold <= r.best_match.match_score < self.config.auto_merge_threshold
        )
        no_match = sum(1 for r in results if not r.best_match or r.recommendation == "no_match")

        logger.info(
            f"Analysis complete: {auto_merge} auto-merge, {review} review, {no_match} no match"
        )

        return results

    def run_analysis(
        self,
        session: AnalysisSession,
        on_progress: Optional[Callable[[int, int], None]] = None
    ) -> List[MatchResult]:
        """
        Synchronous wrapper for analyze_matches

        Args:
            session: The analysis session
            on_progress: Optional progress callback

        Returns:
            List of MatchResults
        """
        return asyncio.run(self.analyze_matches(session, on_progress))

    # =========================================================================
    # MATCH SELECTION
    # =========================================================================

    def select_merge(
        self,
        session: AnalysisSession,
        result_index: int,
        candidate_index: int
    ) -> None:
        """
        Select a specific candidate for merge

        Args:
            session: The analysis session
            result_index: Index in match_results
            candidate_index: Index in candidates list
        """
        if result_index < len(session.match_results):
            result = session.match_results[result_index]
            if candidate_index < len(result.candidates):
                session.selected_merges[result_index] = candidate_index
                logger.debug(f"Selected candidate {candidate_index} for result {result_index}")

    def deselect_merge(
        self,
        session: AnalysisSession,
        result_index: int
    ) -> None:
        """Remove a merge selection"""
        if result_index in session.selected_merges:
            del session.selected_merges[result_index]

    def auto_select_merges(
        self,
        session: AnalysisSession,
        min_score: Optional[float] = None
    ) -> int:
        """
        Automatically select high-confidence merges

        Args:
            session: The analysis session
            min_score: Minimum score threshold (defaults to config.auto_merge_threshold)

        Returns:
            Number of merges selected
        """
        threshold = min_score or self.config.auto_merge_threshold
        count = 0

        for i, result in enumerate(session.match_results):
            if result.best_match and result.best_match.match_score >= threshold:
                session.selected_merges[i] = 0  # Select best match
                count += 1

        logger.info(f"Auto-selected {count} merges above {threshold}% threshold")
        return count

    def get_merge_summary(
        self,
        session: AnalysisSession
    ) -> Dict[str, Any]:
        """
        Get summary of current merge selections

        Returns:
            Summary statistics
        """
        results = session.match_results

        return {
            "total_records": len(results),
            "with_matches": sum(1 for r in results if r.candidates),
            "no_matches": sum(1 for r in results if not r.candidates),
            "selected_for_merge": len(session.selected_merges),
            "by_confidence": {
                "exact": sum(1 for r in results if r.best_match and r.best_match.confidence == MatchConfidence.EXACT),
                "high": sum(1 for r in results if r.best_match and r.best_match.confidence == MatchConfidence.HIGH),
                "medium": sum(1 for r in results if r.best_match and r.best_match.confidence == MatchConfidence.MEDIUM),
                "low": sum(1 for r in results if r.best_match and r.best_match.confidence == MatchConfidence.LOW),
            },
            "by_recommendation": {
                "merge": sum(1 for r in results if r.recommendation == "merge"),
                "review": sum(1 for r in results if r.recommendation == "review"),
                "skip": sum(1 for r in results if r.recommendation == "skip"),
                "no_match": sum(1 for r in results if r.recommendation == "no_match"),
            }
        }

    # =========================================================================
    # MERGE EXECUTION
    # =========================================================================

    async def execute_merges(
        self,
        session: AnalysisSession,
        on_progress: Optional[Callable[[int, int, MergeOperation], None]] = None,
        dry_run: bool = False
    ) -> MergeBatchResult:
        """
        Execute selected merges

        Args:
            session: The analysis session
            on_progress: Optional callback(completed, total, current_op)
            dry_run: If True, validate but don't execute

        Returns:
            MergeBatchResult with all results
        """
        session.status = "merging"

        # Prepare operations from selected merges
        operations = self.merge_executor.prepare_merge_operations(
            match_results=session.match_results,
            selected_indices=session.selected_merges
        )

        if not operations:
            logger.warning("No valid merge operations to execute")
            session.status = "analyzed"
            return MergeBatchResult(
                batch_id="empty",
                total_operations=0,
                successful=0,
                failed=0,
                skipped=0,
                operations=[],
                started_at=datetime.now()
            )

        logger.info(f"Executing {len(operations)} merge operations (dry_run={dry_run})")

        async with ReltioClient(self.reltio_config) as client:
            # Validate first
            issues = await self.merge_executor.validate_operations(operations, client)
            if issues:
                logger.warning(f"Validation found {len(issues)} issues")
                for issue in issues[:5]:  # Log first 5
                    logger.warning(f"  {issue['operation_id']}: {issue['message']}")

            # Execute merges
            result = await self.merge_executor.execute_merges(
                operations=operations,
                reltio_client=client,
                on_progress=on_progress,
                dry_run=dry_run
            )

        session.merge_result = result
        session.status = "completed"

        return result

    def run_merges(
        self,
        session: AnalysisSession,
        on_progress: Optional[Callable[[int, int, MergeOperation], None]] = None,
        dry_run: bool = False
    ) -> MergeBatchResult:
        """
        Synchronous wrapper for execute_merges

        Args:
            session: The analysis session
            on_progress: Optional progress callback
            dry_run: Validation only mode

        Returns:
            MergeBatchResult
        """
        return asyncio.run(self.execute_merges(session, on_progress, dry_run))

    # =========================================================================
    # DIRECT MERGE (without file)
    # =========================================================================

    async def merge_entity_pairs(
        self,
        pairs: List[Dict[str, str]],
        on_progress: Optional[Callable[[int, int, MergeOperation], None]] = None,
        dry_run: bool = False
    ) -> MergeBatchResult:
        """
        Directly merge entity pairs without file analysis

        Args:
            pairs: List of {"winner_uri": ..., "loser_uri": ...}
            on_progress: Optional progress callback
            dry_run: Validation only mode

        Returns:
            MergeBatchResult
        """
        operations = self.merge_executor.prepare_direct_merge_operations(pairs)

        async with ReltioClient(self.reltio_config) as client:
            return await self.merge_executor.execute_merges(
                operations=operations,
                reltio_client=client,
                on_progress=on_progress,
                dry_run=dry_run
            )

    # =========================================================================
    # REPORTING
    # =========================================================================

    def generate_report(
        self,
        session: AnalysisSession,
        include_all_candidates: bool = False
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive report for a session

        Args:
            session: The analysis session
            include_all_candidates: Include all candidates or just selected

        Returns:
            Report dictionary
        """
        report = {
            "session_id": session.session_id,
            "file": session.file_path,
            "created_at": session.created_at.isoformat(),
            "status": session.status,
            "input_summary": {
                "total_records": session.parsed_file.total_records,
                "valid_records": session.parsed_file.valid_records,
                "invalid_records": session.parsed_file.invalid_records,
                "columns_detected": session.parsed_file.detected_columns,
                "column_mapping": session.parsed_file.column_mapping
            }
        }

        if session.match_results:
            report["analysis_summary"] = self.get_merge_summary(session)

            report["match_details"] = []
            for i, result in enumerate(session.match_results):
                detail = {
                    "row": result.input_record.row_number,
                    "input_data": result.input_record.normalized_data,
                    "recommendation": result.recommendation,
                    "processing_time_ms": result.processing_time_ms,
                    "selected_for_merge": i in session.selected_merges
                }

                if result.best_match:
                    detail["best_match"] = {
                        "uri": result.best_match.entity_uri,
                        "label": result.best_match.entity_label,
                        "score": result.best_match.match_score,
                        "confidence": result.best_match.confidence.value,
                        "reasons": [r.value for r in result.best_match.match_reasons]
                    }

                if include_all_candidates:
                    detail["all_candidates"] = [
                        {
                            "uri": c.entity_uri,
                            "label": c.entity_label,
                            "score": c.match_score,
                            "confidence": c.confidence.value
                        }
                        for c in result.candidates
                    ]

                if result.llm_analysis:
                    detail["llm_analysis"] = result.llm_analysis

                report["match_details"].append(detail)

        if session.merge_result:
            report["merge_result"] = self.merge_executor.generate_report(
                session.merge_result,
                include_details=True
            )

        return report

    def export_report(
        self,
        session: AnalysisSession,
        output_path: str,
        format: str = "json"
    ) -> None:
        """
        Export session report to file

        Args:
            session: The analysis session
            output_path: Output file path
            format: 'json' or 'csv'
        """
        report = self.generate_report(session, include_all_candidates=True)

        if format == "json":
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)

        elif format == "csv":
            import csv
            with open(output_path, 'w', newline='') as f:
                if report.get("match_details"):
                    # Flatten for CSV
                    rows = []
                    for detail in report["match_details"]:
                        row = {
                            "row": detail["row"],
                            "recommendation": detail["recommendation"],
                            "selected": detail["selected_for_merge"]
                        }
                        row.update(detail.get("input_data", {}))
                        if detail.get("best_match"):
                            row["match_uri"] = detail["best_match"]["uri"]
                            row["match_score"] = detail["best_match"]["score"]
                            row["match_confidence"] = detail["best_match"]["confidence"]
                        rows.append(row)

                    if rows:
                        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                        writer.writeheader()
                        writer.writerows(rows)

        logger.info(f"Report exported to {output_path}")

    # =========================================================================
    # HEALTH CHECK
    # =========================================================================

    async def health_check(self) -> Dict[str, Any]:
        """
        Check connectivity to Reltio and LLM services

        Returns:
            Health status dictionary
        """
        status = {
            "reltio": {"status": "unknown"},
            "llm": {"status": "not_configured"}
        }

        # Check Reltio
        try:
            async with ReltioClient(self.reltio_config) as client:
                reltio_health = await client.health_check()
                status["reltio"] = reltio_health
        except Exception as e:
            status["reltio"] = {"status": "error", "error": str(e)}

        # Check LLM if configured
        if self.config.use_llm and self.config.llm_api_key:
            try:
                # Simple validation - just check API key format
                if self.config.llm_provider == "openai" and self.config.llm_api_key.startswith("sk-"):
                    status["llm"] = {"status": "configured", "provider": "openai"}
                elif self.config.llm_provider == "anthropic" and "sk-ant" in self.config.llm_api_key:
                    status["llm"] = {"status": "configured", "provider": "anthropic"}
                else:
                    status["llm"] = {"status": "configured", "provider": self.config.llm_provider}
            except Exception as e:
                status["llm"] = {"status": "error", "error": str(e)}

        return status

    def run_health_check(self) -> Dict[str, Any]:
        """Synchronous wrapper for health_check"""
        return asyncio.run(self.health_check())
