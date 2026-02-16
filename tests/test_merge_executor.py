"""Tests for the merge executor module (preparation logic only, no API calls)."""

import pytest
from src.core.merge_executor import (
    MergeExecutor,
    MergeOperation,
    MergeBatchResult,
    MergeStatus,
)
from src.core.match_analyzer import (
    MatchResult,
    MatchCandidate,
    MatchConfidence,
    MatchReason,
)
from src.parsers.file_parser import ParsedRecord


def _make_record(row, npi="1234567890", entity_uri="entities/loser1"):
    """Helper to create a ParsedRecord."""
    return ParsedRecord(
        row_number=row,
        raw_data={"NPI": npi},
        normalized_data={"NPI": npi},
        identifiers={"NPI": npi, "EntityURI": entity_uri},
        attributes={"FirstName": "John", "LastName": "Smith"},
    )


def _make_candidate(uri="entities/winner1", score=95.0, confidence=MatchConfidence.HIGH):
    """Helper to create a MatchCandidate."""
    return MatchCandidate(
        entity_uri=uri,
        entity_label="John Smith",
        entity_type="HCP",
        attributes={},
        match_score=score,
        confidence=confidence,
        match_reasons=[MatchReason.NPI_EXACT],
        matched_attributes={},
    )


def _make_match_result(record, candidate, recommendation="merge"):
    """Helper to create a MatchResult."""
    return MatchResult(
        input_record=record,
        candidates=[candidate],
        best_match=candidate,
        recommendation=recommendation,
    )


class TestPrepareOperations:
    """Tests for preparing merge operations from match results."""

    def test_prepare_from_best_match(self):
        executor = MergeExecutor()
        record = _make_record(1)
        candidate = _make_candidate()
        result = _make_match_result(record, candidate)

        ops = executor.prepare_merge_operations([result])
        assert len(ops) == 1
        assert ops[0].winner_uri == "entities/winner1"
        assert ops[0].loser_uri == "entities/loser1"

    def test_skip_no_match_results(self):
        executor = MergeExecutor()
        record = _make_record(1)
        result = MatchResult(
            input_record=record,
            candidates=[],
            best_match=None,
            recommendation="no_match",
        )

        ops = executor.prepare_merge_operations([result])
        assert len(ops) == 0

    def test_skip_records_without_entity_uri(self):
        executor = MergeExecutor()
        record = ParsedRecord(
            row_number=1,
            raw_data={"NPI": "1234"},
            normalized_data={"NPI": "1234"},
            identifiers={"NPI": "1234"},  # No EntityURI
            attributes={"FirstName": "John"},
        )
        candidate = _make_candidate()
        result = _make_match_result(record, candidate)

        ops = executor.prepare_merge_operations([result])
        assert len(ops) == 0

    def test_selected_indices_override(self):
        executor = MergeExecutor()
        record = _make_record(1)
        candidate1 = _make_candidate("entities/c1", score=90)
        candidate2 = _make_candidate("entities/c2", score=85)

        result = MatchResult(
            input_record=record,
            candidates=[candidate1, candidate2],
            best_match=candidate1,
            recommendation="merge",
        )

        # Select second candidate explicitly
        ops = executor.prepare_merge_operations([result], selected_indices={0: 1})
        assert len(ops) == 1
        assert ops[0].winner_uri == "entities/c2"

    def test_multiple_results(self):
        executor = MergeExecutor()
        records = [_make_record(i, f"NPI{i}", f"entities/loser{i}") for i in range(5)]
        candidates = [_make_candidate(f"entities/winner{i}") for i in range(5)]
        results = [_make_match_result(r, c) for r, c in zip(records, candidates)]

        ops = executor.prepare_merge_operations(results)
        assert len(ops) == 5


class TestDirectMergeOperations:
    """Tests for direct merge pair preparation."""

    def test_prepare_direct_merges(self):
        executor = MergeExecutor()
        pairs = [
            {"winner_uri": "entities/w1", "loser_uri": "entities/l1"},
            {"winner_uri": "entities/w2", "loser_uri": "entities/l2"},
        ]

        ops = executor.prepare_direct_merge_operations(pairs)
        assert len(ops) == 2
        assert ops[0].winner_uri == "entities/w1"
        assert ops[0].loser_uri == "entities/l1"


class TestReportGeneration:
    """Tests for merge report generation."""

    def test_generate_report(self):
        executor = MergeExecutor()
        from datetime import datetime

        batch = MergeBatchResult(
            batch_id="test_batch",
            total_operations=3,
            successful=2,
            failed=1,
            skipped=0,
            operations=[
                MergeOperation(
                    id="op1", winner_uri="w1", loser_uri="l1",
                    source_record_row=1, match_score=95,
                    confidence="high", status=MergeStatus.SUCCESS,
                ),
                MergeOperation(
                    id="op2", winner_uri="w2", loser_uri="l2",
                    source_record_row=2, match_score=88,
                    confidence="medium", status=MergeStatus.SUCCESS,
                ),
                MergeOperation(
                    id="op3", winner_uri="w3", loser_uri="l3",
                    source_record_row=3, match_score=72,
                    confidence="low", status=MergeStatus.FAILED,
                    error="Entity not found",
                ),
            ],
            started_at=datetime.now(),
            completed_at=datetime.now(),
            duration_seconds=1.5,
            errors=["Entity not found"],
        )

        report = executor.generate_report(batch)
        assert report["summary"]["total_operations"] == 3
        assert report["summary"]["successful"] == 2
        assert report["summary"]["failed"] == 1
        assert len(report["operations"]) == 3
        assert report["errors"] == ["Entity not found"]
