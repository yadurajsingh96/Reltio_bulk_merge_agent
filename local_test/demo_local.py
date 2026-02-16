#!/usr/bin/env python3
"""
Bulk Merge Agent - Local Functionality Demo

Demonstrates all functionality that works WITHOUT Reltio API credentials:
1. File parsing (CSV + JSON)
2. Column mapping detection
3. Match scoring engine
4. Merge operation preparation
5. Report generation

No external services required!
"""

import sys
import json
from pathlib import Path

# Add project root to path
PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from src.parsers.file_parser import FileParser, ColumnMapper
from src.core.match_analyzer import MatchScorer, MatchConfidence, MatchReason
from src.core.merge_executor import MergeExecutor, MergeOperation, MergeBatchResult, MergeStatus
from datetime import datetime


def separator(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def demo_column_mapping():
    separator("1. INTELLIGENT COLUMN MAPPING")

    test_columns = [
        "npi", "first_name", "last_name", "provider_npi",
        "zip_code", "phone_number", "merge_with", "custom_field"
    ]

    print("  Input columns -> Mapped Reltio attributes:\n")
    for col in test_columns:
        mapped = ColumnMapper.map_column(col)
        is_id = ColumnMapper.is_identifier(mapped) if mapped else False
        suffix = " [IDENTIFIER]" if is_id else ""
        print(f"    '{col}' -> '{mapped or 'UNMAPPED'}'{suffix}")

    print(f"\n  Result: Auto-mapped {sum(1 for c in test_columns if ColumnMapper.map_column(c))}/{len(test_columns)} columns")


def demo_file_parsing():
    separator("2. FILE PARSING")

    parser = FileParser()

    # CSV parsing
    csv_path = PROJECT_DIR / "sample_data" / "hcp_sample.csv"
    print(f"  Parsing CSV: {csv_path.name}")

    csv_result = parser.parse_file(str(csv_path))
    print(f"    Total records:   {csv_result.total_records}")
    print(f"    Valid records:   {csv_result.valid_records}")
    print(f"    Invalid records: {csv_result.invalid_records}")
    print(f"    Columns:         {', '.join(csv_result.detected_columns)}")

    # Show first record
    first = csv_result.records[0]
    print(f"\n  First record (row {first.row_number}):")
    print(f"    Identifiers: {json.dumps(first.identifiers, indent=6).strip()}")
    print(f"    Attributes:  {json.dumps(dict(list(first.attributes.items())[:3]), indent=6).strip()}...")

    # JSON parsing
    json_path = PROJECT_DIR / "sample_data" / "hcp_sample.json"
    print(f"\n  Parsing JSON: {json_path.name}")

    json_result = parser.parse_file(str(json_path))
    print(f"    Total records: {json_result.total_records}")
    print(f"    Valid records: {json_result.valid_records}")

    print(f"\n  Result: Both CSV and JSON parsing work correctly")


def demo_match_scoring():
    separator("3. MATCH SCORING ENGINE")

    # Simulate matching input records against "Reltio" entities
    input_record = {
        "NPI": "1234567890",
        "FirstName": "John",
        "LastName": "Smith",
        "Specialty": "Cardiology",
        "City": "Boston",
        "State": "MA",
    }

    # Exact match (same person)
    reltio_exact = {
        "NPI": [{"value": "1234567890", "ov": True}],
        "FirstName": [{"value": "John", "ov": True}],
        "LastName": [{"value": "Smith", "ov": True}],
        "Specialty": [{"value": "Cardiology", "ov": True}],
        "City": [{"value": "Boston", "ov": True}],
        "State": [{"value": "MA", "ov": True}],
    }

    # Fuzzy match (similar person, different NPI)
    reltio_fuzzy = {
        "NPI": [{"value": "9876543210", "ov": True}],
        "FirstName": [{"value": "Jon", "ov": True}],
        "LastName": [{"value": "Smithe", "ov": True}],
        "Specialty": [{"value": "Cardiology", "ov": True}],
        "City": [{"value": "Boston", "ov": True}],
    }

    # No match (different person)
    reltio_no_match = {
        "NPI": [{"value": "5555555555", "ov": True}],
        "FirstName": [{"value": "Maria", "ov": True}],
        "LastName": [{"value": "Garcia", "ov": True}],
        "Specialty": [{"value": "Pediatrics", "ov": True}],
        "City": [{"value": "Miami", "ov": True}],
    }

    print(f"  Input: NPI={input_record['NPI']}, {input_record['FirstName']} {input_record['LastName']}, {input_record['Specialty']}")
    print()

    scenarios = [
        ("Exact Match (same NPI + name)", reltio_exact),
        ("Fuzzy Match (different NPI, similar name)", reltio_fuzzy),
        ("No Match (completely different person)", reltio_no_match),
    ]

    for label, reltio_attrs in scenarios:
        score, matched, reasons = MatchScorer.score_match(input_record, reltio_attrs)
        confidence = MatchScorer.determine_confidence(score, reasons)

        flat = MatchScorer._flatten_reltio_attrs(reltio_attrs)
        reltio_name = f"{flat.get('FirstName', '?')} {flat.get('LastName', '?')}"

        print(f"  vs {label}:")
        print(f"    Reltio:     NPI={flat.get('NPI', '?')}, {reltio_name}")
        print(f"    Score:      {score:.1f}%")
        print(f"    Confidence: {confidence.value}")
        print(f"    Reasons:    {', '.join(r.value for r in reasons) or 'None'}")
        print(f"    Matched:    {list(matched.keys())}")
        print()

    print("  Result: Scoring engine correctly differentiates matches")


def demo_merge_preparation():
    separator("4. MERGE OPERATION PREPARATION")

    from src.parsers.file_parser import ParsedRecord
    from src.core.match_analyzer import MatchResult, MatchCandidate

    executor = MergeExecutor()

    # Simulate match results
    results = []
    for i in range(3):
        record = ParsedRecord(
            row_number=i + 1,
            raw_data={"NPI": f"NPI_{i}"},
            normalized_data={"NPI": f"NPI_{i}"},
            identifiers={"NPI": f"NPI_{i}", "EntityURI": f"entities/loser_{i}"},
            attributes={"FirstName": f"Person_{i}"},
        )

        candidate = MatchCandidate(
            entity_uri=f"entities/winner_{i}",
            entity_label=f"Match_{i}",
            entity_type="HCP",
            attributes={},
            match_score=95 - (i * 10),
            confidence=[MatchConfidence.EXACT, MatchConfidence.HIGH, MatchConfidence.MEDIUM][i],
            match_reasons=[MatchReason.NPI_EXACT],
            matched_attributes={},
        )

        results.append(MatchResult(
            input_record=record,
            candidates=[candidate],
            best_match=candidate,
            recommendation=["merge", "merge", "review"][i],
        ))

    # Prepare operations
    ops = executor.prepare_merge_operations(results)
    print(f"  Created {len(ops)} merge operations from {len(results)} match results:\n")

    for op in ops:
        print(f"    {op.id}: {op.loser_uri} -> {op.winner_uri} (score: {op.match_score}%, conf: {op.confidence})")

    print()

    # Generate a mock report
    batch = MergeBatchResult(
        batch_id="demo_batch",
        total_operations=len(ops),
        successful=len(ops) - 1,
        failed=1,
        skipped=0,
        operations=[
            MergeOperation(
                id=op.id, winner_uri=op.winner_uri, loser_uri=op.loser_uri,
                source_record_row=op.source_record_row, match_score=op.match_score,
                confidence=op.confidence,
                status=MergeStatus.SUCCESS if j < len(ops) - 1 else MergeStatus.FAILED,
                error="Simulated failure" if j == len(ops) - 1 else None,
            )
            for j, op in enumerate(ops)
        ],
        started_at=datetime.now(),
        completed_at=datetime.now(),
        duration_seconds=0.5,
    )

    report = executor.generate_report(batch)
    print(f"  Report summary:")
    print(f"    Total:      {report['summary']['total_operations']}")
    print(f"    Successful: {report['summary']['successful']}")
    print(f"    Failed:     {report['summary']['failed']}")
    print(f"    Success %:  {report['summary']['success_rate']:.0f}%")

    print(f"\n  Result: Merge operations prepared and report generated")


def demo_attribute_weights():
    separator("5. ATTRIBUTE WEIGHT REFERENCE")

    print("  Match scoring weights (higher = more important):\n")
    sorted_weights = sorted(
        MatchScorer.ATTRIBUTE_WEIGHTS.items(),
        key=lambda x: x[1],
        reverse=True,
    )
    for attr, weight in sorted_weights:
        bar = "#" * (weight // 5)
        print(f"    {attr:20s} {weight:3d}  {bar}")


def main():
    print("\n" + "=" * 60)
    print("  BULK MERGE AGENT - LOCAL FUNCTIONALITY DEMO")
    print("  Testing all features that work without Reltio API")
    print("=" * 60)

    demo_column_mapping()
    demo_file_parsing()
    demo_match_scoring()
    demo_merge_preparation()
    demo_attribute_weights()

    separator("ALL DEMOS COMPLETED SUCCESSFULLY")
    print("  All local features are working correctly.\n")
    print("  To test with real Reltio API:")
    print("    1. Copy .env.example to .env")
    print("    2. Fill in your Reltio credentials")
    print("    3. Run: python -m src.cli health")
    print("    4. Run: python -m src.cli analyze --file sample_data/hcp_sample.csv")
    print()


if __name__ == "__main__":
    main()
