#!/usr/bin/env python3
"""
Integration Test with Mock Reltio Server

Prerequisites:
    1. Start mock server: python local_test/mock_reltio_server.py
    2. Run this script: python local_test/demo_integration.py

This tests the full pipeline: parse file -> search -> score -> merge (dry run)
"""

import sys
import asyncio
import json
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from src.core.reltio_client import ReltioClient, ReltioConfig
from src.core.match_analyzer import MatchScorer
from src.parsers.file_parser import FileParser


MOCK_PORT = 8888


async def test_mock_connectivity():
    """Test basic connectivity to mock server."""
    print("\n  [1] Testing mock server connectivity...")

    config = ReltioConfig(
        client_id="test",
        client_secret="test",
        tenant_id="test-tenant",
        environment=f"localhost:{MOCK_PORT}",
        auth_server=f"http://localhost:{MOCK_PORT}",
    )

    try:
        async with ReltioClient(config) as client:
            health = await client.health_check()
            print(f"      Status: {health.get('status', 'unknown')}")
            print(f"      Connected: {health.get('connected', False)}")
            return health.get("connected", False)
    except Exception as e:
        print(f"      ERROR: {e}")
        print(f"      Is the mock server running? Start it with:")
        print(f"        python local_test/mock_reltio_server.py")
        return False


async def test_entity_search():
    """Test entity search against mock server."""
    print("\n  [2] Testing entity search...")

    config = ReltioConfig(
        client_id="test",
        client_secret="test",
        tenant_id="test-tenant",
        environment=f"localhost:{MOCK_PORT}",
        auth_server=f"http://localhost:{MOCK_PORT}",
    )

    async with ReltioClient(config) as client:
        results = await client.search_entities(
            filter_expr="equals(attributes.NPI,'1234567890')",
            entity_type="HCP",
            max_results=5,
        )

        if isinstance(results, list):
            print(f"      Found {len(results)} entities")
            for r in results[:3]:
                print(f"        - {r.get('label', 'Unknown')} ({r.get('uri', '?')})")
        else:
            print(f"      Unexpected response type: {type(results)}")


async def test_get_entity():
    """Test getting a single entity."""
    print("\n  [3] Testing entity retrieval...")

    config = ReltioConfig(
        client_id="test",
        client_secret="test",
        tenant_id="test-tenant",
        environment=f"localhost:{MOCK_PORT}",
        auth_server=f"http://localhost:{MOCK_PORT}",
    )

    async with ReltioClient(config) as client:
        entity = await client.get_entity("abc123")
        print(f"      Entity: {entity.get('label', 'Unknown')}")
        print(f"      URI:    {entity.get('uri', '?')}")
        print(f"      Type:   {entity.get('type', '?')}")


async def test_file_to_match_pipeline():
    """Test full pipeline: parse file -> score against mock entities."""
    print("\n  [4] Testing file-to-match pipeline...")

    # Parse the sample CSV
    parser = FileParser()
    csv_path = PROJECT_DIR / "sample_data" / "hcp_sample.csv"
    parsed = parser.parse_file(str(csv_path))

    print(f"      Parsed {parsed.total_records} records from {csv_path.name}")

    # Score first 3 records against mock entity data
    mock_entity_attrs = {
        "NPI": [{"value": "1234567890", "ov": True}],
        "FirstName": [{"value": "John", "ov": True}],
        "LastName": [{"value": "Smith", "ov": True}],
        "Specialty": [{"value": "Cardiology", "ov": True}],
    }

    print(f"      Scoring first 3 records against mock entity (John Smith):\n")

    for record in parsed.records[:3]:
        input_attrs = {**record.identifiers, **record.attributes}
        score, matched, reasons = MatchScorer.score_match(input_attrs, mock_entity_attrs)
        confidence = MatchScorer.determine_confidence(score, reasons)

        name = f"{record.attributes.get('FirstName', '?')} {record.attributes.get('LastName', '?')}"
        print(f"        Row {record.row_number}: {name}")
        print(f"          Score: {score:.1f}%, Confidence: {confidence.value}")
        print(f"          Matched: {list(matched.keys())}")
        print()


async def test_dry_run_merge():
    """Test dry-run merge through mock server."""
    print("\n  [5] Testing dry-run merge...")

    config = ReltioConfig(
        client_id="test",
        client_secret="test",
        tenant_id="test-tenant",
        environment=f"localhost:{MOCK_PORT}",
        auth_server=f"http://localhost:{MOCK_PORT}",
    )

    async with ReltioClient(config) as client:
        result = await client.merge_entities(
            winner_id="entities/abc123",
            loser_id="entities/def456",
        )
        print(f"      Merge result: {json.dumps(result, indent=8).strip()}")


async def main():
    print("\n" + "=" * 60)
    print("  BULK MERGE AGENT - INTEGRATION TEST (MOCK SERVER)")
    print("=" * 60)

    connected = await test_mock_connectivity()
    if not connected:
        print("\n  FAILED: Cannot connect to mock server.")
        print("  Start it first: python local_test/mock_reltio_server.py")
        sys.exit(1)

    await test_entity_search()
    await test_get_entity()
    await test_file_to_match_pipeline()
    await test_dry_run_merge()

    print("\n" + "=" * 60)
    print("  ALL INTEGRATION TESTS PASSED")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
