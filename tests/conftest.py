"""Shared fixtures for Bulk Merge Agent tests."""

import sys
import os
import pytest
from pathlib import Path

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def sample_csv_path():
    """Path to sample CSV file."""
    return str(Path(__file__).parent.parent / "sample_data" / "hcp_sample.csv")


@pytest.fixture
def sample_json_path():
    """Path to sample JSON file."""
    return str(Path(__file__).parent.parent / "sample_data" / "hcp_sample.json")


@pytest.fixture
def sample_hcp_record():
    """A single HCP record as a dict (like from CSV)."""
    return {
        "NPI": "1234567890",
        "FirstName": "John",
        "LastName": "Smith",
        "Specialty": "Cardiology",
        "Address": "123 Main St",
        "City": "Boston",
        "State": "MA",
        "Zip": "02101",
        "Phone": "617-555-0100",
        "EntityURI": "entities/abc123",
    }


@pytest.fixture
def sample_reltio_entity():
    """A mock Reltio entity response."""
    return {
        "uri": "entities/abc123",
        "type": "configuration/entityTypes/HCP",
        "label": "John Smith",
        "attributes": {
            "NPI": [{"value": "1234567890", "ov": True}],
            "FirstName": [{"value": "John", "ov": True}],
            "LastName": [{"value": "Smith", "ov": True}],
            "Specialty": [{"value": "Cardiology", "ov": True}],
            "City": [{"value": "Boston", "ov": True}],
            "State": [{"value": "MA", "ov": True}],
        },
    }


@pytest.fixture
def sample_reltio_entity_different():
    """A different Reltio entity (partial match)."""
    return {
        "uri": "entities/xyz789",
        "type": "configuration/entityTypes/HCP",
        "label": "Jon Smithe",
        "attributes": {
            "NPI": [{"value": "9999999999", "ov": True}],
            "FirstName": [{"value": "Jon", "ov": True}],
            "LastName": [{"value": "Smithe", "ov": True}],
            "Specialty": [{"value": "Cardiology", "ov": True}],
            "City": [{"value": "Boston", "ov": True}],
            "State": [{"value": "MA", "ov": True}],
        },
    }
