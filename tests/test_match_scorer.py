"""Tests for the match scoring engine (no API calls required)."""

import pytest
from src.core.match_analyzer import (
    MatchScorer,
    MatchConfidence,
    MatchReason,
)


class TestSimilarityCalculation:
    """Tests for string similarity calculation."""

    def test_exact_match(self):
        assert MatchScorer.calculate_similarity("John", "John") == 1.0

    def test_case_insensitive_match(self):
        assert MatchScorer.calculate_similarity("JOHN", "john") == 1.0

    def test_fuzzy_match_high(self):
        score = MatchScorer.calculate_similarity("John", "Jon")
        assert score > 0.7

    def test_fuzzy_match_low(self):
        score = MatchScorer.calculate_similarity("John", "Robert")
        assert score < 0.5

    def test_none_values(self):
        assert MatchScorer.calculate_similarity(None, "John") == 0.0
        assert MatchScorer.calculate_similarity("John", None) == 0.0
        assert MatchScorer.calculate_similarity(None, None) == 0.0

    def test_empty_strings(self):
        assert MatchScorer.calculate_similarity("", "John") == 0.0
        assert MatchScorer.calculate_similarity("John", "") == 0.0

    def test_whitespace_handling(self):
        assert MatchScorer.calculate_similarity(" John ", "John") == 1.0


class TestMatchScoring:
    """Tests for weighted match scoring."""

    def test_exact_npi_match(self):
        input_attrs = {"NPI": "1234567890", "FirstName": "John"}
        reltio_attrs = {
            "NPI": [{"value": "1234567890", "ov": True}],
            "FirstName": [{"value": "John", "ov": True}],
        }
        score, matched, reasons = MatchScorer.score_match(input_attrs, reltio_attrs)
        assert score > 90
        assert MatchReason.NPI_EXACT in reasons

    def test_name_only_match(self):
        input_attrs = {"FirstName": "John", "LastName": "Smith"}
        reltio_attrs = {
            "FirstName": [{"value": "John", "ov": True}],
            "LastName": [{"value": "Smith", "ov": True}],
        }
        score, matched, reasons = MatchScorer.score_match(input_attrs, reltio_attrs)
        assert score > 80
        assert MatchReason.NAME_EXACT in reasons

    def test_fuzzy_name_match(self):
        input_attrs = {"FirstName": "John", "LastName": "Smith"}
        reltio_attrs = {
            "FirstName": [{"value": "Jon", "ov": True}],
            "LastName": [{"value": "Smithe", "ov": True}],
        }
        score, matched, reasons = MatchScorer.score_match(input_attrs, reltio_attrs)
        assert score > 50

    def test_no_match(self):
        input_attrs = {"FirstName": "John", "LastName": "Smith"}
        reltio_attrs = {
            "FirstName": [{"value": "Maria", "ov": True}],
            "LastName": [{"value": "Garcia", "ov": True}],
        }
        score, matched, reasons = MatchScorer.score_match(input_attrs, reltio_attrs)
        assert score < 60

    def test_empty_reltio_attrs(self):
        input_attrs = {"NPI": "1234567890"}
        reltio_attrs = {}
        score, matched, reasons = MatchScorer.score_match(input_attrs, reltio_attrs)
        assert score == 0


class TestFlattenReltioAttrs:
    """Tests for Reltio attribute flattening."""

    def test_simple_attribute(self):
        attrs = {"NPI": [{"value": "1234567890", "ov": True}]}
        flat = MatchScorer._flatten_reltio_attrs(attrs)
        assert flat["NPI"] == "1234567890"

    def test_single_value_attribute(self):
        attrs = {"City": [{"value": "Boston"}]}
        flat = MatchScorer._flatten_reltio_attrs(attrs)
        assert flat["City"] == "Boston"

    def test_nested_attribute(self):
        attrs = {
            "Address": [{
                "value": {
                    "City": [{"value": "Boston"}],
                    "State": [{"value": "MA"}],
                },
                "ov": True,
            }]
        }
        flat = MatchScorer._flatten_reltio_attrs(attrs)
        assert flat.get("City") == "Boston"
        assert flat.get("State") == "MA"

    def test_empty_attrs(self):
        assert MatchScorer._flatten_reltio_attrs({}) == {}
        assert MatchScorer._flatten_reltio_attrs(None) == {}


class TestConfidenceDetermination:
    """Tests for confidence level determination."""

    def test_exact_confidence_npi(self):
        confidence = MatchScorer.determine_confidence(
            100, [MatchReason.NPI_EXACT]
        )
        assert confidence == MatchConfidence.EXACT

    def test_high_confidence(self):
        confidence = MatchScorer.determine_confidence(92, [MatchReason.NAME_EXACT])
        assert confidence == MatchConfidence.HIGH

    def test_medium_confidence(self):
        confidence = MatchScorer.determine_confidence(75, [MatchReason.NAME_FUZZY])
        assert confidence == MatchConfidence.MEDIUM

    def test_low_confidence(self):
        confidence = MatchScorer.determine_confidence(55, [])
        assert confidence == MatchConfidence.LOW

    def test_uncertain_confidence(self):
        confidence = MatchScorer.determine_confidence(30, [])
        assert confidence == MatchConfidence.UNCERTAIN

    def test_no_match_confidence(self):
        confidence = MatchScorer.determine_confidence(0, [])
        assert confidence == MatchConfidence.NO_MATCH


class TestCompositeMatching:
    """Tests for composite (multi-attribute) matching."""

    def test_composite_reason_added(self):
        input_attrs = {
            "FirstName": "John",
            "LastName": "Smith",
            "Specialty": "Cardiology",
            "City": "Boston",
        }
        reltio_attrs = {
            "FirstName": [{"value": "John", "ov": True}],
            "LastName": [{"value": "Smith", "ov": True}],
            "Specialty": [{"value": "Cardiology", "ov": True}],
            "City": [{"value": "Boston", "ov": True}],
        }
        score, matched, reasons = MatchScorer.score_match(input_attrs, reltio_attrs)
        assert MatchReason.COMPOSITE in reasons
        assert len(matched) >= 3
