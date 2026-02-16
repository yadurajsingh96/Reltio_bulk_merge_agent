"""Tests for the file parser module."""

import pytest
from pathlib import Path
from src.parsers.file_parser import (
    FileParser,
    FileType,
    ColumnMapper,
    ParsedRecord,
    ParsedFile,
)


class TestColumnMapper:
    """Tests for intelligent column mapping."""

    def test_normalize_column_name(self):
        assert ColumnMapper.normalize_column_name("First Name") == "first_name"
        assert ColumnMapper.normalize_column_name("  NPI  ") == "npi"
        assert ColumnMapper.normalize_column_name("Phone-Number") == "phone_number"

    def test_map_standard_columns(self):
        assert ColumnMapper.map_column("npi") == "NPI"
        assert ColumnMapper.map_column("first_name") == "FirstName"
        assert ColumnMapper.map_column("lastname") == "LastName"
        assert ColumnMapper.map_column("specialty") == "Specialty"
        assert ColumnMapper.map_column("zip") == "PostalCode"
        assert ColumnMapper.map_column("entity_uri") == "EntityURI"

    def test_map_variant_columns(self):
        assert ColumnMapper.map_column("provider_npi") == "NPI"
        assert ColumnMapper.map_column("given_name") == "FirstName"
        assert ColumnMapper.map_column("surname") == "LastName"
        assert ColumnMapper.map_column("postal_code") == "PostalCode"
        assert ColumnMapper.map_column("merge_with") == "MergeTarget"

    def test_unknown_column_returns_none(self):
        assert ColumnMapper.map_column("random_column_xyz") is None

    def test_auto_map_columns(self):
        columns = ["NPI", "FirstName", "LastName", "custom_field"]
        mapping = ColumnMapper.auto_map_columns(columns)
        assert mapping["NPI"] == "NPI"
        assert mapping["FirstName"] == "FirstName"
        assert mapping["LastName"] == "LastName"
        assert mapping["custom_field"] == "custom_field"  # Kept as-is

    def test_is_identifier(self):
        assert ColumnMapper.is_identifier("NPI") is True
        assert ColumnMapper.is_identifier("EntityURI") is True
        assert ColumnMapper.is_identifier("DEA") is True
        assert ColumnMapper.is_identifier("FirstName") is False

    def test_is_merge_target(self):
        assert ColumnMapper.is_merge_target("MergeTarget") is True
        assert ColumnMapper.is_merge_target("FirstName") is False


class TestFileTypeDetection:
    """Tests for file type detection."""

    def test_detect_csv(self):
        parser = FileParser()
        assert parser.detect_file_type("data.csv") == FileType.CSV

    def test_detect_excel(self):
        parser = FileParser()
        assert parser.detect_file_type("data.xlsx") == FileType.EXCEL
        assert parser.detect_file_type("data.xls") == FileType.EXCEL

    def test_detect_json(self):
        parser = FileParser()
        assert parser.detect_file_type("data.json") == FileType.JSON

    def test_detect_unknown(self):
        parser = FileParser()
        assert parser.detect_file_type("data.txt") == FileType.UNKNOWN


class TestCSVParsing:
    """Tests for CSV file parsing using sample data."""

    def test_parse_csv_file(self, sample_csv_path):
        parser = FileParser()
        result = parser.parse_file(sample_csv_path)

        assert isinstance(result, ParsedFile)
        assert result.file_type == FileType.CSV
        assert result.total_records == 10
        assert result.valid_records == 10
        assert result.invalid_records == 0

    def test_csv_columns_detected(self, sample_csv_path):
        parser = FileParser()
        result = parser.parse_file(sample_csv_path)

        assert "NPI" in result.detected_columns
        assert "FirstName" in result.detected_columns
        assert "LastName" in result.detected_columns

    def test_csv_records_have_identifiers(self, sample_csv_path):
        parser = FileParser()
        result = parser.parse_file(sample_csv_path)

        for record in result.records:
            assert "NPI" in record.identifiers
            assert "EntityURI" in record.identifiers

    def test_csv_records_have_attributes(self, sample_csv_path):
        parser = FileParser()
        result = parser.parse_file(sample_csv_path)

        first_record = result.records[0]
        assert "FirstName" in first_record.attributes
        assert "LastName" in first_record.attributes
        assert "Specialty" in first_record.attributes

    def test_csv_npi_validation(self, sample_csv_path):
        parser = FileParser()
        result = parser.parse_file(sample_csv_path)

        for record in result.records:
            npi = record.identifiers.get("NPI", "")
            assert len(str(npi)) == 10


class TestJSONParsing:
    """Tests for JSON file parsing using sample data."""

    def test_parse_json_file(self, sample_json_path):
        parser = FileParser()
        result = parser.parse_file(sample_json_path)

        assert isinstance(result, ParsedFile)
        assert result.file_type == FileType.JSON
        assert result.total_records == 5
        assert result.valid_records == 5

    def test_json_records_have_identifiers(self, sample_json_path):
        parser = FileParser()
        result = parser.parse_file(sample_json_path)

        for record in result.records:
            assert "NPI" in record.identifiers
            assert "EntityURI" in record.identifiers


class TestCustomMapping:
    """Tests for custom column mappings."""

    def test_custom_mapping_overrides_auto(self, sample_csv_path):
        custom = {"Specialty": "CustomSpecialty"}
        parser = FileParser(column_mapping=custom)
        result = parser.parse_file(sample_csv_path)

        assert result.column_mapping.get("Specialty") == "CustomSpecialty"


class TestFileNotFound:
    """Tests for error handling."""

    def test_missing_file_raises_error(self):
        parser = FileParser()
        with pytest.raises(FileNotFoundError):
            parser.parse_file("/nonexistent/file.csv")

    def test_unsupported_file_type(self, tmp_path):
        txt_file = tmp_path / "data.txt"
        txt_file.write_text("some content")
        parser = FileParser()
        with pytest.raises(ValueError, match="Unsupported"):
            parser.parse_file(str(txt_file))
