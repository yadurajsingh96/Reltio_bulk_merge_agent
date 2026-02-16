# Local Testing Guide - Bulk Merge Agent

## Quick Start

```bash
# One-command test (sets up venv, installs deps, runs all tests)
./local_test/run_tests.sh
```

## What Gets Tested

### 1. Unit Tests (`tests/`)
Run with `pytest`:
```bash
PYTHONPATH=. pytest tests/ -v
```

| Test File | What It Tests | External Deps |
|-----------|--------------|---------------|
| `test_file_parser.py` | CSV/JSON parsing, column mapping, NPI validation | None |
| `test_match_scorer.py` | Similarity scoring, confidence levels, attribute weights | None |
| `test_merge_executor.py` | Operation preparation, report generation | None |

### 2. Local Demo (`local_test/demo_local.py`)
Demonstrates all features that work WITHOUT Reltio credentials:
```bash
PYTHONPATH=. python local_test/demo_local.py
```

- Column mapping intelligence
- File parsing (CSV + JSON)
- Match scoring engine
- Merge operation preparation
- Report generation

### 3. Integration Test with Mock Server

Start the mock Reltio API server:
```bash
python local_test/mock_reltio_server.py
```

In another terminal, run the integration test:
```bash
PYTHONPATH=. python local_test/demo_integration.py
```

This tests:
- OAuth token flow
- Entity search
- Entity retrieval
- File-to-match scoring pipeline
- Dry-run merge execution

## Mock Reltio Server

The mock server (`mock_reltio_server.py`) simulates Reltio API endpoints:

| Endpoint | Description |
|----------|-------------|
| `POST /oauth/token` | Returns mock bearer token |
| `POST /entities/_search` | Returns mock entity search results |
| `GET /entities/{id}` | Returns single mock entity |
| `GET /entities/{id}/_matches` | Returns mock match candidates |
| `POST /entities/{id}/_sameAs` | Simulates merge operation |

Comes pre-loaded with 3 mock HCP entities (John Smith, Jane Doe, Robert Johnson).

## Testing the Streamlit UI

```bash
# Install deps (if not already done)
pip install -r requirements.txt

# Start the UI
PYTHONPATH=. streamlit run src/ui/streamlit_app.py
```

The UI will open at http://localhost:8501. You can:
- Upload the sample CSV from `sample_data/`
- View the parsing results
- Configure thresholds (won't connect to Reltio without real credentials)
