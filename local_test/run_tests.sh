#!/usr/bin/env bash
# =============================================================================
# Bulk Merge Agent - Local Test Runner
# Run this script to set up and test the application locally
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "================================================"
echo "  Bulk Merge Agent - Local Test Runner"
echo "================================================"
echo ""
echo "Project: $PROJECT_DIR"
echo ""

# Step 1: Create virtual environment if needed
if [ ! -d "venv" ]; then
    echo "[1/5] Creating virtual environment..."
    python3 -m venv venv
else
    echo "[1/5] Virtual environment exists"
fi

# Activate
source venv/bin/activate

# Step 2: Install dependencies
echo "[2/5] Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt
pip install -q -e ".[dev]" 2>/dev/null || pip install -q pytest pytest-asyncio pytest-cov

# Step 3: Run unit tests
echo ""
echo "[3/5] Running unit tests..."
echo "----------------------------------------------"
PYTHONPATH="$PROJECT_DIR" python -m pytest tests/ -v --tb=short 2>&1
echo ""

# Step 4: Run local functionality demo
echo "[4/5] Running local functionality demo..."
echo "----------------------------------------------"
PYTHONPATH="$PROJECT_DIR" python "$SCRIPT_DIR/demo_local.py"
echo ""

# Step 5: Show how to start the UI
echo "[5/5] All tests passed!"
echo ""
echo "================================================"
echo "  Next Steps"
echo "================================================"
echo ""
echo "  Start the Streamlit UI:"
echo "    cd $PROJECT_DIR"
echo "    source venv/bin/activate"
echo "    streamlit run src/ui/streamlit_app.py"
echo ""
echo "  Run CLI (requires Reltio credentials in .env):"
echo "    python -m src.cli health"
echo "    python -m src.cli analyze --file sample_data/hcp_sample.csv"
echo ""
echo "  Run mock server + integration test:"
echo "    python local_test/mock_reltio_server.py &"
echo "    python local_test/demo_integration.py"
echo ""
