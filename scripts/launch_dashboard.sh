#!/usr/bin/env bash
# ==============================================================================
# Script: scripts/launch_dashboard.sh
# Description: Launch the Stage 5 Streamlit Interactive Dashboard locally.
# ==============================================================================

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PORT="${1:-8501}"

echo "======================================================================"
echo "Starting Explainable AI Bridge Inspection Dashboard (Stage 5)"
echo "Repository Root: $REPO_ROOT"
echo "URL:             http://localhost:$PORT"
echo "======================================================================"

python3.11 -m streamlit run app.py \
    --server.port "$PORT" \
    --server.headless true \
    --theme.base "dark"
