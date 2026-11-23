"""
Root entrypoint for Streamlit dashboard.
Forwards execution to src.dashboard.app.

Usage:
    streamlit run app.py
"""

from src.dashboard.app import main

if __name__ == "__main__":
    main()
