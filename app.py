"""
Root entrypoint for Streamlit dashboard.
Forwards execution to src.dashboard.app.

Usage:
    streamlit run app.py
    # or: ./scripts/launch_dashboard.sh
    # or: python3.11 -m streamlit run app.py
"""

import os
import shutil
import sys

# PyTorch, Torchvision, and OpenCV are installed in the Python 3.11 environment.
# If launched via a global Python 3.12+ (e.g. macOS Python 3.13) Streamlit binary,
# seamlessly re-execute under Python 3.11 where the computer vision stack is available.
if sys.version_info >= (3, 12):
    py311 = (
        shutil.which("python3.11")
        or "/usr/local/bin/python3.11"
        or "/usr/local/opt/python@3.11/bin/python3.11"
    )
    if py311 and os.path.exists(py311) and os.environ.get("_BRIDGE_AI_REEXEC") != "1":
        os.environ["_BRIDGE_AI_REEXEC"] = "1"
        print(f"[BridgeAI] Python {sys.version.split()[0]} detected. Re-executing under Python 3.11 ({py311}) with PyTorch and OpenCV...")
        
        # Preserve any CLI arguments passed to streamlit
        extra_args = []
        orig = getattr(sys, "orig_argv", sys.argv)
        if "run" in orig:
            run_idx = orig.index("run")
            if run_idx + 2 < len(orig):
                extra_args = orig[run_idx + 2:]
        elif len(sys.argv) > 1:
            extra_args = sys.argv[1:]

        target_cmd = [py311, "-m", "streamlit", "run", os.path.abspath(__file__)] + extra_args
        os.execv(py311, target_cmd)
    elif not (py311 and os.path.exists(py311)):
        print(
            f"⚠️ Warning: Running on Python {sys.version.split()[0]}, but PyTorch/OpenCV requires Python 3.11.\n"
            "Please ensure Python 3.11 is installed or launch via: python3.11 -m streamlit run app.py",
            file=sys.stderr,
        )

from src.dashboard.app import main

if __name__ == "__main__":
    main()
