"""Convenience launcher: run Streamlit UI from project root."""

import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    frontend = Path(__file__).parent / "frontend"
    sys.exit(
        subprocess.call(
            [sys.executable, "-m", "streamlit", "run", "streamlit_app.py"],
            cwd=frontend,
        )
    )
