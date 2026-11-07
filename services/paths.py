"""Absolute runtime paths; deployments and tests may isolate mutable state."""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("TAX_AGENT_DATA_DIR", PROJECT_ROOT / "data")).resolve()
UPLOAD_DIR = Path(os.environ.get("TAX_AGENT_UPLOAD_DIR", PROJECT_ROOT / "uploads")).resolve()
