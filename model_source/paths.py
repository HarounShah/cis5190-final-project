"""Repository root and shared paths (works regardless of current working directory)."""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = REPO_ROOT / "checkpoints"
