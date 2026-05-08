from __future__ import annotations

import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parents[1]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "packages" / "circuit" / "src"))
sys.path.insert(0, str(REPO_ROOT / "packages" / "neurons" / "src"))
