from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
for relative in (
    "packages/benchmarks/src",
    "packages/rl/src",
    "packages/biology/src",
    "packages/circuit/src",
    "packages/neurons/src",
):
    path = str(ROOT / relative)
    if path not in sys.path:
        sys.path.insert(0, path)
