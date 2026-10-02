"""Smoke training stage: writes a 'model' whose only weight is the recipe's boost.

Usage: train.py <run_dir> <inputs>
"""

import json
import sys
from pathlib import Path

run_dir, inputs = Path(sys.argv[1]), Path(sys.argv[2])
recipe = json.loads((inputs / "recipe.json").read_text())
model = run_dir / "outputs" / "model"
model.mkdir(parents=True, exist_ok=True)
(model / "weights.json").write_text(json.dumps({"boost": float(recipe.get("boost", 0.0))}))
print(f"trained smoke model with boost={recipe.get('boost', 0.0)}")
