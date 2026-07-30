"""Config loading and environment capture.

Every artefact this project writes is accompanied by the config that produced it and
by a provenance record of the exact software stack, so a result can never be read
without knowing which revision of which model produced it.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"
OUTPUT_DIR = REPO_ROOT / "outputs"


def load_config(name: str) -> dict[str, Any]:
    """Load configs/<name>.yaml (the .yaml suffix is optional)."""
    path = Path(name)
    if not path.suffix:
        path = CONFIG_DIR / f"{name}.yaml"
    elif not path.is_absolute():
        path = CONFIG_DIR / path.name
    with open(path) as fh:
        return yaml.safe_load(fh)


def _git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def provenance() -> dict[str, Any]:
    """Everything needed to re-create this run's software and hardware context."""
    import numpy
    import rdkit
    import torch
    import transformers

    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "torch": torch.__version__,
        "torch_num_threads": torch.get_num_threads(),
        "mps_available": bool(torch.backends.mps.is_available()),
        "cuda_available": bool(torch.cuda.is_available()),
        "transformers": transformers.__version__,
        "rdkit": rdkit.__version__,
        "numpy": numpy.__version__,
        "git_sha": _git_sha(),
    }


def write_run_context(out_dir: Path, configs: dict[str, Any] | None = None) -> None:
    """Drop a provenance record, and any configs, next to a result.

    Every script must call this, so that no output directory can be read without
    the software stack and settings that produced it.  `RunDir` does the same for
    the scripts that create their directory through it.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "provenance.json", provenance())
    if configs:
        write_json(out_dir / "configs_used.json", configs)


@dataclass
class RunDir:
    """A run directory that always carries its own configs and provenance."""

    path: Path

    @classmethod
    def create(cls, name: str, configs: dict[str, Any]) -> "RunDir":
        path = OUTPUT_DIR / name
        path.mkdir(parents=True, exist_ok=True)
        (path / "configs").mkdir(exist_ok=True)
        for key, cfg in configs.items():
            write_json(path / "configs" / f"{key}.json", cfg)
        write_json(path / "provenance.json", provenance())
        return cls(path)

    def __truediv__(self, other: str) -> Path:
        return self.path / other


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, default=_default)
        fh.write("\n")


def read_json(path: Path) -> Any:
    with open(path) as fh:
        return json.load(fh)


def _default(o: Any) -> Any:
    import numpy as np

    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")
