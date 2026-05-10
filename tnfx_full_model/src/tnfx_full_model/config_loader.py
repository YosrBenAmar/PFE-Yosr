from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ProjectConfig:
    root: Path
    model: dict[str, Any]
    run: dict[str, Any]
    market: dict[str, Any]
    validation: dict[str, Any]

    @property
    def output_dir(self) -> Path:
        path = self.root / self.run["output_dir"]
        path.mkdir(parents=True, exist_ok=True)
        return path


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_project_config(root: str | Path | None = None) -> ProjectConfig:
    project_root = Path(root or Path.cwd()).resolve()
    if project_root.name != "tnfx_full_model" and (project_root / "tnfx_full_model").exists():
        project_root = project_root / "tnfx_full_model"
    config_dir = project_root / "config"
    return ProjectConfig(
        root=project_root,
        model=load_yaml(config_dir / "model_config.yaml"),
        run=load_yaml(config_dir / "run_config.yaml"),
        market=load_yaml(config_dir / "market_data_config.yaml"),
        validation=load_yaml(config_dir / "validation_config.yaml"),
    )

