import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tnfx_full_model.config_loader import load_project_config


@pytest.fixture(scope="session")
def config():
    return load_project_config(ROOT)


@pytest.fixture()
def small_run(config):
    run = dict(config.run)
    run["target_profiles"] = 40
    run["pilot_points"] = 64
    return run

