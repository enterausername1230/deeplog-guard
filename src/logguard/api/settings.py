"""Backend configuration, resolved from the repository rather than the caller's cwd."""
from pathlib import Path
import os
import yaml

ROOT = Path(__file__).resolve().parents[3]


def load_settings(path=None):
    path = Path(path or os.environ.get('LOGGUARD_API_CONFIG', ROOT / 'configs/api.yaml'))
    with path.open(encoding='utf-8') as stream:
        settings = yaml.safe_load(stream)
    for key in ('database', 'drain_config', 'drain_state', 'vocab'):
        settings[key] = ROOT / settings[key]
    return settings
