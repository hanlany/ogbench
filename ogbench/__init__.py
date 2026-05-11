"""OGBench: Benchmarking Offline Goal-Conditioned RL"""

import sys
import types
from pathlib import Path

import ogbench.locomaze as locomaze
import ogbench.manipspace as manipspace
import ogbench.powderworld as powderworld
from ogbench.utils import download_datasets, load_dataset, make_env_and_datasets

_LOCAL_DIR = Path(__file__).resolve().parents[1] / 'local'
if _LOCAL_DIR.exists() and 'ogbench.local' not in sys.modules:
    local_module = types.ModuleType('ogbench.local')
    local_module.__path__ = [str(_LOCAL_DIR)]
    sys.modules['ogbench.local'] = local_module

__all__ = (
    'locomaze',
    'manipspace',
    'powderworld',
    'download_datasets',
    'load_dataset',
    'make_env_and_datasets',
)
