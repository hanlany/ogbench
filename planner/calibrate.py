"""Create versioned Phase 1 metric and cycle-error calibration artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .artifacts import (
    LatentModelAdapter,
    cache_identity,
    calibrate_cycle_error,
    calibrate_metrics,
    load_planner_cache,
)


def run_calibration(
    checkpoint_path: str | Path,
    dataset_path: str | Path,
    cache_path: str | Path,
    output_dir: str | Path,
    *,
    max_edge_horizon: int,
    latent_weight: float = 1.0,
) -> Path:
    """Calibrate only from held-out cache queries and write one JSON artifact."""
    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f'Refusing to mix calibration artifacts into non-empty directory: {output_dir}.')
    output_dir.mkdir(parents=True, exist_ok=False)
    adapter = LatentModelAdapter.from_checkpoint(checkpoint_path, max_horizon=max_edge_horizon)
    cache = load_planner_cache(cache_path, expected=cache_identity(adapter, dataset_path, max_edge_horizon))
    adapter.latent_mean = np.asarray(cache.metadata['latent_mean'], dtype=np.float32)
    adapter.latent_std = np.asarray(cache.metadata['latent_std'], dtype=np.float32)
    artifact = {
        'schema_version': 1,
        'checkpoint_path': str(Path(checkpoint_path).expanduser().resolve()),
        'dataset_path': str(Path(dataset_path).expanduser().resolve()),
        'cache_path': str(Path(cache_path).expanduser().resolve()),
        'cache_schema_version': cache.metadata['schema_version'],
        'max_edge_horizon': int(max_edge_horizon),
        'latent_weight': float(latent_weight),
        'metrics': calibrate_metrics(cache, latent_weight=latent_weight),
        'cycle_error': calibrate_cycle_error(adapter, cache),
    }
    path = output_dir / 'calibration_v1.json'
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + '\n')
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint_path', required=True)
    parser.add_argument('--dataset_path', required=True)
    parser.add_argument('--cache_path', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--max_edge_horizon', type=int, default=10)
    parser.add_argument('--latent_weight', type=float, default=1.0)
    args = parser.parse_args()
    run_calibration(**vars(args))


if __name__ == '__main__':
    main()
