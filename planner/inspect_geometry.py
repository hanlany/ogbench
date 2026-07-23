"""Inspect the AntMaze grid-center coordinate transform on real data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ogbench import make_env_and_datasets

from .collision import MazeGeometry, save_geometry_overlay


def run(
    dataset_path: str | Path, output_path: str | Path, *, seed: int = 0, sample_count: int = 5000
) -> dict[str, object]:
    dataset_path = Path(dataset_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    if sample_count <= 0:
        raise ValueError('sample_count must be positive.')
    with np.load(dataset_path, allow_pickle=False) as data:
        observations = np.asarray(data['observations'], dtype=np.float32)
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(observations), size=min(sample_count, len(observations)), replace=False))
    env = make_env_and_datasets('antmaze-large-navigate-v0', dataset_path=str(dataset_path), env_only=True)
    try:
        geometry = MazeGeometry.from_env(env, clearance=0.0)
        points = observations[indices, :2]
        accepted = np.asarray([geometry.point_valid(point) for point in points], dtype=bool)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_geometry_overlay(geometry, points, str(output_path), accepted=accepted)
        summary = {
            'dataset_path': str(dataset_path),
            'output_path': str(output_path),
            'seed': int(seed),
            'sample_count': int(len(points)),
            'accepted_count': int(np.sum(accepted)),
            'rejected_count': int(np.sum(~accepted)),
            'maze_shape': list(geometry.shape),
            'maze_unit': geometry.maze_unit,
            'offset_x': geometry.offset_x,
            'offset_y': geometry.offset_y,
            'clearance': geometry.clearance,
        }
    finally:
        env.close()
    summary_path = output_path.with_suffix('.json')
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset_path', required=True)
    parser.add_argument('--output_path', required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--sample_count', type=int, default=5000)
    args = parser.parse_args()
    run(**vars(args))


if __name__ == '__main__':
    main()
