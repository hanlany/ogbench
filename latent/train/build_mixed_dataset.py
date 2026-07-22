from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def episode_slices(terminals: np.ndarray) -> list[tuple[int, int]]:
    """Return half-open episode slices from terminal-marked OGBench data."""
    terminals = np.asarray(terminals).reshape(-1)
    if len(terminals) == 0:
        raise ValueError('Dataset must be non-empty.')
    ends = np.flatnonzero(terminals > 0)
    if len(ends) == 0 or ends[-1] != len(terminals) - 1:
        raise ValueError('Dataset must mark the final transition as terminal.')
    starts = np.concatenate(([0], ends[:-1] + 1))
    return [(int(start), int(end + 1)) for start, end in zip(starts, ends)]


def validate_dataset(data: dict[str, np.ndarray], name: str) -> list[tuple[int, int]]:
    """Validate aligned fields and episode boundaries."""
    required = {'observations', 'actions', 'terminals'}
    missing = sorted(required - data.keys())
    if missing:
        raise ValueError(f'{name} dataset is missing keys: {missing}.')
    size = len(data['observations'])
    for key, value in data.items():
        if len(value) != size:
            raise ValueError(f'{name} field {key!r} has length {len(value)}, expected {size}.')
    return episode_slices(data['terminals'])


def select_episode_ids(
    slices: list[tuple[int, int]],
    target_transitions: float,
    seed: int,
) -> np.ndarray:
    """Select a deterministic random episode subset closest to a transition target."""
    if target_transitions <= 0:
        raise ValueError('target_transitions must be positive.')
    lengths = np.asarray([end - start for start, end in slices], dtype=np.int64)
    order = np.random.default_rng(seed).permutation(len(slices))
    cumulative = np.cumsum(lengths[order])
    count = int(np.searchsorted(cumulative, target_transitions, side='left')) + 1
    count = min(count, len(order))
    if count > 1:
        with_last = abs(float(cumulative[count - 1]) - target_transitions)
        without_last = abs(float(cumulative[count - 2]) - target_transitions)
        if without_last <= with_last:
            count -= 1
    return np.sort(order[:count]).astype(np.int64)


def build_mixture(
    primary: dict[str, np.ndarray],
    secondary: dict[str, np.ndarray],
    secondary_fraction: float,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Mix all primary episodes with sampled secondary episodes and shuffle episodes."""
    if not 0 < secondary_fraction < 1:
        raise ValueError('secondary_fraction must be in (0, 1).')
    if set(primary) != set(secondary):
        raise ValueError(
            f'Dataset keys must match; primary-only={sorted(set(primary) - set(secondary))}, '
            f'secondary-only={sorted(set(secondary) - set(primary))}.'
        )
    primary_slices = validate_dataset(primary, 'Primary')
    secondary_slices = validate_dataset(secondary, 'Secondary')
    for key in primary:
        if primary[key].shape[1:] != secondary[key].shape[1:]:
            raise ValueError(
                f'Field {key!r} shapes are incompatible: {primary[key].shape} versus {secondary[key].shape}.'
            )

    primary_transitions = len(primary['observations'])
    secondary_target = primary_transitions * secondary_fraction / (1.0 - secondary_fraction)
    secondary_ids = select_episode_ids(secondary_slices, secondary_target, seed)
    episode_refs = [(0, index) for index in range(len(primary_slices))]
    episode_refs.extend((1, int(index)) for index in secondary_ids)
    np.random.default_rng(seed + 1).shuffle(episode_refs)

    mixed = {}
    for key in primary:
        sources = (primary[key], secondary[key])
        slices = (primary_slices, secondary_slices)
        mixed[key] = np.concatenate(
            [sources[source][slice(*slices[source][index])] for source, index in episode_refs],
            axis=0,
        )

    secondary_transitions = sum(secondary_slices[index][1] - secondary_slices[index][0] for index in secondary_ids)
    total_transitions = primary_transitions + secondary_transitions
    manifest = {
        'seed': seed,
        'requested_secondary_fraction': secondary_fraction,
        'actual_secondary_fraction': secondary_transitions / total_transitions,
        'primary': {
            'episodes': len(primary_slices),
            'transitions': primary_transitions,
        },
        'secondary': {
            'available_episodes': len(secondary_slices),
            'selected_episode_ids': secondary_ids.tolist(),
            'selected_episodes': len(secondary_ids),
            'transitions': secondary_transitions,
        },
        'mixture': {
            'episodes': len(episode_refs),
            'transitions': total_transitions,
        },
    }
    return mixed, manifest


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def validation_path(path: Path) -> Path:
    if path.suffix != '.npz':
        raise ValueError(f'Expected an .npz path, got {path}.')
    return path.with_name(f'{path.stem}-val.npz')


def write_split(
    primary_path: Path,
    secondary_path: Path,
    output_path: Path,
    secondary_fraction: float,
    seed: int,
) -> dict[str, Any]:
    primary = load_npz(primary_path)
    secondary = load_npz(secondary_path)
    mixed, manifest = build_mixture(primary, secondary, secondary_fraction, seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **mixed)
    manifest.update(
        {
            'primary_path': str(primary_path.resolve()),
            'secondary_path': str(secondary_path.resolve()),
            'output_path': str(output_path.resolve()),
        }
    )
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Build an episode-safe mixture of two OGBench datasets.')
    parser.add_argument('--primary_path', type=Path, required=True)
    parser.add_argument('--secondary_path', type=Path, required=True)
    parser.add_argument('--output_path', type=Path, required=True)
    parser.add_argument('--secondary_fraction', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=0)
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    manifests = {
        'train': write_split(
            args.primary_path,
            args.secondary_path,
            args.output_path,
            args.secondary_fraction,
            args.seed,
        ),
        'validation': write_split(
            validation_path(args.primary_path),
            validation_path(args.secondary_path),
            validation_path(args.output_path),
            args.secondary_fraction,
            args.seed + 1,
        ),
    }
    manifest_path = args.output_path.with_suffix('.manifest.json')
    manifest_path.write_text(json.dumps(manifests, indent=2, sort_keys=True) + '\n')
    print(f'Wrote mixed training dataset to {args.output_path}')
    print(f'Wrote mixed validation dataset to {validation_path(args.output_path)}')
    print(f'Wrote mixture manifest to {manifest_path}')


if __name__ == '__main__':
    main()
