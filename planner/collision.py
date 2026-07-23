"""AntMaze geometry and learned-state validity checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class MazeGeometry:
    """Immutable grid-center approximation of a locomaze occupancy map."""

    maze_map: np.ndarray
    maze_unit: float
    offset_x: float
    offset_y: float
    clearance: float = 0.0

    def __post_init__(self) -> None:
        maze_map = np.asarray(self.maze_map)
        if maze_map.ndim != 2 or not np.all(np.isin(maze_map, (0, 1))):
            raise ValueError('maze_map must be a 2D binary occupancy array.')
        if self.maze_unit <= 0 or self.clearance < 0:
            raise ValueError('maze_unit must be positive and clearance must be non-negative.')
        object.__setattr__(self, 'maze_map', maze_map.astype(np.int8, copy=True))
        self.maze_map.setflags(write=False)

    @classmethod
    def from_env(cls, env: Any, clearance: float = 0.0) -> 'MazeGeometry':
        unwrapped = getattr(env, 'unwrapped', env)
        required = ('maze_map', '_maze_unit', '_offset_x', '_offset_y')
        missing = [name for name in required if not hasattr(unwrapped, name)]
        if missing:
            raise ValueError(f'AntMaze environment is missing geometry attributes: {missing}.')
        return cls(
            np.asarray(unwrapped.maze_map),
            float(unwrapped._maze_unit),
            float(unwrapped._offset_x),
            float(unwrapped._offset_y),
            float(clearance),
        )

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self.maze_map.shape)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        rows, cols = self.maze_map.shape
        return (
            -self.offset_x - self.maze_unit / 2,
            (cols - 1) * self.maze_unit - self.offset_x + self.maze_unit / 2,
            -self.offset_y - self.maze_unit / 2,
            (rows - 1) * self.maze_unit - self.offset_y + self.maze_unit / 2,
        )

    def xy_to_cell(self, xy: np.ndarray) -> tuple[int, int] | None:
        xy = np.asarray(xy, dtype=np.float64)
        if xy.shape != (2,) or not np.all(np.isfinite(xy)):
            return None
        i = int(np.floor((xy[1] + self.offset_y + self.maze_unit / 2) / self.maze_unit))
        j = int(np.floor((xy[0] + self.offset_x + self.maze_unit / 2) / self.maze_unit))
        if not (0 <= i < self.maze_map.shape[0] and 0 <= j < self.maze_map.shape[1]):
            return None
        return i, j

    def cell_center(self, cell: tuple[int, int]) -> np.ndarray:
        i, j = cell
        return np.asarray([j * self.maze_unit - self.offset_x, i * self.maze_unit - self.offset_y], dtype=np.float32)

    def point_valid(self, xy: np.ndarray) -> bool:
        cell = self.xy_to_cell(xy)
        if cell is None:
            return False
        return bool(self.maze_map[cell] == 0)

    @staticmethod
    def _segment_intersects_rect(start: np.ndarray, end: np.ndarray, rect: tuple[float, float, float, float]) -> bool:
        """Liang-Barsky segment/closed-rectangle intersection."""
        x0, y0 = map(float, start)
        x1, y1 = map(float, end)
        xmin, xmax, ymin, ymax = rect
        dx, dy = x1 - x0, y1 - y0
        t0, t1 = 0.0, 1.0
        for p, q in ((-dx, x0 - xmin), (dx, xmax - x0), (-dy, y0 - ymin), (dy, ymax - y0)):
            if abs(p) < 1e-12:
                if q < 0:
                    return False
                continue
            ratio = q / p
            if p < 0:
                if ratio > t1:
                    return False
                t0 = max(t0, ratio)
            else:
                if ratio < t0:
                    return False
                t1 = min(t1, ratio)
        return t0 <= t1

    def segment_valid(self, start: np.ndarray, end: np.ndarray) -> bool:
        start = np.asarray(start, dtype=np.float64)
        end = np.asarray(end, dtype=np.float64)
        if start.shape != (2,) or end.shape != (2,) or not np.all(np.isfinite([start, end])):
            return False
        if not self.point_valid(start) or not self.point_valid(end):
            return False
        half = self.maze_unit / 2 + self.clearance
        for i, j in zip(*np.where(self.maze_map == 1)):
            center = self.cell_center((int(i), int(j)))
            rect = (float(center[0] - half), float(center[0] + half), float(center[1] - half), float(center[1] + half))
            if self._segment_intersects_rect(start, end, rect):
                return False
        return True

    def point_clearance(self, xy: np.ndarray) -> float:
        """Return free-space clearance to walls and the map boundary."""
        xy = np.asarray(xy, dtype=np.float64)
        if not self.point_valid(xy):
            return 0.0
        xmin, xmax, ymin, ymax = self.bounds
        distances = [float(xy[0] - xmin), float(xmax - xy[0]), float(xy[1] - ymin), float(ymax - xy[1])]
        half = self.maze_unit / 2
        for i, j in zip(*np.where(self.maze_map == 1)):
            center = self.cell_center((int(i), int(j)))
            rect = (float(center[0] - half), float(center[0] + half), float(center[1] - half), float(center[1] + half))
            dx = max(rect[0] - xy[0], 0.0, xy[0] - rect[1])
            dy = max(rect[2] - xy[1], 0.0, xy[1] - rect[3])
            distances.append(float(np.hypot(dx, dy)))
        return float(max(0.0, min(distances) - self.clearance))

    def polyline_clearance(self, points: np.ndarray, samples_per_segment: int = 32) -> float:
        """Estimate minimum configured clearance along a polyline."""
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[-1] != 2 or len(points) == 0 or samples_per_segment <= 0:
            return 0.0
        sampled = [points[0]]
        for start, end in zip(points[:-1], points[1:]):
            sampled.extend(np.linspace(start, end, samples_per_segment + 1)[1:])
        return float(min(self.point_clearance(point) for point in sampled))

    def polyline_valid(self, points: np.ndarray) -> bool:
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[-1] != 2 or len(points) == 0:
            return False
        return (
            all(self.segment_valid(start, end) for start, end in zip(points[:-1], points[1:]))
            if len(points) > 1
            else self.point_valid(points[0])
        )


def save_geometry_overlay(
    geometry: MazeGeometry,
    points: np.ndarray,
    output_path: str,
    *,
    accepted: np.ndarray | None = None,
    title: str = 'AntMaze grid-center geometry overlay',
) -> str:
    """Render wall cells and sampled XY points for coordinate inspection."""
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[-1] != 2:
        raise ValueError('Overlay points must have shape (N, 2).')
    if accepted is None:
        accepted = np.asarray([geometry.point_valid(point) for point in points], dtype=bool)
    accepted = np.asarray(accepted, dtype=bool).reshape(-1)
    if len(accepted) != len(points):
        raise ValueError('Overlay points and accepted mask are not aligned.')
    figure, axis = plt.subplots(figsize=(10, 7), constrained_layout=True)
    half = geometry.maze_unit / 2
    for i, j in zip(*np.where(geometry.maze_map == 1)):
        center = geometry.cell_center((int(i), int(j)))
        axis.add_patch(
            Rectangle(
                (center[0] - half, center[1] - half),
                geometry.maze_unit,
                geometry.maze_unit,
                facecolor='black',
                edgecolor='black',
            )
        )
    if np.any(accepted):
        axis.scatter(points[accepted, 0], points[accepted, 1], s=5, c='tab:green', label='free-cell point')
    if np.any(~accepted):
        axis.scatter(points[~accepted, 0], points[~accepted, 1], s=5, c='tab:red', label='rejected point')
    xmin, xmax, ymin, ymax = geometry.bounds
    axis.set(xlim=(xmin, xmax), ylim=(ymin, ymax), aspect='equal', title=title, xlabel='world x', ylabel='world y')
    axis.legend(loc='upper right')
    output_path = str(output_path)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return output_path


@dataclass(frozen=True)
class ValidityResult:
    valid: np.ndarray
    reasons: np.ndarray
    cycle_errors: np.ndarray


class AntMazeValidity:
    """Validate decoded Ant predictions and every intermediate XY segment."""

    REASONS = ('non_finite', 'off_manifold', 'collision', 'invalid_decoded_state', 'valid')

    def __init__(
        self,
        adapter: Any,
        geometry: MazeGeometry,
        cycle_error_threshold: float,
        *,
        expected_observation_dim: int = 29,
    ):
        if cycle_error_threshold <= 0:
            raise ValueError('cycle_error_threshold must be positive.')
        self.adapter = adapter
        self.geometry = geometry
        self.cycle_error_threshold = float(cycle_error_threshold)
        self.expected_observation_dim = int(expected_observation_dim)
        latent_mean = np.asarray(getattr(adapter, 'latent_mean', np.zeros(adapter.latent_dim)), dtype=np.float32)
        latent_std = np.asarray(getattr(adapter, 'latent_std', np.ones(adapter.latent_dim)), dtype=np.float32)
        self.latent_mean = latent_mean
        self.latent_std = np.maximum(latent_std, 1e-6)

    def _cycle_error(self, latents: np.ndarray, observations: np.ndarray) -> np.ndarray:
        flat_obs = observations.reshape(-1, observations.shape[-1])
        encoded = self.adapter.encode(flat_obs).reshape(latents.shape)
        return np.linalg.norm((encoded - latents) / self.latent_std, axis=-1)

    def validate(
        self,
        start_observations: np.ndarray,
        predicted_latents: np.ndarray,
        predicted_observations: np.ndarray,
        lengths: np.ndarray,
    ) -> ValidityResult:
        predicted_latents = np.asarray(predicted_latents, dtype=np.float32)
        predicted_observations = np.asarray(predicted_observations, dtype=np.float32)
        lengths = np.asarray(lengths, dtype=np.int64).reshape(-1)
        if predicted_latents.ndim != 3 or predicted_observations.ndim != 3:
            raise ValueError('Predicted latent and observation arrays must be (N, T, dimension).')
        if predicted_latents.shape[:2] != predicted_observations.shape[:2] or len(lengths) != len(predicted_latents):
            raise ValueError('Predictions and lengths have inconsistent shapes.')
        n, horizon = predicted_latents.shape[:2]
        mask = np.arange(horizon)[None, :] < lengths[:, None]
        finite = np.all(np.isfinite(predicted_latents), axis=-1) & np.all(np.isfinite(predicted_observations), axis=-1)
        shape_ok = np.ones((n, horizon), dtype=bool)
        if predicted_observations.shape[-1] != self.expected_observation_dim:
            shape_ok[:] = False
        cycle = np.full((n, horizon), np.inf, dtype=np.float32)
        valid_indices = np.flatnonzero(mask & finite & shape_ok)
        if len(valid_indices):
            rows, cols = np.unravel_index(valid_indices, (n, horizon))
            cycle_values = self._cycle_error(predicted_latents[rows, cols], predicted_observations[rows, cols])
            cycle[rows, cols] = cycle_values
        manifold = cycle <= self.cycle_error_threshold
        points = predicted_observations[..., :2]
        collision = np.zeros((n, horizon), dtype=bool)
        for row in range(n):
            previous = np.asarray(start_observations[row, :2], dtype=np.float32)
            for col in range(horizon):
                if not mask[row, col]:
                    continue
                if finite[row, col] and shape_ok[row, col] and manifold[row, col]:
                    collision[row, col] = self.geometry.segment_valid(previous, points[row, col])
                    previous = points[row, col]
                else:
                    previous = points[row, col] if finite[row, col] and shape_ok[row, col] else previous
        active_valid = finite & shape_ok & manifold & collision
        row_valid = np.all(~mask | active_valid, axis=1)
        reasons = np.full(n, 'valid', dtype='<U24')
        for row in range(n):
            active = mask[row]
            if not np.all(finite[row][active]):
                reasons[row] = 'non_finite'
            elif not np.all(shape_ok[row][active]):
                reasons[row] = 'invalid_decoded_state'
            elif not np.all(manifold[row][active]):
                reasons[row] = 'off_manifold'
            elif not np.all(collision[row][active]):
                reasons[row] = 'collision'
        return ValidityResult(row_valid, reasons, cycle)

    def __call__(self, start_observations, predicted_latents, predicted_observations, lengths):
        result = self.validate(start_observations, predicted_latents, predicted_observations, lengths)
        return result.valid, result.reasons
