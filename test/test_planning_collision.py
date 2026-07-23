import numpy as np

from planner.collision import AntMazeValidity, MazeGeometry


def test_maze_geometry_points_and_continuous_segments():
    geometry = MazeGeometry(np.array([[1, 1, 1], [0, 0, 0], [1, 1, 1]]), 2.0, 2.0, 2.0)
    free = [geometry.cell_center((1, j)) for j in range(3)]
    assert all(geometry.point_valid(point) for point in free)
    assert not geometry.point_valid(geometry.cell_center((0, 1)))
    assert not geometry.point_valid(np.array([100.0, 100.0]))
    assert geometry.segment_valid(free[0], free[-1])
    assert not geometry.segment_valid(np.array([-1.0, -2.0]), np.array([1.0, 2.0]))


def test_maze_geometry_rejects_interior_wall_crossing_and_clearance():
    geometry = MazeGeometry(np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]]), 2.0, 2.0, 2.0)
    left = geometry.cell_center((0, 0))
    right = geometry.cell_center((0, 2))
    assert geometry.point_valid(left)
    assert geometry.point_valid(right)
    assert not geometry.segment_valid(left, right)
    corridor = geometry.cell_center((1, 0)), geometry.cell_center((1, 2))
    assert geometry.segment_valid(*corridor)
    near_wall = np.array([-1.0, -0.9]), np.array([1.0, -0.9])
    assert not MazeGeometry(geometry.maze_map, 2.0, 2.0, 2.0, clearance=0.2).segment_valid(*near_wall)


def test_maze_geometry_reports_path_clearance():
    geometry = MazeGeometry(np.array([[1, 1, 1], [0, 0, 0], [1, 1, 1]]), 2.0, 2.0, 2.0)
    points = np.stack([geometry.cell_center((1, 0)), geometry.cell_center((1, 2))])
    assert geometry.point_clearance(points[0]) > 0.0
    assert geometry.polyline_clearance(points) > 0.0
    assert geometry.point_clearance(geometry.cell_center((0, 1))) == 0.0


class _CycleAdapter:
    latent_dim = 1
    latent_std = np.ones(1, dtype=np.float32)

    def encode(self, observations):
        return np.asarray(observations)[..., :1].astype(np.float32)


def test_antmaze_validity_ignores_inactive_padding_but_rejects_active_nonfinite():
    geometry = MazeGeometry(np.zeros((3, 3), dtype=np.int8), 2.0, 2.0, 2.0)
    checker = AntMazeValidity(_CycleAdapter(), geometry, cycle_error_threshold=0.01, expected_observation_dim=2)
    start = np.array([[-2.0, -2.0], [-2.0, -2.0]], dtype=np.float32)
    latents = np.array([[[0.0], [np.nan], [np.nan]], [[0.0], [1.0], [np.nan]]], dtype=np.float32)
    observations = np.array(
        [
            [[0.0, 0.0], [np.nan, np.nan], [np.nan, np.nan]],
            [[0.0, 0.0], [1.0, 0.0], [np.nan, np.nan]],
        ],
        dtype=np.float32,
    )
    valid, reasons = checker(start, latents, observations, np.array([1, 2]))
    assert np.array_equal(valid, np.array([True, True]))
    assert np.array_equal(reasons, np.array(['valid', 'valid']))
    latents[1, 0, 0] = np.nan
    valid, reasons = checker(start, latents, observations, np.array([1, 2]))
    assert not valid[1]
    assert reasons[1] == 'non_finite'
