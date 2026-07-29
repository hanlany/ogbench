import numpy as np

from planner.collision import MazeGeometry
from planner.l2rrt import L2RRTPlanner, Node, PlannerConfig, make_metric
from planner.plan_antmaze import _plot_xy, _save_tree_growth_gif, _write_tree


def _system_propagate(initial, actions, lengths):
    current = initial.copy()
    predictions = np.zeros((len(actions), actions.shape[1], 1), dtype=np.float32)
    for step in range(actions.shape[1]):
        current = current + actions[:, step]
        predictions[:, step] = current
    mask = np.arange(actions.shape[1])[None, :] < lengths[:, None]
    return predictions, mask


def _validity(start_observations, predicted_latents, predicted_observations, lengths):
    mask = np.arange(predicted_latents.shape[1])[None, :] < lengths[:, None]
    valid = np.all(np.where(mask, predicted_observations[..., 0] < 4, True), axis=1)
    reasons = np.where(valid, 'valid', 'collision')
    return valid, reasons


def _planner(seed):
    snippets = np.array([[[1.0], [1.0]], [[-1.0], [-1.0]]], dtype=np.float32)
    lengths = np.array([2, 1], dtype=np.int64)
    return L2RRTPlanner(
        config=PlannerConfig(
            metric='latent',
            best_near_radius=0.0,
            goal_bias=1.0,
            max_iterations=10,
            timeout_seconds=10,
            max_plan_steps=6,
            num_control_candidates=2,
        ),
        state_latents=np.array([[2.0], [3.0]], dtype=np.float32),
        state_observations=np.array([[2.0], [3.0]], dtype=np.float32),
        action_snippets=snippets,
        action_lengths=lengths,
        propagation_fn=_system_propagate,
        metric_fn=make_metric('latent', np.zeros(1), np.ones(1)),
        validity_fn=_validity,
        goal_fn=lambda observations: np.asarray(observations[..., 0] >= 2.0),
        goal_latent=np.array([2.0], dtype=np.float32),
        goal_observation=np.array([2.0], dtype=np.float32),
        seed=seed,
    )


def test_synthetic_planner_reconstructs_exact_variable_length_plan():
    planner = _planner(4)
    result = planner.plan(np.array([0.0], dtype=np.float32), np.array([0.0], dtype=np.float32))
    assert result.status == 'success'
    assert result.goal_node is not None
    assert result.actions.shape == (2, 1)
    assert np.array_equal(result.predicted_latents[:, 0], np.array([0.0, 1.0, 2.0], dtype=np.float32))
    assert result.node_chain[0] == 0
    assert len(result.node_chain) == 2
    assert result.goal_node in result.tree_summary['goal_nodes']
    assert result.tree_summary['iterations'] >= 1


def test_synthetic_plan_replays_exactly_through_propagator():
    result = _planner(4).plan(np.array([0.0], dtype=np.float32), np.array([0.0], dtype=np.float32))
    replayed, mask = _system_propagate(
        np.array([[0.0]], dtype=np.float32), result.actions[None], np.array([len(result.actions)])
    )
    assert np.array_equal(mask, np.array([[True, True]]))
    assert np.array_equal(replayed[0], result.predicted_latents[1:])


def test_multiple_goal_nodes_choose_lowest_cost_then_index():
    planner = _planner(0)
    planner.nodes = [
        Node(0, -1, np.array([0.0]), np.array([0.0]), 0.0, 0, 0),
        Node(1, 0, np.array([2.0]), np.array([2.0]), 3.0, 2, 1),
        Node(2, 0, np.array([2.0]), np.array([2.0]), 2.0, 2, 2),
        Node(3, 0, np.array([2.0]), np.array([2.0]), 2.0, 2, 3),
    ]
    planner.goal_nodes = [1, 2, 3]
    assert planner._best_goal_node() == 2


def test_synthetic_planner_is_reproducible():
    first = _planner(9).plan(np.array([0.0]), np.array([0.0]))
    second = _planner(9).plan(np.array([0.0]), np.array([0.0]))
    assert first.status == second.status
    assert first.node_chain == second.node_chain
    assert np.array_equal(first.actions, second.actions)
    assert first.rejection_counters == second.rejection_counters


def test_best_near_prefers_lowest_cost_and_fallback_breaks_ties_by_index():
    planner = _planner(0)
    planner.nodes = [
        Node(0, -1, np.array([0.0]), np.array([0.0]), 5.0, 0, 0),
        Node(1, 0, np.array([0.5]), np.array([0.5]), 1.0, 1, 1),
    ]
    planner.config = PlannerConfig(best_near_radius=1.0)
    assert planner._best_near(np.array([0.4]), np.array([0.4])) == 1
    planner.config = PlannerConfig(best_near_radius=0.0)
    planner.nodes[1].latent[:] = 2.0
    planner.nodes[1].observation[:] = 2.0
    assert planner._best_near(np.array([1.0]), np.array([1.0])) == 0


def test_goal_bias_sampling_is_reproducible_and_counted():
    first = _planner(12)
    second = _planner(12)
    first.config = PlannerConfig(goal_bias=0.25)
    second.config = PlannerConfig(goal_bias=0.25)
    first_targets = [first._target() for _ in range(100)]
    second_targets = [second._target() for _ in range(100)]
    assert first.target_sampling == second.target_sampling
    assert np.array_equal(np.asarray(first_targets), np.asarray(second_targets))
    assert sum(first.target_sampling.values()) == 100


def test_invalid_candidate_and_intermediate_state_are_rejected():
    snippets = np.array([[[1.0], [1.0]], [[3.0], [3.0]]], dtype=np.float32)
    lengths = np.array([2, 2], dtype=np.int64)

    def propagate(initial, actions, candidate_lengths):
        return _system_propagate(initial, actions, candidate_lengths)

    def candidate_validity(start, latents, observations, candidate_lengths):
        mask = np.arange(latents.shape[1])[None, :] < candidate_lengths[:, None]
        valid = np.all(np.where(mask, observations[..., 0] < 1.5, True), axis=1)
        return valid, np.where(valid, 'valid', 'collision')

    planner = L2RRTPlanner(
        config=PlannerConfig(goal_bias=1.0, max_iterations=1, max_plan_steps=4, num_control_candidates=2),
        state_latents=np.array([[10.0]]),
        state_observations=np.array([[10.0]]),
        action_snippets=snippets,
        action_lengths=lengths,
        propagation_fn=propagate,
        metric_fn=make_metric('latent', np.zeros(1), np.ones(1)),
        validity_fn=candidate_validity,
        goal_fn=lambda observations: np.asarray(observations[..., 0] >= 10.0),
        goal_latent=np.array([10.0]),
        goal_observation=np.array([10.0]),
        seed=0,
    )
    result = planner.plan(np.array([0.0]), np.array([0.0]))
    assert result.status == 'exhausted'
    assert result.tree_summary['nodes'] == 1
    assert result.rejection_counters['collision'] >= 1


def test_invalid_start_and_goal_return_structured_statuses():
    invalid_start = _planner(0).plan(np.array([np.nan]), np.array([0.0]))
    assert invalid_start.status == 'invalid_start'
    invalid_goal_planner = _planner(0)
    invalid_goal_planner.goal_observation[:] = 0.0
    invalid_goal = invalid_goal_planner.plan(np.array([0.0]), np.array([0.0]))
    assert invalid_goal.status == 'invalid_goal'


def test_synthetic_output_bundle_artifacts(tmp_path):
    planner = _planner(4)
    result = planner.plan(np.array([0.0], dtype=np.float32), np.array([0.0], dtype=np.float32))
    tree_path = tmp_path / 'tree.npz'
    _write_tree(tree_path, planner)
    with np.load(tree_path, allow_pickle=False) as arrays:
        assert set(('node_latents', 'node_observations', 'parent_indices', 'edge_offsets', 'edge_actions')) <= set(
            arrays.files
        )
        assert len(arrays['node_latents']) == result.tree_summary['nodes']
        assert arrays['edge_actions'].ndim == 2
        assert len(arrays['edge_actions']) >= len(result.actions)
    plot_path = tmp_path / 'predicted_plan_xy.png'
    _plot_xy(
        plot_path,
        'synthetic',
        goal_xy=np.array([2.0, 0.0]),
        points=np.column_stack((result.predicted_observations[:, 0], np.zeros(len(result.predicted_observations)))),
    )
    assert plot_path.read_bytes().startswith(b'\x89PNG\r\n\x1a\n')


def test_xy_plot_accepts_maze_geometry(tmp_path):
    geometry = MazeGeometry(np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]]), 2.0, 2.0, 2.0, clearance=0.1)
    plot_path = tmp_path / 'maze_plot.png'
    _plot_xy(
        plot_path,
        'maze geometry',
        goal_xy=np.array([0.0, 0.0]),
        points=np.array([[0.0, 0.0]], dtype=np.float32),
        geometry=geometry,
    )
    assert plot_path.read_bytes().startswith(b'\x89PNG\r\n\x1a\n')


def test_xy_plot_can_render_unconnected_tree_nodes(tmp_path):
    plot_path = tmp_path / 'tree_plot.png'
    _plot_xy(
        plot_path,
        'tree',
        goal_xy=np.array([2.0, 0.0]),
        points=np.array([[0.0, 0.0], [2.0, 0.0], [1.0, 1.0]], dtype=np.float32),
        tree_lines=[(np.array([0.0, 0.0]), np.array([1.0, 1.0]))],
        connect_points=False,
    )
    assert plot_path.read_bytes().startswith(b'\x89PNG\r\n\x1a\n')


def test_tree_growth_gif(tmp_path):
    geometry = MazeGeometry(np.zeros((3, 3), dtype=np.int8), 2.0, 2.0, 2.0)
    gif_path = tmp_path / 'tree.gif'
    _save_tree_growth_gif(
        gif_path,
        geometry=geometry,
        goal_xy=np.array([2.0, 2.0]),
        node_points=np.array([[-2.0, -2.0], [0.0, 0.0], [2.0, 2.0]]),
        node_iterations=np.array([0, 2, 4]),
        segments=np.array([[[-2.0, -2.0], [0.0, 0.0]], [[0.0, 0.0], [2.0, 2.0]]]),
        segment_iterations=np.array([2, 4]),
        max_frames=3,
        fps=2,
    )
    assert gif_path.read_bytes().startswith(b'GIF8')
