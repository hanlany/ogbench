import numpy as np

from ogbench import make_env_and_datasets
from planner.evaluation_protocol import (
    classify_failure,
    demonstrated_random_actions,
    select_offline_configuration,
    summarize_offline_rows,
    uniform_random_actions,
)
from planner.evaluate_planner import write_summary
from planner.plan_antmaze import reset_antmaze


def test_antmaze_reset_helper_is_deterministic_for_fixed_task():
    env = make_env_and_datasets(
        'antmaze-large-navigate-v0',
        dataset_path='/workspace/ogbench/data_gen_scripts/data/antmaze-large-navigate-v0.npz',
        env_only=True,
    )
    try:
        first, first_info = reset_antmaze(env, task_id=1, seed=17)
        second, second_info = reset_antmaze(env, task_id=1, seed=17)
        assert env.unwrapped.cur_task_id == 1
        assert np.array_equal(first, second)
        assert np.array_equal(first_info['goal'], second_info['goal'])
        assert np.array_equal(first[:2], second[:2])
    finally:
        env.close()


def test_matched_baseline_proposals_respect_fixed_budgets():
    snippets = np.arange(12, dtype=np.float32).reshape(3, 2, 2)
    lengths = np.array([1, 2, 2], dtype=np.int64)
    actions, sources = demonstrated_random_actions(np.random.default_rng(4), snippets, lengths, 5)
    assert actions.shape == (5, 2)
    assert len(sources) >= 2
    uniform = uniform_random_actions(np.random.default_rng(4), np.array([-1.0, -2.0]), np.array([1.0, 2.0]), 5)
    assert uniform.shape == (5, 2)
    assert np.all(uniform >= np.array([-1.0, -2.0]))
    assert np.all(uniform <= np.array([1.0, 2.0]))


def test_baseline_proposals_reject_invalid_pools():
    with np.testing.assert_raises(ValueError):
        demonstrated_random_actions(np.random.default_rng(0), np.empty((0, 2, 1)), np.empty(0), 2)
    with np.testing.assert_raises(ValueError):
        demonstrated_random_actions(np.random.default_rng(0), np.zeros((1, 2, 1)), np.array([0]), 2)
    with np.testing.assert_raises(ValueError):
        uniform_random_actions(np.random.default_rng(0), np.array([np.nan]), np.array([1.0]), 2)


def test_failure_taxonomy_and_offline_summary_are_measurement_based(tmp_path):
    assert classify_failure({'status': 'exhausted', 'goal_node': None}) == 'search_exhausted'
    assert (
        classify_failure(
            {'status': 'success', 'goal_node': 2, 'predicted_plan_steps': 10},
            {'success': False, 'first_divergent_step': 1, 'steps': 10},
        )
        == 'early_model_execution_divergence'
    )
    rows = [
        {
            'status': 'success',
            'goal_node': 2,
            'predicted_plan_steps': 5,
            'nodes': 3,
            'planning_time': 0.2,
            'rejection_counters': {'collision': 2},
            'candidate_plan': True,
            'first_goal_iteration': 4,
            'nodes_to_candidate': 3,
            'predicted_action_energy': 2.0,
            'model_propagation_calls': 3,
            'iterations': 6,
        },
        {
            'status': 'exhausted',
            'goal_node': None,
            'predicted_plan_steps': 0,
            'nodes': 1,
            'planning_time': 0.1,
            'rejection_counters': {'no_valid_candidate': 4},
        },
    ]
    summary = summarize_offline_rows(rows)
    assert summary['candidate_plan_rate'] == 0.5
    assert summary['median_iterations_to_first_goal'] == 4.0
    assert summary['median_predicted_action_energy'] == 2.0
    assert summary['mean_model_propagation_calls'] == 3.0
    write_summary(rows, tmp_path / 'offline')
    assert (tmp_path / 'offline/metrics.json').is_file()
    assert (tmp_path / 'offline/summary.csv').is_file()
    assert (tmp_path / 'offline/summary.md').is_file()
    assert (tmp_path / 'offline/plots/status_counts.png').is_file()


def test_offline_selection_is_deterministic_and_simulator_blind():
    selection = select_offline_configuration(
        [
            {
                'config_id': 'b',
                'candidate_plan_rate': 0.8,
                'mean_rejections': 3,
                'median_planning_time': 1.0,
                'median_predicted_plan_steps': 10,
                'execution_success': 1.0,
            },
            {
                'config_id': 'a',
                'candidate_plan_rate': 0.8,
                'mean_rejections': 3,
                'median_planning_time': 1.0,
                'median_predicted_plan_steps': 10,
                'execution_success': 0.0,
            },
        ]
    )
    assert selection['selected_config_id'] == 'a'
    assert selection['simulator_fields_used'] is False
