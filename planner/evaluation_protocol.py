"""Matched evaluation helpers, baseline proposals, and failure taxonomy."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


FAILURE_TAXONOMY = (
    'search_exhausted',
    'model_goal_false_positive',
    'early_model_execution_divergence',
    'late_compounding_divergence',
    'wall/contact_stall',
    'unstable_ant_state',
    'plan_exhausted_short_of_goal',
    'environment_or_artifact_error',
)


def demonstrated_random_actions(
    rng: np.random.Generator,
    action_snippets: np.ndarray,
    action_lengths: np.ndarray,
    max_plan_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate complete or truncated episode-safe snippets to a fixed budget."""
    action_snippets = np.asarray(action_snippets, dtype=np.float32)
    action_lengths = np.asarray(action_lengths, dtype=np.int64).reshape(-1)
    if action_snippets.ndim != 3 or len(action_snippets) != len(action_lengths) or not len(action_snippets):
        raise ValueError('Action snippets must be a non-empty 3D array aligned with lengths.')
    if np.any(action_lengths < 1) or np.any(action_lengths > action_snippets.shape[1]):
        raise ValueError('Action lengths must be within the fixed snippet width.')
    if not np.all(np.isfinite(action_snippets)):
        raise ValueError('Action snippets must be finite.')
    if max_plan_steps <= 0:
        raise ValueError('max_plan_steps must be positive.')
    actions = []
    source_indices = []
    total = 0
    while total < max_plan_steps:
        index = int(rng.integers(len(action_snippets)))
        length = min(int(action_lengths[index]), max_plan_steps - total)
        actions.append(action_snippets[index, :length])
        source_indices.append(index)
        total += length
    return np.concatenate(actions, axis=0), np.asarray(source_indices, dtype=np.int64)


def uniform_random_actions(
    rng: np.random.Generator,
    low: np.ndarray,
    high: np.ndarray,
    max_plan_steps: int,
) -> np.ndarray:
    """Generate the explicitly out-of-distribution uniform-action baseline."""
    low = np.asarray(low, dtype=np.float32)
    high = np.asarray(high, dtype=np.float32)
    if low.shape != high.shape or low.ndim != 1 or not np.all(np.isfinite(low)) or not np.all(np.isfinite(high)):
        raise ValueError('Uniform baseline requires finite, aligned one-dimensional action bounds.')
    if np.any(high < low) or max_plan_steps <= 0:
        raise ValueError('Invalid action bounds or max_plan_steps.')
    return rng.uniform(low, high, size=(max_plan_steps, len(low))).astype(np.float32)


def execute_action_sequence(env: Any, actions: np.ndarray, *, record_frames: bool = False) -> dict[str, Any]:
    """Execute a fixed action sequence without clipping or extending it."""
    actions = np.asarray(actions, dtype=np.float32)
    low = np.asarray(env.action_space.low, dtype=np.float32)
    high = np.asarray(env.action_space.high, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[-1] != len(low):
        raise ValueError('Actions do not match the environment action shape.')
    if len(actions) and (not np.all(np.isfinite(actions)) or np.any(actions < low) or np.any(actions > high)):
        raise ValueError('Baseline actions contain non-finite or out-of-bounds values.')
    observations = [np.asarray(env.unwrapped.get_ob(), dtype=np.float32)]
    executed = []
    rewards = []
    successes = []
    terminated = []
    truncated = []
    frames = [np.asarray(env.render())] if record_frames else []
    for action in actions:
        observation, reward, is_terminated, is_truncated, info = env.step(action)
        executed.append(action.copy())
        observations.append(np.asarray(observation, dtype=np.float32))
        rewards.append(float(reward))
        successes.append(float(info.get('success', 0.0)))
        terminated.append(bool(is_terminated))
        truncated.append(bool(is_truncated))
        if record_frames:
            frames.append(np.asarray(env.render()))
        if is_terminated or is_truncated or successes[-1] > 0:
            break
    return {
        'observations': np.asarray(observations, dtype=np.float32),
        'actions': np.asarray(executed, dtype=np.float32),
        'rewards': np.asarray(rewards, dtype=np.float32),
        'successes': np.asarray(successes, dtype=np.float32),
        'terminated': np.asarray(terminated, dtype=bool),
        'truncated': np.asarray(truncated, dtype=bool),
        'success': bool(np.any(np.asarray(successes) > 0)),
        'steps': int(len(executed)),
        'frames': frames,
    }


def classify_failure(result: Mapping[str, Any], execution: Mapping[str, Any] | None = None) -> str | None:
    """Assign one taxonomy label using serialized measurements, not video intuition."""
    if execution is not None and bool(execution.get('success', False)):
        return None
    status = str(result.get('status', 'error'))
    if status in ('error', 'invalid_start', 'invalid_goal'):
        return 'environment_or_artifact_error'
    goal_node = result.get('goal_node')
    if goal_node is None:
        return 'search_exhausted'
    if execution is None:
        # A predicted goal is a candidate-plan outcome, not an execution
        # failure. Simulator failure labels require actual execution evidence.
        return None
    if bool(execution.get('wall_contact', False)):
        return 'wall/contact_stall'
    if bool(execution.get('unstable_ant_state', False)):
        return 'unstable_ant_state'
    first_divergent = execution.get('first_divergent_step')
    steps = max(int(execution.get('steps', 0)), 1)
    if first_divergent is not None:
        return (
            'early_model_execution_divergence'
            if int(first_divergent) <= max(1, steps // 3)
            else 'late_compounding_divergence'
        )
    if int(execution.get('steps', 0)) >= int(result.get('predicted_plan_steps', 0)):
        return 'plan_exhausted_short_of_goal'
    return 'model_goal_false_positive'


def summarize_offline_rows(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Compute predeclared offline screening measures without simulator fields."""
    if not rows:
        raise ValueError('At least one offline row is required.')
    candidate = [bool(row.get('candidate_plan', row.get('goal_node') is not None)) for row in rows]
    candidate_rows = [row for row, is_candidate in zip(rows, candidate) if is_candidate]

    def numeric(key: str, source: list[Mapping[str, Any]] = rows) -> list[float]:
        values = []
        for row in source:
            value = row.get(key)
            if value is not None and np.isfinite(float(value)):
                values.append(float(value))
        return values

    rejection_total = []
    invalid_rates = []
    for row in rows:
        counters = row.get('rejection_counters', {})
        if isinstance(counters, Mapping):
            total = float(sum(int(value) for value in counters.values()))
            rejection_total.append(total)
            iterations = float(row.get('iterations', row.get('tree_summary', {}).get('iterations', 0)) or 0)
            invalid_rates.append(total / max(iterations, 1.0))

    failure_distances = [
        value
        for value in numeric(
            'best_predicted_goal_distance', [row for row, is_candidate in zip(rows, candidate) if not is_candidate]
        )
    ]

    def median(values: list[float]) -> float | None:
        return float(np.median(values)) if values else None

    return {
        'runs': len(rows),
        'candidate_plan_rate': float(np.mean(candidate)),
        'median_iterations_to_first_goal': median(numeric('first_goal_iteration', candidate_rows)),
        'median_nodes_to_candidate': median(numeric('nodes_to_candidate', candidate_rows)),
        'median_predicted_plan_steps': median(numeric('predicted_plan_steps', candidate_rows)),
        'median_predicted_action_energy': median(numeric('predicted_action_energy', candidate_rows)),
        'median_planning_time': median(numeric('planning_time')),
        'mean_model_propagation_calls': float(np.mean(numeric('model_propagation_calls')))
        if numeric('model_propagation_calls')
        else None,
        'mean_rejections': float(np.mean(rejection_total)) if rejection_total else None,
        'mean_invalid_edge_rate': float(np.mean(invalid_rates)) if invalid_rates else None,
        'median_failure_best_predicted_goal_distance': median(failure_distances),
        'median_wall_clearance': median(numeric('minimum_wall_clearance', candidate_rows)),
        'median_cycle_error_margin': median(numeric('minimum_cycle_error_margin', candidate_rows)),
        'execution_fields_ignored': True,
    }


def select_offline_configuration(cohorts: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Apply the predeclared, simulator-blind selection rule to cohort summaries."""
    if not cohorts:
        raise ValueError('At least one offline cohort is required.')
    required = ('config_id', 'candidate_plan_rate', 'mean_rejections', 'median_planning_time')
    for cohort in cohorts:
        missing = [key for key in required if key not in cohort]
        if missing:
            raise ValueError(f'Offline cohort is missing selection fields: {missing}.')

    def ranking_key(cohort: Mapping[str, Any]) -> tuple[float, float, float, float, str]:
        return (
            -float(cohort['candidate_plan_rate']),
            float(cohort['mean_rejections']),
            float(cohort['median_planning_time']),
            float(cohort.get('median_predicted_plan_steps', np.inf)),
            str(cohort['config_id']),
        )

    ranking = sorted((dict(cohort) for cohort in cohorts), key=ranking_key)
    return {
        'selection_rule': (
            'maximize candidate_plan_rate; then minimize mean_rejections, median_planning_time, '
            'median_predicted_plan_steps, and config_id'
        ),
        'selected_config_id': ranking[0]['config_id'],
        'ranking': ranking,
        'simulator_fields_used': False,
    }
