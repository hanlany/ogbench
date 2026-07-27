"""Single-problem learned-latent AntMaze planner and artifact writer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from ogbench import make_env_and_datasets

from .artifacts import LatentModelAdapter, build_planner_cache, cache_identity, load_planner_cache
from .collision import AntMazeValidity, MazeGeometry
from .evaluation_protocol import classify_failure
from .l2rrt import L2RRTPlanner, PlannerConfig, make_metric


CHECKPOINT_DEFAULT = (
    '/workspace/ogbench/latent/train/exp/OGBench/LatentDynamicsResidualXYw5/sd001_20260721_042416/params_1000000.pkl'
)
DATASET_DEFAULT = '/workspace/ogbench/data_gen_scripts/data/antmaze-large-navigate-v0.npz'


def reset_antmaze(env: Any, task_id: int, seed: int) -> tuple[np.ndarray, dict[str, Any]]:
    """Reset task selection, Gym RNGs, action RNG, and legacy global NumPy RNG."""
    if not 1 <= task_id <= 5:
        raise ValueError('AntMaze task_id must be in [1, 5].')
    np.random.seed(seed)
    env.action_space.seed(seed)
    observation, info = env.reset(seed=seed, options={'task_id': task_id})
    observation = np.asarray(observation, dtype=np.float32)
    if observation.shape != (29,):
        raise ValueError(f'Expected AntMaze observation shape (29,), got {observation.shape}.')
    if 'goal' not in info or np.asarray(info['goal']).shape != (29,):
        raise ValueError('AntMaze reset did not provide a 29-dimensional goal observation.')
    return observation, info


def _json_dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=_json_default) + '\n')


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    raise TypeError(f'Cannot serialize {type(value).__name__}.')


def _write_tree(path: Path, planner: L2RRTPlanner) -> None:
    nodes = planner.nodes
    edge_offsets = [0]
    edge_latents = []
    edge_observations = []
    edge_actions = []
    for index in range(len(nodes)):
        if index in planner.edges:
            edge = planner.edges[index]
            edge_latents.append(edge.latents)
            edge_observations.append(edge.observations)
            edge_actions.append(edge.actions)
            edge_offsets.append(edge_offsets[-1] + edge.length)
        else:
            edge_offsets.append(edge_offsets[-1])
    np.savez_compressed(
        path,
        node_latents=np.stack([node.latent for node in nodes]),
        node_observations=np.stack([node.observation for node in nodes]),
        parent_indices=np.asarray([node.parent for node in nodes], dtype=np.int64),
        cumulative_costs=np.asarray([node.cumulative_cost for node in nodes], dtype=np.float32),
        depths=np.asarray([node.depth for node in nodes], dtype=np.int64),
        creation_iterations=np.asarray([node.creation_iteration for node in nodes], dtype=np.int64),
        edge_offsets=np.asarray(edge_offsets, dtype=np.int64),
        edge_latents=np.concatenate(edge_latents)
        if edge_latents
        else np.empty((0, nodes[0].latent.shape[-1]), dtype=np.float32),
        edge_observations=np.concatenate(edge_observations)
        if edge_observations
        else np.empty((0, nodes[0].observation.shape[-1]), dtype=np.float32),
        edge_actions=np.concatenate(edge_actions)
        if edge_actions
        else np.empty((0, planner.action_snippets.shape[-1]), dtype=np.float32),
    )


def _plot_xy(
    path: Path,
    title: str,
    *,
    goal_xy: np.ndarray,
    points: np.ndarray,
    tree_lines: list[tuple[np.ndarray, np.ndarray]] | None = None,
    actual: np.ndarray | None = None,
) -> None:
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    points = np.asarray(points, dtype=np.float32)
    figure, axis = plt.subplots(figsize=(10, 7), constrained_layout=True)
    if tree_lines:
        for start, end in tree_lines:
            axis.plot([start[0], end[0]], [start[1], end[1]], color='0.75', linewidth=0.5)
    if len(points):
        axis.plot(points[:, 0], points[:, 1], '-o', color='tab:blue', markersize=2, label='predicted')
        axis.scatter(points[0, 0], points[0, 1], color='tab:green', label='start', zorder=3)
    if actual is not None and len(actual):
        axis.plot(actual[:, 0], actual[:, 1], '--', color='tab:orange', label='actual')
    goal_xy = np.asarray(goal_xy, dtype=np.float32)
    axis.scatter(goal_xy[0], goal_xy[1], marker='*', s=100, color='tab:red', label='goal', zorder=3)
    axis.set(title=title, xlabel='world x', ylabel='world y', aspect='equal')
    axis.legend(loc='best')
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _minimum_tree_goal_xy_distance(planner: L2RRTPlanner, goal_xy: np.ndarray) -> float:
    """Return the nearest decoded tree endpoint to the AntMaze goal in world XY."""
    if not planner.nodes:
        return float('inf')
    node_xy = np.stack([node.observation[:2] for node in planner.nodes])
    return float(np.min(np.linalg.norm(node_xy - np.asarray(goal_xy, dtype=np.float32), axis=-1)))


def _execute(
    env: Any,
    actions: np.ndarray,
    predicted_observations: np.ndarray,
    *,
    record_video: bool = False,
    divergence_threshold: float = 1.0,
) -> dict[str, Any]:
    lows = np.asarray(env.action_space.low, dtype=np.float32)
    highs = np.asarray(env.action_space.high, dtype=np.float32)
    actions = np.asarray(actions, dtype=np.float32)
    if len(actions) and (np.any(actions < lows) or np.any(actions > highs) or not np.all(np.isfinite(actions))):
        raise ValueError('Returned plan contains an out-of-bounds or non-finite action.')
    observations = []
    rewards = []
    infos = []
    successes = []
    executed_actions = []
    frames = []
    actual = np.asarray(env.unwrapped.get_ob(), dtype=np.float32)
    observations.append(actual)
    if record_video:
        frames.append(np.asarray(env.render()))
    for action in actions:
        executed_actions.append(np.asarray(action, dtype=np.float32).copy())
        observation, reward, terminated, truncated, info = env.step(action)
        observation = np.asarray(observation, dtype=np.float32)
        observations.append(observation)
        rewards.append(float(reward))
        infos.append(info)
        successes.append(float(info.get('success', 0.0)))
        if record_video:
            frames.append(np.asarray(env.render()))
        if terminated or truncated or successes[-1] > 0:
            break
    actual_array = np.asarray(observations, dtype=np.float32)
    predicted_xy = np.asarray(predicted_observations[: len(actual_array), :2], dtype=np.float32)
    actual_xy = actual_array[:, :2]
    xy_error = np.linalg.norm(predicted_xy - actual_xy, axis=-1) if len(predicted_xy) else np.empty(0)
    goal = np.asarray(env.unwrapped.cur_goal_xy, dtype=np.float32)
    goal_distances = np.linalg.norm(actual_xy - goal, axis=-1)
    first_divergent = np.flatnonzero(xy_error > divergence_threshold)
    successful_steps = np.flatnonzero(np.asarray(successes) > 0)
    return {
        'observations': actual_array,
        'actions': np.asarray(executed_actions, dtype=np.float32),
        'rewards': np.asarray(rewards, dtype=np.float32),
        'successes': np.asarray(successes, dtype=np.float32),
        'xy_error': xy_error.astype(np.float32),
        'minimum_goal_distance': float(np.min(goal_distances)),
        'final_goal_distance': float(goal_distances[-1]),
        'success': bool(np.any(np.asarray(successes) > 0)),
        'steps': int(len(rewards)),
        'time_to_success': int(successful_steps[0] + 1) if len(successful_steps) else None,
        'endpoint_xy_error': float(xy_error[-1]) if len(xy_error) else None,
        'first_divergent_step': int(first_divergent[0]) if len(first_divergent) else None,
        'divergence_threshold': float(divergence_threshold),
        'infos': infos,
        'frames': frames,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f'Refusing to mix planner outputs into non-empty directory: {output_dir}.')
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / 'plots').mkdir()
    env = make_env_and_datasets(args.env_name, dataset_path=args.dataset_path, env_only=True)
    try:
        adapter = LatentModelAdapter.from_checkpoint(args.checkpoint_path, max_horizon=args.max_edge_horizon)
        cache_path = Path(args.cache_path).expanduser().resolve()
        if not cache_path.exists():
            if not args.build_cache:
                raise FileNotFoundError(f'Cache does not exist: {cache_path}; pass --build_cache to create it.')
            cache = build_planner_cache(
                adapter,
                args.dataset_path,
                cache_path,
                max_snippet_length=args.max_edge_horizon,
                seed=args.planner_seed,
            )
        else:
            expected_cache_identity = cache_identity(adapter, args.dataset_path, args.max_edge_horizon)
            cache = load_planner_cache(
                cache_path,
                expected=expected_cache_identity,
            )
        adapter.latent_mean = np.asarray(cache.metadata['latent_mean'], dtype=np.float32)
        adapter.latent_std = np.asarray(cache.metadata['latent_std'], dtype=np.float32)
        observation, info = reset_antmaze(env, args.task_id, args.environment_seed)
        goal_observation = np.asarray(info['goal'], dtype=np.float32)
        goal_latent = adapter.encode(goal_observation[None])[0]
        start_latent = adapter.encode(observation[None])[0]
        geometry = MazeGeometry.from_env(env, clearance=args.clearance)
        validity = AntMazeValidity(adapter, geometry, args.cycle_error_threshold)
        metric = make_metric(
            args.metric,
            np.asarray(cache.metadata['latent_mean'], dtype=np.float32),
            np.asarray(cache.metadata['latent_std'], dtype=np.float32),
            np.asarray(cache.metadata['xy_scale'], dtype=np.float32),
        )

        def propagation(initial_latents, actions, lengths):
            return adapter.propagate(initial_latents, actions, lengths)

        def decode(latents, mask):
            return adapter.decode_rollout(latents, mask)

        def goal_fn(observations):
            return (
                np.linalg.norm(observations[..., :2] - np.asarray(env.unwrapped.cur_goal_xy), axis=-1)
                <= args.goal_tolerance
            )

        action_snippets = cache.action_snippets
        proposal_label = 'demonstrated_navigation_snippets'
        if args.control_proposal == 'uniform':
            if not np.all(np.isfinite(env.action_space.low)) or not np.all(np.isfinite(env.action_space.high)):
                raise ValueError('Uniform proposal requires finite environment action bounds.')
            proposal_rng = np.random.default_rng(args.planner_seed)
            action_snippets = proposal_rng.uniform(
                np.asarray(env.action_space.low, dtype=np.float32),
                np.asarray(env.action_space.high, dtype=np.float32),
                size=cache.action_snippets.shape,
            ).astype(np.float32)
            proposal_label = 'unsafe_uniform_ablation'

        planner = L2RRTPlanner(
            config=PlannerConfig(
                metric=args.metric,
                best_near_radius=args.best_near_radius,
                goal_bias=args.goal_bias,
                max_iterations=args.max_iterations,
                timeout_seconds=args.timeout_seconds,
                max_plan_steps=args.max_plan_steps,
                num_control_candidates=args.num_control_candidates,
                cost_mode=args.cost_mode,
                energy_weight=args.energy_weight,
            ),
            state_latents=cache.state_latents,
            state_observations=cache.state_observations,
            action_snippets=action_snippets,
            action_lengths=cache.action_lengths,
            propagation_fn=propagation,
            decode_fn=decode,
            metric_fn=metric,
            validity_fn=validity,
            goal_fn=goal_fn,
            goal_latent=goal_latent,
            goal_observation=goal_observation,
            seed=args.planner_seed,
        )
        result = planner.plan(start_latent, observation)
        result.best_predicted_goal_distance = _minimum_tree_goal_xy_distance(
            planner, np.asarray(env.unwrapped.cur_goal_xy)
        )
        path_xy = result.predicted_observations[:, :2] if len(result.predicted_observations) else observation[None, :2]
        minimum_wall_clearance = geometry.polyline_clearance(path_xy)
        minimum_cycle_error_margin = None
        if len(result.predicted_observations):
            cycle_errors = validity._cycle_error(result.predicted_latents, result.predicted_observations)
            minimum_cycle_error_margin = float(args.cycle_error_threshold - np.max(cycle_errors))
        result_path = {
            'status': result.status,
            'reason': result.reason,
            'goal_node': result.goal_node,
            'tree_summary': result.tree_summary,
            'rejection_counters': result.rejection_counters,
            'target_sampling': result.target_sampling,
            'best_predicted_goal_distance': result.best_predicted_goal_distance,
            'selected_cost': result.selected_cost,
            'planning_time': result.planning_time,
            'predicted_plan_steps': int(len(result.actions)),
            'control_proposal_label': proposal_label,
            'predicted_action_energy': float(np.sum(result.actions**2)),
            'minimum_wall_clearance': minimum_wall_clearance,
            'minimum_cycle_error_margin': minimum_cycle_error_margin,
            'predicted_path_xy_length': float(
                np.sum(np.linalg.norm(np.diff(result.predicted_observations[:, :2], axis=0), axis=-1))
            )
            if len(result.predicted_observations) > 1
            else 0.0,
        }
        result_path['failure_taxonomy'] = classify_failure(result_path)
        tree_lines = [
            (planner.nodes[node.parent].observation[:2], node.observation[:2])
            for node in planner.nodes
            if node.parent >= 0
        ]
        fallback_points = np.asarray([observation], dtype=np.float32)
        _plot_xy(
            output_dir / 'plots/tree_xy.png',
            'L2RRT predicted tree XY',
            goal_xy=np.asarray(env.unwrapped.cur_goal_xy),
            points=np.stack([node.observation[:2] for node in planner.nodes]),
            tree_lines=tree_lines,
        )
        _plot_xy(
            output_dir / 'plots/predicted_plan_xy.png',
            'Predicted plan XY',
            goal_xy=np.asarray(env.unwrapped.cur_goal_xy),
            points=result.predicted_observations[:, :2] if len(result.predicted_observations) else fallback_points,
        )
        execution = None
        if args.execute and result.goal_node is not None:
            execution = _execute(
                env,
                result.actions,
                result.predicted_observations,
                record_video=args.video,
                divergence_threshold=args.divergence_threshold,
            )
            result_path.update(
                {
                    'execution_success': execution['success'],
                    'execution_steps': execution['steps'],
                    'minimum_actual_goal_distance': execution['minimum_goal_distance'],
                    'final_actual_goal_distance': execution['final_goal_distance'],
                    'predicted_vs_actual_xy_error_final': float(execution['xy_error'][-1])
                    if len(execution['xy_error'])
                    else None,
                    'execution_time_to_success': execution['time_to_success'],
                    'execution_endpoint_xy_error': execution['endpoint_xy_error'],
                    'execution_first_divergent_step': execution['first_divergent_step'],
                    'execution_divergence_threshold': execution['divergence_threshold'],
                }
            )
            result_path['failure_taxonomy'] = classify_failure(result_path, execution)
            np.savez_compressed(
                output_dir / 'execution.npz',
                **{key: value for key, value in execution.items() if isinstance(value, np.ndarray)},
            )
            _plot_xy(
                output_dir / 'plots/predicted_vs_actual.png',
                'Predicted versus actual XY',
                goal_xy=np.asarray(env.unwrapped.cur_goal_xy),
                points=result.predicted_observations[:, :2],
                actual=execution['observations'][:, :2],
            )
            if args.video:
                import imageio.v2 as imageio

                imageio.mimsave(output_dir / 'execution.mp4', execution['frames'], fps=10)
        config = vars(args).copy()
        config.update(
            {
                'checkpoint_path': str(Path(args.checkpoint_path).expanduser().resolve()),
                'dataset_path': str(Path(args.dataset_path).expanduser().resolve()),
                'cache_path': str(cache_path),
                'resolved_environment_id': env.spec.id,
                'environment_task_id': int(env.unwrapped.cur_task_id),
                'goal_xy': np.asarray(env.unwrapped.cur_goal_xy).tolist(),
                'start_observation': observation.tolist(),
                'planner_config': result.config,
                'control_proposal_label': proposal_label,
            }
        )
        _json_dump(output_dir / 'config.json', config)
        _json_dump(output_dir / 'result.json', result_path)
        _write_tree(output_dir / 'tree.npz', planner)
        if result.goal_node is not None:
            np.savez_compressed(
                output_dir / 'plan.npz',
                actions=result.actions,
                predicted_latents=result.predicted_latents,
                predicted_observations=result.predicted_observations,
            )
        (output_dir / 'summary.txt').write_text(
            f'status: {result.status}\nreason: {result.reason}\nnodes: {len(planner.nodes)}\n'
            f'predicted_plan_steps: {len(result.actions)}\nplanning_time: {result.planning_time:.6f}\n'
        )
        return result_path
    finally:
        env.close()


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint_path', default=CHECKPOINT_DEFAULT)
    parser.add_argument('--env_name', default='antmaze-large-navigate-v0')
    parser.add_argument('--dataset_path', default=DATASET_DEFAULT)
    parser.add_argument('--cache_path', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--build_cache', action='store_true')
    parser.add_argument('--task_id', type=int, choices=range(1, 6), required=True)
    parser.add_argument('--environment_seed', type=int, default=0)
    parser.add_argument('--planner_seed', type=int, default=0)
    parser.add_argument('--metric', choices=('latent', 'xy', 'hybrid'), default='hybrid')
    parser.add_argument('--best_near_radius', type=float, required=True)
    parser.add_argument('--goal_bias', type=float, default=0.1)
    parser.add_argument('--goal_tolerance', type=float, default=0.5)
    parser.add_argument('--max_edge_horizon', type=int, default=10)
    parser.add_argument('--max_plan_steps', type=int, required=True)
    parser.add_argument('--num_control_candidates', type=int, default=32)
    parser.add_argument('--control_proposal', choices=('demonstrated', 'uniform'), default='demonstrated')
    parser.add_argument('--cost_mode', choices=('steps', 'steps_plus_energy'), default='steps')
    parser.add_argument('--energy_weight', type=float, default=0.0)
    parser.add_argument('--cycle_error_threshold', type=float, required=True)
    parser.add_argument('--clearance', type=float, default=0.0)
    parser.add_argument('--max_iterations', type=int, default=200)
    parser.add_argument('--timeout_seconds', type=float, default=30.0)
    parser.add_argument('--divergence_threshold', type=float, default=1.0)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--video', action='store_true', help='Reserved for the execution-video follow-up.')
    return parser


def main() -> None:
    args = make_parser().parse_args()
    run(args)


if __name__ == '__main__':
    main()
