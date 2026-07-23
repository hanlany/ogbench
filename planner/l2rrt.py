"""Generic, dependency-light L2RRT-style search core."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any, Callable

import numpy as np


MetricFn = Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray]
PropagationFn = Callable[[np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]
ValidityFn = Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]
GoalFn = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class PlannerConfig:
    metric: str = 'latent'
    best_near_radius: float = 1.0
    goal_bias: float = 0.1
    max_iterations: int = 200
    timeout_seconds: float = 30.0
    max_plan_steps: int = 100
    num_control_candidates: int = 32
    cost_mode: str = 'steps'
    energy_weight: float = 0.0
    duplicate_distance: float = 1e-6

    def __post_init__(self) -> None:
        if self.metric not in ('latent', 'xy', 'hybrid'):
            raise ValueError(f'Unknown planner metric: {self.metric!r}.')
        if self.cost_mode not in ('steps', 'steps_plus_energy'):
            raise ValueError(f'Unknown cost mode: {self.cost_mode!r}.')
        if self.best_near_radius < 0 or not 0 <= self.goal_bias <= 1:
            raise ValueError('best_near_radius and goal_bias must be non-negative, with goal_bias <= 1.')
        if self.max_iterations <= 0 or self.timeout_seconds <= 0 or self.max_plan_steps <= 0:
            raise ValueError('Planner resource limits must be positive.')
        if self.num_control_candidates <= 0 or self.duplicate_distance < 0:
            raise ValueError('Candidate count must be positive and duplicate distance non-negative.')


@dataclass
class Edge:
    parent: int
    latents: np.ndarray
    observations: np.ndarray
    actions: np.ndarray
    length: int
    cost: float


@dataclass
class Node:
    index: int
    parent: int
    latent: np.ndarray
    observation: np.ndarray
    cumulative_cost: float
    depth: int
    creation_iteration: int


@dataclass
class PlannerResult:
    status: str
    reason: str
    goal_node: int | None
    tree_summary: dict[str, Any]
    rejection_counters: dict[str, int]
    node_chain: list[int]
    actions: np.ndarray
    predicted_latents: np.ndarray
    predicted_observations: np.ndarray
    planning_time: float
    seed: int
    config: dict[str, Any]
    target_sampling: dict[str, int]
    best_predicted_goal_distance: float
    selected_cost: float | None

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        for key in ('actions', 'predicted_latents', 'predicted_observations'):
            result[key] = result[key].tolist()
        return result


def make_metric(
    name: str,
    latent_mean: np.ndarray | None = None,
    latent_std: np.ndarray | None = None,
    xy_scale: np.ndarray | None = None,
    latent_weight: float = 1.0,
) -> MetricFn:
    """Create one of the calibrated latent, XY, or hybrid distances."""
    if name not in ('latent', 'xy', 'hybrid'):
        raise ValueError(f'Unknown metric: {name!r}.')
    latent_mean = np.zeros(1, dtype=np.float32) if latent_mean is None else np.asarray(latent_mean, dtype=np.float32)
    latent_std = (
        np.ones_like(latent_mean) if latent_std is None else np.maximum(np.asarray(latent_std, dtype=np.float32), 1e-6)
    )
    xy_scale = (
        np.ones(2, dtype=np.float32) if xy_scale is None else np.maximum(np.asarray(xy_scale, dtype=np.float32), 1e-6)
    )

    def metric(latents, observations, target_latent, target_observation):
        if name == 'latent':
            return np.linalg.norm((latents - target_latent) / latent_std, axis=-1)
        xy_delta = observations[..., :2] - target_observation[..., :2]
        xy = np.linalg.norm(xy_delta, axis=-1)
        if name == 'xy':
            return xy
        whitened = (latents - target_latent) / latent_std
        scaled_xy = np.sum((xy_delta / xy_scale) ** 2, axis=-1)
        return np.sqrt(scaled_xy + latent_weight * np.mean(whitened**2, axis=-1))

    return metric


class L2RRTPlanner:
    """Search over an injected learned propagator and validity checker."""

    def __init__(
        self,
        *,
        config: PlannerConfig,
        state_latents: np.ndarray,
        state_observations: np.ndarray,
        action_snippets: np.ndarray,
        action_lengths: np.ndarray,
        propagation_fn: PropagationFn,
        decode_fn: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
        metric_fn: MetricFn,
        validity_fn: ValidityFn,
        goal_fn: GoalFn,
        goal_latent: np.ndarray,
        goal_observation: np.ndarray,
        seed: int = 0,
    ):
        self.config = config
        self.state_latents = np.asarray(state_latents, dtype=np.float32)
        self.state_observations = np.asarray(state_observations, dtype=np.float32)
        self.action_snippets = np.asarray(action_snippets, dtype=np.float32)
        self.action_lengths = np.asarray(action_lengths, dtype=np.int64).reshape(-1)
        if self.state_latents.ndim != 2 or self.state_observations.ndim != 2:
            raise ValueError('State pools must be 2D arrays.')
        if len(self.state_latents) != len(self.state_observations) or not len(self.state_latents):
            raise ValueError('State pools must be non-empty and aligned.')
        if self.action_snippets.ndim != 3 or len(self.action_snippets) != len(self.action_lengths):
            raise ValueError('Action snippets and lengths are not aligned.')
        if np.any(self.action_lengths < 1) or np.any(self.action_lengths > self.action_snippets.shape[1]):
            raise ValueError('Action lengths must be within the fixed snippet width.')
        self.propagation_fn = propagation_fn
        self.decode_fn = decode_fn
        self.metric_fn = metric_fn
        self.validity_fn = validity_fn
        self.goal_fn = goal_fn
        self.goal_latent = np.asarray(goal_latent, dtype=np.float32)
        self.goal_observation = np.asarray(goal_observation, dtype=np.float32)
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.nodes: list[Node] = []
        self.edges: dict[int, Edge] = {}
        self.rejections = {
            'non_finite': 0,
            'off_manifold': 0,
            'collision': 0,
            'invalid_decoded_state': 0,
            'no_valid_candidate': 0,
            'duplicate/no-progress': 0,
        }
        self.target_sampling = {'goal': 0, 'data': 0}
        self.goal_nodes: list[int] = []
        self.iterations = 0
        self.best_goal_distance = np.inf
        self.model_propagation_calls = 0

    def _best_near(self, target_latent: np.ndarray, target_observation: np.ndarray) -> int:
        latents = np.stack([node.latent for node in self.nodes])
        observations = np.stack([node.observation for node in self.nodes])
        distances = np.asarray(
            self.metric_fn(latents, observations, target_latent, target_observation), dtype=np.float64
        )
        eligible = np.flatnonzero(distances <= self.config.best_near_radius)
        candidates = eligible if len(eligible) else np.arange(len(self.nodes))
        return min(
            candidates.tolist(),
            key=lambda index: (self.nodes[index].cumulative_cost if len(eligible) else distances[index], int(index)),
        )

    def _target(self) -> tuple[np.ndarray, np.ndarray]:
        if self.rng.random() < self.config.goal_bias:
            self.target_sampling['goal'] += 1
            return self.goal_latent.copy(), self.goal_observation.copy()
        index = int(self.rng.integers(len(self.state_latents)))
        self.target_sampling['data'] += 1
        return self.state_latents[index].copy(), self.state_observations[index].copy()

    def _record_rejections(self, reasons: np.ndarray) -> None:
        for reason in np.asarray(reasons).reshape(-1).tolist():
            if reason in self.rejections:
                self.rejections[reason] += 1
            else:
                self.rejections[str(reason)] = self.rejections.get(str(reason), 0) + 1

    def _reconstruct(self, node_index: int) -> tuple[list[int], np.ndarray, np.ndarray, np.ndarray]:
        chain = []
        cursor = node_index
        while cursor >= 0:
            chain.append(cursor)
            cursor = self.nodes[cursor].parent
        chain.reverse()
        actions = []
        latents = [self.nodes[chain[0]].latent[None]]
        observations = [self.nodes[chain[0]].observation[None]]
        for child in chain[1:]:
            edge = self.edges[child]
            actions.append(edge.actions)
            latents.append(edge.latents)
            observations.append(edge.observations)
        return (
            chain,
            np.concatenate(actions, axis=0)
            if actions
            else np.empty((0, self.action_snippets.shape[-1]), dtype=np.float32),
            np.concatenate(latents, axis=0),
            np.concatenate(observations, axis=0),
        )

    def plan(self, start_latent: np.ndarray, start_observation: np.ndarray) -> PlannerResult:
        started = time.perf_counter()
        start_latent = np.asarray(start_latent, dtype=np.float32).reshape(-1)
        start_observation = np.asarray(start_observation, dtype=np.float32).reshape(-1)
        if not np.all(np.isfinite(start_latent)) or not np.all(np.isfinite(start_observation)):
            return self._failure('invalid_start', 'start is non-finite', started)
        if not np.all(np.isfinite(self.goal_latent)) or not np.all(np.isfinite(self.goal_observation)):
            return self._failure('invalid_goal', 'goal is non-finite', started)
        try:
            goal_is_valid = bool(np.asarray(self.goal_fn(self.goal_observation)).reshape(-1)[0])
        except Exception as exc:
            return self._failure('invalid_goal', f'goal checker failed: {exc}', started)
        if not goal_is_valid:
            return self._failure('invalid_goal', 'goal observation is not a member of the configured goal set', started)
        self.nodes = [Node(0, -1, start_latent.copy(), start_observation.copy(), 0.0, 0, 0)]
        self.edges = {}
        self.goal_nodes = []
        self.iterations = 0
        self.best_goal_distance = np.inf
        self.model_propagation_calls = 0
        try:
            start_goal = bool(np.asarray(self.goal_fn(start_observation)).reshape(-1)[0])
        except Exception as exc:
            return self._failure('error', f'goal checker failed at start: {exc}', started)
        if start_goal:
            self.goal_nodes.append(0)
            self.best_goal_distance = 0.0
            return self._success(0, started)
        self.best_goal_distance = float(
            np.asarray(
                self.metric_fn(start_latent[None], start_observation[None], self.goal_latent, self.goal_observation)
            )[0]
        )
        reason = 'iteration budget exhausted'
        for iteration in range(self.config.max_iterations):
            self.iterations = iteration + 1
            if time.perf_counter() - started >= self.config.timeout_seconds:
                reason = 'timeout_seconds reached'
                break
            target_latent, target_observation = self._target()
            parent_index = self._best_near(target_latent, target_observation)
            parent = self.nodes[parent_index]
            remaining = self.config.max_plan_steps - parent.depth
            if remaining <= 0:
                self.rejections['no_valid_candidate'] += 1
                continue
            candidate_count = self.config.num_control_candidates
            selected = self.rng.integers(len(self.action_snippets), size=candidate_count)
            actions = self.action_snippets[selected].copy()
            lengths = np.minimum(self.action_lengths[selected], remaining).astype(np.int64)
            keep = lengths > 0
            if not np.any(keep):
                self.rejections['no_valid_candidate'] += 1
                continue
            try:
                initial = np.repeat(parent.latent[None], candidate_count, axis=0)
                self.model_propagation_calls += 1
                predicted_latents, mask = self.propagation_fn(initial, actions, lengths)
                predicted_observations = self._decode_from_propagation(predicted_latents, mask)
                valid, reasons = self.validity_fn(
                    np.repeat(parent.observation[None], candidate_count, axis=0),
                    predicted_latents,
                    predicted_observations,
                    lengths,
                )
            except Exception as exc:
                return self._failure('error', f'propagation or validity failed: {exc}', started)
            valid = np.asarray(valid, dtype=bool)
            self._record_rejections(np.asarray(reasons)[~valid])
            if not np.any(valid):
                self.rejections['no_valid_candidate'] += 1
                continue
            endpoint_latents = predicted_latents[np.arange(candidate_count), lengths - 1]
            endpoint_observations = predicted_observations[np.arange(candidate_count), lengths - 1]
            endpoint_distance = np.asarray(
                self.metric_fn(endpoint_latents, endpoint_observations, target_latent, target_observation)
            )
            endpoint_distance[~valid] = np.inf
            choice = int(np.argmin(endpoint_distance))
            if not np.isfinite(endpoint_distance[choice]):
                self.rejections['no_valid_candidate'] += 1
                continue
            endpoint = endpoint_latents[choice]
            endpoint_observation = endpoint_observations[choice]
            all_distances = np.asarray(
                self.metric_fn(
                    np.asarray([node.latent for node in self.nodes]),
                    np.asarray([node.observation for node in self.nodes]),
                    endpoint,
                    endpoint_observation,
                )
            )
            if np.min(all_distances) <= self.config.duplicate_distance:
                self.rejections['duplicate/no-progress'] += 1
                continue
            length = int(lengths[choice])
            edge_actions = actions[choice, :length].copy()
            edge_latents = predicted_latents[choice, :length].copy()
            edge_observations = predicted_observations[choice, :length].copy()
            edge_cost = float(length)
            if self.config.cost_mode == 'steps_plus_energy':
                edge_cost += self.config.energy_weight * float(np.sum(edge_actions**2))
            node_index = len(self.nodes)
            self.nodes.append(
                Node(
                    node_index,
                    parent_index,
                    endpoint.copy(),
                    endpoint_observation.copy(),
                    parent.cumulative_cost + edge_cost,
                    parent.depth + length,
                    iteration,
                )
            )
            self.edges[node_index] = Edge(
                parent_index, edge_latents, edge_observations, edge_actions, length, edge_cost
            )
            self.best_goal_distance = min(
                self.best_goal_distance,
                float(
                    np.asarray(
                        self.metric_fn(
                            endpoint[None], endpoint_observation[None], self.goal_latent, self.goal_observation
                        )
                    )[0]
                ),
            )
            if bool(np.asarray(self.goal_fn(endpoint_observation)).reshape(-1)[0]):
                self.goal_nodes.append(node_index)
        if self.goal_nodes:
            return self._success(self._best_goal_node(), started)
        return self._failure(
            'timeout' if reason == 'timeout_seconds reached' else 'exhausted',
            reason,
            started,
            self.best_goal_distance,
        )

    def _decode_from_propagation(self, predicted_latents: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if self.decode_fn is not None:
            return np.asarray(self.decode_fn(predicted_latents, mask), dtype=np.float32)
        # A same-shaped latent system can use latents as observations directly.
        return np.asarray(predicted_latents, dtype=np.float32)

    def _success(self, node_index: int, started: float) -> PlannerResult:
        chain, actions, latents, observations = self._reconstruct(node_index)
        return PlannerResult(
            'success',
            'goal reached by predicted endpoint',
            node_index,
            {
                'nodes': len(self.nodes),
                'edges': len(self.edges),
                'depth': self.nodes[node_index].depth,
                'iterations': self.iterations,
                'goal_nodes': list(self.goal_nodes),
                'first_goal_iteration': min(
                    (self.nodes[index].creation_iteration for index in self.goal_nodes), default=None
                ),
                'model_propagation_calls': self.model_propagation_calls,
            },
            dict(self.rejections),
            chain,
            actions,
            latents,
            observations,
            time.perf_counter() - started,
            self.seed,
            asdict(self.config),
            dict(self.target_sampling),
            float(self.best_goal_distance),
            self.nodes[node_index].cumulative_cost,
        )

    def _best_goal_node(self) -> int:
        if not self.goal_nodes:
            raise RuntimeError('Cannot select a goal node from an empty goal set.')
        return min(self.goal_nodes, key=lambda index: (self.nodes[index].cumulative_cost, index))

    def _failure(self, status: str, reason: str, started: float, best_goal_distance: float = np.inf) -> PlannerResult:
        return PlannerResult(
            status,
            reason,
            None,
            {
                'nodes': len(self.nodes),
                'edges': len(self.edges),
                'depth': max((node.depth for node in self.nodes), default=0),
                'iterations': self.iterations,
                'goal_nodes': list(self.goal_nodes),
                'first_goal_iteration': min(
                    (self.nodes[index].creation_iteration for index in self.goal_nodes), default=None
                ),
                'model_propagation_calls': self.model_propagation_calls,
            },
            dict(self.rejections),
            [],
            np.empty((0, self.action_snippets.shape[-1]), dtype=np.float32),
            np.empty((0, self.state_latents.shape[-1]), dtype=np.float32),
            np.empty((0, self.state_observations.shape[-1]), dtype=np.float32),
            time.perf_counter() - started,
            self.seed,
            asdict(self.config),
            dict(self.target_sampling),
            float(best_goal_distance),
            None,
        )


Planner = L2RRTPlanner
