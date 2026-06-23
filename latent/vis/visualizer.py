import glob
import json
import pickle
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import gymnasium
import numpy as np
from gymnasium.spaces import Box, Discrete


_LOCAL_VIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _LOCAL_VIS_DIR.parents[1]
_IMPLS_DIR = _REPO_ROOT / 'impls'

for _path in (_REPO_ROOT, _IMPLS_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import ogbench  # noqa: E402


@dataclass
class RolloutResult:
    """Container returned by OGBenchVisualizer.rollout."""

    frames: List[np.ndarray]
    trajectories: List[Dict[str, List[Any]]]
    stats: Dict[str, float]


class OGBenchVisualizer:
    """Headless rollout visualizer for OGBench environments and policies."""

    def __init__(self, seed: int = 0):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        random.seed(seed)
        np.random.seed(seed)

        self.env = None
        self.env_name = None
        self.env_kwargs = {}
        self.agent = None
        self.policy_config = None
        self.policy_flags = None
        self.frame_stack = None

    @staticmethod
    def available_envs() -> List[str]:
        """Return registered Gymnasium environment IDs provided by OGBench."""
        # Importing ogbench registers its environment families with Gymnasium.
        import ogbench as _ogbench  # noqa: F401

        env_ids = []
        for env_id, spec in gymnasium.envs.registration.registry.items():
            entry_point = getattr(spec, 'entry_point', None)
            if isinstance(entry_point, str) and entry_point.startswith('ogbench.'):
                env_ids.append(env_id)
        return sorted(env_ids)

    def load_env(self, name: str, **env_kwargs):
        """Load an OGBench environment by registered env ID or dataset-style name."""
        if self.env is not None:
            self.close()

        self.env_name = name
        self.env_kwargs = dict(env_kwargs)

        frame_stack = self.env_kwargs.pop('frame_stack', self._policy_frame_stack())
        env = self._make_env(name, **self.env_kwargs)
        if frame_stack is not None:
            env = self._wrap_frame_stack(env, frame_stack)

        self.env = env
        self.frame_stack = frame_stack
        return self.env

    def load_policy(self, run_dir: str, epoch: Optional[int] = None):
        """Load an agent from an OGBench impls run directory."""
        run_dir = Path(run_dir).expanduser()
        flags_path = run_dir / 'flags.json'
        if not flags_path.exists():
            raise FileNotFoundError(f'Missing flags.json in policy run directory: {run_dir}')

        with flags_path.open('r') as f:
            flags = json.load(f)

        config = flags.get('agent')
        if config is None:
            raise ValueError(f'Policy flags do not contain an agent config: {flags_path}')

        if self.env is None:
            env_name = flags.get('env_name')
            if env_name is None:
                raise ValueError('No environment is loaded and flags.json does not contain env_name.')
            self.policy_config = config
            self.load_env(env_name)
        else:
            self._validate_policy_env(flags)
            policy_frame_stack = config.get('frame_stack')
            if policy_frame_stack != self.frame_stack:
                self.policy_config = config
                self.load_env(self.env_name, frame_stack=policy_frame_stack, **self.env_kwargs)

        self.policy_flags = flags
        self.policy_config = config

        agent = self._create_agent(config, seed=flags.get('seed', self.seed))
        checkpoint_path = self._checkpoint_path(run_dir, epoch)

        with checkpoint_path.open('rb') as f:
            load_dict = pickle.load(f)

        import flax.serialization

        self.agent = flax.serialization.from_state_dict(agent, load_dict['agent'])
        return self.agent

    def use_random_policy(self):
        """Use the environment's action sampler as the current policy."""
        self.agent = None
        return self

    def rollout(
        self,
        num_episodes: int = 1,
        task_id: Optional[int] = None,
        max_steps: Optional[int] = None,
        render_goal: bool = True,
        video_frame_skip: int = 1,
        temperature: float = 0.0,
        gaussian: Optional[float] = None,
    ) -> RolloutResult:
        """Run headless rollouts and return frames, trajectories, and aggregate stats."""
        if self.env is None:
            raise ValueError('Call load_env(...) before rollout(...).')
        if num_episodes < 1:
            raise ValueError('num_episodes must be at least 1.')
        if video_frame_skip < 1:
            raise ValueError('video_frame_skip must be at least 1.')

        frames = []
        trajectories = []
        raw_stats = []

        for _ in range(num_episodes):
            reset_options = {}
            if task_id is not None:
                reset_options['task_id'] = task_id
            if render_goal:
                reset_options['render_goal'] = True

            if reset_options:
                observation, info = self.env.reset(options=reset_options)
            else:
                observation, info = self.env.reset()

            goal = info.get('goal')
            goal_frame = info.get('goal_rendered')
            episode_frames = []
            traj = {
                'observations': [],
                'actions': [],
                'rewards': [],
                'next_observations': [],
                'dones': [],
                'infos': [],
            }

            done = False
            step = 0
            final_info = info
            while not done:
                action = self._sample_action(observation, goal, temperature, gaussian)
                next_observation, reward, terminated, truncated, info = self.env.step(action)
                done = bool(terminated or truncated)
                step += 1
                if max_steps is not None and step >= max_steps:
                    done = True

                if step % video_frame_skip == 0 or done:
                    frame = self.env.render().copy()
                    if render_goal and goal_frame is not None:
                        frame = np.concatenate([goal_frame, frame], axis=0)
                    episode_frames.append(frame.astype(np.uint8, copy=False))

                traj['observations'].append(observation)
                traj['actions'].append(action)
                traj['rewards'].append(reward)
                traj['next_observations'].append(next_observation)
                traj['dones'].append(done)
                traj['infos'].append(info)

                observation = next_observation
                final_info = info

            frames.append(np.asarray(episode_frames, dtype=np.uint8))
            trajectories.append(traj)
            raw_stats.append(self._episode_stats(final_info, traj))

        return RolloutResult(frames=frames, trajectories=trajectories, stats=self._mean_stats(raw_stats))

    @staticmethod
    def save_video(path: str, frames: np.ndarray, fps: int = 15):
        """Save rollout frames to MP4/GIF using moviepy, imported only when needed."""
        if len(frames) == 0:
            raise ValueError('Cannot save an empty frame sequence.')

        try:
            from moviepy.editor import ImageSequenceClip
        except ImportError:
            from moviepy import ImageSequenceClip

        path = Path(path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)

        clip = ImageSequenceClip(list(frames), fps=fps)
        suffix = path.suffix.lower()
        if suffix == '.gif':
            clip.write_gif(str(path), fps=fps)
        else:
            clip.write_videofile(str(path), fps=fps, codec='libx264', audio=False, logger=None)

    def close(self):
        """Close the current environment, if any."""
        if self.env is not None:
            self.env.close()
            self.env = None

    def _make_env(self, name: str, **env_kwargs):
        if name in self.available_envs():
            return gymnasium.make(name, **env_kwargs)
        return ogbench.make_env_and_datasets(name, env_only=True, **env_kwargs)

    def _policy_frame_stack(self):
        if self.policy_config is None:
            return None
        return self.policy_config.get('frame_stack')

    def _wrap_frame_stack(self, env, frame_stack: int):
        try:
            from utils.env_utils import FrameStackWrapper
        except ImportError as exc:
            raise ImportError('Frame stacking requires ogbench/impls on PYTHONPATH.') from exc
        return FrameStackWrapper(env, frame_stack)

    def _validate_policy_env(self, flags: Dict[str, Any]):
        policy_env_name = flags.get('env_name')
        if policy_env_name is not None and self.env_name != policy_env_name:
            # Dataset-style names can map to the same registered env, so allow both
            # names if they instantiate compatible observation/action spaces.
            policy_env = self._make_env(policy_env_name)
            try:
                self._validate_spaces(self.env, policy_env)
            finally:
                policy_env.close()

    @staticmethod
    def _validate_spaces(env, policy_env):
        if env.observation_space.shape != policy_env.observation_space.shape:
            raise ValueError(
                f'Loaded environment observation shape {env.observation_space.shape} does not match '
                f'policy environment observation shape {policy_env.observation_space.shape}.'
            )
        if type(env.action_space) is not type(policy_env.action_space):
            raise ValueError(
                f'Loaded environment action space {env.action_space} does not match policy action space '
                f'{policy_env.action_space}.'
            )
        if isinstance(env.action_space, Box) and env.action_space.shape != policy_env.action_space.shape:
            raise ValueError(
                f'Loaded environment action shape {env.action_space.shape} does not match '
                f'policy action shape {policy_env.action_space.shape}.'
            )
        if isinstance(env.action_space, Discrete) and env.action_space.n != policy_env.action_space.n:
            raise ValueError(
                f'Loaded environment action count {env.action_space.n} does not match '
                f'policy action count {policy_env.action_space.n}.'
            )

    def _create_agent(self, config: Dict[str, Any], seed: int):
        from agents import agents

        agent_name = config.get('agent_name')
        if agent_name not in agents:
            raise ValueError(f'Unknown agent_name {agent_name!r}. Available agents: {sorted(agents)}')

        example_observations = self._example_observations()
        example_actions = self._example_actions(config)

        agent_class = agents[agent_name]
        return agent_class.create(seed, example_observations, example_actions, config)

    def _example_observations(self):
        space = self.env.observation_space
        if not hasattr(space, 'shape') or space.shape is None:
            raise ValueError(f'Unsupported observation space for policy loading: {space}')
        return np.zeros((1, *space.shape), dtype=space.dtype)

    def _example_actions(self, config: Dict[str, Any]):
        action_space = self.env.action_space
        if config.get('discrete'):
            if not isinstance(action_space, Discrete):
                raise ValueError(f'Policy expects discrete actions, but env action space is {action_space}.')
            return np.asarray([action_space.n - 1], dtype=np.int32)
        if not isinstance(action_space, Box):
            raise ValueError(f'Only Box action spaces are supported for continuous policies; got {action_space}.')
        return np.zeros((1, *action_space.shape), dtype=action_space.dtype)

    @staticmethod
    def _checkpoint_path(run_dir: Path, epoch: Optional[int]):
        if epoch is not None:
            path = run_dir / f'params_{epoch}.pkl'
            if not path.exists():
                raise FileNotFoundError(f'Missing checkpoint: {path}')
            return path

        checkpoints = glob.glob(str(run_dir / 'params_*.pkl'))
        if not checkpoints:
            raise FileNotFoundError(f'No params_*.pkl checkpoints found in {run_dir}')

        def checkpoint_epoch(path):
            stem = Path(path).stem
            return int(stem.split('_')[-1])

        return Path(max(checkpoints, key=checkpoint_epoch))

    def _sample_action(self, observation, goal, temperature: float, gaussian: Optional[float]):
        if self.agent is None:
            return self.env.action_space.sample()

        import jax

        key = jax.random.PRNGKey(int(self.rng.integers(0, 2**31 - 1)))
        action = self.agent.sample_actions(
            observations=observation,
            goals=goal,
            temperature=temperature,
            seed=key,
        )
        action = np.asarray(action)
        if not self.policy_config.get('discrete'):
            if gaussian is not None:
                action = self.rng.normal(action, gaussian)
            action = np.clip(action, -1, 1)
        return action

    @staticmethod
    def _episode_stats(info: Dict[str, Any], traj: Dict[str, List[Any]]):
        stats = {'length': float(len(traj['rewards'])), 'return': float(np.sum(traj['rewards']))}
        for key, value in info.items():
            if isinstance(value, (int, float, np.number, bool)):
                stats[key] = float(value)
        return stats

    @staticmethod
    def _mean_stats(stats: List[Dict[str, float]]):
        keys = sorted({key for stat in stats for key in stat})
        return {
            key: float(np.mean([stat[key] for stat in stats if key in stat]))
            for key in keys
        }
