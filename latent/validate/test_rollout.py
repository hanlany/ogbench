import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault('MUJOCO_GL', 'egl')

REPO_ROOT = Path(__file__).resolve().parents[2]
IMPLS_DIR = REPO_ROOT / 'impls'
for path in (REPO_ROOT, IMPLS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from agents import SACAgent  # noqa: E402
from latent.vis import OGBenchVisualizer  # noqa: E402
from utils.flax_utils import restore_agent  # noqa: E402


class AntMazeExpertPolicy:
    """Adapter for the directional ant SAC expert used by generate_locomaze.py."""

    def __init__(self, agent, env):
        self.agent = agent
        self.env = env

    def sample_actions(self, observations, goals=None, seed=None, temperature=0.0):
        del observations, goals

        env = self.env.unwrapped
        subgoal_xy, _ = env.get_oracle_subgoal(env.get_xy(), env.cur_goal_xy)
        subgoal_dir = subgoal_xy - env.get_xy()
        subgoal_dir = subgoal_dir / (np.linalg.norm(subgoal_dir) + 1e-6)

        agent_ob = env.get_ob(ob_type='states')
        agent_ob = np.concatenate([agent_ob[2:], subgoal_dir])
        return self.agent.sample_actions(
            observations=agent_ob,
            temperature=temperature,
            seed=seed,
        )


def load_ant_expert(env, seed=0):
    expert_dir = REPO_ROOT / 'data_gen_scripts' / 'experts' / 'ant'
    with (expert_dir / 'flags.json').open('r') as f:
        agent_config = json.load(f)['agent']

    agent = SACAgent.create(
        seed,
        np.zeros(env.observation_space.shape[0]),
        env.action_space.sample(),
        agent_config,
    )
    return restore_agent(agent, str(expert_dir), 400000)


def main():
    vis = OGBenchVisualizer(seed=0)
    vis.load_env('antmaze-large-navigate-v0')

    ant_expert = load_ant_expert(vis.env, seed=0)
    vis.agent = AntMazeExpertPolicy(ant_expert, vis.env)
    vis.policy_config = {'discrete': False}

    result = vis.rollout(task_id=5, num_episodes=1, max_steps=1000, video_frame_skip=3)
    vis.save_video('antmaze_large_expert_rollout.mp4', result.frames[0])
    vis.close()

    print(result.stats)


if __name__ == '__main__':
    main()
