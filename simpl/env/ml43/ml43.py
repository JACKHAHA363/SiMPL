from contextlib import contextmanager
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import SawyerXYZEnv
import gym
from gym.spaces import Box
import numpy as np


class ML43Task:
    def __init__(self, name):
        self.name = name
    def __repr__(self):
        return f"ML43Task({self.name})"

_INITIALIZED_ENVS = {}


def get_metaworld_env(name):
    """ return env if not initialized """
    if name not in _INITIALIZED_ENVS:
        max_ep_len = 200
        env_name = name + '-goal-observable'
        print('Initializing -------task_name', env_name)
        goal_observable_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[env_name]
        env = goal_observable_cls(seed=1)  
        env.max_path_length = max_ep_len
        _INITIALIZED_ENVS[name] = env
    return _INITIALIZED_ENVS[name]

class ML43Env(gym.Env):
    """ A wrapper of ML43 """

    @property
    def action_space(self):
        return self._env.action_space

    @property
    def observation_space(self):
        return self._env.observation_space

    @property
    def _env(self):
        return get_metaworld_env(self.task)

    def __init__(self):
        self.task = "basketball-v2"

    @contextmanager
    def set_task(self, task):
        if type(task) != ML43Task:
            raise TypeError(f'task should ML43Task but {type(task)} is given')

        # Just change the inner env
        prev_task = self.task
        self.task = task.name
        yield
        self.task = prev_task

    def reset(self):
        return self._env.reset()

    def step(self, action):
        if self.task is None:
            raise RuntimeError('task is not set')
        return self._env.step(action)
