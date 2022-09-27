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


class ML43Env:
    """ A wrapper of ML43 """

    @property
    def action_space(self):
        return self._env.action_space

    @property
    def observation_space(self):
        return self._env.observation_space

    def __init__(self):
        self.task = None
        self._env = None

    @contextmanager
    def set_task(self, task):
        if type(task) != ML43Task:
            raise TypeError(f'task should ML43Task but {type(task)} is given')
    
        prev_task = self.task
        self.task = task
        yield
        self.task = prev_task
        
    def set_render_options(self, width, height, device, fps=30, frame_drop=1):
        raise NotImplementedError

    def reset_model(self):
        if self.task is None:
            raise RuntimeError('task is not set')
        goal_observable_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[self.task.name + "-goal-observable"]
        self._env = goal_observable_cls(seed=1)  
        self._env.max_path_length = 200
        return self._env.reset()

    def step(self, action):
        if self.task is None:
            raise RuntimeError('task is not set')
        return self._env.step(action)
