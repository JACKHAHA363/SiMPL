import gym
import simpl.env.ml43
from simpl.env.ml43 import ML43Tasks

env = gym.make('simpl-ml43-v0')
tasks = ML43Tasks.test_tasks
config = dict(
    constrained_sac=dict(auto_alpha=True, kl_clip=20,
                         target_kl=5, increasing_alpha=True,
                         policy_lr=5e-5, qf_lr=5e-5, alpha_lr=5e-5),
    buffer_size=20000,
    n_prior_episode=20,
    time_limit=280,
    n_episode=5000,
    train=dict(batch_size=256, reuse_rate=256)
)
visualize_env = None
