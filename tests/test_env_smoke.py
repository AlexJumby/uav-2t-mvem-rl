import numpy as np

from engine_rl.envs.mvem2t_env import MVEM2TEnv
from engine_rl.envs.params import MVEMParams


def test_env_reset_and_step():
    env = MVEM2TEnv(MVEMParams())
    obs, info = env.reset(seed=42)
    assert obs.shape == (4,)
    assert 0.0 <= obs[2] <= 1.0  # throttle
    assert 0.5 <= obs[3] <= 2.0  # pm in bar

    # прогоним несколько шагов
    for _ in range(10):
        action = np.array([0.0], dtype=np.float32)
        obs, r, term, trunc, info = env.step(action)
        assert obs.shape == (4,)
        assert isinstance(r, float)
        assert term in [False, True]
        assert trunc in [False, True]


def test_env_episode_length():
    env = MVEM2TEnv(MVEMParams(), episode_seconds=0.1)
    obs, _ = env.reset(seed=1)
    steps = 0
    done = False
    while not done:
        action = np.array([0.5], dtype=np.float32)
        obs, r, term, trunc, _ = env.step(action)
        steps += 1
        done = term or trunc
    assert steps == env.steps_per_ep
