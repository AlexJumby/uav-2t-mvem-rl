from __future__ import annotations

import os

import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv

from engine_rl.envs.mvem2t_env import MVEM2TEnv
from engine_rl.envs.params import MVEMParams


def make_env(seed: int = 0, drand: bool = False, seconds: float = 10.0):
    params = MVEMParams()
    params.drand.enable = drand
    env = MVEM2TEnv(params, episode_seconds=seconds)
    env = Monitor(env)
    env.reset(seed=seed)
    return env


def main():
    seed = 42
    np.random.seed(seed)

    # векторизуем (SB3 любит VecEnv)
    train_env = DummyVecEnv([lambda: make_env(seed=seed, drand=False, seconds=10.0)])
    eval_env = DummyVecEnv([lambda: make_env(seed=seed + 1, drand=True, seconds=10.0)])

    # шум на действие
    n_actions = train_env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    model = TD3(
        "MlpPolicy",
        train_env,
        learning_rate=1e-3,
        buffer_size=200_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=(1, "step"),
        gradient_steps=1,
        action_noise=action_noise,
        policy_kwargs=dict(net_arch=[64, 64]),
        tensorboard_log="runs/td3",
        verbose=1,
        seed=seed,
    )

    os.makedirs("models", exist_ok=True)
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path="models",
        log_path="models",
        eval_freq=2_000,
        deterministic=True,
    )
    ckpt_cb = CheckpointCallback(save_freq=10_000, save_path="models", name_prefix="td3_ckpt")

    total_steps = 200_000
    model.learn(total_timesteps=total_steps, callback=[eval_cb, ckpt_cb])

    model.save("models/td3_last")


if __name__ == "__main__":
    main()
