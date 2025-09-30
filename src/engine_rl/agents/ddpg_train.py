from __future__ import annotations

import os

import numpy as np
from stable_baselines3 import DDPG
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from engine_rl.envs.mvem2t_env import MVEM2TEnv
from engine_rl.envs.params import MVEMParams


def make_env(seed: int = 0, drand: bool = False, seconds: float = 10.0):
    params = MVEMParams()
    params.drand.enable = drand
    env = MVEM2TEnv(params, episode_seconds=seconds)
    env = Monitor(env)
    return env


def main():
    seed = 7
    np.random.seed(seed)

    # Векторизация + нормализация наблюдений/награды
    train_env = DummyVecEnv([lambda: make_env(seed=seed, drand=False, seconds=10.0)])
    eval_env = DummyVecEnv([lambda: make_env(seed=seed + 1, drand=True, seconds=10.0)])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=5.0)
    eval_env = VecNormalize(eval_env, training=False, norm_obs=True, norm_reward=False)

    n_actions = train_env.action_space.shape[-1]
    ou_noise = OrnsteinUhlenbeckActionNoise(
        mean=np.zeros(n_actions),
        sigma=0.3 * np.ones(n_actions),  # было 0.2
        theta=0.1,  # было 0.15
        dt=0.01,  # шаг симуляции (если менял dt — поставь self.p.time.dt)
    )

    model = DDPG(
        "MlpPolicy",
        train_env,
        learning_rate=5e-4,  # чуть мягче
        buffer_size=200_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=(1, "step"),
        gradient_steps=2,  # было 1
        learning_starts=10_000,  # прогрев
        action_noise=ou_noise,
        policy_kwargs=dict(net_arch=[64, 64]),
        tensorboard_log="runs/ddpg",
        verbose=1,
        seed=seed,
    )

    os.makedirs("models", exist_ok=True)
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path="models",
        log_path="models",
        eval_freq=2000,
        deterministic=True,
    )
    ckpt_cb = CheckpointCallback(save_freq=10_000, save_path="models", name_prefix="ddpg_ckpt")

    model.learn(total_timesteps=200_000, callback=[eval_cb, ckpt_cb])
    model.save("models/ddpg_last")

    # Сохраняем статистику нормализации, чтобы использовать на валидации/в проде
    train_env.save("models/vecnorm_train.pkl")


if __name__ == "__main__":
    main()
