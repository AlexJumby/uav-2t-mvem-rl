from __future__ import annotations

import glob
import os

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import TD3

from engine_rl.envs.mvem2t_env import MVEM2TEnv
from engine_rl.envs.params import MVEMParams


# --- простые метрики ---
def iae(y: np.ndarray, ref: float = 1.0) -> float:
    return float(np.sum(np.abs(y - ref)))


def overshoot(y: np.ndarray, ref: float = 1.0) -> float:
    return float(max(0.0, np.max(y) - ref))


def violations(y: np.ndarray, lo: float = 0.9, hi: float = 1.1) -> int:
    return int(np.sum((y < lo) | (y > hi)))


def find_model_path() -> str:
    cand = []
    cand += glob.glob("models/best_model.zip")
    cand += glob.glob("models/td3_last.zip")
    cand += glob.glob("models/*.zip")
    if not cand:
        raise FileNotFoundError("Не найден *.zip в папке models/. Сначала запусти обучение.")
    # берём самый свежий
    cand.sort(key=os.path.getmtime, reverse=True)
    return cand[0]


def rollout(model_path: str, seconds: float = 10.0, seed: int = 0):
    env = MVEM2TEnv(MVEMParams(), episode_seconds=seconds)
    obs, _ = env.reset(seed=seed)
    model = TD3.load(model_path)
    T = int(seconds / env.p.time.dt)

    n_hist, lam_hist, a_hist, pm_hist, act_hist, r_hist = [], [], [], [], [], []
    for _ in range(T):
        action, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, _ = env.step(action)
        n_hist.append(obs[0])
        lam_hist.append(obs[1])
        a_hist.append(obs[2])
        pm_hist.append(obs[3])
        act_hist.append(float(action[0]))
        r_hist.append(r)
        if term or trunc:
            break

    t = np.arange(len(n_hist)) * env.p.time.dt
    return (
        t,
        np.array(n_hist),
        np.array(lam_hist),
        np.array(a_hist),
        np.array(pm_hist),
        np.array(act_hist),
        np.array(r_hist),
    )


def main():
    model_path = find_model_path()
    print(f"Using model: {model_path}")
    t, n, lam, a, pm, act, r = rollout(model_path, seconds=10.0, seed=0)

    # метрики
    print(
        f"IAE(λ): {iae(lam):.3f}, Overshoot: {overshoot(lam):.3f}, Violations±10%: {violations(lam,0.9,1.1)}"
    )

    # графики
    os.makedirs("outputs", exist_ok=True)
    plt.figure()
    plt.plot(t, lam)
    plt.axhline(1.0, linestyle="--")
    plt.title("Lambda")
    plt.xlabel("s")
    plt.ylabel("λ")
    plt.figure()
    plt.plot(t, n)
    plt.title("Speed n (rps)")
    plt.xlabel("s")
    plt.figure()
    plt.plot(t, a)
    plt.title("Throttle a")
    plt.xlabel("s")
    plt.figure()
    plt.plot(t, pm)
    plt.title("Manifold pressure (bar)")
    plt.xlabel("s")
    plt.figure()
    plt.plot(t, act)
    plt.title("Action u")
    plt.xlabel("s")
    plt.figure()
    plt.plot(t, r)
    plt.title("Reward")
    plt.xlabel("s")
    plt.tight_layout()
    plt.savefig("outputs/eval_summary.png", dpi=150)


if __name__ == "__main__":
    main()
