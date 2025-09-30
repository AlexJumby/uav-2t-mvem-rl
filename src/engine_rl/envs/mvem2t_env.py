from __future__ import annotations

import math
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from engine_rl.envs.params import MVEMParams


class MVEM2TEnv(gym.Env):
    """
    AFR-only среда:
      действие: u∈[-1,1] → m_fi∈[0,mfi_max]
      наблюдение: [n_rps, lambda, a, p_m_bar]
    """

    metadata = {"render_modes": []}

    def __init__(self, params: MVEMParams | None = None, episode_seconds: float | None = None):
        super().__init__()
        self.p = params or MVEMParams()
        self.steps_per_ep = int((episode_seconds or self.p.time.episode_seconds) / self.p.time.dt)

        # --- spaces ---
        self.mfi_max = self.p.lim.mfi_max
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        # obs = [n_rps, lambda, a, p_m_bar]
        high = np.array([1_000.0, 3.0, 1.0, 2.0], dtype=np.float32)
        low = np.array([0.0, 0.2, 0.0, 0.5], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, shape=(4,), dtype=np.float32)

        # --- internal state ---
        self._t = 0
        self._n = self.p.op.n0_rps
        self._lambda = self.p.op.lambda0
        self._a = self.p.op.a0
        self._pm = self.p.phys.p0  # Pa
        self._delay = [0.0] * self.p.time.inj_delay_steps
        self._rng = np.random.default_rng()
        self._mff = 0.0  # масса топлива на стенке (экв.)

    # ---------- Helpers ----------
    def _rho_from_pmT(self, pm_pa: float, T: float) -> float:
        return pm_pa / (self.p.phys.R * T)

    # ---------- Fuel film (Aquino) ----------
    def _fuel_film_step(self, mfi: float) -> float:
        tau_ff = self.p.film.tau_ff_const
        x = self.p.film.x_const
        dmff = -self._mff / tau_ff + x * mfi
        self._mff += self.p.time.dt * dmff
        mf = self._mff / tau_ff + (1 - x) * mfi
        return float(max(mf, 0.0))

    # ---------- Throttle / intake (2)-(5) ----------
    def _b1(self, a: float) -> float:
        return 1.0 - self.p.thr.a1 * np.cos(a) + self.p.thr.a2 * (np.cos(a) ** 2)

    def _b2(self, pr: float) -> float:
        pr = float(np.clip(pr, 1e-6, 1.0))
        if pr <= self.p.thr.pc:
            b2 = ((pr**self.p.thr.p1) - (pr**self.p.thr.p2)) ** 0.5 / self.p.thr.pn
            return float(max(b2, 0.0))
        return 1.0

    def _m_at(self, a: float, pm_pa: float) -> float:
        pr = max(pm_pa / self.p.phys.p0, 1e-6)
        return float(
            max(
                self.p.thr.mat1
                * (self.p.phys.p0 / (self.p.phys.T0**0.5))
                * self._b1(a)
                * self._b2(pr),
                0.0,
            )
        )

    # ---------- Port flow to cylinder (Hendricks, 6) ----------
    def _m_as(self, n_rps: float, pm_pa: float) -> float:
        n, pm = self.p.hendricks_inputs(n_rps, pm_pa)  # n в rpm/krpm, pm в bar
        c0, c1, c2, c3 = self.p.port.c0, self.p.port.c1, self.p.port.c2, self.p.port.c3
        npm = n * pm
        mas = c0 + c1 * npm + c2 * (npm**2) + c3 * (n**2) * pm
        return float(max(mas, 0.0))

    # ---------- Manifold ODE (1) ----------
    def _dp_m(self, pm_pa: float, a: float, n_rps: float) -> float:
        Tm = self.p.phys.T0
        Vm = self.p.geo.Vm
        return (self.p.phys.R * Tm / Vm) * (self._m_at(a, pm_pa) - self._m_as(n_rps, pm_pa))

    # ---------- Shaft losses power ----------
    def _shaft_losses_power(self, omega: float, m_as_air: float, pm_pa: float) -> float:
        # Friction torque ~ c0 + c1*ω + c2*ω^2 → Pf = τ_f * ω
        tau_f = self.p.loss.c0 + self.p.loss.c1 * omega + self.p.loss.c2 * (omega**2)
        Pf = max(tau_f * omega, 0.0)
        # Pumping power ~ k * Δp * volumetric_flow
        rho_m = self._rho_from_pmT(pm_pa, self.p.phys.T0)
        vol_flow = m_as_air / max(rho_m, 1e-6)  # м^3/с
        Pp = self.p.loss.k_pump * max(pm_pa - self.p.phys.p0, 0.0) * vol_flow
        return Pf + max(Pp, 0.0)

    # ---------- Profiles ----------
    def _a_profile(self, k: int) -> float:
        return 0.46 if k < self.steps_per_ep // 2 else 0.40

    # ---------- One integration step ----------
    def _step_dynamics(self, mfi_cmd: float):
        # задержка впрыска
        self._delay.append(mfi_cmd)
        mfi = self._delay.pop(0)
        mf = self._fuel_film_step(mfi)

        # впуск
        self._pm += self.p.time.dt * self._dp_m(self._pm, self._a, self._n)
        self._pm = float(np.clip(self._pm, 0.5 * self.p.phys.p0, 1.2 * self.p.phys.p0))
        m_air = self._m_as(self._n, self._pm)

        # AFR
        eps = 1e-8
        self._lambda = float(
            np.clip(
                m_air / (mf * self.p.phys.Lth + eps), self.p.lim.lambda_min, self.p.lim.lambda_max
            )
        )

        # валовая динамика через мощности
        omega = max(self._n * 2.0 * math.pi, 1.0)  # рад/с
        P_in = self.p.shaft.Hu * (1 - self.p.shaft.kf) * mf * self.p.shaft.eta_i  # Вт
        P_loss = self._shaft_losses_power(omega, m_air, self._pm)  # Вт
        domega = (P_in - P_loss) / (self.p.shaft.J * omega)
        self._n += self.p.time.dt * domega / (2.0 * math.pi)
        self._n = float(np.clip(self._n, 0.0, 1000.0))

    # ---------- Gym API ----------
    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict]:
        self._rng = np.random.default_rng(seed)
        self._t = 0

        # domain randomization (робастность)
        if self.p.drand.enable:

            def scale(x, pct):
                return x * float(self._rng.uniform(1.0 - pct, 1.0 + pct))

            self.p.shaft.J = scale(self.p.shaft.J, self.p.drand.J_pct)
            self.p.shaft.Hu = scale(self.p.shaft.Hu, self.p.drand.Hu_pct)
            self.p.geo.Vm = scale(self.p.geo.Vm, self.p.drand.Vm_pct)
            self.p.shaft.kf = scale(self.p.shaft.kf, self.p.drand.kf_pct)

        self._n = self.p.op.n0_rps * float(self._rng.uniform(0.9, 1.1))
        self._a = float(self._rng.uniform(0.35, 0.5))
        self._lambda = float(self._rng.uniform(0.9, 1.1))
        self._pm = self.p.phys.p0 * float(self._rng.uniform(0.95, 1.05))
        self._delay = [0.0] * self.p.time.inj_delay_steps
        self._mff = 0.0

        obs = np.array(
            [self._n, self._lambda, self._a, self.p.pa_to_bar(self._pm)], dtype=np.float32
        )
        return obs, {}

    def step(self, action: np.ndarray):
        self._t += 1
        self._a = self._a_profile(self._t)

        # squash: [-1,1] → [0, mfi_max]
        u = float(np.clip(action[0], -1.0, 1.0))
        mfi_cmd = (u + 1.0) / 2.0 * self.mfi_max

        self._step_dynamics(mfi_cmd)

        obs = np.array(
            [self._n, self._lambda, self._a, self.p.pa_to_bar(self._pm)], dtype=np.float32
        )
        # reward: удержание λ≈1 + мягкий штраф за энергию действия и выход из коридора
        err = abs(self._lambda - 1.0)
        r = -(err + 0.5 * max(0.0, err - 0.1)) - 0.01 * (u**2)

        terminated = False
        truncated = self._t >= self.steps_per_ep
        info = {}
        return obs, float(r), terminated, truncated, info
