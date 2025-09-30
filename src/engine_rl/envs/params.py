from __future__ import annotations

from dataclasses import asdict, dataclass, field


# ---------- Универсальные физические константы ----------
@dataclass
class Physical:
    R: float = 287.05
    k: float = 1.4
    Lth: float = 14.7
    p0: float = 101325.0
    T0: float = 298.15
    g: float = 9.80665


# ---------- Сетка времени ----------
@dataclass
class Timing:
    dt: float = 0.01
    inj_delay_steps: int = 2
    episode_seconds: float = 10.0


# ---------- Геометрия ----------
@dataclass
class Geometry:
    Vm: float = 1.0e-3
    Vz: float = 6.0e-4
    D: float = 0.036
    Vd: float = 5.7e-5
    i: int = 1
    t: int = 2


# ---------- Дроссель ----------
@dataclass
class Throttle:
    mat1: float = 1.0
    a1: float = 1.4073
    a2: float = 0.4087
    p1: float = 0.4404
    p2: float = 2.3143
    pn: float = 0.7404
    pc: float = 0.4125


# ---------- Порт (Hendricks) ----------
@dataclass
class PortFlow:
    n_unit: str = "rpm"
    pm_unit: str = "bar"
    c0: float = -0.366
    c1: float = 0.08979
    c2: float = -0.0337
    c3: float = 0.0001


# ---------- Плёночная модель ----------
@dataclass
class FuelFilm:
    use_regressions: bool = True
    tau_ff_const: float = 0.6
    x_const: float = 0.4


# ---------- Вал / мощности ----------
@dataclass
class Shaft:
    J: float = 0.015
    Hu: float = 43e6
    kf: float = 0.05
    eta_i: float = 0.25  # <-- добавили сюда
    hiu: float = 0.98


@dataclass
class ShaftLoss:
    c0: float = 0.1  # Н*м эквивалент (через перевод в мощность выйдет Вт); подберём
    c1: float = 0.02  # Н*м·с
    c2: float = 2e-5  # Н*м·с^2
    k_pump: float = 0.5  # безразмерный масштаб для P_p


# в MVEMParams:
loss: ShaftLoss = field(default_factory=ShaftLoss)


# ---------- Ограничения ----------
@dataclass
class Limits:
    n_min_rpm: float = 100.0
    n_max_rpm: float = 9000.0
    a_min: float = 0.0
    a_max: float = 1.0
    mfi_min: float = 0.0
    mfi_max: float = 3.0e-3
    lambda_min: float = 0.6
    lambda_max: float = 1.6


# ---------- Домрандомизация ----------
@dataclass
class DomainRand:
    enable: bool = True
    J_pct: float = 0.2
    Hu_pct: float = 0.1
    Vm_pct: float = 0.2
    Vz_pct: float = 0.2
    kf_pct: float = 0.2


# ---------- Стартовые условия ----------
@dataclass
class Operating:
    n0_rps: float = 300.0  # стартовая «скорость» в rps (эквивалент)
    a0: float = 0.4  # стартовый дроссель
    lambda0: float = 1.0


# ---------- Групповой контейнер ----------
@dataclass
class MVEMParams:
    phys: Physical = field(default_factory=Physical)
    time: Timing = field(default_factory=Timing)
    geo: Geometry = field(default_factory=Geometry)
    thr: Throttle = field(default_factory=Throttle)
    port: PortFlow = field(default_factory=PortFlow)
    film: FuelFilm = field(default_factory=FuelFilm)
    shaft: Shaft = field(default_factory=Shaft)
    loss: ShaftLoss = field(default_factory=ShaftLoss)
    lim: Limits = field(default_factory=Limits)
    drand: DomainRand = field(default_factory=DomainRand)
    op: Operating = field(default_factory=Operating)

    def to_dict(self) -> dict:  # вспомогательно
        return asdict(self)

    # --- конвертеры ---
    @staticmethod
    def rpm_to_rps(n_rpm: float) -> float:
        return n_rpm / 60.0

    @staticmethod
    def rps_to_rpm(n_rps: float) -> float:
        return n_rps * 60.0

    @staticmethod
    def pa_to_bar(p_pa: float) -> float:
        return p_pa / 1e5

    @staticmethod
    def bar_to_pa(p_bar: float) -> float:
        return p_bar * 1e5

    # Hendricks inputs (формула (6))
    def hendricks_inputs(self, n_rps: float, pm_pa: float) -> tuple[float, float]:
        n_rpm = self.rps_to_rpm(n_rps)
        pm_bar = self.pa_to_bar(pm_pa)
        n = n_rpm / 1000.0 if self.port.n_unit == "krpm" else n_rpm
        return n, pm_bar
