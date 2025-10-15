from __future__ import annotations

from dataclasses import asdict, dataclass, field


# ---------- Универсальные физические константы ----------
@dataclass
class Physical:
    R: float = 287.05  # J/(kg·K) газовая постоянная для воздуха
    k: float = 1.4  # Cp/Cv для воздуха
    Lth: float = 14.7  # стехиометрический AFR (массовый) для бензина
    p0: float = 101325.0  # Па, атмосферное давление
    T0: float = 298.15  # K, температура окружающей среды (25 °C)
    g: float = 9.80665  # м/с², ускорение свободного падения


# ---------- Сетка времени ----------
@dataclass
class Timing:
    dt: float = 0.01  # с, шаг интегрирования
    inj_delay_steps: int = 2  # задержка впрыска в шагах
    episode_seconds: float = 10.0  # длительность эпизода по умолчанию


# ---------- Геометрия ----------
@dataclass
class Geometry:
    Vm: float = 1.0e-3  # м^3, объём ВПУСКНОГО КОЛЛЕКТОРА (используется в dp_m)
    Vz: float = 6.0e-4  # м^3, «камера/картер» (для 2Т — объём под поршнем), если понадобится
    D: float = 0.036  # м, диаметр цилиндра
    Vd: float = 5.7e-5  # м^3, объём цилиндра (рабочий объём)
    i: int = 1  # число цилиндров
    t: int = 2  # число тактов (2-тактный двигатель)


# ---------- Дроссель ----------
@dataclass
class Throttle:
    mat1: float = 1.0  # кг/(с·Па^0.5), масштаб в (2)
    a1: float = 1.4073  # коэф. в формуле (3)
    a2: float = 0.4087  # коэф. в формуле (3)
    p1: float = 0.4404  # коэф. в формуле (4)
    p2: float = 2.3143  # коэф. в формуле (4)
    pn: float = 0.7404  # коэф. в формуле (4)
    pc: float = 0.4125  # критическое отношение давлений (формула (4))


# ---------- Порт (Hendricks) ----------
@dataclass
class PortFlow:
    n_unit: str = "rpm"  # какие единицы ожидает регрессия по n: "rpm" или "krpm"
    pm_unit: str = "bar"  # какие единицы ожидает регрессия по p_m: "bar" или "Pa"
    c0: float = -0.366  # коэф. в формуле (6)
    c1: float = 0.08979  # коэф. в формуле (6)
    c2: float = -0.0337  # коэф. в формуле (6)
    c3: float = 0.0001  # коэф. в формуле (6)


# ---------- Плёночная модель ----------
@dataclass
class FuelFilm:
    use_regressions: bool = True  # пока используем константы; позже можно ввести зависимости
    tau_ff_const: float = 0.6  # с, постоянная времени плёночной модели (Aquino)
    x_const: float = 0.4  # доля мгновенного испарения


# ---------- Вал / мощности ----------
@dataclass
class Shaft:
    J: float = 0.015  # кг·м², момент инерции вращающихся масс
    Hu: float = 43e6  # Дж/кг, удельная теплота сгорания топлива
    kf: float = 0.05  # долю потерь/продувки топлива (безразмерная)
    eta_i: float = 0.25  # <-- добавили сюда -# КПД двигателя (внутренний)
    hiu: float = 0.98  # КПД топливной системы (впрыска)


@dataclass
class ShaftLoss:
    c0: float = 0.1  # Н*м сухое трение (эквивалент)
    c1: float = 0.02  # Н*м·с — вязкое трение
    c2: float = 2e-5  # Н*м·с^2 — квадратичные потери (аэродин., накачка)
    k_pump: float = 0.5  # безразмерный масштаб мощности «помпы»


# ---------- Ограничения ----------
@dataclass
class Limits:
    n_min_rpm: float = 100.0  # минимальная скорость вращения в rpm
    n_max_rpm: float = 9000.0  # максимальная скорость вращения в rpm
    a_min: float = 0.0  # минимальное положение дросселя
    a_max: float = 1.0  # максимальное положение дросселя
    mfi_min: float = 1.0e-3  # кг/с — реалистичный низ (ориентир по практике/статье)
    mfi_max: float = 5.0e-3  # кг/с — реалистичный верх
    lambda_min: float = 0.6  # минимальное значение λ
    lambda_max: float = 1.6  # максимальное значение λ


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
    n0_rps: float = 5000.0 / 60.0  # ≈83.33 rps (≈5000 rpm)
    a0: float = 0.4  # стартовый дроссель
    lambda0: float = 1.0  # стартовая AFR


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
