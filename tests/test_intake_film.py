from engine_rl.envs.mvem2t_env import MVEM2TEnv
from engine_rl.envs.params import MVEMParams


def test_massflows_nonnegative():
    env = MVEM2TEnv(MVEMParams())
    pm = env.p.phys.p0
    n = env.p.op.n0_rps
    assert env._m_at(0.5, pm) >= 0.0
    assert env._m_as(n, pm) >= 0.0


def test_dp_m_sign():
    env = MVEM2TEnv(MVEMParams())
    pm_low = env.p.phys.p0 * 0.9
    dpm = env._dp_m(pm_low, a=0.7, n_rps=env.p.op.n0_rps)
    assert dpm > 0.0  # при низком давлении и приоткрытом дросселе растет


def test_fuel_film_step_response():
    env = MVEM2TEnv(MVEMParams())
    mfi = 1e-3
    mfs = []
    for _ in range(200):
        mfs.append(env._fuel_film_step(mfi))
    # стационарное mf должно быть ~ (x + (1-x)) * mfi = mfi (без утечек)
    assert 0.8e-3 < mfs[-1] < 1.2e-3
