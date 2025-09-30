from engine_rl.envs.params import MVEMParams


def test_units_helpers():
    p = MVEMParams()
    assert abs(p.pa_to_bar(101325) - 1.01325) < 1e-6
    rpm = 6000.0
    assert abs(p.rps_to_rpm(p.rpm_to_rps(rpm)) - rpm) < 1e-9
