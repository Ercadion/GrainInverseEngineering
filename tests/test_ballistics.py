"""내탄도 — 챔버 충전 ODE 검증."""
import numpy as np
import pytest

from gre.core.ballistics import MotorSpec, quasi_steady_pressure, simulate
from gre.core.burnback2d import SegmentBurnback2D
from gre.core.burnrate import model_for
from gre.core.grain import Grain, Segment
from gre.core.nozzle import (area_ratio_to_pressure_ratio, gamma_function,
                             optimum_area_ratio, rt_from_cstar, thrust_coefficient)
from gre.geometry.port import circular_port

D_OUTER, L_SEG = 60.0, 120.0


def _spec(**kw):
    sb = SegmentBurnback2D(circular_port(12.0), d_outer=D_OUTER, n=401)
    grain = Grain([Segment(sb, L_SEG, ends_burn=True)])
    return MotorSpec(grain=grain, burn_rate=model_for("KNSB"), a_throat=80.0,
                     chamber_volume=np.pi / 4 * D_OUTER ** 2 * (L_SEG + 40.0), **kw)


@pytest.fixture(scope="module")
def spec():
    return _spec()


@pytest.fixture(scope="module")
def result(spec):
    return simulate(spec)


def test_mass_balance(spec, result):
    """∫ṁ_out dt = 추진제 질량 — 적분 전체의 정합성."""
    m_out = float(np.trapezoid(result.mdot_out, result.t)) * 1000.0   # g
    assert abs(m_out - spec.grain.mass()) / spec.grain.mass() < 0.02


def test_pressure_matches_quasi_steady_in_plateau(spec, result):
    """정상 구간에서는 ODE 해가 준정상 평형압과 일치해야 한다."""
    w_mid = 0.5 * spec.grain.web_max()
    p_qs = float(quasi_steady_pressure(spec, w_mid)[0])
    i = int(np.argmin(np.abs(result.web - w_mid)))
    assert abs(result.p_c[i] - p_qs) / p_qs < 0.03


def test_ignition_transient_is_short(spec, result):
    """충전 시상수 τ = c*·V_free/(R·T·A_t) 의 수 배 안에 평형에 도달한다."""
    v_free = (spec.chamber_volume - float(spec.grain.volume(0.0)[0])) * 1e-9
    tau = spec.c_star_actual * v_free / (spec.rt * spec.a_throat * 1e-6)
    p_plateau = float(quasi_steady_pressure(spec, 0.0)[0])
    idx = np.where(result.p_c >= 0.9 * p_plateau)[0]
    assert idx.size > 0
    assert result.t[idx[0]] < 20.0 * tau
    assert tau < 0.1            # 수 ms ~ 수십 ms 수준


def test_burn_time_and_average_thrust(result):
    tb = result.burn_time(0.05)
    assert 0.5 < tb < 10.0
    assert abs(result.average_thrust(0.05) * tb - result.total_impulse) / result.total_impulse < 1e-6


def test_thrust_positive_and_bounded(result):
    assert np.all(result.thrust >= 0.0)
    assert result.f_max < 1e5


def test_nozzle_roundtrip():
    """ε → P_e/P_c → C_f 가 물리적으로 맞는 범위인가."""
    g = 1.133
    for eps in (2.0, 4.0, 8.0):
        x = area_ratio_to_pressure_ratio(eps, g)
        assert 0.0 < x < 1.0
        cf = thrust_coefficient(5.0, 0.1013, eps, g)
        assert 1.0 < cf < 2.0
    # 최적 팽창비를 넣으면 P_e = P_a 가 나와야 한다
    eps_opt = optimum_area_ratio(5.0, 0.1013, g)
    x = area_ratio_to_pressure_ratio(eps_opt, g)
    assert abs(x * 5.0 - 0.1013) / 0.1013 < 1e-3


def test_rt_from_cstar_consistency():
    """c* = √(RT)/Γ 의 역산이 일관적인가."""
    g, c_star = 1.133, 885.0
    rt = rt_from_cstar(c_star, g)
    assert abs(np.sqrt(rt) / gamma_function(g) - c_star) / c_star < 1e-12


def test_cf_matches_mass_flow_derivation():
    """C_f 를 질량유량·배기속도 경로로 독립 재유도해 대조한다 (8장 2겹)."""
    from gre.core.nozzle import (gamma_function, rt_from_cstar,
                                 area_ratio_to_pressure_ratio, thrust_coefficient)
    g, cstar, pa = 1.2, 889.0, 0.101325
    G, rt = gamma_function(g), rt_from_cstar(cstar, g)
    for pc in (2.0, 4.5, 7.0):
        for eps in (3.0, 6.0):
            x = area_ratio_to_pressure_ratio(eps, g)
            cf_direct = thrust_coefficient(pc, pa, eps, g, pe_pc=x)
            at = 1e-4                                   # [m²] 임의
            pc_pa = pc * 1e6
            mdot = pc_pa * at * G / np.sqrt(rt)
            ve = np.sqrt(2 * g / (g - 1) * rt * (1 - x ** ((g - 1) / g)))
            f = mdot * ve + (x * pc_pa - pa * 1e6) * eps * at
            assert abs(cf_direct - f / (pc_pa * at)) < 1e-12


def test_expansion_ratio_does_not_change_pressure():
    """ε 는 추력만 바꾸고 압력에는 영향이 없어야 한다."""
    spec_a = _spec(expansion_ratio=3.0)
    spec_b = _spec(expansion_ratio=9.0)
    ra, rb = simulate(spec_a, t_end=8.0), simulate(spec_b, t_end=8.0)
    assert abs(ra.p_max - rb.p_max) / ra.p_max < 1e-9
    assert abs(ra.f_max - rb.f_max) / ra.f_max > 1e-3


def test_auto_expansion_ratio_is_optimum_at_mean_pressure():
    """ε 미지정이면 작동 구간 평균압의 최적 팽창비가 쓰여야 한다."""
    from gre.core.nozzle import optimum_area_ratio
    res = simulate(_spec(), t_end=8.0)
    assert res.expansion_ratio is not None and res.p_mean is not None
    expect = optimum_area_ratio(res.p_mean, 0.1013, 1.133)
    assert abs(res.expansion_ratio - expect) / expect < 1e-9
