"""V1-b — 기존 시뮬레이터와의 대조를 회귀 테스트로 고정.

reference/ 사본이 없으면 건너뛴다.
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "reference"))

srm = pytest.importorskip("SRM_simul")
sbf = pytest.importorskip("srm_bates_fast")

from gre.core.ballistics import MotorSpec, simulate
from gre.core.burnback2d import SegmentBurnback2D
from gre.core.burnrate import model_for
from gre.core.grain import Grain, Segment
from gre.core.nozzle import rt_from_cstar
from gre.geometry.port import circular_port

D_G, D_GC, L_G = 60.0, 24.0, 120.0
A_T, DEN, C_STAR, EFF, GAMMA, T_AD, P_ATM = 80.0, 1.841 * 0.92, 885.0, 0.95, 1.133, 1600.0, 0.1013
V_C = np.pi / 4 * D_G ** 2 * (L_G + 40.0)


@pytest.fixture(scope="module")
def runs():
    br = model_for("KNSB")
    sb = SegmentBurnback2D(circular_port(D_GC / 2), d_outer=D_G, n=601)
    grain = Grain([Segment(sb, L_G, True)], density=DEN)
    spec = MotorSpec(grain=grain, burn_rate=br, a_throat=A_T, chamber_volume=V_C,
                     expansion_ratio=4.0, c_star=C_STAR, c_star_eff=EFF,
                     cf_eff=0.95, gamma=GAMMA, p_ambient=P_ATM)
    ours = simulate(spec)

    orig = srm.R_dot_calculation
    srm.R_dot_calculation = lambda a, n, p: float(br.rate(float(p))[0])
    try:
        v_g = srm.V_g_calculation(D_G, D_GC, L_G)
        out = sbf.simulate_bates(V_C, D_G, D_GC, 1, L_G, v_g, DEN,
                                 srm.m_g_calculation(v_g, DEN), A_T, 0.95,
                                 rt_from_cstar(C_STAR, GAMMA) / T_AD, GAMMA,
                                 C_STAR * EFF, T_AD, 3.0, P_ATM, 0.0,
                                 3.907, 0.535, 10.0, 1e-4, verbose=False)
    finally:
        srm.R_dot_calculation = orig
    return ours, grain, out


def test_burn_area_matches_reference(runs):
    ours, grain, out = runs
    w = (out[1] - D_GC) / 2.0
    a_ref = out[5]
    ok = w < 0.95 * grain.web_max()
    d = np.abs(grain.burn_area(w)[ok] - a_ref[ok]) / a_ref[ok].max()
    assert d.max() < 0.01          # 1 %


def test_chamber_pressure_matches_reference(runs):
    ours, _, out = runs
    t_ref, p_ref = out[0], out[11]
    tg = np.linspace(0.05, min(ours.t[-1], t_ref[-1]), 2000)
    p_ours = np.interp(tg, ours.t, ours.p_c)
    p_refi = np.interp(tg, t_ref, p_ref)
    late = tg > 0.5 * tg[-1]
    t_drop = tg[late][np.argmax(np.abs(np.gradient(p_refi, tg)[late]))]
    plateau = (p_refi > 0.5 * p_refi.max()) & (tg < t_drop - 0.05)
    d = np.abs(p_ours - p_refi)[plateau] / p_refi.max()
    # 자유체적 증가항을 넣은 뒤 정상 구간은 0.1 % 수준에서 일치한다
    assert d.mean() < 2e-3
    assert d.max() < 5e-3
    assert abs(ours.p_max - p_ref.max()) / p_ref.max() < 2e-3
