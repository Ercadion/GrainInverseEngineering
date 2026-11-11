"""설계 대역(3.79~7.03 MPa)에서 두 구현의 압력 차이 원인 분해.

2.3 MPa 케이스에서는 0.4% 였던 최대압 차이가 4.5 MPa 케이스에서 1.8% 로
커졌다. 원인 후보는 넷이다.

    (a) 기하(A_b) 차이            (b) 연소속도 모델 차이
    (c) 시간 적분 방식·스텝 차이   (d) 챔버 모델 차이

각각을 따로 떼어 본다. 판정 기준은 **준정상 평형압** 이다. 평형은
ṁ_gen = ṁ_out 의 대수 방정식이라 적분 방식과 무관한 제3의 기준이 된다.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "reference"))

import numpy as np

import SRM_simul as srm
import srm_bates_fast as sbf

from gre.core.ballistics import MotorSpec, quasi_steady_pressure, simulate
from gre.core.burnback2d import SegmentBurnback2D
from gre.core.burnrate import model_for
from gre.core.grain import Grain, Segment
from gre.core.nozzle import rt_from_cstar
from gre.geometry.port import circular_port

D_G, D_GC, L_G = 60.0, 24.0, 120.0
A_T = 45.0                      # 설계 대역으로 올리기 위해 목을 줄였다
DEN, C_STAR, EFF, GAMMA, T_AD, P_ATM = 1.841 * 0.92, 885.0, 0.95, 1.133, 1600.0, 0.1013
V_C = np.pi / 4 * D_G ** 2 * (L_G + 40.0)

br = model_for("KNSB")
sb = SegmentBurnback2D(circular_port(D_GC / 2), d_outer=D_G, n=601)
grain = Grain([Segment(sb, L_G, True)], density=DEN)
spec = MotorSpec(grain=grain, burn_rate=br, a_throat=A_T, chamber_volume=V_C,
                 expansion_ratio=4.0, c_star=C_STAR, c_star_eff=EFF, cf_eff=0.95,
                 gamma=GAMMA, p_ambient=P_ATM)


def run_ref(dt, patch=True):
    orig = srm.R_dot_calculation
    if patch:
        srm.R_dot_calculation = lambda a, n, p: float(br.rate(float(p))[0])
    try:
        vg = srm.V_g_calculation(D_G, D_GC, L_G)
        return sbf.simulate_bates(V_C, D_G, D_GC, 1, L_G, vg, DEN,
                                  srm.m_g_calculation(vg, DEN), A_T, 0.95,
                                  rt_from_cstar(C_STAR, GAMMA) / T_AD, GAMMA,
                                  C_STAR * EFF, T_AD, 5.0, P_ATM, 0.0,
                                  3.907, 0.535, 10.0, dt, verbose=False)
    finally:
        srm.R_dot_calculation = orig


print("=" * 78)
print("(a) 기하 — 같은 웹에서 A_b 비교")
out = run_ref(1e-4)
w_ref = (out[1] - D_GC) / 2.0
ok = w_ref < 0.95 * grain.web_max()
d_a = np.abs(grain.burn_area(w_ref)[ok] - out[5][ok]) / out[5][ok].max()
print(f"    A_b 최대 차이 {d_a.max()*100:.3f} %  → 기하는 원인이 아니다")

print("\n(b) 연소속도 모델 — 기존 코드를 구간표 그대로 둔 경우와 비교")
out_raw = run_ref(1e-4, patch=False)
print(f"    P_max  통일 {out[11].max():.4f} MPa   구간표 {out_raw[11].max():.4f} MPa"
      f"   ({abs(out[11].max()-out_raw[11].max())/out[11].max()*100:.2f} %)")

print("\n(c) 시간 적분 — 기존 코드의 스텝을 줄이며 수렴 확인")
for dt in (2e-4, 1e-4, 3e-5, 1e-5):
    o = run_ref(dt)
    print(f"    dt = {dt:.0e} s →  P_max {o[11].max():.4f} MPa")

print("\n    우리 쪽 적분 허용오차를 조이며 확인")
for ms in (2e-3, 5e-4, 1e-4):
    r = simulate(spec, max_step=ms)
    print(f"    max_step = {ms:.0e} s →  P_max {r.p_max:.4f} MPa")

print("\n(d) 제3의 기준 — 준정상 평형압 (적분 방식과 무관한 대수 해)")
ours = simulate(spec)
i_peak_ours = int(np.argmax(ours.p_c))
w_peak = ours.web[i_peak_ours]
p_eq = float(quasi_steady_pressure(spec, w_peak)[0])
# 기존 구현의 같은 웹 지점
w_ref_all = (out[1] - D_GC) / 2.0
j = int(np.argmin(np.abs(w_ref_all - w_peak)))
print(f"    웹 {w_peak:.2f} mm 에서")
print(f"      평형해(기준)  {p_eq:.4f} MPa")
print(f"      우리 구현     {ours.p_c[i_peak_ours]:.4f} MPa  ({(ours.p_c[i_peak_ours]-p_eq)/p_eq*100:+.2f} %)")
print(f"      기존 구현     {out[11][j]:.4f} MPa  ({(out[11][j]-p_eq)/p_eq*100:+.2f} %)")
print("=" * 78)
