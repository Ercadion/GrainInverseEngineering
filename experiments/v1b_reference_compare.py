"""V1-b — 기존 시뮬레이터(reference/SRM_simul)와의 독립 구현 대조.

공정한 비교를 위해 **연소속도 모델을 같게 맞춘 뒤** 기하·적분 차이만 본다.
기존 코드의 `R_dot_calculation` 은 인자 a, n 을 무시하고 Nakka 구간표를
하드코딩으로 쓰므로, 우리 전 구간 연속 모델로 monkey-patch 한다.

⚠ 추력계수 규약이 다르다. 기존 코드의 `Cf_calculation` 은 팽창비와 무관하게
   **완전팽창(P_e = P_a)** 을 가정한다. 우리 쪽은 팽창비에서 P_e 를 푼다.
   따라서 1차 비교 대상은 **챔버압**이고, 추력은 같은 Cf 규약으로 다시 계산해
   비교한다.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "reference"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import SRM_simul as srm
import srm_bates_fast as sbf

from gre.core.ballistics import MotorSpec, simulate
from gre.core.burnback2d import SegmentBurnback2D
from gre.core.burnrate import model_for
from gre.core.grain import Grain, Segment
from gre.core.nozzle import rt_from_cstar
from gre.geometry.port import circular_port

# ── 공통 조건 ────────────────────────────────────────────────────────────
D_G, D_GC, L_G, N_SEG = 60.0, 24.0, 120.0, 1
A_THROAT = 80.0
DEN_RATIO, DEN_IDEAL = 0.92, 1.841
C_STAR, C_STAR_EFF, GAMMA, T_ACTUAL = 885.0, 0.95, 1.133, 1600.0
P_ATM, ER, NOZ_EFF = 0.1013, 4.0, 0.95
V_C = np.pi / 4 * D_G ** 2 * (L_G + 40.0)

br = model_for("KNSB")
rt = rt_from_cstar(C_STAR, GAMMA)
R_SPECIFIC = rt / T_ACTUAL          # 두 코드가 같은 R·T 를 쓰도록 맞춘다

# ── 우리 구현 ────────────────────────────────────────────────────────────
sb = SegmentBurnback2D(circular_port(D_GC / 2.0), d_outer=D_G, n=601)
grain = Grain([Segment(sb, L_G, ends_burn=True)], density=DEN_IDEAL * DEN_RATIO)
spec = MotorSpec(grain=grain, burn_rate=br, a_throat=A_THROAT, chamber_volume=V_C,
                 expansion_ratio=ER, c_star=C_STAR, c_star_eff=C_STAR_EFF,
                 cf_eff=NOZ_EFF, gamma=GAMMA, p_ambient=P_ATM)
ours = simulate(spec)

# ── 기존 구현 (연소속도 모델을 우리 것으로 교체) ──────────────────────────
orig_rdot = srm.R_dot_calculation
srm.R_dot_calculation = lambda a, n, P_chamber: float(br.rate(float(P_chamber))[0])
try:
    V_g = srm.V_g_calculation(D_G, D_GC, L_G)
    den_actual = DEN_IDEAL * DEN_RATIO
    m_g = srm.m_g_calculation(V_g, den_actual)
    out = sbf.simulate_bates(
        V_C, D_G, D_GC, N_SEG, L_G, V_g, den_actual, m_g,
        A_THROAT, NOZ_EFF, R_SPECIFIC, GAMMA, C_STAR * C_STAR_EFF, T_ACTUAL,
        3.0, P_ATM, 0.0, 3.907, 0.535, 10.0, 1e-4, verbose=False)
finally:
    srm.R_dot_calculation = orig_rdot

t_ref = out[0]
a_bg_ref = out[5]
p_ref = out[11]
f_ref = out[14]

# ── 비교 ────────────────────────────────────────────────────────────────
t_hi = min(ours.t[-1], t_ref[-1])
tg = np.linspace(0.0, t_hi, 4000)
p_ours = np.interp(tg, ours.t, ours.p_c)
p_refi = np.interp(tg, t_ref, p_ref)

d_p = np.abs(p_ours - p_refi) / p_refi.max() * 100.0
# 소진 급강하 구간은 시간축이 ms 단위로만 어긋나도 점별 차이가 크게 잡힌다.
late = tg > 0.5 * tg[-1]
t_drop = tg[late][np.argmax(np.abs(np.gradient(p_refi, tg)[late]))]
plateau = (tg > 0.05) & (p_refi > 0.5 * p_refi.max()) & (tg < t_drop - 0.05)
drop = (tg >= t_drop - 0.05) & (p_refi > 0.5 * p_refi.max())

print(f"우리 구현 : P_max {ours.p_max:6.3f} MPa   t_b(5%) {ours.burn_time():.3f} s   "
      f"추진제 {grain.mass():.1f} g")
print(f"기존 구현 : P_max {p_ref.max():6.3f} MPa   (동일 연소속도 모델로 교체 후)")
print()
print(f"정상 구간 챔버압 최대 차이 : {d_p[plateau].max():.3f} % (P_max 대비)")
print(f"정상 구간 평균 차이        : {d_p[plateau].mean():.3f} %")
if drop.any():
    print(f"소진 급강하 구간           : {d_p[drop].max():.3f} %  (시간축 어긋남의 영향)")
print(f"최대압 차이                : {abs(ours.p_max - p_ref.max())/p_ref.max()*100:.3f} %")

# 연소면적 비교 (기하 자체)
w_ref = (out[1] - D_GC) / 2.0             # D_gc 진행 → 웹
a_ours = grain.burn_area(w_ref)
ok = w_ref < 0.95 * grain.web_max()
d_a = np.abs(a_ours[ok] - a_bg_ref[ok]) / a_bg_ref[ok].max() * 100.0
print(f"연소면적 A_b 최대 차이     : {d_a.max():.3f} % (A_b 최대값 대비, 소진 직전 제외)")

fig, ax = plt.subplots(1, 3, figsize=(13, 4))
ax[0].plot(w_ref[ok], a_ours[ok], label="ours (distance field)")
ax[0].plot(w_ref[ok], a_bg_ref[ok], "--", label="reference (SRM_simul)")
ax[0].set_xlabel("web [mm]"); ax[0].set_ylabel("A_b [mm2]"); ax[0].set_title("Burn area"); ax[0].legend()
ax[1].plot(ours.t, ours.p_c, label="ours")
ax[1].plot(t_ref, p_ref, "--", label="reference")
ax[1].set_xlabel("t [s]"); ax[1].set_ylabel("P_c [MPa]"); ax[1].set_title("Chamber pressure"); ax[1].legend()
ax[2].plot(tg, d_p); ax[2].set_xlabel("t [s]"); ax[2].set_ylabel("|dP| / P_max [%]")
ax[2].set_title("Pressure difference")
plt.tight_layout(); plt.savefig("out/v1b_compare.png", dpi=130)
print("\nsaved out/v1b_compare.png")
