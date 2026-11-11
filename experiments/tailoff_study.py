"""테일오프 정량 확인.

문제
----
웹이 소진되는 순간, 해석해(원형 포트)는 연소면적이 **계단**으로 0 이 되지만
거리장 모델은 완화 폭 ε 만큼 **번져서** 떨어진다. V1-a 대조에서 그 한 점의
오차가 A_b 최대값 대비 34 % 였다.

질문: 그 차이가 **최종 압력·추력 곡선에 얼마나 남는가?**
압력은 A_b 를 그대로 따라가지 않고 챔버 배출 시상수 τ 를 거치므로,
A_b 의 계단이 압력에서는 완만해진다. 그 정도를 숫자로 확인한다.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gre.core import analytic
from gre.core.ballistics import MotorSpec, simulate
from gre.core.burnback2d import SegmentBurnback2D
from gre.core.burnrate import model_for
from gre.core.grain import Grain, Segment
from gre.geometry.port import circular_port

R_PORT, D_OUTER, L_SEG = 12.0, 60.0, 120.0
R_OUTER = D_OUTER / 2.0

sb = SegmentBurnback2D(circular_port(R_PORT), d_outer=D_OUTER, n=601)
grain = Grain([Segment(sb, L_SEG, ends_burn=True)])
spec = MotorSpec(grain=grain, burn_rate=model_for("KNSB"), a_throat=80.0,
                 chamber_volume=np.pi / 4 * D_OUTER ** 2 * (L_SEG + 40.0))

# (1) 거리장 (ε 로 번진 A_b)  (2) 해석해 (계단 A_b)
res_field = simulate(spec)
ab_step = lambda w: analytic.bates_burn_area(np.asarray(w, dtype=float),
                                             R_PORT, R_OUTER, L_SEG, ends_burn=True)
res_step = simulate(spec, a_burn_fn=ab_step)

v_free = (spec.chamber_volume - float(grain.volume(0.0)[0])) * 1e-9
tau = spec.c_star_actual * v_free / (spec.rt * spec.a_throat * 1e-6)


def summary(res, name):
    return (f"{name:10s} P_max {res.p_max:6.3f} MPa   F_max {res.f_max:7.2f} N   "
            f"I_t {res.total_impulse:7.2f} N·s   t_b(5%) {res.burn_time(0.05):.3f} s")


print(f"챔버 충전 시상수 τ = {tau*1e3:.2f} ms")
print(summary(res_field, "거리장"))
print(summary(res_step, "해석해"))
d_it = abs(res_field.total_impulse - res_step.total_impulse) / res_step.total_impulse
d_pm = abs(res_field.p_max - res_step.p_max) / res_step.p_max
d_tb = abs(res_field.burn_time(0.05) - res_step.burn_time(0.05))
print(f"\n총역적 차이   {d_it*100:.3f} %")
print(f"최대압 차이   {d_pm*100:.3f} %")
print(f"연소시간 차이 {d_tb*1e3:.1f} ms")

# 꼬리 구간에서의 추력 차이
t_grid = np.linspace(0, min(res_field.t[-1], res_step.t[-1]), 4000)
f_field = np.interp(t_grid, res_field.t, res_field.thrust)
f_step = np.interp(t_grid, res_step.t, res_step.thrust)
# 급강하 구간은 시간축이 몇 ms 만 어긋나도 점별 차이가 크게 나온다.
# 정상 구간 / 급강하 / 꼬리를 나눠 본다.
late = t_grid > 0.5 * t_grid[-1]          # 점화 상승과 헷갈리지 않도록 후반부에서 찾는다
t_drop = t_grid[late][np.argmax(np.abs(np.gradient(f_step, t_grid)[late]))]   # 소진 순간
plateau = t_grid < t_drop - 0.05
drop = (t_grid >= t_drop - 0.05) & (f_step >= 0.05 * res_step.f_max)
tail = f_step < 0.05 * res_step.f_max
d = np.abs(f_field - f_step) / res_step.f_max * 100.0
print(f"\n정상 구간 추력 최대 차이  {d[plateau].max():.3f} % (F_max 대비)")
if drop.any():
    print(f"소진 급강하 구간          {d[drop].max():.3f} %  <- 시간축 {d_tb*1e3:.1f} ms 어긋남의 영향")
if tail.any():
    print(f"꼬리 구간                 {d[tail].max():.3f} %")

fig, ax = plt.subplots(1, 3, figsize=(13, 4))
w = np.linspace(0, grain.web_max(), 600)
ax[0].plot(w, grain.burn_area(w), label="distance field (smeared)")
ax[0].plot(w, ab_step(w), "--", label="analytic (step)")
ax[0].set_xlabel("web w [mm]"); ax[0].set_ylabel("A_b [mm2]")
ax[0].set_title("Burn area at burnout"); ax[0].legend()

for r, st, lbl in ((res_field, "-", "distance field"), (res_step, "--", "analytic step")):
    ax[1].plot(r.t, r.p_c, st, label=lbl)
    ax[2].plot(r.t, r.thrust, st, label=lbl)
ax[1].set_xlabel("t [s]"); ax[1].set_ylabel("P_c [MPa]"); ax[1].set_title("Chamber pressure"); ax[1].legend()
ax[2].set_xlabel("t [s]"); ax[2].set_ylabel("F [N]"); ax[2].set_title("Thrust"); ax[2].legend()
plt.tight_layout(); plt.savefig("out/tailoff_study.png", dpi=130)
print("\nsaved out/tailoff_study.png")
