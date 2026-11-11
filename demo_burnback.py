"""프로토타입 데모 — 원형/스타 포트의 A_b(w) 곡선과 검증 그래프 생성."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gre.core import analytic
from gre.core.burnback2d import SegmentBurnback2D
from gre.geometry.port import circular_port, star_port

D_OUTER, LENGTH = 60.0, 80.0

circ = SegmentBurnback2D(circular_port(10.0), d_outer=D_OUTER, n=601)
star = SegmentBurnback2D(star_port(12.0, 6, 3.0), d_outer=D_OUTER, n=601)

fig, ax = plt.subplots(1, 3, figsize=(13, 4))

# (1) 거리장 + 등고선
im = ax[0].contourf(star.X, star.Y, np.where(star.mask_prop, star.phi, np.nan), 24)
ax[0].contour(star.X, star.Y, np.where(star.mask_prop, star.phi, np.nan),
              levels=[2, 4, 6, 8, 10], colors="k", linewidths=0.6)
ax[0].set_aspect("equal"); ax[0].set_title("Star port: distance field and burn fronts")
fig.colorbar(im, ax=ax[0], label="φ [mm]")

# (2) 원형 포트 검증
w = circ.web_grid(200)
ax[1].plot(w, circ.burn_area(w, LENGTH), label="numeric (distance field)")
ax[1].plot(w, analytic.bates_burn_area(w, 10.0, 30.0, LENGTH), "--", label="analytic")
ax[1].set_xlabel("web w [mm]"); ax[1].set_ylabel("A_b [mm²]")
ax[1].set_title("Circular port check (V1-a)"); ax[1].legend()

# (3) 형상별 A_b(w)
for sb, name in ((circ, "circular (BATES)"), (star, "star, 6 points")):
    w = sb.web_grid(200)
    ax[2].plot(w, sb.burn_area(w, LENGTH), label=name)
ax[2].set_xlabel("web w [mm]"); ax[2].set_ylabel("A_b [mm²]")
ax[2].set_title("Burn area by shape"); ax[2].legend()

plt.tight_layout()
plt.savefig("out/burnback_demo.png", dpi=130)
print("saved out/burnback_demo.png")

w = circ.web_grid(200)
ana = analytic.bates_burn_area(w, 10.0, 30.0, LENGTH)
num = circ.burn_area(w, LENGTH)
main = w <= 0.95 * circ.w_max      # 웹 소진 직전까지
err_main = np.max(np.abs(num[main] - ana[main])) / ana.max()
err_all = np.max(np.abs(num - ana)) / ana.max()
print(f"격자 {circ.n}^2, h={circ.h:.3f} mm")
print(f"  본 구간(w <= 0.95 w_max) 최대 오차: {err_main*100:.3f} % (A_b 최대값 대비)")
print(f"  전 구간 최대 오차:                  {err_all*100:.3f} %  <- 웹 소진 순간의 불연속")
print("  해석해는 소진 순간 연소면이 계단으로 0이 되지만, 거리장은 완화 폭 ε 만큼")
print("  매끄럽게 떨어진다. 테일오프를 따로 보고하기로 한 이유가 여기서도 보인다.")
