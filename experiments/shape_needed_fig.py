"""어떤 곡선이 비원형 포트를 요구하는가 — 설명 그림.

`shape_needed_sweep.py` 가 만든 `.npz` 를 읽어 네 장을 그린다.

    (a) 원형 포트의 A_b(w) 는 **반드시 아래로 오목한 포물선**이다 — 유도까지
    (b) 그래서 원형이 만들 수 있는 모양은 2-자유도뿐 → 못 만드는 곡선족
    (c) 비원형이 실제로 버는 것: **초기 둘레 과잉**뿐. 후반은 Steiner 로 수렴
    (d) 곡선 공간 지도 — 실제 형상 N 개를 돌려 "원형으로 충분한가" 를 색칠
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from gre.core.analytic import bates_burn_area
from gre.optimize.forward import DiffMotor

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
torch.set_default_dtype(torch.float64)
plt.rcParams.update({"font.family": "Noto Sans CJK JP", "axes.unicode_minus": False,
                     "font.size": 9})

C_CIR, C_BAD, C_STAR, C_GRID = "#1F4FA3", "#B4441F", "#7A3EA8", "#9AA0A6"
R_OUT, L0 = 30.0, 95.0

fig = plt.figure(figsize=(13.6, 10.2))
gs = fig.add_gridspec(2, 2, hspace=0.33, wspace=0.24,
                      left=0.065, right=0.975, top=0.880, bottom=0.115)

# ═════════════════════════════════════ (a) 원형 = 포물선, 곡률은 항상 −12π
ax = fig.add_subplot(gs[0, 0])
for r0, ls in zip((8.0, 12.0, 16.0, 20.0), ("-", "--", "-.", ":")):
    w = np.linspace(0.0, R_OUT - r0, 400)
    ax.plot(w, bates_burn_area(w, r0, R_OUT, L0) / 100.0, color=C_CIR, ls=ls, lw=1.9,
            label=f"원형 r0 = {r0:.0f} mm")
    # 포물선 식과 일치하는지 그림 안에서 확인
    poly = 2 * np.pi * ((r0 * L0 + R_OUT ** 2 - r0 ** 2) + (L0 - 4 * r0) * w - 3 * w ** 2)
    assert np.allclose(poly, bates_burn_area(w, r0, R_OUT, L0), atol=1e-9)
ax.set_xlabel("웹 w [mm]")
ax.set_ylabel("A_b [cm²]")
ax.set_title("(a) 원형 포트의 연소면적은 언제나 아래로 오목한 포물선", fontsize=10.5)
ax.legend(fontsize=8, loc="lower left")
ax.grid(alpha=0.22)
ax.text(0.98, 0.97,
        "A_b(w) = 2π(r0+w)(L−2w) + 2π(R²−(r0+w)²)\n"
        "        = 2π[ (r0L + R² − r0²)  +  (L − 4r0)·w  −  3w² ]\n\n"
        "→ d²A_b/dw² = −12π,  r0 · L · R 과 무관한 상수\n"
        "   즉 원형이 고를 수 있는 것은 상수항과 1차항 둘뿐이다.",
        transform=ax.transAxes, ha="right", va="top", fontsize=8.2,
        bbox=dict(boxstyle="round,pad=0.45", fc="#F4F7FC", ec=C_CIR, alpha=0.95))

# ═════════════════════════ (b) 2-자유도로 못 만드는 곡선족
#
# ⚠ 초판은 "모든 원형을 쓸어 만든 도달 가능 영역" 을 음영으로 깔았다. 그것은
#   **합집합의 포락선**이라, 포물선이 아닌 곡선도 그 안에 들어갈 수 있다
#   (실제로 ① 이 영역 안으로 들어가 버려 그림이 주장을 뒷받침하지 못했다).
#   요구는 "포물선 하나로 맞출 수 있는가" 이므로, 목표마다 **최적 원형을
#   직접 맞춰** 남는 차이를 보이는 쪽이 옳다.
from scipy.optimize import minimize                                  # noqa: E402

u = np.linspace(0.0, 1.0, 400)
BAD = [("① 아래로 볼록한 점진형", 1.0 + 1.1 * u ** 2.6),
       ("② 쌍봉 — 극값이 둘",
        1.0 + 0.55 * np.exp(-((u - 0.22) / 0.13) ** 2)
        + 0.72 * np.exp(-((u - 0.70) / 0.14) ** 2)),
       ("③ 평탄 후 급락 (cliff)",
        1.0 + 0.1 * u - 1.05 / (1 + np.exp(-(u - 0.80) / 0.035)))]


def best_circle(y):
    """정규화 목표 y(u) 에 가장 가까운 원형 포트 (r0, L) 와 그 곡선."""
    def shape(r0, L):
        w = u * (R_OUT - r0)
        ab = bates_burn_area(w, r0, R_OUT, L)
        return ab / ab[0] if ab[0] > 0 else np.full_like(u, 1e3)

    def rms(x):
        r0, L = float(x[0]), float(x[1])
        if not (3.0 < r0 < R_OUT - 2.0) or not (15.0 < L < 400.0):
            return 1e6
        return float(np.sqrt(np.mean((shape(r0, L) - y) ** 2)))

    seed = min(((rms((r0, L)), r0, L) for r0 in np.linspace(4, 26, 23)
                for L in np.geomspace(20, 350, 25)))
    r = minimize(rms, x0=[seed[1], seed[2]], method="Nelder-Mead",
                 options={"xatol": 1e-4, "fatol": 1e-12, "maxiter": 800})
    return shape(float(r.x[0]), float(r.x[1])), float(r.fun), float(r.x[0]), float(r.x[1])


gsb = gs[0, 1].subgridspec(3, 1, hspace=0.42)
for i, (lab, y) in enumerate(BAD):
    axb = fig.add_subplot(gsb[i])
    fit, rms_, r0f, Lf = best_circle(y)
    axb.fill_between(u, y, fit, color=C_BAD, alpha=0.17, lw=0)
    axb.plot(u, y, color=C_BAD, lw=2.1, label="요구한 A_b(w)")
    axb.plot(u, fit, color=C_CIR, lw=1.8, ls="--",
             label=f"가장 가까운 원형 (r0 {r0f:.1f}, L {Lf:.0f})")
    axb.set_ylabel("A_b / A_b(0)", fontsize=8)
    axb.tick_params(labelsize=7.5)
    axb.grid(alpha=0.2)
    axb.set_title(f"{lab}   — 최적 원형으로도 RMS {100*rms_:.0f} % 가 남는다",
                  fontsize=9, pad=3)
    axb.legend(fontsize=6.8, loc="best", framealpha=0.9)
    if i == 0:
        axb.annotate("(b) 포물선 하나로는 맞출 수 없는 요구들",
                     xy=(0.5, 1.46), xycoords="axes fraction", ha="center",
                     fontsize=10.5)
    if i == 2:
        axb.set_xlabel("정규화 웹  w / w_소진", fontsize=8.5)
        axb.annotate("포물선은 ① 위로 휠 수 없고 ② 극값이 하나뿐이며 "
                     "③ 정점 근처에서 2차로 완만하게 떨어진다.\n"
                     "※ 단 노즐목 A_t 가 자유면 w→t 사상까지 바뀐다 — "
                     "A_b 의 불가능이 곧 F(t) 의 불가능은 아니다. 그래서 (d).",
                     xy=(0.5, -0.62), xycoords="axes fraction", ha="center",
                     va="top", fontsize=7.6,
                     bbox=dict(boxstyle="round,pad=0.35", fc="#FDF5F2",
                               ec=C_BAD, alpha=0.95))

# ═════════════════════ (c) 비원형이 버는 것 = 초기 둘레 과잉뿐 (Steiner)
ax = fig.add_subplot(gs[1, 0])
M = DiffMotor(d_outer=2 * R_OUT, n_grid=161, n_samples=512)
w = torch.linspace(0.0, 20.0, 150)
pairs = [("원형 r0 = 12", 12.0, torch.zeros(6), C_CIR, "-"),
         ("3-로브 스타 (r0 8, a3 = 4.5)", 8.0, None, C_STAR, "--"),
         ("4-로브 스타 (r0 8, a4 = 5.0)", 8.0, None, C_BAD, "-.")]
amps = [None, (3, 4.5), (4, 5.0)]
for (lab, r0, _, col, ls), am in zip(pairs, amps):
    a = torch.zeros(6)
    if am:
        a[am[0] - 1] = am[1]
    with torch.no_grad():
        phi = M.ds.phi(torch.tensor(r0), a, torch.zeros(6))
        per = M.ds.perimeter(phi, w).numpy()
    ax.plot(w.numpy(), per, color=col, ls=ls, lw=2.0, label=lab)
wline = np.linspace(0, 20, 50)
ax.plot(wline, 2 * np.pi * (12.0 + wline), color=C_GRID, lw=1.2, ls=":",
        label="기울기 2π (Steiner 극한)")
ax.set_xlabel("웹 w [mm]")
ax.set_ylabel("포트 둘레  P_2D(w)  [mm]")
ax.set_title("(c) 비원형이 버는 것은 '초기 둘레 과잉' 과 그것이 사라지는 속도뿐",
             fontsize=10.5)
ax.legend(fontsize=8, loc="upper left")
ax.grid(alpha=0.22)
ax.annotate("두께 w 만큼 옵셋한 곡선의 둘레는\n"
            "모든 형상에서 P(w) → P(0) + 2πw 로 수렴한다 (Steiner).\n"
            "→ 웹 후반부는 어떤 형상을 써도 원형과 구별되지 않는다.\n"
            "   형상이 곡선에 남기는 흔적은 앞쪽에만 있다.",
            xy=(0.98, 0.04), xycoords="axes fraction", ha="right", va="bottom",
            fontsize=8, bbox=dict(boxstyle="round,pad=0.4", fc="#F7F4FB",
                                  ec=C_STAR, alpha=0.95))

# ══════════════════════════════════ (d) 곡선 공간 지도
ax = fig.add_subplot(gs[1, 1])
npz_path = os.path.join(OUT, "shape_needed.npz")
D = np.load(npz_path)
res = np.clip(D["residual"], 0.0, 40.0)
cir = D["n_lobe"] == 0
floor = float(res[cir].max())                  # 모델 오차 바닥 (대조군 최댓값)
sc = ax.scatter(D["peakiness"][~cir], D["front_back"][~cir], c=res[~cir],
                cmap="viridis_r", s=46, vmin=0, vmax=20, edgecolor="white", lw=0.5,
                zorder=3)
ax.scatter(D["peakiness"][cir], D["front_back"][cir], marker="o", s=30,
           facecolor="none", edgecolor=C_GRID, lw=1.2, zorder=2,
           label=f"대조군: 실제 원형 포트 ({int(cir.sum())}개, 잔차 ≤ {floor:.1f} %)")
cb = fig.colorbar(sc, ax=ax, pad=0.02)
cb.set_label("원형+노즐목 2변수로 맞췄을 때의 잔차 [%]", fontsize=8.5, labelpad=2)
cb.ax.axhline(floor, color=C_BAD, lw=1.8)
cb.ax.text(-0.35, floor, "모델 오차\n바닥 %.1f %%" % floor, color=C_BAD,
           fontsize=7.0, ha="right", va="center",
           transform=cb.ax.get_yaxis_transform())
need = res > 10.0
if (need & ~cir).any():
    ax.scatter(D["peakiness"][need & ~cir], D["front_back"][need & ~cir], marker="x",
               s=30, color="black", lw=0.9, zorder=4,
               label="잔차 > 10 % — 형상이 필요한 곡선")
ax.axhline(1.0, color=C_GRID, lw=1.0, ls=":")
ax.set_xlabel("뾰족함  F_max / F_평균")
ax.set_ylabel("앞뒤 비  F(25 % 시점) / F(75 % 시점)")
ax.set_title("(d) 실제 형상 %d 개의 곡선을 '원형으로 충분한가' 로 색칠" % len(res),
             fontsize=10.5)
ax.grid(alpha=0.22)
ax.set_ylim(0.72, None)
ax.legend(fontsize=7.4, loc="upper left", framealpha=0.95)
ax.annotate("중립 (앞 = 뒤)", xy=(0.995, 1.0), xycoords=("axes fraction", "data"),
            ha="right", va="bottom", color=C_GRID, fontsize=7.5)
fig.text(0.5, 0.022,
         "(d) 잔차는 손실의 ω(t) 와 같은 창(점화 끝 ~ F ≥ 20 %% F_max)에서 쟀다. 전 구간으로 재면 실제 원형조차 잔차 중앙값 %.1f %% 가 나오는데,\n"
         "이는 준정상 근사가 점화 과도와 블로다운을 못 담아 생기는 모델 오차이지 형상 정보가 아니다. 비원형 %d 개 중 잔차 ≤ 10 %% 가 %.0f %% —\n"
         "형상이 꼭 필요한 곡선은 이 설계 공간에서 소수이고, 그런 곡선은 '뾰족하면서 앞뒤 비가 1 에서 크게 벗어난' 쪽에 몰려 있다."
         % (np.median(D["residual_all"][cir]), int((~cir).sum()),
            100 * np.mean(res[~cir] <= 10.0)),
         ha="center", va="bottom", fontsize=8.0,
         bbox=dict(boxstyle="round,pad=0.45", fc="#FFFFFF", ec="#CCCCCC"))

fig.suptitle("어떤 추력 곡선이 비원형 포트를 요구하는가\n"
             "— 원형은 A_b 를 포물선으로만 만들 수 있고(a·b), 비원형이 더하는 것은 "
             "웹 앞쪽의 둘레 과잉뿐이며(c), 그 차이는 노즐목이 상당 부분 흡수한다(d)",
             fontsize=11.5, y=0.985)
path = os.path.join(OUT, "shape_needed.png")
fig.savefig(path, dpi=160)
print("saved", path)
