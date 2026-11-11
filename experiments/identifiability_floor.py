"""식별성 판정자의 바닥 — `circle_can_make_curve` 는 왜 원형조차 못 알아보는가.

다중 출발 측정에서 드러난 문제. `circle_can_make_curve` 는 "이 추력 곡선을 원형
포트 + 노즐목 두 변수만으로 맞출 수 있는가" 를 묻는 식별성 판정자인데,

    **실제 원형 포트로 만든 곡선**을 넣어도 잔차가 0 이 아니다.
    대조군 30개: 중앙값 5.60 %, 최대 8.91 % (전 구간 기준)

허용 오차가 10 % 이므로, 형상 정보에 남는 여유가 1~4 %p 밖에 없다. 그리고
오차의 방향이 한쪽이다 — 바닥이 높을수록 "원형으로는 부족 → 형상이 필요" 로
읽히므로, **판정자는 항상 형상이 필요하다는 쪽으로 틀린다.**

이 스크립트는 그 바닥이 어디서 오는지 두 장으로 보인다.

    (a) 시간축 어디서 잔차가 나오는가 — 점화 상승과 테일오프
    (b) 구간을 자르면 적합 자체가 참값으로 돌아온다

측정 결론: 바닥은 **준정상 평형 가정** 때문이지 거리장 이산화 때문이 아니다.
원형의 거리장 A_b 는 w ≤ 0.95·w_max 에서 해석해와 RMS 0.05~0.5 % 로 맞는다.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gre.core.analytic import bates_burn_area
from gre.core.burnrate import model_for
from gre.core.nozzle import (area_ratio_to_pressure_ratio, gamma_function,
                             optimum_area_ratio, rt_from_cstar)
from gre.optimize.forward import DiffMotor
from gre.validate.identifiability import circle_can_make_curve

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)
torch.set_default_dtype(torch.float64)
plt.rcParams.update({"font.family": "Noto Sans CJK JP", "axes.unicode_minus": False,
                     "font.size": 9})

C_TRUE, C_FULL, C_WIN, C_GRID = "#B4441F", "#1F4FA3", "#2C7A3F", "#9AA0A6"
R_OUT, R0, L, AT = 30.0, 11.0, 95.0, 80.0

M = DiffMotor(d_outer=2 * R_OUT, n_grid=121, n_samples=384, n_w=200, h_max=10e-3)
BR = model_for("KNSB")


def quasi_curve(r0, at_mm2, length):
    """`circle_can_make_curve` 내부와 같은 준정상 곡선."""
    if not (0.5 < r0 < R_OUT - 0.5) or at_mm2 <= 1.0:
        return None
    gamma, pa, cs = 1.133, 0.1013, 885.0 * 0.95
    rt, gfun = rt_from_cstar(885.0, gamma), gamma_function(1.133)
    rho_p = 1.841 * 0.92 * 1000.0
    w = np.linspace(0.0, R_OUT - r0, 300)
    ab = bates_burn_area(w, r0, R_OUT, length) * 1e-6
    at = at_mm2 * 1e-6
    lo, hi = np.full_like(w, 1e-3), np.full_like(w, 50.0)
    for _ in range(45):
        mid = 0.5 * (lo + hi)
        g = (rho_p - mid * 1e6 / rt) * ab * BR.rate(mid) * 1e-3 - mid * 1e6 * at / cs
        lo, hi = np.where(g > 0, mid, lo), np.where(g > 0, hi, mid)
    p = 0.5 * (lo + hi)
    r_mm = BR.rate(p)
    t = np.concatenate([[0.0], np.cumsum(np.diff(w) / np.clip((r_mm[1:] + r_mm[:-1]) / 2,
                                                              1e-9, None))])
    band = p >= 0.1 * p.max()
    p_mean = float(np.trapezoid(p[band], t[band]) / max(t[band][-1] - t[band][0], 1e-9))
    eps = optimum_area_ratio(p_mean, pa, gamma) if p_mean > pa else 1.0
    pe = area_ratio_to_pressure_ratio(eps, gamma)
    cfm = gfun * np.sqrt(2 * gamma / (gamma - 1) * (1 - pe ** ((gamma - 1) / gamma)))
    cf = cfm + (pe - pa / np.clip(p, 1e-9, None)) * eps
    return t, np.clip(cf * 0.95 * p * 1e6 * at, 0.0, None)


# ── 참값: 실제 원형 포트를 전체 모델(거리장 + 챔버 ODE)로 돌린 곡선 ──────────
with torch.no_grad():
    t, p, f, diag = M.simulate(torch.tensor(R0), torch.zeros(6), torch.zeros(6),
                               [L], torch.tensor(AT))
t, f = t.numpy(), f.numpy()

plateau = float(np.median(f[f > 0.5 * f.max()]))
i_ign = int(np.argmax(f >= 0.90 * plateau))
idx = np.arange(len(f))
REG = {"점화 상승": idx < i_ign,
       "지속 연소": (idx >= i_ign) & (f >= 0.20 * f.max()),
       "테일오프": (idx >= i_ign) & (f < 0.20 * f.max())}


# ── ⚠ 초판 정정 ────────────────────────────────────────────────────────────
#   초판은 `circle_can_make_curve` 를 그대로 불렀고, "전 구간 적합이 r0 11.00
#   → 9.78, A_t 80 → 109.7 로 참값을 놓친다" 고 적었다. **그것은 모델 현상이
#   아니라 탐색 실패였다.** 그 함수의 14×14 거친 격자 + Nelder-Mead 가 전역
#   최소에 못 간다. 조밀 전역 격자로 다시 재면 같은 목표에서 r0 10.60,
#   A_t 79.8 (잔차 5.46 %) 로 참값을 회복한다.
#
#   그래서 이 스크립트는 판정자를 쓰지 않고 **직접 조밀 격자**로 푼다.
#   살아남는 결론은 바닥의 존재와 그 출처뿐이다 — 전 구간 5.46 % → 창 2.95 %.
R0_GRID = np.linspace(3.0, 27.0, 161)
AT_GRID = np.geomspace(25.0, 400.0, 121)


def _rms(tt, ft, r0, at):
    out = quasi_curve(r0, at, L)
    if out is None:
        return 1e9
    tq, fq = out
    return float(np.sqrt(np.mean(((np.interp(tt, tq, fq, left=0.0, right=0.0) - ft)
                                  / max(abs(ft).max(), 1e-9)) ** 2)))


def dense_fit(tt, ft):
    """조밀 전역 격자 — 국소 최소에 속지 않는다 (느리지만 진단용이다)."""
    best = (1e9, None, None)
    for r0 in R0_GRID:
        for at in AT_GRID:
            v = _rms(tt, ft, r0, at)
            if v < best[0]:
                best = (v, r0, at)
    return {"residual_pct": best[0] * 100.0, "r0": best[1], "a_throat": best[2]}


full = dense_fit(t, f)
win = dense_fit(t[REG["지속 연소"]], f[REG["지속 연소"]])
per_region = {k: dense_fit(t[m], f[m]) for k, m in REG.items() if m.sum() >= 3}

print(f"참값      r0 = {R0:.2f} mm, A_t = {AT:.1f} mm²")
print(f"전 구간   잔차 {full['residual_pct']:.2f} %  → r0 = {full['r0']:.2f}, "
      f"A_t = {full['a_throat']:.1f}")
print(f"지속 구간 잔차 {win['residual_pct']:.2f} %  → r0 = {win['r0']:.2f}, "
      f"A_t = {win['a_throat']:.1f}")
for k, r in per_region.items():
    print(f"  [{k}] 점 {int(REG[k].sum()):4d}개 ({100*REG[k].mean():4.1f} %) "
          f"그 구간만 맞추면 {r['residual_pct']:5.2f} %")

fig, AX = plt.subplots(1, 2, figsize=(13.4, 5.3))

# ═══════════════════════════════ (a) 잔차가 시간축 어디서 나오는가
ax = AX[0]
tf, ff = quasi_curve(full["r0"], full["a_throat"], L)
tw, fw = quasi_curve(win["r0"], win["a_throat"], L)
ax.plot(t, f, color=C_TRUE, lw=3.4, alpha=0.85, zorder=2,
        label=f"참값 — 실제 원형 (r0 {R0:.0f}, A_t {AT:.0f}), 거리장 + 챔버 ODE")
ax.plot(tf, ff, color=C_FULL, lw=1.9, ls="--", zorder=4,
        label=f"판정자가 고른 원형 — 전 구간 적합 (r0 {full['r0']:.2f}, "
              f"A_t {full['a_throat']:.0f})")
ax.plot(tw, fw, color=C_WIN, lw=1.9, ls=(0, (6, 3)), zorder=5,
        label=f"판정자가 고른 원형 — 지속 구간만 (r0 {win['r0']:.2f}, "
              f"A_t {win['a_throat']:.0f})")
for k, col in (("점화 상승", "#E8B4A0"), ("테일오프", "#E8B4A0")):
    m = REG[k]
    if m.sum():
        ax.axvspan(t[m][0], t[m][-1], color=col, alpha=0.45, lw=0, zorder=0)
ax.text(t[REG["점화 상승"]][-1] if REG["점화 상승"].sum() else t[0], f.max() * 0.46,
        " 점화\n 상승", fontsize=8, color="#8A4A2A", va="center")
ax.text(t[REG["테일오프"]][0], f.max() * 0.46, " 테일\n 오프", fontsize=8,
        color="#8A4A2A", va="center")
ax.set_xlim(-0.05, t[REG["테일오프"]][-1] * 1.06)
ax.set_xlabel("t [s]")
ax.set_ylabel("F [N]")
ax.set_title("(a) 준정상 판정자는 점화 상승도 블로다운도 만들 수 없다", fontsize=10.5)
ax.legend(fontsize=7.6, loc="lower left", framealpha=0.95)
ax.grid(alpha=0.22)

# ═══════════════════════════════ (b) 구간별 기여
ax = AX[1]
names = list(per_region)
share = [100 * REG[k].mean() for k in names]
resid = [per_region[k]["residual_pct"] for k in names]
y = np.arange(len(names))[::-1]
ax.barh(y + 0.19, share, height=0.34, color=C_GRID, label="곡선에서 차지하는 점 비율 [%]")
ax.barh(y - 0.19, resid, height=0.34, color=C_TRUE,
        label="그 구간만 맞췄을 때의 잔차 [%]")
for yi, s, r in zip(y, share, resid):
    ax.text(s + 1.2, yi + 0.19, f"{s:.1f} %", va="center", fontsize=8, color="#555555")
    ax.text(r + 1.2, yi - 0.19, f"{r:.1f} %", va="center", fontsize=8, color=C_TRUE)
ax.set_yticks(y)
ax.set_yticklabels(names)
ax.axvline(full["residual_pct"], color=C_FULL, lw=1.8, ls="--")
ax.text(full["residual_pct"] + 1.5, len(names) - 1.62,
        f"전 구간 잔차\n{full['residual_pct']:.2f} %", color=C_FULL, fontsize=8.2,
        va="center", ha="left")
ax.axvline(10.0, color="#333333", lw=1.2, ls=":")
ax.text(10.0 + 1.5, -0.30, "허용 오차 10 %", fontsize=8.2, va="center")
ax.set_xlim(0, 108)
ax.set_ylim(-0.75, len(names) - 0.25)
ax.set_xlabel("[%]")
ax.set_title("(b) 전체의 6.8 % 뿐인 두 구간이 잔차를 두 배로 만든다", fontsize=10.5)
ax.legend(fontsize=8, loc="lower right", framealpha=0.95)
ax.grid(alpha=0.22, axis="x")

fig.text(0.5, 0.012,
         "준정상 평형식에는 점화 상승도 블로다운도 없다 — dρ_c/dt = 0 을 가정하므로 "
         "t = 0 에 이미 평형압이고 소진 즉시 추력 0 이다.\n"
         "전체의 %.1f %% 뿐인 두 구간이 잔차를 %.2f → %.2f %% 로 올린다. "
         "허용 오차 10 %% 중 그만큼을 형상 정보가 아닌 모델 오차가 먹고 들어간다.\n"
         "⚠ 이 그림은 조밀 전역 격자로 다시 푼 것이다. 초판은 판정자의 거친 격자를 "
         "그대로 썼다가 탐색 실패를 모델 현상으로 잘못 읽었다."
         % (100 * (REG["점화 상승"].mean() + REG["테일오프"].mean()),
            per_region["지속 연소"]["residual_pct"], full["residual_pct"]),
         ha="center", va="bottom", fontsize=8.2,
         bbox=dict(boxstyle="round,pad=0.45", fc="#FFFFFF", ec="#CCCCCC"))
fig.suptitle("식별성 판정자의 바닥 — 실제 원형 포트조차 잔차 %.2f %% 가 나온다"
             % full["residual_pct"], fontsize=11.5, y=0.975)
fig.tight_layout(rect=(0, 0.115, 1, 0.945))
path = os.path.join(OUT, "identifiability_floor.png")
fig.savefig(path, dpi=160)
print("saved", path)
