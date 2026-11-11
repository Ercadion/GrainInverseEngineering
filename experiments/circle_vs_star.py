"""최소 복잡도 탐색 결과의 시각 확인 — 스타 목표 vs 복원된 원형.

`minimal_k.py` 가 두 목표 모두 **K = 0 (원형)** 으로 통과시켰다. 숫자만
보면 "정말 같은 곡선인가" 가 확인되지 않으므로, 세 가지를 나란히 그린다.

    (1) 포트 단면    — 모양은 전혀 다르다
    (2) A_b(w)       — 연소면적 자체도 다르다  ← 중요
    (3) F(t)         — 그런데 추력 곡선은 겹친다

(2)와 (3)이 함께 있어야 "노즐목 A_t 가 A_b 의 차이를 흡수한다" 는 결론이
그림으로 보인다. A_b 가 같아서 곡선이 같은 것이 아니다.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gre.optimize.forward import DiffMotor

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)
torch.set_default_dtype(torch.float64)

plt.rcParams.update({"font.family": "Noto Sans CJK JP", "axes.unicode_minus": False,
                     "font.size": 9})

C_TGT, C_CIR, C_CASE = "#B4441F", "#1F4FA3", "#9AA0A6"   # 목표 / 원형 / 케이스
R_OUT, L = 30.0, 95.0
M = DiffMotor(d_outer=2 * R_OUT, n_grid=121, n_samples=384, n_w=200, h_max=10e-3)

# (이름, 목표 r0, 목표 a, 목표 A_t, 복원 원형 r0, 복원 A_t)  — minimal_k.py 결과
CASES = [
    ("얕은 스타", 11.0, [0.0, 0.0, 2.2, 0.0], 78.0, 12.292, 81.60),
    ("깊은 스타", 8.0, [0.0, 0.0, 0.0, 5.0], 120.0, 13.756, 75.16),
]


def contour(r0, a, n=721):
    """포트 윤곽 (x, y) [mm]."""
    th = np.linspace(0.0, 2 * np.pi, n)
    k = np.arange(1, len(a) + 1)[:, None]
    r = float(r0) + (np.asarray(a, dtype=float)[:, None] * np.cos(k * th)).sum(0)
    return r * np.cos(th), r * np.sin(th)


def run(r0, a, at):
    t, p, f, d = M.simulate(torch.tensor(float(r0)), torch.tensor(a, dtype=torch.float64),
                            torch.zeros(len(a)), [L], torch.tensor(float(at)))
    return dict(t=t.numpy(), p=p.detach().numpy(), f=f.detach().numpy(),
                w=d["w_tab"].numpy(), ab=d["a_b"].detach().numpy(),
                I=float(torch.trapezoid(f, t)), mass=d["mass_g"])


fig, AX = plt.subplots(2, 3, figsize=(13.0, 7.6))

for row, (tag, r0t, at_, att, r0c, atc) in enumerate(CASES):
    tgt = run(r0t, at_, att)
    cir = run(r0c, [0.0], atc)

    # ── (1) 포트 단면 ───────────────────────────────────────────────
    ax = AX[row, 0]
    th = np.linspace(0, 2 * np.pi, 400)
    ax.plot(R_OUT * np.cos(th), R_OUT * np.sin(th), color=C_CASE, lw=2.0)
    ax.text(0, -R_OUT - 3.4, "케이스 내벽 Ø60", color=C_CASE, ha="center", fontsize=8)
    xt, yt = contour(r0t, at_)
    ax.fill(xt, yt, color=C_TGT, alpha=0.13)
    ax.plot(xt, yt, color=C_TGT, lw=2.0, label=f"목표 {tag}")
    xc, yc = contour(r0c, [0.0])
    ax.plot(xc, yc, color=C_CIR, lw=2.0, ls="--", label=f"복원 원형 r0={r0c:.2f}")
    ax.set_aspect("equal")
    ax.set_xlim(-R_OUT - 6, R_OUT + 6)
    ax.set_ylim(-R_OUT - 6, R_OUT + 6)
    ax.set_title(f"({chr(97+row)}1) 포트 단면 — 모양은 전혀 다르다", fontsize=10)
    ax.legend(fontsize=8, loc="upper right", framealpha=0.9)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.grid(alpha=0.18)

    # ── (2) A_b(w) ────────────────────────────────────────────────
    ax = AX[row, 1]
    ax.plot(tgt["w"], tgt["ab"] / 100.0, color=C_TGT, lw=2.0, label="목표")
    ax.plot(cir["w"], cir["ab"] / 100.0, color=C_CIR, lw=2.0, ls="--", label="복원 원형")
    ax.set_title(f"({chr(97+row)}2) 연소면적 A_b(w) — 이것도 다르다", fontsize=10)
    ax.set_xlabel("웹 w [mm]")
    ax.set_ylabel("A_b [cm²]")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.22)
    d_ab = 100.0 * abs(tgt["ab"][0] - np.interp(0.0, cir["w"], cir["ab"])) / tgt["ab"][0]
    ax.annotate(f"초기 A_b 차이 {d_ab:.0f} %\n웹 소진 {tgt['w'][-1]:.1f} → {cir['w'][-1]:.1f} mm",
                xy=(0.97, 0.06), xycoords="axes fraction", ha="right", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#CCCCCC"))

    # ── (3) F(t) ──────────────────────────────────────────────────
    ax = AX[row, 2]
    ax.plot(tgt["t"], tgt["f"], color=C_TGT, lw=2.4, label="목표")
    ax.plot(cir["t"], cir["f"], color=C_CIR, lw=2.0, ls="--", label="복원 원형")
    ax.set_title(f"({chr(97+row)}3) 추력 F(t) — 곡선은 겹친다", fontsize=10)
    live = tgt["t"][tgt["f"] > 0.01 * tgt["f"].max()]
    ax.set_xlim(-0.05, float(live[-1]) * 1.12)
    ax.set_xlabel("t [s]")
    ax.set_ylabel("F [N]")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.22)
    ax.annotate(f"A_t  {att:.0f} → {atc:.1f} mm²\n"
                f"I  {tgt['I']:.0f} → {cir['I']:.0f} N·s\n"
                f"F_max  {tgt['f'].max():.0f} → {cir['f'].max():.0f} N",
                xy=(0.97, 0.62), xycoords="axes fraction", ha="right", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#CCCCCC"))

    AX[row, 0].text(-R_OUT - 5, R_OUT + 2.0, tag, fontsize=12, fontweight="bold",
                    color=C_TGT)

fig.suptitle("최소 복잡도 탐색: 두 스타 목표가 모두 K = 0 (원형) 으로 통과한 이유\n"
             "— 포트 모양도 A_b 도 다르지만, 노즐목 A_t 가 그 차이를 흡수해 같은 추력 곡선을 만든다",
             fontsize=11.5, y=0.995)
fig.tight_layout(rect=(0, 0, 1, 0.945))
path = os.path.join(OUT, "circle_vs_star.png")
fig.savefig(path, dpi=160)
print("saved", path)
