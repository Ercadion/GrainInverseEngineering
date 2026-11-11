"""정방향 입력 설계의 **코어(=포트) 형상** 그림.

예시 목표 곡선 CSV 를 만든 그 설계가 어떤 코어인지 보여 준다.
세 패널: ① 단면(코어·추진제·케이스) ② 코어 확대 + 치수 ③ 코어 3D.

⚠ 그림의 수치는 **손으로 적지 않는다.** `gre.optimize.constraints` 에서 읽는다 —
  여기서 다시 계산하면 그림과 도구가 조용히 갈라진다 (STEP 과 STL 이
  그렇게 갈라졌다). 곡률반경만 극좌표 식으로 따로 구하고, 볼록 쪽이
  `min_tip_radius` 와 일치하는지 교차 확인해 출력한다.

사용:
  python experiments/forward_core_fig.py                      # 예시 1 (Ø60·1×90)
  python experiments/forward_core_fig.py --preset d44_3seg    # 예시 2 (Ø44·3×74)
  python experiments/forward_core_fig.py --r0 13 --a 0,0,3.2,0 \
      --d-outer 44 --length 74 --n-seg 3 --out out/x.png
"""
import argparse
import math
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon as MPoly
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CORE, PROP, CASE = "#0f62fe", "#ff832b", "#161616"
RHO = 1.841 * 0.92                       # g/cc — DiffMotor 기본 밀도

PRESETS = {
    # 이름            r0     a                    d_outer  length  n_seg  csv
    "d60_1seg": dict(r0=10.0, a=[0, 0, 3.2, 0], d_outer=60.0, length=90.0,
                     n_seg=1, tag="example_target_3lobe.csv"),
    "d44_3seg": dict(r0=13.0, a=[0, 0, 3.2, 0], d_outer=44.0, length=74.0,
                     n_seg=3, tag="example_target_D44_3seg74.csv"),
}


def pick_font():
    have = {f.name for f in matplotlib.font_manager.fontManager.ttflist}
    for c in ("Malgun Gothic", "AppleGothic", "NanumGothic",
              "Noto Sans CJK KR", "Noto Sans KR"):
        if c in have:
            plt.rcParams["font.family"] = c
            return c
    return None


def metrics(r0, a, b, d_outer, length):
    """제작 지표는 **코어에서 읽는다** — 여기서 다시 적지 않는다."""
    import torch
    from gre.optimize.constraints import (core_slenderness, lobe_geometry,
                                          min_port_radius, min_tip_radius,
                                          web_thickness)
    dt = torch.float64
    R0 = torch.tensor(float(r0), dtype=dt)
    A = torch.tensor(list(a), dtype=dt)
    B = torch.tensor(list(b), dtype=dt)
    R_out = float(d_outer) / 2.0
    g = lobe_geometry(R0, A, B)
    return {
        "web": float(web_thickness(R0, A, B, R_out)),
        "r_min": float(min_port_radius(R0, A, B)),
        "r_tip": float(min_tip_radius(R0, A, B)),
        "slender": float(core_slenderness(R0, A, B, float(length))),
        "depth": float(g["depth"]), "width": float(g["width"]),
        "n_lobes": int(g["n_lobes"]),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", choices=sorted(PRESETS), default="d60_1seg")
    ap.add_argument("--r0", type=float), ap.add_argument("--a", type=str)
    ap.add_argument("--b", type=str, help="sin 계수 (쉼표). 역설계 결과는 b 가 있다")
    ap.add_argument("--d-outer", type=float), ap.add_argument("--length", type=float)
    ap.add_argument("--n-seg", type=int), ap.add_argument("--out", type=str)
    args = ap.parse_args()

    cfg = dict(PRESETS[args.preset])
    if args.r0 is not None:
        cfg["r0"] = args.r0
    if args.a:
        cfg["a"] = [float(x) for x in args.a.replace(",", " ").split()]
    if args.b:
        cfg["b"] = [float(x) for x in args.b.replace(",", " ").split()]
    if args.d_outer is not None:
        cfg["d_outer"] = args.d_outer
    if args.length is not None:
        cfg["length"] = args.length
    if args.n_seg is not None:
        cfg["n_seg"] = args.n_seg
    r0, a = cfg["r0"], cfg["a"]
    b = cfg.get("b") or [0.0] * len(a)
    if len(b) != len(a):
        raise SystemExit(f"a 와 b 의 길이가 다르다: {len(a)} vs {len(b)}")
    R_OUT, L, NSEG = cfg["d_outer"] / 2.0, cfg["length"], cfg["n_seg"]
    out = args.out or f"out/mold_demold_D40/forward_core_{args.preset}.png"

    k = np.arange(1, len(a) + 1)[:, None]
    A = np.asarray(a, float)[:, None]
    B = np.asarray(b, float)[:, None]

    def rth(t):
        t = np.atleast_1d(t)
        return r0 + (A * np.cos(k * t) + B * np.sin(k * t)).sum(0)

    def curv(t):
        t = np.atleast_1d(t)
        r = rth(t)
        rp = (-A * k * np.sin(k * t) + B * k * np.cos(k * t)).sum(0)
        rpp = (-A * k * k * np.cos(k * t) - B * k * k * np.sin(k * t)).sum(0)
        return ((r**2 + rp**2) ** 1.5 / (r**2 + 2 * rp**2 - r * rpp))

    M = metrics(r0, a, b, cfg["d_outer"], L)
    N = M["n_lobes"]
    th = np.linspace(0, 2 * np.pi, 2001)
    r = rth(th)
    x, y = r * np.cos(th), r * np.sin(th)
    r_max, r_min = float(r.max()), float(r.min())
    # ⚠ b 계수가 있으면 봉우리·골이 θ=0, π/N 에 있지 않다 — **수치로 찾는다.**
    t_pk, t_vl = float(th[int(r.argmax())]), float(th[int(r.argmin())])
    pk = (r[r.argmax()] * np.cos(t_pk), r[r.argmax()] * np.sin(t_pk))
    # ⚠⚠ **가장 뾰족한 볼록점 ≠ r_max 지점이다.** `min_tip_radius` 가 보는 것은
    #   곡률이 가장 작은 **볼록** 자리이고, 로브 크기가 서로 다르면(= b 계수가
    #   있으면) 그게 가장 먼 점이 아니다. 초판은 r_max 자리의 곡률을 적었고
    #   아래 assert 가 그것을 잡았다. 두 자리를 **따로** 찾는다.
    cv = curv(th)
    i_tip = int(np.where(cv > 0, cv, np.inf).argmin())      # 볼록 중 가장 뾰족
    i_val = int(np.where(cv < 0, cv, -np.inf).argmax())     # 오목 중 가장 뾰족
    R_tip, R_val = float(cv[i_tip]), float(cv[i_val])
    p_tip = (r[i_tip] * np.cos(th[i_tip]), r[i_tip] * np.sin(th[i_tip]))
    p_val = (r[i_val] * np.cos(th[i_val]), r[i_val] * np.sin(th[i_val]))

    tt = np.linspace(0, 2 * np.pi, 20001)
    A_port = float(np.trapezoid(rth(tt) ** 2 / 2, tt))
    A_case = math.pi * R_OUT**2
    A_prop = A_case - A_port
    mass = A_prop * L * NSEG * RHO / 1000.0

    assert abs(R_tip - M["r_tip"]) < 1e-3, "볼록 곡률이 min_tip_radius 와 다르다"
    pick_font()
    plt.rcParams["axes.unicode_minus"] = False

    bits = []
    for i in range(len(a)):
        if abs(A[i, 0]) > 1e-9:
            bits.append(f"{A[i,0]:+.2f}·cos({i+1}θ)")
        if abs(B[i, 0]) > 1e-9:
            bits.append(f"{B[i,0]:+.2f}·sin({i+1}θ)")
    terms = " ".join(bits) or "+0"
    fig = plt.figure(figsize=(16.2, 5.6), dpi=140)
    fig.suptitle(f"코어 형상 —  r(θ) = {r0:.2f} {terms}  ·  "
                 f"케이스 Ø{cfg['d_outer']:.0f} · 세그먼트 {NSEG} × {L:.0f} mm",
                 fontsize=13, y=0.99)

    # ── ① 단면 ──────────────────────────────────────────────────────
    ax = fig.add_subplot(1, 3, 1)
    tc = np.linspace(0, 2 * np.pi, 721)
    outer = np.c_[R_OUT * np.cos(tc), R_OUT * np.sin(tc)]
    inner = np.c_[rth(tc) * np.cos(tc), rth(tc) * np.sin(tc)][::-1]
    ax.add_patch(MPoly(np.vstack([outer, inner]), closed=True,
                       facecolor=PROP, alpha=.28, edgecolor="none"))
    ax.add_patch(MPoly(np.c_[x, y], closed=True, facecolor=CORE, alpha=.90,
                       edgecolor=CORE, lw=1.5))
    ax.add_patch(Circle((0, 0), R_OUT, fill=False, ec=CASE, lw=2.0))
    ax.text(0, 0, "코어\n(포트)", ha="center", va="center", color="#ffffff",
            fontsize=11, fontweight="bold")
    lim1 = R_OUT * 1.12
    ax.text(0, -R_OUT - lim1 * 0.14,
            f"추진제\n{A_prop:.0f} mm² × {L:.0f} mm × {NSEG} = {mass:.0f} g",
            ha="center", va="center", fontsize=9.5, color="#8a3800")
    ax.annotate(f"케이스 내벽 Ø{cfg['d_outer']:.1f}",
                xy=(-R_OUT * 0.707, R_OUT * 0.707),
                xytext=(-lim1 * 0.98, R_OUT * 0.90), fontsize=9.5, color=CASE,
                arrowprops=dict(arrowstyle="->", color=CASE, lw=1.1))
    ax.set_xlim(-lim1, lim1)
    ax.set_ylim(-lim1 * 1.08, lim1)
    ax.set_aspect("equal")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title("① 단면 — 코어 · 추진제 · 케이스", fontsize=11)
    ax.grid(alpha=.2, lw=.6)

    # ── ② 코어 확대 + 치수 ──────────────────────────────────────────
    ax = fig.add_subplot(1, 3, 2)
    ax.add_patch(MPoly(np.c_[x, y], closed=True, facecolor=CORE, alpha=.14,
                       edgecolor=CORE, lw=2.4))
    for rr, col, ls in ((r_max, "#da1e28", "--"), (r_min, "#007d79", ":")):
        ax.add_patch(Circle((0, 0), rr, fill=False, ec=col, lw=1.3, ls=ls))
    pv = (r_min * np.cos(t_vl), r_min * np.sin(t_vl))
    ax.plot([0, pk[0]], [0, pk[1]], color="#da1e28", lw=1.6, zorder=4)
    ax.plot([0, pv[0]], [0, pv[1]], color="#007d79", lw=1.6, zorder=4)
    ax.scatter([pk[0]], [pk[1]], s=34, color="#da1e28", zorder=5)
    ax.scatter([pv[0]], [pv[1]], s=34, color="#007d79", zorder=5)
    ax.text(pk[0] * 1.12, pk[1] * 1.12, f"r_max\n{r_max:.2f}", color="#da1e28",
            fontsize=9.5, ha="center", va="center")
    ax.text(pv[0] + 0.5, pv[1] + 1.1, f"r_min {r_min:.2f}", color="#007d79",
            fontsize=9.5, ha="left")
    lim2 = r_max * 1.38
    ax.annotate(f"가장 뾰족한 볼록 {R_tip:.2f} mm (= r_tip_min)",
                xy=p_tip, xytext=(lim2 * 0.08, -lim2 * 0.82),
                fontsize=9, color="#525252", ha="center",
                arrowprops=dict(arrowstyle="->", color="#8d8d8d", lw=1.0,
                                connectionstyle="arc3,rad=-0.25"))
    ax.annotate(f"가장 뾰족한 오목 {abs(R_val):.2f} mm",
                xy=p_val, xytext=(-lim2 * 0.92, lim2 * 0.72), fontsize=9,
                color="#525252", ha="left",
                arrowprops=dict(arrowstyle="->", color="#8d8d8d", lw=1.0,
                                connectionstyle="arc3,rad=0.2"))
    ax.text(0, -lim2 * 1.02,
            f"로브 {N} 개 · 깊이 {M['depth']:.2f} · 폭 {M['width']:.2f} mm   |   "
            f"포트 단면 {A_port:.1f} mm² · 웹 두께 {M['web']:.2f} mm",
            ha="center", fontsize=9.5, color="#161616")
    ax.set_xlim(-lim2, lim2)
    ax.set_ylim(-lim2 * 1.14, lim2 * 0.92)
    ax.set_aspect("equal")
    ax.set_xlabel("x [mm]")
    ax.set_title("② 코어 확대 — 이 형상이 역설계의 정답", fontsize=11)
    ax.grid(alpha=.2, lw=.6)

    # ── ③ 코어 3D ───────────────────────────────────────────────────
    ax = fig.add_subplot(1, 3, 3, projection="3d")
    tm = np.linspace(0, 2 * np.pi, 181)
    xm, ym = rth(tm) * np.cos(tm), rth(tm) * np.sin(tm)
    gap = L * 0.10 if NSEG > 1 else 0.0
    for s in range(NSEG):
        z0 = s * (L + gap)
        side = [[(xm[i], ym[i], z0), (xm[i + 1], ym[i + 1], z0),
                 (xm[i + 1], ym[i + 1], z0 + L), (xm[i], ym[i], z0 + L)]
                for i in range(len(tm) - 1)]
        ax.add_collection3d(Poly3DCollection(side, facecolor="#2f6fe0",
                                             alpha=1.0, edgecolor="none"))
        for z in (z0, z0 + L):
            ax.add_collection3d(Poly3DCollection(
                [list(zip(xm, ym, np.full_like(xm, z)))],
                facecolor="#0043ce", alpha=1.0, edgecolor="#002d9c", lw=.8))
    ztop = NSEG * L + (NSEG - 1) * gap
    ax.set_xlim(-r_max * 1.2, r_max * 1.2)
    ax.set_ylim(-r_max * 1.2, r_max * 1.2)
    ax.set_zlim(0, ztop)
    ax.set_box_aspect((1, 1, 2.3))
    ax.set_xlabel("x [mm]", fontsize=9)
    ax.set_ylabel("y [mm]", fontsize=9)
    ax.set_zlabel("길이 [mm]", fontsize=9)
    ax.view_init(elev=18, azim=-58)
    ttl = f"③ 코어 — {L:.0f} mm × {NSEG} (세장비 {M['slender']:.2f})"
    if NSEG > 1:
        ttl += "\n※ 세그먼트 간격은 표시용"
    ax.set_title(ttl, fontsize=11)
    ax.tick_params(labelsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"저장: {out}")
    print(f"  웹 {M['web']:.2f} · 포트 최소 {M['r_min']:.2f} · "
          f"볼록 최소 {M['r_tip']:.3f} ≡ 곡률 {R_tip:.3f} ✔ · "
          f"오목 최소 {abs(R_val):.3f} · 세장비 {M['slender']:.2f}")
    print(f"  포트 {A_port:.2f} · 추진제 {A_prop:.2f} mm² → {mass:.1f} g")


if __name__ == "__main__":
    main()
