"""제조 제약이 어디서 걸리는가 — 요구 연소면적에 따른 지도.

시연에서 제약이 전부 비활성이었다(웹 16.6 vs 한계 3.0). 그렇다면
제약은 언제 일을 하는가? 답은 '목표가 많은 연소면적을 요구할 때' 다.

연소면적은 둘레로 번다. 같은 케이스 안에서 둘레를 늘리려면 핀을 깊게·많이
파야 하고, 그러면 (1) 핀 끝과 케이스 벽 사이 추진제가 얇아지고(w_web_min)
(2) 핀 사이 슬롯이 좁아지며(t_void_min) (3) 꼭짓점이 뾰족해진다(r_tip_min).
이 셋이 둘레의 상한을 정한다.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from gre.optimize.forward import DiffMotor
from gre.optimize.constraints import (ManufacturingLimits, min_tip_radius,
                                      thin_feature_penalty, web_thickness)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)
torch.set_default_dtype(torch.float64)
M = DiffMotor(d_outer=60.0, n_grid=161, n_samples=512)
LIM = ManufacturingLimits()
R_OUT, L = 30.0, 95.0
W0 = torch.linspace(0.05, 0.4, 4)        # 초기 둘레 측정용 (w ≈ 0)

rows = []
t0 = time.perf_counter()
for n_lobe in (3, 4, 6):
    for r0 in (8.0, 11.0, 14.0, 17.0, 20.0):
        for amp in np.arange(0.0, 9.01, 1.0):
            if r0 - amp < 1.0 or r0 + amp > R_OUT - 0.5:
                continue
            a = torch.zeros(6); a[n_lobe-1] = float(amp)
            b = torch.zeros(6); r0t = torch.tensor(r0)
            phi = M.ds.phi(r0t, a, b)
            P0 = float(M.ds.perimeter(phi, W0)[0])
            web = float(web_thickness(r0t, a, b, R_OUT))
            tip = min_tip_radius(r0t, a, b)
            tp_, tv_ = thin_feature_penalty(M.ds, phi, LIM)
            rows.append(dict(n=n_lobe, r0=r0, amp=amp, P0=P0, web=web, tip=tip,
                             thin_prop=float(tp_), thin_void=float(tv_)))
print(f"({time.perf_counter()-t0:.0f} s, {len(rows)} 형상)\n")

ok = lambda r: (r["web"] >= LIM.w_web_min and r["tip"] >= LIM.r_tip_min
                and r["thin_prop"] == 0.0 and r["thin_void"] == 0.0)
feas = [r for r in rows if ok(r)]
infeas = [r for r in rows if not ok(r)]
print(f"제작 가능 {len(feas)} / {len(rows)}")
print(f"달성 가능한 최대 초기 둘레: 제약 있음 {max(r['P0'] for r in feas):.1f} mm"
      f"   /  제약 없음 {max(r['P0'] for r in rows):.1f} mm"
      f"   → 제약이 {100*(1-max(r['P0'] for r in feas)/max(r['P0'] for r in rows)):.0f} % 를 잘라낸다")

print("\n둘레를 늘려갈 때 먼저 걸리는 제약은 무엇인가 (lobe 수별)")
for n_lobe in (3, 4, 6):
    sub = sorted([r for r in rows if r["n"] == n_lobe], key=lambda r: -r["P0"])
    first = {}
    for r in sub:
        if ok(r):
            continue
        for key, cond in (("w_web_min", r["web"] < LIM.w_web_min),
                          ("r_tip_min", r["tip"] < LIM.r_tip_min),
                          ("t_prop_min", r["thin_prop"] > 0),
                          ("t_void_min", r["thin_void"] > 0)):
            if cond:
                first[key] = first.get(key, 0) + 1
    best_ok = max((r["P0"] for r in sub if ok(r)), default=0)
    print(f"  {n_lobe} 갈래: 제작 가능 최대 둘레 {best_ok:6.1f} mm   위반 분포 {first}")

fig, ax = plt.subplots(1, 2, figsize=(11.4, 4.4))
for r in rows:
    c = "#2C7A7B" if ok(r) else "#B4441F"
    ax[0].scatter(r["P0"], r["web"], s=16, c=c, alpha=.75)
    ax[1].scatter(r["P0"], r["tip"], s=16, c=c, alpha=.75)
ax[0].axhline(LIM.w_web_min, color="k", ls="--", lw=1.2)
ax[0].text(ax[0].get_xlim()[0]+3, LIM.w_web_min*1.12, "w_web_min = 3.0 mm", fontsize=8)
ax[0].set_xlabel("initial perimeter $P_0$ [mm]  (≈ burn-area demand)")
ax[0].set_ylabel("web thickness [mm]"); ax[0].grid(alpha=.25)
ax[0].set_title("(a) more perimeter → thinner web", fontsize=10)
ax[1].axhline(LIM.r_tip_min, color="k", ls="--", lw=1.2); ax[1].set_yscale("log")
ax[1].text(ax[1].get_xlim()[0]+3, LIM.r_tip_min*1.15, "r_tip_min = 0.8 mm", fontsize=8)
ax[1].set_xlabel("initial perimeter $P_0$ [mm]"); ax[1].set_ylabel("tip radius [mm]")
ax[1].grid(alpha=.25, which="both")
ax[1].set_title("(b) more perimeter → sharper tips", fontsize=10)
import matplotlib.lines as ml
ax[0].legend(handles=[ml.Line2D([],[],marker='o',ls='',color="#2C7A7B",label="manufacturable"),
                      ml.Line2D([],[],marker='o',ls='',color="#B4441F",label="violates a limit")],
             fontsize=8, loc="upper right")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "constraint_map.png"), dpi=160)
print(f"\nsaved {OUT}/constraint_map.png")
