"""K 상승 탐색이 '단순한 해'를 고르는가, '탈출에 실패'하는가.

`minimal_complexity_search` 는 K = 0(원형)에서 시작해 웜 스타트로 K 를 올린다.
그 docstring 은 "원형 안장점을 K=0 단계에서 이미 지나왔으므로 대칭이 자동으로
깨진다" 고 적고 있다. **거꾸로다.** K=0 해는 a = b = 0, 즉 **안장점 위**다.
지나온 것이 아니라 거기 앉아 있는 상태에서 출발한다. K 를 하나 올릴 때 새로
열린 모드에만 N(0, 0.25) 를 주는데, `seed_design` 이 전 모드에 주는 0.3~0.4 mm
섭동보다 약하다.

그래서 의심: **K 상승 탐색이 저-K 해를 고르는 이유가 "충분해서" 가 아니라
"탈출을 못 해서" 일 수 있다.**

공정한 비교를 위해 **예산을 같게** 둔다.

    A (웜 스타트)  : K = 0 → 1 → 2 → 3, 각 45 iter = 180 iter, 궤적 1개
    B (독립 출발)  : K = 3 에서 서로 다른 씨앗 4개 × 45 iter = 180 iter, 궤적 4개

같은 180 iter 를 쓰고 B 가 더 좋은 해를 찾으면, K 상승 탐색은 "단순함을 고르는
장치" 가 아니라 **"탈출 실패를 단순함으로 오인하는 장치"** 다.

목표는 원형으로 안 되는 쪽을 쓴다 — 6-로브(circle_screen: 노즐목 자유 8.20 %,
고정 13.96 %). 원형으로 되는 목표에서는 이 질문 자체가 성립하지 않는다.

⚠ **초판 정정 (2026-10-07).** 이 측정은 **초판 손실(가중 제곱합)** 기준이다.
   손실을 힌지로 바꾼 뒤 다시 재 보니 6-로브 목표는 **K = 0(원형)으로도 실현
   가능**했다 (K=0 에 300 반복: 모양 4.59 %, I 9.42 %, F 5.51 % → 집합 안).
   즉 "원형으로 안 되는 목표" 라는 전제가 틀렸고, 이전의 "K=3 까지 올려도 부족"
   은 형상 문제가 아니라 **손실 문제**였다.

   그래도 이 파일의 **결론 자체는 유효하다** — 같은 예산에서 웜 스타트가 독립
   다중 출발보다 불리하지 않다는 것은 손실과 무관한 성질이다. 다만 숫자는 초판
   손실에서 나온 것이므로 그대로 인용하면 안 된다. 힌지 기준 재측정은
   `k_under_hinge.py` 에 있다.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from gre.optimize.constraints import ManufacturingLimits
from gre.optimize.forward import DiffMotor, resample
from gre.optimize.loss import LossWeights, Target
from gre.optimize.runner import (DesignVars, final_report,

                                 minimal_complexity_search, optimize,
                                 seed_design, shape_distance)
from gre.optimize.constraints import ThroatSpec  # noqa: E402

torch.set_default_dtype(torch.float64)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)

M = DiffMotor(d_outer=60.0, n_grid=121, n_samples=384, n_w=200, h_max=10e-3)
LIM, WTS = ManufacturingLimits(), LossWeights()
L, K, ITERS = 95.0, 3, 45

#: (이름, 참 r0, 참 계수, 참 A_t) — 원형으로는 부족한 목표들
CASES = [("6-로브", 14.0, [0, 0, 0, 0, 0, 4.5], 90.0),
         ("4-로브 깊음", 8.0, [0, 0, 0, 5.0, 0, 0], 120.0)]


def make_target(r0, arr, at):
    a = torch.tensor(arr, dtype=torch.float64)
    with torch.no_grad():
        t, p, f, d = M.simulate(torch.tensor(r0), a, torch.zeros(6), [L],
                                torch.tensor(at))
    tg = torch.linspace(0.0, float(t[-1]), 320)
    return Target(t=tg, f=resample(t, f, tg).detach(), p_max=float(p.max())), (r0, arr, at)


def amp(dv):
    return np.hypot(dv.a.detach().numpy(), dv.b.detach().numpy())


for tag, r0t, arrt, att in CASES:
    print(f"\n{'='*72}\n{tag}  (참 r0={r0t}, A_t={att}, 계수 {arrt})\n{'='*72}")
    target, truth = make_target(r0t, arrt, att)
    true_shape = (r0t, np.array(arrt[:K] + [0.0] * max(0, K - len(arrt))),
                  np.zeros(K))

    # ── A: 웜 스타트 (K 상승) ─────────────────────────────────────────
    t0 = time.perf_counter()
    dv_a, rep_a = minimal_complexity_search(M, target, LIM, WTS, k_max=K,
                          seeding="warm",
                                            throat=ThroatSpec.band(2.0 * (50.0 / 3.141592653589793) ** 0.5,
                         2.0 * (140.0 / 3.141592653589793) ** 0.5),
                                            iters_per_k=ITERS, lengths=(L,),
                                            verbose=False)
    ta = time.perf_counter() - t0
    k_tab = {r["K"]: r for r in rep_a["k_table"]}
    at_k = k_tab.get(K)
    print(f"\n[A] 웜 스타트 K 상승  ({ta:.0f} s, {ITERS*(max(k_tab)+1)} iter)")
    for r in rep_a["k_table"]:
        print(f"    K={r['K']}: 모양 {r['rms_shape_pct']:6.2f}%  복잡도 {r['complexity']:5.2f}"
              f"  {'통과' if r['passed'] else '부족'}")
    print(f"    선택: K={rep_a['k_chosen']}  — {rep_a['note']}")

    # ── B: 독립 다중 출발 (K 고정) ────────────────────────────────────
    print(f"\n[B] 독립 출발 K={K} × 4 씨앗  (같은 예산 {4*ITERS} iter)")
    t0 = time.perf_counter()
    best_b, runs = None, []
    for s in range(4):
        rng = np.random.default_rng(1000 + s)
        dv = seed_design(r0=float(rng.uniform(8.0, 18.0)), n_modes=K,
                         a_throat=float(rng.uniform(50.0, 140.0)), lengths=(L,),
                         throat=ThroatSpec.band(2.0 * (50.0 / 3.141592653589793) ** 0.5,
                         2.0 * (140.0 / 3.141592653589793) ** 0.5),
                         seed_amplitude=0.4, seed=s)
        dv, _ = optimize(M, dv, target, LIM, WTS, iters=ITERS, verbose=False,
                         throat=ThroatSpec.band(2.0 * (50.0 / 3.141592653589793) ** 0.5,
                         2.0 * (140.0 / 3.141592653589793) ** 0.5))
        rep, _ = final_report(M, dv, target, LIM, WTS)
        feas = all(rep[x] for x in ("ok_web", "ok_tip", "ok_mass", "ok_case",
                                    "ok_connected"))
        runs.append((rep, dv, feas))
        print(f"    seed {s}: 모양 {rep['rms_shape_pct']:6.2f}%  I {rep['I_err_pct']:5.2f}%"
              f"  복잡도 {rep['complexity']:5.2f}  |c_k| {np.round(amp(dv), 2)}"
              f"  {'제작가능' if feas else '제작불가'}")
        if best_b is None or rep["rms_shape_pct"] < best_b[0]["rms_shape_pct"]:
            best_b = (rep, dv, feas)
    tb = time.perf_counter() - t0

    # ── 판정 ──────────────────────────────────────────────────────────
    print(f"\n[비교]  ({ta:.0f} s vs {tb:.0f} s)")
    if at_k is not None:
        print(f"  K={K} 모양 오차 :  웜 스타트 {at_k['rms_shape_pct']:6.2f} %"
              f"   독립 출발 최선 {best_b[0]['rms_shape_pct']:6.2f} %"
              f"   차이 {at_k['rms_shape_pct']-best_b[0]['rms_shape_pct']:+6.2f} %p")
        print(f"  K={K} 복잡도    :  웜 스타트 {at_k['complexity']:6.2f}"
              f"   독립 출발 최선 {best_b[0]['complexity']:6.2f}")
    d_true = shape_distance(true_shape,
                            (float(best_b[1].r0), best_b[1].a.detach().numpy(),
                             best_b[1].b.detach().numpy()))
    print(f"  독립 출발 최선이 참 형상과 떨어진 거리: {d_true:.2f} mm")
    if at_k is not None and best_b[0]["rms_shape_pct"] < at_k["rms_shape_pct"] - 1.0:
        print("  → **웜 스타트가 손해다.** 같은 예산으로 독립 출발이 더 좋은 해를 찾는다.")
        print("     K 상승 탐색이 고르는 저-K 해는 '충분해서' 가 아니라 '탈출 실패' 일 수 있다.")
    else:
        print("  → 웜 스타트가 불리하지 않다. K 상승 탐색의 저-K 선호는 정당하다.")
