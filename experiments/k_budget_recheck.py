"""K 상승 탐색 — "K 가 올라가서" 인가 "예산이 늘어서" 인가.

`k_under_hinge.py` 에서 **6-로브 깊음**이 K=1 로 통과했는데 복잡도가 0.01 mm
였다. 모드가 놀았다는 뜻이다. 초판은 경고만 띄웠다. 이전 수정에서 `k0_recheck` 로
**재다**: K=0 에 K 상승 탐색이 실제로 쓴 예산(과 같은 재시작 횟수)을 그대로 주고
다시 돌려, K=0 이 거기서 통과하면 `k_chosen` 을 0 으로 되돌린다.

공정한 비교의 정의가 중요하다. K 를 하나 올리는 것은 **Adam 재시작**이기도 하다
(모멘트 초기화 + 새 모드 섭동). 그래서 K=0 에도 같은 라운드 수를 주고 해를
이어가며 돌린다. 차이는 **모드를 0 에 얼려둔 것 하나**뿐이다.

가설 두 개가 갈린다.
    H1 (예산이 한 일) : K=0 이 같은 누적 예산에서 통과한다 → K>0 판정은 거짓 양성
    H2 (모드가 한 일) : K=0 이 같은 누적 예산에서도 부족하다 → K>0 판정이 옳다

측정 결과 (2026-10-12, iters_per_k=120, 둘레 반 스텝 중심차분 수정 후)

    6-로브                                                                396 s
        K        모양%      I%      F%   복잡도   쓴/예산   누적   종료
        0        8.55   11.74   13.74   0.000   120/120    120   반복 수 소진  부족 F 3.7 %p
        1        9.85    9.43    2.93   0.020    17/120    137   집합 진입    통과
        0(재확인) 9.57    9.19    3.58   0.000   139/240    139   집합 진입    통과
        → 선택 K = 0.   **H1 — 예산이 한 일이었다.** K=1 은 17 반복만 더 썼고 그
          사이 모드는 0.020 mm 밖에 안 움직였다. K=0 은 2 반복만 더 주면 들어간다.

    6-로브 깊음                                                           1044 s
        0       12.42   19.61    4.05   0.000   120/120    120   반복 수 소진  부족 I 9.6 %p
        1       10.53    9.84   10.54   0.003    59/120    179   손실 정체    부족 F 0.5 %p
        2       10.35   10.13   10.44   0.566    26/120    205   손실 정체    부족 F 0.4 %p
        3        9.88    5.93    9.97   0.428    90/120    295   집합 진입    통과
        0(재확인)10.46    9.45   10.42   0.000   238/480    238   손실 정체    부족 모양 0.5 %p
        → 선택 K = 3.   **H2 — 모드가 한 일이었다.** 같은 예산을 줘도 K=0 이 0.5 %p
          모자라고, 채택된 설계의 복잡도 0.428 mm 는 격자 한 칸(0.5 mm)에 가깝다.

**절차는 양방향으로 작동한다** — 거짓 양성(6-로브)을 걸러내고, 진짜 모드 요구
(6-로브 깊음)는 확정해 준다. 재확인이 없으면 둘 다 "K>0 필요" 로 보고된다.

⚠ **이전에는 6-로브 깊음도 K = 0 으로 나왔다.** 그때는 소진 근처 둘레의
1차 후방차분이 `A_b` 를 3 % 부풀려 **원형이 닿을 수 있는 곡선의 범위를 과대평가**
하고 있었다. 둘레를 반 스텝 중심차분으로 고치자 원형으로는 이 곡선을
낼 수 없다는 것이 드러났다. 모델이 정확해지면서 판정이 바뀐 사례다.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

torch.set_default_dtype(torch.float64)

from gre.optimize.constraints import ManufacturingLimits
from gre.optimize.forward import DiffMotor, resample
from gre.optimize.loss import LossWeights, Target, Tolerances
from gre.optimize.runner import minimal_complexity_search
from gre.optimize.constraints import ThroatSpec  # noqa: E402


M = DiffMotor(d_outer=60.0, n_grid=121, n_samples=384, n_w=200, h_max=10e-3)
LIM, WTS, TOL, L = ManufacturingLimits(), LossWeights(), Tolerances(), 95.0

#: `k_under_hinge.py` 에서 K>0 을 고른 목표들. K=0 으로 통과한 목표는 재확인이
#: 돌지 않으므로(추가 비용 0) 여기서 다시 잴 필요가 없다.
CASES = [("6-로브 깊음", 14.0, [0, 0, 0, 0, 0, 7.0], 90.0),
         ("6-로브", 14.0, [0, 0, 0, 0, 0, 4.5], 90.0)]
ITERS = 120


def target_of(r0, arr, at):
    a = torch.tensor(arr, dtype=torch.float64)
    with torch.no_grad():
        t, p, f, d = M.simulate(torch.tensor(r0), a, torch.zeros(6), [L],
                                torch.tensor(at))
    g = torch.linspace(0.0, float(t[-1]), 320)
    return Target(t=g, f=resample(t, f, g).detach(), p_max=float(p.max()))


if __name__ == "__main__":
    for tag, r0, arr, at in CASES:
        print("=" * 78)
        print(f"{tag}  (참 r0={r0}, A_t={at}, 계수 {arr})   iters_per_k={ITERS}")
        print("=" * 78)
        tgt = target_of(r0, arr, at)
        t0 = time.perf_counter()
        dv, rep = minimal_complexity_search(M, tgt, LIM, WTS, k_max=3,
                              seeding="warm",
                                            throat=ThroatSpec.fixed(2.0 * (at / 3.141592653589793) ** 0.5),
                                            iters_per_k=ITERS, lengths=(L,),
                                            tol=TOL, verbose=True)
        dt = time.perf_counter() - t0

        print(f"\n  {'K':>10} {'모양%':>7} {'I%':>7} {'F%':>7} {'복잡도':>8}"
              f" {'쓴/예산':>9} {'누적':>6} {'종료':>10}  판정")
        for r in rep["k_table"]:
            name = "0 (재확인)" if r.get("recheck") else str(r["K"])
            verdict = ("통과" if r["passed"]
                       else f"{r['binding'] or '제작'} {r['worst_gap_pp']:.1f} %p")
            print(f"  {name:>10} {r['rms_shape_pct']:>7.2f} {r['I_err_pct']:>7.2f}"
                  f" {r['F_err_pct']:>7.2f} {r['complexity']:>8.3f}"
                  f" {r['iters_used']:>4}/{r['iters_budget']:<4} {r['iters_cum']:>6}"
                  f" {r['stop_reason']:>10}  {verdict}")
        print(f"\n  선택 K = {rep['k_chosen']}   ({dt:.0f} s)")
        print(f"  {rep['note']}\n")

        # ── 가설 판정 ────────────────────────────────────────────────────
        rc = [r for r in rep["k_table"] if r.get("recheck")]
        if not rc:
            print("  재확인이 돌지 않았다 — K=0 으로 통과했거나 아무 K 도 통과하지 "
                  "못했다. 가를 것이 없다.\n")
        elif rep["k_chosen"] == 0:
            print("  → **H1: 예산이 한 일이었다.** K>0 판정은 거짓 양성이다. "
                  "같은 예산을 받으면 원형으로 집합에 들어간다.\n")
        else:
            print(f"  → **H2: 모드가 한 일이었다.** 같은 예산을 줘도 K=0 은 "
                  f"{rc[0]['binding'] or '제작 제약'} 이 {rc[0]['worst_gap_pp']:.1f} %p "
                  f"모자랐다. K={rep['k_chosen']} 판정이 옳다.\n")
