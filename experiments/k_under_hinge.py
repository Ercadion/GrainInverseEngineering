"""힌지 아래에서 K 상승 탐색은 무엇을 고르는가.

손실을 힌지로 바꾼 뒤 K 상승 탐색을 다시 돌린다. 이전의 "6-로브는 K=3 까지
올려도 부족" 은 형상 문제가 아니라 손실 문제였으므로 전부 다시 재야 한다.

측정 결과 (2026-10-07, iters_per_k = 120):

    목표              K     모양%    I%     F%   복잡도   시간    K 표(초과 %p)
    얕은 스타          0    3.76   2.17   1.66   0.00     7s    K0 통과
    깊은 스타(4-로브)   0    9.76   1.07   8.25   0.00     6s    K0 통과
    6-로브          None   5.98  10.19   7.18   0.68   605s    K0 3 → K1 1 → K2 0.2 → K3 0.2
    6-로브 깊음        1    9.32   8.93   5.43   0.01   462s    K0 10 → K1 통과
    3+5 복합          0    9.74   3.35   7.02   0.00   127s    K0 통과

세 가지가 보인다.

1. **통과하는 목표는 7 초에 끝난다.** 조기 종료(집합 진입) 덕이다. 통과 못 하는
   목표만 전 예산을 쓴다 (462~605 초). 이전에는 전부 전 예산을 썼다.
2. **6-로브는 0.2 %p 차이로 경계에 걸린다.** 같은 목표를 K=0 에 300 반복 주면
   총역적 9.42 % 로 **들어간다**. 즉 실현 불가가 아니라 예산 문제다. 판정이
   "0.2 %p 초과" 라고 숫자를 주기 때문에 그 사실이 보인다 — 이진 판정이었으면
   '불가' 로 묻혔을 것이다.
3. ⚠ **6-로브 깊음의 K=1 통과는 모드가 한 일이 아니다.** 복잡도가 0.01 mm 로
   사실상 0 이다. 웜 스타트는 K 를 올릴 때마다 `iters_per_k` 를 새로 주므로
   K=1 의 해는 K=0 의 예산을 물려받은 상태에서 120 반복을 더 쓴 것이다.
   새 모드의 초기 섭동 |c₁| = 0.546 mm 가 0.010 mm 로 **55 배 줄었다** — 모드를
   못 써서 0 인 것이 아니라 최적화기가 적극적으로 0 으로 밀어낸 것이다.
   **K 가 올라가서 통과한 것인지 예산이 늘어서 통과한 것인지 구분해야 한다.**
   이전 수정에서 `minimal_complexity_search` 에 예산 회계(`k_table` 의 `iters_cum`,
   `stop_reason`)와 `k0_recheck`(K=0 에 같은 누적 예산·같은 재시작 횟수)를 넣었다.
   재확인 결과는 `k_budget_recheck.py` 에 있다.

⚠ **위 표는 낡았다 (2026-10-08).** 두 가지가 바뀌었다.

   1. 이전 판의 K=0 재확인이 켜진 뒤 '6-로브 깊음' 은 K = **0** 이다 (거짓 양성).
   2. `optimize` 의 되돌림 규칙을 고친 뒤 '6-로브' 는 "실현 불가(총역적 0.2 %p)" 가
      아니라 K = **0 통과**다. 초판은 집합 안 반복을 방문했는데 `argmin loss` 가
      집합 밖 반복을 골라 버렸다 (`best_iterate_bug.py`).

   현행 값은 `k_budget_recheck.py` 의 docstring 에 있다. 이 파일을 다시 돌리면
   '얕은 스타 / 깊은 스타 / 3+5 복합' 도 숫자가 조금 달라질 수 있다 (판정은 K=0 유지
   예상 — 셋 다 K=0 에서 조기 종료하므로 되돌림 규칙의 영향이 작다).
"""
import sys, time; sys.path.insert(0,'/home/claude/gre_build')
import numpy as np, torch
torch.set_default_dtype(torch.float64)
from gre.optimize.forward import DiffMotor, resample
from gre.optimize.loss import LossWeights, Target, Tolerances
from gre.optimize.constraints import ManufacturingLimits
from gre.optimize.runner import minimal_complexity_search
from gre.optimize.constraints import ThroatSpec  # noqa: E402

M=DiffMotor(d_outer=60., n_grid=121, n_samples=384, n_w=200, h_max=10e-3)
LIM, WTS, TOL, L = ManufacturingLimits(), LossWeights(), Tolerances(), 95.
CASES=[("얕은 스타",11.,[0,0,2.2,0,0,0],78.),
       ("깊은 스타(4-로브)",8.,[0,0,0,5.,0,0],120.),
       ("6-로브",14.,[0,0,0,0,0,4.5],90.),
       ("6-로브 깊음",14.,[0,0,0,0,0,7.0],90.),
       ("3+5 복합",12.,[0,0,3.0,0,2.5,0],85.)]
print(f"{'목표':<18}{'K':>4}{'모양%':>8}{'I%':>8}{'F%':>8}{'복잡도':>8}{'시간':>7}  비고")
for tag,r0,arr,at in CASES:
    a=torch.tensor(arr,dtype=torch.float64)
    with torch.no_grad(): t,p,f,d=M.simulate(torch.tensor(r0),a,torch.zeros(6),[L],torch.tensor(at))
    g=torch.linspace(0.,float(t[-1]),320)
    T=Target(t=g,f=resample(t,f,g).detach(),p_max=float(p.max()))
    t0=time.perf_counter()
    dv,rep=minimal_complexity_search(M,T,LIM,WTS,k_max=3,iters_per_k=120,
                   seeding="warm",
                                     throat=ThroatSpec.fixed(2.0 * (85.0 / 3.141592653589793) ** 0.5),
                                     lengths=(L,),tol=TOL,verbose=False)
    dt=time.perf_counter()-t0
    k=rep["k_chosen"]
    ktab=" ".join(f"K{r['K']}:{'O' if r['passed'] else f'{r[chr(119)+chr(111)+chr(114)+chr(115)+chr(116)+chr(95)+chr(103)+chr(97)+chr(112)+chr(95)+chr(112)+chr(112)]:.0f}'}" for r in rep["k_table"])
    print(f"{tag:<18}{str(k):>4}{rep['rms_shape_pct']:>8.2f}{rep['I_err_pct']:>8.2f}"
          f"{rep['F_err_pct']:>8.2f}{rep['complexity']:>8.2f}{dt:>6.0f}s  {ktab}")
    if k is None: print(f"      {rep['note'][:150]}")
