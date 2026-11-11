"""45 iter 는 충분한가 — K 상승 탐색의 판정이 예산을 재고 있던 문제.

`minimal_complexity_search` 의 기본값은 `iters_per_k = 45` 다. 그 값으로
"두 목표 모두 K = 0 (원형) 으로 통과" 라는 결론을 냈다. 그 결론이 **형상 충분성**
을 잰 것인지 **최적화 예산**을 잰 것인지 확인한다.

K = 0 (원형) 으로 고정하고 300 iter 까지 돌려 모양 오차의 궤적을 본다.

─────────────────────────────────────────────────────────────────────────────
측정 결과 (2026-10-06)

  얕은 스타      45 iter  3.09 %  →  300 iter 최선  2.82 %   (+0.27 %p)
  6-로브   (둘레를 많이 요구)      45 iter 19.42 %  →  300 iter 최선  3.01 %  (+16.42 %p)

**미수렴은 목표에 따라 다르다.** 원형에 가까운 쉬운 목표는 45 iter 에 수렴하지만,
어려운 목표는 45 iter 에서 19 % 로 멈춘다. 그래서 시연 두 목표의 결론
("원형으로 충분") 은 **유효하다** — 둘 다 쉬운 쪽이었다. 위험한 것은 그 다음이다:
어려운 목표에서 K 상승 탐색은 "K=0 부족" 이라 판정하고 K 를 올리는데, 그 판정의
상당 부분이 **수렴을 안 한 탓**이다. 그리고 K 를 올려도 같은 미수렴이 반복된다.

두 번째로 드러난 것 — **손실 가중치와 합격 기준이 어긋나 있다.**

  6-로브 K=0 궤적:  iter  45  모양 18.99 %   I  4.09 %
                    iter 135  모양  3.26 %   I 24.35 %
                    iter 300  모양  3.09 %   I 24.35 %

`LossWeights` 는 shape 1.0 : impulse 1e-1 이라 최적화기가 **총역적을 희생해 모양을
맞추는 것이 옳은 행동**이다. 그런데 합격 기준(`minimal_complexity_search`)은
모양·I·F 를 **모두 10 % 이하**로 요구한다. 즉 최적화기가 열심히 할수록 합격에서
멀어지는 구간이 있다. 가중치를 고치든 합격 기준을 고치든 둘을 맞춰야 한다.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
torch.set_default_dtype(torch.float64)
from gre.optimize.forward import DiffMotor, resample
from gre.optimize.loss import Target, LossWeights
from gre.optimize.constraints import ManufacturingLimits
from gre.optimize.runner import DesignVars, optimize, final_report
from gre.optimize.constraints import ThroatSpec  # noqa: E402

M=DiffMotor(d_outer=60., n_grid=121, n_samples=384, n_w=200, h_max=10e-3)
LIM,WTS=ManufacturingLimits(),LossWeights(); L=95.
CASES=[("6-로브",14.,[0,0,0,0,0,4.5],90.), ("얕은 스타",11.,[0,0,2.2,0,0,0],78.)]
mk=lambda v: torch.tensor(v, dtype=torch.float64, requires_grad=True)
for tag,r0t,arr,att in CASES:
    a=torch.tensor(arr,dtype=torch.float64)
    with torch.no_grad():
        t,p,f,d=M.simulate(torch.tensor(r0t),a,torch.zeros(6),[L],torch.tensor(att))
    tg=torch.linspace(0.,float(t[-1]),320)
    target=Target(t=tg, f=resample(t,f,tg).detach(), p_max=float(p.max()))
    dv=DesignVars(r0=mk(12.0), a=mk(np.zeros(1)), b=mk(np.zeros(1)),
                  a_throat=mk(85.0), lengths=[L])
    dv.a.requires_grad_(False); dv.b.requires_grad_(False)   # K=0: 원형 고정
    t0=time.perf_counter()
    dv,hist=optimize(M,dv,target,LIM,WTS,iters=300,verbose=False,
                     throat=ThroatSpec.fixed(2.0 * (85.0 / 3.141592653589793) ** 0.5))
    print(f"\n{tag}  K=0 (원형) 300 iter, {time.perf_counter()-t0:.0f} s")
    print(f"{'iter':>6}{'모양 %':>9}{'I %':>8}{'F %':>8}{'r0':>8}{'A_t':>8}")
    for it in (10,25,45,90,135,180,240,299):
        h=hist[it]
        print(f"{it:>6}{h['rms_shape_pct']:>9.2f}{h['I_err_pct']:>8.2f}{h['F_err_pct']:>8.2f}"
              f"{'':>8}{'':>8}")
    np.savez(os.path.join(os.path.dirname(os.path.abspath(__file__)), "out",
                          f"conv_{'star' if '스타' in tag else 'lobe'}.npz"),
             shape=np.array([h['rms_shape_pct'] for h in hist]),
             imp=np.array([h['I_err_pct'] for h in hist]),
             fmax=np.array([h['F_err_pct'] for h in hist]))
    best=min(hist,key=lambda h:h['loss'])
    print(f"  최선 iter {best['iter']}: 모양 {best['rms_shape_pct']:.2f}%  I {best['I_err_pct']:.2f}%")
    print(f"  45 iter 시점 {hist[44]['rms_shape_pct']:.2f}%  →  300 iter 최선 "
          f"{min(h['rms_shape_pct'] for h in hist):.2f}%   "
          f"(차이 {hist[44]['rms_shape_pct']-min(h['rms_shape_pct'] for h in hist):+.2f} %p)", flush=True)
