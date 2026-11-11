"""최소 복잡도 탐색 — 가장 단순한 형상으로 목표를 맞춘다.

설계 원칙: 간단한 형상으로 목표 곡선을 만들 수 있으면 복잡하게 갈 이유가 없다.
K = 0(원형)부터 올려가며 처음 허용 오차를 통과하는 K 에서 멈춘다.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from gre.optimize.forward import DiffMotor, resample
from gre.optimize.loss import Target, LossWeights
from gre.optimize.constraints import ManufacturingLimits
from gre.optimize.runner import DesignVars, minimal_complexity_search
from gre.validate.identifiability import assess
from gre.optimize.constraints import ThroatSpec  # noqa: E402


torch.set_default_dtype(torch.float64)
M = DiffMotor(d_outer=60.0, n_grid=121, n_samples=384, n_w=200, h_max=10e-3)
LIM, WTS = ManufacturingLimits(), LossWeights()
L = 95.0

CASES = [("쉬운 목표 (얕은 스타)",  11.0, [0., 0., 2.2, 0.], 78.0),
         ("깊은 스타 (둘레를 많이 요구)", 8.0, [0., 0., 0., 5.0], 120.0)]

for tag, r0, a, at in CASES:
    print(f"\n{'='*66}\n{tag}:  r0={r0}, a={a}, A_t={at}\n{'='*66}")
    tg = DesignVars(r0=torch.tensor(r0), a=torch.tensor(a), b=torch.zeros(4),
                    a_throat=torch.tensor(at), lengths=[L])
    tt, tp, tf, td = M.simulate(tg.r0, tg.a, tg.b, tg.lengths, tg.a_throat)
    print(f"[목표] P_max {float(tp.max()):.3f} MPa  F_max {float(tf.max()):.1f} N  "
          f"I {float(torch.trapezoid(tf,tt)):.1f} N·s  장약 {td['mass_g']:.0f} g")
    wt = td["w_tab"].numpy(); abt = td["a_b"].detach().numpy()
    ident = assess(wt, abt, 30.0, L)
    print(f"[식별성] {ident.note}")
    tgrid = torch.linspace(0.0, float(tt[-1]), 320)
    target = Target(t=tgrid, f=resample(tt, tf, tgrid).detach(), p_max=float(tp.max()))
    t0 = time.perf_counter()
    dv, rep = minimal_complexity_search(M, target, LIM, WTS, k_max=4,
                      seeding="warm",
                                        throat=ThroatSpec.fixed(2.0 * (at / 3.141592653589793) ** 0.5),
                                        iters_per_k=45, lengths=(L,))
    print(f"[결과] {rep['note']}   ({time.perf_counter()-t0:.0f} s)")
    print(f"  r0={float(dv.r0):.3f}  |c_k|={np.round(np.hypot(dv.a.detach().numpy(), dv.b.detach().numpy()),3)}"
          f"  A_t={float(dv.a_throat):.2f}")
    print(f"  제약: 웹 {rep['web_mm']:.2f}  꼭짓점 {rep['r_tip_mm']:.2f}  장약 {rep['mass_g']:.0f}g  "
          f"최대압 {rep['P_max_sim']:.2f}  슬리버 {100*rep['sliver_frac']:.2f}%")
