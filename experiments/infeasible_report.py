"""실현 불가 목표 — 억지로 설계를 내놓으면 무엇이 나빠지는가.

사용자 결정(2026-10-07): 모든 사양을 허용 오차 안에 넣을 수 없으면 우선순위가
낮은 항목을 양보해 설계 하나를 내놓는 것이 아니라, **"이 곡선으로는 주어진
조건에서 맞는 형상을 찾을 수 없습니다"** 라고 보고한다.

그 결정이 무엇을 고치는지 잰다. 핵심은 **반환값이 목표가 아니라 최적화 경로에
의해 정해진다**는 것이다. 실현 불가 목표에 씨앗만 바꿔 던지면 초판은 매번 다른
설계를 내놓는데, 사용자는 그것을 '답' 으로 읽는다.

실현 불가 목표 만들기 — 6-로브 곡선의 추력을 1.6 배로 키운다. 시간축은 그대로이므로
총역적도 1.6 배가 되고, 같은 케이스·장약 상한으로는 낼 수 없다. (6-로브 원본은
힌지 손실에서 K = 0 으로도 **실현 가능**하다는 것이 밝혀졌으므로,
'실현 불가' 예시로 쓸 수 없다 — 그래서 배율을 건다.)
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
from gre.optimize.runner import (DesignVars, admissibility, final_report,

                                 optimize, verdict_text)
from gre.optimize.constraints import ThroatSpec  # noqa: E402

M = DiffMotor(d_outer=60.0, n_grid=121, n_samples=384, n_w=200, h_max=10e-3)
LIM, TOL, L = ManufacturingLimits(), Tolerances(), 95.0
OLD = LossWeights(shape=1.0, impulse=1e-1, f_max=1e-2)   # 초판 — 우선순위 가중치
NEW = LossWeights()                                      # 힌지
NO_TOL = Tolerances(shape=0.0, impulse=0.0, f_max=0.0)   # 힌지를 끈 상태
mk = lambda v: torch.tensor(v, dtype=torch.float64, requires_grad=True)
SCALE = 1.6


def target_scaled():
    a = torch.zeros(6)
    a[5] = 4.5
    with torch.no_grad():
        t, p, f, d = M.simulate(torch.tensor(14.0), a, torch.zeros(6), [L],
                                torch.tensor(90.0))
    tg = torch.linspace(0.0, float(t[-1]), 320)
    return Target(t=tg, f=SCALE * resample(t, f, tg).detach(),
                  p_max=float(p.max()) * SCALE)


def run(target, weights, tol, stop, seed, iters=120, modes=3):
    rng = np.random.default_rng(700 + seed)
    dv = DesignVars(r0=mk(float(rng.uniform(8.0, 20.0))),
                    a=mk(rng.normal(0, 0.4, modes)), b=mk(rng.normal(0, 0.4, modes)),
                    a_throat=mk(float(rng.uniform(50.0, 160.0))), lengths=[L])
    dv, _ = optimize(M, dv, target, LIM, weights, iters=iters, verbose=False,
                     throat=ThroatSpec.band(2.0 * (50.0 / 3.141592653589793) ** 0.5,
                         2.0 * (160.0 / 3.141592653589793) ** 0.5),
                     tol=tol, stop_in_set=stop, patience=10 ** 9)
    rep, _ = final_report(M, dv, target, LIM, weights, tol=TOL)
    return dv, rep


if __name__ == "__main__":
    tgt = target_scaled()
    print(f"목표: 6-로브 곡선 × {SCALE}  "
          f"(I {float(torch.trapezoid(tgt.f, tgt.t)):.0f} N·s, "
          f"F_max {tgt.f_max:.0f} N)\n")

    rows = {"초판 (우선순위 가중치)": [], "힌지 (판정)": []}
    for tag, w, tol, stop in (("초판 (우선순위 가중치)", OLD, NO_TOL, False),
                              ("힌지 (판정)", NEW, TOL, True)):
        print("=" * 76)
        print(tag)
        print("=" * 76)
        t0 = time.perf_counter()
        for s in range(4):
            dv, rep = run(tgt, w, tol, stop, s)
            adm = rep["admissibility"]
            rows[tag].append((float(dv.r0), float(dv.a_throat), rep, adm))
            amp = np.hypot(dv.a.detach().numpy(), dv.b.detach().numpy())
            print(f"  seed {s}: r0 {float(dv.r0):6.2f}  A_t {float(dv.a_throat):6.1f}  "
                  f"|c_k| {np.round(amp, 2)}  "
                  f"모양 {rep['rms_shape_pct']:6.2f}%  I {rep['I_err_pct']:6.2f}%  "
                  f"F {rep['F_err_pct']:6.2f}%  "
                  f"{'집합 안' if adm['in_set'] else '집합 밖'}")
        print(f"  ({time.perf_counter()-t0:.0f} s)")

        r0s = np.array([x[0] for x in rows[tag]])
        ats = np.array([x[1] for x in rows[tag]])
        binds = [x[3]["binding"] for x in rows[tag]]
        gaps = np.array([x[3]["worst_gap_pp"] for x in rows[tag]])
        print(f"\n  설계 흩어짐 : r0 {np.ptp(r0s):.2f} mm 폭   "
              f"A_t {np.ptp(ats):.1f} mm² 폭 ({ats.max()/max(ats.min(),1e-9):.1f} 배)")
        print(f"  판정 일관성 : 걸리는 사양 {set(binds)}   "
              f"초과량 {gaps.min():.1f} ~ {gaps.max():.1f} %p "
              f"(폭 {np.ptp(gaps):.1f} %p)\n")

    print("=" * 76)
    print("사용자에게 나가는 것")
    print("=" * 76)
    print("\n[초판] 설계 하나를 답으로 내놓는다 — 씨앗마다 다른 설계다:")
    for s, (r0, at, rep, _) in enumerate(rows["초판 (우선순위 가중치)"]):
        print(f"   seed {s}: 포트 반경 {r0:.2f} mm, 노즐목 {at:.1f} mm² "
              f"(모양 {rep['rms_shape_pct']:.1f} %, I {rep['I_err_pct']:.1f} %)")
    print("\n[힌지] 설계 대신 판정을 내놓는다:")
    print("  " + verdict_text(rows["힌지 (판정)"][0][3], k_max=3).replace("  ", "\n  "))
