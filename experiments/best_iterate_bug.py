"""`optimize` 가 집합 안 반복을 버리고 집합 밖 반복을 돌려주는가.

`k_budget_recheck.py` 의 예산 회계가 **모순된 행**을 드러냈다 (6-로브 목표).

    K=1: 모양 5.97%  I 10.70%  F 6.65%  [19/120 반복, 누적 139, 집합 진입]  부족 — 총역적 0.7 %p
    K=2: 모양 6.03%  I 10.47%  F 6.82%  [ 9/120 반복, 누적 148, 집합 진입]  부족 — 총역적 0.5 %p

`stop_reason == "집합 진입"` 은 그 반복에서 `in_set` 이 True 였다는 뜻인데 최종
판정은 "부족" 이다. 둘 중 하나가 거짓말을 한다.

의심: `optimize` 는 끝에서 **손실이 가장 낮았던** 반복으로 되돌려 놓는다.

    if best is None or rep["loss"] < best[0]: best = (...)
    ...
    with torch.no_grad(): dv.r0.copy_(best[1]); ...

힌지에서는 **손실 최소 ≠ 집합 안**이다. 집합 안에서 사양 항은 정확히 0 이므로
손실 = `w_pmax · (ΔP/P)²` + 희소성뿐이고, 집합 **밖**의 반복이 사양 초과가 작으면서
(0.7 %p → hinge² = 4.9e-5) 2차 기준이 더 작으면 손실이 더 낮을 수 있다. 그러면
집합에 들어가서 멈췄는데 **집합 밖 설계를 돌려준다.**

재는 방법: K 상승 탐색의 K=0 → K=1 경로를 **그대로 재현**하고 (`minimal_complexity_search`
와 같은 초기값·같은 난수 씨앗), K=1 구간의 모든 반복에서 집합 조건과 손실을 비교한다.
무작위 씨앗에서 차갑게 출발하면 집합에 들어가지 못해(측정: 씨앗 0, 906 s, 집합 안
반복 0 개) 이 결함과 무관한 결과만 나온다 — 그래서 경로를 재현해야 한다.
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

                                 optimize)
from gre.optimize.constraints import ThroatSpec  # noqa: E402

M = DiffMotor(d_outer=60.0, n_grid=121, n_samples=384, n_w=200, h_max=10e-3)
LIM, WTS, TOL, L = ManufacturingLimits(), LossWeights(), Tolerances(), 95.0
ITERS, R0_INIT, AT0 = 120, 12.0, 85.0
mk = lambda v: torch.tensor(v, dtype=torch.float64, requires_grad=True)


def in_set_of(rep):
    """`optimize` 의 조기 종료 조건과 **글자 그대로 같은** 식."""
    return (rep["gap_shape_pp"] == 0.0 and rep["gap_impulse_pp"] == 0.0
            and rep["gap_f_max_pp"] == 0.0 and rep["penalty"] < 1e-9
            and rep["case_over"] == 0.0)


def target_of(r0, arr, at):
    a = torch.tensor(arr, dtype=torch.float64)
    with torch.no_grad():
        t, p, f, d = M.simulate(torch.tensor(r0), a, torch.zeros(6), [L],
                                torch.tensor(at))
    g = torch.linspace(0.0, float(t[-1]), 320)
    return Target(t=g, f=resample(t, f, g).detach(), p_max=float(p.max()))


def segment(dv, tgt, tag):
    """한 구간을 돌리고 '멈춘 반복' 과 '되돌린 반복' 을 비교한다."""
    t0 = time.perf_counter()
    dv, hist = optimize(M, dv, tgt, LIM, WTS, iters=ITERS, verbose=False,
                        tol=TOL, throat=ThroatSpec.fixed(2.0 * (AT0 / 3.141592653589793) ** 0.5))
    rep, _ = final_report(M, dv, tgt, LIM, WTS, tol=TOL)
    adm = rep["admissibility"]
    flags = [in_set_of(h) for h in hist]
    losses = np.array([h["loss"] for h in hist])
    i_best = int(losses.argmin())
    ins = [i for i, f in enumerate(flags) if f]
    print(f"\n[{tag}]  {len(hist)} 반복, {time.perf_counter()-t0:.0f} s"
          f"   종료 {hist[-1].get('stop_reason')} @ {hist[-1].get('stop_iter')}")
    print(f"  집합 안이었던 반복 : {ins if len(ins) <= 10 else str(ins[:10]) + ' …'}"
          f"  (총 {len(ins)})")
    print(f"  손실 최소 반복     : {i_best}  손실 {losses[i_best]:.4e}  "
          f"집합 {'안' if flags[i_best] else '밖'}")
    for i in ins[:3]:
        h = hist[i]
        print(f"    · 집합 안 반복 {i:3d}: 손실 {h['loss']:.4e}  "
              f"모양 {h['rms_shape_pct']:5.2f}%  I {h['I_err_pct']:5.2f}%  "
              f"F {h['F_err_pct']:5.2f}%  P오차 {h['P_err_pct']:5.2f}%")
    if not flags[i_best]:
        h = hist[i_best]
        print(f"    · 손실 최소 {i_best:3d}: 손실 {h['loss']:.4e}  "
              f"모양 {h['rms_shape_pct']:5.2f}%  I {h['I_err_pct']:5.2f}%  "
              f"F {h['F_err_pct']:5.2f}%  P오차 {h['P_err_pct']:5.2f}%")
    print(f"  **반환된 설계**     : 모양 {rep['rms_shape_pct']:.2f}%  "
          f"I {rep['I_err_pct']:.2f}%  F {rep['F_err_pct']:.2f}%  → "
          + ("집합 안" if adm["in_set"]
             else f"집합 밖 ({adm['binding']} {adm['worst_gap_pp']:.2f} %p)"))
    hit = bool(ins) and not adm["in_set"]
    if hit:
        print("  → ⚠ **결함 확인.** 집합 안 반복을 방문했는데 집합 밖 설계를 "
              "돌려줬다. 손실 최소 규칙이 힌지와 맞지 않는다.")
    elif ins:
        print("  → 정상 (집합 안 설계를 돌려줬다)")
    else:
        print("  → 이 구간은 집합에 들어간 적이 없다 — 이 결함과 무관")
    return dv, hist, hit


if __name__ == "__main__":
    tgt = target_of(14.0, [0, 0, 0, 0, 0, 4.5], 90.0)      # 6-로브
    hits = 0

    # ── K=0 구간 (minimal_complexity_search 와 동일한 초기값) ─────────────
    dv = DesignVars(r0=mk(R0_INIT), a=mk(np.zeros(1)), b=mk(np.zeros(1)),
                    a_throat=mk(AT0), lengths=[L])
    dv.a.requires_grad_(False)
    dv.b.requires_grad_(False)
    dv, _, hit = segment(dv, tgt, "K=0")
    hits += hit
    prev = (float(dv.r0.detach()), float(dv.a_throat.detach()))

    # ── K=1 구간 (웜 스타트 + 새 모드에 g.normal(0, 0.25), 씨앗 100+1) ────
    g = np.random.default_rng(101)
    a0, b0 = np.zeros(1), np.zeros(1)
    a0[0] += g.normal(0, 0.25)
    b0[0] += g.normal(0, 0.25)
    print(f"\n  (K=1 새 모드 초기 섭동 |c₁| = {np.hypot(a0[0], b0[0]):.3f} mm)")
    dv = DesignVars(r0=mk(prev[0]), a=mk(a0), b=mk(b0), a_throat=mk(prev[1]),
                    lengths=[L])
    dv, _, hit = segment(dv, tgt, "K=1 (웜 스타트)")
    hits += hit

    print(f"\n{'='*72}\n결함 재현 {hits} / 2 구간")
    if hits:
        print("고칠 곳: `optimize` 가 되돌릴 반복을 고르는 규칙.")
        print("  현행 : 손실 최소        ← 힌지에서 집합 안을 보장하지 않는다")
        print("  제안 : 집합 안 반복이 있으면 그중 2차 기준이 가장 좋은 것,")
        print("         하나도 없으면 집합까지의 거리(worst_gap_pp)가 가장 작은 것")
        print("곁가지: `optimize` 의 조기 종료 조건은 `admissibility` 보다 좁다 —")
        print("  웹 두께·꼭짓점 반경·장약 질량·연결성을 보지 않는다. 같은 식을 쓰게 한다.")
