"""힌지 손실 A/B — 사양을 '한 점' 이 아니라 '허용 집합' 으로 두면 무엇이 달라지는가.

세 가지를 잰다.

  (1) 집합 안에 들어간 뒤에도 계속 미는가      — 6-로브 K=0 궤적의 총역적
  (2) 실현 가능한 목표는 여전히 통과하는가      — 얕은 스타 회귀 확인
  (3) 실현 불가 목표에서 답이 흔들리는가        — 씨앗 4개로 반환 설계 비교

(3)이 핵심이다. 억지로 설계를 내놓으면 그 설계는 **목표가 아니라 최적화 경로**가
정한다. 돌릴 때마다 다른 답이 나오는데 사용자는 그것을 '답' 으로 읽는다.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
torch.set_default_dtype(torch.float64)
from gre.optimize.forward import DiffMotor, resample
from gre.optimize.loss import LossWeights, Target, Tolerances
from gre.optimize.constraints import ManufacturingLimits
from gre.optimize.runner import (DesignVars, final_report, optimize, seed_design,

                                 shape_distance, verdict_text)
from gre.optimize.constraints import ThroatSpec  # noqa: E402

M = DiffMotor(d_outer=60.0, n_grid=121, n_samples=384, n_w=200, h_max=10e-3)
LIM, TOL, L = ManufacturingLimits(), Tolerances(), 95.0
#: 초판 가중치 — λ_모양 : λ_I : λ_F = 10 : 1 : 0.1, 힌지 없음
OLD = LossWeights(shape=1.0, impulse=1e-1, f_max=1e-2)
NEW = LossWeights()                      # 힌지 + 가중치 동일
mk = lambda v: torch.tensor(v, dtype=torch.float64, requires_grad=True)


def make_target(r0, arr, at):
    a = torch.tensor(arr, dtype=torch.float64)
    with torch.no_grad():
        t, p, f, d = M.simulate(torch.tensor(r0), a, torch.zeros(6), [L], torch.tensor(at))
    tg = torch.linspace(0.0, float(t[-1]), 320)
    return Target(t=tg, f=resample(t, f, tg).detach(), p_max=float(p.max()))


def run_k0(target, weights, iters, hinge):
    """K = 0 (원형) 고정. hinge=False 면 초판 손실 경로(힌지 문턱을 0 으로)."""
    dv = DesignVars(r0=mk(12.0), a=mk(np.zeros(1)), b=mk(np.zeros(1)),
                    a_throat=mk(85.0), lengths=[L])
    dv.a.requires_grad_(False); dv.b.requires_grad_(False)
    tol = TOL if hinge else Tolerances(shape=0.0, impulse=0.0, f_max=0.0)
    dv, hist = optimize(M, dv, target, LIM, weights, iters=iters, verbose=False,
                        throat=ThroatSpec.band(2.0 * (50.0 / 3.141592653589793) ** 0.5,
                         2.0 * (140.0 / 3.141592653589793) ** 0.5),
                        tol=tol, stop_in_set=hinge, patience=10**9)
    rep, _ = final_report(M, dv, target, LIM, weights, tol=TOL)
    return dv, rep, hist


print("=" * 78)
print("(1) 6-로브 — 집합 안에 들어간 뒤에도 계속 미는가")
print("=" * 78)
tgt6 = make_target(14.0, [0, 0, 0, 0, 0, 4.5], 90.0)
for tag, w, hinge in (("초판 (가중 제곱합)", OLD, False), ("힌지", NEW, True)):
    t0 = time.perf_counter()
    dv, rep, hist = run_k0(tgt6, w, 300, hinge)
    sr = hist[-1].get("stop_reason", "반복 수 소진")
    print(f"\n[{tag}]  ({time.perf_counter()-t0:.0f} s, 종료: {sr})")
    print(f"  {'반복':>6}{'모양 %':>9}{'I %':>9}{'F %':>9}")
    for it in (45, 90, 135, 200, len(hist) - 1):
        if it < len(hist):
            h = hist[it]
            print(f"  {it:>6}{h['rms_shape_pct']:>9.2f}{h['I_err_pct']:>9.2f}{h['F_err_pct']:>9.2f}")
    print(f"  최종: 모양 {rep['rms_shape_pct']:.2f} %  I {rep['I_err_pct']:.2f} %  "
          f"F {rep['F_err_pct']:.2f} %  → {'집합 안' if rep['in_set'] else '집합 밖'}")

print("\n" + "=" * 78)
print("(2) 얕은 스타 (실현 가능) — 회귀 확인")
print("=" * 78)
tgtS = make_target(11.0, [0, 0, 2.2, 0, 0, 0], 78.0)
for tag, w, hinge in (("초판", OLD, False), ("힌지", NEW, True)):
    t0 = time.perf_counter()
    dv, rep, hist = run_k0(tgtS, w, 300, hinge)
    sr = hist[-1].get("stop_reason", "반복 수 소진")
    print(f"[{tag}] 모양 {rep['rms_shape_pct']:5.2f} %  I {rep['I_err_pct']:5.2f} %  "
          f"F {rep['F_err_pct']:5.2f} %  {'집합 안' if rep['in_set'] else '집합 밖'}  "
          f"({time.perf_counter()-t0:.0f} s, {len(hist)} 반복, 종료: {sr})")

print("\n" + "=" * 78)
print("(3) 실현 불가 목표 — 반환되는 '답' 이 씨앗에 따라 흔들리는가")
print("=" * 78)
print("목표: 6-로브 곡선 (K=0 원형으로는 어떤 씨앗으로도 집합에 못 들어간다)")
for tag, w, hinge in (("초판 (모양 오차로 최선 선택)", OLD, False),
                      ("힌지 (집합까지의 거리로 선택)", NEW, True)):
    shapes, reps = [], []
    for s in range(4):
        rng = np.random.default_rng(500 + s)
        dv = DesignVars(r0=mk(float(rng.uniform(9.0, 20.0))), a=mk(np.zeros(1)),
                        b=mk(np.zeros(1)),
                        a_throat=mk(float(rng.uniform(50.0, 140.0))), lengths=[L])
        dv.a.requires_grad_(False); dv.b.requires_grad_(False)
        tol = TOL if hinge else Tolerances(shape=0.0, impulse=0.0, f_max=0.0)
        dv, _ = optimize(M, dv, tgt6, LIM, w, iters=120, verbose=False, tol=tol,
                         throat=ThroatSpec.band(2.0 * (50.0 / 3.141592653589793) ** 0.5,
                         2.0 * (140.0 / 3.141592653589793) ** 0.5),
                         stop_in_set=hinge, patience=10**9)
        rep, _ = final_report(M, dv, tgt6, LIM, w, tol=TOL)
        shapes.append((float(dv.r0), float(dv.a_throat)))
        reps.append(rep)
        print(f"  [{tag[:6]}] seed {s}: r0 {float(dv.r0):6.2f}  A_t {float(dv.a_throat):6.1f}  "
              f"모양 {rep['rms_shape_pct']:6.2f} %  I {rep['I_err_pct']:6.2f} %")
    r0s = np.array([x[0] for x in shapes]); ats = np.array([x[1] for x in shapes])
    print(f"  → [{tag}]  r0 범위 {r0s.min():.2f}~{r0s.max():.2f} "
          f"(폭 {np.ptp(r0s):.2f} mm)   A_t 범위 {ats.min():.1f}~{ats.max():.1f} "
          f"(폭 {np.ptp(ats):.1f} mm²)\n")

print("=" * 78)
print("실현 불가 보고 문구")
print("=" * 78)
_, rep6, _ = run_k0(tgt6, NEW, 150, True)
print(verdict_text(rep6["admissibility"], k_max=0))
