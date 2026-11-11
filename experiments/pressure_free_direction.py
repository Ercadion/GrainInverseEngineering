"""압력은 왜 65~103 % 나 틀려도 통과하는가.

`best_iterate_bug.py` 에서 반환 설계의 `P_err_pct` 가 65~103 % 였다. 사양 세 항
(모양·총역적·최대추력)은 전부 10 % 안인데 압력만 2 배 가까이 틀린다.

가설: **추력은 압력과 노즐목의 곱만 묶는다.**

    F = C_f(P, ε) · P_c · A_t

`A_t` 가 자유 설계 변수이므로 같은 추력을 (낮은 P, 큰 A_t) 로도 (높은 P, 작은 A_t)
로도 낼 수 있다. 사양 세 항이 전부 추력에서 나오므로 **P 는 사실상 자유 방향**이고,
붙잡는 것은 가중치 1e-3 의 소프트 2차 기준뿐이다. 즉 설계상 의도된 결과일 수 있다 —
"압력이 사양이 아니다" 는 사실이 숫자로 드러난 것.

재는 방법 (최적화 없이, 정방향만):

    1. 참 설계(6-로브)의 P_max, A_t 를 기준으로 둔다.
    2. `A_t` 를 0.6 ~ 1.6 배로 훑으면서, 각 A_t 에서 **추력 사양 세 항이 10 % 안에
       들어오도록 r0 만** 맞춘다 (1차원 탐색이라 최적화기 없이 격자로 충분).
    3. 통과한 (A_t, P_max) 쌍을 모아, **같은 허용 집합 안에서 P_max 가 몇 배까지
       흩어지는가**를 본다. 흩어짐이 크면 "압력은 자유 방향" 이 확인된다.
    4. P·A_t 의 곱이 보존되는지도 같이 본다 (C_f 가 P 에 약하게 의존하므로 완전히
       일정하지는 않다).
    5. **추가** — 압력 곡선을 입력으로 주면(사양 5개) 같은 훑기에서 몇 개가
       살아남는가. 자유도가 실제로 묶이는지가 그 결정의 근거다.

사용자 결정 (2026-10-08): **A안 + 조건부 B안.** 압력 곡선이 입력으로 들어오면 압력을
네 번째·다섯 번째 사양(각 10 %)으로 올린다. 안 들어오면 지금처럼 소프트로 둔다.
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
from gre.optimize.loss import LossWeights, Target, Tolerances, shape_window

M = DiffMotor(d_outer=60.0, n_grid=121, n_samples=384, n_w=200, h_max=10e-3)
LIM, TOL, L = ManufacturingLimits(), Tolerances(), 95.0
TRUE_R0, TRUE_A6, TRUE_AT = 14.0, 4.5, 90.0


def sim(r0, a6, at):
    a = torch.zeros(6)
    a[5] = a6
    with torch.no_grad():
        t, p, f, d = M.simulate(torch.tensor(float(r0)), a, torch.zeros(6), [L],
                                torch.tensor(float(at)))
    return t, p, f, d


def errors(t, p, f, tgt):
    """사양 세 항과 압력 오차. `curve_loss` 와 같은 식."""
    with torch.no_grad():
        f_on = resample(t, f, tgt.t)
        w = shape_window(t, p, None, None, LossWeights())
        w_on = resample(t, w, tgt.t)
        num = (w_on * (f_on - tgt.f) ** 2).sum()
        den = torch.clamp(w_on.sum(), min=1.0)
        e_shape = float(torch.sqrt(num / den / tgt.f_max ** 2 + 1e-12))
        e_imp = abs(float(torch.trapezoid(f, t)) - tgt.total_impulse) / tgt.total_impulse
        e_f = abs(float(f.max()) - tgt.f_max) / tgt.f_max
        e_p = abs(float(p.max()) - tgt.p_max) / tgt.p_max
    return 100 * e_shape, 100 * e_imp, 100 * e_f, 100 * e_p, float(p.max())


def pshape_err(t, p, f, tgt):
    """압력 곡선 모양 오차 [%] — `curve_loss` 의 `p_shape` 와 같은 식."""
    with torch.no_grad():
        p_on = resample(t, p, tgt.t)
        w = shape_window(t, p, None, None, LossWeights())
        w_on = resample(t, w, tgt.t)
        num = (w_on * (p_on - tgt.p) ** 2).sum()
        den = torch.clamp(w_on.sum(), min=1.0)
        return 100 * float(torch.sqrt(num / den / tgt.p_max ** 2 + 1e-12))


if __name__ == "__main__":
    t, p, f, d = sim(TRUE_R0, TRUE_A6, TRUE_AT)
    g = torch.linspace(0.0, float(t[-1]), 320)
    tgt = Target(t=g, f=resample(t, f, g).detach(),
                 p=resample(t, p, g).detach())      # 압력 곡선까지 목표로 둔다
    print(f"참 설계 (6-로브): r0 {TRUE_R0} mm, A_t {TRUE_AT} mm², "
          f"P_max {tgt.p_max:.3f} MPa, F_max {tgt.f_max:.0f} N, "
          f"I {tgt.total_impulse:.0f} N·s")
    print(f"허용 오차: 모양 {100*TOL.shape:.0f} %, 총역적 {100*TOL.impulse:.0f} %, "
          f"최대추력 {100*TOL.f_max:.0f} %  (+ 압력 곡선 입력 시 압력 모양 "
          f"{100*TOL.p_shape:.0f} %, 최대압 {100*TOL.p_max:.0f} %)")
    print(f"케이스 허용압 (하드 제약): {LIM.p_case_allow:.1f} MPa\n")

    print(f"{'A_t':>7} {'A_t/참':>7} {'최선 r0':>8} {'모양%':>7} {'I%':>7} {'F%':>7}"
          f" {'P_max':>7} {'P오차%':>8} {'P·A_t':>8}  판정")
    ok = []
    t0 = time.perf_counter()
    for scale in np.arange(0.6, 1.65, 0.1):
        at = TRUE_AT * scale
        best = None
        for r0 in np.arange(6.0, 24.01, 0.25):       # 1차원 격자 탐색
            try:
                tt, pp, ff, dd = sim(r0, TRUE_A6, at)
            except Exception:
                continue
            es, ei, ef, ep, pmax = errors(tt, pp, ff, tgt)
            worst = max(es - 100 * TOL.shape, ei - 100 * TOL.impulse,
                        ef - 100 * TOL.f_max, 0.0)
            if best is None or worst < best[0]:
                best = (worst, r0, es, ei, ef, ep, pmax)
        if best is None:
            continue
        worst, r0, es, ei, ef, ep, pmax = best
        passed = worst == 0.0 and pmax <= LIM.p_case_allow
        if passed:
            ok.append((at, r0, pmax, ep))
        print(f"{at:>7.1f} {scale:>7.2f} {r0:>8.2f} {es:>7.2f} {ei:>7.2f} {ef:>7.2f}"
              f" {pmax:>7.3f} {ep:>8.2f} {pmax*at:>8.0f}  "
              + ("통과" if passed
                 else (f"케이스압 초과" if pmax > LIM.p_case_allow
                       else f"초과 {worst:.1f} %p")))
    print(f"\n({time.perf_counter()-t0:.0f} s)")

    if len(ok) >= 2:
        ats = np.array([x[0] for x in ok])
        pms = np.array([x[2] for x in ok])
        eps = np.array([x[3] for x in ok])
        prod = pms * ats
        print(f"\n{'='*78}")
        print(f"**같은 허용 집합 안에서** (추력 사양 세 항 전부 10 % 안, 케이스압 이내)")
        print(f"  노즐목  : {ats.min():.1f} ~ {ats.max():.1f} mm²  "
              f"({ats.max()/ats.min():.2f} 배)")
        print(f"  최대압  : {pms.min():.3f} ~ {pms.max():.3f} MPa "
              f"({pms.max()/pms.min():.2f} 배)  ← 압력이 이만큼 자유롭다")
        print(f"  P 오차  : {eps.min():.1f} ~ {eps.max():.1f} %  "
              f"(참값 {tgt.p_max:.3f} MPa 기준)")
        print(f"  P·A_t   : {prod.min():.0f} ~ {prod.max():.0f}  "
              f"(변동 {100*(prod.max()/prod.min()-1):.1f} % — 거의 보존된다면 "
              f"추력이 곱만 묶는다는 뜻)")
        print("\n→ 압력은 사양이 아니라 **소프트 2차 기준(가중치 1e-3)** 과 "
              "**케이스 허용압 하드 제약** 두 층으로만 잡혀 있다.")
        print("  추력 세 항이 P·A_t 의 곱만 묶으므로, A_t 가 자유 변수인 동안 "
              "P 는 자유 방향이다.")
    else:
        print("\n통과한 A_t 가 2개 미만 — 1차원 격자로는 가를 수 없다. "
              "r0 격자를 촘촘히 하거나 모드를 열어야 한다.")

    # ═══════════════ 압력 곡선을 사양으로 주면
    print(f"\n{'='*78}")
    print("압력 곡선을 **사양**으로 주면 (사양 5개, 각 10 %)")
    print(f"{'='*78}")
    print(f"{'A_t':>7} {'A_t/참':>7} {'최선 r0':>8} {'모양%':>7} {'I%':>7} {'F%':>7}"
          f" {'P모양%':>8} {'P최대%':>8}  판정")
    ok2 = []
    for scale in np.arange(0.6, 1.65, 0.1):
        at = TRUE_AT * scale
        best = None
        for r0 in np.arange(6.0, 24.01, 0.25):
            try:
                tt, pp, ff, dd = sim(r0, TRUE_A6, at)
            except Exception:
                continue
            es, ei, ef, ep, pmax = errors(tt, pp, ff, tgt)
            eps_ = pshape_err(tt, pp, ff, tgt)
            worst = max(es - 100 * TOL.shape, ei - 100 * TOL.impulse,
                        ef - 100 * TOL.f_max, eps_ - 100 * TOL.p_shape,
                        ep - 100 * TOL.p_max, 0.0)
            if best is None or worst < best[0]:
                best = (worst, r0, es, ei, ef, eps_, ep, pmax)
        if best is None:
            continue
        worst, r0, es, ei, ef, eps_, ep, pmax = best
        passed = worst == 0.0 and pmax <= LIM.p_case_allow
        if passed:
            ok2.append((at, r0, pmax))
        print(f"{at:>7.1f} {scale:>7.2f} {r0:>8.2f} {es:>7.2f} {ei:>7.2f} {ef:>7.2f}"
              f" {eps_:>8.2f} {ep:>8.2f}  "
              + ("통과" if passed else f"초과 {worst:.1f} %p"))

    print(f"\n{'='*78}")
    if ok2:
        a2 = np.array([x[0] for x in ok2]); p2 = np.array([x[2] for x in ok2])
        print(f"압력 사양 **켬** : 노즐목 {a2.min():.0f}~{a2.max():.0f} mm² "
              f"({a2.max()/a2.min():.2f} 배)   최대압 {p2.min():.3f}~{p2.max():.3f} MPa "
              f"({p2.max()/max(p2.min(),1e-9):.2f} 배)   통과 {len(ok2)} 개")
    else:
        print("압력 사양 켬 : 통과 0 개 — 이 격자(r0 만, K=0)로는 다섯 사양을 동시에 "
              "만족하지 못한다. 최적화기가 모드를 열어야 할 수 있다.")
    if len(ok) >= 2:
        ats = np.array([x[0] for x in ok]); pms = np.array([x[2] for x in ok])
        print(f"압력 사양 **끔** : 노즐목 {ats.min():.0f}~{ats.max():.0f} mm² "
              f"({ats.max()/ats.min():.2f} 배)   최대압 {pms.min():.3f}~{pms.max():.3f} MPa "
              f"({pms.max()/pms.min():.2f} 배)   통과 {len(ok)} 개")
    print("\n→ 압력 곡선을 주면 P–A_t 축퇴가 묶인다. 그 대가로 허용 집합이 좁아지므로 "
          "\n  실현 불가 보고가 늘어난다 — 그래서 **선택 입력**이다.")
