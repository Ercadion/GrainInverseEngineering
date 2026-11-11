"""노즐목을 입력으로 고정하면 식별성 판정이 좋아지는가 — 조밀 전역 격자 측정.

⚠ **이 스크립트는 느리다 (8 사례 × 4 적합에 약 57 분).** 그럴 만한 이유가 있다.

`circle_can_make_curve` 의 탐색(14×14 거친 격자 + Nelder-Mead)이 **전역 최소에
가지 못한다.** 그걸 모르고 쓰면 탐색 실패를 모델 현상으로 읽게 된다. 실제로
그렇게 틀렸다.

    원형 r0 11, A_t 80 (실제 원형이므로 참값을 찾는 것이 정답)
      14×14 + Nelder-Mead : 잔차 6.15 %  → r0  9.78, A_t 109.7   ← 참값을 놓침
      조밀 전역 격자        : 잔차 5.46 %  → r0 10.60, A_t  79.8   ← 회복

    원형 r0 17, L 140, A_t 80 · A_t 고정
      Brent(bounded)      : 잔차 23.42 % → r0 15.46              ← 국소 최소
      조밀 주사            : 잔차  6.73 % → r0 16.60

그래서 여기서는 r0 0.1 mm × A_t 1.7 % 간격 격자를 전부 훑는다. 느린 대신
**결론이 탐색 품질에 의존하지 않는다.**

─────────────────────────────────────────────────────────────────────────────
측정 결과 (2026-10-06, 아래 CASES 8개)

                        원형 바닥 중앙/최대   비원형 최소   분리폭
    전 구간 · 2변수          5.46 / 6.72 %      7.36 %     +0.63 %p
    전 구간 · A_t 고정        5.46 / 6.79 %      8.85 %     +2.06 %p
    창    · 2변수          2.79 / 4.25 %      7.71 %     +3.46 %p
    창    · A_t 고정         3.00 / 4.25 %      9.18 %     +4.93 %p   ← 최선

    원형 r0 복원 오차 [mm]   평균 / 최대
    전 구간 · 2변수          1.180 / 2.500
    전 구간 · A_t 고정        0.360 / 0.400
    창    · 2변수          1.160 / 2.500
    창    · A_t 고정         0.340 / 0.400

읽는 법 세 가지.

1. **A_t 를 고정해도 바닥은 거의 그대로다** (5.46 → 5.46, 2.79 → 3.00).
   바닥은 준정상 모델이 점화·블로다운을 못 담아 생기는 것이지, 자유도가
   많아서 생기는 것이 아니다. 바닥을 내리는 것은 **창**뿐이다.

2. **A_t 를 고정하면 비원형의 잔차가 오른다** (7.36 → 8.85, 7.71 → 9.18).
   비원형이 "원형인 척" 하는 수단을 뺏기 때문이다. 바닥은 그대로이고 비원형만
   올라가므로 분리폭이 넓어진다.

3. **r0 복원은 A_t 고정이 압도적이다** (평균 1.18 → 0.36 mm). 2변수 적합은
   (r0, A_t) 축퇴 골짜기를 타고 전혀 다른 설계로 간다. 예: 실제 원형
   r0 20, A_t 80 을 2변수로 맞추면 r0 17.6, A_t 229.7 (잔차 6.31 %) 이 나온다
   — A_t 고정(6.79 %)보다 0.48 %p 싸게 사면서 r0 가 2.4 mm 틀린다.

즉 노즐목 자유도는 **오적합의 원인이 아니라 축퇴의 통로**다. 원인(모델에 없는
구간)은 창으로 제거하고, 통로는 A_t 고정으로 막는다 — 둘은 하는 일이 다르므로
같이 써야 한다.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from gre.core.analytic import bates_burn_area
from gre.core.burnrate import model_for
from gre.core.nozzle import (area_ratio_to_pressure_ratio, gamma_function,
                             optimum_area_ratio, rt_from_cstar)
from gre.optimize.forward import DiffMotor

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)
torch.set_default_dtype(torch.float64)

R, G, PA = 30.0, 1.133, 0.1013
CS, RT = 885.0 * 0.95, rt_from_cstar(885.0, 1.133)
GF, RHO, CFE = gamma_function(1.133), 1.841 * 0.92 * 1000.0, 0.95
M = DiffMotor(d_outer=2 * R, n_grid=121, n_samples=384, n_w=200, h_max=10e-3)
BR = model_for("KNSB")

R0_GRID = np.linspace(3.0, 27.0, 241)        # 0.1 mm
AT_GRID = np.geomspace(25.0, 400.0, 161)     # 1.7 %

CASES = [("원형", 11.0, 95.0, 80.0, 0, 0.0), ("원형", 14.0, 95.0, 120.0, 0, 0.0),
         ("원형", 17.0, 140.0, 80.0, 0, 0.0), ("원형", 8.0, 55.0, 80.0, 0, 0.0),
         ("원형", 20.0, 95.0, 80.0, 0, 0.0),
         ("3-로브", 11.0, 95.0, 78.0, 3, 2.2), ("4-로브", 8.0, 95.0, 120.0, 4, 5.0),
         ("6-로브", 14.0, 95.0, 90.0, 6, 4.5)]


def quasi(r0, at_mm2, length):
    """`circle_can_make_curve` 내부와 같은 준정상 곡선."""
    if not (0.5 < r0 < R - 0.5) or at_mm2 <= 1.0:
        return None
    w = np.linspace(0.0, R - r0, 300)
    ab = bates_burn_area(w, r0, R, length) * 1e-6
    at = at_mm2 * 1e-6
    lo, hi = np.full_like(w, 1e-3), np.full_like(w, 50.0)
    for _ in range(45):
        mid = 0.5 * (lo + hi)
        g = (RHO - mid * 1e6 / RT) * ab * BR.rate(mid) * 1e-3 - mid * 1e6 * at / CS
        lo, hi = np.where(g > 0, mid, lo), np.where(g > 0, hi, mid)
    p = 0.5 * (lo + hi)
    rm = BR.rate(p)
    t = np.concatenate([[0.0], np.cumsum(np.diff(w)
                                         / np.clip((rm[1:] + rm[:-1]) / 2, 1e-9, None))])
    band = p >= 0.1 * p.max()
    pm = float(np.trapezoid(p[band], t[band]) / max(t[band][-1] - t[band][0], 1e-9))
    eps = optimum_area_ratio(pm, PA, G) if pm > PA else 1.0
    pe = area_ratio_to_pressure_ratio(eps, G)
    cfm = GF * np.sqrt(2 * G / (G - 1) * (1 - pe ** ((G - 1) / G)))
    cf = cfm + (pe - PA / np.clip(p, 1e-9, None)) * eps
    return t, np.clip(cf * CFE * p * 1e6 * at, 0.0, None)


def rms(tt, ft, r0, at, length):
    o = quasi(r0, at, length)
    if o is None:
        return 1e9
    tq, fq = o
    return float(np.sqrt(np.mean(((np.interp(tt, tq, fq, left=0.0, right=0.0) - ft)
                                  / max(abs(ft).max(), 1e-9)) ** 2)))


def dense2(tt, ft, length):
    """(r0, A_t) 둘 다 자유 — 전역 격자."""
    best = (1e9, None, None)
    for r0 in R0_GRID:
        for at in AT_GRID:
            v = rms(tt, ft, r0, at, length)
            if v < best[0]:
                best = (v, r0, at)
    return best[0] * 100.0, best[1], best[2]


def dense1(tt, ft, length, at_fixed):
    """A_t 를 입력으로 고정 — r0 만 자유."""
    v = np.array([rms(tt, ft, r0, at_fixed, length) for r0 in R0_GRID])
    return float(v.min()) * 100.0, float(R0_GRID[int(v.argmin())])


def window(f):
    """손실의 ω(t)(§4.5)와 같은 창 — 점화 끝 이후 · F ≥ 20 % F_max."""
    plateau = float(np.median(f[f > 0.5 * f.max()]))
    i0 = int(np.argmax(f >= 0.90 * plateau))
    m = np.zeros(len(f), dtype=bool)
    m[i0:] = True
    return m & (f >= 0.20 * f.max())


if __name__ == "__main__":
    rows, t0 = [], time.perf_counter()
    print(f"{'형상':<7}{'r0':>4}{'At':>5} ||{'전구간 2변수':>21}|{'전구간 At고정':>15}||"
          f"{'창 2변수':>21}|{'창 At고정':>15}")
    for tag, r0, L, at, nl, amp in CASES:
        a = torch.zeros(6)
        if nl:
            a[nl - 1] = amp
        with torch.no_grad():
            t, p, f, d = M.simulate(torch.tensor(r0), a, torch.zeros(6),
                                    [L], torch.tensor(at))
        t, f = t.numpy(), f.numpy()
        m = window(f)
        a2, r2, t2 = dense2(t, f, L)
        a1, r1 = dense1(t, f, L, at)
        w2, rw2, tw2 = dense2(t[m], f[m], L)
        w1, rw1 = dense1(t[m], f[m], L, at)
        rows.append((tag, nl, r0, at, a2, r2, t2, a1, r1, w2, rw2, tw2, w1, rw1))
        print(f"{tag:<7}{r0:>4.0f}{at:>5.0f} ||{a2:>7.2f}% r0={r2:>5.2f} At={t2:>5.1f}|"
              f"{a1:>7.2f}% r0={r1:>5.2f}||{w2:>7.2f}% r0={rw2:>5.2f} At={tw2:>5.1f}|"
              f"{w1:>7.2f}% r0={rw1:>5.2f}", flush=True)

    cir = [r for r in rows if r[1] == 0]
    non = [r for r in rows if r[1] != 0]
    print(f"\n({time.perf_counter()-t0:.0f} s)\n── 요약 ──")
    for name, i in (("전구간 2변수", 4), ("전구간 At고정", 7),
                    ("창 2변수", 9), ("창 At고정", 12)):
        c = [r[i] for r in cir]
        n = [r[i] for r in non]
        print(f"{name:<14} 원형 바닥 중앙 {np.median(c):5.2f}% 최대 {max(c):5.2f}% | "
              f"비원형 최소 {min(n):5.2f}% | 분리폭 {min(n)-max(c):+6.2f}%p")
    print("\n── 원형 r0 복원 오차 [mm] ──")
    for name, i in (("전구간 2변수", 5), ("전구간 At고정", 8),
                    ("창 2변수", 10), ("창 At고정", 13)):
        e = [abs(r[i] - r[2]) for r in cir]
        print(f"{name:<14} 평균 {np.mean(e):.3f}  최대 {max(e):.3f}")
    np.savez(os.path.join(OUT, "throat_fixed.npz"),
             rows=np.array([r[2:] for r in rows], dtype=float),
             tags=np.array([r[0] for r in rows]))
