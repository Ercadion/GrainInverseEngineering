#!/usr/bin/env python3
"""V2 연소시험 → **c\\* 를 먼저 떼고, 그 다음 a·n** 순서 적합.

출처: HRE1K CFD 이론서 1.7절의 교훈 "입구 난류를 맞춘 것은 예측이 아니다".
고체판으로 옮기면 — **a·n 을 압력 곡선에 바로 맞추면, c\\* 효율·밀도·노즐목
오차가 전부 a 로 흡수된다**. 그 상태의 a 는
V3 사전 예측에 들어가는 순간 다른 형상에서 틀린다.

`uncertainty_budget.py` 측정 결과가 이 순서를 뒷받침한다 (n=121, 1σ 가정):

    총역적 I 의 분산      : c*_eff 31 % · 공표 c* 28 % · cf_eff 20 %  ← a 는 5 %
    P_max·연소시간의 분산 : a 67~86 % · n 14~17 %

즉 **I 는 c\\*·노즐이, 곡선 모양·압력은 연소속도가** 정한다. 둘의 정보원이 다르므로
떼어서 잴 수 있다.

## 순서

    1 단  c*_meas = Ā_t · ∫P dt / m_p          ← a·n 과 무관 (질량 보존만 씀)
          → c_star_eff = c*_meas / c*_공표
    2 단  η_Cf, separation_frac               ← `v2_cf_fit.py` (F/(P·A_t), c* 무관)
    3 단  1 단 값을 고정한 뒤 정상창 P(t) 로 (a 배율, n 이동) 최소제곱
          + 공분산으로 **n 식별성** 판정

⚠ 1 단의 Ā_t 는 시험 **전후 평균**이다 (삭마). m_p 는 **장전 질량 실측**.
⚠ 1 단은 모델 규약상 정확하다: `DiffMotor` 의 배출 유량은 `P·A_t / (c*·η)` 이고
   질량이 보존되므로 `∫P dt · A_t = m_p · c*·η` 가 모델 안에서도 성립한다.

    PYTHONPATH=. python experiments/v2_cstar_an_fit.py --selftest
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from scipy.optimize import least_squares

torch.set_default_dtype(torch.float64)

from experiments.uncertainty_budget import (AT_MM2, D_CORE, N_SEG,  # noqa: E402
                                                SEG_LEN, build, run)

#: 3 단 적합창 — 최대압의 이 비율 이상인 구간 (점화 상승·테일오프 제외, §6.3).
P_WINDOW_FRAC = 0.6


def stage1_cstar(t, p_mpa, m_p_g, at_before, at_after):
    """c*_meas [m/s] 와 그 불확실도(전후 A_t 차이의 절반을 1σ 로)."""
    ip = float(np.trapezoid(p_mpa, t)) * 1e6                      # Pa·s
    at_mean = 0.5 * (at_before + at_after) * 1e-6                 # m²
    cs = ip * at_mean / (m_p_g * 1e-3)
    rel_sd = 0.5 * abs(at_after - at_before) / (at_before + at_after)
    return cs, cs * rel_sd


def stage3_an(motor, t_meas, p_meas, p_win=P_WINDOW_FRAC, x0=(0.0, 0.0)):
    """(a 배율−1, n 이동) 최소제곱. motor 는 1 단 c* 가 이미 들어간 것."""
    win = p_meas >= p_win * p_meas.max()
    tw, pw = t_meas[win], p_meas[win]

    def resid(x):
        t, p, _ = run(motor, {"a": float(x[0]), "n": float(x[1])})
        ps = np.interp(tw, t.numpy(), p.numpy(), left=0.0, right=0.0)
        return (ps - pw) / pw.max()

    sol = least_squares(resid, x0=np.asarray(x0, float), diff_step=1e-3,
                        x_scale=[0.1, 0.05], xtol=1e-6, ftol=1e-8)
    J, r = sol.jac, sol.fun
    dof = max(len(r) - 2, 1)
    s2 = float(r @ r) / dof
    try:
        cov = np.linalg.inv(J.T @ J) * s2
    except np.linalg.LinAlgError:
        cov = np.full((2, 2), np.inf)
    sd = np.sqrt(np.diag(cov))
    corr = cov[0, 1] / max(sd[0] * sd[1], 1e-300)
    return {"a_scale": 1.0 + sol.x[0], "dn": sol.x[1], "sd_a": sd[0], "sd_n": sd[1],
            "corr": corr, "rms_pct": float(np.sqrt(np.mean(r ** 2)) * 100),
            "p_range": (float(pw.min()), float(pw.max())), "n_pts": int(win.sum()),
            "nfev": sol.nfev}


def _report_an(tag, res, truth=None):
    lo, hi = res["p_range"]
    print(f"  [{tag}] a 배율 {res['a_scale']:.4f} ± {res['sd_a']:.4f}   "
          f"Δn {res['dn']:+.4f} ± {res['sd_n']:.4f}   상관 {res['corr']:+.3f}   "
          f"잔차 {res['rms_pct']:.2f} %   (창 {lo:.2f}~{hi:.2f} MPa, "
          f"{res['n_pts']} 점, 호출 {res['nfev']})")
    if truth:
        print(f"         참값 a 배율 {truth[0]:.4f}   Δn {truth[1]:+.4f}   "
              f"→ 오차 a {100*(res['a_scale']/truth[0]-1):+.2f} %   "
              f"Δn {res['dn']-truth[1]:+.4f}")
    # 식별성: 창 안의 압력비가 좁으면 n 은 a 와 묶여 못 가른다 (§6.4)
    if abs(res["corr"]) > 0.95 or res["sd_n"] > 0.05:
        print(f"         ⚠ n 식별 불가 판정 — |상관| {abs(res['corr']):.2f}, "
              f"σ_n {res['sd_n']:.3f}. 창 압력비 {hi/lo:.2f} 에서 a·n 이 서로 "
              f"보상한다.\n           σ_n 이 작아도 **모형 오차가 없을 때만** 믿을 수 "
              f"있다 — 다른 오차(c* 등)가 있으면 n 이 그것을 흡수한다.")


def selftest(grid_truth=161, grid_fit=121, seed=0):
    """합성 V2: 참값을 알고 만든 '측정' 에서 순서 적합이 참값을 되찾는지.

    역범죄(inverse crime)를 피하려고 측정은 **다른 격자**로 만든다.
    """
    rng = np.random.default_rng(seed)
    # 참값 — Nakka 와 일부러 다르게
    A_TRUE, DN_TRUE, ETA_CS_TRUE = 1.08, 0.03, 0.90
    AT_B, AT_A = AT_MM2, AT_MM2 * 1.02          # 삭마 2 %
    truth_m = build(grid_truth)
    truth_m.c_star_eff = ETA_CS_TRUE
    truth_m.c_star_actual = truth_m.c_star * ETA_CS_TRUE
    # 시험 중 A_t 는 평균값으로 근사 (모델은 상수 A_t)
    import experiments.uncertainty_budget as ub
    at_save = ub.AT_MM2
    ub.AT_MM2 = 0.5 * (AT_B + AT_A)
    try:
        t, p, f = run(truth_m, {"a": A_TRUE - 1.0, "n": DN_TRUE})
    finally:
        ub.AT_MM2 = at_save
    t, p = t.numpy(), p.numpy()
    tm = np.arange(0.0, t[-1], 1e-3)                              # TMS 1 kHz
    pm = np.interp(tm, t, p) * (1 + 0.01 * rng.standard_normal(tm.size))
    m_p = float(truth_m.tables(torch.tensor(D_CORE / 2), torch.zeros(6),
                               torch.zeros(6), [SEG_LEN] * N_SEG)[2][0]) \
        * truth_m.density / 1000.0 * (1 + 0.002 * rng.standard_normal())

    print("=" * 78)
    print(f"자가검증 — 참값 a×{A_TRUE}, Δn {DN_TRUE:+}, c* 효율 {ETA_CS_TRUE}, "
          f"A_t 삭마 2 %, 압력 잡음 1 %")
    print("=" * 78)

    # ── 1 단
    cs, cs_sd = stage1_cstar(tm, pm, m_p, AT_B, AT_A)
    eta_fit = cs / truth_m.c_star
    print(f"1 단  c*_meas {cs:.1f} ± {cs_sd:.1f} m/s  → c_star_eff {eta_fit:.4f} "
          f"(참값 {ETA_CS_TRUE}, 오차 {100*(eta_fit/ETA_CS_TRUE-1):+.2f} %)")

    # ── 3 단: 순서 적합 (1 단 값 고정)
    fit_m = build(grid_fit)
    fit_m.c_star_eff = eta_fit
    fit_m.c_star_actual = fit_m.c_star * eta_fit
    ub.AT_MM2 = 0.5 * (AT_B + AT_A)
    try:
        print("3 단  c* 고정 후 a·n 적합", flush=True)
        r_ord = stage3_an(fit_m, tm, pm)
        _report_an("순서", r_ord, (A_TRUE, DN_TRUE))

        # ── 대조: 1 단을 건너뛰고 기본 c* 효율(0.95) 그대로 a·n 만 맞춘 경우
        naive = build(grid_fit)
        print("대조  1 단 생략 (c* 효율 기본 0.95 그대로)", flush=True)
        r_nv = stage3_an(naive, tm, pm)
        _report_an("생략", r_nv, (A_TRUE, DN_TRUE))
    finally:
        ub.AT_MM2 = at_save

    ok_cs = abs(eta_fit / ETA_CS_TRUE - 1) < 0.01
    ok_a = abs(r_ord["a_scale"] / A_TRUE - 1) < max(0.02, 2 * r_ord["sd_a"])
    worse = abs(r_nv["a_scale"] / A_TRUE - 1) > abs(r_ord["a_scale"] / A_TRUE - 1)
    print("-" * 78)
    print(f"판정  1 단 c* 1 % 안 {ok_cs} · 3 단 a 회복 {ok_a} · "
          f"생략 쪽 a 가 더 빗나감 {worse}")
    return ok_cs and ok_a and worse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    if args.selftest:
        ok = selftest()
        print(f"{time.time()-t0:.0f} s — {'통과' if ok else '실패'}")
        sys.exit(0 if ok else 1)
    ap.print_help()


if __name__ == "__main__":
    main()
