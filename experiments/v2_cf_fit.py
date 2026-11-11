#!/usr/bin/env python3
"""V2 연소시험 → **노즐 효율 η 와 박리 문턱 `separation_frac`** 2 단 적합.

측정량은 **함의 추력계수**다.

    C_f,meas(t) = F(t) / (P_c(t) · A_t)          [N] / ([MPa]·[mm²]) → 무차원

⚠ 단위가 그대로 맞는다: `P[MPa]·1e6 · A_t[mm²]·1e-6 = P·A_t` [N].

이 양의 좋은 성질 두 가지:

  · **c\\* 도 연소효율도 들어가지 않는다.** F·P·A_t 셋뿐이다. 그래서 추진제
    물성의 불확실도가 이 측정을 오염시키지 않는다.
  · **노즐 효율 η 와 박리 문턱 frac 의 레버리지가 겹치지 않는다.** 노즐이 꽉
    차서 흐르는 구간에서는 `C_f` 가 frac 과 **무관**하고(그 구간이 η 를 정한다),
    박리 구간에서만 frac 이 갈라진다(그 구간이 frac 을 정한다).

그래서 2 단으로 적합한다.

    1 단 (η 보정창)  P_c > frac_max · p_a / (p_e/p_c)  →  η = mean(C_f,meas / C_f,ideal)
    2 단 (판별창)    그 아래 구간             →  frac = argmin ‖C_f,meas − η·C_f(·,frac)‖

사용:

    python experiments/v2_cf_fit.py --selftest              # 합성 데이터 자가검증
    python experiments/v2_cf_fit.py --csv burn.csv --at 61.75 --eps 8.0
    python experiments/v2_cf_fit.py --csv burn.csv --at-before 61.75 --at-after 63.1 ...

CSV 는 `t_s,F_N,P_c_MPa` 열을 가진다 (헤더 필수).
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 2)[0])

from gre.core.nozzle import (area_ratio_to_pressure_ratio,  # noqa: E402
                             thrust_coefficient)

#: 적합이 훑는 박리 문턱 후보.
FRAC_GRID = np.round(np.arange(0.0, 1.001, 0.02), 3)
#: η 보정창을 정할 때 쓰는 "가장 큰 후보" — 이 값의 박리 시작 압력 **위**는
#: 모든 후보가 같은 `C_f` 를 주므로 frac 과 무관하게 η 를 잴 수 있다.
FRAC_MAX_FOR_ETA = 0.5


def cf_model(p_c, p_a, eps, gamma, frac):
    """η 를 빼고 순수 `C_f` — 벡터."""
    return np.array([max(thrust_coefficient(float(x), p_a, eps, gamma,
                                            separation_frac=float(frac)), 0.0)
                     for x in np.atleast_1d(p_c)])


#: 판별창의 **압력 하한** [MPa]. 이 아래는 적합에 넣지 않는다.
#:
#: ⚠ **깊은 테일은 신호가 큰데도 버려야 한다 — 두 가지 이유가 겹친다.**
#:   ① `C_f = F/(P·A_t)` 는 P 로 **나누므로**, P 가 작아지면 추력 잡음이
#:      그대로 증폭된다. 자가검증에서 하한 없이 추력 잡음 5 N RMS 를 주면
#:      적합이 0.40 → **0.48 로 치우쳤다** (흩어진 게 아니라 **편향**이다).
#:   ② 그 구간은 급감압이라 준정상 가정이 깨지고, 모델에 박리 **이력**이
#:      없어서 가장 안 맞는 곳이기도 하다.
P_FIT_FLOOR_MPA: float = 0.5


def fit(t, f, p, a_throat, eps, gamma, p_a,
        frac_grid=FRAC_GRID, p_floor=P_FIT_FLOOR_MPA, verbose=True):
    """2 단 적합. `a_throat` 는 스칼라거나 `t` 와 같은 길이(삭마 보정)."""
    at = np.full_like(p, float(a_throat)) if np.isscalar(a_throat) \
        else np.asarray(a_throat, dtype=float)
    cf_meas = f / np.maximum(p * at, 1e-12)

    pe_pc = area_ratio_to_pressure_ratio(eps, gamma)
    p_sep_eta = FRAC_MAX_FOR_ETA * p_a / pe_pc          # η 보정창 하한
    live = (f > 0) & (p > p_a)

    # ── 1 단: η
    win_eta = live & (p >= p_sep_eta)
    if win_eta.sum() < 20:
        raise RuntimeError(
            f"η 보정창에 표본이 {int(win_eta.sum())} 개뿐이다 "
            f"(P_c > {p_sep_eta:.2f} MPa). 최대압이 낮아 이 방법을 쓸 수 없다 — "
            f"§6 의 '못 가르는 경우' 참조.")
    cf_ideal = cf_model(p[win_eta], p_a, eps, gamma, 0.0)   # 그 구간은 frac 무관
    eta = float(np.mean(cf_meas[win_eta] / np.maximum(cf_ideal, 1e-12)))
    eta_sd = float(np.std(cf_meas[win_eta] / np.maximum(cf_ideal, 1e-12)))

    # ── 2 단: frac
    win_fit = live & (p < p_sep_eta) & (p >= p_floor)
    if win_fit.sum() < 20:
        raise RuntimeError(f"판별창에 표본이 {int(win_fit.sum())} 개뿐이다.")
    sse = []
    for fr in frac_grid:
        pred = eta * cf_model(p[win_fit], p_a, eps, gamma, fr)
        sse.append(float(np.sum((cf_meas[win_fit] - pred) ** 2)))
    sse = np.asarray(sse)
    i = int(np.argmin(sse))
    frac = float(frac_grid[i])
    rms = float(np.sqrt(sse[i] / win_fit.sum()))

    # 신뢰구간 — SSE 가 최소의 (1 + 1/N) 배 안에 드는 frac 범위 (대략 1σ)
    thr = sse[i] * (1.0 + 1.0 / win_fit.sum())
    inside = frac_grid[sse <= thr]
    lo, hi = float(inside.min()), float(inside.max())

    if verbose:
        print(f"  η 보정창  P_c ≥ {p_sep_eta:.2f} MPa : {int(win_eta.sum())} 표본")
        print(f"     → η = {eta:.4f}  (표본 표준편차 {eta_sd:.4f}, "
              f"{100*eta_sd/max(eta,1e-9):.2f} %)")
        print(f"  판별창    {p_floor:.2f} ≤ P_c < {p_sep_eta:.2f} MPa : "
              f"{int(win_fit.sum())} 표본  (하한 아래는 버린다 — P 로 나누므로 "
              f"잡음이 증폭된다)")
        print(f"     → separation_frac = **{frac:.2f}**  "
              f"(1σ 범위 {lo:.2f}~{hi:.2f})   잔차 RMS {rms:.4f}")
        if lo <= frac_grid[0] or hi >= frac_grid[-1]:
            print("     ⚠ 적합이 격자 끝에 붙었다 — 가르지 못한 것이다 "
                  ".")
        if hi - lo > 0.25:
            print(f"     ⚠ 1σ 범위가 {hi-lo:.2f} 로 넓다 — 이 데이터로는 "
                  f"0.3 과 0.4 를 가르지 못한다.")
    return {"eta": eta, "eta_sd": eta_sd, "frac": frac, "frac_lo": lo,
            "frac_hi": hi, "rms": rms, "n_eta": int(win_eta.sum()),
            "n_fit": int(win_fit.sum()), "p_sep_eta": p_sep_eta,
            "cf_meas": cf_meas}


# ────────────────────────────────────────────────────────── 자가검증
def _synth(frac_true, eta_true, eps, gamma, p_a, at, noise_N=0.0,
           shift_ms=0.0, at_drift_pct=0.0, n=2000, seed=0):
    """합성 연소 — 점화 상승 + 지수 감쇠 압력에 모델 추력을 씌운다."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 2.0, n)
    p = 2.9 * np.where(t < 0.06, (t / 0.06) ** 2, np.exp(-(t - 0.06) / 0.62))
    p = np.maximum(p, 1e-6)
    at_t = at * (1.0 + at_drift_pct / 100.0 * np.clip(t / t[-1], 0, 1))
    f = eta_true * cf_model(p, p_a, eps, gamma, frac_true) * p * at_t
    f = np.maximum(f, 0.0)
    if noise_N:
        f = f + rng.normal(0.0, noise_N, size=f.shape)
    if shift_ms:                       # 압력 채널만 뒤로 민다 (정렬 오차 모사)
        k = int(round(shift_ms * 1e-3 / (t[1] - t[0])))
        p = np.roll(p, k)
        if k > 0:
            p[:k] = p[k]
    return t, f, p, at


def selftest():
    eps, gamma, p_a, at = 8.0, 1.133, 0.1013, 61.75
    print("═" * 72)
    print("자가검증 — 참값을 알고 합성한 데이터에서 되찾는가")
    print("═" * 72)

    print("\n① 깨끗한 데이터 (잡음 0, 정렬 0, 삭마 0)")
    ok = True
    for ft in (0.2, 0.3, 0.4, 0.5):
        t, f, p, a = _synth(ft, 0.85, eps, gamma, p_a, at)
        r = fit(t, f, p, a, eps, gamma, p_a, verbose=False)
        good = abs(r["frac"] - ft) <= 0.02 and abs(r["eta"] - 0.85) <= 0.005
        ok &= good
        print(f"   참 frac {ft:.2f} → 적합 {r['frac']:.2f} "
              f"(1σ {r['frac_lo']:.2f}~{r['frac_hi']:.2f})   "
              f"참 η 0.850 → {r['eta']:.4f}   {'OK' if good else '✗'}")
    print(f"   {'전부 복원' if ok else '⚠ 복원 실패'}")

    print("\n② 추력 잡음 — 잡음은 평균으로 눌린다는 주장의 검증")
    for nz in (0.0, 1.0, 2.0, 5.0):
        t, f, p, a = _synth(0.4, 0.85, eps, gamma, p_a, at, noise_N=nz, seed=1)
        r = fit(t, f, p, a, eps, gamma, p_a, verbose=False)
        print(f"   추력 잡음 {nz:4.1f} N RMS → frac {r['frac']:.2f} "
              f"(1σ 폭 {r['frac_hi']-r['frac_lo']:.2f})   η {r['eta']:.4f}")

    print("\n②-b 압력 하한의 효과 — 깊은 테일을 버리면 잡음 편향이 줄어드는가")
    for nz in (2.0, 5.0, 8.0):
        t, f, p, a = _synth(0.4, 0.85, eps, gamma, p_a, at, noise_N=nz, seed=1)
        r_no = fit(t, f, p, a, eps, gamma, p_a, p_floor=0.0, verbose=False)
        r_fl = fit(t, f, p, a, eps, gamma, p_a, verbose=False)
        print(f"   잡음 {nz:4.1f} N : 하한 없음 frac {r_no['frac']:.2f} "
              f"(오차 {r_no['frac']-0.40:+.2f})  →  하한 {P_FIT_FLOOR_MPA} MPa "
              f"frac {r_fl['frac']:.2f} (오차 {r_fl['frac']-0.40:+.2f})")

    print("\n③ 채널 시간 정렬 오차 — 계통오차는 눌리지 않는다")
    for ms in (0, 2, 5, 10, 20):
        t, f, p, a = _synth(0.4, 0.85, eps, gamma, p_a, at, shift_ms=ms)
        r = fit(t, f, p, a, eps, gamma, p_a, verbose=False)
        print(f"   어긋남 {ms:3d} ms → frac {r['frac']:.2f} "
              f"(참 0.40, 오차 {r['frac']-0.40:+.2f})   η {r['eta']:.4f}")

    print("\n④ 노즐목 삭마 — 보정하지 않으면 박리로 **오인**된다")
    for dr in (0.0, 2.0, 5.0, 10.0):
        t, f, p, a = _synth(0.4, 0.85, eps, gamma, p_a, at, at_drift_pct=dr)
        r = fit(t, f, p, a, eps, gamma, p_a, verbose=False)   # 삭마 모르고 적합
        print(f"   A_t 가 연소 중 +{dr:4.1f} % 커졌는데 상수로 보면 → "
              f"frac {r['frac']:.2f} (참 0.40, 오차 {r['frac']-0.40:+.2f})")
    print("   → **연소 전·후 노즐목을 둘 다 재서 보정해야 한다**.")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--selftest", action="store_true", help="합성 데이터 자가검증")
    p.add_argument("--csv", type=str, help="t_s,F_N,P_c_MPa 열을 가진 CSV")
    p.add_argument("--at", type=float, help="노즐목 면적 [mm²] (상수)")
    p.add_argument("--at-before", type=float, help="연소 **전** 노즐목 [mm²]")
    p.add_argument("--at-after", type=float, help="연소 **후** 노즐목 [mm²]")
    p.add_argument("--eps", type=float, default=8.0, help="팽창비")
    p.add_argument("--gamma", type=float, default=1.133)
    p.add_argument("--p-amb", type=float, default=0.1013, help="대기압 [MPa]")
    p.add_argument("--p-floor", type=float, default=P_FIT_FLOOR_MPA,
                   help="판별창 압력 하한 [MPa] — 그 아래는 버린다")
    args = p.parse_args()

    if args.selftest:
        selftest(); return
    if not args.csv:
        p.error("--csv 또는 --selftest 중 하나가 필요하다")

    d = np.genfromtxt(args.csv, delimiter=",", names=True)
    t, f, pc = d["t_s"], d["F_N"], d["P_c_MPa"]

    if args.at_before and args.at_after:
        # 삭마를 시간에 선형으로 본다 — 가장 단순한 보정. 실제 삭마 이력은
        # 모르므로, **보정 유무 두 값을 함께 보고**해 그 차이를 불확실도로 쓴다.
        at = args.at_before + (args.at_after - args.at_before) * (t - t[0]) / (t[-1] - t[0])
        print(f"노즐목 삭마 보정: {args.at_before:.3f} → {args.at_after:.3f} mm² "
              f"({100*(args.at_after/args.at_before-1):+.2f} %, 시간 선형)")
    elif args.at:
        at = args.at
    else:
        p.error("--at 또는 (--at-before 와 --at-after) 가 필요하다")

    print(f"\n── 보정 적용 ──")
    r = fit(t, f, pc, at, args.eps, args.gamma, args.p_amb,
            p_floor=args.p_floor)
    if args.at_before and args.at_after:
        print(f"\n── 보정 없이 (전 노즐목 {args.at_before:.3f} 상수) ──")
        r0 = fit(t, f, pc, args.at_before, args.eps, args.gamma,
                 args.p_amb, p_floor=args.p_floor)
        print(f"\n  두 값의 차이 {abs(r['frac']-r0['frac']):.2f} 가 "
              f"**삭마에서 오는 불확실도**다.")


if __name__ == "__main__":
    main()
