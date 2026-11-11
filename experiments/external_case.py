"""외부 시뮬레이터 대조 + 그 곡선으로 왕복 역설계.

사용자가 **자신이 전에 만든 시뮬레이터**로 정방향 설계한 사례를 줬다. 지금까지
역설계 검증은 전부 **내 정방향 모델이 만든 곡선**으로 했는데, 그건 모델의
버릇까지 같이 맞히는 자기 대조다. 외부 결과와 맞대 보는 것이 처음이다.

## 입력 (사용자 제공)

    챔버      D_c 50 mm, L_c 106 mm, 목표압 3.0 MPa
    노즐      팽창비 Ae/At = 8.0, 노즐 효율 0.85, 삭마 0.0
    연소      연소 효율 0.85
    그레인    KNSB, 외경 50 mm, 코어 20 mm, 세그먼트 30 mm × 3, 간격 4 mm

## 외부 시뮬레이터의 결과 (대조 기준)

    연소시간 1.92 s · 최대추력 226.9 N · 평균추력 140.0 N · 총역적 268.5 N·s
    최대압력 2.895 MPa · Isp 103.9 s · 추진제 질량 0.2514 kg
    설계 Kn 252 (20~252) · 노즐목 61.75 mm² / Ø8.87 mm
    챔버온도 1360 K · 그레인 밀도 1.6937 g/cc · 체적 충전율 71.3 %

## 매핑 — 손으로 검산한 것부터

세 값이 **독립적으로** 맞아떨어져서 기하 매핑이 맞다고 볼 수 있다.

    추진제 체적  3 × π(25² − 10²) × 30            = 148 440 mm³
    질량         148 440 × 1.6937 g/cc            = **251.4 g**   (외부 0.2514 kg)
    챔버 체적    π × 25² × 106                     = 208 133 mm³
    체적 충전율  148 440 / 208 133                 = **71.3 %**    (외부 71.3 %)
    초기 연소면  코어 3 × π·20·30 = 5 655
                 끝면 6 × π(25² − 10²) = 9 896     = **15 551 mm²**
    설계 Kn      15 551 / 61.75                    = **252**       (외부 252)

끝면이 **6 개 다** 타는 것(세그먼트 사이 간격 4 mm)과 `ends_burn=True` 가 같은
가정이라는 뜻이다.

`chamber_margin` 은 "챔버 체적 / 초기 추진제 체적" 이므로 208133/148440 = 1.4021
로 둔다 (기본값 1.25 는 이 사례의 챔버가 아니다).

## 두 단계

    1부  정방향 대조 — 같은 입력으로 내 모델을 돌려 외부 결과와 맞댄다
    2부  왕복 역설계 — 1부가 낸 추력·압력 곡선을 목표로 주고 역설계
         (노즐목은 입력 Ø8.87 mm 고정, 설계 A 는 판정에 들어가지 않는다)

⚠ 1부가 크게 어긋나면 2부는 의미가 없다 — 먼저 어디서 갈라지는지 본다.

## 쓰는 법

    PYTHONPATH=. python experiments/external_case.py            # 1부만
    PYTHONPATH=. python experiments/external_case.py --inverse --seeding independent
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

torch.set_default_dtype(torch.float64)

from gre.optimize.constraints import (ManufacturingLimits,          # noqa: E402
                                      ThroatSpec)
from gre.optimize.forward import DiffMotor, resample                # noqa: E402
from gre.optimize.loss import LossWeights, Target, Tolerances       # noqa: E402
from gre.optimize.runner import (admissibility, evaluate, margins,  # noqa: E402
                                 minimal_complexity_search)

# ── 사용자 입력 그대로
D_GRAIN, D_CORE = 50.0, 20.0
SEG_LEN, N_SEG, SEG_GAP = 30.0, 3, 4.0
D_CHAMBER, L_CHAMBER = 50.0, 106.0
AT_MM2, D_THROAT = 61.75, 8.87
EPS_RATIO = 8.0
ETA_NOZZLE, ETA_COMBUSTION = 0.85, 0.85

#: ⚠ **연소 효율의 규약이 외부 도구와 다르다 — 측정으로 확인했다.**
#:
#:    `DiffMotor` 는 `c_star_eff` 를 `c*` 에 곱한다. 그러면 챔버압이
#:    `P ∝ (Kn·ρ·a·c*)^(1/(1-n))` 를 통해 같은 비율로 내려간다. 외부 도구는
#:    그렇게 하지 않는다 — 같은 입력에서 `c_star_eff` 를 **1.0 으로 두면**
#:    최대압이 2.4739 → 2.9012 MPa 가 되어 외부의 2.895 와 **0.22 %** 로
#:    맞는다 (0.85 를 곱한 쪽은 −14.6 %).
#:
#:    어느 규약이 "옳은가" 는 이 비교로 정해지지 않는다. 여기서 중요한 것은
#:    **외부 결과를 재현하려면 그쪽 규약을 따라야 한다**는 것이고, 그래서
#:    대조에는 `C_STAR_EFF_FOR_MATCH` 를 쓴다. 프로젝트 기본값
#:    (`c_star_eff=0.95`)은 건드리지 않는다.
C_STAR_EFF_FOR_MATCH = 1.0

#: 외부 시뮬레이터의 결과 — 대조 기준
EXTERNAL = {"t_burn_s": 1.92, "F_max_N": 226.9, "F_mean_N": 140.0,
            "I_total_Ns": 268.5, "P_max_MPa": 2.895, "isp_s": 103.9,
            "mass_kg": 0.2514, "kn_design": 252.0, "at_mm2": 61.75,
            "d_throat_mm": 8.87, "rho_g_cc": 1.6937, "loading_pct": 71.3}

# ══════════════════════════════════════════════════════════════════════════
#  외부 도구의 **집계 규약** — 소스에서 직접 확인한 것만 적는다
#  (`6DOF RocketSim with Unreal\rocket_input_web.py` 로 띄운 웹에서 돌린 값이고,
#   집계는 `rocket_input_builder.py`, 연소 모델은 `srm_multiseg.py`/`SRM_simul.py`)
#
#  ⚠⚠ **이걸 몰라서 서로 다른 정의를 비교했다 — 그게 "11.77 % 불일치" 의 정체다.**
#
#    rocket_input_builder.py:1334-1339
#        "burn_time_s":  float(t[-1])                      ← NFPA 5 % 가 **아니다**
#        "avg_thrust_N": float(total_imp / max(t[-1], 1e-9))
#        "total_impulse_Ns": float(np.trapezoid(F, t))
#        "avg_isp_s":    float(np.mean(Isp_arr[Isp_arr > 0]))
#    SRM_simul.py:518  Isp_calculation(F, m_dot_NOZ) = F / (m_dot_NOZ · g0)
#                      → **순간** Isp. 그 비가중 시간평균이 "평균 Isp" 다.
#    srm_multiseg.py:435,553-562
#        추력이 사라진 뒤 DEPRESS_HOLD_S = 0.05 s 더 돌고 break
#        → `t[-1]` ≈ (추력 소멸 시각) + 0.05
#    srm_multiseg.py:212-216  원형 포트 번아웃은 `D_gc >= D_g` (슬리버 분율 없음)
#    srm_multiseg.py:14-18    끝면 유효 η(g)=0 (g≤g_min) → 1 (g≥g_full),
#                             기본 g_min=0.5 g_full=2.0 mm.
#                             이 사례는 간격 4.0 mm ≥ 2.0 → **η = 1 (완전 유효)**,
#                             즉 `DiffMotor(ends_burn=True)` 와 같은 조건이다.
#
#  ⚠ **"외부 수치가 내부적으로 안 맞는다" 는 주장은 철회됐다.** 초판은
#    `I_total ≠ m·g·Isp` (4.82 % 차이) 를 외부 쪽 오류로 읽었다. 틀렸다 —
#    외부의 Isp 는 **순간 Isp 의 비가중 시간평균**이라 질량가중인 `I/(m·g)` 와
#    같을 이유가 없다. 같은 두 정의를 **내 모델에** 적용해도 −3.65 % 가 나온다
#    (외부 −4.60 %): 같은 방향·비슷한 크기이므로 정의 차이지 오류가 아니다.
#    `total_impulse_Ns` 는 `np.trapezoid(F, t)` 라 그 자체가 권위 있는 값이다.
# ══════════════════════════════════════════════════════════════════════════

#: 추력 소멸 후 외부 적분기가 더 도는 시간 [s] — `srm_multiseg.DEPRESS_HOLD_S`.
DEPRESS_HOLD_S = 0.05

#: 보고용 연소시간의 **내 프로젝트 기준**: NFPA 1125 = 최대 추력의 5 %.
#: 대조표는 외부 규약(`t[-1]`)으로 찍고, 이 값은 **참고로 병기**한다 — 둘을 섞어
#: 비교한 것이 바로 위에서 말한 실수다.
NFPA_THRUST_FRAC = 0.05

G0 = 9.80665

LIM, WTS, TOL = ManufacturingLimits(), LossWeights(), Tolerances()
THROAT = ThroatSpec.fixed(D_THROAT)

#: 역설계 출발점 [mm]. 설계 A 의 코어 반경(10.0)과 **다르게** 잡는다 — 정답을
#: 출발점으로 주면 통과가 역설계의 결과인지 초기값인지 구별할 수 없다.
R0_INIT = 6.0


def _chamber_margin() -> float:
    """챔버 체적 / 초기 추진제 체적 — `DiffMotor` 가 쓰는 정의."""
    v_prop = N_SEG * np.pi * ((D_GRAIN / 2) ** 2 - (D_CORE / 2) ** 2) * SEG_LEN
    v_cham = np.pi * (D_CHAMBER / 2) ** 2 * L_CHAMBER
    return float(v_cham / v_prop)


def motor_of(n_grid: int = 121) -> DiffMotor:
    return DiffMotor(d_outer=D_GRAIN, n_grid=n_grid, n_w=400, h_max=2e-3,
                     c_star_eff=C_STAR_EFF_FOR_MATCH, cf_eff=ETA_NOZZLE,
                     chamber_margin=_chamber_margin())


def forward(motor: DiffMotor):
    """사용자 설계 그대로 — 원형 코어 r0 = 10 mm, 세그먼트 3 개."""
    r0 = torch.tensor(D_CORE / 2.0)
    a = b = torch.zeros(6)
    lengths = [SEG_LEN] * N_SEG
    with torch.no_grad():
        t, p, f, d = motor.simulate(r0, a, b, lengths, torch.tensor(AT_MM2),
                                    expansion_ratio=EPS_RATIO)
    return t.numpy(), p.numpy(), f.numpy(), d


def part1(n_grid: int = 121) -> dict:
    m = motor_of(n_grid)
    print("=" * 78)
    print("1부. 정방향 대조 — 같은 입력, 내 모델 vs 외부 시뮬레이터")
    print("=" * 78)
    print(f"격자 n={m.n_grid}  n_samples={m.n_samples}"
          f"{' (자동 상향)' if m.n_samples_raised else ''}  "
          f"chamber_margin={m.chamber_margin:.4f}")
    print(f"c_star_eff {m.c_star_eff} (외부 규약에 맞춤; 입력 연소효율은 "
          f"{ETA_COMBUSTION})  노즐 효율 {m.cf_eff}  팽창비 {EPS_RATIO}\n",
          flush=True)

    t, p, f, d = forward(m)

    # ── 대조표는 **외부 규약 그대로** 집계한다. 위 규약 블록 참조.
    #    외부 적분기는 추력 소멸 후 DEPRESS_HOLD_S 만큼 더 돌고 멈추므로,
    #    내 곡선도 같은 지점에서 잘라야 `t[-1]` 이 같은 뜻이 된다. (내 곡선은
    #    테일 감쇠를 길게 붙여 두므로 자르지 않으면 t[-1] 이 4.5 s 가 된다.)
    nz = np.nonzero(f > 0.0)[0]
    t_ext = float(t[nz[-1]]) if nz.size else float(t[-1])
    cut = t <= t_ext + DEPRESS_HOLD_S
    tc, pc, fc = t[cut], p[cut], f[cut]

    i_tot = float(np.trapezoid(fc, tc))
    t_span = float(tc[-1])                      # 외부: "burn_time_s" = t[-1]
    f_mean = i_tot / max(t_span, 1e-9)          # 외부: total_imp / t[-1]
    mass_kg = float(d["mass_g"]) / 1000.0
    ab0 = float(d["a_b"][0])          # 진단 키는 `a_b` 다 (`ab_tab` 아님)

    # 외부: avg_isp_s = mean(Isp[Isp>0]),  Isp(t) = F / (ṁ_noz · g0) — **순간** Isp.
    # ṁ_noz = P·A_t / (c*·c_star_eff) 로 같은 양을 만든다 (단위: MPa·mm² → kg/s).
    mdot = pc * 1e6 * (AT_MM2 * 1e-6) / max(float(m.c_star) * float(m.c_star_eff),
                                            1e-9)
    isp_inst = np.where(mdot > 1e-12, fc / np.maximum(mdot, 1e-12) / G0, 0.0)
    isp = float(isp_inst[isp_inst > 0].mean()) if (isp_inst > 0).any() else 0.0

    # 참고용 — **내 프로젝트 기준**. 대조에는 쓰지 않는다.
    on = f >= NFPA_THRUST_FRAC * float(f.max())
    t_nfpa = float(t[on][-1] - t[on][0]) if on.any() else float(t[-1])
    isp_massweighted = i_tot / max(mass_kg * G0, 1e-9)

    mine = {"t_burn_s": t_span, "F_max_N": float(fc.max()), "F_mean_N": f_mean,
            "I_total_Ns": i_tot, "P_max_MPa": float(pc.max()), "isp_s": isp,
            "mass_kg": mass_kg, "kn_design": ab0 / AT_MM2,
            "rho_g_cc": m.density, "loading_pct": 100.0 / m.chamber_margin}

    print("집계 규약: **외부 도구와 동일** — 연소시간 = t[-1] (추력 소멸 "
          f"{t_ext:.3f} s + 홀드 {DEPRESS_HOLD_S} s), 평균추력 = I/t[-1],")
    print("           평균 Isp = 순간 Isp 의 비가중 시간평균.  "
          "(서로 다른 정의를 비교한 것이 옛 '11.77 %' 의 정체였다)\n")
    print(f"{'항목':<14}{'외부':>12}{'내 모델':>12}{'차이':>11}{'':>4}")
    print("-" * 56)
    worst = 0.0
    for key, label, unit in (("mass_kg", "추진제 질량", "kg"),
                             ("loading_pct", "체적 충전율", "%"),
                             ("rho_g_cc", "그레인 밀도", "g/cc"),
                             ("kn_design", "설계 Kn", ""),
                             ("P_max_MPa", "최대 압력", "MPa"),
                             ("F_max_N", "최대 추력", "N"),
                             ("F_mean_N", "평균 추력", "N"),
                             ("I_total_Ns", "총역적", "N·s"),
                             ("t_burn_s", "연소 시간", "s"),
                             ("isp_s", "평균 Isp", "s")):
        ext, got = EXTERNAL[key], mine[key]
        rel = 100.0 * (got - ext) / max(abs(ext), 1e-12)
        if key in ("P_max_MPa", "F_max_N", "I_total_Ns", "t_burn_s"):
            worst = max(worst, abs(rel))
        print(f"{label:<14}{ext:>12.4g}{got:>12.4g}{rel:>10.2f}% {unit}")

    print(f"\n곡선 네 항(최대압·최대추력·총역적·연소시간) 최대 차이 "
          f"= **{worst:.2f} %**")
    print("→ " + ("**10 % 안 — 합격 기준과 같은 문턱을 통과**" if worst <= 10
                  else f"**10 % 밖 — 2 부(역설계)를 돌리기 전에 원인을 봐야 한다**"))

    # ── 두 기준을 **나란히** 찍는다. 섞어 쓰다 11.77 % 를 만들었으므로.
    print(f"\n참고 — 같은 곡선을 기준만 바꿔 집계하면:")
    print(f"  연소시간  외부 규약 t[-1] = {t_span:.3f} s   |   "
          f"내 프로젝트 기준 NFPA 1125({100*NFPA_THRUST_FRAC:.0f} %) "
          f"= {t_nfpa:.3f} s")
    print(f"  평균 Isp  외부 규약 mean(Isp>0) = {isp:.2f} s   |   "
          f"질량가중 I/(m·g) = {isp_massweighted:.2f} s")
    print(f"  (외부 수치도 같은 정의 차이를 갖는다: 103.9 vs "
          f"{EXTERNAL['I_total_Ns'] / (EXTERNAL['mass_kg'] * G0):.2f} s — "
          f"오류가 아니라 정의 차이다)")

    return {"mine": mine, "external": EXTERNAL, "worst_pct": worst,
            "t_nfpa_s": t_nfpa, "isp_massweighted_s": isp_massweighted,
            "t_extinction_s": t_ext,
            "curve": {"t": t.tolist(), "p": p.tolist(), "f": f.tolist()},
            "n_grid": n_grid}


def part2(n_grid: int, seeding: str, iters: int, n_seeds: int) -> dict:
    """1부가 낸 곡선을 목표로 주고 역설계 — 노즐목은 입력."""
    m = motor_of(n_grid)
    t, p, f, _ = forward(m)
    print("\n" + "=" * 78)
    print("2부. 왕복 역설계 — 1부의 곡선 + 노즐목 입력만 넘긴다")
    print("=" * 78)
    print(f"노즐목 입력: **{THROAT.describe()}**   씨앗 전략: **{seeding}**")
    print(f"설계 A (코어 Ø{D_CORE} mm, 세그먼트 {N_SEG}×{SEG_LEN} mm) 는 "
          f"곡선 생성용이고 판정에 들어가지 않는다\n", flush=True)

    g = torch.linspace(0.0, float(t[-1]), 320)
    tt = torch.as_tensor(t)
    target = Target(t=g, f=resample(tt, torch.as_tensor(f), g).detach(),
                    p=resample(tt, torch.as_tensor(p), g).detach())
    # ⚠ **출발점에 정답을 주면 안 된다.** 초판은 `r0_init = D_CORE/2 = 10.0`
    #    을 줬고, 그건 설계 A 의 코어 반경 **그 자체**다. 그러면 1 반복만에
    #    통과하는 것이 당연하고, 그 통과는 역설계가 아니라 초기값을 보고한 것이다.
    #    (독립 다중출발의 씨앗이 `uniform(0.75, 1.25)·r0_init` 이라 더 그렇다.)
    #    정답에서 떨어진 곳에서 출발시켜 **실제로 찾게** 한다.
    t0 = time.time()
    dv, rep0 = minimal_complexity_search(
        m, target, LIM, WTS, k_max=3, iters_per_k=iters, tol=TOL,
        lengths=[SEG_LEN] * N_SEG, r0_init=R0_INIT, throat=THROAT,
        seeding=seeding, n_seeds=n_seeds, verbose=True)
    el = time.time() - t0
    rows = [r for r in (rep0 or {}).get("k_table", []) if not r.get("recheck")]

    if dv is None:
        best = min(rows, key=lambda r: float(r.get("worst_gap_pp", 1e9))) if rows else None
        print(f"\n{'='*78}")
        if best is None:
            print("물리 한계 사전 검사에서 걸러졌다.")
            print((rep0 or {}).get("note", ""))
        else:
            print(f"⚠ 어느 K 도 집합에 못 들어갔다. 가장 가까운 K={int(best['K'])}, "
                  f"부족 {best['worst_gap_pp']:.2f} %p ({best.get('binding')})")
        print(f"총 {el:.0f} s")
        return {"in_set": False, "k_table": rows, "elapsed_s": el,
                "seeding": seeding}

    _, rep, _ = evaluate(m, dv, target, LIM, WTS, tol=TOL)
    adm = admissibility(rep, LIM, TOL)
    r0b, atb = float(dv.r0.detach()), float(dv.a_throat.detach())
    print(f"\n{'='*78}")
    print(f"선택 K = {(rep0 or {}).get('k_chosen')}")
    print(f"설계 B: 코어 반경 {r0b:.3f} mm (Ø{2*r0b:.2f}),  "
          f"A_t {atb:.2f} mm² = Ø{2*np.sqrt(atb/np.pi):.3f} mm")
    print(f"        a {[round(x, 3) for x in dv.a.detach().tolist()]}")
    for name, room in margins(rep, TOL).items():
        print(f"   {name:<8} 여유 {room:6.2f} %p "
              f"{'통과' if room >= 0 else '**초과**'}")
    print(f"   집합 안: {adm['in_set']}   복잡도 {rep['complexity']:.4f} mm")
    print(f"총 {el:.0f} s ({el/60:.1f} min)")
    return {"in_set": bool(adm["in_set"]), "seeding": seeding,
            "chosen_k": (rep0 or {}).get("k_chosen"),
            "design_b": {"r0": r0b, "a": dv.a.detach().tolist(),
                         "b": dv.b.detach().tolist(), "a_throat_mm2": atb},
            "margins": {k: float(v) for k, v in margins(rep, TOL).items()},
            "complexity": float(rep["complexity"]),
            "k_table": rows, "elapsed_s": el}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, default=121)
    ap.add_argument("--inverse", action="store_true", help="2부도 돌린다")
    ap.add_argument("--seeding", choices=("warm", "independent"),
                    default="independent")
    ap.add_argument("--iters", type=int, default=120)
    ap.add_argument("--n-seeds", type=int, default=3)
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    out = {"part1": part1(a.grid)}
    if a.inverse:
        out["part2"] = part2(a.grid, a.seeding, a.iters, a.n_seeds)
    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(out, fh, ensure_ascii=False, indent=2, default=str)
