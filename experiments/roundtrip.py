"""왕복 검증 — 정방향 → 역설계 → 정방향 이 같은 곡선을 내는가.

## 판정의 정의

합격 조건은 **곡선끼리의 비교**다. 역설계가 정방향에 넣은 입력값을 되찾는지는
묻지 않는다 — 같은 곡선을 내는 형상이 여러 개 있을 수 있고, 그 중 **제작
가능한 다른 형상**을 찾는 것이 이 도구의 일이다.

    정방향(설계 A)  →  곡선1 (F₁, P₁)
         ↓  곡선1 + 노즐목 입력만 넘긴다  (설계 A 는 안 넘긴다)
    역설계          →  설계 B            (A 와 전혀 달라도 된다)
         ↓
    정방향(설계 B)  →  곡선2 (F₂, P₂)
         ↓
    판정: 곡선1 vs 곡선2 가 10 % 안인가

설계 A 는 **곡선을 만드는 도구**일 뿐이고 판정에 들어가지 않는다. 그래서
"참 설계를 복원했는가" 라는 질문은 이 스크립트에 없다.

## 두 기준을 같이 적는다

`rms_shape_pct` 는 **창**(점화 완료 ~ 웹 소진) 안의 가중 RMS 다. 합격 기준이
"곡선의 오차 10 %" 라면 **전 구간** 비교도 같이 봐야 한다 — 두 숫자가 많이
다르면 "10 % 안에 들어왔다" 가 어느 기준의 이야기인지 분명히 해야 한다.
그래서 창 기준(사양)과 전 구간 기준을 나란히 보고한다.

⚠ 전 구간 비교는 **시간축을 맞춰야** 한다. 설계 B 의 연소시간이 다르면 두
   곡선의 길이가 다르다. 짧은 쪽에 맞춰 자르면 긴 쪽의 꼬리를 공짜로 버리는
   셈이므로, **두 곡선의 합집합 구간**에서 비교하고 바깥은 0 으로 둔다
   (추력이 0 인 구간은 실제로 0 이다).

## 노즐목은 입력이다

`ThroatSpec` 으로 받는다 — 고정(Ø8 mm) 또는 범위(Ø5~10 mm). 측정에서
목이 자유 변수인 동안 최적화기가 목을 0.40 배로 조여 압력을 2.07 배로
올렸다 (`pressure_results.md`). 목이 입력이면 추력을 맞추는 것이 압력을
맞추는 것과 같은 일이 되어 그 자유도가 사라진다.

## 쓰는 법

    PYTHONPATH=. python experiments/roundtrip.py --fixed 10.7
    PYTHONPATH=. python experiments/roundtrip.py --band 5 12
    PYTHONPATH=. python experiments/roundtrip.py --band 5 12 --no-pressure
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

L = 95.0
CFG = dict(d_outer=60.0, n_grid=121, n_w=200, h_max=10e-3)
ITERS = 120
LIM, WTS, TOL = ManufacturingLimits(), LossWeights(), Tolerances()

#: 곡선1 을 만드는 설계 A. **판정에 들어가지 않는다** — 곡선 생성용이다.
DESIGN_A = ("6-로브 깊음", 14.0, [0, 0, 0, 0, 0, 7.0], 90.0)

ERR_KEY = {"모양": "rms_shape_pct", "총역적": "I_err_pct",
           "최대추력": "F_err_pct", "압력 모양": "P_shape_pct",
           "최대압": "P_err_pct"}


def forward(motor, r0, arr, brr, at):
    """정방향 한 번 — `(t, P, F)` 를 numpy 로."""
    with torch.no_grad():
        t, p, f, _ = motor.simulate(
            torch.tensor(float(r0)),
            torch.tensor(arr, dtype=torch.float64),
            torch.tensor(brr, dtype=torch.float64),
            [L], torch.tensor(float(at)))
    return t.numpy(), p.numpy(), f.numpy()


def curve_distance(t1, y1, t2, y2, n: int = 2000) -> dict:
    """두 곡선의 **전 구간** 차이 [%]. 시간축이 달라도 된다.

    합집합 구간 `[0, max(t1[-1], t2[-1])]` 에 공통 격자를 깔고 각 곡선을
    보간한다. 자기 구간 밖은 0 으로 둔다 — 연소가 끝난 뒤 추력·계기압은
    실제로 0 이므로 이 외삽은 물리적으로 맞다. (짧은 쪽에 맞춰 자르면 긴
    쪽의 꼬리가 공짜로 사라져 **차이를 과소평가**한다.)

    정규화는 `y1` 의 최대값으로 한다 — 곡선1 이 기준이다.
    """
    t_end = max(float(t1[-1]), float(t2[-1]))
    g = np.linspace(0.0, t_end, n)
    a = np.interp(g, t1, y1, left=0.0, right=0.0)
    b = np.interp(g, t2, y2, left=0.0, right=0.0)
    scale = max(float(np.max(np.abs(y1))), 1e-12)
    rms = float(np.sqrt(np.mean((a - b) ** 2))) / scale
    mx = float(np.max(np.abs(a - b))) / scale
    i1 = float(np.trapezoid(a, g))
    i2 = float(np.trapezoid(b, g))
    return {"rms_pct": 100.0 * rms, "max_pct": 100.0 * mx,
            "peak_pct": 100.0 * abs(float(np.max(b)) - float(np.max(a)))
            / max(float(np.max(a)), 1e-12),
            "integral_pct": 100.0 * abs(i2 - i1) / max(abs(i1), 1e-12),
            "t_end_1": float(t1[-1]), "t_end_2": float(t2[-1]),
            "t_end_pct": 100.0 * abs(float(t2[-1]) - float(t1[-1]))
            / max(float(t1[-1]), 1e-12)}


def main(throat: ThroatSpec, with_pressure: bool = True,
         iters: int = ITERS, seeding: str = "warm",
         n_seeds: int = 3) -> dict:
    motor = DiffMotor(**CFG)
    tag, r0a, arra, ata = DESIGN_A
    print(f"격자 n={motor.n_grid}  n_samples={motor.n_samples}"
          f"{' (자동 상향)' if motor.n_samples_raised else ''}")
    print(f"노즐목 입력: **{throat.describe()}**")
    print(f"사양: {'5 항 (압력 곡선 포함)' if with_pressure else '3 항 (추력만)'}")
    print(f"씨앗 전략: **{seeding}**"
          + (f" (씨앗 {n_seeds} 개, 예산을 나눠 쓴다)"
             if seeding == "independent" else " (앞 K 의 해를 물려받는다)"))
    print(f"예산 {iters} 반복 / K\n")

    # ── 1단계: 정방향 → 곡선1
    t1, p1, f1 = forward(motor, r0a, arra, [0.0] * len(arra), ata)
    d_a = 2.0 * np.sqrt(ata / np.pi)
    print(f"1단계 정방향 (설계 A = {tag}, r0 {r0a}, a₆ {arra[5]}, "
          f"A_t {ata} mm² = Ø{d_a:.2f} mm)")
    print(f"   곡선1: 연소 {t1[-1]:.3f} s,  F_max {f1.max():.1f} N,  "
          f"P_max {p1.max():.3f} MPa,  I {np.trapezoid(f1, t1):.1f} N·s")
    print("   ⚠ 설계 A 는 여기까지만 쓰인다 — 판정에 들어가지 않는다\n", flush=True)

    # ── 2단계: 곡선1 + 노즐목 입력만 넘겨 역설계
    g = torch.linspace(0.0, float(t1[-1]), 320)
    tt = torch.as_tensor(t1)
    target = Target(t=g, f=resample(tt, torch.as_tensor(f1), g).detach(),
                    **({"p": resample(tt, torch.as_tensor(p1), g).detach()}
                       if with_pressure else {"p_max": float(p1.max())}))
    assert target.pressure_is_spec == with_pressure

    print("2단계 역설계 (곡선1 + 노즐목 입력만 입력)", flush=True)
    t0 = time.time()
    dv, rep0 = minimal_complexity_search(motor, target, LIM, WTS, k_max=3,
                                         iters_per_k=iters, tol=TOL,
                                         lengths=(L,), r0_init=12.0,
                                         throat=throat, seeding=seeding,
                                         n_seeds=n_seeds, verbose=True)
    el = time.time() - t0
    rows = [r for r in (rep0 or {}).get("k_table", []) if not r.get("recheck")]
    early = [r for r in rows if r.get("loss_still_falling") and not r["passed"]]

    if dv is None:
        print(f"\n{'='*78}")
        if not rows:
            print("물리 한계 사전 검사에서 걸러졌다 — 최적화가 돌지 않았다.")
            print((rep0 or {}).get("note", ""))
            return {"ran": False, "throat": throat.describe(),
                    "note": (rep0 or {}).get("note", ""), "elapsed_s": el}
        best = min(rows, key=lambda r: float(r.get("worst_gap_pp", 1e9)))
        print(f"⚠ 어느 K 도 집합에 못 들어갔다. 가장 가까운 K={int(best['K'])}, "
              f"부족 {best['worst_gap_pp']:.2f} %p ({best.get('binding')})")
        if early:
            print(f"⚠ **조기 종료된 K 가 있다**: {[int(r['K']) for r in early]} "
                  f"— 이 '부족' 은 예산 부족의 값이므로 '실현 불가' 의 근거가 "
                  f"**될 수 없다**.")
        print(f"총 {el:.0f} s")
        return {"ran": True, "in_set": False, "throat": throat.describe(),
                "seeding": seeding, "n_seeds": n_seeds,
                "closest": best, "early_stopped": [int(r["K"]) for r in early],
                "k_table": rows, "elapsed_s": el}

    r0b = float(dv.r0.detach())
    arrb = dv.a.detach().tolist()
    brrb = dv.b.detach().tolist()
    atb = float(dv.a_throat.detach())
    d_b = 2.0 * np.sqrt(atb / np.pi)
    print(f"\n   설계 B: r0 {r0b:.3f} mm,  A_t {atb:.2f} mm² = **Ø{d_b:.3f} mm**")
    print(f"           a {[round(x, 3) for x in arrb]}")
    print(f"           b {[round(x, 3) for x in brrb]}")

    # 노즐목 입력이 실제로 지켜졌는지 — 이것부터 확인한다
    lo, hi = throat.bounds_mm2
    kept = lo - 1e-9 <= atb <= hi + 1e-9
    print(f"   노즐목 입력 준수: {kept}  (입력 상자 {lo:.2f}~{hi:.2f} mm²)")
    if throat.is_fixed:
        print(f"   고정 입력과의 차이: {abs(atb - throat.area_mm2):.3e} mm² "
              f"(얼려 있어야 0)")

    # ── 3단계: 설계 B 로 정방향 다시
    t2, p2, f2 = forward(motor, r0b, arrb, brrb, atb)
    print(f"\n3단계 정방향 (설계 B)")
    print(f"   곡선2: 연소 {t2[-1]:.3f} s,  F_max {f2.max():.1f} N,  "
          f"P_max {p2.max():.3f} MPa,  I {np.trapezoid(f2, t2):.1f} N·s")

    # ── 4단계: 판정
    _, rep, _ = evaluate(motor, dv, target, LIM, WTS, tol=TOL)
    adm = admissibility(rep, LIM, TOL)
    dF = curve_distance(t1, f1, t2, f2)
    dP = curve_distance(t1, p1, t2, p2)

    print(f"\n{'='*78}")
    print("4단계 판정 — 곡선1 vs 곡선2")
    print(f"\n[가] 사양 기준 (창 = 점화 완료 ~ 웹 소진, 가중 RMS)")
    worst_spec = 0.0
    for name, room in margins(rep, TOL).items():
        err = float(rep.get(ERR_KEY[name], float("nan")))
        worst_spec = max(worst_spec, err)
        print(f"   {name:<8} {err:7.2f} %  / 허용 10.0 %   "
              f"{'통과' if room >= 0 else f'**초과 {-room:.2f} %p**'}")
    print(f"   집합 안: {adm['in_set']}")

    print(f"\n[나] 전 구간 기준 (합집합 시간축, 곡선1 최대값으로 정규화)")
    print(f"   {'':<10}{'RMS':>9}{'최대편차':>10}{'최대값':>9}{'적분':>9}")
    for nm, d in (("추력", dF), ("압력", dP)):
        print(f"   {nm:<10}{d['rms_pct']:>8.2f}%{d['max_pct']:>9.2f}%"
              f"{d['peak_pct']:>8.2f}%{d['integral_pct']:>8.2f}%")
    print(f"   연소시간  {dF['t_end_1']:.3f} → {dF['t_end_2']:.3f} s "
          f"({dF['t_end_pct']:.2f} %)")

    worst_full = max(dF["rms_pct"], dF["peak_pct"], dF["integral_pct"],
                     *( (dP["rms_pct"], dP["peak_pct"]) if with_pressure else ()))
    ok_spec = bool(adm["in_set"])
    ok_full = worst_full <= 10.0
    print(f"\n{'─'*78}")
    print(f"사양 기준  : {'**통과**' if ok_spec else '부족'}  "
          f"(최악 {worst_spec:.2f} %)")
    print(f"전 구간 기준: {'**통과**' if ok_full else '부족'}  "
          f"(최악 {worst_full:.2f} %)")
    if ok_spec != ok_full:
        print("⚠ **두 기준의 판정이 다르다.** 합격 주장을 할 때 어느 기준인지 "
              "반드시 명시해야 한다 — 창 밖(점화 과도·테일오프)에서 벌어진다.")
    print(f"\n왕복 판정: {'**합격**' if (ok_spec and ok_full) else '**불합격**'}")
    print(f"총 {el:.0f} s ({el/60:.1f} min)")

    return {"ran": True, "in_set": ok_spec, "throat": throat.describe(),
            "seeding": seeding, "n_seeds": n_seeds,
            "throat_kept": kept, "with_pressure": with_pressure,
            "design_b": {"r0": r0b, "a": arrb, "b": brrb,
                         "a_throat_mm2": atb, "d_throat_mm": d_b},
            "chosen_k": (rep0 or {}).get("k_chosen"),
            "spec": {k: float(rep.get(ERR_KEY[k], float("nan")))
                     for k in margins(rep, TOL)},
            "full_thrust": dF, "full_pressure": dP,
            "worst_spec_pct": worst_spec, "worst_full_pct": worst_full,
            "pass_spec": ok_spec, "pass_full": ok_full,
            "pass_roundtrip": bool(ok_spec and ok_full),
            "early_stopped": [int(r["K"]) for r in early],
            "k_table": rows, "elapsed_s": el}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    gx = ap.add_mutually_exclusive_group(required=True)
    gx.add_argument("--fixed", type=float, metavar="D_MM",
                    help="노즐목 직경 고정 [mm]")
    gx.add_argument("--band", type=float, nargs=2, metavar=("D_MIN", "D_MAX"),
                    help="노즐목 직경 범위 [mm]")
    ap.add_argument("--no-pressure", action="store_true",
                    help="압력 곡선을 사양에서 뺀다 (3 사양) — 비교용")
    ap.add_argument("--seeding", choices=("warm", "independent"),
                    required=True,
                    help="씨앗 전략 — 판정을 바꾸므로 기본값을 두지 않는다")
    ap.add_argument("--n-seeds", type=int, default=3)
    ap.add_argument("--iters", type=int, default=ITERS)
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    thr = (ThroatSpec.fixed(a.fixed) if a.fixed
           else ThroatSpec.band(a.band[0], a.band[1]))
    res = main(thr, not a.no_pressure, a.iters, a.seeding, a.n_seeds)
    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(res, fh, ensure_ascii=False, indent=2, default=str)
