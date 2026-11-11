"""K 근거 복구 — 진폭 바닥 아래로 내려간 판정을 되살린다.

## ⚠ 이 스크립트의 전제가 바뀌었다 (2026-10-17)

머리말 아래 내용은 "격자를 조여 바닥을 내린다" 는 계획으로 쓰였다. **그
계획은 취소됐다** — 바닥이 아니라 **반복 예산**이 문제였다. 같은 격자
(n=121) 같은 목표에 예산만 45 → 120 으로 주면 선택 K 가 3 → **2**, 복잡도가
0.428 → **2.517 mm** (바닥 0.60 mm 의 4.19 배) 가 된다. 자세한 것은
`experiments/gradient_audit.md`.

그래서 이 스크립트는 이제 **`--grid 121` 로 쓰는 것이 기본 용도**다 (기본값은
호환을 위해 241 로 남겨 둔다). n=241/321 결과를 다시 쓸 일은 없다.

## 왜 필요한가

측정에서 **6-로브 깊음**은 `K = 3` 을 골랐고 채택 설계의 복잡도가
**0.428 mm** 였다. 그 숫자로 "스타 형상이 필요하다" 를 세웠다 — 당시 기준은
제작 전사 문턱 δ = 0.020 mm 였고 0.428 은 그것의 21.6 배였다.

두 가지가 바뀌었다.

    ① 진폭 바닥을 **측정**했다 — `0.80 · eps_cells · h`. 운용 격자(n=121)에서
       **0.60 mm** 다. 옛 휴리스틱 0.14 mm 는 바닥을 4.3 배 낮게 봤다.
    ② δ 를 판정에서 **뺐다** — 지배항이 내보내기 공차가 만든 자기충족적 값이다.

그 결과 **0.428 mm 가 바닥(0.60) 아래**가 되어, `amplitude_floor` 가 "K 를 올려
적을 근거가 약하다" 고 경고하는 쪽에 들어간다. 즉 지금 상태로는 K=3 결론을
그 사례로 세울 수 없다.

## 되살릴 수 있다

바닥이 `h` 에 비례하므로 격자를 조이면 내려간다. `0.80 · 1.5 · h < 0.428` 이면

    h < 0.357 mm  →  n ≳ 169        (현 격자 121 의 1.4 배)
    n = 241 이면 h = 0.25, 바닥 **0.300 mm** — 0.428 에 1.43 배 여유

δ 쪽이 n ≳ 3600 (비용 ~900 배) 으로 불가능했던 것과 **대조적으로 현실적이다.**
이 스크립트가 그 재측정을 한다.

## 무엇을 재는가 — 그리고 무엇이 열려 있는가

n=241 에서 K 상승 탐색을 그대로 돌린다. 세 가지를 같이 본다.

    1. 선택된 K 가 여전히 3 인가
    2. 채택 설계의 복잡도가 그 격자의 바닥(0.300 mm) **위**인가
    3. `k0_recheck` 가 여전히 H2(모드가 한 일)를 지지하는가

⚠ **복잡도가 내려갈 수 있다.** 같은 측정에서 거친 격자가 모드의 효과를
   **전 구간에서 깎는다**는 것이 드러났다 (n=121 은 기준 격자 대비 0.42~0.91 만큼
   틀리게 잰다). 모델이 모드 효과를 과소평가하면 최적화기는 같은 추력 변화를 내려고
   진폭을 **더 키운다** — 즉 n=121 의 0.428 mm 는 **과대 추정일 가능성이 높다.**
   격자를 조이면 바닥도 내려가지만 복잡도도 같이 내려간다. 어느 쪽이 더 빨리
   내려가는지는 **예측이 아니라 측정의 문제다.** 그래서 돌린다.

## 목표 곡선을 어느 격자에서 만드는가

**최적화와 같은 격자(n=241)에서 만든다.** 다른 격자에서 만든 곡선을 쓰면 격자
불일치가 잔차에 섞여, 재는 것이 "K 가 필요한가" 가 아니라 "격자가 다른가" 가 된다.
그래서 이 결과는 표와 **직접 비교할 수 없다** — 목표 곡선 자체가 (조금) 다르다.
비교 가능한 것은 **판정**(선택 K, 바닥 대비 복잡도)이지 잔차 숫자가 아니다.

## 쓰는 법

    PYTHONPATH=. python experiments/k3_at_n241.py            # n=241 (기본)
    PYTHONPATH=. python experiments/k3_at_n241.py --grid 169  # 최소 격자
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

torch.set_default_dtype(torch.float64)

from gre.optimize.constraints import ManufacturingLimits          # noqa: E402
from gre.optimize.forward import DiffMotor, resample              # noqa: E402
from gre.optimize.loss import LossWeights, Target, Tolerances     # noqa: E402
from gre.optimize.runner import (admissibility,                   # noqa: E402
                                 amplitude_floor, evaluate,
                                 minimal_complexity_search)
from gre.optimize.constraints import ThroatSpec  # noqa: E402

#: 실험과 **같은 모터 설정** (격자만 바꾼다) — `k_budget_recheck.py`.
#: `h_max=10 ms`, `n_samples=384`, `n_w=200` 은 비용을 낮추려고 고른 값이고
#: 판정에 영향이 없음은 그 실험에서 확인됐다.
CFG = dict(d_outer=60.0, n_samples=384, n_w=200, h_max=10e-3)
LIM, WTS, TOL, L = ManufacturingLimits(), LossWeights(), Tolerances(), 95.0

#: K=3 을 고른 목표. 정의는 그대로 둔다 (설계로 곡선을 만든다).
CASE = ("6-로브 깊음", 14.0, [0, 0, 0, 0, 0, 7.0], 90.0)
ITERS = 120
#: n=121 에서 나온 값 — 비교 기준.
BASE_COMPLEXITY, BASE_K = 0.428, 3


def target_of(motor, r0, arr, at):
    """그 격자에서 그 설계가 내는 추력 곡선을 목표로 만든다."""
    a = torch.tensor(arr, dtype=torch.float64)
    with torch.no_grad():
        t, p, f, _ = motor.simulate(torch.tensor(r0), a, torch.zeros(len(arr)),
                                    [L], torch.tensor(at))
    g = torch.linspace(0.0, float(t[-1]), 320)
    return Target(t=g, f=resample(t, f, g).detach(), p_max=float(p.max()))


def main(n_grid: int = 241, iters: int = ITERS) -> dict:
    tag, r0, arr, at = CASE
    motor = DiffMotor(n_grid=n_grid, **CFG)
    h = float(motor.ds.h)
    floor, _ = amplitude_floor(r0, h=h, eps_cells=float(motor.eps_cells))

    print(f"=== {tag} · n={n_grid} (h={h:.4f} mm) ===")
    print(f"진폭 바닥 {floor:.4f} mm")
    print(f"  (예산 45, n=121): K={BASE_K}  복잡도 {BASE_COMPLEXITY} mm "
          f"→ 바닥 대비 {BASE_COMPLEXITY/floor:.2f} 배 "
          f"({'위' if BASE_COMPLEXITY > floor else '아래'})  ⚠ 예산 artifact")
    print(f"  재측정 (예산 {RECHECK_ITERS}, n=121): K={RECHECK_K}  복잡도 {RECHECK_COMPLEXITY} mm "
          f"→ 바닥 대비 {RECHECK_COMPLEXITY/floor:.2f} 배 "
          f"({'위 — 근거가 선다' if RECHECK_COMPLEXITY > floor else '아래'})")
    print(f"목표 곡선은 **같은 격자**에서 만든다 (격자 불일치를 섞지 않기 위해)\n",
          flush=True)

    target = target_of(motor, r0, arr, at)
    t0 = time.time()
    dv, rep0 = minimal_complexity_search(motor, target, LIM, WTS, k_max=3,
                       seeding="warm",
                                         iters_per_k=iters, tol=TOL,
                                         lengths=(L,), r0_init=12.0,
                                         throat=ThroatSpec.fixed(2.0 * (at / 3.141592653589793) ** 0.5), verbose=True)
    elapsed = time.time() - t0
    k_tab_early = (rep0 or {}).get("k_table", [])
    rows_early = [r for r in k_tab_early if not r.get("recheck")]
    if dv is None and not rows_early:
        print("\n물리 한계 사전 검사에서 걸러졌다 — 최적화가 돌지 않았다.")
        return {"n_grid": n_grid, "infeasible": True, "ran": False,
                "note": (rep0 or {}).get("note", "")}
    if dv is None:
        # ⚠ **최적화는 돌았다.** `dv is None` 은 "어느 K 도 집합에 못 들어갔다" 는
        #    뜻이기도 하다 (초판은 이 경우를 "최적화가 돌지 않았다" 로 적어
        #    오해를 만들었다). 가장 가까운 설계로 판정을 적는다 — 복잡도가
        #    바닥 위인지는 **집합 진입과 별개의 질문**이고, 그게 이 실험의
        #    주된 질문이다.
        best = min(rows_early, key=lambda r: float(r.get("worst_gap_pp", 1e9)))
        cx = float(best.get("complexity", float("nan")))
        print(f"\n{'='*78}")
        print(f"⚠ **어느 K 도 집합에 못 들어갔다** (K=3 까지). 최적화는 돌았다 — "
              f"아래는 가장 가까운 설계다.")
        print(f"가장 가까운 K = {int(best['K'])}   부족 {best.get('worst_gap_pp'):.2f} %p "
              f"({best.get('binding')})")
        print(f"복잡도 {cx:.4f} mm   바닥 {floor:.4f} mm   비 {cx/floor:.1f} 배 → "
              f"{'**바닥 위 — 진폭 근거는 선다**' if cx > floor else '**바닥 아래**'}")
        print(f"n=121 복잡도 {BASE_COMPLEXITY} mm 대비 {cx/BASE_COMPLEXITY:.1f} 배, "
              f"참 설계(a₆ = {arr[5]}) 대비 {cx/arr[5]:.2f} 배")
        print(f"총 {elapsed:.0f} s ({elapsed/60:.1f} min)")
        print(f"\n{rep0.get('note','')}")
        return {"n_grid": n_grid, "h": h, "floor": floor, "ran": True,
                "in_set": False, "closest_k": int(best["K"]),
                "complexity": cx, "ratio_to_floor": cx / floor,
                "ratio_to_base": cx / BASE_COMPLEXITY,
                "ratio_to_truth": cx / arr[5],
                "worst_gap_pp": float(best.get("worst_gap_pp", float("nan"))),
                "binding": best.get("binding"),
                "k_table": rows_early, "elapsed_s": elapsed,
                "note": rep0.get("note", ""),
                "base": {"n_grid": 121, "chosen_k": BASE_K,
                       "complexity": BASE_COMPLEXITY}}

    # 채택 설계를 다시 평가해 복잡도와 집합 판정을 확정한다
    _, rep, _ = evaluate(motor, dv, target, LIM, WTS, tol=TOL)
    adm = admissibility(rep, LIM, TOL)
    k_tab = (rep0 or {}).get("k_table", [])
    chosen = (rep0 or {}).get("k_chosen")
    cx = float(rep["complexity"])

    print(f"\n{'='*78}")
    print(f"선택 K = {chosen}")
    print(f"복잡도 {cx:.4f} mm   바닥 {floor:.4f} mm   "
          f"비 {cx/floor:.2f} 배 → "
          f"{'**바닥 위 — 근거가 선다**' if cx > floor else '**바닥 아래 — 근거가 약하다**'}")
    print(f"집합 안: {adm['in_set']}   모양 {rep['rms_shape_pct']:.2f} % "
          f"I {rep['I_err_pct']:.2f} % F {rep['F_err_pct']:.2f} %")
    print(f"로브 충전: {adm['manufacturing']['로브 충전']}")
    print(f"총 {elapsed:.0f} s ({elapsed/60:.1f} min)")

    if cx <= floor:
        need_h = cx / (0.80 * float(motor.eps_cells))
        need_n = int((motor.ds.r_outer * 2) / max(need_h, 1e-9)) + 1
        print(f"\n⚠ 이 격자로도 바닥 아래다. 복잡도 {cx:.4f} mm 를 넘기려면 "
              f"h < {need_h:.4f} mm → n ≳ {need_n}. "
              f"**복잡도가 격자와 함께 내려간다는 뜻**이고, 그러면 격자를 조이는 "
              f"것으로는 이 결론을 세울 수 없다.")

    return {"n_grid": n_grid, "h": h, "floor": floor, "chosen_k": chosen,
            "complexity": cx, "ratio": cx / floor if floor else None,
            "in_set": bool(adm["in_set"]),
            "rms_shape_pct": float(rep["rms_shape_pct"]),
            "I_err_pct": float(rep["I_err_pct"]),
            "F_err_pct": float(rep["F_err_pct"]),
            "fill_ok": bool(rep.get("fill_ok", True)),
            "lobe_severity": float(rep.get("lobe_severity", 0.0)),
            "k_table": k_tab, "elapsed_s": elapsed,
            "base": {"n_grid": 121, "chosen_k": BASE_K,
                   "complexity": BASE_COMPLEXITY}}


def multistart(n_grid: int = 241, k: int = 3, iters: int = 60,
               seeds: int = 4) -> dict:
    """K 를 고정하고 **독립 씨앗 여러 개**로 집합 진입을 다시 시도한다.

    ⚠ **예산을 늘리는 것은 답이 아니다.** K 상승 탐색에서 K=3 은 예산 120 중
       59 반복만 쓰고 **손실 정체**로 멈췄다 (최대추력 0.2 %p 부족). 정체는
       예산 부족이 아니라 그 국소 최소에서 더 못 내려간다는 뜻이므로,
       `iters_per_k` 를 올려도 같은 자리에 머문다. 필요한 것은 **다른 출발점**이다.

    §4.5 가 이 경우를 다룬다 — "경계에 매우 가깝다(0.24 %p). 반복 예산을
    늘리거나 허용 오차를 재검토하면 들어갈 수 있다 — '실현 불가' 로 단정할
    거리가 아니다." 이 함수가 그 '단정할 거리가 아니다' 를 실제로 확인한다.
    """
    from gre.optimize.runner import DesignVars, optimize, seed_design
    tag, r0, arr, at = CASE
    motor = DiffMotor(n_grid=n_grid, **CFG)
    h = float(motor.ds.h)
    floor, _ = amplitude_floor(r0, h=h, eps_cells=float(motor.eps_cells))
    target = target_of(motor, r0, arr, at)
    print(f"=== {tag} · n={n_grid} · K={k} 고정 · 독립 씨앗 {seeds} 개 "
          f"× {iters} 반복 ===")
    print(f"진폭 바닥 {floor:.4f} mm.  K 상승 탐색의 K=3 은 최대추력 0.2 %p "
          f"부족에서 **손실 정체**로 멈췄다 — 예산이 아니라 출발점 문제다.\n",
          flush=True)

    t0, runs = time.time(), []
    for sd in range(seeds):
        dv = seed_design(12.0, n_modes=k, a_throat=85.0, lengths=(L,),
                         throat=ThroatSpec.fixed(2.0 * (at / 3.141592653589793) ** 0.5),
                         seed_amplitude=0.3, seed=2000 + sd)
        dv, hist = optimize(motor, dv, target, LIM, WTS, iters=iters, tol=TOL,
                            throat=ThroatSpec.fixed(2.0 * (at / 3.141592653589793) ** 0.5),
                            verbose=False, stop_in_set=True)
        _, rep, _ = evaluate(motor, dv, target, LIM, WTS, tol=TOL)
        adm = admissibility(rep, LIM, TOL)
        row = {"seed": 2000 + sd, "in_set": bool(adm["in_set"]),
               "worst_gap_pp": float(adm["worst_gap_pp"]),
               "binding": adm["binding"],
               "rms_shape_pct": float(rep["rms_shape_pct"]),
               "I_err_pct": float(rep["I_err_pct"]),
               "F_err_pct": float(rep["F_err_pct"]),
               "complexity": float(rep["complexity"]),
               "fill_ok": bool(rep.get("fill_ok", True)),
               "lobe_severity": float(rep.get("lobe_severity", 0.0)),
               "iters_used": int(hist[-1].get("stop_iter", iters - 1)) + 1,
               "stop_reason": hist[-1].get("stop_reason", "반복 수 소진")}
        runs.append(row)
        print(f"  씨앗 {row['seed']}: 모양 {row['rms_shape_pct']:6.2f} %  "
              f"I {row['I_err_pct']:6.2f} %  F {row['F_err_pct']:6.2f} %  "
              f"복잡도 {row['complexity']:5.2f}  집합 {row['in_set']}  "
              f"부족 {row['worst_gap_pp']:5.2f} %p  "
              f"[{row['iters_used']}/{iters}, {row['stop_reason']}]", flush=True)

    best = min(runs, key=lambda r: (not r["in_set"], r["worst_gap_pp"]))
    el = time.time() - t0
    print(f"\n{'='*78}")
    print(f"최선 씨앗 {best['seed']}: 집합 {best['in_set']}  "
          f"부족 {best['worst_gap_pp']:.2f} %p  복잡도 {best['complexity']:.4f} mm")
    print(f"복잡도 / 바닥 = {best['complexity']/floor:.1f} 배 → "
          f"{'**바닥 위**' if best['complexity'] > floor else '**바닥 아래**'}")
    print(f"참 설계(a₆ = {arr[5]}) 대비 {best['complexity']/arr[5]:.2f} 배,  "
          f"n=121 복잡도 {BASE_COMPLEXITY} 대비 {best['complexity']/BASE_COMPLEXITY:.1f} 배")
    print(f"로브 충전 통과: {best['fill_ok']} (깊이²/폭 {best['lobe_severity']:.1f})")
    n_in = sum(r["in_set"] for r in runs)
    print(f"집합에 들어간 씨앗 {n_in}/{seeds} → "
          f"{'**K=3 판정이 선다**' if n_in else '**여전히 못 들어간다**'}")
    print(f"총 {el:.0f} s ({el/60:.1f} min)")
    return {"mode": "multistart", "n_grid": n_grid, "k": k, "floor": floor,
            "seeds": seeds, "iters": iters, "runs": runs, "best": best,
            "n_in_set": n_in, "elapsed_s": el,
            "ratio_to_floor": best["complexity"] / floor,
            "ratio_to_truth": best["complexity"] / arr[5]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--multistart", action="store_true",
                    help="K 고정 + 독립 씨앗 (손실 정체를 재시작으로 넘는다)")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--grid", type=int, default=241)
    ap.add_argument("--iters", type=int, default=ITERS)
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    res = (multistart(args.grid, args.k, args.iters, args.seeds)
           if args.multistart else main(args.grid, args.iters))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(res, fh, ensure_ascii=False, indent=2, default=str)
