"""웜 스타트 K 상승 vs 독립 다중 출발 — **같은 K, 같은 총예산**.

## 왜 다시 재는가

§4.4 는 이렇게 적고 있다.

> 측정: 같은 예산 180 반복에서 웜 스타트 K 상승이 독립 다중 출발보다
> **불리하지 않았다** (6-로브 3.66 % vs 8.73 %, 4-로브 2.89 % vs 3.54 %).
> 저-K 선호는 정당하다.

전면 검토에서 이 숫자의 **출처를 찾지 못했다** — 저장소의 어느 실험
스크립트도 3.66 / 8.73 / 2.89 / 3.54 를 만들지 않는다. 게다가 수정이
burnback 기반 측정값 전부를 무효로 만들었으므로, 그 전 측정이라면 값이 바뀐다.

**결론 자체는 버릴 수 없다.** 그 문장이 받치는 것은 바로 다음 줄의 "저-K 선호는
정당하다" 이고, 그게 없으면 K 상승 탐색의 정당성이 빈다 — "K=3 이 통과한 건
모드가 필요해서가 아니라 웜 스타트라는 유리한 출발점을 받았기 때문" 이라는
반론을 막을 수 없다.

## 공정한 비교의 정의 — 두 번 틀렸고 두 번 고쳤다

**(1) 총예산.** K 상승 탐색은 `K` 마다 `iters_per_k` 를 **새로** 주므로 `K = k` 의
해는 앞선 `K` 들의 예산을 **전부 물려받는다** (`k_table` 의 `iters_cum`).
`iters_per_k = 180` 으로 돌리고 독립 쪽에 180 을 주면 웜 스타트는 실제로
`180 × (K+1)` 을 쓴다 — 최대 4 배 유리한 비교다. 원래 측정에 빠져 있었을
가능성이 가장 큰 부분이고(`iters_per_k` 기본값이 45 였던 것도 같이 작용한다),
이 스크립트는 **쓴 반복 수를 세서 맞춘다.**

**(2) 비교할 `K`.** 초판은 "K 상승이 고른 `K`" 에서 비교했다. 그런데 6-로브는
`K = 0` 을 고른다 (모드가 필요 없는 목표 — 의 거짓 양성 사례). `K = 0` 에서는
모드가 얼어 있으므로 독립 씨앗 3 개가 **전부 같은 출발점**이 되고, 예산을 3 등분
해서 같은 계산을 세 번 한다 — 독립 쪽이 실질 예산의 1/3 만 쓰는 셈이라
**웜 스타트가 부당하게 유리해진다.** 실측에서 그 일이 벌어졌다 (6-로브: 웜 124
반복 vs 독립 3 × 41).

   → **`K` 를 고정해서 비교한다** (`--k-compare`, 기본 2). 묻는 것이 "`K` 가
     몇이어야 하나" 가 아니라 "**같은 `K` 에 도달하는 두 경로 중 웜 스타트가
     불리하지 않은가**" 이므로, `K` 를 자유롭게 두면 두 축이 섞인다.

## 두 팔의 정의

    웜 팔    K=0 (모드 얼림) → K=1 (새 모드만 N(0, 0.25) 섭동, 나머지 물려받음)
             → … → K=k_compare.   각 단계 `iters_per_k` 반복.
             `minimal_complexity_search` 가 내부에서 하는 것과 **같은 사슬**이고,
             통과해도 멈추지 않는 것만 다르다 (목표 `K` 까지 가야 비교가 된다).

    독립 팔  `K = k_compare` 고정, 씨앗 `S` 개. 각 씨앗은 `seed_design` 으로
             모드를 새로 뽑는다 — **낮은 K 의 해를 물려받지 않는다**는 것이
             유일한 차이다. 공칭 출발점(r₀ = 12, A_t = 85)은 웜 팔과 같게 둔다.
             그 둘을 다르게 하면 재는 것이 '웜 스타트의 효과' 가 아니라
             '출발점 운' 이 된다. 예산은 웜 팔이 **실제로 쓴** 총 반복 수를
             `S` 로 나눠 준다.

점수는 **집합까지의 거리**(`worst_gap_pp`)로 비교한다 — 모양 오차로 고르면
'총역적을 가장 많이 포기한 설계' 가 뽑힌다 (`runner` 가 같은 이유로 같은 규칙을
쓴다). 모양 오차도 같이 적는다.

⚠ 두 팔 모두 `stop_in_set=False` 로 돌린다. 집합에 들어가면 멈추는 기본 동작은
  **예산을 덜 쓰고 멈추게** 해서 "같은 예산" 비교를 깨뜨린다.

⚠ **단측 질문이다.** 웜 스타트가 불리하지 않다는 것은 저-K 선호를 **정당화**
   하지만, "K=3 이 옳다" 가 증명되지는 않는다. 그쪽은 `k0_recheck` 가 따로 본다.

## 쓰는 법

    PYTHONPATH=. python experiments/warmstart_vs_multistart.py
    PYTHONPATH=. python experiments/warmstart_vs_multistart.py --k-compare 3
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

from gre.optimize.constraints import ManufacturingLimits          # noqa: E402
from gre.optimize.forward import DiffMotor, resample              # noqa: E402
from gre.optimize.loss import LossWeights, Target, Tolerances     # noqa: E402
from gre.optimize.runner import (DesignVars, admissibility,       # noqa: E402

                                 evaluate, optimize, seed_design)
from gre.optimize.constraints import ThroatSpec  # noqa: E402

CFG = dict(d_outer=60.0, n_samples=384, n_w=200, h_max=10e-3)
LIM, WTS, TOL, L = ManufacturingLimits(), LossWeights(), Tolerances(), 95.0

#: 설계 문서가 인용하는 두 목표.
CASES = [("6-로브", 14.0, [0, 0, 0, 0, 0, 4.5], 90.0),
         ("4-로브", 14.0, [0, 0, 0, 3.0, 0, 0], 90.0)]
SEEDS = 3
R0_INIT, AT_INIT = 12.0, 85.0
PERTURB = 0.25        # 새로 열린 모드의 섭동 — `minimal_complexity_search` 와 같다


def target_of(motor, r0, arr, at):
    a = torch.tensor(arr, dtype=torch.float64)
    with torch.no_grad():
        t, p, f, _ = motor.simulate(torch.tensor(r0), a,
                                    torch.zeros(len(arr)), [L],
                                    torch.tensor(at))
    g = torch.linspace(0.0, float(t[-1]), 320)
    return Target(t=g, f=resample(t, f, g).detach(), p_max=float(p.max()))


def _score(motor, dv, target):
    _, rep, _ = evaluate(motor, dv, target, LIM, WTS, tol=TOL)
    adm = admissibility(rep, LIM, TOL)
    return {"rms_shape_pct": float(rep["rms_shape_pct"]),
            "I_err_pct": float(rep["I_err_pct"]),
            "F_err_pct": float(rep["F_err_pct"]),
            "complexity": float(rep["complexity"]),
            "in_set": bool(adm["in_set"]),
            "worst_gap_pp": float(adm["worst_gap_pp"]),
            "binding": adm["binding"]}


def _mk(v):
    return torch.tensor(v, dtype=torch.float64, requires_grad=True)


def warm_chain(motor, target, k_compare: int, iters_per_k: int, n_modes: int):
    """K=0 → 1 → … → `k_compare` 로 **이어가며** 올린다.

    `minimal_complexity_search` 의 사슬과 같다 — 통과해도 멈추지 않는 것만 다르다.
    """
    t0, used, rows = time.time(), 0, []
    g = np.random.default_rng(0)
    prev = None
    for k in range(k_compare + 1):
        a0, b0 = np.zeros(n_modes), np.zeros(n_modes)
        r0v, atv = R0_INIT, AT_INIT
        if prev is not None:                 # 앞 단계 해를 물려받는다
            a0[:len(prev[1])] = prev[1]
            b0[:len(prev[2])] = prev[2]
            r0v, atv = prev[0], prev[3]
            if k > 0:                        # 새로 열린 모드만 작게 흔든다
                a0[k - 1] += g.normal(0, PERTURB)
                b0[k - 1] += g.normal(0, PERTURB)
        dv = DesignVars(r0=_mk(float(r0v)), a=_mk(a0), b=_mk(b0),
                        a_throat=_mk(float(atv)), lengths=[L])
        if k == 0:                           # 원형 — 모드를 얼린다
            dv.a.requires_grad_(False)
            dv.b.requires_grad_(False)
        dv, hist = optimize(motor, dv, target, LIM, WTS, iters=iters_per_k,
                            throat=ThroatSpec.fixed(2.0 * (AT_INIT / 3.141592653589793) ** 0.5),
                            verbose=False, tol=TOL, stop_in_set=False)
        used += int(hist[-1].get("stop_iter", iters_per_k - 1)) + 1
        sc = _score(motor, dv, target)
        sc.update(K=k, iters_cum=used,
                  stop_reason=hist[-1].get("stop_reason", "반복 수 소진"))
        rows.append(sc)
        with torch.no_grad():
            prev = (float(dv.r0), dv.a.detach().numpy().copy(),
                    dv.b.detach().numpy().copy(), float(dv.a_throat))
    out = dict(rows[-1])
    out.update(iters_total=used, elapsed_s=time.time() - t0, chain=rows)
    return out


def multi_start(motor, target, k: int, iters_total: int, n_modes: int,
                seeds: int = SEEDS):
    """`K = k` 고정, 독립 씨앗 `seeds` 개. 예산 `iters_total` 을 나눠 쓴다."""
    t0 = time.time()
    per = max(1, iters_total // seeds)
    runs = []
    for s in range(seeds):
        dv = seed_design(R0_INIT, n_modes=n_modes, a_throat=AT_INIT,
                         throat=ThroatSpec.fixed(2.0 * (AT_INIT / 3.141592653589793) ** 0.5),
                         lengths=(L,), seed_amplitude=0.3, seed=1000 + s)
        with torch.no_grad():                # k 보다 높은 모드는 쓰지 않는다
            dv.a[k:] = 0.0
            dv.b[k:] = 0.0
        optimize(motor, dv, target, LIM, WTS, iters=per, tol=TOL,
                 throat=ThroatSpec.fixed(2.0 * (AT_INIT / 3.141592653589793) ** 0.5),
                 verbose=False, stop_in_set=False)
        sc = _score(motor, dv, target)
        sc["seed"] = 1000 + s
        runs.append(sc)
    best = min(runs, key=lambda r: (not r["in_set"], r["worst_gap_pp"],
                                    r["rms_shape_pct"]))
    return {"best": best, "runs": runs, "iters_total": per * seeds,
            "iters_per_seed": per, "seeds": seeds,
            "elapsed_s": time.time() - t0}


def _ckpt(path: str, tag: str, arm: str, payload: dict) -> None:
    """팔 하나가 끝날 때마다 한 줄 적는다.

    ⚠ 이게 없으면 중간에 끊길 때 **그 목표의 모든 계산을 잃는다.** 실제로
      한 번 잃었다 — 일시정지 중 컨테이너가 프로세스를 회수했고,
      목표 단위 저장이라 끝나지 않은 목표가 통째로 날아갔다.
    """
    if not path:
        return
    with open(path + "l", "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"tag": tag, "arm": arm, "t": time.time(),
                             "payload": payload}, ensure_ascii=False,
                            default=str) + "\n")


def main(n_grid: int = 121, iters_per_k: int = 60, k_compare: int = 2,
         out_json: str = "") -> dict:
    motor = DiffMotor(n_grid=n_grid, **CFG)
    n_modes = max(k_compare, 1)
    out = {}
    print(f"격자 n={n_grid} (h={float(motor.ds.h):.4f} mm), "
          f"iters_per_k={iters_per_k}, **비교 K={k_compare} 고정**, "
          f"씨앗 {SEEDS} 개\n", flush=True)

    for tag, r0, arr, at in CASES:
        target = target_of(motor, r0, arr, at)
        print(f"── {tag} ──────────────────────────────────", flush=True)
        w = warm_chain(motor, target, k_compare, iters_per_k, n_modes)
        _ckpt(out_json, tag, "warm", w)        # 팔 단위 체크포인트
        for r in w["chain"]:
            print(f"    사슬 K={r['K']}: 모양 {r['rms_shape_pct']:6.2f} %  "
                  f"부족 {r['worst_gap_pp']:5.2f} %p  복잡도 {r['complexity']:5.2f}  "
                  f"누적 {r['iters_cum']}  ({r['stop_reason']})", flush=True)
        print(f"  웜 스타트   : 모양 {w['rms_shape_pct']:6.2f} %  집합 {w['in_set']}  "
              f"부족 {w['worst_gap_pp']:5.2f} %p  복잡도 {w['complexity']:5.2f}  "
              f"**반복 {w['iters_total']}**  ({w['elapsed_s']/60:.1f} min)", flush=True)

        m = multi_start(motor, target, k_compare, w["iters_total"], n_modes)
        _ckpt(out_json, tag, "multi", m)
        b = m["best"]
        for r in m["runs"]:
            print(f"    씨앗 {r['seed']}: 모양 {r['rms_shape_pct']:6.2f} %  "
                  f"부족 {r['worst_gap_pp']:5.2f} %p  복잡도 {r['complexity']:5.2f}",
                  flush=True)
        print(f"  독립 다중출발: 모양 {b['rms_shape_pct']:6.2f} %  집합 {b['in_set']}  "
              f"부족 {b['worst_gap_pp']:5.2f} %p  복잡도 {b['complexity']:5.2f}  "
              f"**반복 {m['iters_total']}** ({m['seeds']}×{m['iters_per_seed']})  "
              f"({m['elapsed_s']/60:.1f} min)", flush=True)

        warm_better = (w["in_set"], -w["worst_gap_pp"]) >= (b["in_set"],
                                                            -b["worst_gap_pp"])
        print(f"  → 웜 스타트가 {'불리하지 않다' if warm_better else '**불리하다**'}"
              f"  (부족 {w['worst_gap_pp']:.2f} vs {b['worst_gap_pp']:.2f} %p)",
              flush=True)
        out[tag] = {"warm": w, "multi": m, "warm_not_worse": bool(warm_better),
                    "budget_matched": w["iters_total"] == m["iters_total"],
                    "budget_warm": w["iters_total"],
                    "budget_multi": m["iters_total"]}
        if out_json:                          # 목표마다 중간 저장
            with open(out_json, "w", encoding="utf-8") as fh:
                json.dump(out, fh, ensure_ascii=False, indent=2, default=str)

    print(f"\n{'='*78}")
    print(f"판정 — §4.4 의 '웜 스타트가 불리하지 않았다' (K={k_compare} 고정)")
    for tag, v in out.items():
        w, b = v["warm"], v["multi"]["best"]
        print(f"  {tag:8s} 웜 부족 {w['worst_gap_pp']:5.2f} %p (모양 "
              f"{w['rms_shape_pct']:5.2f} %) vs 독립 부족 {b['worst_gap_pp']:5.2f} %p "
              f"(모양 {b['rms_shape_pct']:5.2f} %)   예산 {v['budget_warm']} = "
              f"{v['budget_multi']} {'✔' if v['budget_matched'] else '✗'}")
    allw = all(v["warm_not_worse"] for v in out.values())
    print(f"\n{'**불리하지 않다 — 저-K 선호가 정당하다**' if allw else '**불리하다 — 저-K 선호의 통계적 근거가 없다**'}")
    print("⚠ 단측 질문이다. 불리하지 않다는 것이 'K 가 옳다' 를 증명하지는 않는다.")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, default=121)
    ap.add_argument("--iters-per-k", type=int, default=60)
    ap.add_argument("--k-compare", type=int, default=2)
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    main(args.grid, args.iters_per_k, args.k_compare, args.json)
