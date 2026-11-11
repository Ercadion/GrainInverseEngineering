"""n=321 에서 K=1,2,3 이 모두 **정확히 26 반복**에서 멈춘 이유.

## 내가 처음에 틀린 진단을 내렸다

처음에는 `n_samples=384` 가 격자에 비해 모자라 경사가 망가졌다고 보고했다.
근거는 `n_samples` 를 올릴 때 경사 **값**이 n=321 에서 36 % 바뀐다는 것이었다.

그 뒤 `sample_resolution.py --fd` 로 **같은 모델의 중심차분**과 맞춰 보니
그 진단이 틀렸다. 매끄러운 탐침(모든 사양이 τ 밖)에서

    n_grid=321, n_samples= 384 : 경사 방향 비 0.94~0.95   경사 노름 1.06×
    n_grid=321, n_samples=2048 : 경사 방향 비 1.01~1.05   경사 노름 1.00×

즉 n=321 에서 `n_samples=384` 의 경사는 **방향·크기 모두 멀쩡하다.** (오히려
n=121 쪽이 비 1.71~1.78 로 더 나쁘다 — 그건 표본 수와 무관한 별개 문제다,
같은 파일 4 부.) 처음 본 36 % 는 **해상도에 따라 모델의 답이 달라지는 것**이고,
그건 "경사가 망가졌다" 와 다른 이야기다. 그래서 다시 찾는다.

## 코드를 읽으면 다른 후보가 있다

`optimize` 의 조기 종료는 이렇게 생겼다 (`runner.py`).

    if adm["in_set"]:                 best_in 갱신 (2차 기준)
    elif best_in is None:             key = (worst_gap_pp, 벌점+케이스초과, loss)
                                      best_out 갱신, best_it = it
    ...
    if it - best_it >= patience:      stop = ("손실 정체", it)

**`best_it` 은 손실이 아니라 사전식 키로 갱신된다.** 그 키의 첫 원소는
`worst_gap_pp` — 집합까지의 거리 [%p] 다. 집합 밖에서는 `worst_gap_pp` 가
좋아지지 않는 한 손실이 아무리 내려가도 `best_it` 이 멈추고, 25 반복 뒤
"**손실 정체**" 로 끝난다. 이름이 사실과 다르다.

세 K 가 모두 같은 26 반복에서 멈춘 것이 이 설명에 맞는다 — 26 = best_it(=1)
+ patience(25) + 1 이고, 이건 최적화가 어려운 목표에서 흔한 모양이다.

## 재는 것

n=321 에서 K=1 을 반복마다 기록해 두 가지를 본다.

    1. 멈춘 시점에 **손실이 아직 내려가고 있었는가** (내려가고 있었다면 조기 종료다)
    2. `worst_gap_pp` 가 정말 1 번 반복 이후 한 번도 안 좋아졌는가

⚠ 이 스크립트는 **진단만** 한다. patience 를 바꾸는 것은 판정 기준을 바꾸는
   일이므로 승인 없이 적용하지 않는다.

## 쓰는 법

    PYTHONPATH=. python experiments/stall_cause.py --grid 321 --k 1
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
from gre.optimize.runner import (DesignVars, optimize,            # noqa: E402
                                 seed_design)
from gre.optimize.constraints import ThroatSpec  # noqa: E402

L = 95.0
CASE = ("6-로브 깊음", 14.0, [0, 0, 0, 0, 0, 7.0], 90.0)
LIM, WTS, TOL = ManufacturingLimits(), LossWeights(), Tolerances()


def target_of(motor, r0, arr, at):
    a = torch.tensor(arr, dtype=torch.float64)
    with torch.no_grad():
        t, p, f, _ = motor.simulate(torch.tensor(r0), a, torch.zeros(len(arr)),
                                    [L], torch.tensor(at))
    g = torch.linspace(0.0, float(t[-1]), 320)
    return Target(t=g, f=resample(t, f, g).detach(), p_max=float(p.max()))


def main(n_grid: int, k: int, n_samples: int, iters: int) -> dict:
    tag, r0, arr, at = CASE
    motor = DiffMotor(d_outer=60.0, n_grid=n_grid, n_samples=n_samples,
                      n_w=200, h_max=10e-3)
    target = target_of(motor, r0, arr, at)
    dv = seed_design(12.0, n_modes=k, a_throat=85.0, lengths=(L,),
                     throat=ThroatSpec.fixed(2.0 * (at / 3.141592653589793) ** 0.5),
                     seed_amplitude=0.3, seed=1234)
    print(f"=== {tag} · n_grid={n_grid} · n_samples={n_samples} · K={k} ===")
    print(f"h={float(motor.ds.h):.4f} mm,  예산 {iters} 반복,  patience=25\n",
          flush=True)
    t0 = time.time()
    dv, hist = optimize(motor, dv, target, LIM, WTS, iters=iters, tol=TOL,
                        throat=ThroatSpec.fixed(2.0 * (at / 3.141592653589793) ** 0.5),
                        verbose=False, stop_in_set=False)
    el = time.time() - t0

    stop_reason = hist[-1].get("stop_reason", "반복 수 소진")
    stop_iter = int(hist[-1].get("stop_iter", len(hist) - 1))
    restored = int(hist[-1].get("restored_iter", -1))

    print(f"{'it':>4}{'손실':>12}{'부족[%p]':>10}{'모양%':>8}{'I%':>8}{'F%':>8}"
          f"  최선 갱신")
    print("-" * 62)
    best_gap, best_it = None, 0
    for h in hist:
        gap = float(h.get("worst_gap_pp", float("nan")))
        upd = ""
        if best_gap is None or gap < best_gap:
            best_gap, best_it, upd = gap, int(h["iter"]), "← 갱신"
        print(f"{int(h['iter']):>4}{h['loss']:>12.4e}{gap:>10.2f}"
              f"{h['rms_shape_pct']:>8.2f}{h['I_err_pct']:>8.2f}"
              f"{h['F_err_pct']:>8.2f}  {upd}")

    print(f"\n{'='*62}")
    print(f"종료: {stop_reason} (반복 {stop_iter}),  되돌린 반복 {restored}")
    print(f"`best_it` 마지막 갱신 = 반복 {best_it} (부족 {best_gap:.2f} %p)")
    print(f"→ 26 = {best_it} + 25 + 1 인가: "
          f"{'**맞다 — patience 가 터뜨렸다**' if stop_iter == best_it + 25 else '아니다'}")

    # ── 1. 멈출 때 손실이 아직 내려가고 있었는가
    losses = [float(h["loss"]) for h in hist]
    if len(losses) >= 11:
        tail = losses[-10:]
        slope = (tail[-1] - tail[0]) / max(len(tail) - 1, 1)
        print(f"\n마지막 10 반복 손실: {tail[0]:.4e} → {tail[-1]:.4e} "
              f"(반복당 {slope:+.3e})")
        print(f"→ {'**아직 내려가고 있었다 — 조기 종료다**' if slope < 0 else '평평하거나 올라갔다'}")
    gaps = [float(h.get("worst_gap_pp", float('nan'))) for h in hist]
    print(f"부족[%p] 최소 {min(gaps):.2f} (반복 {gaps.index(min(gaps))}), "
          f"마지막 {gaps[-1]:.2f}")
    print(f"총 {el:.0f} s")
    return {"n_grid": n_grid, "n_samples": n_samples, "k": k,
            "stop_reason": stop_reason, "stop_iter": stop_iter,
            "restored_iter": restored, "best_it": best_it,
            "best_gap_pp": best_gap, "losses": losses, "gaps": gaps,
            "elapsed_s": el}


def warm(n_grid: int, n_samples: int, iters: int, k_max: int = 2) -> dict:
    """K 상승 탐색과 **같은 방식**으로 웜 스타트를 만들어 조기 종료를 본다.

    `minimal_complexity_search` 는 K=0 을 먼저 풀고(모드를 얼린 원형), 그
    해답에서 K=1 을 웜 스타트한다 — 새로 열린 모드만 `N(0, 0.25)` 로 흔든다.

    그 출발점은 `worst_gap_pp` 가 이미 그 지역의 최선에 가깝다. 새 모드가
    0.25 mm 에서 자라는 동안 `worst_gap_pp` 는 거의 안 좋아지는데, `best_it`
    은 그 값으로만 갱신된다. 25 반복이 지나면 patience 가 터진다 — **새 모드가
    일을 시작하기도 전에.** 그러면 높은 K 가 체계적으로 불리해지고, K 상승
    탐색의 결론(필요한 K, '0.2 %p 부족')이 그 편향 위에 서게 된다.

    이 함수가 그 구조를 그대로 재현해 반복별로 기록한다.
    """
    tag, r0, arr, at = CASE
    motor = DiffMotor(d_outer=60.0, n_grid=n_grid, n_samples=n_samples,
                      n_w=200, h_max=10e-3)
    target = target_of(motor, r0, arr, at)
    import numpy as np
    out = {"n_grid": n_grid, "n_samples": n_samples, "segments": []}
    prev = None
    print(f"=== 웜 스타트 재현 · {tag} · n_grid={n_grid} "
          f"· n_samples={n_samples} ===\n", flush=True)
    for k in range(0, k_max + 1):
        n_modes = max(k, 1)
        g = np.random.default_rng(100 + k)
        mk = lambda v: torch.tensor(v, dtype=torch.float64,   # noqa: E731
                                    requires_grad=True)
        if prev is None:
            a0, b0, r0v, atv = np.zeros(n_modes), np.zeros(n_modes), 12.0, 85.0
        else:
            a0, b0 = np.zeros(n_modes), np.zeros(n_modes)
            a0[:len(prev[1])], b0[:len(prev[1])] = prev[1], prev[2]
            r0v, atv = prev[0], prev[3]
            if k > 0:
                a0[k - 1] += g.normal(0, 0.25)
                b0[k - 1] += g.normal(0, 0.25)
        dv = DesignVars(r0=mk(float(r0v)), a=mk(a0), b=mk(b0),
                        a_throat=mk(float(atv)), lengths=[L])
        if k == 0:
            dv.a.requires_grad_(False)
            dv.b.requires_grad_(False)
        dv, hist = optimize(motor, dv, target, LIM, WTS, iters=iters, tol=TOL,
                            throat=ThroatSpec.fixed(2.0 * (at / 3.141592653589793) ** 0.5),
                        throat=ThroatSpec.fixed(2.0 * (at / 3.141592653589793) ** 0.5),
                            verbose=False)
        used = int(hist[-1].get("stop_iter", iters - 1)) + 1
        reason = hist[-1].get("stop_reason", "반복 수 소진")
        losses = [float(h["loss"]) for h in hist]
        gaps = [float(h.get("worst_gap_pp", float("nan"))) for h in hist]
        tail = losses[-min(10, len(losses)):]
        slope = (tail[-1] - tail[0]) / max(len(tail) - 1, 1)
        seg = {"K": k, "iters_used": used, "iters_budget": iters,
               "stop_reason": reason, "loss_first": losses[0],
               "loss_last": losses[-1], "tail_slope": slope,
               "gap_first": gaps[0], "gap_last": gaps[-1],
               "gap_min": min(gaps), "gap_argmin": gaps.index(min(gaps)),
               "amp_opened": float(abs(a0[k - 1])) if k > 0 else 0.0,
               "amp_final": float(abs(dv.a.detach()[k - 1])) if k > 0 else 0.0}
        out["segments"].append(seg)
        print(f"K={k}: {used}/{iters} 반복, {reason}")
        print(f"   손실 {losses[0]:.4e} → {losses[-1]:.4e} "
              f"(마지막 {len(tail)} 반복 기울기 {slope:+.2e})")
        print(f"   부족 {gaps[0]:.2f} → {gaps[-1]:.2f} %p   "
              f"최소 {min(gaps):.2f} (반복 {gaps.index(min(gaps))})")
        if k > 0:
            print(f"   새 모드 진폭 {abs(a0[k-1]):.3f} → "
                  f"{abs(float(dv.a.detach()[k-1])):.3f} mm")
        verdict = ("**조기 종료 — 손실이 아직 내려가는 중이었다**"
                   if reason != "반복 수 소진" and slope < 0 else
                   "예산 소진" if reason == "반복 수 소진" else "정체 맞음")
        print(f"   → {verdict}\n", flush=True)
        prev = (float(dv.r0.detach()), dv.a.detach().numpy(),
                dv.b.detach().numpy(), float(dv.a_throat.detach()))
    print("=" * 62)
    early = [s for s in out["segments"]
             if s["stop_reason"] != "반복 수 소진" and s["tail_slope"] < 0]
    print(f"조기 종료된 K: {[s['K'] for s in early]} / "
          f"{[s['K'] for s in out['segments']]}")
    print("→ 조기 종료가 있으면 K 상승 탐색의 '필요한 K' 와 '부족 %p' 는 "
          "그만큼 **높은 K 에 불리하게** 나온다.")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--warm", action="store_true",
                    help="K 상승 탐색의 웜 스타트를 재현해 조기 종료를 본다")
    ap.add_argument("--k-max", type=int, default=2)
    ap.add_argument("--grid", type=int, default=321)
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--n-samples", type=int, default=384)
    ap.add_argument("--iters", type=int, default=120)
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    res = (warm(a.grid, a.n_samples, a.iters, a.k_max) if a.warm
           else main(a.grid, a.k, a.n_samples, a.iters))
    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(res, fh, ensure_ascii=False, indent=2, default=str)
