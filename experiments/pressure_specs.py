"""압력 10 % 가 실제로 설계에 반영되고 있는가.

## 왜 이걸 재는가

합격 조건은 처음부터 **"역설계로 뽑은 그레인 형상이 요구 추력 곡선 및
압력 곡선의 오차 10 % 이내로 들어오는가"** 다. 압력 쪽은 개발이 진행된 뒤에
논의돼서, 코드에 **경로는 있지만 측정된 적이 없다.**

    $ grep -rn "Target(.*p=" experiments/ tests/
    tests/test_optimize.py   8 곳   ← 전부 배선 시험
    experiments/             0 곳   ← 하나도 없다

즉 지금까지의 모든 역설계 측정(K 선택, 복잡도, 허용 집합 진입, 바닥 비교)은
**3 사양**으로만 돌았다. 배선을 시험이 지키는 것은 "압력까지 10 % 안에
들어온다" 의 증거가 아니다.

## 압력이 자유 방향이라는 것을 이미 알고 있다

추력만 사양이면 `F = C_f·P·A_t` 가 **곱만 묶는다.** `A_t` 가 자유인 동안
`P` 는 추력 사양이 보지 않는 방향이고, 그 크기를 쟀다 — 같은 허용
집합 안에서 **최대압이 2.64 배** 벌어지고 `P·A_t` 는 1.5 % 안에서 같았다. 그러므로 3 사양 설계가 압력 10 % 를 지킬 **이유가 없다.**
얼마나 벗어나는지가 질문이다.

## 두 단계로 잰다

    1부. 3 사양으로 역설계한 **그 설계**를 5 사양 보고서로 다시 평가한다.
         → "지금 파이프라인이 내놓는 답이 압력 기준을 지키는가" 에 답한다.
         ⚠ 목표 곡선은 **같은 참 설계**에서 나온 것이므로 압력 목표도 참값이다.
            즉 비교가 공정하다 — 3 사양 설계가 압력을 못 맞히면 그것은 사양에
            없어서 안 맞춘 것이다.

    2부. 압력 곡선을 **사양으로 주고** 역설계를 다시 돌린다 (5 사양).
         → 선택 K, 복잡도, 5 사양 오차, 집합 진입이 어떻게 바뀌는가.

비교 기준은 3 사양 재측정값이다 (같은 격자 n=121, 같은 예산 120):
선택 K=2, 복잡도 2.517 mm, 모양 9.09 / I 9.82 / F 9.68 %
(`experiments/gradient_audit.md` §3).

## 쓰는 법

    PYTHONPATH=. python experiments/pressure_specs.py            # 1부+2부
    PYTHONPATH=. python experiments/pressure_specs.py --part 1    # 1부만
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
from gre.optimize.runner import (admissibility, evaluate,         # noqa: E402
                                 margins, minimal_complexity_search)
from gre.optimize.constraints import ThroatSpec  # noqa: E402

L = 95.0
#: K 재측정과 **같은 목표**. 숫자를 직접 비교하려면 같아야 한다.
CASE = ("6-로브 깊음", 14.0, [0, 0, 0, 0, 0, 7.0], 90.0)
#: 운용 격자. `n_samples` 는 `DiffMotor` 가 알아서 올린다.
CFG = dict(d_outer=60.0, n_grid=121, n_w=200, h_max=10e-3)
ITERS = 120
#: 3 사양 재측정값 (n=121, 예산 120) — 비교 기준.
REF3 = {"K": 2, "complexity": 2.517, "shape": 9.09, "I": 9.82, "F": 9.68}

LIM, WTS, TOL = ManufacturingLimits(), LossWeights(), Tolerances()


def targets(motor):
    """같은 참 설계에서 **3 사양용·5 사양용** 목표를 한 쌍으로 만든다.

    둘의 차이는 `p` 를 넘기는지 **하나뿐**이다 — 곡선 자체는 같은 시뮬에서
    나온 같은 것이다. 그래야 "압력을 사양으로 올린 효과" 만 재진다.
    """
    _, r0, arr, at = CASE
    a = torch.tensor(arr, dtype=torch.float64)
    with torch.no_grad():
        t, p, f, _ = motor.simulate(torch.tensor(r0), a, torch.zeros(len(arr)),
                                    [L], torch.tensor(at))
    g = torch.linspace(0.0, float(t[-1]), 320)
    fr = resample(t, f, g).detach()
    pr = resample(t, p, g).detach()
    t3 = Target(t=g, f=fr, p_max=float(p.max()))
    t5 = Target(t=g, f=fr, p=pr)
    assert not t3.pressure_is_spec and t5.pressure_is_spec
    return t3, t5


def _row(rep, adm):
    return {"in_set": bool(adm["in_set"]),
            "worst_gap_pp": float(adm["worst_gap_pp"]),
            "binding": adm["binding"],
            "rms_shape_pct": float(rep["rms_shape_pct"]),
            "I_err_pct": float(rep["I_err_pct"]),
            "F_err_pct": float(rep["F_err_pct"]),
            "P_shape_pct": float(rep.get("P_shape_pct", float("nan"))),
            "P_err_pct": float(rep["P_err_pct"]),
            "complexity": float(rep["complexity"]),
            "pressure_is_spec": bool(rep.get("pressure_is_spec", False))}


#: 사양 이름 → 보고서의 오차 키. `margins` 는 **여유**(허용 − 오차)를 주므로
#: 오차 자체는 보고서에서 직접 읽는다.
ERR_KEY = {"모양": "rms_shape_pct", "총역적": "I_err_pct",
           "최대추력": "F_err_pct", "압력 모양": "P_shape_pct",
           "최대압": "P_err_pct"}


def show(tag, rep, adm, tol=TOL):
    print(f"\n── {tag}")
    for name, room in margins(rep, tol).items():
        err = float(rep.get(ERR_KEY[name], float("nan")))
        mark = f"여유 {room:5.2f} %p" if room >= 0 else f"**초과 {-room:.2f} %p**"
        print(f"   {name:<8} 오차 {err:6.2f} %  / 허용 10.0 %   {mark}")
    man = adm["manufacturing"]
    bad = [f"{k}: {v[1]}" for k, v in man.items() if not v[0]]
    print(f"   복잡도 {rep['complexity']:.4f} mm   집합 안: {adm['in_set']}"
          + (f"   제작 미충족 → {'; '.join(bad)}" if bad else ""))


def part1(motor, t3, t5) -> dict:
    """3 사양으로 역설계한 설계를 **5 사양 보고서**로 다시 평가한다."""
    print("=" * 78)
    print("1부. 3 사양 역설계 → 그 설계의 압력 오차를 잰다")
    print("=" * 78)
    print("압력은 추력 사양이 보지 않는 방향이다 (F = C_f·P·A_t, A_t 자유).")
    print("실측: 같은 허용 집합 안에서 최대압 2.64 배, P·A_t 는 1.5 % 동일.")
    print("그러므로 3 사양 설계가 압력 10 % 를 지킬 이유가 없다 — 얼마나")
    print("벗어나는지가 질문이다.\n", flush=True)

    t0 = time.time()
    dv, rep0 = minimal_complexity_search(motor, t3, LIM, WTS, k_max=3,
                                         seeding="warm",
                                         throat=ThroatSpec.fixed(2.0 * (CASE[3] / 3.141592653589793) ** 0.5),
                                         iters_per_k=ITERS, tol=TOL,
                                         lengths=(L,), r0_init=12.0,
                                         verbose=True)
    el = time.time() - t0
    if dv is None:
        print("\n⚠ 3 사양에서도 집합에 못 들어갔다 — 재측정과 다르다. "
              "먼저 그 차이를 설명해야 한다.")
        return {"ran": True, "in_set": False, "elapsed_s": el,
                "k_table": (rep0 or {}).get("k_table", [])}

    # ── 같은 설계를 두 보고서로 평가한다. 차이는 목표의 `p` 하나뿐이다.
    _, rep_3, _ = evaluate(motor, dv, t3, LIM, WTS, tol=TOL)
    adm_3 = admissibility(rep_3, LIM, TOL)
    _, rep_5, _ = evaluate(motor, dv, t5, LIM, WTS, tol=TOL)
    adm_5 = admissibility(rep_5, LIM, TOL)

    chosen = (rep0 or {}).get("k_chosen")
    print(f"\n선택 K = {chosen}")
    show("3 사양 판정 (지금 파이프라인이 쓰는 것)", rep_3, adm_3)
    show("같은 설계를 5 사양으로 판정하면", rep_5, adm_5)

    pm = float(rep_5["P_err_pct"])
    ps = float(rep_5.get("P_shape_pct", float("nan")))
    print(f"\n{'─'*78}")
    print(f"**압력 오차: 모양 {ps:.2f} %, 최대압 {pm:.2f} %**  (허용 10 %)")
    if adm_5["in_set"]:
        print("→ 3 사양으로 뽑은 설계가 **우연히** 압력 기준도 지킨다. "
              "그래도 보장은 아니다 — 사양에 없으면 지킬 이유가 없다.")
    else:
        print(f"→ **합격 기준 위반이다.** 지금 파이프라인은 추력 3 항만 보고 "
              f"답을 내놓는데, 그 답이 압력 기준({adm_5['binding']}) 을 "
              f"{adm_5['worst_gap_pp']:.2f} %p 넘긴다.")
    print(f"총 {el:.0f} s ({el/60:.1f} min)")
    return {"ran": True, "in_set": True, "chosen_k": chosen,
            "spec3": _row(rep_3, adm_3), "spec5_same_design": _row(rep_5, adm_5),
            "k_table": (rep0 or {}).get("k_table", []), "elapsed_s": el,
            "design": {"r0": float(dv.r0.detach()),
                       "a": dv.a.detach().tolist(), "b": dv.b.detach().tolist(),
                       "a_throat": float(dv.a_throat.detach())}}


def part2(motor, t5) -> dict:
    """압력을 **사양으로 주고** 역설계를 다시 돌린다 (5 사양)."""
    print("\n" + "=" * 78)
    print("2부. 압력 곡선을 사양으로 주고 역설계 (5 사양)")
    print("=" * 78)
    print("목표 곡선은 1부와 **같은 것**이다 — `p` 를 넘기는지만 다르다.")
    print("압력이 사양이면 `A_t` 의 자유도가 묶이므로, 같은 추력 곡선을 내면서도")
    print("더 어려운 문제가 된다. K 가 올라가거나 집합에 못 들어갈 수 있다.\n",
          flush=True)

    t0 = time.time()
    dv, rep0 = minimal_complexity_search(motor, t5, LIM, WTS, k_max=3,
                                         seeding="warm",
                                         throat=ThroatSpec.fixed(2.0 * (CASE[3] / 3.141592653589793) ** 0.5),
                                         iters_per_k=ITERS, tol=TOL,
                                         lengths=(L,), r0_init=12.0,
                                         verbose=True)
    el = time.time() - t0
    rows = [r for r in (rep0 or {}).get("k_table", []) if not r.get("recheck")]

    if dv is None:
        if not rows:
            print("\n물리 한계 사전 검사에서 걸러졌다 — 최적화가 돌지 않았다.")
            print((rep0 or {}).get("note", ""))
            return {"ran": False, "note": (rep0 or {}).get("note", ""),
                    "elapsed_s": el}
        best = min(rows, key=lambda r: float(r.get("worst_gap_pp", 1e9)))
        print(f"\n⚠ **5 사양으로는 어느 K 도 집합에 못 들어갔다** (K=3 까지).")
        print(f"가장 가까운 K = {int(best['K'])}  부족 {best['worst_gap_pp']:.2f} %p "
              f"({best.get('binding')})  복잡도 {best.get('complexity'):.4f} mm")
        print(f"→ 추력 3 항은 10 % 안에 들어오지만 압력까지 묶으면 이 케이스로는 "
              f"부족하다는 뜻이다. §4.5 의 경계 규칙을 적용해 "
              f"'실현 불가' 로 단정할 거리인지 본다.")
        print(f"총 {el:.0f} s ({el/60:.1f} min)")
        return {"ran": True, "in_set": False, "closest": best,
                "k_table": rows, "elapsed_s": el}

    _, rep, _ = evaluate(motor, dv, t5, LIM, WTS, tol=TOL)
    adm = admissibility(rep, LIM, TOL)
    chosen = (rep0 or {}).get("k_chosen")
    print(f"\n선택 K = {chosen}  (3 사양: {REF3['K']})")
    show("5 사양 판정", rep, adm)
    print(f"\n복잡도 {rep['complexity']:.4f} mm  "
          f"(3 사양 {REF3['complexity']} mm 대비 "
          f"{float(rep['complexity'])/REF3['complexity']:.2f} 배)")
    print(f"로브 충전: {adm['manufacturing'].get('로브 충전')}")
    print(f"총 {el:.0f} s ({el/60:.1f} min)")
    return {"ran": True, "in_set": True, "chosen_k": chosen,
            "spec5": _row(rep, adm), "k_table": rows, "elapsed_s": el,
            "design": {"r0": float(dv.r0.detach()),
                       "a": dv.a.detach().tolist(), "b": dv.b.detach().tolist(),
                       "a_throat": float(dv.a_throat.detach())}}


def main(part: int = 0) -> dict:
    motor = DiffMotor(**CFG)
    print(f"격자 n={motor.n_grid}  h={float(motor.ds.h):.4f} mm  "
          f"n_samples={motor.n_samples}"
          f"{' (자동 상향)' if motor.n_samples_raised else ''}")
    print(f"목표: {CASE[0]}  r0={CASE[1]} a₆={CASE[2][5]} A_t={CASE[3]}  "
          f"예산 {ITERS} 반복\n")
    t3, t5 = targets(motor)
    out = {"n_grid": motor.n_grid, "n_samples": motor.n_samples,
           "iters": ITERS, "case": CASE, "ref3": REF3}
    if part in (0, 1):
        out["part1"] = part1(motor, t3, t5)
    if part in (0, 2):
        out["part2"] = part2(motor, t5)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", type=int, default=0, choices=(0, 1, 2))
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    res = main(a.part)
    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(res, fh, ensure_ascii=False, indent=2, default=str)
