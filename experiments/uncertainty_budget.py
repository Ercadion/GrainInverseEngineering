#!/usr/bin/env python3
"""불확실성 예산 — 물성·운용 파라미터가 곡선 지표를 얼마나 흔드는가.

출처: HRE1K CFD 이론서 26.6절 "불확실도 예산과 민감도 우선순위" 를 고체 0D 모델에
옮긴 것. 원문의 원칙은 "정확한 답이 아니라 **얼마나 틀렸는지 아는 답**" 이고,
그 첫 단계가 입력별 민감도 순위다.

## 무엇을 하나

기준 설계
에서 아래 파라미터를 하나씩 ±1σ 흔들어 지표 변화를 잰다.

    지표: 총역적 I, 최대압 P_max, 최대추력 F_max, 연소시간 t_b(NFPA 5 %),
          곡선 모양 RMS(기준 곡선 대비, F_max 정규화)

두 경로로 잰다.

  · **중앙차분 (모든 파라미터)** — 판정에 쓰는 값.
  · **자동미분 (텐서로 바꿀 수 있는 5 개)** — 중앙차분과 맞는지 대조.
    `DiffMotor` 의 float 필드를 실험 안에서만 텐서로 덮어쓴다. 소스는 건드리지 않는다.

## 1σ 는 가정이다

σ 값은 측정 전 "그럴듯한 폭" 이다(아래 `PARAMS`). 결론은 **순위**와 **σ 를 얼마나
줄여야 하나**이지 절대 오차가 아니다. σ 를 바꾸려면 표만 고치면 된다 — 1차
민감도라 기여는 σ 에 정비례한다.

    PYTHONPATH=. python experiments/uncertainty_budget.py            # n=121
    PYTHONPATH=. python experiments/uncertainty_budget.py --grid 241
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

from gre.optimize.forward import DiffMotor  # noqa: E402

# ── 기준 설계 (external_case 와 같은 형상, 효율은 프로젝트 기본값)
D_GRAIN, D_CORE = 50.0, 20.0
SEG_LEN, N_SEG = 30.0, 3
D_CHAMBER, L_CHAMBER = 50.0, 106.0
AT_MM2, EPS_RATIO = 61.75, 8.0

#: (이름, 설명, 1σ, 종류) — 종류 "rel" 은 상대(배율), "abs" 는 절대 이동.
#: σ 근거는 모두 [가정]. 측정으로 바뀌면 이 표만 고친다.
PARAMS = [
    ("a",        "연소속도 계수 (전 구간 배율)",        0.10, "rel"),  # strand↔motor 차 흔히 ±10 %
    ("n",        "압력 지수 (전 구간 가산)",            0.05, "abs"),
    ("rho",      "밀도비 (0.92 기준)",                  0.015, "rel"), # 질량/체적 측정 ±1.5 %
    ("cstar_eff", "c* 효율 (0.95)",                     0.03, "abs"),
    ("cf_eff",   "노즐 효율 (0.95)",                    0.03, "abs"),
    ("At",       "노즐목 면적 (직경 ±0.5 %)",           0.010, "rel"),
    ("cstar",    "공표 c* (T_c 불확실)",                0.03, "rel"),
    ("gamma",    "비열비 γ (1.133)",                    0.01, "abs"),
    ("sep",      "박리 문턱 separation_frac (0.4)",     0.10, "abs"),
]
AD_ABLE = {"a", "n", "rho", "cstar_eff", "cf_eff", "At"}


def chamber_margin() -> float:
    v_prop = N_SEG * np.pi * ((D_GRAIN / 2) ** 2 - (D_CORE / 2) ** 2) * SEG_LEN
    return float(np.pi * (D_CHAMBER / 2) ** 2 * L_CHAMBER / v_prop)


def build(n_grid, gamma=1.133, cstar=885.0, sep=0.4):
    return DiffMotor(d_outer=D_GRAIN, n_grid=n_grid, n_w=400, h_max=2e-3,
                     c_star=cstar, gamma=gamma, separation_frac=sep,
                     chamber_margin=chamber_margin())


def run(motor, d=None, grad=False):
    """d: 파라미터 이동량 dict. grad=True 면 텐서 이동량을 받아 그래프를 유지한다."""
    d = d or {}
    z = lambda k: d.get(k, 0.0)
    br = motor.br
    ln_a0, n0 = br.ln_a.clone(), br.n.clone()
    rho0, cse0, cfe0 = motor.rho_p, motor.c_star_actual, motor.cf_eff
    try:
        # 연소속도: ln r = ln a + n ln p  → 배율 a(1+δ) = ln a + ln(1+δ), 지수 n + δ
        br.ln_a = ln_a0 + torch.log1p(torch.as_tensor(z("a")))
        br.n = n0 + z("n")
        motor.rho_p = rho0 * (1.0 + z("rho"))
        motor.c_star_actual = motor.c_star * (motor.c_star_eff + z("cstar_eff"))
        motor.cf_eff = cfe0 + z("cf_eff")
        at = torch.as_tensor(AT_MM2) * (1.0 + z("At"))
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            t, p, f, _ = motor.simulate(torch.tensor(D_CORE / 2), torch.zeros(6),
                                        torch.zeros(6), [SEG_LEN] * N_SEG, at,
                                        expansion_ratio=EPS_RATIO)
        return t, p, f
    finally:
        br.ln_a, br.n = ln_a0, n0
        motor.rho_p, motor.c_star_actual, motor.cf_eff = rho0, cse0, cfe0


def metrics(t, p, f, ref=None):
    t_np = t.detach().numpy()
    f_np, p_np = f.detach().numpy(), p.detach().numpy()
    out = {"I": float(np.trapezoid(f_np, t_np)), "Pmax": float(p_np.max()),
           "Fmax": float(f_np.max())}
    on = np.nonzero(f_np >= 0.05 * f_np.max())[0]
    out["tb"] = float(t_np[on[-1]] - t_np[on[0]])
    if ref is not None:
        tr, fr = ref
        fi = np.interp(tr, t_np, f_np, left=0.0, right=0.0)
        out["rms"] = float(np.sqrt(np.mean((fi - fr) ** 2)) / fr.max() * 100.0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, default=121)
    ap.add_argument("--json", default="experiments/out/uncertainty_budget.json")
    args = ap.parse_args()
    t0 = time.time()

    base = build(args.grid)
    t, p, f = run(base)
    ref = (t.numpy(), f.numpy())
    m0 = metrics(t, p, f, ref)
    print(f"기준  I {m0['I']:.2f} N·s  Pmax {m0['Pmax']:.4f} MPa  "
          f"Fmax {m0['Fmax']:.2f} N  tb {m0['tb']:.3f} s   (n={args.grid})", flush=True)

    keys = ["I", "Pmax", "Fmax", "tb", "rms"]
    rows = {}
    for name, desc, sig, kind in PARAMS:
        res = []
        for s in (+1, -1):
            if name == "gamma":
                mot, d = build(args.grid, gamma=1.133 + s * sig), {}
            elif name == "cstar":
                mot, d = build(args.grid, cstar=885.0 * (1 + s * sig)), {}
            elif name == "sep":
                mot, d = build(args.grid, sep=0.4 + s * sig), {}
            else:
                mot, d = base, {name: s * sig}
            res.append(metrics(*run(mot, d), ref))
        hi, lo = res
        row = {}
        for k in keys:
            base_v = m0[k] if k != "rms" else None
            if k == "rms":
                row[k] = 0.5 * (hi[k] + lo[k])           # 1σ 이동 시 모양 이탈 [%]
            else:
                row[k] = (hi[k] - lo[k]) / 2.0 / base_v * 100.0   # 1σ 당 상대변화 [%]
        rows[name] = {"desc": desc, "sigma": sig, "kind": kind, **row}
        print(f"  {name:9s} σ={sig:<6g} " + "  ".join(
            f"{k} {row[k]:+7.2f}%" for k in keys), flush=True)

    # ── 자동미분 대조: dI/dδ · σ 를 중앙차분과 비교
    print("\n자동미분 대조 (총역적, 1σ 당 %)", flush=True)
    ad = {}
    for name, _, sig, _ in PARAMS:
        if name not in AD_ABLE:
            continue
        x = torch.zeros((), requires_grad=True)
        tt, pp, ff = run(base, {name: x}, grad=True)
        i_tot = torch.trapezoid(ff, tt)
        (g,) = torch.autograd.grad(i_tot, x)
        ad_pct = float(g) * sig / m0["I"] * 100.0
        fd_pct = rows[name]["I"]
        ad[name] = ad_pct
        print(f"  {name:9s} AD {ad_pct:+7.3f}%   FD {fd_pct:+7.3f}%   "
              f"차이 {abs(ad_pct - fd_pct):.3f}%p", flush=True)

    # ── 예산 (제곱합) — 독립 가정
    print("\n예산 (독립 가정, 1σ 제곱합)", flush=True)
    budget = {}
    for k in ["I", "Pmax", "Fmax", "tb"]:
        contrib = sorted(((abs(rows[n][k]), n) for n in rows), reverse=True)
        tot = float(np.sqrt(sum(c * c for c, _ in contrib)))
        budget[k] = {"total_pct": tot,
                     "rank": [(n, c, (c * c) / (tot * tot) * 100) for c, n in contrib]}
        top = ", ".join(f"{n} {s:.0f}%" for n, _, s in budget[k]["rank"][:3])
        print(f"  {k:5s} 합성 1σ {tot:5.2f}%   분산 상위: {top}", flush=True)

    os.makedirs(os.path.dirname(args.json), exist_ok=True)
    with open(args.json, "w", encoding="utf-8") as fh:
        json.dump({"grid": args.grid, "base": m0, "rows": rows, "ad": ad,
                   "budget": budget}, fh, ensure_ascii=False, indent=1)
    print(f"\n{time.time() - t0:.0f} s  →  {args.json}")


if __name__ == "__main__":
    main()
