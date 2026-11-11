"""포트 윤곽 표본 수 `n_samples` 는 격자 `n_grid` 와 **함께** 올라가야 한다.

## 어쩌다 찾았는가

n=321 에서 K 상승 탐색을 돌렸더니 K=1,2,3 이 **모두 정확히 26 반복**에서 "손실
정체"로 멈췄다. 오차는 n=241 의 두 배였다 (I 24.5 % vs 10.2 %). 세 K 가 같은
반복 수에서 멈추는 것은 최적화의 성질이 아니다 — 경사가 망가진 것이다.

원인은 격자가 아니라 **거리장의 기준점**이었다. `DiffSegment2D.phi` 는 윤곽을
`n_samples` 개 점으로 샘플해 그 점들까지의 soft-min 거리를 쓴다. 표본 간격이
격자 칸 `h` 보다 크면, 격자점이 "가장 가까운 표본" 을 볼 때 윤곽의 매끄러움이
아니라 **표본 격자의 톱니**를 본다. 그 톱니가 경사에 들어간다.

    표본 간격 ≒ 2π·r_port / n_samples

이 값은 `n_samples` 만의 함수인데 `h` 는 `n_grid` 로 줄어든다. 그래서 고정
`n_samples=384` 로 격자만 조이면 비(ratio)가 커진다.

    n_grid=121  h=0.5000 mm   표본간격 0.2291 mm = 0.46×h
    n_grid=241  h=0.2500 mm   표본간격 0.2291 mm = 0.92×h
    n_grid=321  h=0.1875 mm   표본간격 0.2291 mm = 1.22×h

## 무엇을 재는가

격자마다 `n_samples` 를 두 배씩 올리며 **총역적의 경사** ∂I/∂(r0, a_k, A_t) 가
어디서 수렴하는지 본다. 판정은 `2× 올렸을 때 상대변화 < TOL_REL` 이다.

경사의 **크기**만이 아니라 **방향**(코사인)도 같이 본다. 방향이 틀리면 선
탐색이 내려갈 쪽을 못 찾아 정체하고, 크기만 틀리면 느려질 뿐이다. n=321 의
증상(세 K 가 같은 반복에서 정체)은 방향 쪽이다.

## 왜 설계를 여러 개 보는가

표본 간격은 `r_port` 에 비례하므로 **포트가 크고 로브가 깊은 설계가 가장
불리하다.** 한 설계만 재면 그 설계에 맞는 규칙이 나와 다른 설계에서 깨진다
(`cf_eff` 를 놓쳤던 F/P 측정과 같은 실패). 그래서 포트 반경과 로브
깊이를 양쪽으로 벌린 설계들을 같이 본다.

## ⚠ 어느 부분을 믿을 것인가 — 먼저 읽을 것

이 파일은 네 부분이고, **앞의 세 부분은 탐침을 잘못 잡아 쓸 수 없는 답을 냈다.**
남겨 두는 이유는 왜 못 쓰는지가 그 자체로 결론이기 때문이다.

| 부 | 무엇 | 쓸 수 있나 |
|---|---|---|
| 1 (`기본`) | 총역적의 경사 수렴 | ✗ `∂I/∂a_k` 가 설계에 따라 0 을 지나 상대 오차가 무의미 |
| 2 (`--loss`) | 손실의 경사 수렴 | △ 탐침 교체 후에만. 격자 간 결과가 들쭉날쭉하다 |
| 3 (`--tau`) | 지배 길이가 τ 인지 | ✗ straight-through 때문에 τ 가 **손실 값**을 안 바꾼다 |
| 4 (`--fd`) | **autograd vs 중심차분** | ○ **이것이 결론이다** |

4 부만 기준 해상도도 참값도 필요 없다 — 같은 모델의 자기 일관성이다.
결과와 결론은 `experiments/gradient_audit.md` 에 있다.

## 쓰는 법

    PYTHONPATH=. python experiments/sample_resolution.py --fd          # 본론
    PYTHONPATH=. python experiments/sample_resolution.py --fd --grids 121,241,321 \
        --samples 384,768,1536,2048
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

torch.set_default_dtype(torch.float64)

from gre.optimize.forward import DiffMotor                       # noqa: E402

L = 95.0

#: 경사가 수렴했다고 볼 **성분별** 상대변화. 2× 올려서 모든 성분이 이보다
#: 덜 바뀌면 수렴.
#:
#: ⚠ **노름이나 코사인으로 판정하면 안 된다.** 초판이 그렇게 했고 놓쳤다.
#:    n=241, 384→1024 에서
#:        ∂I/∂r0 : −10.83 → −12.14   (12 %)
#:        ∂I/∂a₆ :  −1.099 → −1.744  (**59 %**)
#:        ∂I/∂A_t:  −0.664 → −0.660  (0.7 %)
#:    노름은 `∂I/∂r0` 가 지배하므로 상대변화가 11.9 %, 코사인은 0.9991 로
#:    "거의 맞다" 처럼 보인다. 그러나 최적화기가 **형상**을 고치려면 쓰는 것은
#:    `∂I/∂a` 이고 그게 59 % 틀렸다. 노름 판정은 그 성분을 가려 준다.
TOL_REL = 0.05
#: 참고용(판정에는 쓰지 않는다) — 노름 상대변화와 1−cos 도 같이 적는다.

#: 포트 반경과 로브 깊이를 양쪽으로 벌린다. 표본 간격 ∝ r_port 이므로
#: **r0 + A 가 큰 설계가 가장 불리하다** — 마지막 둘이 그 경우다.
CASES = [("원형 r0=12", 12.0, [0] * 6, 85.0),
         ("얕은 스타(3-로브)", 11.0, [0, 0, 2.2, 0, 0, 0], 78.0),
         ("깊은 스타(4-로브)", 8.0, [0, 0, 0, 5.0, 0, 0], 120.0),
         ("6-로브", 14.0, [0, 0, 0, 0, 0, 4.5], 90.0),
         ("6-로브 깊음", 14.0, [0, 0, 0, 0, 0, 7.0], 90.0)]


def grad_of(n_grid: int, n_samples: int, r0: float, arr, at: float):
    """∂(총역적)/∂(r0, a_k*, A_t) — `k*` 는 진폭이 가장 큰 모드."""
    m = DiffMotor(d_outer=60.0, n_grid=n_grid, n_samples=n_samples,
                  n_w=200, h_max=10e-3)
    kstar = int(np.argmax(np.abs(arr))) if np.any(arr) else 2
    p_r0 = torch.tensor(float(r0), requires_grad=True)
    p_a = torch.tensor(arr, dtype=torch.float64, requires_grad=True)
    p_at = torch.tensor(float(at), requires_grad=True)
    t, p, f, d = m.simulate(p_r0, p_a, torch.zeros(len(arr)), [L], p_at)
    # 총역적 — 스텝별 `h_list` 로 적분한다 (평균 스텝은 웹 소진 시각을 2 배까지
    # 틀리게 만든다, 전면 검토).
    hs = torch.as_tensor(np.asarray(d["h_list"], dtype=float),
                         dtype=f.dtype)[:f.shape[0] - 1]
    impulse = (0.5 * (f[:-1] + f[1:]) * hs).sum()
    g = torch.autograd.grad(impulse, [p_r0, p_a, p_at])
    out = np.array([float(g[0]), float(g[1][kstar]), float(g[2])])
    spacing = 2 * np.pi * (r0 + max(abs(x) for x in arr)) / n_samples
    res = (out, float(m.ds.h), spacing, kstar, float(impulse.detach()))
    del m, t, p, f, d, g
    gc.collect()
    return res


#: 경사 성분의 이름 — 표와 판정에 그대로 쓴다.
COMPS = ("∂I/∂r0", "∂I/∂a*", "∂I/∂A_t")

#: 0 에 가까운 성분의 상대 오차는 무의미하다 — 가장 큰 성분의 이 비율을
#: 분모의 바닥으로 둔다. 바닥보다 작은 성분은 선 탐색을 움직일 수 없다.
ETA_FLOOR = 0.01


def rel_cos_floored(g_a, g_b):
    """`(성분별, 최대, 노름, 1−cos)` — 분모에 `ETA_FLOOR` 바닥을 깐다."""
    g_a, g_b = np.asarray(g_a, float), np.asarray(g_b, float)
    na, nb = np.linalg.norm(g_a), np.linalg.norm(g_b)
    scale = np.maximum(np.abs(g_b), ETA_FLOOR * np.abs(g_b).max())
    per = np.abs(g_a - g_b) / np.maximum(scale, 1e-30)
    rel = float(np.linalg.norm(g_a - g_b) / max(nb, 1e-30))
    cos = float(np.dot(g_a, g_b) / max(na * nb, 1e-30))
    return per.tolist(), float(per.max()), rel, 1.0 - cos


def rel_cos(g_a, g_b):
    """`(성분별 상대변화, 최대 성분 상대변화, 노름 상대변화, 1−cos)`.

    판정은 **최대 성분 상대변화**로 한다 — 노름과 코사인은 참고용이다
    (`TOL_REL` 주석의 n=241 사례가 왜인지 보여 준다).
    """
    na, nb = np.linalg.norm(g_a), np.linalg.norm(g_b)
    per = np.abs(g_a - g_b) / np.maximum(np.abs(g_b), 1e-30)
    rel = float(np.linalg.norm(g_a - g_b) / max(nb, 1e-30))
    cos = float(np.dot(g_a, g_b) / max(na * nb, 1e-30))
    return per.tolist(), float(per.max()), rel, 1.0 - cos


def sweep(grids, samples, cases=None) -> dict:
    cases = cases or CASES
    rows = []
    print(f"{'설계':<18}{'n_g':>5}{'h[mm]':>8}{'n_s':>6}{'간격/h':>8}"
          f"{'∂r0':>9}{'∂a*':>9}{'∂A_t':>9}{'최대':>9}{'노름':>8}{'1−cos':>10}")
    print("-" * 99)
    for tag, r0, arr, at in cases:
        for ng in grids:
            prev = None
            for ns in samples:
                try:
                    g, h, sp, kstar, I = grad_of(ng, ns, r0, arr, at)
                except (RuntimeError, MemoryError) as exc:
                    print(f"{tag:<18}{ng:>5}{'':>8}{ns:>6}   건너뜀 ({exc})")
                    break
                if prev is None:
                    per, mx, rel, dcos = [float("nan")] * 3, *[float("nan")] * 3
                else:
                    per, mx, rel, dcos = rel_cos(prev, g)
                rows.append({"tag": tag, "n_grid": ng, "h": h, "n_samples": ns,
                             "spacing": sp, "ratio": sp / h, "grad": g.tolist(),
                             "per_comp": per, "max_comp": mx,
                             "rel_change": rel, "one_minus_cos": dcos,
                             "kstar": kstar, "impulse": I})
                f3 = lambda v: "—" if prev is None else f"{100*v:.1f}%"   # noqa: E731
                print(f"{tag:<18}{ng:>5}{h:>8.4f}{ns:>6}{sp/h:>8.2f}"
                      f"{f3(per[0]):>9}{f3(per[1]):>9}{f3(per[2]):>9}"
                      f"{f3(mx):>9}{f3(rel):>8}"
                      f"{('—' if prev is None else f'{dcos:.1e}'):>10}",
                      flush=True)
                prev = g
    return {"rows": rows, "grids": list(grids), "samples": list(samples)}


def verdict(res: dict) -> dict:
    """격자마다 **모든 설계**가 수렴하는 최소 `n_samples` 를 찾는다."""
    rows = res["rows"]
    out = {}
    print(f"\n{'='*80}")
    print(f"판정 기준: 2× 올렸을 때 **모든 경사 성분**의 상대변화 "
          f"< {100*TOL_REL:.0f} %")
    print(f"{'n_grid':>7}{'h[mm]':>8}{'최소 n_s':>10}{'그 간격/h':>11}"
          f"  불리한 설계")
    print("-" * 80)
    for ng in res["grids"]:
        sub = [r for r in rows if r["n_grid"] == ng]
        if not sub:
            continue
        h = sub[0]["h"]
        need = {}
        for tag in {r["tag"] for r in sub}:
            srt = sorted((r for r in sub if r["tag"] == tag),
                         key=lambda r: r["n_samples"])
            # 처음으로 "직전 단계에서 이만큼밖에 안 바뀜" 인 지점의 **직전** 값이
            # 수렴한 값이다 (변화는 ns 와 ns/2 사이의 것이다).
            ok = None
            for i, r in enumerate(srt):
                if i == 0:
                    continue
                if r["max_comp"] < TOL_REL:
                    ok = srt[i - 1]["n_samples"]
                    break
            need[tag] = ok
        worst_tag = max(need, key=lambda k: (need[k] is None,
                                             need[k] or 0))
        worst = need[worst_tag]
        out[ng] = {"h": h, "n_samples": worst, "worst_case": worst_tag,
                   "per_case": need}
        txt = f"> {max(res['samples'])}" if worst is None else f"{worst}"
        ratio = (2 * np.pi * 21.0 / worst / h) if worst else float("nan")
        print(f"{ng:>7}{h:>8.4f}{txt:>10}{ratio:>11.2f}  {worst_tag}")
    return out


def rule(out: dict) -> None:
    """측정값에서 `n_samples(n_grid)` 규칙을 세운다."""
    pts = [(ng, v["n_samples"], v["h"]) for ng, v in out.items()
           if v["n_samples"]]
    if len(pts) < 2:
        print("\n규칙을 세울 점이 부족하다 (수렴 못 한 격자가 있다).")
        return
    print(f"\n{'='*80}")
    print("**규칙 후보** — 표본 간격을 `h` 의 몇 배 아래로 둘 것인가")
    print("간격 = 2π·r_max / n_s,  r_max = 21 mm (가장 불리한 설계의 r0+A)")
    for ng, ns, h in pts:
        print(f"  n_grid={ng:<5} 필요 n_s={ns:<6} "
              f"간격/h = {2*np.pi*21.0/ns/h:.3f}")
    worst_ratio = min(2 * np.pi * 21.0 / ns / h for ng, ns, h in pts)
    print(f"\n모든 격자에서 통한 가장 느슨한 비 = {worst_ratio:.3f}·h")
    print(f"→ 규칙:  n_samples ≥ 2π·r_outer / ({worst_ratio:.2f}·h)"
          f"  (r_outer 를 쓰면 어떤 포트에도 안전하다)")
    for ng, ns, h in pts:
        need = int(np.ceil(2 * np.pi * 30.0 / (worst_ratio * h)))
        print(f"  n_grid={ng:<5} 규칙값 {need:<6} (측정 필요값 {ns})")


# ══════════════════════════════════════════════════ 2부: **손실**의 경사 (본론)
#
# 1부는 총역적의 경사를 쟀다. 그건 틀린 탐침이었다 — `∂I/∂a_k` 는 설계에 따라
# 0 을 지나간다 (총역적은 거의 장약 질량만 보고, 모드는 면적을 2 차로만 바꾼다).
# 0 근처 성분의 **상대** 오차는 수백 % 로 튀지만 최적화에 아무 의미가 없다.
#
# 최적화기가 실제로 쓰는 것은 **손실**의 경사다. 모양 항은 F(t) 곡선 전체를
# 보므로 모드에 민감하다. 그래서 같은 질문을 손실로 다시 묻는다.
#
# ⚠ 목표 곡선은 격자마다 **한 번** 만들어 고정한다. `n_samples` 를 바꿀 때마다
#    목표를 다시 만들면 재는 것이 "경사가 수렴했나" 가 아니라 "목표가 바뀌었나"
#    가 된다.

#: 참 설계에서 떨어진 지점 — 최적화기가 실제로 서 있을 자리. 최적점에서 재면
#: 경사가 0 에 가까워 상대 오차가 무의미해진다.
PROBE_TRUTH = (14.0, [0, 0, 0, 0, 0, 7.0], 90.0)

#: ⚠ 탐침은 **모든 성분이 대칭으로 0 이 아니어야** 한다.
#:
#:    초판은 `b = 0`, `a = [0,0,0,0,0,5]` 에서 쟀다. 그 점에서 `∂L/∂b_k` 는
#:    **대칭으로 정확히 0** 이다 (b → −b 는 형상을 거울로 뒤집을 뿐이라 곡선이
#:    같다). 14 개 성분 중 6 개가 그렇게 0 이었고, 거기 남는 것은 전부 수치
#:    잡음이다. "성분별 상대 오차의 최대" 를 그 잡음이 차지해서 어떤
#:    `n_samples` 로도 수렴하지 않는 것처럼 보였다 (n=241, 3072 에서 171 %
#:    인데 노름은 4.6 %). 1 부에서 `∂I/∂a_k` 가 0 을 지나가 같은 함정에
#:    빠졌는데, 같은 실수를 손실에서 한 번 더 한 것이다.
#:
#:    그래서 모든 모드에 a, b 를 섞어 둔다 — 대칭으로 0 이 되는 성분이 없다.
#:
#: ⚠ 탐침은 **힌지 꺾임에서 떨어져 있어야** 한다.
#:
#:    두 번째 판은 `r0=12, a₆=5, A_t=85` 였다. 그 점의 모양 오차는 **10.4 %**,
#:    총역적 오차는 **9.9 %** — 둘 다 허용 τ = 10 % 에 붙어 있다. 손실이
#:    `max(0, err − τ)²` 이므로 그 자리에 **꺾임**이 있고, 중심차분은 꺾임의
#:    양쪽을 평균해 버린다. 그래서 autograd 와의 비가 ε 에 따라
#:    1.26 → −0.76 → 0.41 로 뒤집혔다 — 모델 결함이 아니라 **측정 위치 결함**이다.
#:    (경사 수렴 측정도 같은 점에서 했으므로 2·3 부의 초판 숫자도 이 영향을 받았다.)
#:
#:    아래 탐침은 모양 21~23 %, 총역적 21 % 로 모든 사양이 τ 밖에 있어 힌지가
#:    전부 활성이고 손실이 매끄럽다. ε 를 100 배 바꿔도 비가 1.66~1.77 로
#:    안정한 것이 그 증거다.
PROBE_AT = (9.0,
            [0.4, 0.0, 1.5, 0.0, 0.0, 1.0],
            [0.3, 0.5, 0.0, 0.7, 0.0, 0.5],
            70.0)


def _r_probe_max() -> float:
    """탐침 윤곽의 최대 반경 — 표본 간격 `2π·r_max/n_s` 의 분자."""
    import numpy as _np
    th = _np.linspace(0, 2 * _np.pi, 4096, endpoint=False)
    k = _np.arange(1, 7)[:, None]
    r = (PROBE_AT[0]
         + (_np.array(PROBE_AT[1])[:, None] * _np.cos(k * th[None, :])
            + _np.array(PROBE_AT[2])[:, None] * _np.sin(k * th[None, :])).sum(0))
    return float(r.max())
#: 목표 곡선을 만들 때 쓰는 표본 수 — 어떤 시험값보다 촘촘해야 한다.
REF_SAMPLES = 2048


def loss_grad(n_grid: int, n_samples: int, target):
    """∂(손실)/∂(r0, a, b, A_t) — 최적화기가 실제로 받는 경사."""
    from gre.optimize.constraints import ManufacturingLimits
    from gre.optimize.loss import LossWeights, Tolerances
    from gre.optimize.runner import DesignVars, evaluate

    m = DiffMotor(d_outer=60.0, n_grid=n_grid, n_samples=n_samples,
                  n_w=200, h_max=10e-3)
    r0, arr, brr, at = PROBE_AT
    dv = DesignVars(r0=torch.tensor(float(r0), requires_grad=True),
                    a=torch.tensor(arr, dtype=torch.float64, requires_grad=True),
                    b=torch.tensor(brr, dtype=torch.float64, requires_grad=True),
                    lengths=(L,),
                    a_throat=torch.tensor(float(at), requires_grad=True))
    loss, rep, _ = evaluate(m, dv, target, ManufacturingLimits(), LossWeights(),
                            tol=Tolerances())
    g = torch.autograd.grad(loss, [dv.r0, dv.a, dv.b, dv.a_throat])
    out = np.concatenate([[float(g[0])], g[1].numpy(), g[2].numpy(),
                          [float(g[3])]])
    res = (out, float(m.ds.h), float(loss.detach()), float(rep["rms_shape_pct"]))
    del m, dv, loss, rep, g
    gc.collect()
    return res


def loss_target(n_grid: int):
    """그 격자에서 참 설계가 내는 곡선 — `REF_SAMPLES` 로 **한 번** 만든다."""
    from gre.optimize.forward import resample
    from gre.optimize.loss import Target

    m = DiffMotor(d_outer=60.0, n_grid=n_grid, n_samples=REF_SAMPLES,
                  n_w=200, h_max=10e-3)
    r0, arr, at = PROBE_TRUTH
    with torch.no_grad():
        t, p, f, _ = m.simulate(torch.tensor(r0),
                                torch.tensor(arr, dtype=torch.float64),
                                torch.zeros(len(arr)), [L], torch.tensor(at))
    g = torch.linspace(0.0, float(t[-1]), 320)
    tg = Target(t=g, f=resample(t, f, g).detach(), p_max=float(p.max()))
    del m
    gc.collect()
    return tg


def loss_sweep(grids, samples) -> dict:
    """격자마다 `n_samples` 를 올리며 **손실 경사**의 수렴을 본다."""
    rows = []
    print(f"손실 경사 수렴 — 참 r0={PROBE_TRUTH[0]} a6={PROBE_TRUTH[1][5]} "
          f"A_t={PROBE_TRUTH[2]},  탐침 r0={PROBE_AT[0]} A_t={PROBE_AT[3]} "
          f"(a, b 모두 비영 — 대칭 0 성분 없음, r_max={_r_probe_max():.1f} mm)")
    print(f"목표 곡선은 격자마다 n_samples={REF_SAMPLES} 로 한 번 만들어 고정\n")
    print(f"{'n_g':>5}{'h[mm]':>8}{'n_s':>6}{'간격/h':>8}{'손실':>12}"
          f"{'모양%':>8}{'성분최대':>10}{'노름':>8}{'1−cos':>10}")
    print("-" * 76)
    for ng in grids:
        target = loss_target(ng)
        prev = None
        for ns in samples:
            try:
                g, h, lo, sh = loss_grad(ng, ns, target)
            except (RuntimeError, MemoryError) as exc:
                print(f"{ng:>5}{'':>8}{ns:>6}   건너뜀 ({exc})")
                break
            sp = 2 * np.pi * _r_probe_max() / ns
            if prev is None:
                mx = rel = dcos = float("nan")
            else:
                _, mx, rel, dcos = rel_cos_floored(prev, g)
            rows.append({"n_grid": ng, "h": h, "n_samples": ns, "ratio": sp / h,
                         "grad": g.tolist(), "loss": lo, "rms_shape_pct": sh,
                         "max_comp": mx, "rel_change": rel,
                         "one_minus_cos": dcos})
            fm = lambda v: "—" if prev is None else f"{100*v:.1f}%"  # noqa: E731
            print(f"{ng:>5}{h:>8.4f}{ns:>6}{sp/h:>8.2f}{lo:>12.5f}{sh:>8.2f}"
                  f"{fm(mx):>10}{fm(rel):>8}"
                  f"{('—' if prev is None else f'{dcos:.1e}'):>10}", flush=True)
            prev = g
        del target
        gc.collect()
    return {"rows": rows, "grids": list(grids), "samples": list(samples)}


def loss_verdict(res: dict) -> dict:
    """격자마다 수렴하는 최소 `n_samples` 와 그때의 간격/h."""
    out = {}
    print(f"\n{'='*76}")
    print(f"판정: 2× 올렸을 때 **모든 손실 경사 성분**의 상대변화 "
          f"< {100*TOL_REL:.0f} % (η 바닥 {ETA_FLOOR:.0%})")
    print(f"{'n_grid':>7}{'h[mm]':>8}{'수렴 n_s':>10}{'그 간격/h':>11}")
    print("-" * 44)
    for ng in res["grids"]:
        srt = sorted((r for r in res["rows"] if r["n_grid"] == ng),
                     key=lambda r: r["n_samples"])
        ok = None
        for i, r in enumerate(srt):
            if i and r["max_comp"] < TOL_REL:
                ok = srt[i - 1]
                break
        if ok is None:
            print(f"{ng:>7}{srt[0]['h']:>8.4f}{'> ' + str(max(res['samples'])):>10}"
                  f"{'—':>11}")
            out[ng] = {"h": srt[0]["h"], "n_samples": None}
        else:
            print(f"{ng:>7}{ok['h']:>8.4f}{ok['n_samples']:>10}"
                  f"{ok['ratio']:>11.2f}")
            out[ng] = {"h": ok["h"], "n_samples": ok["n_samples"],
                       "ratio": ok["ratio"]}
    rs = [v["ratio"] for v in out.values() if v.get("ratio")]
    if rs:
        c = max(rs)
        print(f"\n모든 격자에서 통한 가장 느슨한 간격/h = **{c:.2f}**")
        print(f"→ 규칙:  n_samples ≥ 2π·r_outer / ({c:.2f}·h)")
        for ng, v in out.items():
            need = int(np.ceil(2 * np.pi * 30.0 / (c * v["h"])))
            print(f"   n_grid={ng:<5} h={v['h']:.4f} → n_samples ≥ {need}"
                  f"   (측정 수렴값 {v['n_samples']})")
    return out


# ══════════════════════════════════════ 3부: 지배 길이는 `h` 가 아니라 `τ` 다
#
# 2부에서 n=121(h=0.5)의 손실 경사가 `n_samples ≒ 1536~3072` 에서야 수렴했다.
# 그때 표본 간격은 0.14·h 로, "0.5·h 면 된다" 는 1 부의 어림보다 3.6 배 촘촘하다.
# 왜 그렇게 촘촘해야 하는가 — `phi` 가 **soft-min** 이기 때문이다.
#
#     φ(x) ≒ −τ·logsumexp(−d(x, s_i)/τ),     τ = tau_cells·h = 0.25·h
#
# soft-min 의 경사는 **τ 창 안에 든 표본들**에 분배된다. 표본 간격이 τ 보다
# 크면 창 안에 표본이 한 개뿐이어서 soft-min 이 hard-min 으로 퇴화하고, 경사는
# 그 한 점만 본다 — 윤곽을 따라 톱니가 생긴다. 즉 기준 길이는 `h` 가 아니라
# **`τ = 0.25·h`** 이고, 1 부가 `h` 로 규칙을 세우려 한 것이 틀렸다.
#
# 확인 방법: `τ` 를 바꿨을 때 필요한 `n_samples` 가 **`τ` 에 반비례**하면
# 기전이 맞다. 그러면 격자마다 비싼 수렴 측정을 다시 하지 않고 규칙을 세울 수 있다.
# 또 이 측정은 **더 싼 길**도 같이 본다 — `τ` 를 키우면 필요한 표본이 줄어든다.
# 대가는 거리장이 더 뭉개져 `A_b` 가 편향되는 것이므로, 손실·모양%도 같이 적는다.

TAUS = (0.25, 0.5, 1.0)


def tau_sweep(n_grid: int, samples, taus=TAUS) -> dict:
    """`τ` 를 바꿔 가며 손실 경사가 수렴하는 `n_samples` 를 찾는다."""
    target = loss_target(n_grid)
    rows = []
    print(f"τ 의 효과 — n_grid={n_grid}.  표본 간격을 τ 로 재면 규칙이 서는가\n")
    print(f"{'τ/h':>6}{'τ[mm]':>8}{'n_s':>6}{'간격/τ':>8}{'손실':>11}"
          f"{'모양%':>8}{'성분최대':>10}{'노름':>8}{'1−cos':>10}")
    print("-" * 75)
    for tc in taus:
        prev = None
        for ns in samples:
            from gre.optimize.constraints import ManufacturingLimits
            from gre.optimize.loss import LossWeights, Tolerances
            from gre.optimize.runner import DesignVars, evaluate
            m = DiffMotor(d_outer=60.0, n_grid=n_grid, n_samples=ns,
                          n_w=200, h_max=10e-3)
            m.ds.tau = tc * m.ds.h          # ← 재는 대상
            r0, arr, brr, at = PROBE_AT
            dv = DesignVars(r0=torch.tensor(float(r0), requires_grad=True),
                            a=torch.tensor(arr, dtype=torch.float64,
                                           requires_grad=True),
                            b=torch.tensor(brr, dtype=torch.float64,
                                           requires_grad=True),
                            lengths=(L,),
                            a_throat=torch.tensor(float(at), requires_grad=True))
            loss, rep, _ = evaluate(m, dv, target, ManufacturingLimits(),
                                    LossWeights(), tol=Tolerances())
            g = torch.autograd.grad(loss, [dv.r0, dv.a, dv.b, dv.a_throat])
            gv = np.concatenate([[float(g[0])], g[1].numpy(), g[2].numpy(),
                                 [float(g[3])]])
            sp = 2 * np.pi * _r_probe_max() / ns
            if prev is None:
                mx = rel = dcos = float("nan")
            else:
                _, mx, rel, dcos = rel_cos_floored(prev, gv)
            rows.append({"tau_cells": tc, "tau": m.ds.tau, "n_samples": ns,
                         "sp_over_tau": sp / m.ds.tau, "grad": gv.tolist(),
                         "loss": float(loss.detach()),
                         "rms_shape_pct": float(rep["rms_shape_pct"]),
                         "max_comp": mx, "rel_change": rel,
                         "one_minus_cos": dcos})
            fm = lambda v: "—" if prev is None else f"{100*v:.1f}%"  # noqa: E731
            print(f"{tc:>6.2f}{m.ds.tau:>8.4f}{ns:>6}{sp/m.ds.tau:>8.2f}"
                  f"{float(loss.detach()):>11.5f}{float(rep['rms_shape_pct']):>8.2f}"
                  f"{fm(mx):>10}{fm(rel):>8}"
                  f"{('—' if prev is None else f'{dcos:.1e}'):>10}", flush=True)
            prev = gv
            del m, dv, loss, rep, g
            gc.collect()
    print(f"\n{'='*75}")
    print(f"판정: 성분최대 < {100*TOL_REL:.0f} % 가 된 **직전** n_samples 가 수렴값")
    out = {}
    for tc in taus:
        srt = sorted((r for r in rows if r["tau_cells"] == tc),
                     key=lambda r: r["n_samples"])
        ok = next((srt[i - 1] for i, r in enumerate(srt)
                   if i and r["max_comp"] < TOL_REL), None)
        if ok is None:
            print(f"  τ={tc:.2f}h : 수렴 못 함 (> {max(samples)})")
            out[tc] = None
            continue
        out[tc] = ok
        print(f"  τ={tc:.2f}h : n_samples {ok['n_samples']:>5}  "
              f"간격/τ {ok['sp_over_tau']:.2f}  "
              f"손실 {ok['loss']:.5f}  모양 {ok['rms_shape_pct']:.2f} %")
    got = [(tc, v) for tc, v in out.items() if v]
    if len(got) >= 2:
        same = {round(v["sp_over_tau"], 1) for _, v in got}
        print(f"\n간격/τ 가 τ 와 무관하게 일정한가: {sorted(same)}  → "
              f"{'**기전 확인 — 기준 길이는 τ 다**' if len(same) <= 2 else '일정하지 않다'}")
        print("  (일정하면 n_samples ∝ 1/τ, 즉 τ 를 키우면 표본을 줄일 수 있다)")
        base = next((v for tc, v in got if tc == 0.25), None)
        if base:
            print(f"\nτ 를 키운 대가 (τ=0.25h 기준 손실 {base['loss']:.5f}, "
                  f"모양 {base['rms_shape_pct']:.2f} %):")
            for tc, v in got:
                print(f"  τ={tc:.2f}h : 손실 {v['loss']:.5f} "
                      f"({100*(v['loss']/base['loss']-1):+.2f} %)  "
                      f"모양 {v['rms_shape_pct']:.2f} % "
                      f"({v['rms_shape_pct']-base['rms_shape_pct']:+.3f} %p)  "
                      f"표본 {v['n_samples']} ({v['n_samples']/base['n_samples']:.2f}×)")
    return {"rows": rows, "converged": {str(k): v for k, v in out.items()}}


# ═══════════════════════════ 4부: 경사가 **그 손실의** 경사인가 (결정적 측정)
#
# 2·3 부는 `n_samples` 를 올려 가며 경사가 수렴하는지 봤다. 그 방식에는 약점이
# 있다 — 두 해상도의 경사를 비교하는 것이라, "수렴했다" 가 "맞다" 를 뜻하지
# 않는다. 그리고 격자마다 결과가 들쭉날쭉했다 (n=121 이 n=161 보다 좋다).
#
# `DiffSegment2D.phi` 가 **straight-through** 를 쓴다는 데 답이 있다.
#
#     outs.append(d_hard.detach() + (d_soft - d_soft.detach()))
#     ↑ 값은 hard-min,  경사는 soft-min
#
# 그래서 이 모델에서 `경사 ≠ ∇(손실)` 이다. **의도된 설계**이지만(hard-min 값이
# 면적 계산에 맞다), 둘이 얼마나 벌어지는지는 `n_samples` 에 달려 있다 — 표본이
# 촘촘해지면 soft-min → hard-min 이므로 벌어짐이 줄어든다.
#
# 이것이 n=321 의 증상을 설명한다. 선 탐색은 **손실 값**으로 내려갈지 판단하고
# **경사 방향**으로 걷는다. 둘이 어긋나면 어느 보폭에서도 손실이 안 줄어 "손실
# 정체" 로 멈춘다 — K=1,2,3 이 모두 같은 26 반복에서 멈춘 모양 그대로다.
#
# 그래서 재는 것: **중심차분**으로 같은 모델의 손실을 수치 미분해 autograd 와
# 맞춰 본다. 기준 해상도도, 참값도 필요 없다 — 자기 자신과의 일관성이다.
#
#     · 무작위 방향 `d` 몇 개에 대해  (L(θ+εd) − L(θ−εd)) / 2ε   vs   g·d
#     · 경사 자신의 방향에 대해도 (선 탐색이 실제로 걷는 방향)
#
# 비가 1 에서 멀면 그 보폭에서 선 탐색이 속는다. **0 에 가깝거나 음수면 치명적**
# — 내려간다고 믿고 걷는 방향이 실제로는 평평하거나 올라간다.

#: 중심차분 보폭. 설계 변수의 크기가 mm 단위(r0 ≒ 12, a ≒ 1~5, A_t ≒ 85)이므로
#: 상대 보폭으로 잡는다. 너무 작으면 hard-min 의 불연속이 잡음으로 들어오고,
#: 너무 크면 2 차 항이 섞인다 — 두 보폭에서 같은 답이 나오는지 같이 본다.
FD_STEPS = (1e-3, 1e-2)
#: 방향 수 — 무작위 방향은 재현을 위해 고정 씨앗을 쓴다.
FD_DIRS = 4


def fd_check(n_grid: int, n_samples: int, target, steps=FD_STEPS,
             n_dirs: int = FD_DIRS, seed: int = 7) -> dict:
    """autograd 경사 vs 중심차분 — **같은** 모델, 같은 점."""
    from gre.optimize.constraints import ManufacturingLimits
    from gre.optimize.loss import LossWeights, Tolerances
    from gre.optimize.runner import DesignVars, evaluate

    m = DiffMotor(d_outer=60.0, n_grid=n_grid, n_samples=n_samples,
                  n_w=200, h_max=10e-3)
    lim, wts, tol = ManufacturingLimits(), LossWeights(), Tolerances()
    r0, arr, brr, at = PROBE_AT
    th0 = np.concatenate([[r0], arr, brr, [at]])

    def make(th, grad=False):
        return DesignVars(
            r0=torch.tensor(float(th[0]), requires_grad=grad),
            a=torch.tensor(th[1:7], dtype=torch.float64, requires_grad=grad),
            b=torch.tensor(th[7:13], dtype=torch.float64, requires_grad=grad),
            lengths=(L,),
            a_throat=torch.tensor(float(th[13]), requires_grad=grad))

    def loss_at(th):
        with torch.no_grad():
            lo, _, _ = evaluate(m, make(th), target, lim, wts, tol=tol)
        return float(lo)

    dv = make(th0, grad=True)
    lo, _, _ = evaluate(m, dv, target, lim, wts, tol=tol)
    gt = torch.autograd.grad(lo, [dv.r0, dv.a, dv.b, dv.a_throat])
    g = np.concatenate([[float(gt[0])], gt[1].numpy(), gt[2].numpy(),
                        [float(gt[3])]])

    rng = np.random.default_rng(seed)
    # 변수 크기가 달라 그대로 섞으면 r0·A_t 가 방향을 독점한다 — 각 변수의
    # 크기로 정규화한 뒤 단위벡터로 만든다.
    scale = np.maximum(np.abs(th0), 1.0)
    dirs = [g / max(np.linalg.norm(g), 1e-30)]          # 선 탐색이 걷는 방향
    for _ in range(n_dirs):
        d = rng.normal(size=th0.shape) * scale
        dirs.append(d / np.linalg.norm(d))

    rows = []
    for j, d in enumerate(dirs):
        gd = float(np.dot(g, d))
        for eps in steps:
            st = eps * float(np.linalg.norm(scale))
            num = (loss_at(th0 + st * d) - loss_at(th0 - st * d)) / (2 * st)
            rows.append({"dir": "경사 방향" if j == 0 else f"무작위 {j}",
                         "eps": eps, "g_dot_d": gd, "fd": num,
                         "ratio": num / gd if abs(gd) > 1e-30 else float("nan")})
    out = {"n_grid": n_grid, "n_samples": n_samples, "h": float(m.ds.h),
           "loss": float(lo.detach()), "grad_norm": float(np.linalg.norm(g)),
           "rows": rows}
    del m, dv, lo, gt
    gc.collect()
    return out


def fd_sweep(grids, samples) -> dict:
    allr = []
    print("autograd vs 중심차분 — **같은 모델**의 자기 일관성")
    print("비 = 중심차분 / (g·d).  1 이면 일치, 0 근처나 음수면 선 탐색이 속는다.")
    print("`phi` 가 straight-through(값 hard-min, 경사 soft-min)이므로 둘은 "
          "원리적으로 다르다 — 얼마나 다른지가 질문이다.\n")
    print(f"{'n_g':>5}{'n_s':>6}{'간격/h':>8}{'ε':>8}"
          f"{'경사방향 비':>12}{'무작위 비 (최소~최대)':>24}{'음수':>6}")
    print("-" * 71)
    for ng in grids:
        target = loss_target(ng)
        for ns in samples:
            try:
                r = fd_check(ng, ns, target)
            except (RuntimeError, MemoryError) as exc:
                print(f"{ng:>5}{ns:>6}   건너뜀 ({exc})")
                break
            allr.append(r)
            sp = 2 * np.pi * _r_probe_max() / ns
            for eps in FD_STEPS:
                sel = [x for x in r["rows"] if x["eps"] == eps]
                gdir = next(x for x in sel if x["dir"] == "경사 방향")
                rnd = [x["ratio"] for x in sel if x["dir"] != "경사 방향"]
                nneg = sum(1 for x in sel if x["ratio"] < 0)
                print(f"{ng:>5}{ns:>6}{sp/r['h']:>8.2f}{eps:>8.0e}"
                      f"{gdir['ratio']:>12.3f}"
                      f"{f'{min(rnd):.3f} ~ {max(rnd):.3f}':>24}{nneg:>6}",
                      flush=True)
        del target
        gc.collect()
    print(f"\n{'='*71}")
    print("판정: **경사 방향의 비**가 1 에 가까워야 한다. 그 방향으로 걷는 것이")
    print("      선 탐색이고, 비가 작으면 예상만큼 안 내려가 보폭을 계속 줄이다")
    print("      정체한다. 음수면 올라간다.")
    for r in allr:
        for eps in FD_STEPS:
            gdir = next(x for x in r["rows"]
                        if x["dir"] == "경사 방향" and x["eps"] == eps)
            flag = ("치명적" if gdir["ratio"] < 0.2 else
                    "의심" if gdir["ratio"] < 0.8 else "정상")
            print(f"  n={r['n_grid']:<4} n_s={r['n_samples']:<5} ε={eps:.0e}  "
                  f"비 {gdir['ratio']:>7.3f}  → {flag}")

    # ── 결정적 진단: **경사 노름의 부풀림**
    #
    # 표본이 격자에 비해 모자라면 경사의 크기 자체가 터진다. 같은 탐침에서
    # 격자만 올리면 참 경사는 거의 안 변해야 하는데(같은 물리, 같은 점), 노름이
    # 격자와 함께 커지면 커진 만큼이 전부 톱니다. 이게 "손실 정체" 의 기전이다 —
    # Armijo 선 탐색은 `L(θ − α g)` 가 `α|g|²` 만큼 줄기를 기대하는데, `|g|` 가
    # 10 배 부풀면 어떤 `α` 에서도 그만큼 줄지 않아 전부 거부한다.
    print(f"\n{'─'*71}")
    print("**경사 노름의 부풀림** — 같은 탐침, 같은 물리. 표본이 모자라면 터진다.")
    print(f"{'n_grid':>7}{'n_s':>6}{'|g|':>11}{'가장 촘촘한 n_s 대비':>20}")
    print("-" * 46)
    for ng in grids:
        sub = sorted((r for r in allr if r["n_grid"] == ng),
                     key=lambda r: r["n_samples"])
        if not sub:
            continue
        ref = sub[-1]["grad_norm"]
        for r in sub:
            infl = r["grad_norm"] / max(ref, 1e-30)
            mark = "  ← 부풀었다" if infl > 1.5 else ""
            print(f"{ng:>7}{r['n_samples']:>6}{r['grad_norm']:>11.5f}"
                  f"{infl:>19.2f}×{mark}")
    return {"runs": allr}


# ══════════════════════════════ 5부: 규칙 상수를 **공식이 아니라 측정**으로 정한다
#
# 4 부는 "간격/h = 0.40 은 통과, 0.81 은 깨짐" 까지만 좁혔다. 그 사이는 2 배
# 구간이고, 거기에 `0.5·h` 를 찍는 것은 측정이 아니라 어림이다. 게다가 공식이
# `r_outer = 30 mm` 를 쓰는데 실제 포트는 깊은 6-로브에서도 `r ≤ 21 mm` 라
# 1.4 배 과하게 나온다. 운용 격자는 n=121 하나로 고정됐으므로(①② 취소)
# **그 격자에서 필요한 값만 재면 된다.**
#
# 재는 것: `n_samples` 를 올리며 경사 벡터가 기준(가장 촘촘한 값)으로 수렴하는가.
#
#     |g(n_s) − g(ref)| / |g(ref)|        ← 노름으로 정규화한 벡터 차이
#
# 이 지표는 성분이 0 에 가까워도 터지지 않고(1·2 부가 그래서 못 쓸 답을 냈다),
# 크기와 방향을 같이 본다. 문턱은 **2 %** — 4 부에서 본 straight-through 편향이
# 1.4~1.8 배이므로, 표본 때문에 생기는 오차를 그보다 훨씬 작게 묶어 두면
# `n_samples` 는 더 이상 지배 요인이 아니다.
#
# ⚠ 탐침은 4 부와 **같은** 점이다 (힌지 꺾임 밖). 목표도 `REF_SAMPLES` 로
#    고정한다 — 목표를 매번 다시 만들면 재는 것이 "목표가 바뀌었나" 가 된다.
#    초판 측정이 바로 그 실수를 해서 "손실이 9.2 % 달라진다" 를 보고했다.
#    목표를 고정하면 손실 차이는 0.01 % 다.

#: 표본 때문에 생기는 경사 오차를 이 아래로 묶는다.
RULE_TOL = 0.02


def rule_sweep(grids, samples) -> dict:
    """격자마다 경사가 기준으로 수렴하는 최소 `n_samples` 를 **재서** 고른다."""
    out, rows = {}, []
    print("경사 벡터의 수렴 — 기준은 그 격자의 가장 촘촘한 n_samples")
    print(f"지표 = |g(n_s) − g(ref)| / |g(ref)|,  문턱 {100*RULE_TOL:.0f} %")
    print(f"탐침은 4 부와 같은 점(힌지 밖), 목표는 n_samples={REF_SAMPLES} 로 고정\n")
    print(f"{'n_g':>5}{'h[mm]':>8}{'n_s':>6}{'간격/h':>8}{'|g|':>11}"
          f"{'기준 대비':>11}{'판정':>8}")
    print("-" * 57)
    for ng in grids:
        target = loss_target(ng)
        gs = {}
        for ns in samples:
            try:
                g, h, lo, sh = loss_grad(ng, ns, target)
            except (RuntimeError, MemoryError) as exc:
                print(f"{ng:>5}{'':>8}{ns:>6}   건너뜀 ({exc})")
                break
            gs[ns] = (np.asarray(g, float), h)
        if not gs:
            del target
            gc.collect()
            continue
        ref_ns = max(gs)
        gref = gs[ref_ns][0]
        nref = np.linalg.norm(gref)
        ok = None
        for ns in sorted(gs):
            g, h = gs[ns]
            dev = float(np.linalg.norm(g - gref) / max(nref, 1e-30))
            sp = 2 * np.pi * _r_probe_max() / ns
            good = dev < RULE_TOL
            if good and ok is None:
                ok = ns
            rows.append({"n_grid": ng, "n_samples": ns, "h": h, "ratio": sp / h,
                         "grad_norm": float(np.linalg.norm(g)), "dev": dev,
                         "ok": good, "ref_n_samples": ref_ns})
            print(f"{ng:>5}{h:>8.4f}{ns:>6}{sp/h:>8.2f}"
                  f"{np.linalg.norm(g):>11.5f}{100*dev:>10.2f}%"
                  f"{('통과' if good else '부족'):>8}")
        h = gs[ref_ns][1]
        out[ng] = {"h": h, "n_samples": ok, "ref_n_samples": ref_ns}
        del target, gs
        gc.collect()
    print(f"\n{'='*57}")
    print("격자별 최소 `n_samples` (측정값):")
    for ng, v in out.items():
        if v["n_samples"] is None:
            print(f"  n_grid={ng:<5} 기준({v['ref_n_samples']})까지 "
                  f"수렴 못 함 — 더 촘촘히 재야 한다")
            continue
        sp = 2 * np.pi * _r_probe_max() / v["n_samples"]
        print(f"  n_grid={ng:<5} n_samples ≥ **{v['n_samples']}**  "
              f"(간격 {sp:.4f} mm = {sp/v['h']:.2f}·h,  탐침 r_max "
              f"{_r_probe_max():.1f} mm)")
    got = [(ng, v) for ng, v in out.items() if v["n_samples"]]
    if got:
        c = max(2 * np.pi * _r_probe_max() / v["n_samples"] / v["h"]
                for _, v in got)
        print(f"\n모든 격자에서 통한 가장 느슨한 간격/h = **{c:.2f}**")
        print(f"→ 안전 규칙:  n_samples ≥ 2π·r_outer / ({c:.2f}·h)"
              f"   (r_outer 를 쓰면 어떤 포트에도 성립)")
        for ng, v in got:
            need = int(np.ceil(2 * np.pi * 30.0 / (c * v["h"])))
            print(f"   n_grid={ng:<5} 규칙값 {need:<6} vs 측정 필요값 "
                  f"{v['n_samples']}  (보수성 {need/v['n_samples']:.2f} 배)")
    return {"rows": rows, "verdict": {str(k): v for k, v in out.items()}}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rule", action="store_true",
                    help="5 부 — 규칙 상수를 측정으로 정한다 (가장 실용적)")
    ap.add_argument("--fd", action="store_true",
                    help="4 부 — autograd vs 중심차분 (결정적 측정)")
    ap.add_argument("--tau", action="store_true",
                    help="3 부 — 지배 길이가 τ 인지 확인 (싼 길도 같이 본다)")
    ap.add_argument("--loss", action="store_true",
                    help="**본론** — 손실 경사의 수렴 (총역적 대신)")
    ap.add_argument("--grids", default="121,241,321")
    ap.add_argument("--samples", default="192,384,768,1536")
    ap.add_argument("--only", default="", help="설계 태그 부분일치 필터")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    gs = [int(x) for x in a.grids.split(",")]
    ss = [int(x) for x in a.samples.split(",")]
    cs = [c for c in CASES if a.only in c[0]] if a.only else CASES
    t0 = time.time()
    if a.rule:
        res = rule_sweep(gs, ss)
        out = res.pop("verdict")
    elif a.fd:
        res = fd_sweep(gs, ss)
        out = {}
    elif a.tau:
        res = tau_sweep(gs[0], ss)
        out = res.pop("converged")
    elif a.loss:
        res = loss_sweep(gs, ss)
        out = loss_verdict(res)
    else:
        res = sweep(gs, ss, cs)
        out = verdict(res)
        rule(out)
        print("\n⚠ 1 부(총역적)의 판정은 믿지 말 것 — `∂I/∂a_k` 가 설계에 따라"
              " 0 을 지나가서 상대 오차가 무의미하다. `--loss` 가 본론이다.")
    print(f"\n총 {time.time()-t0:.0f} s")
    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump({"mode": "loss" if a.loss else "impulse", "sweep": res,
                       "verdict": {str(k): v for k, v in out.items()}},
                      fh, ensure_ascii=False, indent=2, default=str)
