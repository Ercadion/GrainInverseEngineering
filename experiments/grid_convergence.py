"""격자 수렴 — **모델이 구분할 수 있는 최소 모드 진폭**을 잰다.

## 왜 재는가

`runner.amplitude_floor()` 는 모드 진폭이 바닥 아래면 "K 를 올려 적을 근거가
약하다" 고 경고한다. 그 바닥이 둘인데 —

    제작  `DELTA_TRANSFER_MM` = 0.020 mm   시편에서 확정
    모델  격자·완화 폭 아래                 **이 스크립트가 재는 것**

모델 쪽이 지금까지 `max(0.01·r₀, 0.05)` 라는 휴리스틱이었다. ⚠ **그 식에는
격자 크기 `h` 가 들어가지 않는다.** 이름은 "격자 분해능" 인데 격자를 조여도
값이 안 변한다 — 설계 문서 수정이 "격자를 조이면 바닥이 내려가고 그때 δ 가
판정한다" 고 쓴 문장이 그 식 아래에서는 **구조적으로 성립할 수 없다.**

## 무엇을 재는가

모드 진폭 `A` 는 추력 곡선을 바꾼다. 모델도 이산화 오차를 낸다. 둘이 같아지는
지점이 바닥이다.

    신호  S(A, h) = dist( F_h(A) , F_h(0) )      진폭이 곡선을 바꾸는 양
    오차  E(A, h) = dist( F_h(A) , F_ref(A) )    그 격자의 이산화 오차

바닥을 **두 가지로** 잰다. 어느 쪽을 쓰느냐가 값을 두 배 이상 바꾼다.

    A_abs(h)  S(A, h) = E(A, h) 인 A
              "모드가 바꾸는 양 > 격자 오차 전체". 보수적이다 — E 의 대부분이
              진폭과 무관한 **상수 치우침**이기 때문이다 (n=121 에서 진폭 0 일
              때 이미 3.88 %, 진폭을 줄여도 안 내려간다). 상수 치우침은 A 와
              0 을 구별하는 데는 지워진다.

    A_sig(h)  |S(A, h) − S_ref(A)| = S_ref(A) 인 A        ← **이걸 쓴다**
              "그 격자가 추정한 모드의 효과가 효과 자체만큼 틀린다". 식별
              가능성이 실제로 묻는 것 — `amplitude_floor()` 가 답해야 하는
              질문이 "K 를 올려 적을 근거가 있나" 이므로 치우침이 아니라
              **응답의 신뢰도**가 기준이다.

`dist` 는 **손실이 쓰는 그 측도**다 (`curve_loss` 의 `rms_shape_pct`) — 다른
측도로 재면 최적화기가 보는 것과 다른 바닥이 나온다.

⚠ **필요조건이지 충분조건이 아니다.** S > E 는 "모드가 곡선을 격자 오차보다
   크게 바꾼다" 는 뜻이지, "최적화기가 그 차이를 **쓸 수 있다**" 는 뜻은
   아니다. r₀ 와 `A_t` 가 자유롭기 때문에 작은 모드 변화는 보상될 수 있다.
   그 질문은 훨씬 비싸고 여기서 묻지 않는다.

## 쓰는 법

    python experiments/grid_convergence.py            # 전체 (약 6 분)
    python experiments/grid_convergence.py --quick    # 격자 2개 (약 1 분)
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from gre.optimize.forward import DiffMotor
from gre.optimize.loss import LossWeights, Target, curve_loss, shape_window

# ── 시험 대상 — 운용 설정과 같은 모터 ─────────────────────────────────────
D_OUTER, LENGTH, A_THROAT, R0 = 60.0, 95.0, 80.0, 14.0
N_MODES, K_MODE = 8, 3           # k=3 — 진짜 로브 모드 (k=1 은 포트 평행이동이다)
GRIDS = (121, 161, 241, 321)     # 운용 격자는 121 (h = 0.5 mm)
GRID_REF = 481                   # 기준 격자 (h = 0.125 mm)
AMPS = tuple(float(x) for x in np.geomspace(2e-3, 1.0, 13))


EPS_CELLS = 1.5                  # 운용 완화 폭 (DiffMotor 기본값)
EPS_SWEEP = (1.0, 1.5, 2.5)      # --eps 에서 훑는 완화 폭


def simulate(n_grid: int, amp: float, eps_cells: float = EPS_CELLS):
    """한 격자·한 진폭의 추력 곡선."""
    m = DiffMotor(d_outer=D_OUTER, n_grid=n_grid, eps_cells=eps_cells)
    a = torch.zeros(N_MODES, dtype=m.dtype)
    b = torch.zeros(N_MODES, dtype=m.dtype)
    a[K_MODE - 1] = amp
    t, p, f, diag = m.simulate(torch.tensor(R0, dtype=m.dtype), a, b, [LENGTH],
                               torch.tensor(A_THROAT, dtype=m.dtype))
    return {"t": t, "p": p, "f": f, "diag": diag, "motor": m}


def dist(ref: dict, got: dict) -> float:
    """두 곡선 사이 거리 [%] — **손실이 쓰는 그 측도**.

    `ref` 를 목표로 삼고 `got` 을 평가한다. 창(ω)은 `got` 쪽에서 나온다 —
    `curve_loss` 가 최적화 중에 하는 것과 같은 규약이다.
    """
    w = LossWeights()
    tgt = Target(t=ref["t"], f=ref["f"])
    m, d = got["motor"], got["diag"]
    w_cap = float(d["w_tab"][-1])
    r_of_t = m.br.rate(got["p"])
    # ⚠ 스텝별 `h_list` 로 적분한다 — `d["h"]`(평균)는 웹 소진 시각을 2 배까지
    #   틀리게 만들고, 그 시각이 `shape_window` 의 테일오프 경계다 (전면 검토). `runner.evaluate` 와 **같은 방식**이어야 여기서 재는
    #   거리가 최적화기가 보는 거리와 같다.
    hs = torch.as_tensor(np.asarray(d["h_list"], dtype=float),
                         dtype=got["p"].dtype)[:r_of_t.shape[0] - 1]
    traj = torch.cat([torch.zeros(1, dtype=got["p"].dtype),
                      torch.cumsum(r_of_t[:-1] * hs, 0)])
    traj = torch.clamp(traj, max=w_cap)
    omega = shape_window(got["t"], got["p"], traj, w_cap, w)
    _, _, rep = curve_loss(got["t"], got["f"], got["p"], omega, tgt, w)
    return rep["rms_shape_pct"]


def crossing(amps, ratio) -> float:
    """`ratio`(A) = 1 이 되는 A — log–log 선형 보간.

    `ratio` 는 "신호 / 문턱" 이고 A 가 커질수록 커진다고 본다. 작은 A 쪽은
    ratio < 1 (바닥 아래), 큰 A 쪽은 > 1. 교차점이 바닥이다.

    못 만나면: 모든 점이 1 위면 nan (바닥이 측정 구간보다 아래),
    모두 1 아래면 inf (바닥이 구간보다 위).
    """
    a, r = np.asarray(amps, float), np.asarray(ratio, float)
    ok = np.isfinite(r)
    a, r = a[ok], r[ok]
    if a.size < 2:
        return float("nan")
    below = r < 1.0
    if not below.any():
        return float("nan")
    if below.all():
        return float("inf")
    i = int(np.flatnonzero(~below)[0])           # 처음으로 1 을 넘는 점
    if i == 0:
        return float(a[0])
    y0, y1 = np.log(max(r[i - 1], 1e-12)), np.log(r[i])
    if y1 == y0:
        return float(a[i])
    x0, x1 = np.log(a[i - 1]), np.log(a[i])
    return float(np.exp(x0 + (0.0 - y0) * (x1 - x0) / (y1 - y0)))


def main(quick: bool = False) -> dict:
    grids = GRIDS[:2] if quick else GRIDS
    amps = AMPS[::3] if quick else AMPS
    t_start = time.time()

    print(f"기준 격자 n={GRID_REF} (h={2*D_OUTER/2/(GRID_REF-1):.4f} mm) 로 "
          f"진폭 {len(amps)} 개 계산 중…", flush=True)
    ref = {a: simulate(GRID_REF, a) for a in amps}
    ref0 = simulate(GRID_REF, 0.0)
    sig_ref = [dist(ref0, ref[a]) for a in amps]      # 기준 격자가 본 모드의 효과

    rows, out = [], {}
    for n in grids:
        base = simulate(n, 0.0)
        h = float(base["motor"].ds.h)
        sig, err, rel = [], [], []
        for a, s_r in zip(amps, sig_ref):
            got = simulate(n, a)
            s = dist(base, got)                  # 진폭이 바꾸는 양
            sig.append(s)
            err.append(dist(ref[a], got))        # 그 격자의 이산화 오차 (전체)
            rel.append(abs(s - s_r) / s_r if s_r > 0 else float("nan"))
        e0 = dist(ref0, base)                    # 진폭 0 에서의 격자 오차
        a_abs = crossing(amps, [s / e for s, e in zip(sig, err)])
        a_sig = crossing(amps, [1.0 / r if r > 0 else float("inf") for r in rel])
        out[n] = {"h": h, "amps": list(amps), "signal": sig, "error": err,
                  "signal_ref": sig_ref, "rel_signal_error": rel,
                  "error_at_zero": e0, "a_abs": a_abs, "a_sig": a_sig,
                  "a_abs_over_h": a_abs / h if np.isfinite(a_abs) else None,
                  "a_sig_over_h": a_sig / h if np.isfinite(a_sig) else None}
        rows.append((n, h, e0, float(np.median(err)), a_abs, a_sig))
        print(f"  n={n:4d}  h={h:.4f} mm  격자오차 중앙 {np.median(err):.4f} % "
              f"(진폭0 {e0:.4f} %)  →  A_abs = {a_abs:.4f} ({a_abs/h:.2f}·h)  "
              f"A_sig = {a_sig:.4f} ({a_sig/h:.2f}·h)", flush=True)

    print()
    print(f"{'n':>5} {'h [mm]':>8} {'격자오차%':>10} "
          f"{'A_abs':>8} {'/h':>6} {'A_sig':>8} {'/h':>6}")
    for n, h, e0, em, aa, as_ in rows:
        print(f"{n:5d} {h:8.4f} {em:10.4f} {aa:8.4f} {aa/h:6.2f} "
              f"{as_:8.4f} {as_/h:6.2f}")

    for label, idx in (("A_abs", 4), ("A_sig", 5)):
        rr = [r[idx] / r[1] for r in rows if np.isfinite(r[idx])]
        if len(rr) >= 2:
            spread = max(rr) / min(rr)
            print(f"\n{label}/h = {min(rr):.2f} ~ {max(rr):.2f} "
                  f"({'h 에 비례한다' if spread < 1.5 else 'h 비례가 아니다'}"
                  f", 폭 {spread:.2f}×)")
    print(f"\n총 {time.time()-t_start:.0f} s")
    return out


def eps_study(n_grid: int = 161) -> dict:
    """바닥이 `h` 를 따라가는지 완화 폭 `eps_cells·h` 를 따라가는지 가른다.

    격자는 하나로 고정하고 `eps_cells` 만 바꾼다. 기준은 **운용 완화 폭의**
    고운 격자 — `eps_cells` 는 이산화 매개변수이므로 기준도 같은 규약을 쓴다.
    바닥이 `h` 에만 의존하면 세 줄이 같게 나오고, 완화 폭에 의존하면
    `A/(eps_cells·h)` 가 같게 나온다.
    """
    amps = AMPS
    print(f"기준 격자 n={GRID_REF}, eps_cells={EPS_CELLS} 계산 중…", flush=True)
    ref = {a: simulate(GRID_REF, a) for a in amps}
    ref0 = simulate(GRID_REF, 0.0)
    sig_ref = [dist(ref0, ref[a]) for a in amps]

    rows, out = [], {}
    for ec in EPS_SWEEP:
        base = simulate(n_grid, 0.0, ec)
        h = float(base["motor"].ds.h)
        sig, err = [], []
        for a in amps:
            got = simulate(n_grid, a, ec)
            sig.append(dist(base, got))
            err.append(dist(ref[a], got))
        a_abs = crossing(amps, [s / e for s, e in zip(sig, err)])
        out[ec] = {"h": h, "eps_mm": ec * h, "amps": list(amps), "signal": sig,
                   "error": err, "signal_ref": sig_ref, "a_abs": a_abs}
        rows.append((ec, h, a_abs))
        print(f"  eps_cells={ec:4.1f}  eps={ec*h:.4f} mm  →  A_abs = {a_abs:.4f} "
              f"= {a_abs/h:.2f}·h = {a_abs/(ec*h):.2f}·eps", flush=True)

    print(f"\n{'eps_cells':>9} {'eps [mm]':>9} {'A_abs':>8} {'A/h':>6} {'A/eps':>7}")
    for ec, h, a in rows:
        print(f"{ec:9.1f} {ec*h:9.4f} {a:8.4f} {a/h:6.2f} {a/(ec*h):7.2f}")
    byh = [a / h for _, h, a in rows if np.isfinite(a)]
    bye = [a / (ec * h) for ec, h, a in rows if np.isfinite(a)]
    if len(byh) >= 2:
        sh, se = max(byh) / min(byh), max(bye) / min(bye)
        print(f"\nA/h   폭 {sh:.2f}×   (0.84~1.04 = 격자 훑기에서 본 폭)")
        print(f"A/eps 폭 {se:.2f}×")
        print("→ " + ("바닥은 h 를 따라간다 — eps_cells 는 식에 안 들어간다"
                      if sh < se else
                      "바닥은 완화 폭을 따라간다 — 식에 eps_cells 가 들어가야 한다"))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--eps", action="store_true",
                    help="격자 고정, eps_cells 만 훑는다")
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    res = eps_study() if args.eps else main(args.quick)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump({str(k): v for k, v in res.items()}, fh,
                      ensure_ascii=False, indent=2)
