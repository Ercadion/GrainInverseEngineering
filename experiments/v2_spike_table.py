#!/usr/bin/env python3
"""V2 탈형 관찰 판독표 — **가시 끝 반경 `r_break` → `t_prop_min`** 환산.

얇은 추진제는 케이스 쪽이 아니라 **포트 최소반경 자리**에 있다. 포트 윤곽이
`r(θ) = r₀ + Σ aₖcos(kθ) + Σ bₖsin(kθ)` 이고 추진제가 `r ≥ r(θ)` 쪽이므로,
`r(θ)` 가 **극소**가 되는 각도마다 추진제가 포트 안으로 찌르고 들어간 **가시**가
생긴다. 6-로브면 가시 6 개다.

    가시는 끝으로 갈수록 **연속적으로 가늘어진다.**
    → 형상 자체가 이미 연속 두께 사다리다. 사다리를 따로 만들 필요가 없다.

그래서 탈형한 그레인의 끝면에서 **가시가 어느 반경까지 남아 있는가(`r_break`)**
를 재면 그 반경에서의 두께가 곧 `t_prop_min` 이다. 이 스크립트가 그 환산표를
만든다.

────────────────────────────────────────────────────────────────────────────
⚠ **`constraints.thin_feature_penalty` 를 쓰면 안 된다 — 그건 계측 도구가
   아니다.** 그 함수는 최적화 **벌점**이라 두 가지 장치가 들어 있다.

     · 구조요소 반경이 셀 단위로 **올림**된다 (`k = 2·⌈ρ/h⌉+1`) → 두께 분해능이
       `2h` 로 뭉친다. 운용 격자(h=0.5)에서 1.0 mm 다.
     · 격자 바닥 `min_cells = 12` 아래면 **0 으로 만든다** (깨끗한 원형이 위반으로
       찍히는 것을 막으려고 일부러 넣은 것).

   이전에 그 함수로 "최소 두께" 를 이분해 보다가, 진폭 6.0 에서 0.25 mm,
   6.5 에서 0.75 mm, 7.0 에서 0.25 mm 라는 **단조롭지 않은** 값을 얻었다.
   물리가 아니라 위 두 장치를 재고 있었던 것이다. 그래서 여기서는 벌점 함수를
   쓰지 않고 **순수 기하**(이진 마스크 + 참 원판 열림)로 잰다.

⚠ **케이스 경계 계단을 빼야 한다.** 바깥 원을 사각 격자에 올리면 경계에 계단
   셀이 남고, 열림이 그것을 얇은 곳으로 집어낸다. 면적에는 거의 영향이 없지만
   (4.80 vs 4.98 mm²) **최대 반경 통계는 완전히 망가진다** — 빼지 않으면
   `r_break` 가 전부 케이스 반경(30.00)으로 나온다. 실제로 그렇게 나왔다.
────────────────────────────────────────────────────────────────────────────

사용:

    python experiments/v2_spike_table.py --r0 14.0 --a6 7.0
    python experiments/v2_spike_table.py --r0 14.0 --a "0,0,0,0,0,7.0" --csv out.csv

V2 설계가 확정되면 **그 설계의 계수로 다시 돌려서** 판독표를 뽑는다. 아래 기본값
(r₀ 14.0 · a₆ 7.0)은 §4.4 의 6-로브 깊음 사례일 뿐 V2 설계가 아니다.
"""
from __future__ import annotations

import argparse

import numpy as np
from scipy import ndimage

#: 케이스 경계에서 이만큼 안쪽까지만 통계에 넣는다 [mm]. 위 ⚠ 참조.
BOUNDARY_MARGIN_MM = 2.0


def port_radius(theta, r0: float, a, b) -> np.ndarray:
    """포트 윤곽 r(θ) — `DiffSegment2D.phi` 와 같은 급수."""
    r = np.full_like(theta, float(r0))
    for k, (ak, bk) in enumerate(zip(a, b), start=1):
        if ak:
            r = r + float(ak) * np.cos(k * theta)
        if bk:
            r = r + float(bk) * np.sin(k * theta)
    return r


def spike_table(r0: float, a, b, r_case: float, n: int = 1001,
                thicknesses=None):
    """각 두께 문턱에서 '그보다 얇은 추진제' 의 최대 반경과 면적.

    반환: `[(t, r_break, area_mm2), ...]`, 셀 크기 `h`.
    `r_break` 는 그 두께 아래 영역이 닿는 **가장 바깥 반경**이다 — 탈형 후
    가시가 거기까지 사라져 있으면 `t_prop_min` 이 그 `t` 라는 뜻이다.
    """
    thicknesses = (np.arange(0.5, 3.51, 0.25) if thicknesses is None
                   else np.asarray(thicknesses, dtype=float))
    h = 2.0 * r_case / (n - 1)
    y, x = np.mgrid[-r_case:r_case:n * 1j, -r_case:r_case:n * 1j]
    rr = np.hypot(x, y)
    th = np.arctan2(y, x)
    prop = (rr >= port_radius(th, r0, a, b)) & (rr <= r_case)
    inner = rr < (r_case - BOUNDARY_MARGIN_MM)          # 경계 계단 제외

    rows = []
    for t in thicknesses:
        rho = float(t) / 2.0
        k = int(np.ceil(rho / h))
        yy, xx = np.mgrid[-k:k + 1, -k:k + 1]
        disk = (xx ** 2 + yy ** 2) * h * h <= rho * rho + 1e-12
        thin = prop & ~ndimage.binary_opening(prop, structure=disk) & inner
        rows.append((float(t),
                     float(rr[thin].max()) if thin.any() else float("nan"),
                     float(thin.sum()) * h * h))
    return rows, h


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--r0", type=float, default=14.0, help="포트 기본 반경 [mm]")
    p.add_argument("--a6", type=float, default=None,
                   help="6 차 cos 진폭만 줄 때 [mm] (--a 와 배타)")
    p.add_argument("--a", type=str, default=None,
                   help="cos 계수 6 개, 쉼표 구분 (예: 0,0,0,0,0,7.0)")
    p.add_argument("--b", type=str, default=None, help="sin 계수 6 개")
    p.add_argument("--d-case", type=float, default=60.0, help="케이스 내경 [mm]")
    p.add_argument("--n", type=int, default=1001, help="격자 한 변 셀 수")
    p.add_argument("--csv", type=str, default=None, help="CSV 로도 저장")
    args = p.parse_args()

    if args.a and args.a6 is not None:
        p.error("--a 와 --a6 은 같이 쓸 수 없다")
    a = ([float(v) for v in args.a.split(",")] if args.a
         else [0.0] * 5 + [args.a6 if args.a6 is not None else 7.0])
    b = [float(v) for v in args.b.split(",")] if args.b else [0.0] * len(a)
    if len(b) != len(a):
        p.error("--a 와 --b 의 길이가 다르다")

    r_case = args.d_case / 2.0
    rows, h = spike_table(args.r0, a, b, r_case, n=args.n)

    th = np.linspace(0.0, 2 * np.pi, 4096, endpoint=False)
    rp = port_radius(th, args.r0, a, b)
    print(f"포트 {rp.min():.2f} ~ {rp.max():.2f} mm   "
          f"케이스 내경 {args.d_case:.1f} (R={r_case:.1f})   "
          f"로브끝 웹 {r_case - rp.max():.2f} mm")
    print(f"격자 {args.n}²  h = {h:.4f} mm   "
          f"경계 {BOUNDARY_MARGIN_MM} mm 제외\n")
    print(f"  {'t_prop_min 후보':>15} │ {'가시 끝 반경 r_break':>21} │ "
          f"{'그 아래 면적':>13}")
    print("  " + "─" * 55)
    for t, rb, area in rows:
        rs = f"{rb:.2f} mm" if np.isfinite(rb) else "없음"
        print(f"  {t:>12.2f} mm │ {rs:>21} │ {area:>10.2f} mm²")

    # ── 판독표가 쓸 수 있는 것인가: `r_break` 가 두께에 대해 **단조 증가**해야 한다.
    #
    # ⚠ 양 끝만 비교하면 안 된다. 얕은 형상에서는 중간이 되돌아간다 (실측:
    #   r₀=15·a₆=6.0 에서 9.82 → 9.21 → 9.82). 그러면 같은 `r_break` 가 두
    #   두께를 가리켜 역으로 읽을 수 없다. **모든 이웃 쌍**을 본다.
    fin = [(t, rb) for t, rb, _ in rows if np.isfinite(rb)]
    if len(fin) >= 2:
        drops = [(fin[i][0], fin[i + 1][0], fin[i][1], fin[i + 1][1])
                 for i in range(len(fin) - 1) if fin[i + 1][1] < fin[i][1] - 1e-9]
        flats = [(fin[i][0], fin[i + 1][0]) for i in range(len(fin) - 1)
                 if abs(fin[i + 1][1] - fin[i][1]) <= 1e-9]
        (t0, r0_), (t1, r1_) = fin[0], fin[-1]
        if drops or flats:
            print("\n  ⚠⚠ **이 설계에는 이 방법을 쓸 수 없다.** `r_break` 가 두께에 "
                  "대해 단조롭지 않아 역으로 읽을 수 없다.")
            for ta, tb_, ra, rb_ in drops:
                print(f"       되돌아감: t {ta:.2f} → {tb_:.2f} mm 인데 "
                      f"r_break {ra:.2f} → {rb_:.2f} mm")
            for ta, tb_ in flats:
                print(f"       평평함:   t {ta:.2f} → {tb_:.2f} mm 에서 "
                      f"r_break 가 같다")
            print("       보통 **가시가 뚜렷하지 않은 얕은 형상**이거나 격자가 "
                  "거칠 때다. `--n` 을 올려 보고, 그래도 남으면 이 설계의 "
                  "`t_prop_min` 은 이 시험으로 못 잰다.")
        elif r1_ > r0_:
            slope = (t1 - t0) / (r1_ - r0_)
            print(f"\n  단조 증가 확인 ({len(fin)} 점) — 역으로 읽을 수 있다.")
            print(f"  감도 d(t)/d(r_break) ≈ {slope:.2f} mm/mm  →  "
                  f"r_break 를 ±0.2 mm 로 재면 t_prop_min 은 "
                  f"±{0.2 * slope:.2f} mm")
            print(f"  (도구 자신의 격자 분해능은 운용 격자에서 1.0 mm 다 — "
                  f"§4.6.1)")

    if args.csv:
        with open(args.csv, "w", encoding="utf-8") as fh:
            fh.write("t_prop_min_mm,r_break_mm,area_mm2\n")
            for t, rb, area in rows:
                fh.write(f"{t:.3f},{rb:.4f},{area:.4f}\n")
        print(f"\n  → {args.csv}")


if __name__ == "__main__":
    main()
