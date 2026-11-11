"""F/P 는 얼마나 일정한가 — 두 입력 곡선의 모순 검사 문턱을 정하기 위한 측정.

추력과 압력을 **둘 다** 입력으로 받으면 둘이 서로 모순일 수 있다. 노즐목 `A_t` 는
시간에 대해 고정이므로

    F(t) = C_f(P(t), ε) · P_c(t) · A_t     →     F/P = C_f · A_t

이고, `C_f` 가 압력에 약하게 의존하므로 `F/P` 는 **거의** 일정해야 한다. "거의" 가
얼마인지 모르면 모순 검사 문턱을 정할 수 없다 — 그래서 실제 설계들에서 재 본다.

측정 대상
    1. 창(점화 끝 ~ 웹 소진) 안에서 `F/P` 의 변동폭  ptp/mean
    2. 그 변동이 `C_f` 때문인지 확인 — `F/(C_f·P)` 는 정확히 `A_t` 여야 한다
    3. 목표마다, 노즐목마다 달라지는가

문턱은 **관측된 최대 변동의 2~3 배**로 잡는다. 그보다 크면 "두 곡선이 한 모터에서
나온 것일 수 없다" 고 말할 수 있다.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

torch.set_default_dtype(torch.float64)

from gre.optimize.forward import DiffMotor
from gre.optimize.loss import LossWeights, fp_ratio_spread, shape_window

M = DiffMotor(d_outer=60.0, n_grid=121, n_samples=384, n_w=200, h_max=10e-3)
L, W = 95.0, LossWeights()

CASES = [("원형 r0=12", 12.0, [0] * 6, 85.0),
         ("원형 r0=12, A_t 54", 12.0, [0] * 6, 54.0),
         ("원형 r0=12, A_t 126", 12.0, [0] * 6, 126.0),
         ("얕은 스타", 11.0, [0, 0, 2.2, 0, 0, 0], 78.0),
         ("깊은 스타(4-로브)", 8.0, [0, 0, 0, 5.0, 0, 0], 120.0),
         ("6-로브", 14.0, [0, 0, 0, 0, 0, 4.5], 90.0),
         ("6-로브 깊음", 14.0, [0, 0, 0, 0, 0, 7.0], 90.0),
         ("3+5 복합", 12.0, [0, 0, 3.0, 0, 2.5, 0], 85.0)]


def run(r0, arr, at):
    a = torch.tensor(arr, dtype=torch.float64)
    with torch.no_grad():
        t, p, f, d = M.simulate(torch.tensor(float(r0)), a, torch.zeros(6), [L],
                                torch.tensor(float(at)))
        w_cap = float(d["w_tab"][-1])
        rate = M.br.rate(p)
        # ⚠ 스텝별 `h_list` (평균 `d["h"]` 가 아니다) — `runner.evaluate` 와
        #   같은 방식. 평균으로 쌓으면 웹 소진 시각이 2 배까지 틀리고 그
        #   시각이 창의 테일오프 경계다.
        hs = torch.as_tensor(np.asarray(d["h_list"], dtype=float),
                             dtype=p.dtype)[:rate.shape[0] - 1]
        wtraj = torch.cat([torch.zeros(1, dtype=p.dtype),
                           torch.cumsum(rate[:-1] * hs, 0)])
        wtraj = torch.clamp(wtraj, max=w_cap)
        om = shape_window(t, p, wtraj, w_cap, W)
    return (t.numpy(), p.numpy(), f.numpy(), om.numpy(), d)


if __name__ == "__main__":
    print(f"{'목표':<22}{'A_t':>6}{'창 점수':>8}{'함의 A_t':>10}"
          f"{'A_t 변동':>10}{'F/P 변동':>10}{'ε':>7}{'C_f 범위':>14}")
    worst_at, worst_fp = 0.0, 0.0
    for tag, r0, arr, at in CASES:
        t, p, f, om, d = run(r0, arr, at)
        fp = fp_ratio_spread(torch.tensor(t), torch.tensor(f), torch.tensor(p),
                             gamma=M.gamma, p_ambient=M.p_ambient,
                             cf_eff=M.cf_eff)
        if fp is None:
            print(f"{tag:<22}{at:>6.0f}   창이 너무 좁다 — 건너뜀")
            continue
        cf = (f / np.maximum(p, 1e-12))[om > 0.5] / at
        print(f"{tag:<22}{at:>6.0f}{fp['n']:>8}{fp['at_mean']:>10.2f}"
              f"{100*fp['spread']:>9.2f}%{100*fp['fp_spread']:>9.2f}%"
              f"{fp['eps']:>7.2f}{cf.min():>7.3f}~{cf.max():<6.3f}")
        worst_at = max(worst_at, fp["spread"])
        worst_fp = max(worst_fp, fp["fp_spread"])

    print(f"\n{'='*86}")
    print("**추정자: 함의 노즐목** A_t = (F/P)/C_f(P, ε) 의 5~95 분위폭.")
    print("창은 **구간**이다 (점화 완료 ~ 연소 끝), 준위 마스크가 아니다.")
    print(f"  함의 A_t 변동폭 최대 : {100*worst_at:.2f} %")
    print(f"  (참고) F/P 변동폭 최대 : {100*worst_fp:.2f} %"
          f"  ← 옛 추정자. 차이가 C_f 의 압력 의존이다")
    print("→ A_t 는 시간에 고정이므로 함의 A_t 의 변동은 **전부 모순**이다.")
    print("   남는 것은 자동 ε 가 참 ε 와 다른 데서 오는 잔차뿐이다.")
    thr = max(0.05, np.ceil(100 * worst_at * 4 / 5) * 5 / 100)
    print(f"→ 모순 검사 문턱 제안 : 관측 최대의 4 배 ≒ **{100*thr:.0f} %**")
    print("  (멀쩡한 입력을 거부하는 쪽이 더 나쁘므로 느슨하게 잡는다)")
    print("  ⚠ 자기 일관된 쌍을 넣어 거부되지 않는지 반드시 확인할 것 — 판 둘이")
    print("     이 확인을 빠뜨려 실패했다: 창을 f > 0.05·F_max 로 잡은 판은")
    print("     자기 일관된 쌍을 64.5 % 로 거부했고, 준위 마스크 판은 반대로")
    print("     부스트 압력만 2 배 틀린 쌍을 0.00 % 로 통과시켰다.")
    print("  ⚠ 실측 데이터에는 침식연소·목 침식이 더 보탠다 — 시험 후 재확인.")
