"""형상 식별성 — 이 목표 곡선이 형상을 얼마나 정하는가.

연소속도 피팅에는 대해 이미 같은 개념을 두고 있다.

> 압력이 거의 일정한 중립 곡선이면 데이터가 `n` 정보를 담지 못하므로
> **"n 식별 불가"** 로 표시 (차이가 크다는 결론을 내리면 안 된다)

형상에도 똑같이 적용해야 한다. 다중 출발 측정에서, 변동폭이 8 % 뿐인
거의 중립인 목표에 대해 **출발점 8개가 모두 잔차 0.24 % 이내로 수렴했는데
형상은 회전을 제거하고도 최대 5.97 mm(포트 반경의 55 %) 떨어져 있었다.**
곡선이 형상 정보를 담고 있지 않았던 것이다.

═══════════════════════════════════════════════════════════════════════════
⚠ **원형 기반 검사는 모두 단측(one-sided)이다.** 이 모듈을 쓰기 전에 읽어라.

    원형으로 맞춰진다  →  **식별 불가가 증명된다** (원형이 되면 더 풍부한
                          형상족은 당연히 되므로 해집합이 크다)
    원형으로 안 맞는다  →  **아무것도 증명되지 않는다**

후자가 핵심이다. 6-로브 목표를 원형이 못 맞춘다고 해서 그 목표가 형상을 하나로
정한다는 뜻이 아니다 — 서로 다른 4·5·6-로브 형상들이 똑같이 잘 맞출 수 있다.
**식별 "가능" 판정은 원형 검사로 할 수 없고**, 다중 출발 + `runner.solution_family`
(형상족에 무관하게 해집합의 지름을 직접 잰다) 로 해야 한다.

이 모듈이 답하는 질문은 하나뿐이다: **"이 목표에 형상을 논할 이유가 있는가."**
"어떤 형상인가" 도 "그 형상이 유일한가" 도 답하지 않는다.
═══════════════════════════════════════════════════════════════════════════

쓸 함수 고르기.

    circle_screen()        **권장.** 추력 곡선 기준, 챔버 ODE, 정방향 모델과
                           같은 기하 이산화. 노즐목 자유/고정 두 숫자를 같이 준다.
                           목표 1개 판정에 ~50 s.
    assess()               `A_b(w)` 기준, 노즐목 **고정** 가정. 배열 두 개로 끝나
                           사실상 공짜다. 대량 쓸기의 1차 선별용.
    circle_can_make_curve() 준정상 평형식. 바닥 5.5 %, 탐색이 전역 최소에 못 간다.
                           초고속 1차 선별용으로만 남긴다 (아래 경고 참조).

⚠ **목표 1개를 제대로 판정할 거면 판정자를 쓰지 말고 최적화기를 K = 0 으로
   한 번 돌리는 것도 선택지다.** 측정: `circle_screen` 52 s / 오차 +2.47 %p,
   `minimal_complexity_search(k_max=0)` 153 s. 목표가 하나뿐이면 후자가 낫다.
   판정자가 값어치를 하는 곳은 **파라미터 스터디처럼 목표가 수백 개일 때**다.

   ⚠⚠ 단 그 153 s 는 `iters_per_k = 45` 기준이고, **어려운 목표에서는 미수렴이다**
       (`experiments/convergence_budget.py`). 원형에 가까운 목표는 45 iter 에
       수렴하지만(얕은 스타 45 iter 3.09 % → 300 iter 2.82 %), 어려운 목표는
       그렇지 않다(6-로브 45 iter 19.42 % → 300 iter 3.01 %, 16.4 %p 차이).
       최적화기를 판정에 쓸 거면 **수렴을 확인하고 써야 한다.** 역설적으로,
       미수렴한 최적화기보다 `circle_screen` 이 더 정확한 구간이 있다
       (6-로브: 판정자 8.20 %, 최적화기 45 iter 18.99 %, 수렴값 ~3 %).
       ⚠ 이 세 숫자도 이전이다. 재측정에서 `circle_screen` 의
         6-로브가 **5.32 / 9.32 %** (자유/고정) 로 나왔다 — 허용 10 % 기준으로
         자유·고정 **둘 다 집합 안**이라 §4.6.3 의 '판정이 뒤집힌다'
         예시가 사라진다. 최적화기 쪽 두 숫자는 미확인이다.

⚠ 아래 문턱값은 **경험적 기준**이다. V2/V3 실측과 더 많은 목표 곡선을 겪은
   뒤 다시 잡아야 한다. 지금은 "경고를 띄울지" 판단에만 쓰고, 숫자 자체를
   결론으로 쓰지 않는다.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize_scalar

from ..core.analytic import bates_burn_area
from ..core.nozzle import (area_ratio_to_pressure_ratio, gamma_function,
                           optimum_area_ratio, rt_from_cstar, thrust_coefficient)

#: (변동폭 문턱, 등급) — 작을수록 식별성이 낮다
VARIATION_BANDS = ((0.15, "낮음"), (0.40, "보통"))


@dataclass
class Identifiability:
    variation: float            # A_b 변동폭 (max−min)/mean
    grade: str                  # 높음 / 보통 / 낮음
    circle_residual: float      # 최적 원형 포트의 상대 잔차 [%] (A_t 고정 가정)
    circle_r0: float            # 그 원형의 반경 [mm]
    circle_suffices_fixed_throat: bool   # A_t 를 고정했을 때 원형으로 충분한가
    note: str

    def warnings(self) -> list:
        out = []
        if self.circle_suffices_fixed_throat:
            out.append(
                f"형상 식별 불가 — 반경 {self.circle_r0:.2f} mm 인 **원형 포트**만으로 "
                f"목표 A_b 를 잔차 {self.circle_residual:.2f} % 로 맞출 수 있다 "
                f"(노즐목 고정 가정). 이 목표는 포트 모양을 정하지 못하므로, 역설계 "
                f"결과의 모양을 '이 곡선이 요구한 모양' 으로 읽으면 안 된다.")
        if self.grade == "낮음":
            out.append(
                f"형상 식별성 낮음 — A_b 변동폭이 {100*self.variation:.1f} % 로 "
                f"거의 중립이다. 같은 곡선을 내는 형상이 많으므로 해 족(族)으로 "
                f"보고하고, 2차 기준(복잡도·슬리버 비율)으로 고르는 것을 권한다.")
        if not self.circle_suffices_fixed_throat:
            out.append(
                "⚠ 원형이 못 맞췄다는 것은 **식별 가능하다는 뜻이 아니다.** 이 검사는 "
                "단측이다 — 원형이 되면 식별 불가를 증명하지만, 안 되면 아무것도 "
                "증명하지 못한다. 비원형 형상끼리 서로 바꿔치기 가능할 수 있으므로, "
                "식별 가능 판정은 다중 출발 + `runner.solution_family` 로 해야 한다.")
        return out


def assess(w, a_b, r_outer: float, length: float, ends_burn: bool = True,
           tol_pct: float = 10.0) -> Identifiability:
    """목표 `A_b(w)` 가 형상을 얼마나 정하는지 — **노즐목 고정 가정**의 1차 선별.

    두 지표를 낸다.

    1. **변동폭** `(max − min)/mean` — 곡선이 얼마나 중립인가. 이건 가정에
       의존하지 않는 멀쩡한 지표다. 중립 곡선은 어떤 판정자로도 형상을
       식별하지 못한다.
    2. **최적 원형의 A_b 잔차** — `A_b` 를 비교한다는 것은 **노즐목이 고정**
       이라고 암묵 가정하는 것이다. `A_t` 를 바꾸면 압력 → 연소속도 → 연소시간이
       함께 바뀌어 **`A_b` 가 달라도 같은 추력 곡선**이 나오기 때문이다.
       즉 이 값은 `circle_screen().residual_fixed` 의 싸구려 근사이고,
       노즐목이 자유로운 설계에서는 **체계적으로 비관적**이다. 이 함수가
       12.4 %("원형으로는 부족")라 한 목표를 K 상승 탐색은 모양 오차 3.1 %
       ("원형으로 충분")로 통과시켰다.

    판정은 `circle_screen()` 으로 한다. 이 함수는 대량 쓸기의 1차 선별용이다.
    그리고 모듈 머리의 **단측 검정 경고**가 여기에도 그대로 적용된다.

    Parameters
    ----------
    w, a_b : array_like   목표 연소면적 곡선 [mm], [mm²]
    r_outer : float       그레인 바깥 반경 [mm]
    length : float        세그먼트 길이 [mm]
    tol_pct : float       수렴 허용 오차 [%]
    """
    w = np.asarray(w, dtype=float)
    a_b = np.asarray(a_b, dtype=float)
    scale = float(np.abs(a_b).max())
    variation = float((a_b.max() - a_b.min()) / max(a_b.mean(), 1e-12))

    grade = "높음"
    for thr, g in VARIATION_BANDS:
        if variation < thr:
            grade = g
            break

    # ── 원형 포트 하나로 얼마나 맞출 수 있는가 (BATES 해석해라 사실상 공짜) ──
    #
    # ⚠ **`w` 는 목표의 전 연소 구간(0 ~ 소진)을 덮어야 한다.** 중간에서 끊긴
    #    구간만 주면, 연소시간이 전혀 다른 형상도 "맞는다" 고 나온다. 범위를
    #    끝까지 주면 소진 시점 불일치가 잔차로 드러난다 — BATES 해석해는 웹을
    #    다 태운 뒤 A_b = 0 을 주므로, 원형이 먼저 소진되면 그만큼 큰 잔차가
    #    남는다. (바깥을 0 으로 채워 범위를 늘리는 것은 **안 된다.** 목표가
    #    아직 타고 있는 구간을 0 으로 단정하는 셈이라, 원형이 자기 자신조차
    #    못 찾는다 — 실제로 그렇게 짰다가 r0=11 목표에서 16.93 이 나왔다.)
    def resid(r0: float) -> float:
        if not (0.5 < r0 < r_outer - 0.5):
            return 1e9
        pred = bates_burn_area(w, r0, r_outer, length, ends_burn)
        return float(np.sqrt(np.mean(((pred - a_b) / scale) ** 2)))

    opt = minimize_scalar(resid, bounds=(0.6, r_outer - 0.6), method="bounded",
                          options={"xatol": 1e-4})
    circ_res = float(opt.fun) * 100.0
    circ_ok = circ_res <= tol_pct

    note = (f"A_b 변동폭 {100*variation:.1f} % → 식별성 {grade}. "
            f"최적 원형 포트 r0 = {float(opt.x):.2f} mm 의 잔차 {circ_res:.2f} % "
            f"({'허용 오차 안 — 원형으로 충분' if circ_ok else '원형으로는 부족'})")
    return Identifiability(variation=variation, grade=grade,
                           circle_residual=circ_res, circle_r0=float(opt.x),
                           circle_suffices_fixed_throat=circ_ok, note=note)


# ═══════════════════════════════════ 추력 곡선 기준 (노즐목을 자유로 둔다)
def circle_can_make_curve(t_target, f_target, r_outer: float, length: float,
                          burn_rate, density: float = 1.841 * 0.92,
                          c_star: float = 885.0, c_star_eff: float = 0.95,
                          cf_eff: float = 0.95, gamma: float = 1.133,
                          p_ambient: float = 0.1013, tol_pct: float = 10.0):
    """**원형 포트 + 노즐목**만으로 목표 추력 곡선을 얼마나 맞출 수 있는가.

    ⚠⚠ **판정에는 `circle_screen()` 을 써라.** 이 함수는 두 가지 결함이 있고,
        둘 다 측정으로 확인됐다.

        1. **바닥이 5.5 %**. 준정상 평형식이라 점화 상승도 블로다운도 모델에
           없어서, **실제 원형 포트로 만든 곡선을 넣어도** 잔차가 중앙값 5.46 %,
           최대 6.72 % 나온다. 허용 오차 10 % 중 절반을 모델 오차가 먹는다.
        2. **탐색이 전역 최소에 못 간다**. 14×14 격자 + Nelder-Mead 가
           국소 최소에 갇힌다 — 같은 목표에서 조밀 격자 5.46 % vs 이 함수 6.15 %,
           노즐목은 79.8 vs 109.7 로 **참값을 놓쳤다.**

        `circle_screen()` 은 챔버 ODE 를 벡터로 적분해 바닥을 1.9 % 로 낮추면서
        **오히려 더 빠르다** (설계당 0.78 ms → 전역 격자를 통째로 훑을 수 있다).
        이 함수는 거리장도 ODE 도 필요 없는 **초고속 1차 선별**로만 남긴다.

    ⚠ `assess()` 는 `A_b(w)` 를 고정 `A_t` 로 비교하므로 **과대평가**한다.
       노즐목은 설계 변수이고, 그것을 바꾸면 압력이 바뀌어 연소속도와 연소시간이
       함께 바뀐다. 즉 `A_b` 가 달라도 추력 곡선은 맞출 수 있다. 실제로 시험에서 `assess` 가 "원형으로는 부족(잔차 12.4 %)" 이라 한 목표를
       K 상승 탐색은 **원형으로 모양 오차 3.1 % 에 통과**시켰다.

    그래서 식별성은 **추력 곡선** 에서, `(r0, A_t)` 를 **둘 다 자유**로 두고
    판정해야 한다. BATES 해석해 + 준정상 평형식을 쓰므로 여전히 싸다
    (거리장도 ODE 도 쓰지 않는다).
    """
    t_target = np.asarray(t_target, dtype=float)
    f_target = np.asarray(f_target, dtype=float)
    f_scale = max(float(np.abs(f_target).max()), 1e-9)
    rt = rt_from_cstar(c_star, gamma)
    cs = c_star * c_star_eff
    rho_p = density * 1000.0
    gfun = gamma_function(gamma)

    def curve(r0: float, at_mm2: float):
        w_max = r_outer - r0
        if w_max <= 0.2 or at_mm2 <= 1.0:
            return None
        w = np.linspace(0.0, w_max, 300)
        ab = bates_burn_area(w, r0, r_outer, length) * 1e-6        # m²
        at = at_mm2 * 1e-6
        # 준정상 평형압 (이분법, 벡터화)
        lo, hi = np.full_like(w, 1e-3), np.full_like(w, 50.0)
        for _ in range(45):
            mid = 0.5 * (lo + hi)
            r = burn_rate.rate(mid) * 1e-3
            g = (rho_p - mid * 1e6 / rt) * ab * r - mid * 1e6 * at / cs
            lo, hi = np.where(g > 0, mid, lo), np.where(g > 0, hi, mid)
        p = 0.5 * (lo + hi)
        r_mm = burn_rate.rate(p)
        dt = np.diff(w) / np.clip((r_mm[1:] + r_mm[:-1]) / 2, 1e-9, None)
        t = np.concatenate([[0.0], np.cumsum(dt)])
        band = p >= 0.1 * p.max()
        p_mean = float(np.trapezoid(p[band], t[band])
                       / max(t[band][-1] - t[band][0], 1e-9))
        eps = optimum_area_ratio(p_mean, p_ambient, gamma) if p_mean > p_ambient else 1.0
        pe = area_ratio_to_pressure_ratio(eps, gamma)
        cfm = gfun * np.sqrt(2 * gamma / (gamma - 1) * (1 - pe ** ((gamma - 1) / gamma)))
        cf = cfm + (pe - p_ambient / np.clip(p, 1e-9, None)) * eps
        f = np.clip(cf * cf_eff * p * 1e6 * at, 0.0, None)
        return t, f

    def resid(x):
        out = curve(float(x[0]), float(x[1]))
        if out is None:
            return 1e9
        t, f = out
        pred = np.interp(t_target, t, f, left=0.0, right=0.0)
        return float(np.sqrt(np.mean(((pred - f_target) / f_scale) ** 2)))

    best = (1e9, None, None)
    for r0 in np.linspace(2.0, r_outer - 1.5, 14):          # 거친 격자 → 국소 최적
        for at in np.geomspace(20.0, 400.0, 14):
            v = resid((r0, at))
            if v < best[0]:
                best = (v, r0, at)
    from scipy.optimize import minimize
    res = minimize(resid, x0=[best[1], best[2]], method="Nelder-Mead",
                   options={"xatol": 1e-3, "fatol": 1e-9, "maxiter": 400})
    rms = float(res.fun) * 100.0
    return {"residual_pct": rms, "r0": float(res.x[0]), "a_throat": float(res.x[1]),
            "suffices": rms <= tol_pct,
            "note": (f"원형 포트 r0 = {res.x[0]:.2f} mm + 노즐목 {res.x[1]:.1f} mm² 로 "
                     f"목표 곡선을 모양 오차 {rms:.2f} % 에 맞출 수 있다"
                     + (" — **형상이 필요 없다**" if rms <= tol_pct else
                        " — 원형으로는 부족하므로 형상이 의미를 갖는다"))}


# ═════════════════════════════ 권장 판정자 — ODE + 노즐목 두 가지 기준
@dataclass
class CircleScreen:
    """`circle_screen()` 의 결과 — **두 숫자를 같이** 본다.

    `residual_free` 는 노즐목을 설계 변수로 둔 기준, `residual_fixed` 는
    노즐목을 입력으로 고정한 기준이다. **둘의 차이가 곧 "노즐목이 흡수한 양"**
    이다. 어느 하나보다 쌍이 더 많은 것을 말해 준다.
    """

    residual_free: float            # [%] (r0, A_t) 둘 다 자유
    r0_free: float                  # [mm]
    a_throat_free: float            # [mm²]
    residual_fixed: float = None    # [%] A_t 를 입력으로 고정했을 때
    r0_fixed: float = None          # [mm]
    a_throat_given: float = None    # [mm²] 고정한 값
    tol_pct: float = 10.0

    @property
    def throat_gap(self):
        """노즐목이 흡수한 양 [%p] — 고정했을 때 잔차가 얼마나 더 남는가."""
        if self.residual_fixed is None:
            return None
        return self.residual_fixed - self.residual_free

    @property
    def suffices(self) -> bool:
        return self.residual_free <= self.tol_pct

    def warnings(self) -> list:
        out = []
        if self.suffices:
            out.append(
                f"형상 식별 불가 — 반경 {self.r0_free:.2f} mm 원형 포트 + 노즐목 "
                f"{self.a_throat_free:.1f} mm² 만으로 목표를 잔차 "
                f"{self.residual_free:.2f} % 에 맞출 수 있다. 이 목표는 포트 모양을 "
                f"정하지 못하므로, 역설계 결과의 모양을 '이 곡선이 요구한 모양' 으로 "
                f"읽으면 안 된다.")
        gap = self.throat_gap
        if gap is not None and gap > 2.0:
            out.append(
                f"노즐목이 형상 정보를 {gap:.1f} %p 흡수하고 있다 — 노즐목을 "
                f"{self.a_throat_given:.1f} mm² 로 고정하면 잔차가 "
                f"{self.residual_free:.2f} → {self.residual_fixed:.2f} % 로 오른다. "
                f"노즐목을 설계 변수로 둘지 입력으로 둘지가 형상 식별성을 좌우한다.")
        return out

    @property
    def note(self) -> str:
        s = (f"원형 포트 r0 = {self.r0_free:.2f} mm + 노즐목 "
             f"{self.a_throat_free:.1f} mm² 로 목표 곡선을 모양 오차 "
             f"{self.residual_free:.2f} % 에 맞출 수 있다"
             + (" — **형상이 필요 없다**" if self.suffices else
                " — 원형으로는 부족하므로 형상이 의미를 갖는다"))
        if self.residual_fixed is not None:
            s += (f" / 노즐목을 {self.a_throat_given:.1f} mm² 로 고정하면 "
                  f"{self.residual_fixed:.2f} % (r0 = {self.r0_fixed:.2f} mm)")
        return s


def circle_screen(motor, t_target, f_target, length: float,
                  a_throat: float = None, tol_pct: float = 10.0,
                  n_r0: int = 81, n_at: int = 51,
                  r0_range=None, at_range=(25.0, 400.0)) -> CircleScreen:
    """**원형 귀무가설 선별** — 챔버 ODE 로, 정방향 모델과 같은 기하 이산화로.

    ⚠ **용도가 한정돼 있다.** 이 함수는 단측 검정이고 비싸다.

        목표가 수백 개 (파라미터 스터디, 곡선 공간 쓸기)  →  **이것을 써라**
        목표가 하나, 판정이 목적                        →  `minimal_complexity_search(k_max=0)`
        "형상이 하나로 정해지는가" 가 질문               →  `shape_identifiability()`

    목표 1개 기준 측정: 이 함수 52 s / 오차 +2.47 %p,
    `minimal_complexity_search(k_max=0)` 153 s / 정확. 2.9 배를 아끼려고 허용
    오차 10 % 중 2.5 %p 를 쓰는 셈이라, 목표가 하나면 손해다.

    그리고 이 함수가 "원형으로는 부족" 이라고 해도 **식별 가능하다는 뜻이 아니다**
    (모듈 머리의 단측 검정 경고). 그 답은 `shape_identifiability()` 에만 있다.

    `motor` 는 최적화기가 쓰는 `DiffMotor` 다. 판정자는 그것으로부터 ε 완화 폭과
    둘레 차분 스텐실까지 물려받는다 (`OdeCircleJudge.from_motor`). 판정자의 질문이
    **"최적화기가 쓰는 그 모델에서 원형이 이 곡선을 낼 수 있는가"** 이므로,
    기하 이산화까지 같아야 답이 맞는다.

    측정:

        판정자                     바닥 중앙/최대   비원형 최소   r0 복원오차 평균
        준정상 (circle_can_make_curve) 5.46 / 6.72 %   7.36 %      1.180 mm
        ODE, 기하 불일치               2.25 / 3.20 %   6.29 %      0.280 mm
        ODE, 기하 일치 (이것)           1.94 / 3.90 %   4.70 %      0.160 mm

    **분리폭만 보면 기하 불일치 쪽이 넓어 보이지만 그건 비관 편향이다.**
    K 상승 탐색이 실제 `DiffMotor` 로 낸 결과와 대조하면

        얕은 스타 (실제 3.08 %) : 기하 불일치 7.62 % (+4.54 %p) / 일치 5.54 % (+2.46 %p)
        깊은 스타 (실제 2.89 %) : 기하 불일치 6.29 % (+3.40 %p) / 일치 4.70 % (+1.81 %p)

    기하를 맞춘 쪽이 **양쪽 지표 모두** 실제 최적화 결과에 가깝다. 비원형 잔차가
    내려간 것은 판정이 관대해진 것이 아니라 **비관 편향이 빠진 것**이다.

    ⚠⚠ **정정: 위 두 표는 이전 측정이고 '바닥' 열이 무효다.**
        수정이 둘레를 반 스텝 중심차분으로 고치면서 판정자 바닥이 내려갔다.
        재측정 (`OdeCircleJudge.from_motor`, n_grid=121, 원형 목표 3 개):

            ODE, 기하 일치 (이것)   **0.199 / 0.405 / 0.270 %**   (표의 1.94/3.90 → 10 배 낮다)
            ODE, 기하 불일치        **3.32 / 5.71 / 3.42 %**      (표의 2.25/3.20 → 약 2 배 높다)

        `tests/test_ode_judge.py` 가 이미 그 사실을 적고 `residual_pct < 1.0` 을
        단언한다 ("이전에는 ε 일치가 1.94~3.90 % 였다").

    ⚠⚠ **그래서 위 "목표 1개 기준 측정 (52 s / +2.47 %p)" 과 거기서 끌어낸
        권고("목표가 하나면 손해다") 도 근거가 사라졌다.** +2.47 %p 는 무효가 된
        바닥 열에서 나온 숫자다. 재측정하지 않았으므로 **지금 이 권고는 미확인
        이다** — 판정자를 고를 때 이 문장을 근거로 쓰지 말 것.
        🔜 재측정하면 기하 일치 판정자의 바닥이 0.2~0.4 % 이므로 오차 쪽 논거는
        오히려 **판정자에 유리**해질 가능성이 높다 (비용 논거만 남는다).

    ⚠ **창(ω(t))을 씌우지 않는다.** 준정상 판정자에는 점화 상승과 블로다운이
       없어서 그 구간을 잘라내야 했지만(바닥 5.5 → 3.0 %), ODE 판정자에는 둘 다
       있으므로 자를 것이 없다 — 측정에서 전 구간 2.25 % vs 창 2.29 % 로 차이가
       없었다. 창은 준정상의 결함을 가리는 반창고였지 그 자체로 옳은 조치가
       아니었다.

    Parameters
    ----------
    motor : DiffMotor      최적화기가 쓰는 정방향 모델
    t_target, f_target     목표 추력 곡선 [s], [N]
    length : float         세그먼트 길이 [mm]
    a_throat : float       노즐목을 입력으로 둘 때 그 값 [mm²]. 주면 두 번째
                           기준(고정)도 함께 재서 `throat_gap` 을 돌려준다.
    """
    from .ode_judge import OdeCircleJudge          # 순환 import 방지

    judge = OdeCircleJudge.from_motor(motor)
    r_out = float(motor.ds.r_outer)
    lo, hi = r0_range if r0_range else (3.0, r_out - 3.0)
    r0_grid = np.linspace(lo, hi, int(n_r0))
    at_grid = np.geomspace(at_range[0], at_range[1], int(n_at))
    if a_throat is not None:
        # ⚠ 자유 탐색 격자에 **고정값을 반드시 넣는다.** 안 넣으면 '자유' 가
        #   '고정' 의 상위집합이 아니게 되어, 자유도가 더 많은데 잔차가 더 큰
        #   모순이 난다 (실측: 자유 0.259 % vs 고정 0.151 %). 격자 탐색에서
        #   흔한 함정이고, 모델이 정확해질수록 드러나기 쉽다.
        at_grid = np.unique(np.append(at_grid, float(a_throat)))

    free = judge.fit(t_target, f_target, length, r0_grid, at_grid)
    out = CircleScreen(residual_free=free["residual_pct"], r0_free=free["r0"],
                       a_throat_free=free["a_throat"], tol_pct=float(tol_pct))
    if a_throat is not None:
        fixed = judge.fit(t_target, f_target, length, r0_grid,
                          at_fixed=float(a_throat))
        out.residual_fixed = fixed["residual_pct"]
        out.r0_fixed = fixed["r0"]
        out.a_throat_given = float(a_throat)
    return out


# ══════════════════════════ 진짜 식별성 — 해집합의 지름을 직접 잰다
@dataclass
class ShapeIdentifiability:
    """`shape_identifiability()` 의 결과 — **원형과 무관한 양측 측정**.

    원형 기반 검사(`assess`, `circle_screen`)는 단측이라 "원형으로 안 된다" 에서
    멈춘다. 그 다음 질문 — **"그럼 형상이 하나로 정해지는가"** — 에는 답하지
    못한다. 여기서는 서로 다른 출발점에서 수렴한 해들이 **얼마나 흩어져 있는지**
    를 직접 재서 답한다. 형상족을 가정하지 않으므로 양측이다.

    ⚠ `spread_mm` 은 해집합 지름의 **하한**이다. 출발점이 유한하므로, 못 찾은
       해가 더 멀리 있을 수 있다. "좁다" 는 결론은 "출발점 N 개로는 못 벌렸다"
       로 읽어야 한다.
    """

    n_starts: int
    n_accepted: int                 # 허용 오차 + 제작 제약을 통과한 해의 수
    spread_mm: dict = None          # {min, median, max} 해끼리의 형상 거리 [mm]
    spread_frac: float = None       # max 지름 / 평균 포트 반경
    grade: str = "판정 불가"        # 높음 / 보통 / 낮음 / 판정 불가
    recommended: dict = None        # 2차 기준(복잡도·슬리버)으로 고른 해
    family: list = field(default_factory=list)
    circle: object = None           # 참고용 CircleScreen (있으면)

    #: (지름/포트반경 문턱, 등급) — 작을수록 해가 하나에 가깝다
    BANDS = ((0.10, "높음"), (0.30, "보통"))

    def warnings(self) -> list:
        out = []
        if self.n_accepted == 0:
            out.append("허용 오차와 제작 제약을 함께 만족하는 해가 없다 — 목표가 "
                       "이 케이스에서 실현 불가이거나 제약이 너무 빡빡하다.")
            return out
        if self.n_accepted == 1:
            out.append(f"출발점 {self.n_starts} 개 중 하나만 통과해 **흩어짐을 잴 수 "
                       f"없다.** 식별성이 높다는 뜻이 아니라 표본이 없다는 뜻이다 — "
                       f"출발점을 늘리거나 제약을 완화해 다시 재라.")
            return out
        if self.grade == "낮음":
            out.append(
                f"형상 식별성 낮음 — 같은 곡선을 내는 해들이 최대 "
                f"{self.spread_mm['max']:.2f} mm (포트 반경의 "
                f"{100*self.spread_frac:.0f} %) 떨어져 있다. 해 하나를 '이 곡선이 "
                f"요구한 형상' 으로 내놓으면 안 된다. 족(族)으로 보고하고 2차 "
                f"기준(복잡도·슬리버 비율)으로 고르게 하라.")
        if self.circle is not None and self.circle.suffices:
            out.append("원형 선별도 '형상 불필요' 로 나왔다 — 두 측정이 일치한다.")
        return out

    @property
    def note(self) -> str:
        if self.spread_mm is None:
            return (f"출발점 {self.n_starts} 개 중 {self.n_accepted} 개 통과 — "
                    f"흩어짐을 잴 수 없다")
        return (f"출발점 {self.n_starts} 개 중 {self.n_accepted} 개가 허용 오차와 "
                f"제작 제약을 통과했고, 서로 최대 {self.spread_mm['max']:.2f} mm "
                f"(포트 반경의 {100*self.spread_frac:.0f} %) 떨어져 있다 "
                f"→ 식별성 {self.grade}")


def _grade_of_spread_frac(frac: float) -> str:
    """해집합 지름 / 포트 반경 → 식별성 등급.

    ⚠ **함수로 빼 둔 이유가 있다.** 초판은 이 루프가 `shape_identifiability`
      본문에 인라인되어 있었고, 시험은 같은 루프를 **시험 본문에 다시 구현해**
      자기 출력을 단언했다 — 그래서 실제 채점 코드가 한 번도 실행되지 않았다. 함수로 꺼내면 시험이 구현을 직접 부를 수 있다.
    """
    for thr, g in ShapeIdentifiability.BANDS:
        if float(frac) < thr:
            return g
    return "낮음"


def shape_identifiability(motor, target, n_starts: int = 6, n_modes: int = 4,
                          lengths=(95.0,), limits=None, weights=None,
                          iters: int = 90, tol=None,
                          circle: object = None, *, throat,
                          verbose: bool = True) -> ShapeIdentifiability:
    """**권장 식별성 측정** — 해집합의 지름을 직접 잰다 (원형과 무관).

    `runner.solution_family` 를 돌려 통과한 해들의 형상 거리를 모으고, 포트
    반경으로 무차원화해 등급을 매긴다. 거리는 `shape_distance` 가 **회전을 최적
    정렬한 뒤** 재므로, 정확한 대칭인 회전을 "다른 해" 로 세지 않는다.

    왜 이것이 기준이어야 하는가 — 원형 검사는 단측이다:

        원형으로 맞춰진다  → 식별 불가가 증명된다
        원형으로 안 맞는다  → **아무것도 증명되지 않는다**

    후자에 답할 수 있는 것은 이 측정뿐이다. 측정에서 출발점 8 개가 전부
    잔차 0.24 % 이내로 수렴했는데 형상은 최대 5.97 mm (포트 반경의 55 %)
    떨어져 있었다 — 곡선이 형상 정보를 담지 않았던 것이다.

    비용이 실질적인 문제다 (출발점 하나당 `iters` 회 정방향+역방향). 목표 하나를
    제대로 판정할 때 쓰고, 목표가 수백 개인 쓸기에는 `assess` 나 `circle_screen`
    으로 거른다.

    Parameters
    ----------
    circle : CircleScreen, optional
        `circle_screen()` 결과를 주면 두 측정을 나란히 보고한다. 원형 검사가
        "충분" 이라 했는데 여기서 흩어짐이 작게 나오면 **둘 중 하나가 틀린
        것이므로** 출발점을 늘려 다시 재야 한다.
    """
    from ..optimize.runner import solution_family        # 순환 import 방지

    fam = solution_family(motor, target, n_starts=n_starts, n_modes=n_modes,
                          lengths=lengths, limits=limits, weights=weights,
                          iters=iters, tol=tol, verbose=verbose, throat=throat)
    keep = fam["accepted"]
    out = ShapeIdentifiability(n_starts=n_starts, n_accepted=len(keep),
                               recommended=fam["recommended"],
                               family=fam["family"], circle=circle)
    if fam["spread_mm"] is None:
        return out
    out.spread_mm = fam["spread_mm"]
    r_mean = float(np.mean([r["r0"] for r in keep]))
    out.spread_frac = fam["spread_mm"]["max"] / max(r_mean, 1e-9)
    out.grade = _grade_of_spread_frac(out.spread_frac)
    return out
