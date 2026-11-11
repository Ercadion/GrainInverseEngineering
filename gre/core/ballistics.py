"""내탄도 — 챔버 충전 방정식을 푸는 시간 적분.

상태 변수 두 개.

    dw/dt   = r(P_c)                                                웹 진행
    dρ_c/dt = (ṁ_gen − ṁ_out − ρ_c · dV_free/dt) / V_free           챔버 충전
    P_c     = ρ_c · R·T

⚠ 세 번째 항을 빠뜨리면 안 된다. 추진제가 타면 **자유체적이 커지므로**,
   생성된 가스의 일부는 노즐로 나가는 대신 그 늘어난 공간을 채운다.
   보존식은 밀도가 아니라 질량에 대해 세워야 한다: d(ρ_c·V)/dt = ṁ_gen − ṁ_out.
   dV_free/dt = A_b · r 이므로 이 항은 ρ_gas·A_b·r 이고, 실질적으로
   유효 생성 유량이 (ρ_p − ρ_gas)·A_b·r 로 줄어드는 것과 같다.
   누락 시 압력을 약 2% 높게 예측한다 (V1-b 대조에서 발견).

준정상 평형식을 바로 쓰지 않는 이유는 두 가지다.

1. 점화 직후 압력 상승이 물리적으로 나온다. 준정상은 t=0 에 평형으로
   점프해 버려서, 곡선의 앞부분이 임의의 계단이 된다.
2. 웹 소진처럼 A_b 가 급히 변하는 순간에도 압력이 시상수
   τ = c*·V_free/(R·T·A_t) 에 걸쳐 부드럽게 반응한다.

단위는 내부에서 SI 로 통일한다 (입력은 mm, MPa, g/cc).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from .burnrate import BurnRateModel
from .grain import Grain
from .nozzle import (area_ratio_to_pressure_ratio, optimum_area_ratio,
                     rt_from_cstar, thrust_coefficient)

MM3_TO_M3 = 1e-9
MM2_TO_M2 = 1e-6
MPA_TO_PA = 1e6


@dataclass
class MotorSpec:
    grain: Grain
    burn_rate: BurnRateModel
    a_throat: float                  # [mm²]
    chamber_volume: float            # [mm³] 그레인 포함 챔버 내부 총 체적
    expansion_ratio: float = None    # A_e / A_t. None 이면 자동 산출 (§6.5)
    c_star: float = 885.0            # [m/s] 공표값
    c_star_eff: float = 0.95         # 전달 효율
    cf_eff: float = 0.95             # 추력계수 손실
    gamma: float = 1.133
    p_ambient: float = 0.1013        # [MPa]

    @property
    def rt(self) -> float:
        """R·T [J/kg].

        ⚠ 규약: **공표 c\* 로 역산한다.** 배출 유량은 실효 c\*(= 공표 × c_star_eff)
           를 쓰므로 둘은 c_star_eff² 만큼(기본값에서 9.8 %) 어긋난다. 이는 c*
           효율을 '노즐 유량 손실' 로만 해석하고 연소 온도는 이상값으로 두는
           규약이다. 결과 영향은 작다 — R·T 가 들어가는 곳은 ρ_gas = P/R·T 뿐이고
           그것이 1 % 짜리 보정항이라, 실효 c* 로 바꿔도 최대압 0.087 %,
           총역적 0.024 % 만 변한다. 다만 **시상수 τ 는 9.8 % 달라지므로**
           과도 구간을 실측과 맞출 때는 이 규약을 다시 보아야 한다.
           (2026-10-05 결정: 현행 유지, 명시)
        """
        return rt_from_cstar(self.c_star, self.gamma)

    @property
    def c_star_actual(self) -> float:
        return self.c_star * self.c_star_eff


@dataclass
class BurnResult:
    t: np.ndarray
    p_c: np.ndarray        # [MPa]
    thrust: np.ndarray     # [N]
    web: np.ndarray        # [mm]
    a_burn: np.ndarray     # [mm²]
    mdot_out: np.ndarray   # [kg/s]
    expansion_ratio: float = None   # 실제로 쓰인 ε (자동 산출이면 그 결과)
    p_mean: float = None            # ε 산출 기준이 된 작동 구간 평균 압력 [MPa]
    tau: np.ndarray = None          # [s] 챔버 충전 시상수 τ(t) — 아래 참조

    @property
    def total_impulse(self) -> float:
        return float(np.trapezoid(self.thrust, self.t))

    @property
    def p_max(self) -> float:
        return float(np.max(self.p_c))

    @property
    def f_max(self) -> float:
        return float(np.max(self.thrust))

    @property
    def tau_max(self) -> float:
        """작동 구간 안에서 가장 큰 챔버 충전 시상수 [s]."""
        if self.tau is None:
            return float("nan")
        band = self.p_c >= 0.1 * float(self.p_c.max())
        return float(np.max(self.tau[band])) if band.any() else float(np.max(self.tau))

    def burn_time(self, frac: float = 0.05) -> float:
        """기준 비율(기본 NFPA 1125 의 5%) 교차점 사이의 시간 [s]."""
        thr = frac * self.f_max
        idx = np.where(self.thrust >= thr)[0]
        if idx.size < 2:
            return 0.0
        return float(self.t[idx[-1]] - self.t[idx[0]])

    def average_thrust(self, frac: float = 0.05) -> float:
        tb = self.burn_time(frac)
        return self.total_impulse / tb if tb > 0 else 0.0


def simulate(spec: MotorSpec, t_end: float = 10.0, max_step: float = 2e-3,
             a_burn_fn=None, p_cut_frac: float = 1e-3) -> BurnResult:
    """모터 연소를 시간 적분한다.

    a_burn_fn : callable(w[mm]) -> A_b[mm²]
        기본은 그레인 테이블. 테일오프 실험처럼 A_b 를 바꿔 끼울 때 쓴다.
    """
    grain, br = spec.grain, spec.burn_rate
    if a_burn_fn is None:
        w_tab = np.linspace(0.0, grain.web_max(), 800)
        ab_tab = grain.burn_area(w_tab)
        a_burn_fn = lambda w: np.interp(w, w_tab, ab_tab, left=ab_tab[0], right=0.0)

    w_tab2 = np.linspace(0.0, grain.web_max(), 800)
    v_tab = grain.volume(w_tab2)
    vol_fn = lambda w: np.interp(w, w_tab2, v_tab, left=v_tab[0], right=0.0)

    # r(P) 를 매 스텝 스플라인으로 평가하면 적분이 수십 초로 늘어난다.
    # **log 간격 격자** 위에 한 번 펼쳐두고 **P 에 대해 선형** 보간한다
    # (0.01 % 이내 일치). ⚠ 초판 주석은 "log P 격자 선형 보간" 이라 적어
    # 'log P 에 대해 선형' 으로 읽혔다 — 격자만 로그고 보간은 P 선형이다.
    #
    # 표의 양 끝을 **모델 자신의 유효 범위에 맞춘다.** `np.interp` 는 범위 밖에서
    # 끝값으로 평탄 외삽하는데, 초판은 하한을 1e-3 으로 하드코딩해 두고 그것이
    # 모델의 `p_min` 과 같은지 확인하지 않았다. 같으면 표 ≡ 모델이고, 다르면
    # 표만 조용히 다른 값을 준다. 명시적으로 묶고 `left`/`right` 도 적어 둔다.
    p_lo = float(getattr(br, "p_min", 1e-3))
    p_tab = np.geomspace(p_lo, 60.0, 4000)
    r_tab = np.asarray(br.rate(p_tab), dtype=float)
    rate_fast = lambda p: float(np.interp(p, p_tab, r_tab,
                                          left=r_tab[0], right=r_tab[-1]))

    rt = spec.rt
    at = spec.a_throat * MM2_TO_M2
    rho_p = grain.density * 1000.0                  # g/cc → kg/m³
    cstar = spec.c_star_actual
    rho_amb = spec.p_ambient * MPA_TO_PA / rt
    w_max = grain.web_max()

    p_floor_pa = spec.p_ambient * MPA_TO_PA

    def rhs(t, y):
        w, rho_c = y
        # ⚠ 챔버압의 하한은 **대기압**이지 1 Pa 가 아니다. 노즐로 대기와 통해
        #   있으므로 챔버가 대기압 아래로 내려갈 수 없다. 초판은 1 Pa 로 두어
        #   적분기가 아래로 흔들릴 때 r(P) 가 모델 하한(KNSB: 0.143 mm/s)으로
        #   평탄 외삽되는 구간이 생겼다 — "0 압력에 0 아닌 연소속도".
        #   이제 그 구간 자체가 없어진다.
        #   (모델의 저압 바닥 자체는 남아 있다 — KNSB 는 P ≤ p_min 에서 r 이
        #    상수다. 소화 한계를 모델링하지 않는다는 뜻이고, 그 가정은
        #    `burnrate.py` 에 적혀 있다. 🔜 실측으로 소화압을 잡으면 재검토.)
        p_pa = max(rho_c * rt, p_floor_pa)
        p_mpa = p_pa / MPA_TO_PA
        r = rate_fast(p_mpa)                               # mm/s
        a_b = float(np.asarray(a_burn_fn(w)).ravel()[0])   # mm²
        if w >= w_max:
            a_b = 0.0
        v_free = spec.chamber_volume - float(vol_fn(w))    # mm³
        v_free = max(v_free, 1.0) * MM3_TO_M3
        a_b_m2 = a_b * MM2_TO_M2
        r_ms = r * 1e-3
        mdot_gen = rho_p * a_b_m2 * r_ms                   # kg/s
        mdot_out = p_pa * at / cstar
        dv_free = a_b_m2 * r_ms                            # m³/s — 자유체적 증가
        return [r, (mdot_gen - mdot_out - rho_c * dv_free) / v_free]

    def burnt_out(t, y):
        # 추진제가 다 타고 압력이 대기압 근처로 내려오면 종료
        return (y[1] * rt / MPA_TO_PA) - (spec.p_ambient * (1.0 + p_cut_frac) + 1e-4) \
            if y[0] >= w_max else 1.0
    burnt_out.terminal = True
    burnt_out.direction = -1

    sol = solve_ivp(rhs, (0.0, t_end), [0.0, rho_amb], max_step=max_step,
                    rtol=1e-6, atol=[1e-9, 1e-9], dense_output=False,
                    events=burnt_out, method="LSODA")

    w = sol.y[0]
    p_pa = np.maximum(sol.y[1] * rt, 1.0)
    p_mpa = p_pa / MPA_TO_PA

    # ── 팽창비 ──────────────────────────────────────────────────────────
    # ε 는 압력에 전혀 영향을 주지 않는다(초킹 유량 ṁ = P·A_t/c* 에 ε 가 없다).
    # 추력 높이만 바꾸므로, 압력을 먼저 풀고 나서 정하면 된다.
    # 규약(2026-10-05): **작동 구간의 시간 평균 압력에 대한 최적 팽창비.**
    # 작동 구간은 §4.5 의 고전 기준(최대압의 10 % 교차)을 쓴다.
    eps_ratio, p_mean = spec.expansion_ratio, None
    if eps_ratio is None:
        band = p_mpa >= 0.1 * float(p_mpa.max())
        p_mean = (float(np.trapezoid(p_mpa[band], sol.t[band])
                        / max(sol.t[band][-1] - sol.t[band][0], 1e-12))
                  if band.sum() > 1 else float(p_mpa.max()))
        if p_mean <= spec.p_ambient:
            eps_ratio = 1.0
        else:
            eps_ratio = optimum_area_ratio(p_mean, spec.p_ambient, spec.gamma)
    pe_pc = area_ratio_to_pressure_ratio(eps_ratio, spec.gamma)
    a_b = np.array([float(np.asarray(a_burn_fn(wi)).ravel()[0]) if wi < w_max else 0.0
                    for wi in w])
    cf = np.array([thrust_coefficient(max(p, 1e-9), spec.p_ambient,
                                      eps_ratio, spec.gamma, pe_pc)
                   for p in p_mpa]) * spec.cf_eff
    thrust = np.maximum(cf * p_pa * at, 0.0)               # N
    mdot_out = p_pa * at / cstar

    # ── 챔버 충전 시상수 τ(t) [s] ──────────────────────────────────────────
    # 모듈 머리말과 §4.3 이 τ = c*·V_free/(R·T·A_t) 를 핵심 척도로 적어 두고도
    # 코드 어디서도 계산하지 않았다. 그래서 (a) 적분 스텝이 τ 대비 충분히 작은지,
    # (b) `rt` 규약(공표 c* 역산)이 바꾸는 τ 9.8 % 가 과도 구간에 얼마나
    # 영향을 주는지를 **진단할 수단이 없었다**. 값은 여기서 공짜로 나온다.
    v_free_t = np.maximum(spec.chamber_volume - vol_fn(w), 1.0) * MM3_TO_M3
    tau = cstar * v_free_t / (rt * at)
    return BurnResult(sol.t, p_mpa, thrust, w, a_b, mdot_out,
                      expansion_ratio=float(eps_ratio), p_mean=p_mean, tau=tau)


def quasi_steady_pressure(spec: MotorSpec, w) -> np.ndarray:
    """준정상 평형 압력 [MPa] — 비교·진단용.

    자유체적 증가분을 채우는 몫을 뺀 **유효 생성 유량**이 배출과 같아지는
    P 를 이분법으로 푼다.

        (ρ_p − ρ_gas) · A_b · r(P) = P · A_t / c*,    ρ_gas = P / (R·T)

    전 구간 연속 r(P) 를 쓰므로 근이 여러 개인 경우에도 단조 구간에서 잡힌다.
    """
    w = np.atleast_1d(np.asarray(w, dtype=float))
    grain, br = spec.grain, spec.burn_rate
    rho_p = grain.density * 1000.0
    at = spec.a_throat * MM2_TO_M2
    out = np.zeros_like(w)
    for i, wi in enumerate(w):
        a_b = float(grain.burn_area(wi)[0]) * MM2_TO_M2
        if a_b <= 0:
            out[i] = spec.p_ambient
            continue

        def g(p_mpa):
            r = float(br.rate(p_mpa)[0]) * 1e-3
            rho_gas = (p_mpa * MPA_TO_PA) / spec.rt
            return ((rho_p - rho_gas) * a_b * r
                    - (p_mpa * MPA_TO_PA) * at / spec.c_star_actual)

        # ⚠ 이분법은 **부호가 바뀌는 구간**에서만 근을 준다. 초판은 `g(lo) < 0`
        #   만 확인하고 `g(hi)` 는 보지 않아, 50 MPa 에서도 아직 g > 0 이면
        #   (= 평형압이 50 MPa 를 넘는 폭주 설계) 200 번 쪼갠 끝에 **50 MPa 를
        #   근이라고 반환**했다. 근이 아닌 값이 진단표에 그대로 실린다.
        lo, hi = 1e-3, 50.0
        if g(lo) < 0:
            out[i] = spec.p_ambient
            continue
        if g(hi) > 0:
            raise RuntimeError(
                f"준정상 평형압이 탐색 상한 {hi:g} MPa 를 넘는다 "
                f"(w = {wi:.3f} mm, A_b = {a_b / MM2_TO_M2:.0f} mm², "
                f"A_t = {spec.a_throat:.1f} mm²). 노즐목이 너무 작거나 "
                f"연소면적이 너무 큰 설계다 — 이 압력은 어떤 케이스도 견디지 "
                f"못하므로 근을 돌려주는 것이 의미가 없다.")
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if g(mid) > 0:
                lo = mid
            else:
                hi = mid
        out[i] = 0.5 * (lo + hi)
    return out
