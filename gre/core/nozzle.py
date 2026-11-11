"""노즐 — 추력계수와 팽창.

이상 기체·등엔트로피 팽창을 가정한다. 침식(삭마)은 범위 밖이며,
필요하면 목 면적을 시간에 따라 바꾸는 방식으로 나중에 얹는다.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq


#: **박리 문턱** `p_e / p_a` 의 기본값 — 이 아래로 과팽창하면 유동이 박리한다고 본다.
#:
#: 이상 등엔트로피 `C_f` 는 과팽창이 심해지면 **음수**로 간다. 가압된 챔버가 음의
#: 추력을 낼 수는 없으므로 그건 물리가 아니라 **식이 깨진 것**이다. 실제 노즐은
#: 출구압이 대기압보다 충분히 낮아지면 벽에서 유동이 떨어져(박리), 박리 지점
#: 하류는 추력에 기여하지 않는다. 그래서 손실에 물리적 상한이 생긴다.
#:
#: 이 값은 **0~1 사용자 입력**이고 (`DiffMotor.separation_frac`), 양 끝이 정확히
#: 두 극단 모델이다 — 수치로 확인했다:
#:
#:     frac = 0    "절대 박리 안 함"  → 이상 과팽창식 그대로 (가장 비관적)
#:     frac = 0.3  Kalt–Bendall
#:     frac = 0.4  Summerfield ← 기본값
#:     frac = 1    "p_e < p_a 즉시 박리" → 항상 최적팽창 C_f (가장 낙관적)
#:
#:                      p_c      frac=0    이상식      frac=1   최적팽창식
#:                    2.895      1.4710   1.4710       1.4954   1.4954
#:                    1.000      0.9404   0.9404       1.2728   1.2728
#:                    0.300     −0.9510  −0.9510       0.9068   0.9068
#:
#: ⚠ **박리가 손실을 없애는 것이 아니다.** 노즐이 아직 꽉 차서 흐르는 구간에서는
#:   박리 모델이 이상식과 **완전히 같다** (위 표 p_c = 2.895 · 2.0 에서 손실 100 %
#:   적용). 이상식이 애초에 유효하지 않은 아래쪽에서만 갈라진다.
#:
#: ⚠ **박리하면 ε 이 구별력을 잃는다.** 박리 지점 하류 노즐은 추력에 기여하지
#:   않으므로 `frac=0.3, p_c=1.0` 에서 ε=8 과 ε=16 이 둘 다 `C_f = 1.1099` 다.
#:   물리적으로 옳지만, 테일 구간에서 ε 민감도가 사라진다는 뜻이기도 하다.
#:
#: ══════════════════════════════════════════════════════════════════════
#: **기본값 0.4 의 근거 (2026-10-29 확정)**
#:
#: 이 값은 **측정하지 않고** 쓰고, 근거는 **Summerfield 박리 기준**(문헌값)이다.
#: 직접 측정은 **보류**다 — 이 값이 결론을
#: 가르는 자리에 오면 그때 꺼낸다.
#:
#: ⚠⚠ **"NAKKA 시트가 0.4 를 쓰므로" 는 근거가 아니다 — 시트를 확인했고,
#:    시트는 박리를 아예 보지 않는다 (2026-10-29, `SRM_2023.xls` 실물 확인).**
#:
#:      · 워크북 전체에 박리 관련 용어가 **0 건**
#:        (separation / detached / overexpanded / Summerfield / Kalt–Bendall)
#:      · `Performance` 표의 `Pe` 열이 **모든 행에서 대기압(101000 Pa) 고정**.
#:        `Ae/At = 8` 이 고정인데 출구압이 챔버압을 따라 변하지 않는다.
#:      · 시트의 `CF` 를 두 식과 대조하면 **최적팽창식과 소수점 4 자리까지
#:        일치**하고 ε=8 이상식과는 한 번도 맞지 않는다:
#:
#:            Po[MPa]   시트 CF   최적팽창   ε=8 이상
#:             1.937    1.4180    1.4180     1.3314
#:             2.186    1.4418    1.4418     1.3790
#:             2.230    1.4456    1.4456     1.3862
#:
#:      · 시트는 **실제 출구 지름(29.67 mm, ε=8)과 "최적" 출구 지름(22.17 mm)을
#:        둘 다 적어 놓고 성능은 최적 쪽으로 계산한다.**
#:
#:    즉 시트는 "이 노즐이 낼 추력" 을 예측하는 도구가 아니라 **"최적 팽창이면
#:    얼마 나오나" 를 보여주는 설계 도구**다. 우리 파라미터로 번역하면
#:    **`separation_frac = 1.0`** 이지 0.4 가 아니다.
#:
#:    **이 도구는 예측 도구이므로 그 가정을 따르지 않는다.** 이 모터는 ε=8 에
#:    최대압 2.9 MPa 라 연소 내내 과팽창인데, `frac = 1` 은 그 손실을 전혀 보지
#:    않고 ε 을 추력에 무의미하게 만든다.
#:
#: ⚠ **외부 대조가 `frac = 1` 에서 가장 잘 맞는 것(0.61 % vs 2.98 %)은 근거가
#:   아니다.** 사용자의 외부 시뮬레이터도 같은 가정을 쓰므로(아래 참조) 그
#:   일치는 **같은 모델링 선택끼리 맞는 것**이지 실측과 맞는 것이 아니다.
#:
#: ⚠ **사용자의 파이썬 포팅본은 시트를 충실히 재현한 것이다.**
#:   `SRM_simul.Cf_calculation` 이 `P_e` 자리에 `P_atm` 을 넣고 `ER` 을 쓰지 않는
#:   것은 **시트와 같은 거동**이다 (위 대조로 확인). 이것을 포팅 결함처럼
#:   서술했으나, 코드는 출처에 충실하고 **docstring 쪽이 시트보다 완전한 모델을
#:   적어 둔 것**이다. 코드 ≠ docstring 은 여전히 사실이다.
#: ══════════════════════════════════════════════════════════════════════
PE_PA_SEPARATION_DEFAULT: float = 0.4


def gamma_function(gamma: float) -> float:
    """Γ(γ) = √γ · (2/(γ+1))^((γ+1)/(2(γ−1)))  — 초킹 유량 계수."""
    g = float(gamma)
    return np.sqrt(g) * (2.0 / (g + 1.0)) ** ((g + 1.0) / (2.0 * (g - 1.0)))


def rt_from_cstar(c_star: float, gamma: float) -> float:
    """c* 와 γ 로부터 R·T [J/kg] 를 역산한다.

    c* = √(R T) / Γ  →  R T = (c* Γ)²
    챔버 충전 방정식에서 P = ρ R T 로 쓰기 위해 필요하다.
    """
    return (float(c_star) * gamma_function(gamma)) ** 2


def critical_pressure_ratio(gamma: float) -> float:
    """목(M=1)에서의 압력비 P*/P_c."""
    g = float(gamma)
    return (2.0 / (g + 1.0)) ** (g / (g - 1.0))


def area_ratio_to_pressure_ratio(eps: float, gamma: float) -> float:
    """팽창비 ε = A_e/A_t → 압력비 P_e/P_c (**초음속 해**).

    ε(x) 는 x = P*/P_c 에서 최소값 1 을 갖고 양쪽으로 발산한다. 초음속
    해는 x < P*/P_c 쪽 가지이므로 그 구간에서만 근을 찾는다.
    """
    g = float(gamma)
    G = gamma_function(g)
    if float(eps) < 1.0:
        raise ValueError("팽창비는 1 이상이어야 한다")

    def f(x):  # x = Pe/Pc
        return (G / np.sqrt(2 * g / (g - 1) * x ** (2 / g) * (1 - x ** ((g - 1) / g)))
                - float(eps))

    x_crit = critical_pressure_ratio(g)
    # ε = 1 이면 근이 정확히 임계점이다. 브래킷 오른쪽 끝이라 수치적으로
    # 잡히지 않으므로 바로 돌려준다.
    if abs(float(eps) - 1.0) < 1e-12:
        return float(x_crit)
    return float(brentq(f, 1e-12, x_crit * (1 - 1e-9)))


def thrust_coefficient(p_c: float, p_a: float, eps: float, gamma: float,
                       pe_pc: float = None,
                       separation_frac: float = 0.0) -> float:
    """추력계수 C_f. 압력 단위는 서로 같기만 하면 된다.

    `separation_frac` (0~1) 는 **박리 문턱** `p_e/p_a` 다. 꽉 찬 흐름의 출구압이
    `separation_frac · p_a` 아래로 내려가면 그 지점에서 박리했다고 보고, 유효
    팽창비를 `p_e = separation_frac · p_a` 가 되는 값으로 낮춰 계산한다.
    **0 이면 박리를 보지 않는다** — 기존 호출부는 전부 이 기본값이라 거동이
    바뀌지 않는다. 자세한 근거는 `PE_PA_SEPARATION_DEFAULT` 주석.
    """
    g = float(gamma)
    G = gamma_function(g)
    if pe_pc is None:
        pe_pc = area_ratio_to_pressure_ratio(eps, g)

    frac = float(separation_frac)
    if frac > 0.0:
        pc = max(float(p_c), 1e-12)
        if pe_pc * pc < frac * float(p_a):
            # 박리했다 — 유효 팽창비는 출구압이 `frac·p_a` 가 되는 ε.
            # `optimum_area_ratio(pc, frac·p_a, γ)` 가 바로 그 ε 다.
            if pc <= frac * float(p_a):
                return 0.0            # 챔버압이 박리압 이하 — 추력 없음
            eps_eff = optimum_area_ratio(pc, frac * float(p_a), g)
            x = frac * float(p_a) / pc
            cf_mom = G * np.sqrt(2 * g / (g - 1) * (1 - x ** ((g - 1) / g)))
            return float(cf_mom + (x - float(p_a) / pc) * eps_eff)

    cf_mom = G * np.sqrt(2 * g / (g - 1) * (1 - pe_pc ** ((g - 1) / g)))
    return float(cf_mom + (pe_pc - p_a / max(p_c, 1e-12)) * eps)


def optimum_area_ratio(p_c: float, p_a: float, gamma: float) -> float:
    """주어진 챔버압에서 대기압과 맞물리는(최적 팽창) 팽창비."""
    g = float(gamma)
    if float(p_c) <= float(p_a):
        raise ValueError("챔버압이 대기압보다 커야 최적 팽창비가 정의된다")
    x = float(p_a) / float(p_c)
    G = gamma_function(g)
    return float(G / np.sqrt(2 * g / (g - 1) * x ** (2 / g) * (1 - x ** ((g - 1) / g))))
