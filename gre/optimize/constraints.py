"""제조 제약 — 미분 가능한 벌점.

제약은 네 개의 **독립 파라미터**로 분리되어 있다. 파괴 모드가 서로 다르고,
한쪽 기준을 반대쪽에 적용하면 원하는 형상이 뭉개지기 때문이다.

| 파라미터 | 대상 | 파괴 모드 | 구현 |
|---|---|---|---|
| `w_web_min` | 포트 벽 ↔ 케이스 벽 추진제 두께 | 케이스 조기 노출 | 해석 (반경 푸리에에서 바로) |
| `t_prop_min` | 추진제 쪽 최소 살 두께 | 추진제 파단 → 조각 이탈 | 형태학적 열림 |
| `t_void_min` | 공극 쪽 최소 슬롯 폭 | 실리콘 코어 찢어짐 | 형태학적 열림 |
| `r_tip_min` | 볼록 모서리 반경 | 프린팅·몰드 해상도 | 해석 (곡률) |

**형태학적 열림(opening)** — §4.5 가 지정한 방법이다. 폭이 2ρ 보다
얇은 부분은 ρ 만큼 침식하면 사라지고, 다시 ρ 만큼 팽창해도 돌아오지 않는다.
따라서 `S − open(S, ρ)` 의 면적이 곧 "너무 얇은 부분의 양" 이고, 침식·팽창이
min/max 풀링이라 **그대로 미분 가능**하다. 완화된 지시함수에 적용하므로
경계에서도 경사가 끊기지 않는다.

⚠ 벌점은 "위반량" 이지 불리언이 아니다. 경사 기반 최적화기가 제약을 향해
   밀려가려면 얼마나 위반했는지가 연속적으로 보여야 한다.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as tfn

if TYPE_CHECKING:                       # 순환 참조 방지 — 실행 시에는 불러오지 않는다
    from ..core.diff_burnback import DiffSegment2D

#: 포트 최소 반경의 퇴화 방지 하한 [mm]. 제작 한계가 아니라 **기하 유효성**이다 —
#: r(θ) 가 0 에 닿으면 윤곽이 자기교차하고 거리장이 의미를 잃는다.
R_PORT_MIN: float = 0.5


#: 노즐목 직경의 물리적 하한 [mm]. 이보다 작은 목은 가공·연소 양쪽에서
#: 의미가 없다 (`optimize` 가 옛날에 `a_throat.clamp_(min=5.0 mm²)` 로
#: 쓰던 값과 같은 자리 — 면적 5 mm² 는 Ø2.52 mm 다).
D_THROAT_MIN_MM: float = 2.52


@dataclass(frozen=True)
class ThroatSpec:
    """노즐목 — **입력**이다.

    ## 왜 입력이어야 하는가

    `F = C_f · P_c · A_t` 는 **곱만 묶는다.** `A_t` 가 최적화 변수인 동안
    추력 곡선을 맞추는 방법이 무한히 많고, 그 중 하나를 고르는 기준이 추력
    사양에는 없다. 그래서 압력이 **추력 사양이 보지 않는 자유 방향**이 된다.

    그 결과를 쟀다 (`experiments/pressure_results.md`). 추력 3 항만
    사양으로 주면 최적화기가 목을 Ø10.7 → **Ø6.8 mm** (면적 90 → 36.06 mm²,
    0.40 배) 로 조여 압력을 **2.07 배**로 올리고도 "추력 10 % 안" 을 보고했다.
    최대압 오차 **106.87 %** — 합격 기준(추력 **및 압력** 10 %)을 97 %p 넘긴다.
    케이스압 하드 제약도 못 잡았다 (5.8 MPa < 한계 11 MPa).

    목이 입력이면 이 자유도가 사라진다. `A_t` 가 주어지면 추력을 맞추는 것이
    **압력을 맞추는 것과 같은 일**이 된다.

    ## 두 모드

        ThroatSpec.fixed(8.0)        보유 노즐 Ø8 mm 를 쓴다 — `A_t` 를 얼린다
        ThroatSpec.band(5.0, 10.0)   Ø5~10 mm 중에서 고른다 — 상자 제약

    **'자유' 모드는 없다.** 이전에는 `A_t` 가 그냥 최적화 변수였고 그것이
    위에 적은 압력 106.87 % 이탈의 원인이다. 그 동작을 `free()` 라는 이름으로
    남겨 두면 "선택지 하나" 로 보여 다시 쓰이게 되므로 삭제했다. 이전
    실험들의 숫자가 그 모드의 것이라는 사실은 `pressure_results.md` 에 적혀
    있고, 그것이 기록으로 충분하다.

    ## 범위 모드는 **연속**이다

    구간 안의 어떤 값이든 고를 수 있다 (상자 제약, 매 스텝 clamp). 가공 가능
    목록으로 스냅하지 않는다 — 스냅은 스냅 후 사양을 다시 판정해야 하고,
    그 판정 규칙이 아직 정해지지 않았다. 나온 목 직경이 가공하기 어려운
    수치일 수 있으므로 보고에 직경을 **항상 mm 로** 적는다.
    """

    #: 고정 모드의 목 직경 [mm]. 세 필드가 다 None 이면 거부된다 —
    #: 기본 생성 `ThroatSpec()` 이 조용히 '아무 목' 이 되지 않게 하기 위해서다.
    d_mm: float | None = None
    #: 범위 모드의 하한·상한 직경 [mm]
    d_min_mm: float | None = None
    d_max_mm: float | None = None

    # ── 생성자 — 모드를 이름으로 못질한다 (불리언 인자 조합보다 읽기 쉽다)
    @classmethod
    def fixed(cls, d_mm: float) -> "ThroatSpec":
        return cls(d_mm=float(d_mm))

    @classmethod
    def band(cls, d_min_mm: float, d_max_mm: float) -> "ThroatSpec":
        return cls(d_min_mm=float(d_min_mm), d_max_mm=float(d_max_mm))

    def __post_init__(self):
        if self.d_mm is None and self.d_min_mm is None and self.d_max_mm is None:
            raise ValueError("노즐목은 입력이다 — 고정 직경(`fixed`) 또는 "
                             "범위(`band`) 중 하나를 줘야 한다")
        if self.d_mm is not None:
            if self.d_min_mm is not None or self.d_max_mm is not None:
                raise ValueError("고정 직경과 범위를 같이 줄 수 없다")
            if not (self.d_mm >= D_THROAT_MIN_MM):
                raise ValueError(f"목 직경 {self.d_mm} mm 는 하한 "
                                 f"{D_THROAT_MIN_MM} mm 아래다")
        elif (self.d_min_mm is None) != (self.d_max_mm is None):
            raise ValueError("범위는 하한과 상한을 **둘 다** 줘야 한다 — "
                             "한쪽만 주면 반대쪽이 조용히 무제한이 된다")
        elif self.d_min_mm is not None:
            if not (self.d_min_mm >= D_THROAT_MIN_MM):
                raise ValueError(f"목 직경 하한 {self.d_min_mm} mm 는 물리 하한 "
                                 f"{D_THROAT_MIN_MM} mm 아래다")
            if not (self.d_max_mm > self.d_min_mm):
                raise ValueError(f"목 직경 범위가 뒤집혔다: "
                                 f"{self.d_min_mm} ~ {self.d_max_mm} mm")

    # ── 조회
    @property
    def mode(self) -> str:
        return "고정" if self.d_mm is not None else "범위"

    @property
    def is_fixed(self) -> bool:
        return self.d_mm is not None

    @property
    def area_mm2(self) -> float | None:
        """고정 모드의 목 면적 [mm²]. 다른 모드에서는 None."""
        if self.d_mm is None:
            return None
        return float(np.pi * (self.d_mm / 2.0) ** 2)

    @property
    def bounds_mm2(self) -> tuple:
        """`A_t` 를 가둘 상자 [mm²] — 두 모드 다 유한하다.

        ⚠ 고정 모드도 상자를 돌려준다 (lo == hi). 호출자가 모드별로 분기하지
           않고 그대로 clamp 할 수 있게 하기 위한 것이고, 얼린 변수에 clamp 를
           걸어도 값이 안 바뀌므로 안전하다.
        """
        if self.d_mm is not None:
            a = self.area_mm2
            return (a, a)
        return (float(np.pi * (self.d_min_mm / 2.0) ** 2),
                float(np.pi * (self.d_max_mm / 2.0) ** 2))

    def start_mm2(self) -> float:
        """최적화 출발값 [mm²]. 고정이면 그 값, 범위면 **직경의 중앙**."""
        if self.d_mm is not None:
            return self.area_mm2
        d_mid = 0.5 * (self.d_min_mm + self.d_max_mm)
        return float(np.pi * (d_mid / 2.0) ** 2)

    def describe(self) -> str:
        if self.d_mm is not None:
            return f"고정 Ø{self.d_mm:.2f} mm ({self.area_mm2:.2f} mm²)"
        lo, hi = self.bounds_mm2
        return (f"범위 Ø{self.d_min_mm:.2f}~{self.d_max_mm:.2f} mm "
                f"({lo:.2f}~{hi:.2f} mm²)")


@dataclass
class ManufacturingLimits:
    """제작 허용치.

    ⚠ **값마다 출처가 다르다.** 시험으로 확정된 것과 아직 임시값인 것을 섞어
       놓으면 나중에 무엇을 다시 재야 하는지 모르게 된다.

           t_void_min  시편 T1 **실측** — 아래 주석
           나머지      시편 시험 전 임시값
    """

    w_web_min: float = 3.0      # [mm] 임시 — 시편 시험 대상 아님
    #: [mm] 임시값. **측정하지 않기로 했다** — 시편 대신 V2 그레인
    #: 탈형 관찰로 본다. 실제 형상에서 보는 게 대표성이 높고 추가 용융이 없다.
    #: ⚠ 버리지 않고 **보수적 값으로 남겨 둔다**. 얇은 추진제는 케이스 쪽이 아니라
    #:   **로브 사이 쐐기 끝**(포트 최소반경 근처)에 생기고, K=3 판정이 걸려 있는
    #:   6-로브 깊음(amp 7.0)에서 얇은 추진제가 남는다. 얕은 형상(3-로브 2.2,
    #:   원형)에서는 0 이다. V2 에서 그 쐐기가 살아남는지가 이 값의 시험이다.
    #:
    #: ⚠⚠ **이 값은 격자에 양자화된다 — 분해능 2h.** `_thin_area` 의
    #:   구조요소가 `k = 2·⌈(t_prop_min/2)/h⌉ + 1` 이라, 같은 `k` 로 떨어지는
    #:   값들은 **구별되지 않는다**:
    #:
    #:       n_grid = 121 (h 0.50 mm) : 2.0·1.75·1.5·1.25 → k=5,  1.0·0.75·0.5 → k=3
    #:       n_grid = 241 (h 0.25 mm) : 2.0 → k9, 1.5·1.25 → k7, 1.0·0.75 → k5, 0.5 → k3
    #:
    #:   즉 **운용 격자에서 `t_prop_min` 의 분해능은 1.0 mm** 다. 사용자가
    #:   1.25 를 넣든 2.0 을 넣든 같은 판정이 나온다 (§4.6.1 에 이 한계를 적었다).
    #:   초판 주석의 "6-로브 깊음에서 **4.6 mm²**" 는 격자를 안 밝힌 값이었다 —
    #:   실측은 **n=121 에서 3.64, n=241 에서 4.82 mm²** 로 격자마다 다르다.
    t_prop_min: float = 2.0
    #: [mm] **T1 실측.** 0.5 mm 폭 지느러미가 FDM 0.4 마스터에서 찍히고,
    #: 실리콘이 차고, 찢김 없이 탈형됐다. 이 사슬에서 가장 좁은 목은 실리콘이
    #: 아니라 **프린터**다 — 0.4 노즐의 슬롯 하한이 0.5 mm 라 그보다 얇은 포트는
    #: 마스터부터 못 뽑는다. 그래서 0.5 는 "실리콘이 버티는 한계" 가 아니라
    #: "지금 장비로 만들 수 있는 한계" 이고, 더 얇은 쪽은 **미시험**이다.
    #: ⚠ T2(추진제에서 탈형)는 이 값을 **올릴 수만 있다** — 추진제는 달라붙고
    #:   부서진다. T2 전까지 이 값은 하한이지 확정이 아니다.
    t_void_min: float = 0.5
    #: [mm] 임시값. T3-a(설계→마스터→실리콘)는 시편으로 재지만, 그레인 단계는
    #: V2 로 옮겼다. 전제: 압축 성형의 압력이 형상을 고정한다.
    r_tip_min: float = 0.8
    slenderness_max: float = 15.0
    mass_max: float = 400.0     # [g]
    p_case_allow: float = 11.0  # [MPa] 하드 제약


# ════════════════════════════════════════════════════ 해석적 제약 (푸리에에서)
def _radius_and_derivs(r0, a, b, theta):
    """r(θ), r'(θ), r''(θ) — 모두 해석 미분."""
    k = torch.arange(1, a.shape[0] + 1, dtype=a.dtype, device=a.device)
    ang = k[:, None] * theta[None, :]
    c, s = torch.cos(ang), torch.sin(ang)
    r = r0 + (a[:, None] * c + b[:, None] * s).sum(0)
    rp = (k[:, None] * (-a[:, None] * s + b[:, None] * c)).sum(0)
    rpp = (k[:, None] ** 2 * (-a[:, None] * c - b[:, None] * s)).sum(0)
    return r, rp, rpp


def _st_max(x, beta: float):
    """값은 hard max, 경사는 soft-max (straight-through).

    soft-max 만 쓰면 값이 ln(N)/β 만큼 들뜬다. 실측에서 웹 두께가 0.084 mm
    얇게 보고되었고, 그 편향이 제약 판정에 그대로 들어간다. 9 장의 거리장과
    형태학 벌점에서 쓴 것과 같은 수법으로 분리한다.
    """
    soft = torch.logsumexp(beta * x, dim=0) / beta
    return x.max().detach() + (soft - soft.detach())


def _st_min(x, beta: float):
    return -_st_max(-x, beta)


def web_thickness(r0, a, b, r_outer: float, n_theta: int = 2048, beta: float = 50.0):
    """포트 벽과 케이스 벽 사이 최소 추진제 두께 [mm].

    = R_outer − max_θ r(θ). 어느 θ 가 최대인지 바뀌는 순간에도 경사가 끊기지
    않아야 한다 (스타 꼭짓점은 동점이 n 개라 특히 중요) → straight-through.
    """
    th = torch.linspace(0.0, 2 * np.pi, n_theta + 1, dtype=a.dtype, device=a.device)[:-1]
    r, _, _ = _radius_and_derivs(r0, a, b, th)
    return r_outer - _st_max(r, beta)


def min_port_radius(r0, a, b, n_theta: int = 2048, beta: float = 50.0):
    """포트 최소 반경 [mm] — 0 이하가 되면 윤곽이 퇴화한다 (하드 하한)."""
    th = torch.linspace(0.0, 2 * np.pi, n_theta + 1, dtype=a.dtype, device=a.device)[:-1]
    r, _, _ = _radius_and_derivs(r0, a, b, th)
    return _st_min(r, beta)


def tip_curvature_penalty(r0, a, b, r_tip_min: float, n_theta: int = 2048):
    """볼록 모서리 반경이 `r_tip_min` 미만인 구간에만 벌점.

    극좌표 곡률  κ = (r² + 2r'² − r·r'') / (r² + r'²)^{3/2}
    κ > 0 이 포트 쪽으로 볼록(= 추진제 쪽에서 보면 뾰족한 꼭짓점)이다.

    ⚠ 전 구간 평활화는 금지다. 스타 꼭짓점을 뭉개면 의도한
       형상이 나오지 않는다. 한계보다 뾰족한 곳만 벌한다.
    """
    th = torch.linspace(0.0, 2 * np.pi, n_theta + 1, dtype=a.dtype, device=a.device)[:-1]
    r, rp, rpp = _radius_and_derivs(r0, a, b, th)
    kappa = (r ** 2 + 2 * rp ** 2 - r * rpp) / (r ** 2 + rp ** 2) ** 1.5
    k_max = 1.0 / float(r_tip_min)
    over = torch.clamp(kappa - k_max, min=0.0)
    return (over ** 2).mean() * float(r_tip_min) ** 2           # 무차원화


def core_slenderness(r0, a, b, length: float, n_theta: int = 2048) -> float:
    """코어 핀의 세장비 L / d_eq — 보고용 진단값 (경사 없음).

    코어 핀(실리콘 음형)은 포트 단면을 세그먼트 길이만큼 뽑은 기둥이다. 너무
    가늘고 길면 캐스팅 중 휘어 포트가 한쪽으로 쏠린다. `slenderness_max` 는
    `ManufacturingLimits` 에 **정의만 되고 어디서도 쓰이지 않았다** — 미구현
    제약인지 죽은 설정인지 코드만 봐서는 알 수 없었다.

        A_port = ∮ r²/2 dθ = π(r₀² + ½Σ(a_k² + b_k²))      (해석)
        d_eq   = 2√(A_port/π)                              등가 지름
        세장비  = L / d_eq

    ⚠ **등가 지름은 스타 코어의 세장비를 과소평가한다.** 실제로는 가장 얇은
      날개가 먼저 휘므로 단면적이 아니라 최소 폭이 기준이어야 한다. 그 최소
      폭은 `t_void_min` 형태학 벌점이 이미 따로 보고 있고, 휨 한계를 폭으로
      쓰려면 실리콘 탄성률이 필요하다. 그래서 지금은 **등가 지름 기준으로
      보고만 하고 하드 게이트로 쓰지 않는다** — 제작 상수 4개와 같은 상태다.
      🔜 시편 시험에서 휨량을 재면 기준을 확정한다.
    """
    with torch.no_grad():
        a_port = float(np.pi * (float(r0) ** 2
                                + 0.5 * float((a ** 2 + b ** 2).sum())))
        d_eq = 2.0 * np.sqrt(max(a_port, 1e-12) / np.pi)
    return float(length) / max(d_eq, 1e-9)


#: [mm] 실리콘 충전 한계 — **`깊이² / 폭`** 의 상한. 시편 18 개 실측.
#:
#: 깊이도 세장비도 **단독으로는 데이터를 분리하지 못한다.** 세 후보를 다 재 봤다.
#:
#:     기준            찬 것 최대   안 찬 것 최소   분리
#:     세장비 d/w          20.0        20.0       ✗ (B1 찼고 C2 안 찼다)
#:     깊이 d              20.0        20.0       ✗ (C2 안 찼고 C3 찼다)
#:     **깊이²/폭**       **266.7**   **400.0**   ✔ 여유 **1.50 배**
#:
#: 관측 (18 개 전부):
#:
#:     행   폭 [mm]      깊이   깊이²/폭      결과
#:     A    0.5 ~ 4.0     5     6.2 ~  50    전부 찼다
#:     B    0.5 ~ 4.0    10    25.0 ~ 200    전부 찼다
#:     C    1.5 ~ 4.0    20   100.0 ~ 267    전부 찼다  ← C3 이 267 로 최대
#:     C2   1.0          20          400     **안 찼다** (왼쪽 꼭짓점 미충전)
#:     C1   0.5          20          800     **안 찼다** (10 mm 정도만 형성)
#:
#: 한계는 **찬 것의 최대**를 쓴다 (266.7 → 267). 참 경계는 267 과 400 사이
#: 어딘가이고 그 구간은 **미시험**이다.
#:
#: ⚠⚠ **두 번 고쳤다.** 이 값의 역사가 결함의 역사다.
#:     ① 이전 판은 `FILL_ASPECT_MAX = 20` 하나만 두고 **C2 를 "세장비 20 의
#:        근거" 로 세었다.** C2 는 그 세장비에서의 **실패 사례**다 — 한계값을
#:        실패가 관찰된 지점에 놓고 있었다.
#:     ② 그래서 처음엔 "깊이 ≤ 10 mm 이고 세장비 ≤ 20" 이라는 곱집합으로
#:        고쳤는데, 그러면 **C3~C6 이 거부된다** — 깊이 20 mm 에서 멀쩡히 찬
#:        시편들이다. 깊이 단독 기준도 틀렸다.
#:     위 표를 다 재고 나서야 `깊이²/폭` 이 나왔다. 교훈: 한계를 정하기 전에
#:     **찬 것과 안 찬 것을 모두** 분리하는지 확인할 것.
#:
#: ⚠ FDM 0.4 마스터 + Mold Max 30 + 중력 주입 조건의 값이다. 공정이 바뀌면
#:   다시 재야 한다 (`data/coupon/conditions.md`).
FILL_SEVERITY_MAX: float = 267.0

#: [-] 참고용 세장비 상한 — **게이트가 아니다.** 찬 시편의 최대 세장비가 20
#: 이지만 같은 20 에서 C2 가 안 찼으므로 이 값만으로 판정하면 틀린다.
#: `export/coupon.py` 의 시편 보고가 "경계" 표시에 쓴다.
FILL_ASPECT_MAX: float = 20.0


def lobe_geometry(r0, a, b, n_theta: int = 2048) -> dict:
    """포트 로브의 **깊이와 폭** — 실리콘 충전 한계 판정용 (경사 없음).

    시편이 잰 것과 **같은 양**을 재야 한다. 시편의 지느러미는 마스터의 슬롯에
    실리콘이 들어가 생긴 기둥이고, 실제 그레인에서 그 기둥에 해당하는 것은
    **포트의 로브**다 (실리콘 코어의 날개 = 포트의 오목한 골).

        깊이  로브 봉우리 r_peak − 이웃 골 r_valley 중 깊은 쪽
        폭    반(半)깊이 높이에서의 **호 길이** — 지느러미를 중간 높이에서
              캘리퍼스로 재는 것과 같은 규약
        세장비 깊이 / 폭

    ⚠ `core_slenderness` 와 **다른 양이다.** 그쪽은 코어 핀 **전체**의
      `L/d_eq` (축방향 세장비, 캐스팅 중 휨 판정)이고, 이쪽은 단면 안에서 로브
      하나의 깊이/폭 (실리콘 충전 판정)이다. 전면 검토가 둘을 같은 것으로 적어
      둔 것을 지적했다 — 이름이 비슷해서 생긴 혼동이다.

    반환: `{"depth", "width", "aspect", "n_lobes"}`. 로브가 없으면(원형 포트)
    깊이 0, 폭 inf, 세장비 0 — 원형은 항상 채워진다.
    """
    with torch.no_grad():
        th = torch.linspace(0.0, 2 * np.pi, n_theta + 1,
                            dtype=a.dtype, device=a.device)[:-1]
        r, _, _ = _radius_and_derivs(r0, a, b, th)
        rv = r.detach().cpu().numpy()
        n = rv.size
        prev, nxt = np.roll(rv, 1), np.roll(rv, -1)
        peaks = np.flatnonzero((rv >= prev) & (rv > nxt))
        valleys = np.flatnonzero((rv <= prev) & (rv < nxt))
        if peaks.size == 0 or valleys.size == 0:
            return {"depth": 0.0, "width": float("inf"), "aspect": 0.0,
                    "severity": 0.0, "n_lobes": 0}

        worst = {"depth": 0.0, "width": float("inf"), "aspect": 0.0,
                 "severity": 0.0, "n_lobes": int(peaks.size)}
        dth = 2 * np.pi / n
        for ip in peaks:
            # 이웃 골 두 개 (원형 배열이므로 감싸서 찾는다)
            d_fwd = (valleys - ip) % n
            d_bwd = (ip - valleys) % n
            v_r = valleys[int(np.argmin(np.where(d_fwd > 0, d_fwd, n)))]
            v_l = valleys[int(np.argmin(np.where(d_bwd > 0, d_bwd, n)))]
            r_peak = float(rv[ip])
            r_val = float(min(rv[v_l], rv[v_r]))      # 깊은 쪽 골
            depth = r_peak - r_val
            if depth <= 1e-9:
                continue
            # 반깊이 높이에서의 폭 — 봉우리 양쪽으로 교차점을 찾는다
            half = r_peak - 0.5 * depth
            k_r = 0
            while k_r < n and rv[(ip + k_r) % n] > half:
                k_r += 1
            k_l = 0
            while k_l < n and rv[(ip - k_l) % n] > half:
                k_l += 1
            r_mid = 0.5 * (r_peak + r_val)
            width = (k_l + k_r) * dth * r_mid          # 호 길이 [mm]
            aspect = depth / max(width, 1e-9)
            severity = depth * depth / max(width, 1e-9)    # 깊이²/폭 [mm]
            if severity > worst["severity"]:
                worst.update(depth=depth, width=float(width),
                             aspect=float(aspect), severity=float(severity))
        return worst


def fill_feasible(r0, a, b, limits=None, n_theta: int = 2048) -> dict:
    """포트 로브가 실리콘으로 **채워지는가** — 시편 기준 `깊이²/폭`.

    반환: `{"ok", "depth", "width", "aspect", "severity", "over_pp", "n_lobes"}`.
    `over_pp` 는 한계를 넘은 양 [%p] 이라 보고서에 "몇 %p 모자랐다" 를 그대로
    쓸 수 있다 (사양 여유와 같은 단위).

    ⚠ **현 설계 족에서는 이 게이트가 사실상 걸리지 않는다.** 푸리에 로브는
       넓다 — `r₀ + A·cos(kθ)` 의 로브는 깊이 `2A`, 반깊이 폭이 `≈ πr/k` 라
       `깊이²/폭` 이 수십 mm 수준이고 한계 267 과 한 자리 이상 떨어져 있다.
       (6-로브 깊음 amp 7.0: 깊이 14.0, 폭 7.39 → **26.5**.)
       시편이 시험한 기하(폭 0.5~4 mm 의 직사각 슬롯)가 설계 공간이 만드는
       기하보다 훨씬 극단적이었기 때문이다.

       그래도 게이트를 둔다. 이유 둘 — ① 모드 수 `K` 를 크게 올리면 로브가
       좁아지므로 언젠가 걸린다 (`k` 가 커지면 폭 ∝ 1/k, severity ∝ k).
       ② 시편 결과가 **도구 어디에도 연결되지 않은 상태**였고, 그게
       전면 검토의 지적이었다. 안 걸리는 게이트와 없는 게이트는 다르다.

       실제로 포트를 묶는 것은 `t_void_min`(포트 최소 틈) 쪽이고 그건 이미
       형태학 벌점으로 게이트되어 있다.
    """
    g = lobe_geometry(r0, a, b, n_theta=n_theta)
    over = 100.0 * (g["severity"] / FILL_SEVERITY_MAX - 1.0)
    return {"ok": g["severity"] <= FILL_SEVERITY_MAX,
            "depth": g["depth"], "width": g["width"], "aspect": g["aspect"],
            "severity": g["severity"], "n_lobes": g["n_lobes"],
            "over_pp": max(over, 0.0)}


def min_tip_radius(r0, a, b, n_theta: int = 2048) -> float:
    """가장 뾰족한 볼록 모서리의 반경 [mm] — 보고용 진단값 (경사 없음)."""
    with torch.no_grad():
        # ⚠ `device=a.device` 가 빠져 있었다 — 이 모듈에서 유일하게. CUDA 텐서를
        #   넘기면 `_radius_and_derivs` 안에서 device 불일치로 터진다.
        th = torch.linspace(0.0, 2 * np.pi, n_theta + 1,
                            dtype=a.dtype, device=a.device)[:-1]
        r, rp, rpp = _radius_and_derivs(r0, a, b, th)
        kappa = (r ** 2 + 2 * rp ** 2 - r * rpp) / (r ** 2 + rp ** 2) ** 1.5
        k = float(kappa.max())
    return float("inf") if k <= 0 else 1.0 / k


# ══════════════════════════════════════════════════ 형태학적 열림 (거리장에서)
def _to_image(ds, values):
    """선택된 셀 값을 (1,1,n,n) 이미지로 되돌린다."""
    img = torch.zeros(ds.n * ds.n, dtype=values.dtype, device=values.device)
    img = img.masked_scatter(ds.sel, values)
    return img.reshape(1, 1, ds.n, ds.n)


_DISK_CACHE: dict = {}


def _disk(k: int, dtype, device):
    """반지름 k//2 의 원판 구조요소 — (1, k*k, 1, 1) 불리언 가중치.

    ⚠ 정사각 커널을 쓰면 안 된다. 정사각 격자 + 정사각 커널은 4중 대칭이라
       k = 4 모드에 가짜 경사를 만든다 (측정값: 원형 포트에서 ∂/∂a₄ 가 7.27 —
       같은 자리의 곡선 손실 경사 3.4e-5 의 20 만 배). 최적화기가 목표와
       무관하게 4갈래 형상으로 끌려간다.
    """
    key = (k, dtype, device)
    if key not in _DISK_CACHE:
        r = k // 2
        y, x = torch.meshgrid(torch.arange(k) - r, torch.arange(k) - r, indexing="ij")
        _DISK_CACHE[key] = ((x ** 2 + y ** 2) <= r * r).reshape(-1).to(device)
    return _DISK_CACHE[key]


def _erode_disk(img, k: int):
    pad = k // 2
    patches = tfn.unfold(img, kernel_size=k, padding=pad)      # (1, k*k, L)
    mask = _disk(k, img.dtype, img.device)
    big = torch.where(mask[None, :, None], patches,
                      torch.full_like(patches, float("inf")))
    return big.min(dim=1).values.reshape(1, 1, img.shape[-2], img.shape[-1])


def _dilate_disk(img, k: int):
    pad = k // 2
    patches = tfn.unfold(img, kernel_size=k, padding=pad)
    mask = _disk(k, img.dtype, img.device)
    big = torch.where(mask[None, :, None], patches,
                      torch.full_like(patches, float("-inf")))
    return big.max(dim=1).values.reshape(1, 1, img.shape[-2], img.shape[-1])


def _open(img, k: int):
    """열림 = 침식 후 팽창. 구조요소는 원판."""
    return _dilate_disk(_erode_disk(img, k), k)


def _thin_area(ds, indicator, rho: float, min_cells: int = 12):
    """폭이 2ρ 보다 얇은 곳에 남아 있는 재료의 면적 [mm²].

    두 단계로 나눈다.

    1. **어디가 얇은가** — 지시함수를 0.5 에서 이진화해 원판 구조요소로 열고,
       `hard − open(hard)` 로 '너무 얇은 셀' 마스크를 만든다. 이 마스크는
       `detach` 한다.
    2. **얼마나 있는가** — 그 마스크 위에서 **완화된** 지시함수를 적분한다.
       경사는 여기로만 흐른다.

    ⚠ 이렇게 나누지 않으면 두 가지가 샌다.
       (a) min/max 풀링의 경사는 승자독식이라 창마다 한 셀로만 흐르고,
           얇은 곳이 하나도 없는 형상에서도 0 이 아닌 경사를 남긴다.
       (b) 완화 폭 ε 짜리 경계 링 자체가 '얇은 것' 으로 집계된다. 원형
           포트에서 공극 쪽 0.126 mm² 가 그 값이었고, 경사도 2.82 였다.
       이진 마스크를 detach 하면 둘 다 사라진다 — 제약이 **실제로 걸렸을
       때만** 값과 경사가 생긴다. 9 장의 straight-through 와 같은 수법이다.

    ⚠ **격자 바닥(min_cells)이 필요하다.** 이진 마스크를 detach 해도 (b) 가 완전히
       사라지지는 않는다. 원판을 정사각 격자에 올리면 경계가 계단이 되고, 그 계단이
       반경에 따라 열림 연산에서 몇 셀 남는다. 깨끗한 원형 포트 61개를 쓸어 잰 값:

           n_grid=121 (셀 0.250 mm²) : 61개 중 7개가 0 아님, 최대 1.281 mm² (셀 5.1개분)
           n_grid=161 (셀 0.141 mm²) : 61개 중 6개가 0 아님, 최대 0.708 mm² (셀 5.0개분)
           n_grid=241 (셀 0.062 mm²) : 61개 중 15개가 0 아님, 최대 0.628 mm² (셀 10.0개분)

       **셀 수로 보면 한 자릿수**이고 격자를 촘촘히 해도 그대로다 — 물리가 아니라
       이산화다. 그래서 바닥을 면적이 아니라 **셀 개수**로 둔다. 바닥 아래면 값과
       경사를 모두 0 으로 만든다 (마스크가 이미 detach 된 계단이라 불연속이 새로
       생기지는 않는다).

       이것을 안 두면 깨끗한 원형 설계가 "제작 제약 위반" 으로 판정되고(실측:
       r0 = 12.30 에서 벌점 1.31e-3), 그 벌점이 가짜 경사로 최적화기를 민다.

       바닥 12 셀에서 n_grid = 121(운용 설정)의 61개는 전부 0 이 된다. n_grid = 241
       에서는 1개가 0.628 mm² 로 남는다 — 더 올리면 n_grid = 121 에서 진짜 위반까지
       가릴 수 있어 여기서 멈춘다. 61개 표본으로 더 맞추는 것은 과적합이다.
    """
    # ⚠ **구조요소 반경이 셀 단위로 올림되므로 `rho` 는 격자에 양자화된다.**
    #   `ceil` 때문에 `rho ∈ (0, h]` 가 전부 k=3, `(h, 2h]` 가 전부 k=5 … 로
    #   뭉친다. 즉 이 함수가 구별할 수 있는 두께 분해능은 **2h** 다 (허용치가
    #   `2·rho` 이므로). 운용 격자 n_grid=121 (h=0.5 mm) 에서 1.0 mm —
    #   `t_prop_min` 1.25 와 2.0 이 같은 판정을 낸다. 호출자(제작 한계)의
    #   주석에 이 한계를 적어 두었다. 더 가는 분해능이 필요하면 격자를 조여야
    #   하고, 그 비용은 `samples_for_grid` 의 자동 상향까지 따라온다.
    k = 2 * int(np.ceil(rho / ds.h)) + 1
    img = _to_image(ds, indicator)
    with torch.no_grad():
        hard = (img > 0.5).to(img.dtype)
        mask = torch.clamp(hard - _open(hard, k), min=0.0)
        if float(mask.sum()) < float(min_cells):          # 격자 분해능 아래 — 무시
            mask = torch.zeros_like(mask)                 # 그래프는 유지 (경사도 0)
    return (img * mask).sum() * ds.cell_area


def thin_feature_penalty(ds, phi, limits: ManufacturingLimits):
    """추진제 쪽·공극 쪽 얇은 부분의 면적 [mm²] 두 개.

    같은 거리장에서 지시함수만 뒤집어 양쪽을 본다. 재료가 달라(추진제 vs
    실리콘) 허용치가 다르므로 **반드시 따로** 계산한다.
    """
    prop = torch.clamp(phi / ds.eps + 0.5, 0.0, 1.0) * ds.w_case     # 추진제 쪽
    void = ds.w_case - prop                                          # 공극 쪽
    return (_thin_area(ds, prop, limits.t_prop_min / 2.0),
            _thin_area(ds, void, limits.t_void_min / 2.0))


def connected_components(ds: "DiffSegment2D", phi, min_cells: int = 12) -> int:
    """추진제 영역의 연결 요소 개수 — 진단값 (경사 없음).

    2 이상이면 조각이 떨어져 나간다. 노즐 막힘·예측 밖 연소면의 원인이라
    하드 제약이지만, 정수라 미분할 수 없다. 얇은 부분 벌점이 간접적으로
    막아 주고, 최종 판정은 이 값으로 한다.

    ⚠ **최소 면적 필터가 필요하다.** 완화 폭 ε 짜리 경계 링을 0.5 에서 자르면
      격자 계단 때문에 한두 셀짜리 파편이 떨어져 나올 수 있고, 초판은 그것을
      '조각' 으로 세어 멀쩡한 설계를 연결성 위반으로 판정했다. `_thin_area` 의
      격자 바닥과 **같은 계열의 문제이고 같은 처방**이다 — 면적이 아니라 셀
      개수로 바닥을 둔다 (기본 12 셀, `_thin_area` 와 같은 값).

      실제로 떨어져 나온 조각은 이 바닥보다 훨씬 크다 — 추진제 조각이 12 셀
      (n_grid = 121 에서 3 mm²) 미만이면 그것이 노즐을 막을 일도 없다.
    """
    from scipy import ndimage
    with torch.no_grad():
        img = _to_image(ds, (phi > 0).to(phi.dtype) * (ds.w_case > 0.5).to(phi.dtype))
        mask = img[0, 0].cpu().numpy() > 0.5
    lab, n = ndimage.label(mask)
    if n == 0:
        return 0
    # ⚠ `n == 1` 도 지름길로 빠지지 않고 같은 필터를 지난다. 특수 경우를 두면
    #   "조각 하나는 크기와 무관하게 통과" 라는 숨은 규칙이 생긴다.
    sizes = np.bincount(lab.ravel())[1:]              # 배경(0) 제외
    return int((sizes >= int(min_cells)).sum())


# ═══════════════════════════════════════════════════════════════ 묶음
def manufacturing_penalty(ds: "DiffSegment2D", phi, r0, a, b,
                          limits: ManufacturingLimits, r_outer: float,
                          length: float = None):
    """제조 제약 벌점 합과 항목별 내역.

    각 항은 **위반량의 제곱**이고 길이 단위로 무차원화되어 있어, 가중치
    하나로 함께 다룰 수 있다.

    `ds` 는 `gre.core.diff_burnback.DiffSegment2D` (torch 판)다. numpy 판
    `SegmentBurnback2D` 를 넘기면 `_to_image` 안에서 원인을 알기 어려운 오류가
    난다 — 초판에는 타입 주석조차 없었다. 아래에서 미리 확인한다.
    """
    for attr in ("sel", "w_case", "n", "cell_area", "h", "eps", "r_outer"):
        if not hasattr(ds, attr):
            raise TypeError(
                f"`ds` 는 torch 판 DiffSegment2D 여야 한다 — '{attr}' 가 없다 "
                f"(받은 것: {type(ds).__name__}). numpy 판 SegmentBurnback2D 는 "
                f"경사를 흘리지 못하므로 여기서 쓸 수 없다.")
    web = web_thickness(r0, a, b, r_outer)
    r_min = min_port_radius(r0, a, b)
    thin_p, thin_v = thin_feature_penalty(ds, phi, limits)
    tip = tip_curvature_penalty(r0, a, b, limits.r_tip_min)

    pen_web = torch.clamp(limits.w_web_min - web, min=0.0) ** 2 / limits.w_web_min ** 2
    # ⚠ 이 항만 무차원화가 빠져 있었다 (mm² 단위). 다른 항은 전부 상대 위반량의
    #   제곱인데 이것만 절대 길이라, 같은 가중치 아래에서 **다른 척도로** 더해졌다.
    #   `pen_web` 과 같은 꼴로 맞춘다 — 한계값 제곱으로 나눈다.
    pen_pos = torch.clamp(R_PORT_MIN - r_min, min=0.0) ** 2 / R_PORT_MIN ** 2
    scale = float(ds.r_outer) ** 2
    terms = {
        "web": pen_web,
        "port_positive": pen_pos,
        "thin_propellant": thin_p / scale,
        "thin_void": thin_v / scale,
        "tip_radius": tip,
    }
    report = {"web_mm": float(web.detach()),
              "r_port_min_mm": float(r_min.detach()),
              "thin_prop_mm2": float(thin_p.detach()),
              "thin_void_mm2": float(thin_v.detach()),
              "r_tip_mm": min_tip_radius(r0, a, b)}
    if length is not None:                       # 세장비는 길이를 알아야 나온다
        report["slenderness"] = core_slenderness(r0, a, b, length)
        report["slenderness_max"] = float(limits.slenderness_max)
    return sum(terms.values()), terms, report
