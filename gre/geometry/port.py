"""포트 윤곽 표현 — 1단계: 반경 푸리에.

r(θ) = r0 + Σ_k [a_k cos(kθ) + b_k sin(kθ)]

이 표현은 중심에서 뻗은 어떤 반직선도 윤곽을 한 번만 지나므로
자기교차가 구조적으로 불가능하다 (r(θ) > 0 이기만 하면 된다).
"""
from __future__ import annotations

import numpy as np


class FourierPort:
    """반경 푸리에로 표현한 단일 중심 포트 윤곽.

    Parameters
    ----------
    r0 : float
        평균 반경 [mm]
    a, b : array_like
        cos·sin 계수 [mm]. 길이 K (k = 1..K)
    """

    def __init__(self, r0: float, a=None, b=None):
        self.r0 = float(r0)
        self.a = np.atleast_1d(np.asarray(a if a is not None else [], dtype=float))
        self.b = np.atleast_1d(np.asarray(b if b is not None else [], dtype=float))
        if self.a.size != self.b.size:
            raise ValueError("a 와 b 의 길이가 같아야 한다")

    # ---------------------------------------------------------------- 기본량
    @property
    def n_modes(self) -> int:
        return int(self.a.size)

    def radius(self, theta):
        """θ 에서의 반경 [mm]."""
        theta = np.asarray(theta, dtype=float)
        r = np.full(theta.shape, self.r0, dtype=float)
        for k in range(1, self.n_modes + 1):
            r += self.a[k - 1] * np.cos(k * theta) + self.b[k - 1] * np.sin(k * theta)
        return r

    def min_radius(self, n: int = 4096) -> float:
        return float(np.min(self.radius(np.linspace(0.0, 2 * np.pi, n, endpoint=False))))

    def max_radius(self, n: int = 4096) -> float:
        return float(np.max(self.radius(np.linspace(0.0, 2 * np.pi, n, endpoint=False))))

    def is_valid(self, eps: float = 1e-9) -> bool:
        """반경이 어디서도 0 이하가 되지 않는가."""
        return self.min_radius() > eps

    # ---------------------------------------------------------------- 샘플링
    def sample(self, spacing: float) -> np.ndarray:
        """윤곽을 대략 `spacing` [mm] 간격으로 샘플링한 점 (N, 2).

        거리장 계산의 기준점이 된다. 간격이 격자 간격보다 충분히 작아야
        '폴리라인까지의 거리'와 '샘플점까지의 거리'의 차이가 무시된다.
        """
        # 1차: 균일 θ 로 둘레 추정 → 필요한 점 수 산출
        th = np.linspace(0.0, 2 * np.pi, 2048, endpoint=False)
        r = self.radius(th)
        xy = np.column_stack([r * np.cos(th), r * np.sin(th)])
        seg = np.linalg.norm(np.diff(np.vstack([xy, xy[:1]]), axis=0), axis=1)
        perimeter = float(seg.sum())
        n = max(256, int(np.ceil(perimeter / max(spacing, 1e-6))))
        th2 = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
        r2 = self.radius(th2)
        return np.column_stack([r2 * np.cos(th2), r2 * np.sin(th2)])

    def area(self, n: int = 20000) -> float:
        """포트 단면적 [mm²] — ∮ r²/2 dθ.

        **주기 함수는 균일 격자 합이 곧 사다리꼴이다.** 끝점이 처음점과 같으므로
        모든 표본이 가중치 dθ 를 똑같이 받는다 (사다리꼴의 1/2 가중치가 양 끝에서
        합쳐져 1 이 된다). 이 경우 사다리꼴은 지수적으로 정확하다.

        ⚠ 초판은 `np.trapezoid(...) + r[0]²/2·dθ` 였다. `trapezoid` 가 이미 `r[0]`
          에 dθ/2 를 주는데 닫힘 항으로 dθ 를 또 더해 **r[0] 만 1.5·dθ** 를 받았고,
          마지막 표본 `r[n−1]` 은 dθ/2 밖에 못 받았다.

          **원형에서는 드러나지 않는다** — r 이 상수면 두 식이 항등적으로 같다.
          비원형에서 r = 12 + 3.5·cos 3θ (해석해 π(r₀² + a²/2) = 471.631597 mm²)
          로 재면:

              n       초판 상대오차   수정판
              16      1.30e-02      0
              64      2.42e-04      0
              512     4.78e-07      0
              20000   8.03e-12      0

          O(dθ²) 로 줄어들긴 하므로 운용 기본값 n = 20000 에서는 무해하다. 다만
          수정판은 **표본 수와 무관하게 기계 정밀도**다 (주기 사다리꼴의 지수
          수렴). 포트 면적은 식별성 판정과 노즐목/포트 비 제약에 쓰이므로
          공짜로 얻는 정확도를 버릴 이유가 없다.
        """
        th = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
        r = self.radius(th)
        return float(np.sum(r ** 2 / 2.0) * (2 * np.pi / n))

    def __repr__(self) -> str:
        return f"FourierPort(r0={self.r0:.4g}, modes={self.n_modes})"


def circular_port(radius: float) -> FourierPort:
    """원형 포트 (BATES)."""
    return FourierPort(radius)


def star_port(r_mean: float, n_points: int, amplitude: float) -> FourierPort:
    """단순 스타 — k = n_points 모드 하나만 준 형상."""
    a = np.zeros(n_points)
    a[n_points - 1] = amplitude
    return FourierPort(r_mean, a=a, b=np.zeros(n_points))
