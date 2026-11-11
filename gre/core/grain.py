"""그레인 = 세그먼트의 적층.

각 세그먼트는 단면이 일정하므로 §4.2 의 해석적 분리가 그대로 쓰인다.
전체 그레인의 A_b(w), V_p(w) 는 세그먼트 합이다.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .burnback2d import SegmentBurnback2D


@dataclass
class Segment:
    burnback: SegmentBurnback2D
    length: float               # [mm] 초기 길이
    ends_burn: bool = True      # 인히비터 미사용이면 True (§5.1)


@dataclass
class Grain:
    segments: list = field(default_factory=list)
    density: float = 1.841 * 0.92   # [g/cc] 설계 밀도 = 이론 × 밀도비

    def __post_init__(self):
        """세그먼트가 같은 케이스에 들어가는지 확인한다.

        ⚠ 초판은 검사하지 않았다. `d_outer` 가 다른 세그먼트를 쌓아도 조용히
          통과하고, `A_b` 와 체적은 각자의 바깥지름으로 계산되어 **물리적으로
          존재할 수 없는 그레인**이 답으로 나온다. 한 케이스 안의 적층이라는
          전제가 깨지는데 어디서도 드러나지 않는 것이 문제다.
        """
        outs = {round(float(s.burnback.d_outer), 9) for s in self.segments}
        if len(outs) > 1:
            raise ValueError(
                f"세그먼트의 그레인 바깥지름이 서로 다르다: {sorted(outs)} mm. "
                f"한 케이스 안의 적층이므로 모두 같아야 한다.")

    # ------------------------------------------------------------- 테이블화
    @staticmethod
    def segment_web(seg: Segment) -> float:
        """세그먼트가 소진되는 웹 [mm].

        끝면이 타면 두 가지로 소진될 수 있고, **먼저 오는 쪽**이 수명이다.
          - 반경 방향: 웹을 다 태움          w = w_max
          - 축방향  : 양쪽 끝면이 만남       L - 2w = 0  →  w = L/2
        끝면을 막았으면(인히비터) 축방향 소진이 없으므로 w_max 뿐이다.
        """
        if not seg.ends_burn:
            return float(seg.burnback.w_max)
        return float(min(seg.burnback.w_max, seg.length / 2.0))

    def web_max(self) -> float:
        return float(max(self.segment_web(s) for s in self.segments))

    def burn_area(self, w) -> np.ndarray:
        w = np.atleast_1d(np.asarray(w, dtype=float))
        total = np.zeros_like(w)
        for s in self.segments:
            inside = w <= self.segment_web(s)
            a = np.zeros_like(w)
            a[inside] = s.burnback.burn_area(w[inside], s.length, s.ends_burn)
            total += a
        return total

    def volume(self, w) -> np.ndarray:
        w = np.atleast_1d(np.asarray(w, dtype=float))
        total = np.zeros_like(w)
        for s in self.segments:
            inside = w <= self.segment_web(s)
            v = np.zeros_like(w)
            v[inside] = s.burnback.volume(w[inside], s.length, s.ends_burn)
            total += v
        return total

    def mass(self, w=0.0):
        """남은 추진제 질량 [g] — 체적 mm³ × 밀도 g/cc (1 cc = 1000 mm³).

        스칼라 입력이면 스칼라를, 배열 입력이면 **같은 모양의 배열**을 돌려준다.

        ⚠ 초판은 `float(self.volume(w)[0] · ρ/1000)` 이라, 배열을 넘겨도 조용히
          `w[0]` 하나의 질량만 돌려줬다. 질량 궤적을 뽑으려고 `mass(w_tab)` 을
          부르면 상수를 받고도 알 수 없다 — 예외도, 모양 불일치도 없다.
        """
        v = self.volume(w) * self.density / 1000.0
        return float(v[0]) if np.ndim(w) == 0 else v

    def table(self, n_steps: int = 400):
        w = np.linspace(0.0, self.web_max(), n_steps)
        return {"w": w, "A_b": self.burn_area(w), "V": self.volume(w)}
