"""2D 거리장 기반 burnback — 단면이 일정한 세그먼트용.

핵심 아이디어
-------------------------------
연소는 모든 연소면에서 같은 속도로 법선 방향으로 진행하므로, 누적 웹 `w`
시점의 연소면은 **거리장의 등고선** `{φ = w}` 이다. 따라서 φ 를 한 번만
계산해 두면 시간 전진 없이 `A_b(w)` 를 얻는다.

단면이 일정한 세그먼트는 축방향이 해석적으로 분리된다.

    끝면 연소:   A_b(w) = P₂D(w) · max(L − 2w, 0) + 2 · A_end(w)
    끝면 인히비트: A_b(w) = P₂D(w) · L

여기서
    A_end(w) : 웹 w 에서 남은 추진제 **단면적**
    P₂D(w)   : 등고선 {φ = w} 의 **둘레**  =  −dA_end/dw

둘레를 따로 추적하지 않고 면적의 미분으로 얻는 것이 요점이다. 면적은
격자 위에서 부드러운 지시함수로 적분하므로, 나중에 자동미분으로 그대로
넘어간다 (ε = 완화 폭).
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from ..geometry.port import FourierPort

#: 둘레 중심차분 스텐실 폭 — torch 판에서 재서 정했다. 근거와 측정표는
#: `gre.core.diff_burnback.STENCIL_CELLS` 의 주석에 있다. **셋이 갈라지면 안 된다.**
from .diff_burnback import STENCIL_CELLS  # noqa: E402


class SegmentBurnback2D:
    """단면 일정 세그먼트의 2D 거리장 burnback.

    Parameters
    ----------
    port : FourierPort
        포트 윤곽 (연소면)
    d_outer : float
        그레인 바깥지름 [mm]. 외벽은 **타지 않는다** (지관·케이스에 접함)
    n : int
        한 변의 격자 수
    eps_cells : float
        완화 폭을 격자 간격의 몇 배로 둘지. 1.0 이면 ε = h
    """

    def __init__(self, port: FourierPort, d_outer: float, n: int = 401,
                 eps_cells: float = 1.0):
        if not port.is_valid():
            raise ValueError("포트 반경이 0 이하가 되는 구간이 있다")
        self.port = port
        self.d_outer = float(d_outer)
        self.r_outer = self.d_outer / 2.0
        self.n = int(n)
        self.eps_cells = float(eps_cells)

        if port.max_radius() >= self.r_outer:
            raise ValueError("포트가 그레인 바깥지름을 벗어난다")

        self._build_field()

    # ------------------------------------------------------------------ 격자
    def _build_field(self) -> None:
        r = self.r_outer
        xs = np.linspace(-r, r, self.n)
        self.h = float(xs[1] - xs[0])
        self.cell_area = self.h ** 2
        X, Y = np.meshgrid(xs, xs, indexing="xy")
        self.X, self.Y = X, Y

        rr = np.hypot(X, Y)
        th = np.arctan2(Y, X)
        r_port = self.port.radius(th)

        # 케이스(그레인 외곽) 안쪽 전체. 외벽은 연소면이 아니라 경계다.
        self.mask_case = rr <= r
        # 추진제 영역: 포트 바깥 ~ 그레인 외벽 안쪽
        self.mask_prop = (rr >= r_port) & self.mask_case

        # 연소면(포트 윤곽)까지의 **부호 있는** 거리장.
        #   포트 내부(이미 공극) < 0,  추진제 쪽 > 0
        # 부호를 주는 이유: A_end(w) 를 w < 0 으로도 매끄럽게 이어 두어야
        # w = 0 에서 중심차분으로 둘레를 구할 때 한쪽만 잘리는 편향이 없다.
        pts = self.port.sample(spacing=self.h / 4.0)
        tree = cKDTree(pts)
        d, _ = tree.query(np.column_stack([X.ravel(), Y.ravel()]), workers=-1)
        d = d.reshape(X.shape)
        self.phi = np.where(rr >= r_port, d, -d)

        # 외벽(그레인 바깥지름)도 계단으로 자르면 면적 오차가 1차로 남아
        # 격자를 촘촘히 해도 줄지 않는다. 같은 폭 ε 로 완화해 가중치를 준다.
        eps0 = self.eps_cells * self.h
        self.w_case = np.clip((r - rr) / eps0 + 0.5, 0.0, 1.0)
        self.sel = self.w_case > 0.0
        self.phi_case = self.phi[self.sel]
        self.wt_case = self.w_case[self.sel]

        # 정렬 + 누적합을 미리 만들어 두면 A_end(w) 가 O(log N) 이 된다.
        # (직접 합산하면 w 격자 800점 × 셀 13만개 = 1억 원소라 표 하나
        #  만드는 데 10초가 넘는다. 수식은 완전히 같다.)
        order = np.argsort(self.phi_case)
        self._phi_s = self.phi_case[order]
        wt_s = self.wt_case[order]
        self._cw = np.concatenate([[0.0], np.cumsum(wt_s)])
        self._cwp = np.concatenate([[0.0], np.cumsum(wt_s * self._phi_s)])

        # ── 소진 웹 — torch 판 `DiffSegment2D.web_max` 와 **같은 정의** ──────
        # `area_end` 가 정확히 0 이 되는 지점(= max φ + ε/2). 여기까지 태워야
        # 장전 질량이 전부 소비된다. 꺾임점을 비켜 두는 이유와, 이 값이 끝단
        # 둘레 보정(`endgame_scale`)과 짝이라는 것은 torch 판 주석에 있다 —
        # **한쪽만 고치면 총역적 편향이 커진다.**
        eps0 = self.eps_cells * self.h
        self.w_max_geom = float(np.max(self.phi[self.mask_prop]))   # 하드 기준(참고)
        self.w_max = float(self._phi_s[-1] + eps0 / 2.0 * (1.0 + 1e-9))

        self.area_prop0 = float(self.area_end(0.0)[0])

    # ------------------------------------------------------------- 기본 질의
    def area_end(self, w) -> np.ndarray:
        """웹 w 에서 남은 추진제 단면적 A_end(w) [mm²].

        {φ > w} 를 폭 ε 의 선형 완화로 적분한다. w < 0 으로도 매끄럽게
        이어진다 (포트가 그만큼 작았을 때의 면적).
        """
        w = np.atleast_1d(np.asarray(w, dtype=float))
        eps = self.eps_cells * self.h
        phi_s, cw, cwp = self._phi_s, self._cw, self._cwp

        lo = np.searchsorted(phi_s, w - eps / 2.0, side="left")
        hi = np.searchsorted(phi_s, w + eps / 2.0, side="right")

        above = cw[-1] - cw[hi]                       # 완전히 남은 셀 (가중치 1)
        band_w = cw[hi] - cw[lo]                      # 완화 밴드 안 셀의 가중치 합
        band_wp = cwp[hi] - cwp[lo]                   # 같은 셀의 가중치×φ 합
        band = band_wp / eps - (w / eps - 0.5) * band_w
        return (above + band) * self.cell_area

    def perimeter(self, w, richardson: bool = True) -> np.ndarray:
        """등고선 {φ = w} 의 둘레 P₂D(w) [mm] = −dA_end/dw.

        면적을 중심차분해서 얻는다. 셀을 직접 세는 방식(밴드 카운트)은 격자
        양자화 때문에 값이 계단처럼 튀어서 쓰지 않는다 — torch 판에서는 그 방식이
        **경사를 아예 못 주기도 한다**(지시함수의 미분이 0). `richardson=True` 면
        간격 d 와 2d 의 중심차분을 외삽해 O(d⁴) 로 올린다.

        소진 근처 3d 는 **반 스텝 중심차분**. 근거와 측정표는 torch 판
        `DiffSegment2D.perimeter` 주석에 있다 — **두 판이 갈라지면 안 된다.**
        """
        w = np.atleast_1d(np.asarray(w, dtype=float))
        d = STENCIL_CELLS * self.h          # torch 판·ODE 판정자와 **같은 값**

        def central(step):
            return -(self.area_end(w + step) - self.area_end(w - step)) / (2.0 * step)

        p1 = central(d)
        p = p1 if not richardson else (4.0 * p1 - central(2.0 * d)) / 3.0

        near_end = w > (self.w_max - 3.0 * d)
        if np.any(near_end):
            ww = w[near_end]
            p = p.copy()
            p[near_end] = -(self.area_end(ww + 0.5 * d)
                            - self.area_end(ww - 0.5 * d)) / d
        return np.maximum(p, 0.0)

    # -------------------------------------------------------------- 연소면적
    def burn_area(self, w, length: float, ends_burn: bool = True) -> np.ndarray:
        """세그먼트 연소면적 A_b(w) [mm²].

        ends_burn=True  : A_b = P₂D(w)·max(L − 2w, 0) + 2·A_end(w)·[L − 2w > 0]
        ends_burn=False : A_b = P₂D(w)·L

        ⚠ **끝면 항에 지시자가 필요하다.** `w = L/2` 에서 두 끝면이 만나 축방향으로
          소진되므로 `volume = A_end·(L − 2w) = 0` 인데, 초판은 `A_b = 0 + 2·A_end`
          로 **추진제가 없는데 연소면이 있다**고 보고했다. `A_b = −dV/dw` 가 성립
          하려면 `l_rem > 0` 일 때만 끝면을 세야 한다. `l_rem = 0` 에서 A_b 가
          0 으로 떨어지는 것은 수치 결함이 아니라 물리다 — 마지막 조각이 사라진다.
        """
        w = np.atleast_1d(np.asarray(w, dtype=float))
        p2d = self.perimeter(w)
        if not ends_burn:
            return p2d * float(length)
        l_rem = np.clip(float(length) - 2.0 * w, 0.0, None)
        ends = np.where(l_rem > 0.0, 2.0 * self.area_end(w), 0.0)
        return p2d * l_rem + ends

    def volume(self, w, length: float, ends_burn: bool = True) -> np.ndarray:
        """남은 추진제 체적 [mm³]."""
        w = np.atleast_1d(np.asarray(w, dtype=float))
        a_end = self.area_end(w)
        if not ends_burn:
            return a_end * float(length)
        l_rem = np.clip(float(length) - 2.0 * w, 0.0, None)
        return a_end * l_rem

    def web_grid(self, n_steps: int = 200) -> np.ndarray:
        """0 ~ w_max 의 웹 격자."""
        return np.linspace(0.0, self.w_max, n_steps)

    def table(self, length: float, ends_burn: bool = True, n_steps: int = 200):
        """(w, A_b, A_end, P₂D, V) 표를 한 번에 계산."""
        w = self.web_grid(n_steps)
        return {
            "w": w,
            "A_b": self.burn_area(w, length, ends_burn),
            "A_end": self.area_end(w),
            "P2D": self.perimeter(w),
            "V": self.volume(w, length, ends_burn),
        }
