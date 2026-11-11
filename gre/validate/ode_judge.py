"""ODE 판정자 — 준정상 대신 챔버 ODE 로 "원형으로 충분한가" 를 묻는다.

**왜 만들었나.** 기존 `circle_can_make_curve` 는 준정상 평형식을 쓴다. 싸지만
점화 상승도 블로다운도 모델에 없어서, **실제 원형 포트로 만든 곡선을 넣어도
잔차가 5.5 % 나온다** (전 구간 기준). 허용 오차 10 % 중 절반이 모델 오차로
깔리는 셈이다. 그 바닥을 없애려면 판정자도 목표와 같은 물리를 써야 한다.

**왜 그동안 못 했나 — 그리고 그 판단이 틀렸던 이유.** 처음엔 스칼라 루프로
짜서 한 설계당 ODE 한 번을 돌렸다. 격자 탐색에 20~34 s 가 걸려 "13~18 배
비싸니 못 쓴다" 고 결론냈는데, 그건 **구현 문제였지 방법 문제가 아니었다.**
BATES 는 A_b 와 잔여 체적이 모두 닫힌 식이라 `w` 만 알면 보간표가 필요 없다.
그래서 **설계 수천 개의 ODE 를 한꺼번에 벡터로 적분**할 수 있다. 상태는
(w, ρ_c) 두 개뿐이고 RK4 한 스텝은 전부 원소별 연산이다.

    스칼라 루프 : 설계 1개당 ODE 1회      → 격자 1만 점에 ~13 분
    벡터 적분   : 설계 1만개를 동시에 적분 → 같은 격자에 수 초

**경계 처리.** 설계마다 연소시간이 다르므로 소진한 설계는 마스크로 꺼 둔다
(`A_b = 0`). 그 뒤로는 생성항이 사라지고 배출항만 남아 **블로다운이 저절로
나온다** — 준정상이 못 하던 바로 그 부분이다. 점화 상승도 마찬가지로 초기
조건 ρ_c = p_a/RT 에서 저절로 나온다.

⚠ 여기서 쓰는 물리는 `optimize/forward.py` 의 `DiffMotor.simulate` 와 같아야
   한다. 다른 것은 A_b 를 거리장 대신 BATES 해석해로 얻는다는 점뿐이다.
   달라지면 바닥이 다시 생기므로, `test_floor_is_small_without_any_window` 가 그것을
   지킨다.
"""
from __future__ import annotations

import numpy as np

from ..core.burnrate import model_for
from ..core.diff_burnback import STENCIL_CELLS
from ..core.nozzle import gamma_function, rt_from_cstar

MM2_M2, MM3_M3, MPA_PA = 1e-6, 1e-9, 1e6


class OdeCircleJudge:
    """원형 포트 격자의 추력 곡선을 한꺼번에 적분한다.

    Parameters
    ----------
    r_outer : float      그레인 바깥 반경 [mm]
    propellant, density, c_star, ... : `DiffMotor` 와 같은 규약
    """

    def __init__(self, r_outer: float = 30.0, propellant: str = "KNSB",
                 density: float = 1.841 * 0.92, c_star: float = 885.0,
                 c_star_eff: float = 0.95, cf_eff: float = 0.95,
                 gamma: float = 1.133, p_ambient: float = 0.1013,
                 chamber_margin: float = 1.25, h: float = 2.0e-3,
                 eps_mm: float = 0.0, fd_h: float = None):
        self.r_outer = float(r_outer)
        self.br = model_for(propellant)
        self.rho_p = density * 1000.0
        self.rt = rt_from_cstar(c_star, gamma)
        self.cs = c_star * c_star_eff
        self.cf_eff, self.gamma, self.pa = cf_eff, gamma, p_ambient
        self.margin, self.h = chamber_margin, h
        self.gfun = gamma_function(gamma)
        self.eps = float(eps_mm)
        #: 정방향 모델의 격자 간격 — 주면 `perimeter` 의 차분 스텐실까지 흉내낸다
        self.fd_h = None if fd_h is None else float(fd_h)

    @classmethod
    def from_motor(cls, motor, **kw):
        """정방향 모델과 **같은 ε** 로 맞춘 판정자.

        판정자의 일은 "최적화기가 쓰는 정방향 모델로 이 곡선을 원형이 낼 수
        있는가" 이므로, 기하 이산화까지 같아야 바닥이 0 에 가까워진다.
        """
        kw.setdefault("eps_mm", float(motor.ds.eps))
        kw.setdefault("fd_h", float(motor.ds.h))
        kw.setdefault("propellant", motor.propellant)
        kw.setdefault("density", motor.density)
        kw.setdefault("c_star", motor.c_star)
        kw.setdefault("c_star_eff", motor.c_star_eff)
        kw.setdefault("cf_eff", motor.cf_eff)
        kw.setdefault("gamma", motor.gamma)
        kw.setdefault("p_ambient", motor.p_ambient)
        kw.setdefault("chamber_margin", motor.chamber_margin)
        return cls(r_outer=motor.ds.r_outer, **kw)

    # ------------------------------------------------------------ 기하 (BATES)
    #: 2점 가우스-르장드르 노드 (구간 [0,1] 기준) — 3차까지 정확하다
    _GN = (0.5 - 0.5 / np.sqrt(3.0), 0.5 + 0.5 / np.sqrt(3.0))

    def _geom(self, w, r0, length):
        """A_b [mm²] 와 남은 추진제 체적 [mm³].

        `eps` 가 0 이면 순수 BATES 해석해다. 0 이 아니면 `DiffSegment2D` 와
        **같은 ε 완화**를 연속 극한에서 닫힌 형태로 넣는다.

        거리장 쪽은 지시함수를 두 개의 경사로(ramp)로 완화한다.

            g_in(r)  = clip((r − (r0+w))/ε + 0.5, 0, 1)    포트 쪽 (타고 남은 면)
            g_out(r) = clip((R − r)/ε + 0.5, 0, 1)         케이스 벽

            A_end(w) = 2π ∫ g_in·g_out·r dr
            P₂D(w)   = −dA_end/dw = 2π ∫ (1/ε)·1[|r−(r0+w)| < ε/2]·g_out·r dr

        두 경사로가 겹치지 않는 **구간 내부에서는 정확히 BATES 와 같다**
        (P₂D = 2π(r0+w)). 달라지는 곳은 소진 근처뿐이고, 거기서 A_b 가 계단이
        아니라 매끄럽게 0 으로 간다 — 목표를 만든 거리장이 하는 일과 같다.

        적분은 {A, B, C, D} = {r0+w∓ε/2, R∓ε/2} 를 정렬해 만든 3개 구간에서
        2점 가우스로 한다. 각 구간 안에서 피적분함수는 최대 3차라 **정확**하다.
        """
        l_rem = np.clip(length - 2.0 * w, 0.0, None)
        R = self.r_outer
        if self.eps <= 0.0:                       # 순수 BATES
            r_in = np.minimum(r0 + w, R)
            burnt = (r0 + w) >= R
            per = np.where(burnt, 0.0, 2.0 * np.pi * r_in)
            a_end = np.pi * (R ** 2 - r_in ** 2)
            return np.where(burnt, 0.0, per * l_rem + 2.0 * a_end), a_end * l_rem

        a_end = self._area_end(r0 + w)
        if self.fd_h is None:                     # 해석 미분 (−dA_end/dw)
            p2d = self._dminus_area_end(r0 + w)
        else:
            # `DiffSegment2D.perimeter` 와 **같은 스텐실·같은 끝단 보정**을 쓴다.
            # 판정자가 정방향 모델과 같은 수치 거동을 가져야, "원형이 낼 수 있는
            # 곡선" 을 같은 자로 비교하게 된다. 셋(torch·numpy·판정자)이 갈라지면
            # 같은 설계에서 서로 다른 A_b 가 나온다.
            d = STENCIL_CELLS * self.fd_h
            # 소진 웹은 `DiffSegment2D.web_max` 와 **같은 정의** — `area_end` 가
            # 정확히 0 이 되는 `max(φ) + ε/2` 이고, `max(φ) = (R + ε/2) − r0` 이므로
            # `R + ε − r0` 다.
            w_cap = R + self.eps - r0
            central = -(self._area_end(r0 + w + d) - self._area_end(r0 + w - d)) / (2.0 * d)
            # 소진 3d 는 **반 스텝 중심차분** (정방향 모델과 동일). 초판의 1차
            # 후방차분은 P 를 w − d/2 에서 추정해 적분이 접히지 않았다.
            half = -(self._area_end(r0 + w + 0.5 * d)
                     - self._area_end(r0 + w - 0.5 * d)) / d
            p2d = np.where(w > (w_cap - 3.0 * d), half, central)
        p2d = np.clip(p2d, 0.0, None)
        ends = np.where(l_rem > 0.0, 2.0 * a_end, 0.0)
        return p2d * l_rem + ends, a_end * l_rem

    # ------------------------------------------------- ε 완화된 남은 단면적
    def _area_end(self, c):
        """A_end = 2π ∫ g_in(r)·g_out(r)·r dr — `DiffSegment2D.area_end` 의 연속 극한.

        `c = r0 + w` 는 안쪽 경사로의 중심이다. `c < r0` (음수 웹)도 그대로
        돌아간다 — 중심차분이 w = 0 에서 한쪽으로 치우치지 않으려면 필요하다.
        """
        R, e = self.r_outer, self.eps
        c = np.asarray(c, dtype=float)
        A, B = c - 0.5 * e, c + 0.5 * e
        C = np.full_like(A, R - 0.5 * e)
        D = np.full_like(A, R + 0.5 * e)
        bp = np.sort(np.stack([A, B, C, D]), axis=0)
        out = np.zeros_like(c)
        for i in range(3):
            lo = np.clip(bp[i], 0.0, None)
            hi = np.clip(bp[i + 1], 0.0, None)
            width = hi - lo
            for nd in self._GN:
                r = lo + width * nd
                g_in = np.clip((r - c) / e + 0.5, 0.0, 1.0)
                g_out = np.clip((R - r) / e + 0.5, 0.0, 1.0)
                out += 0.5 * width * g_in * g_out * r
        return 2.0 * np.pi * out

    def _dminus_area_end(self, c):
        """−dA_end/dw = 2π ∫ (1/ε)·1[|r−c| < ε/2]·g_out(r)·r dr (해석 미분)."""
        R, e = self.r_outer, self.eps
        c = np.asarray(c, dtype=float)
        A, B = c - 0.5 * e, c + 0.5 * e
        C = np.full_like(A, R - 0.5 * e)
        D = np.full_like(A, R + 0.5 * e)
        bp = np.sort(np.stack([A, B, C, D]), axis=0)
        out = np.zeros_like(c)
        for i in range(3):
            lo = np.clip(bp[i], 0.0, None)
            hi = np.clip(bp[i + 1], 0.0, None)
            width = hi - lo
            for nd in self._GN:
                r = lo + width * nd
                g_out = np.clip((R - r) / e + 0.5, 0.0, 1.0)
                box = np.where((r > A) & (r < B), 1.0 / e, 0.0)
                out += 0.5 * width * box * g_out * r
        return 2.0 * np.pi * out

    # ------------------------------------------------------------------ 적분
    def curves(self, r0, a_throat, length, n_steps: int = 2400):
        """설계 배열 `(r0, A_t)` 전체의 F(t) 를 **동시에** 적분한다.

        Returns
        -------
        t : (n_steps+1,)        공통 시간축 (고정 스텝)
        f : (N, n_steps+1)      추력 [N]
        """
        r0 = np.asarray(r0, dtype=float).ravel()
        at_mm2 = np.asarray(a_throat, dtype=float).ravel()
        at = at_mm2 * MM2_M2
        n = r0.size
        # 챔버 체적은 **초기 추진제 체적** × 여유 — ε 완화가 들어가면 그 체적도
        # 조금 달라지므로 같은 `_geom` 에서 받아 온다 (DiffMotor 와 같은 규약).
        v_cham = self._geom(np.zeros(n), r0, length)[1] * self.margin

        w = np.zeros(n)
        rho = np.full(n, self.pa * MPA_PA / self.rt)
        h = self.h

        def rhs(w_, rho_):
            # 챔버압 하한은 **대기압** — 노즐로 대기와 통해 있으므로 그 아래로
            # 내려갈 수 없다 (`ballistics.simulate` 와 같은 규약, 같은 결함이었다).
            p_pa = np.clip(rho_ * self.rt, self.pa * MPA_PA, None)
            r_mm = self.br.rate(p_pa / MPA_PA)
            ab, vol = self._geom(w_, r0, length)
            v_free = np.clip(v_cham - vol, 1.0, None) * MM3_M3
            gen = self.rho_p * ab * MM2_M2 * r_mm * 1e-3
            out = p_pa * at / self.cs
            dv = ab * MM2_M2 * r_mm * 1e-3
            return np.where(ab > 0, r_mm, 0.0), (gen - out - rho_ * dv) / v_free

        ps = np.empty((n, n_steps + 1))
        ps[:, 0] = self.pa
        for k in range(n_steps):
            k1w, k1r = rhs(w, rho)
            k2w, k2r = rhs(w + 0.5 * h * k1w, rho + 0.5 * h * k1r)
            k3w, k3r = rhs(w + 0.5 * h * k2w, rho + 0.5 * h * k2r)
            k4w, k4r = rhs(w + h * k3w, rho + h * k3r)
            w = w + (h / 6.0) * (k1w + 2 * k2w + 2 * k3w + k4w)
            rho = rho + (h / 6.0) * (k1r + 2 * k2r + 2 * k3r + k4r)
            ps[:, k + 1] = np.clip(rho * self.rt, 1.0, None) / MPA_PA
        t = np.arange(n_steps + 1) * h

        # ── 팽창비: 작동 구간 시간 평균 압력의 최적값 (§4.4, DiffMotor 와 같은 규약)
        g, G, pa = self.gamma, self.gfun, self.pa
        band = ps >= 0.1 * ps.max(axis=1, keepdims=True)
        span = np.clip(band.sum(axis=1) * h, 1e-9, None)
        p_mean = (ps * band).sum(axis=1) * h / span
        # 최적 팽창이면 P_e = P_a 이므로 압력비는 바로 p_a/p_mean (브렌트 불필요)
        x = np.clip(pa / np.clip(p_mean, pa * 1.0001, None), 1e-12, None)
        eps = G / np.sqrt(2 * g / (g - 1) * x ** (2 / g) * (1 - x ** ((g - 1) / g)))
        cfm = G * np.sqrt(2 * g / (g - 1) * (1 - x ** ((g - 1) / g)))
        cf = cfm[:, None] + (x[:, None] - pa / np.clip(ps, 1e-9, None)) * eps[:, None]
        return t, np.clip(cf * self.cf_eff * ps * MPA_PA * at[:, None], 0.0, None)

    # ------------------------------------------------------------------ 판정
    def fit(self, t_target, f_target, length, r0_grid=None, at_grid=None,
            at_fixed: float = None, mask=None, chunk: int = 4000):
        """목표 곡선에 가장 가까운 원형 설계를 **전역 격자**로 찾는다.

        `at_fixed` 를 주면 노즐목을 입력으로 고정하고 `r0` 만 푼다.
        `mask` 는 잔차를 잴 구간(창) — 목표 배열에 대한 불리언.
        """
        t_target = np.asarray(t_target, dtype=float)
        f_target = np.asarray(f_target, dtype=float)
        if mask is not None:
            t_target, f_target = t_target[mask], f_target[mask]
        scale = max(float(np.abs(f_target).max()), 1e-9)

        if r0_grid is None:
            r0_grid = np.linspace(3.0, self.r_outer - 3.0, 141)
        at_grid = (np.array([float(at_fixed)]) if at_fixed is not None
                   else (at_grid if at_grid is not None
                         else np.geomspace(25.0, 400.0, 101)))
        RR, AA = np.meshgrid(np.asarray(r0_grid, float),
                             np.asarray(at_grid, float), indexing="ij")
        rr, aa = RR.ravel(), AA.ravel()

        best = (np.inf, None, None)
        for s in range(0, rr.size, chunk):
            r0c, atc = rr[s:s + chunk], aa[s:s + chunk]
            t, f = self.curves(r0c, atc, length)
            # 공통 고정 스텝이라 보간이 곧 인덱싱이다
            pred = np.empty((r0c.size, t_target.size))
            for i in range(r0c.size):
                pred[i] = np.interp(t_target, t, f[i], left=0.0, right=0.0)
            res = np.sqrt(np.mean(((pred - f_target) / scale) ** 2, axis=1))
            j = int(res.argmin())
            if res[j] < best[0]:
                best = (float(res[j]), float(r0c[j]), float(atc[j]))
        rms = best[0] * 100.0
        return {"residual_pct": rms, "r0": best[1], "a_throat": best[2],
                "suffices": rms <= 10.0}


def shape_window(f, ignition_frac: float = 0.90, tail_frac: float = 0.20):
    """작동 구간 창 — 점화 끝 이후 · F ≥ tail_frac·F_max.

    ⚠ **손실의 ω(t) 와 '같은 창' 이 아니다**. `loss.shape_window` 의
      꼬리 경계는 **웹 소진**(`w ≥ 0.999·w_cap`)이고 여기는 추력 준위
      (`F ≥ tail_frac·F_max`)다. 점화 쪽 규칙만 같다. 초판 서술이 둘을 같다고
      적어, 판정자 잔차와 손실 잔차가 같은 구간에서 나온다고 오해하게 만들었다.

    ODE 판정자에는 **원리적으로 필요 없다** (점화도 블로다운도 모델에 있다).
    준정상 판정자와 같은 조건에서 비교하기 위해 남겨 둔다.
    """
    f = np.asarray(f, dtype=float)
    plateau = float(np.median(f[f > 0.5 * f.max()]))
    i0 = int(np.argmax(f >= ignition_frac * plateau))
    m = np.zeros(f.size, dtype=bool)
    m[i0:] = True
    return m & (f >= tail_frac * f.max())
