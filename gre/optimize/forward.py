"""미분가능 정방향 모델 — 형상 계수에서 추력 곡선까지.

    θ = (r0, a, b, L, A_t)
        → φ(θ)                     거리장      (diff_burnback)
        → A_b(w), V(w)             연소량      (diff_burnback)
        → ρ_c(t), w(t)             챔버 ODE    ← 이 파일
        → P(t), F(t)               곡선

결정: **챔버 ODE 를 그대로 미분한다.** 준정상 평형식은
점화 상승을 평형값으로 점프시키므로 과도 구간을 맞출 수 없다. 측정 결과
ODE(RK4, h = 5 ms)는 기준 LSODA 대비 최대압·최대추력 0.00 %, 총역적 0.14 %
이고 경사당 비용은 준정상의 2.4 배였다.

⚠ **강성(stiff) 가드가 필수다.** 챔버 충전 시상수

        τ = c*·V_free / (R·T·A_t)

는 설계에 따라 변한다. 추진제를 더 채우면 V_free 가 줄어 τ 가 작아지고,
고정 스텝 RK4 는 h > 2.78τ 에서 발산한다. 최적화기가 장약을 늘리는 쪽으로
움직이면 조용히 터질 수 있으므로, 매 정방향마다 그 설계의 τ_min 을 구해
h = min(h_max, SAFETY·τ_min) 으로 줄인다. SAFETY = 1.5 는 안정 한계(2.78)의
1.85 배 여유이고, 정확도는 h = 1.33τ 에서 이미 확인했다.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from ..core.burnrate import DEFAULT_DELTA, NAKKA_TABLE
from ..core.diff_burnback import DiffSegment2D
from ..core.nozzle import (PE_PA_SEPARATION_DEFAULT,
                           area_ratio_to_pressure_ratio, gamma_function,
                           optimum_area_ratio, rt_from_cstar)

DTYPE = torch.float64
MM2_M2, MM3_M3, MPA_PA = 1e-6, 1e-9, 1e6

#: RK4 안정 한계는 h < 2.78τ. 그 1.85 배 여유.
STIFF_SAFETY = 1.5

# ────────────────────────────────── 윤곽 표본 수는 격자와 **함께** 정해야 한다
#
# `DiffSegment2D.phi` 는 포트 윤곽을 `n_samples` 점으로 샘플해 그 점들까지의
# soft-min 거리로 거리장을 만든다. 표본 간격은 `2π·r_port / n_samples` 로
# **격자와 무관**한데 격자 칸 `h` 는 `n_grid` 로 줄어든다. 그래서 `n_samples` 를
# 고정한 채 격자만 조이면 격자점이 윤곽의 매끄러움이 아니라 **표본 격자의
# 톱니**를 보게 되고, 그 톱니가 경사에 들어간다.
#
# 이것이 실제로 터졌다. `n_samples=384` 를 격자와 무관하게 쓰고 있었고
# n=241 로 올렸을 때 경사가 34.7 % 틀렸다 (`experiments/sample_resolution.py
# --rule`, 지표는 `|g(n_s) − g(ref)| / |g(ref)|`, 기준 `n_s = 2048`):
#
#     n_grid=121 :  256  3.21 %   384  4.19 %   512  3.44 %   **768  0.25 %**
#     n_grid=241 :  256 10.26 %   384 34.72 %   512  2.89 %   **768  1.21 %**
#
# n=121 의 3~4 % 는 표본 때문이 아니라 ODE 스케줄의 이산성이다 — 같은 설계에서
# 적분 스텝 수가 993 ↔ 994 로 하나 흔들린다. 정방향 양(총역적·장약·웹)은
# 1e-5 로 안정하지만 역방향은 그만큼 흔들린다. 즉 **그 3 %가 바닥이고**
# `n_samples` 로는 더 못 내린다. n=241 의 34.7 % 는 그 바닥의 10 배라
# 표본 결함이 분명하다.
#
# ⚠ **비용은 사실상 없다.** 한 반복(정방향+역방향) 실측 n=121 에서
#    384 → 768 이 6.62 s → 6.60 s 다 (비용은 ODE 스텝 ~870 이 지배한다).
#    메모리는 1239 MB → 1328 MB (+7 %). 그래서 "필요한 만큼" 이 아니라
#    "재서 통한 값" 을 쓰는 쪽이 맞다.

#: 재서 통한 표본 수. n=121·n=241 **둘 다** 768 에서 통과했다 — 이 구간에서
#: 요구량은 `1/h` 에 비례하지 않고 거의 일정하다. 그래서 공식이 아니라
#: 측정값을 쓴다.
SAMPLES_MEASURED = 768
#: 위 측정이 덮는 격자 상한. 이보다 촘촘한 격자는 **재 본 적이 없다** —
#: 그때는 보수적 공식으로 넘어간다 (아래 `samples_for_grid`).
SAMPLES_MEASURED_MAX_GRID = 241
#: 측정 범위 밖에서 쓰는 보수적 기준 — 표본 간격 ≤ 이 비율 × `h`.
#: 4부 측정에서 간격/h = 0.40 은 통과, 0.81 은 깨졌다. 통과한 쪽을 쓴다.
SAMPLES_SPACING_RATIO = 0.40


def samples_for_grid(n_grid: int, d_outer: float) -> int:
    """그 격자에서 **최소로 필요한** `n_samples`.

    측정이 덮는 구간(`n_grid ≤ 241`)에서는 측정값을 그대로 쓴다. 그 밖에서는
    `간격 ≤ 0.40·h` 를 `r_port ≤ r_outer` 로 보수적으로 풀어 쓴다 — 포트의
    실제 반경은 미리 알 수 없고 `theta_s` 는 생성 시점에 고정되기 때문이다.
    보수적인 쪽은 느려지기만 하고, 모자란 쪽은 조용히 경사를 망친다.
    """
    if int(n_grid) <= SAMPLES_MEASURED_MAX_GRID:
        return SAMPLES_MEASURED
    h = float(d_outer) / (int(n_grid) - 1)
    need = 2 * np.pi * (float(d_outer) / 2.0) / (SAMPLES_SPACING_RATIO * h)
    return max(SAMPLES_MEASURED, int(np.ceil(need / 64.0)) * 64)


# ════════════════════════════════════════════════════════════ 연소속도 (torch)
class TorchBurnRate:
    """`core.burnrate.BurnRateModel` 과 **같은 식**을 텐서로 다시 쓴 것.

    블렌드가 해석적 닫힌 형태라 이식이 그대로 된다. PCHIP 이었다면 스플라인
    계수를 옮겨야 했다 — 경계 전이를 블렌드로 바꾼 덤이 여기서 나온다.
    """

    def __init__(self, propellant: str = "KNSB", delta: float = DEFAULT_DELTA,
                 dtype=DTYPE, device="cpu"):
        tab = NAKKA_TABLE[propellant]
        self.ln_a = torch.tensor([np.log(t[2]) for t in tab], dtype=dtype, device=device)
        self.n = torch.tensor([t[3] for t in tab], dtype=dtype, device=device)
        self.ln_pb = torch.tensor([np.log(t[0]) for t in tab[1:]], dtype=dtype, device=device)
        self.delta = float(delta)

    def rate(self, p_mpa):
        """r [mm/s]."""
        lp = torch.log(torch.clamp(p_mpa, min=1e-3))
        out = self.ln_a[0] + self.n[0] * lp
        d = self.delta
        for k in range(self.ln_pb.shape[0]):
            u = torch.clamp((lp - (self.ln_pb[k] - d)) / (2.0 * d), 0.0, 1.0)
            s = u * u * (3.0 - 2.0 * u)
            out = (1.0 - s) * out + s * (self.ln_a[k + 1] + self.n[k + 1] * lp)
        return torch.exp(out)


# ════════════════════════════════════════════════════════════════════ 모터
@dataclass
class DiffMotor:
    """미분가능 모터 — 형상 계수 → 추력 곡선.

    세그먼트는 **같은 격자를 공유**한다 (거리장 계산이 전체 비용의 ~98 % 라
    공유가 그대로 성능이다). 세그먼트마다 다른 것은 포트 계수와 길이뿐이다.

    ⚠ `n_samples` 는 **격자에 비해 모자라면 조용히 올려진다** (`samples_for_grid`).
       `n_samples=384` 를 격자와 무관하게 쓰다가 n=241 에서 경사가
       34.7 % 틀렸다. 호출자가 비용을 낮추려 작은 값을 넘기는 일이 잦으므로
       경고만으로는 부족하다 — 실제로 그렇게 해서 터졌다. 올려졌는지는
       `motor.n_samples_requested` 와 `motor.n_samples` 를 비교하면 된다.
    """

    d_outer: float                      # [mm]
    n_grid: int = 241
    eps_cells: float = 1.5
    #: 0 또는 None 이면 격자에서 자동으로 정한다. 명시한 값이 필요량보다
    #: 작으면 필요량으로 올린다 (낮추지는 않는다 — 더 촘촘한 쪽은 해롭지 않다).
    n_samples: int = 0
    n_w: int = 400                      # 웹 격자 (A_b 표)
    propellant: str = "KNSB"
    density: float = 1.841 * 0.92       # [g/cc]
    c_star: float = 885.0               # [m/s] 공표값
    c_star_eff: float = 0.95
    cf_eff: float = 0.95
    gamma: float = 1.133
    p_ambient: float = 0.1013           # [MPa]
    chamber_margin: float = 1.25        # 챔버 체적 = 초기 추진제 체적 × 이 값
    #: **박리 문턱 `p_e/p_a`, 0~1 사용자 입력.** 0 = 박리를 안 봄(이상 과팽창식,
    #: 가장 비관적) · 1 = 즉시 박리(항상 최적팽창, 가장 낙관적). 기본값 0.4 는
    #: Summerfield 기준. 근거와 양 끝 검증은 `nozzle.PE_PA_SEPARATION_DEFAULT`.
    separation_frac: float = PE_PA_SEPARATION_DEFAULT
    ends_burn: bool = True
    h_max: float = 5e-3                 # [s] 적분 스텝 상한
    dtype: torch.dtype = DTYPE
    device: str = "cpu"

    ds: DiffSegment2D = field(init=False)
    br: TorchBurnRate = field(init=False)

    def __post_init__(self):
        # ── 표본 수를 격자에 묶는다 (위 `samples_for_grid` 주석의 측정 근거)
        self.n_samples_requested = int(self.n_samples or 0)
        need = samples_for_grid(self.n_grid, self.d_outer)
        self.n_samples = max(self.n_samples_requested, need)
        self.n_samples_raised = self.n_samples != self.n_samples_requested

        if not (0.0 <= float(self.separation_frac) <= 1.0):
            raise ValueError(
                f"박리 문턱 separation_frac 는 0~1 이어야 한다 "
                f"(받은 값 {self.separation_frac}). 0 = 박리 안 봄, "
                f"1 = 즉시 박리.")

        self.ds = DiffSegment2D(self.d_outer, n=self.n_grid, eps_cells=self.eps_cells,
                                n_samples=self.n_samples, dtype=self.dtype,
                                device=self.device)
        self.br = TorchBurnRate(self.propellant, dtype=self.dtype, device=self.device)
        self.rt = rt_from_cstar(self.c_star, self.gamma)      # §6.5 규약: 공표 c*
        self.c_star_actual = self.c_star * self.c_star_eff
        self.rho_p = self.density * 1000.0                    # [kg/m³]

    # ------------------------------------------------------------------ 기하
    def tables(self, r0, a, b, lengths):
        """웹 격자 위의 A_b(w), V(w) [mm², mm³]. 세그먼트 합."""
        w_max_geom = self.ds.r_outer - float(torch.as_tensor(r0).detach())
        segs = len(lengths)
        r0 = torch.atleast_1d(r0)
        a = a if a.ndim == 2 else a.unsqueeze(0)
        b = b if b.ndim == 2 else b.unsqueeze(0)
        if r0.shape[0] == 1 and segs > 1:
            r0 = r0.expand(segs)
            a, b = a.expand(segs, -1), b.expand(segs, -1)

        # 웹 상한 — 반경 소진과 축방향 소진 중 먼저 오는 쪽 (core.grain 과 같은 규칙)
        #
        # ⚠ 반경 소진 웹은 `ds.web_max(phi)` 를 쓴다 — `area_end` 와 **같은 완화**로
        #   정의된 값이다. 초판은 여기서 `phi.max()` 를 직접 불러서 `web_max()` 를
        #   우회했고, 그 지점에서 `area_end` 가 아직 0 이 아니었다. 결과는 두 가지다.
        #     · `A_b(w_cap) ≠ 0` 인 채로 표가 끊긴다
        #     · 장전 질량은 `volume(0)`(완화 기준)으로 세면서 소진은 그보다 일찍
        #       멈추므로 **장전한 추진제의 일부가 타지 않고 사라진다**
        #   차이는 ε/2 (운용 설정 0.375 mm) 이고 격자를 촘촘히 하면 사라진다.
        w_cap = 0.0
        phis = []
        for i in range(segs):
            phi = self.ds.phi(r0[i], a[i], b[i])
            phis.append(phi)
            wr = self.ds.web_max(phi)
            wa = float(lengths[i]) / 2.0 if self.ends_burn else wr
            w_cap = max(w_cap, min(wr, wa))
        w = torch.linspace(0.0, w_cap, self.n_w, dtype=self.dtype, device=self.device)

        ab = torch.zeros_like(w)
        vol = torch.zeros_like(w)
        for i, phi in enumerate(phis):
            wr = self.ds.web_max(phi)
            wa = float(lengths[i]) / 2.0 if self.ends_burn else wr
            alive = (w <= min(wr, wa)).to(self.dtype)
            ab = ab + alive * self.ds.burn_area(phi, w, float(lengths[i]),
                                                self.ends_burn, w_max=wr)
            vol = vol + alive * self.ds.volume(phi, w, float(lengths[i]), self.ends_burn)
        return w, ab, vol, phis

    # ------------------------------------------------------------------ 적분
    def simulate(self, r0, a, b, lengths, a_throat, expansion_ratio=None,
                 tail_taus: float = 6.0):
        """형상 → (t, P, F, 진단).  a_throat [mm²] 는 텐서여도 된다 (설계 변수)."""
        w_tab, ab_tab, v_tab, phis = self.tables(r0, a, b, lengths)
        at_mm2 = torch.as_tensor(a_throat, dtype=self.dtype, device=self.device)
        at = at_mm2 * MM2_M2
        v_chamber = v_tab[0] * self.chamber_margin                 # [mm³]

        # ── 적분 구간과 스텝 일정: 준정상 추정으로 한 번만 가늠한다 ──
        with torch.no_grad():
            p_eq = self._equilibrium(ab_tab, at)
            r_eq = self.br.rate(p_eq)                              # mm/s
            dt_eq = torch.diff(w_tab) / torch.clamp((r_eq[1:] + r_eq[:-1]) / 2, min=1e-9)
            t_of_w = torch.cat([torch.zeros(1, dtype=self.dtype, device=self.device),
                                torch.cumsum(dt_eq, 0)])
            t_burn = float(t_of_w[-1])
            # τ(w) = c*·V_free(w)/(R·T·A_t) — 추진제가 타면 V_free 가 커져 τ 도 커진다
            tau_w = (self.c_star_actual * torch.clamp(v_chamber - v_tab, min=1.0) * MM3_M3
                     / (self.rt * at))
            tau_min = float(tau_w[0])
            t_end = t_burn + tail_taus * tau_min * 20.0
            h_list = _build_schedule(t_of_w, tau_w, self.h_max, t_end)
            n_steps = len(h_list)
            # ⚠ **보고용 스칼라다. 적분에 쓰면 안 된다.** 스케줄은 강성 가드
            #    때문에 일부러 불균일하므로(`_build_schedule`) 이 평균으로 무엇을
            #    누적하면 틀린 시각이 나온다. 웹 궤적처럼 시간축 누적이 필요하면
            #    **`diag["h_list"]` 를 쓸 것.**
            #
            #    전면 검토: `runner.evaluate` 가 이 값으로 웹 궤적을 쌓고
            #    있었고, 그 궤적이 `shape_window` 의 테일오프 경계를 정했다.
            #    웹 소진 시각이 4.26 s vs 실제 8.76 s (2.1 배), 모양 오차에
            #    1.3 %p. 초판 주석이 "보고용 평균" 이라고만 적어 두어 이 값을
            #    적분에 써도 안전하다고 오해하게 만들었다.
            h = float(np.mean(h_list))

        y = torch.stack([torch.zeros((), dtype=self.dtype, device=self.device),
                         torch.as_tensor(self.p_ambient * MPA_PA / self.rt,
                                         dtype=self.dtype, device=self.device)])
        w_cap = float(w_tab[-1])

        def rhs(state):
            w, rho = state[0], state[1]
            p_pa = torch.clamp(rho * self.rt, min=1.0)
            p_mpa = p_pa / MPA_PA
            r_mm = self.br.rate(p_mpa)
            ab = _interp(w, w_tab, ab_tab) * (w < w_cap).to(self.dtype)
            vol = _interp(w, w_tab, v_tab) * (w < w_cap).to(self.dtype)
            v_free = torch.clamp(v_chamber - vol, min=1.0) * MM3_M3
            ab_m, r_m = ab * MM2_M2, r_mm * 1e-3
            mdot_gen = self.rho_p * ab_m * r_m
            mdot_out = p_pa * at / self.c_star_actual
            dv_free = ab_m * r_m                                   # 자유체적 증가
            return torch.stack([r_mm,
                                (mdot_gen - mdot_out - rho * dv_free) / v_free])

        ps = [torch.as_tensor(self.p_ambient, dtype=self.dtype, device=self.device)]
        for hi in h_list:
            k1 = rhs(y)
            k2 = rhs(y + 0.5 * hi * k1)
            k3 = rhs(y + 0.5 * hi * k2)
            k4 = rhs(y + hi * k3)
            y = y + (hi / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            ps.append(torch.clamp(y[1] * self.rt, min=1.0) / MPA_PA)
        p = torch.stack(ps)
        t = torch.as_tensor(np.concatenate([[0.0], np.cumsum(h_list)]),
                            dtype=self.dtype, device=self.device)

        # ── 팽창비: 작동 구간(최대압 10 % 교차) 시간 평균 압력의 최적값 (§4.4) ──
        if expansion_ratio is None:
            with torch.no_grad():
                band = p >= 0.1 * p.max()
                span = torch.clamp(t[band][-1] - t[band][0], min=1e-9)
                p_mean = float(torch.trapezoid(p[band], t[band]) / span)
            eps_ratio = (optimum_area_ratio(p_mean, self.p_ambient, self.gamma)
                         if p_mean > self.p_ambient else 1.0)
        else:
            eps_ratio, p_mean = float(expansion_ratio), None

        pe_pc = area_ratio_to_pressure_ratio(eps_ratio, self.gamma)
        g, G = self.gamma, gamma_function(self.gamma)
        cf_mom = G * np.sqrt(2 * g / (g - 1) * (1 - pe_pc ** ((g - 1) / g)))
        cf = cf_mom + (pe_pc - self.p_ambient / torch.clamp(p, min=1e-9)) * eps_ratio

        if self.separation_frac > 0.0:
            # ── 박리 인지 C_f (`nozzle.PE_PA_SEPARATION_DEFAULT` 주석 참조).
            #    꽉 찬 흐름의 출구압 `pe_pc·p` 가 `frac·p_a` 아래면 박리로 보고
            #    유효 팽창비를 `p_e = frac·p_a` 인 값으로 낮춘다.
            #
            # ⚠ **두 가지가 미분을 망가뜨릴 수 있어서 그렇게 안 짰다.**
            #   ① `torch.where` 는 **고르지 않은 가지도 계산**하고, 그 가지에
            #      nan 이 있으면 역전파에서 nan 이 **선택된 쪽까지 오염**시킨다.
            #      `x = frac·p_a/p` 는 p 가 작으면 1 을 넘고 `1 - x^((γ-1)/γ)`
            #      가 음수가 되어 `sqrt` 가 nan 이다. 그래서 **먼저 clamp** 한다.
            #   ② `eps_eff` 분모의 `u` 도 0 에 닿으면 경사가 발산하므로 같이 막는다.
            pa = self.p_ambient
            p_safe = torch.clamp(p, min=1e-9)
            x = torch.clamp(float(self.separation_frac) * pa / p_safe,
                            max=1.0 - 1e-9)                 # ← ① nan 차단
            u = torch.clamp(1.0 - x ** ((g - 1) / g), min=1e-12)   # ← ②
            cf_mom_sep = G * torch.sqrt(2 * g / (g - 1) * u)
            # ε_eff = Γ / √(2γ/(γ−1)·x^(2/γ)·u)  — `optimum_area_ratio` 의 텐서판
            eps_eff = G / torch.sqrt(torch.clamp(
                2 * g / (g - 1) * x ** (2 / g) * u, min=1e-12))
            cf_sep = cf_mom_sep + (x - pa / p_safe) * eps_eff
            separated = (pe_pc * p_safe) < (float(self.separation_frac) * pa)
            cf = torch.where(separated, cf_sep, cf)

        f = torch.clamp(cf * self.cf_eff * p * MPA_PA * at, min=0.0)

        diag = {"h": h, "h_list": h_list, "h_min": float(np.min(h_list)),
                "h_max_used": float(np.max(h_list)), "n_steps": n_steps,
                "tau_min": tau_min, "t_burn_est": t_burn,
                "expansion_ratio": eps_ratio, "p_mean": p_mean,
                "w_tab": w_tab, "a_b": ab_tab, "volume": v_tab, "phis": phis,
                "mass_g": float(v_tab[0].detach()) * self.density / 1000.0}
        return t, p, f, diag

    # -------------------------------------------------------------- 보조
    def _equilibrium(self, ab_mm2, at, iters: int = 30):
        """준정상 평형압 [MPa] — **적분 구간을 가늠하는 용도로만** 쓴다.

        블렌드 모델에서 설계 대역의 국소 지수가 n < 1 이므로 평형이 유일하고,
        이분법이 항상 같은 근으로 간다.
        """
        ab = ab_mm2 * MM2_M2
        lo = torch.full_like(ab, 1e-3)
        hi = torch.full_like(ab, 50.0)
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            r = self.br.rate(mid) * 1e-3
            g = ((self.rho_p - mid * MPA_PA / self.rt) * ab * r
                 - mid * MPA_PA * at / self.c_star_actual)
            lo = torch.where(g > 0, mid, lo)
            hi = torch.where(g > 0, hi, mid)
        return 0.5 * (lo + hi)


def _build_schedule(t_of_w, tau_w, h_max: float, t_end: float):
    """단계별 적분 스텝 [s] 목록.

    τ 는 연소가 진행되며 커지므로(V_free 증가), τ_min 하나로 전 구간을 묶으면
    평탄한 구간에서 불필요하게 잘게 썬다. 준정상 추정으로 τ(t) 를 받아
    구간마다 h 를 다시 정한다.

    **보수성**: τ 가 t 에 대해 단조 증가하므로, 구간 시작점의 τ 로 h 를 정하면
    그 구간 전체에서 h ≤ 1.5·τ 가 자동으로 보장된다. 일정 자체는 추정값에서
    나오지만 detach 되어 있어 경사에는 영향이 없고, 정확도는 고정 스텝 결과와
    대조해 확인한다.
    """
    t_np = t_of_w.detach().cpu().numpy()
    tau_np = tau_w.detach().cpu().numpy()
    hs, t = [], 0.0
    while t < t_end:
        tau_here = float(np.interp(t, t_np, tau_np, left=tau_np[0], right=tau_np[-1]))
        hs.append(min(h_max, STIFF_SAFETY * tau_here))
        t += hs[-1]
    return np.asarray(hs)


def _interp(x, xp, fp):
    """스칼라 x 에 대한 선형 보간 (미분 가능)."""
    i = torch.clamp(torch.searchsorted(xp, x.reshape(1).contiguous()) - 1,
                    0, xp.shape[0] - 2)
    t = (x - xp[i]) / (xp[i + 1] - xp[i])
    return (fp[i] * (1 - t) + fp[i + 1] * t)[0]


def resample(t_src, f_src, t_dst):
    """곡선을 목표 시간 격자로 옮긴다 (범위 밖은 0). 미분 가능."""
    i = torch.clamp(torch.searchsorted(t_src.contiguous(), t_dst.contiguous()) - 1,
                    0, t_src.shape[0] - 2)
    u = (t_dst - t_src[i]) / (t_src[i + 1] - t_src[i])
    out = f_src[i] * (1 - u) + f_src[i + 1] * u
    return torch.where((t_dst >= t_src[0]) & (t_dst <= t_src[-1]),
                       out, torch.zeros_like(out))
