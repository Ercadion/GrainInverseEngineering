"""역설계 최적화 루프 — Adam.

설계 변수 (연속): 포트 윤곽 계수 `(r0, a, b)`, 세그먼트 길이 `L`, 노즐목 `A_t`.
이산 변수(세그먼트 수 등)는 바깥에서 열거하고, 여기는 그 안쪽 루프다.

⚠ **원형 포트에서 출발하면 안 된다.** `a = b = 0` 에서 모든 경사가 정확히 0 이다
   (측정값 ~1e-13, 유한차분으로도 f(+δ) = f(−δ) 확인). 포트 면적이
   ∮r²/2 dθ = πr₀² + (π/2)Σ(a_k²+b_k²) 라 계수에 대해 **짝함수**이기 때문이다.
   원형은 안장점이므로 경사 하강이 한 발도 못 뗀다. §4.6.5 의
   "B 후보 형상을 초기값으로" 가 수렴 속도 문제가 아니라 **필수 조건**이다.
   초기 형상이 원형에 가까우면 `seed_amplitude` 로 대칭을 깨고 시작한다.

⚠ 변수들의 크기가 다르다 (r0 ~ 10 mm, a_k ~ 1 mm, A_t ~ 80 mm²). Adam 은
   좌표별로 스텝을 정규화하므로 비교적 견디지만, 그래도 변수군마다 학습률을
   따로 둔다.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from .constraints import (D_THROAT_MIN_MM, FILL_SEVERITY_MAX,
                          ManufacturingLimits, ThroatSpec,
                          connected_components, fill_feasible,
                          manufacturing_penalty)
from .forward import DiffMotor
from .loss import (LossWeights, Target, Tolerances, curve_loss, fp_ratio_spread,
                   group_l1,
                   hard_pressure_penalty, mass_penalty, shape_window,
                   sparsity_ramp)

#: [mm] **전사 오차의 상한 추정** — 설계 형상이 그레인에 옮겨질 때의 오차.
#:
#: ⚠⚠ ** 더 이상 진폭 유의성 판정에 쓰지 않는다** (`amplitude_floor` 참조).
#:     이유는 이 값이 **측정이 아니라 설정값**이기 때문이다.
#:
#:     사슬은 이렇게 적혀 있었다 —
#:
#:         테셀레이션 처짐                           0.0198 mm  ← 지배항
#:         수축 보정 잔차 (실측 0.2055 − 적용 0.200)  0.0012
#:         출력 오차                                0      (가정)
#:         압축 성형 전사                            0      (가정)
#:
#:     전제 두 개가 나머지를 0 으로 만든 뒤에는 지배항만 남는데, 그 0.0198 은
#:     `export/profile.py` 의 `DEFAULT_CHORD_TOL = 0.02` 가 만든 값이다.
#:     확인 (6-로브 r₀14): 공차 0.02 → 처짐 0.0198, 0.01 → 0.0099,
#:     0.005 → 0.0050 으로 **1:1 로 따라간다.** 즉 δ 는 내가 고른 렌더링
#:     공차이고, 실리콘·추진제의 전사 정밀도에 대한 정보가 아니다.
#:     전제를 받아들이는 한 **δ 는 측정될 수 없다** — 사슬에 측정항이 없다.
#:
#: 그래도 상수를 남겨 두는 이유: 공차를 그 값으로 고정해 두는 한 "전사 오차는
#: 적어도 이보다 크지 않다" 는 **상한**으로는 읽을 수 있고, STL 공차가 설계
#: 의도보다 커지는 것을 막는 기준점이 필요하다. 쓰는 곳은 그 둘뿐이다.
#:
#: 교차 확인: 쐐기 시편 R1~R4 (ρ 0.2 / 0.4 / 0.8 / 1.5 mm) 가 네 개 모두
#: 전사됐다 — 가장 작은 R1 이 ρ 0.2 mm 다. ⚠ 다만 `data/coupon/T3_tips.csv` 의
#: 반경 칸이 비어 있어 "전사 오차가 0.02 mm 보다 훨씬 작다" 는 **정량** 주장은
#: 원 피팅 수치 없이 서 있다 (형성 여부는 실측, 반경 값은 미측정).
DELTA_TRANSFER_MM: float = 0.020

#: 추력·압력 두 곡선의 모순 판정 문턱 — **함의 노즐목** `A_t = (F/P)/C_f(P, ε)`
#: 의 허용 변동폭 (상대, 1.0 = 100 %). `A_t` 는 시간에 대해 **고정**이므로 두
#: 곡선이 한 모터에서 나왔다면 이 값이 상수다 — 남는 변동은 **전부 모순**이다.
#:
#: 실측 (`fp_ratio.py`, 설계 8 개, `fp_ratio_spread` 추정자, 재측정):
#:
#:     함의 A_t 변동폭   0.01 ~ **2.13 %**   ← 문턱은 이것의 4 배
#:     (참고) F/P 변동폭  1.52 ~ 20.29 %     옛 추정자. 차이가 C_f 의 압력 의존
#:
#: 문턱은 관측 최대의 4 배 — 멀쩡한 입력을 거부하는 쪽이 더 나쁘고, 실측
#: 데이터에는 센서 잡음·침식연소·목 침식이 더 보태기 때문이다.
#: 🔜 실제 연소시험 데이터로 재확인.
#:
#: ⚠⚠ ** 추정자를 `F/P` 에서 함의 노즐목으로 바꾸고 문턱을 20 % → 10 %
#:     로 내렸다.** 이유가 둘인데 둘 다 측정이다.
#:
#:     ① 창을 구간으로 고치자 **옛 추정자의 최악이 20.29 % 가 되어 옛 문턱
#:        20 % 를 넘었다** — 6-로브 깊음은 자기 일관된 설계인데 거부된다.
#:        `F/P` 는 `C_f` 의 압력 의존만큼 **정당하게** 변하므로, 압력 범위가
#:        넓은 설계일수록 멀쩡한 입력이 거부된다. 구조적 결함이다.
#:     ② `C_f` 로 나누면 그 변화가 사라진다. 검산: 합성한 쌍에 참 ε 를 주면
#:        함의 `A_t` 가 80.000 mm² (참값 80.0), 변동 **0.000 %** 다. 남는
#:        잔차는 자동 ε 가 참 ε 와 다른 데서만 온다.
#:
#:     부수 효과로 **검사가 세졌다**: 부스트 압력만 2 배 틀린 쌍(함의 노즐목이
#:     절반이 되는 쌍)을 옛 준위 마스크는 0.00 % 로 통과시켰는데 이제 61.45 % 로
#:     걸린다. 문턱을 절반으로 내렸는데 거짓 양성은 오히려 줄었다 —
#:     물리를 나눠 버렸기 때문이다.
#:
#: ⚠ 이 검사가 잡는 것과 못 잡는 것을 구분할 것.
#:    잡는다   : 두 곡선의 시간축 불일치, 서로 다른 시험에서 온 쌍, 목 침식,
#:               한쪽만 필터링된 데이터
#:    못 잡는다 : **균일한 배율 오차** (bar↔MPa, 게이지압↔절대압). 비율이 통째로
#:               옮겨갈 뿐 변동폭은 그대로다. 그래서 함의된 `A_t` 범위를 따로
#:               본다 — 그 검사는 `C_f` 모델을 가정하면 안 되므로 **`F/P` 날것**
#:               (`fp["fp_mean"]`)을 쓴다.
AT_SPREAD_MAX: float = 0.10

#: 조기 종료가 "좋아졌다" 로 셀 **상대 개선폭**. 잡음 방어가 유일한 목적이고,
#: 그래서 부동소수 잡음(1e-15 수준)보다 크고 **어떤 실제 진전보다 작게** 잡는다.
#:
#: 측정으로 정했다 (`loss_noramp` 궤적 120 반복, 고정 목 Ø10.70 mm, 6-로브
#: 깊음 목표). 이 값은 판정에 **민감하지 않다** — K=0 의 종료 반복이
#:
#:     문턱  0 / 1e-9 / 1e-6 / 1e-4 / 1e-3 → 종료 반복 115,  1e-2 → 비슷
#:
#: 로 거의 같다. 그래서 작은 쪽을 고른다.
#:
#: ⚠ 한때 1 % 로 잡고 "허용 10 % 에 대해 손실이 1 % 씩도 못 내려가면 가망이
#:    없다" 고 적었다. **척도를 혼동한 것이다** — 손실은 힌지 **제곱** 합이라
#:    오차 %p 와 단위가 다르고, 손실의 1 % 는 오차로 보면 훨씬 작은 변화다.
#:    철회한다.
STALL_REL_TOL: float = 1e-9

#: 대표 `C_f` 의 물리적 범위 — 함의된 `A_t = (F/P)/C_f` 로 단위 오류를 잡는다.
#: 종이 지관·KNSB 대역에서 `C_f` 는 1.1~1.5 이지만, 넉넉하게 잡아도 이 범위를
#: 벗어날 수 없다 (진공 추력계수 상한과 음속 노즐 하한).
CF_PLAUSIBLE: tuple = (0.8, 2.0)


def canonical_rotation(a: torch.Tensor, b: torch.Tensor):
    """포트를 정규 방향으로 돌린다 — 회전 게이지 고정.

    포트를 δ 만큼 돌려도 A_b(w) 는 **정확히** 같다. 즉 회전은 손실의 1차원
    대칭이고, 그 방향으로는 최적화할 것이 없다. 그런데 거리장이 정사각 격자
    위에서 계산되므로 수치적으로는 대칭이 깨져 있다 — 실측에서 포트를 돌릴 때
    A_b 가 0.05 ~ 0.12 % 흔들렸다. 참 미분이 0 인 방향이라 **그 잡음이 그대로
    경사가 되고**, 최적화기는 회전 궤도를 따라 의미 없이 돌아다닌다.

    그래서 매 스텝 뒤에 **진폭이 가장 큰 모드의 위상을 0 으로** 맞춰 궤도 위
    한 점으로 사영한다. 부수 효과로 출력 형상의 방향이 정규형이 되어 설계끼리
    비교하기 쉬워진다.

    ⚠ Adam 의 모멘텀은 함께 돌리지 않는다. 회전각이 스텝마다 작으므로 영향이
       작지만, 완전히 깔끔하지는 않다.
    """
    with torch.no_grad():
        amp = torch.sqrt(a ** 2 + b ** 2)
        m = int(torch.argmax(amp)) + 1                     # 지배 모드 (1-based)
        if float(amp[m - 1]) < 1e-9:
            return
        delta = -torch.atan2(b[m - 1], a[m - 1]) / m
        k = torch.arange(1, a.shape[0] + 1, dtype=a.dtype, device=a.device)
        c, s_ = torch.cos(k * delta), torch.sin(k * delta)
        a_new = a * c - b * s_
        b_new = a * s_ + b * c
        if float(a_new[m - 1]) < 0:                        # a_m > 0 으로 부호 고정
            delta = delta + np.pi / m
            c, s_ = torch.cos(k * delta), torch.sin(k * delta)
            a_new, b_new = a * c - b * s_, a * s_ + b * c
        a.copy_(a_new)
        b.copy_(b_new)


@dataclass
class DesignVars:
    r0: torch.Tensor
    a: torch.Tensor
    b: torch.Tensor
    a_throat: torch.Tensor
    lengths: list

    def tensors(self):
        return [self.r0, self.a, self.b, self.a_throat]


def seed_design(r0: float, n_modes: int = 4, a_throat: float = None,
                lengths=(120.0,), seed_amplitude: float = 0.3, seed: int = 0,
                dtype=torch.float64, *, throat: ThroatSpec) -> DesignVars:
    """초기 설계 — 대칭을 깬 상태로 만든다 (위 경고 참조).

    `throat` 는 **필수**다. `a_throat` 는 그 안에서의 출발값을 고르는 선택
    인자이고(다중출발에서 유용하다), 주지 않으면 `throat` 에서 나온다 —
    고정이면 그 값, 범위면 구간 중앙이다. 어느 쪽이든 **입력 상자 안으로
    잘라 넣는다**: 범위 밖에서 출발하면 첫 clamp 에서 경계로 튕겨 나가
    출발점을 고른 의미가 없어진다.

    초판은 `a_throat=80.0` 이 **기본 인자**였다. 그래서 호출자가 노즐목
    입력을 전혀 안 넘겨도 조용히 80 mm² 에서 출발해 자유롭게 떠다녔고,
    그것이 측정한 압력 106.87 % 이탈의 출발점이었다.
    """
    g = np.random.default_rng(seed)
    a0 = g.normal(0.0, seed_amplitude, n_modes)
    b0 = g.normal(0.0, seed_amplitude, n_modes)
    mk = lambda v: torch.tensor(v, dtype=dtype, requires_grad=True)
    lo, hi = throat.bounds_mm2
    at0 = throat.start_mm2() if a_throat is None else float(a_throat)
    at0 = min(max(at0, lo), hi)
    return DesignVars(r0=mk(float(r0)), a=mk(a0), b=mk(b0),
                      a_throat=mk(at0), lengths=list(lengths))


def evaluate(motor: DiffMotor, dv: DesignVars, target: Target,
             limits: ManufacturingLimits, weights: LossWeights,
             sparsity_w: float = 0.0, tol: Tolerances = None):
    """한 번의 정방향 + 손실. 경사는 호출자가 backward 로 받는다."""
    t, p, f, diag = motor.simulate(dv.r0, dv.a, dv.b, dv.lengths, dv.a_throat)

    # 웹 궤적을 시간축으로 — ω(t) 의 소진 경계 판정용
    #
    # ⚠ **스텝별 `h_list` 로 적분한다. `diag["h"]`(평균)로 하면 안 된다.**
    #    적분 스케줄은 강성 가드 때문에 일부러 불균일한데(`_build_schedule`),
    #    평균 스텝으로 쌓으면 웹이 엉뚱한 시각에 소진된 것으로 나온다. 그
    #    시각이 `shape_window` 의 테일오프 경계라서 **모양 사양이 보는 구간이
    #    틀어진다.**
    #
    #    실측 (r₀=11, a₃=2.2, A_t=78): 평균 스텝 4.26 s vs 실제 스케줄 8.76 s
    #    (2.1 배). 같은 보고서의 `rms_shape_pct` 가 3.624 % → **2.326 %** 로
    #    바뀐다 — 허용 10 % 중 1.3 %p 가 이 한 줄에서 나오고 있었다.
    #    `forward.py` 가 `diag["h"]` 를 `# 보고용 평균` 이라고 적어 둔 것이
    #    덫이었다.
    with torch.no_grad():
        w_cap = float(diag["w_tab"][-1])
        r_of_t = motor.br.rate(p)
        h_steps = torch.as_tensor(np.asarray(diag["h_list"], dtype=float),
                                  dtype=p.dtype)[:r_of_t.shape[0] - 1]
        w_traj = torch.cat([torch.zeros(1, dtype=p.dtype),
                            torch.cumsum(r_of_t[:-1] * h_steps, 0)])
        w_traj = torch.clamp(w_traj, max=w_cap)

    omega = shape_window(t, p, w_traj, w_cap, weights)
    l_curve, curve_terms, rep = curve_loss(t, f, p, omega, target, weights, tol)

    pen, pen_terms, pen_rep = manufacturing_penalty(
        motor.ds, diag["phis"][0], dv.r0, dv.a, dv.b, limits, motor.ds.r_outer,
        length=float(dv.lengths[0]))
    l_hard = hard_pressure_penalty(p, limits.p_case_allow)
    l_mass = mass_penalty(diag["mass_g"], limits.mass_max)

    l_sparse = group_l1(dv.a, dv.b, motor.ds.r_outer)
    # ── 램프 없는 부분을 따로 들고 있는다 (`loss_noramp`)
    #
    #    `sparsity_ramp` 가 반복마다 `sparsity_w` 를 **단조 증가**시키므로
    #    `loss` 는 **반복 간 비교가 불가능**하다 (설계가 전혀 안 변해도 값이
    #    오른다). 그런데 조기 종료는 "아직 좋아지고 있나" 를 묻는 일이라
    #    반복 간 비교가 필수다. 그래서 램프가 들어가는 항만 뺀 값을 둔다 —
    #    사양 힌지 + 제작 벌점 + 장약 + 케이스압은 모두 가중치가 고정이다.
    l_noramp = (l_curve + weights.manufacturing * (pen + l_mass)
                + weights.case_pressure * l_hard)
    loss = l_noramp + sparsity_w * l_sparse

    # ── 로브 충전 한계 — 경사 없는 진단값. `admissibility` 의
    #    제작 게이트가 이 값을 읽는다.
    fill = fill_feasible(dv.r0, dv.a, dv.b, limits)
    rep.update(pen_rep)
    rep.update({"fill_ok": fill["ok"], "lobe_depth_mm": fill["depth"],
                "lobe_width_mm": fill["width"], "lobe_aspect": fill["aspect"],
                "lobe_severity": fill["severity"],
                "fill_over_pp": fill["over_pp"],
                "loss": float(loss.detach()),
                #: 램프가 빠진 손실 — **반복 간 비교가 유효한 유일한 값**이고
                #: 조기 종료 판정이 이것을 본다 (`optimize`).
                "loss_noramp": float(l_noramp.detach()),
                #: 사양 힌지 합만 — 제작 벌점이 섞이지 않은 진단값
                "l_curve": float(l_curve.detach()),
                "mass_g": diag["mass_g"],
                "expansion_ratio": diag["expansion_ratio"], "h_ms": diag["h"] * 1e3,
                "n_steps": diag["n_steps"], "tau_min_ms": diag["tau_min"] * 1e3,
                "penalty": float(pen.detach()),
                "case_over": float(l_hard.detach()),
                "complexity": float((dv.a ** 2 + dv.b ** 2).sqrt().sum().detach()),
                "sparsity_w": sparsity_w,
                "sliver_frac": float(diag["volume"][-1].detach()
                                     / diag["volume"][0].detach())})
    return loss, rep, (t, p, f, diag, omega)


def optimize(motor: DiffMotor, dv: DesignVars, target: Target,
             limits: ManufacturingLimits = None, weights: LossWeights = None,
             iters: int = 120, lr_shape: float = 0.08, lr_throat: float = 0.4,
             log_every: int = 10, verbose: bool = True,
             tol: Tolerances = None, stop_in_set: bool = True,
             patience: int = 25, *, throat: ThroatSpec):
    """Adam 루프. 반환: (최적 설계, 기록).

    **`throat` 는 필수 입력이다** (`ThroatSpec`, §4.4 결정 D). 기본값을
    두지 않는 이유는 그 대가를 쟀기 때문이다 — `A_t` 가 자유 변수인
    동안 최적화기는 목을 0.40 배로 조여 압력을 2.07 배로 올리고도 "추력 10 %
    안" 을 보고했다 (`experiments/pressure_results.md`). 조용한 기본값은
    그 동작을 되살린다.

    **되돌릴 반복은 손실이 아니라 '집합 안인가' 로 고른다.**

    초판은 `argmin loss` 를 되돌렸다. 힌지에서는 그것이 집합 안을 보장하지 않는다.
    집합 안에서 사양 세 항은 정확히 0 이므로 손실 = 2차 기준(최대압·희소성)뿐이고,
    집합 **밖** 반복이 사양 초과는 작으면서(힌지는 2차식이라 문턱 근처에서 작다)
    2차 기준이 더 좋으면 손실이 더 낮다. 실측(6-로브, K=1 웜 스타트):

        반복 18 (집합 **안**) : 손실 1.062e-03  모양 6.80  I  9.93  F  0.04  P오차 103.0 %
        반복  2 (집합 **밖**) : 손실 6.945e-04  모양 6.61  I 10.51  F 11.19  P오차  72.6 %
        → `argmin loss` 가 반복 2 를 골라 **집합 밖 설계를 반환**했다. 그래서 K 탐색이
          K=2, K=3 까지 올라가고 끝내 "실현 불가 0.2 %p" 를 출력했다 — 허용 설계가
          반복 18 에 이미 있었는데도.

    즉 가중치 1e-3 의 **가장 약한 기준이 합격 판정을 이기고** 있었다. 손실은 또
    `sparsity_ramp` 때문에 반복마다 척도가 달라 **반복 간 비교가 애초에 불가능**하다.

    새 규칙 — 두 묶음을 따로 추적하고 집합 안을 **무조건** 우선한다.

        집합 안 반복 → 2차 기준 `w_pmax·(ΔP/P)² + w_sparse·Σ|c|/R` 최소 (고정 가중치)
        집합 밖 반복 → (집합까지 거리 `worst_gap_pp`, 제약 위반량, 손실) 사전식 최소

    `in_set` 판정은 `admissibility()` 를 **그대로** 쓴다. 초판의 루프 내 판정은
    사양·벌점·케이스압만 보고 **웹 두께·꼭짓점 반경·장약 질량·연결성을 빠뜨려서**,
    `admissibility` 가 떨어뜨릴 설계에서 "집합 진입" 으로 멈출 수 있었다. 연결성
    검사는 13 ms 로 한 반복(4.2 s)의 0.3 % 라 매 반복 계산해도 무리가 없다.

    ⚠ 스냅샷은 `opt.step()` **앞**에서 뜬다. 초판은 스텝 뒤에 떠서 반복 `it` 의
       보고서와 반복 `it+1` 의 설계 변수를 짝지었다 — 집합 판정과 돌려주는 설계가
       다른 반복일 수 있다는 뜻이다.

    **조기 종료는 `loss_noramp` 를 본다.** 되돌릴 반복을 고르는
    규칙과 **분리**되어 있다 — 두 질문이 다르기 때문이다.

        되돌릴 반복 : "지금까지 본 것 중 가장 좋은 설계가 어느 반복인가"
        조기 종료   : "아직 좋아지고 있는가"

    ⚠ 초판은 둘을 같은 카운터(`best_it`)로 처리했고, 그 키의 첫 원소가
       `worst_gap_pp` — **사양 중 최악 하나**였다. `max` 이므로 다른 사양이
       좋아지는 중에도 25 반복 제자리면 끊긴다. 그리고 이름이 `"손실 정체"`
       였다. 그 조합이 n=321 결과를 "실현 불가" 로 오독하게 만들었다
       (`experiments/gradient_audit.md` §3).

       **그 구조는 높은 K 를 체계적으로 불리하게 만든다** — 웜 스타트는 새
       모드를 0.25 mm 로만 흔들어 시작하므로 그 모드가 자라는 동안 최악
       사양이 거의 안 움직인다. 왕복 검증 실측 (고정 목 Ø10.70 mm):

           K=0 120/120   K=1 57/120(조기)   K=2 39/120(조기)   K=3 26/120
           새 모드 진폭 0.01 → 0.61 → 1.03 mm  (진폭 바닥 0.60 mm)

       K 가 올라갈수록 반복이 단조 감소하고, 모드가 바닥을 넘기기도 전에
       잘렸다.

    ⚠ **`rep["loss"]` 로는 이 판정을 할 수 없다.** `sparsity_ramp` 가
       `sparsity_w` 를 단조 증가시키므로 설계가 전혀 안 변해도 `loss` 가
       오른다 — 반복 간 비교가 애초에 불가능하다. `loss_noramp` 는 가중치가
       고정인 항만 모은 값이고, 그래서 비교가 유효하다.
    """
    limits = limits or ManufacturingLimits()
    weights = weights or LossWeights()
    tol = tol or Tolerances()
    # ── 노즐목 입력을 먼저 적용한다 (Adam 을 만들기 **전**에)
    #
    #    고정 모드에서는 변수를 얼리고 값을 입력으로 **덮어쓴다.** 씨앗이 들고
    #    온 값을 그냥 두면 "입력으로 받았다" 가 거짓이 된다.
    at_lo, at_hi = throat.bounds_mm2
    with torch.no_grad():
        if throat.is_fixed:
            dv.a_throat.fill_(throat.area_mm2)
        else:
            dv.a_throat.clamp_(min=at_lo, max=at_hi)
    if throat.is_fixed:
        dv.a_throat.requires_grad_(False)
    opt = torch.optim.Adam([
        {"params": [dv.r0, dv.a, dv.b], "lr": lr_shape},
        {"params": [dv.a_throat], "lr": lr_throat},
    ])
    hist, stop = [], ("반복 수 소진", iters - 1)
    best_in, best_out, best_it = None, None, 0
    # ── 조기 종료 전용 카운터 — `best_it` 과 **분리**한다 (docstring 참조).
    #    `best_it` 은 "되돌릴 반복", 이쪽은 "아직 좋아지는가" 를 센다.
    #
    #    ⚠ `None` 으로 시작한다. `float("inf")` 로 두면 첫 비교가
    #       `inf - tol*abs(inf)` = **nan** 이 되고, `x < nan` 은 항상 False 라
    #       카운터가 **한 번도 갱신되지 않는다** — `stall_it` 이 0 에 고정돼
    #       `patience` 가 반복 25 에서 그냥 터진다. 초판이 그렇게 써서 K=0 이
    #       120 → 26 반복으로 줄었고, 하필 "조기 종료" 증상과 똑같이 보여서
    #       문턱 값을 의심하게 만들었다.
    stall_best, stall_it = None, 0
    for it in range(iters):
        opt.zero_grad()
        sw = sparsity_ramp(it, iters, weights)
        loss, rep, aux = evaluate(motor, dv, target, limits, weights,
                                  sparsity_w=sw, tol=tol)
        # ── 보고서와 **같은 반복**의 설계 변수를 뜬다 (스텝 앞)
        snap = (dv.r0.detach().clone(), dv.a.detach().clone(),
                dv.b.detach().clone(), dv.a_throat.detach().clone())
        rep["components"] = connected_components(motor.ds, aux[3]["phis"][0])
        adm = admissibility(rep, limits, tol)
        rep["iter"], rep["in_set"] = it, adm["in_set"]
        rep["worst_gap_pp"] = adm["worst_gap_pp"]
        # ⚠ 압력이 **사양**이면 2차 기준에서 뺀다. 넣어 두면 사양 밖
        #    기준이 집합 안 선택에 다시 끼어든다 — 이전의 오판과 같은 구조다.
        rep["secondary"] = (weights.sparsity * rep["complexity"]
                            / float(motor.ds.r_outer)
                            + (0.0 if rep.get("pressure_is_spec")
                               else weights.p_max * (rep["P_err_pct"] / 100.0) ** 2))
        hist.append(rep)
        if adm["in_set"]:                       # 집합 안은 무조건 우선
            key = rep["secondary"]
            if best_in is None or key < best_in[0]:
                best_in, best_it = (key, snap, it), it
        elif best_in is None:                   # 집합 안을 한 번도 못 봤을 때만
            key = (adm["worst_gap_pp"], rep["penalty"] + rep["case_over"],
                   rep["loss"])
            if best_out is None or key < best_out[0]:
                best_out, best_it = (key, snap, it), it
        if verbose and (it % log_every == 0 or it == iters - 1):
            print(f"  it {it:4d}  loss {rep['loss']:10.4e}  모양 {rep['rms_shape_pct']:6.2f}%"
                  f"  I {rep['I_err_pct']:6.2f}%  F {rep['F_err_pct']:6.2f}%"
                  f"  벌점 {rep['penalty']:8.2e}  복잡도 {rep['complexity']:5.2f}"
                  f"  장약 {rep['mass_g']:6.1f}g  {'집합 안' if adm['in_set'] else ''}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(dv.tensors(), 10.0)
        opt.step()
        with torch.no_grad():                       # 물리적 하한 (퇴화 방지)
            dv.r0.clamp_(min=2.0, max=motor.ds.r_outer - limits.w_web_min)
            # 노즐목은 **입력이 준 상자** 안에 가둔다. 초판은 여기에
            # `clamp_(min=5.0)` 한 줄이 있었다 — 면적 5 mm²(Ø2.52 mm) 라는
            # 물리 하한뿐이고 상한이 없어서, 목이 어디로든 떠다닐 수 있었다.
            dv.a_throat.clamp_(min=at_lo, max=at_hi)
        canonical_rotation(dv.a, dv.b)          # 회전 게이지 고정
        # ── 종료 조건 — 반복 수가 아니라 집합 진입/정체
        if stop_in_set and adm["in_set"]:
            stop = ("집합 진입", it)
            break
        # ── 아직 좋아지고 있는가 — 램프 없는 손실로 센다
        cmp_now = float(rep["loss_noramp"])
        if (stall_best is None
                or cmp_now < stall_best - STALL_REL_TOL * abs(stall_best)):
            stall_best, stall_it = cmp_now, it
        if it - stall_it >= patience:
            # 이름이 사실과 맞는다 — 이 조건은 실제로 **손실**(램프 제외)을
            # 본다. 이전에는 `worst_gap_pp` 를 보면서 "손실 정체" 라고
            # 적었고, 그 불일치가 n=321 오독의 경로였다 (위 docstring).
            stop = ("손실 정체(램프 제외)", it)
            break
    best = best_in if best_in is not None else best_out
    with torch.no_grad():
        dv.r0.copy_(best[1][0]); dv.a.copy_(best[1][1])
        dv.b.copy_(best[1][2]); dv.a_throat.copy_(best[1][3])
    if hist:
        hist[-1]["stop_reason"], hist[-1]["stop_iter"] = stop
        hist[-1]["restored_iter"] = best[2]
        hist[-1]["restored_from"] = "집합 안" if best_in is not None else "집합 밖"
        # ── 멈출 때 손실이 아직 내려가고 있었는가 (조기 종료인지 가리는 값)
        # ⚠ 기울기도 **램프 없는** 손실로 잰다. 램프가 든 `loss` 로 재면
        #    후반부에 인공적인 상승 드리프트가 섞여 "정체" 가 거짓으로 뜬다.
        tail = [float(h["loss_noramp"]) for h in hist[-min(10, len(hist)):]]
        slope = ((tail[-1] - tail[0]) / (len(tail) - 1)) if len(tail) > 1 else 0.0
        hist[-1]["loss_tail_slope"] = slope
        hist[-1]["stalled_metric"] = "램프 없는 손실"
        hist[-1]["loss_still_falling"] = bool(slope < 0.0)
        hist[-1]["stall_iter"] = stall_it
        hist[-1]["stall_best"] = stall_best
        # 어느 노즐목 모드로 돌았는지 기록 — 같은 보고서를 다른 모드의 것과
        # 섞어 읽는 일을 막는다
        hist[-1]["throat_mode"] = throat.mode
        hist[-1]["throat_desc"] = throat.describe()
        hist[-1]["a_throat_mm2"] = float(dv.a_throat.detach())
        hist[-1]["d_throat_mm"] = float(
            2.0 * np.sqrt(float(dv.a_throat.detach()) / np.pi))
    if verbose:
        print(f"  종료: {stop[0]} (반복 {stop[1]}), "
              f"되돌린 반복 {best[2]} ({'집합 안' if best_in is not None else '집합 밖'})")
    return dv, hist


#: 모델 바닥의 계수 — 바닥 ≈ `MODEL_FLOOR_K · eps_cells · h`.
#:
#: 격자 수렴 시험에서 측정 (`experiments/grid_convergence.py`).
#: 두 번 훑었다 — 격자(n=121·161·241·321, eps_cells 1.5 고정)와 완화 폭
#: (eps_cells 1.0·1.5·2.5, n=161 고정). 바닥은 **완화 폭을 따라간다**:
#:
#:     A/eps  측정 0.56 ~ 0.80   (폭 1.43×)
#:     A/h    측정 0.80 ~ 1.51   (폭 1.88×)  ← 격자만 훑으면 ~1.0 으로 보인다
#:
#: 계수는 측정 **상한**을 쓴다. 이 값은 "근거가 있다" 고 적을 문턱이므로
#: 틀릴 때 '근거 없음' 쪽으로 틀려야 한다.
MODEL_FLOOR_K: float = 0.80


def amplitude_floor(r0_for_scale: float = 12.0, h: float | None = None,
                    eps_cells: float = 1.5) -> tuple:
    """모드 진폭의 **바닥** [mm] 과 그것을 정한 쪽. 반환 `(바닥, "모델"|…)`.

        모델  `MODEL_FLOOR_K · eps_cells · h` — 시뮬레이터가 구분하지 못한다

    `h` 를 주지 않으면 `r0_for_scale` 로 옛 휴리스틱(`max(0.01·r₀, 0.05)`)을
    쓰고 두 번째 반환값에 `"모델(휴리스틱)"` 을 적는다 — **호출자가 격자를
    넘기지 않았다는 사실이 보고서에 남아야** 하기 때문이다. 새 코드는 반드시
    `h=motor.ds.h, eps_cells=motor.eps_cells` 를 넘긴다.

    ═══════════════════════════════════════════════════════════════════════
    ⚠⚠ **바닥은 하나다. 제작 쪽(δ)은 판정에서 뺐다.**

        이전에는 바닥이 둘이었고 "높은 쪽이 판정한다" 였다 — 제작 δ =
        0.020 mm 와 모델 바닥. 그 구조를 버린 이유는 **δ 가 측정이 아니기
        때문**이다.

        δ 의 사슬은 이렇게 적혀 있었다:

            테셀레이션 처짐   0.0198 mm  ← 지배항
            수축 보정 잔차    0.0012     (실측)
            출력 오차        0          (가정)
            압축 성형 전사    0          (가정)

        전제 두 개가 나머지를 0 으로 만든 뒤에는 지배항만 남는다. 그런데 그
        0.0198 은 `export/profile.py` 의 `DEFAULT_CHORD_TOL = 0.02` 가 만든
        값이다 — 공차를 0.01 로 바꾸면 처짐이 0.0099, 0.005 로 바꾸면 0.0050
        으로 **1:1 로 따라간다**. 즉 δ 는 내가 고른
        렌더링 공차이고, 실리콘·추진제의 전사 정밀도에 대한 정보가 아니다.
        **자기충족적인 값을 판정 문턱으로 쓸 수 없다.**

        게다가 δ 는 어차피 판정하지 못한다 — 모델 바닥이 운용 설정에서
        0.60 mm, δ 의 **30 배**다. δ 가 판정하려면 `eps_cells·h ≤ 0.025 mm`,
        즉 n ≳ 3600 (현 격자의 30 배, 비용 ~900 배) 이어야 한다.

        → 그래서 진폭 유의성은 **모델 바닥 하나로만** 판정한다. `DELTA_
          TRANSFER_MM` 은 상수로 남겨 두지만(전사 오차의 **상한 추정**으로서
          의미가 있다) 이 함수는 더 이상 그것과 비교하지 않는다.
    ═══════════════════════════════════════════════════════════════════════

    ⚠ 바닥은 **필요조건이지 충분조건이 아니다.** 같은 시험에서 거친 격자가
       작은 진폭을 '놓치는' 게 아니라 **진폭 전 구간에서 응답을 깎는다**는 것도
       드러났다 (n=121 은 기준 격자 대비 모드 효과를 0.42~0.91 만큼 틀리게 잰다).
       바닥 위에 있다고 진폭이 정확히 복원된다는 뜻은 아니다.
    """
    if h is None:
        return (max(0.01 * float(r0_for_scale), 0.05), "모델(휴리스틱)")
    return (MODEL_FLOOR_K * float(eps_cells) * float(h), "모델")


def feasibility_bounds(motor: DiffMotor, target: Target,
                       limits: ManufacturingLimits, lengths=(95.0,),
                       throat: ThroatSpec = None) -> dict:
    """**목표 자체**가 물리 한계를 넘는가 — 형상·경로와 무관한 사전 검사 (§4.6.3).

    ⚠ 최적화 뒤의 `admissibility()` 와 역할이 다르다. 그쪽은 "찾은 설계가 집합 안인가"
       를 묻고, 못 찾았을 때의 진단이 **경로에 의존한다.** 실측(실현 불가 목표, 씨앗
       4개)에서 걸리는 사양이 씨앗마다 달랐다 — 총역적 / 총역적 / 최대추력 / ….
       '가장 가까운 설계' 가 매번 다른 곳에 멈추니 그 지점에서 무엇이 가장 모자란지도
       달라지는 것이다.

       여기서는 **목표와 케이스만 보고** 판정하므로 경로가 끼어들지 않는다. 넘으면
       "어떤 형상으로도 불가능" 이 확정이고, 안 넘어도 "가능하다" 는 뜻은 아니다 —
       **필요조건**일 뿐이다.

    총역적 한계
        I = ∫F dt = ∫ C_f·c*·ṁ dt ≈ c*_eff · C̄_f · m_prop
        m_prop ≤ min( ρ · π R² L ,  질량 상한 )          (포트가 0 일 때가 최대)
    ⚠ **전면 검토: 최대추력 한계는 여기서 검사하지 않는다.** 이 docstring 이
       `F = C_f·P_c·A_t ≤ C_f·P_허용·A_t,max` 를 2 대 한계의 하나로 제시하고
       있었는데, 반환 딕셔너리에 추력 관련 항이 **하나도 없고** `feasible` 은
       `I_feasible` (압력 곡선이 들어오면 + 압력 항)뿐이다. 총역적 2000 N·s /
       F_max 1e5 N 스파이크 목표로 확인했다 — 통과한다.
       노즐목이 주어졌을 때의 도달 추력 범위는
       `validate.input_check.check_physics` 의 `PHYS_THROAT_FIXED` 가 본다.

    `throat` 를 주면 **함의 노즐목 검사**를 입력과 맞대 본다 (압력 곡선이 같이
    들어온 경우). `A_t = (F/P)/C_f` 는 항등식이므로, 함의된 값이 입력 범위를
    벗어나면 그 (추력, 압력, 노즐목) 세 쌍은 **한 모터에서 나올 수 없다** —
    형상을 어떻게 잡아도 불가능하고, 최적화를 시작할 이유가 없다.
    """
    from ..core.nozzle import (area_ratio_to_pressure_ratio, gamma_function,
                               optimum_area_ratio)
    r_out = float(motor.ds.r_outer)
    l_tot = float(sum(lengths))
    m_geom = motor.density * np.pi * r_out ** 2 * l_tot / 1000.0      # [g]
    m_max = min(m_geom, limits.mass_max)
    g, pa = motor.gamma, motor.p_ambient
    # 대표 C_f — 설계 대역 중앙 압력의 최적 팽창 (낙관적으로 잡아야 '넘으면 확정')
    p_rep = 0.5 * limits.p_case_allow
    eps = optimum_area_ratio(p_rep, pa, g) if p_rep > pa else 1.0
    pe = area_ratio_to_pressure_ratio(eps, g)
    cf = gamma_function(g) * np.sqrt(2 * g / (g - 1) * (1 - pe ** ((g - 1) / g)))
    i_max = motor.c_star * motor.c_star_eff * cf * motor.cf_eff * (m_max / 1000.0)

    i_tgt = float(torch.trapezoid(target.f, target.t))
    out = {"m_max_g": m_max, "m_geom_g": m_geom, "cf_rep": float(cf),
           "I_max": float(i_max), "I_target": i_tgt,
           "I_over_pct": 100.0 * (i_tgt / max(i_max, 1e-9) - 1.0),
           "I_feasible": i_tgt <= i_max}
    out["feasible"] = out["I_feasible"]
    # ── 추력·압력 두 곡선이 서로 모순인가 — 압력 곡선 입력 시만.
    #    A_t 는 시간에 대해 **고정**이므로 F/P = C_f·A_t 이고, C_f 의 압력 의존만큼만
    #    변해야 한다. 문턱은 `AT_SPREAD_MAX` = **10 %** 이고 근거는 그 상수에
    #    적어 두었다 (채택 추정자 실측 0.88~2.39 % 의 약 8 배).
    #
    #    ⚠ 전면 검토 정정: 이 자리에 "원형 2.1~3.2 %, 스타 10.7~19.9 %,
    #      문턱은 관측 최대의 3 배인 60 %" 라고 적혀 있었다. 세 가지가 틀렸다 —
    #      숫자는 설계 문서가 **폐기**로 표시한 `f > 0.05·F_max` 창의 것이고,
    #      배수도 다르고, 60 % 는 코드가 쓰는 문턱이 아니다. 60 % 로 읽으면
    #      20 % 에서 거부되는 입력을 "통과한다" 고 판단하게 된다.
    #
    #    ⚠ 필요조건일 뿐이고, 실측 데이터에는 침식연소·목 침식이 더 보태므로 시험
    #    데이터로 문턱을 재확인해야 한다.
    out["fp_ratio_checked"] = False
    if getattr(target, "p", None) is not None:
        # ── 목표 최대압이 케이스 허용압을 넘으면 그 자체로 확정 불가.
        #    어떤 형상이 이 압력 곡선을 재현하더라도 케이스가 터진다. 균일 배율
        #    오차(bar↔MPa)를 잡는 것도 실질적으로 이 검사다 — F/P 변동폭 검사는
        #    배율에 둔감하고, 함의 A_t 검사는 ×10 을 통과시킨다 (실측 확인).
        out["P_target_max"] = float(target.p_max)
        out["P_case_allow"] = float(limits.p_case_allow)
        if target.p_max > limits.p_case_allow:
            out["P_feasible"] = False
            out["feasible"] = False
            out["note"] = (
                f"**이 압력 곡선으로는 맞는 형상을 찾을 수 없습니다 — 케이스가 "
                f"터집니다.**  목표 최대압 {target.p_max:.2f} MPa 가 케이스 허용압 "
                f"{limits.p_case_allow:.1f} MPa 를 "
                f"{100*(target.p_max/limits.p_case_allow-1):.0f} % 넘습니다. "
                f"형상과 무관하므로 역설계를 돌릴 필요가 없습니다. "
                f"조치: 압력 단위를 확인하거나(bar 면 ÷10, psi 면 ÷145), "
                f"케이스 허용압 입력을 올리거나, 목표 압력을 낮추십시오.")
            return out
        out["P_feasible"] = True
        # ⚠ 노즐목을 입력으로 알면 ε 적합의 퇴화 가지(ε 를 키우면 C_f 가 0 을
        #   지나 함의 A_t 가 발산하면서 변동폭이 **가짜로 0** 이 되는 구간)를
        #   더 좁게 막을 수 있다. 다만 입력 상자를 **그대로** 쓰면 안 된다 —
        #   고정 모드의 상자는 폭이 0 이라 어떤 ε 도 후보가 되지 못하고, 적합이
        #   통째로 실패해 옛 추측으로 되돌아간다 (실제로 그렇게 실패했다).
        #
        #   이 상자의 목적은 **발산 가지 배제**이지 노즐목 검증이 아니다.
        #   노즐목이 입력과 맞는지는 아래 `throat_consistent` 가 따로 본다.
        #   그래서 넉넉하게 ×[1/3, 3] 로 벌린다 — 1e8 mm² 는 확실히 막고 참값은
        #   확실히 통과시키는 폭이다.
        at_box = None
        if throat is not None:
            lo_t, hi_t = throat.bounds_mm2
            at_box = (lo_t / 3.0, hi_t * 3.0)
        fp = fp_ratio_spread(
            target.t, target.f, target.p,
            gamma=motor.gamma, p_ambient=motor.p_ambient,
            cf_eff=motor.cf_eff, at_bounds=at_box,
            # 정방향과 같은 C_f — `input_check` 쪽 같은 주석 참조.
            separation_frac=getattr(motor, "separation_frac", 0.0))
        if fp is not None:
            # 함의된 노즐목 범위 — 단위 오류 검사. F/P = C_f·A_t 이고 C_f 가
            # CF_PLAUSIBLE 안이어야 하므로 A_t ∈ [ratio/C_f,max, ratio/C_f,min].
            # 그 구간이 물리적으로 가능한 목 면적과 겹치지 않으면 단위가 틀렸다
            # (예: 압력을 psi 로 넣으면 비율이 145 배 작아져 A_t < 1 mm² 가 된다).
            # ⚠ 단위 검사는 **`F/P` 날것**을 쓴다 (`fp_mean`), 함의 노즐목이
            #    아니다. 여기서 잡으려는 것이 단위 오류이므로 `C_f` 모델을
            #    가정하면 안 된다 — 넓은 `CF_PLAUSIBLE` 로 감싸는 게 요점이다.
            at_lo = fp["fp_mean"] / CF_PLAUSIBLE[1]
            at_hi = fp["fp_mean"] / CF_PLAUSIBLE[0]
            # 하한 = 목 직경 물리 하한, 상한 = 케이스 단면
            at_phys = (float(np.pi * (D_THROAT_MIN_MM / 2.0) ** 2),
                       np.pi * r_out ** 2)
            at_ok = at_hi >= at_phys[0] and at_lo <= at_phys[1]
            out.update(fp_ratio_checked=True, fp_ratio_mean=fp["fp_mean"],
                       at_implied_mean=fp["at_mean"],
                       at_spread_pct=100.0 * fp["spread"],
                       fp_spread_pct=100.0 * fp["fp_spread"],
                       fp_eps=fp["eps"], fp_n=fp["n"],
                       fp_at_implied=(at_lo, at_hi), fp_units_ok=at_ok,
                       fp_consistent=fp["spread"] <= AT_SPREAD_MAX and at_ok)
            out["feasible"] = out["I_feasible"] and out["fp_consistent"]
            if not at_ok:
                out["note"] = (
                    f"**입력한 압력의 단위가 틀린 것 같습니다.**  F/P = C_f·A_t "
                    f"이므로 입력에서 함의되는 노즐목은 {at_lo:.2f} ~ {at_hi:.2f} mm² "
                    f"인데 (C_f 를 {CF_PLAUSIBLE[0]}~{CF_PLAUSIBLE[1]} 로 잡음), "
                    f"물리적으로 가능한 범위 {at_phys[0]:.0f} ~ {at_phys[1]:.0f} mm² "
                    f"(가공 한계 ~ 케이스 단면)와 겹치지 않습니다. "
                    f"압력은 **MPa 절대압**, 추력은 **N** 으로 넣으십시오 "
                    f"(bar 면 ÷10, psi 면 ÷145).")
                return out
            if not out["fp_consistent"]:
                out["note"] = (
                    f"**입력한 두 곡선이 서로 모순입니다 — 한 모터에서 나올 수 "
                    f"없습니다.**  노즐목 A_t 는 시간에 대해 고정이므로 "
                    f"입력에서 함의되는 A_t = (F/P)/C_f(P, ε) 가 **상수여야** "
                    f"하는데 {100*fp['spread']:.1f} % 흔들립니다 "
                    f"(허용 {100*AT_SPREAD_MAX:.0f} %, 모델 설계 8 종 실측 최대 "
                    f"2.13 %, 함의 A_t ≈ {fp['at_mean']:.1f} mm², ε {fp['eps']:.2f}). "
                    f"C_f 의 압력 의존은 이미 나눠 냈으므로 이 변동은 그것으로 "
                    f"설명되지 않습니다 (참고: 나누기 전 F/P 변동 "
                    f"{100*fp['fp_spread']:.1f} %). "
                    f"조치: 두 곡선이 같은 시험의 같은 시간축에서 온 것인지, "
                    f"압력이 절대압 [MPa] 인지(게이지압이면 대기압을 더해야 합니다), "
                    f"목 침식으로 A_t 가 변한 시험이 아닌지 확인하십시오.")
                return out
            # ── 함의 노즐목 vs **입력한 노즐목**
            #
            #    `A_t = (F/P)/C_f` 는 항등식이다. 두 곡선과 노즐목이 전부
            #    입력이면 셋이 서로를 결정하므로, 함의된 값이 입력 범위를
            #    벗어나면 그 세 쌍은 **한 모터에서 나올 수 없다.** 형상으로
            #    메울 수 없으니 최적화를 시작할 이유가 없다.
            #
            #    ⚠ 범위 모드에서는 구간 전체를 본다 — 구간 안에 들어오기만
            #       하면 통과다.
            if throat is not None:
                lo, hi = throat.bounds_mm2
                at_imp = float(fp["at_mean"])
                # 함의 추정자 자신의 잔차(자동 ε 가 참 ε 와 다른 데서 온다)를
                # 감싸 준다 — 실측 최대 2.13 %, 그 2 배로 잡는다.
                slack = 1.0 + 2.0 * 0.0213
                out.update(throat_mode=throat.mode,
                           throat_input_mm2=(lo, hi),
                           throat_implied_mm2=at_imp,
                           throat_consistent=(lo / slack <= at_imp <= hi * slack))
                if not out["throat_consistent"]:
                    out["feasible"] = False
                    d_imp = 2.0 * np.sqrt(at_imp / np.pi)
                    out["note"] = (
                        f"**입력한 추력·압력 곡선이 입력한 노즐목과 맞지 "
                        f"않습니다 — 한 모터에서 나올 수 없습니다.**  "
                        f"F = C_f·P·A_t 이므로 두 곡선에서 함의되는 노즐목은 "
                        f"{at_imp:.2f} mm² (Ø{d_imp:.2f} mm) 인데, 입력한 "
                        f"노즐목은 {throat.describe()} 입니다. 형상을 어떻게 "
                        f"잡아도 이 셋을 동시에 만족시킬 수 없습니다. "
                        f"조치: 노즐목 입력을 Ø{d_imp:.2f} mm 쪽으로 고치거나, "
                        f"두 곡선이 같은 시험에서 온 것인지 확인하십시오.")
                    return out
    if not out["I_feasible"]:
        out["note"] = (
            f"**이 곡선으로는 주어진 조건에서 맞는 형상을 찾을 수 없습니다 — "
            f"형상 문제가 아니라 물리 한계입니다.**  "
            f"목표 총역적 {i_tgt:.0f} N·s 는 이 케이스가 낼 수 있는 최대 "
            f"{i_max:.0f} N·s 를 {out['I_over_pct']:.0f} % 넘습니다 "
            f"(장약 상한 {m_max:.0f} g = min(케이스 가득 {m_geom:.0f} g, "
            f"입력 상한 {limits.mass_max:.0f} g), 대표 C_f {cf:.2f}). "
            f"포트 모양을 어떻게 바꿔도 넘을 수 없으므로 역설계를 돌릴 필요가 없습니다. "
            f"조치: 목표 총역적을 낮추거나, 케이스를 키우거나, 장약 상한을 올리십시오.")
    else:
        out["note"] = (f"물리 한계 안 — 목표 총역적 {i_tgt:.0f} N·s 는 상한 "
                       f"{i_max:.0f} N·s 의 {100*i_tgt/i_max:.0f} % 다. "
                       f"(필요조건일 뿐 실현 가능을 보장하지 않는다.)")
        if out["fp_ratio_checked"]:
            out["note"] += (f" 입력한 두 곡선은 서로 모순이 아니다 — 함의 노즐목 "
                            f"변동 {out['at_spread_pct']:.2f} % "
                            f"(허용 {100*AT_SPREAD_MAX:.0f} %), 함의 A_t ≈ "
                            f"{out['at_implied_mean']:.1f} mm² (ε {out['fp_eps']:.2f}). "
                            f"단위 검사: F/P ≈ {out['fp_ratio_mean']:.1f} mm² → "
                            f"노즐목 {out['fp_at_implied'][0]:.0f}~"
                            f"{out['fp_at_implied'][1]:.0f} mm² 대역. "
                            f"압력이 사양이므로 사양은 **5개**다.")
    return out


def admissibility(rep: dict, limits: ManufacturingLimits,
                  tol: Tolerances = None) -> dict:
    """허용 집합 A = S_사양 ∩ S_제작 안에 있는가, 아니면 **무엇이 얼마나** 모자라는가.

    ⚠ **못 들어가면 억지로 끼워맞추지 않는다**. 모든 사양을
       허용 오차 안에 넣을 수 없으면 우선순위가 낮은 항목을 양보해 설계 하나를
       내놓는 것이 아니라, **"이 곡선으로는 주어진 조건에서 맞는 형상을 찾을 수
       없습니다"** 라고 보고하고 무엇이 얼마나 모자라는지 적는다.

       억지로 내놓으면 두 가지가 나쁘다.

       1. 반환된 설계가 **목표가 아니라 최적화 경로**(씨앗·반복 수)에 의해 정해진다.
          실현 불가 목표에서는 돌릴 때마다 다른 설계가 나오는데, 사용자는 그것을
          '답' 으로 읽는다.
       2. 어느 사양을 얼마나 포기했는지가 숫자 하나(손실)에 뭉개져 보이지 않는다.
          "총역적이 14 %p 모자랍니다" 는 조치가 가능하지만 "손실 0.047" 은 아니다.
    """
    tol = tol or Tolerances()
    spec = {
        "모양":     (rep["rms_shape_pct"], 100 * tol.shape, rep["gap_shape_pp"]),
        "총역적":   (rep["I_err_pct"],     100 * tol.impulse, rep["gap_impulse_pp"]),
        "최대추력": (rep["F_err_pct"],     100 * tol.f_max,  rep["gap_f_max_pp"]),
    }
    # ── 압력 곡선이 입력으로 들어온 경우에만 사양이 둘 늘어난다.
    #    `curve_loss` 가 `gap_p_*` 를 넣어 줄 때만 집어간다 — 그래야 압력 곡선이
    #    없는 호출에서 "압력 사양 통과" 라는 빈 주장을 하지 않는다.
    if "gap_p_shape_pp" in rep:
        spec["압력 모양"] = (rep["P_shape_pct"], 100 * tol.p_shape,
                             rep["gap_p_shape_pp"])
    if "gap_p_max_pp" in rep:
        spec["최대압"] = (rep["P_err_pct"], 100 * tol.p_max, rep["gap_p_max_pp"])
    man = {
        "웹 두께":   (rep["web_mm"] >= limits.w_web_min,
                      f"{rep['web_mm']:.2f} / 한계 {limits.w_web_min:.1f} mm"),
        "꼭짓점 반경": (rep["r_tip_mm"] >= limits.r_tip_min,
                      f"{rep['r_tip_mm']:.2f} / 한계 {limits.r_tip_min:.1f} mm"),
        "장약 질량": (rep["mass_g"] <= limits.mass_max,
                      f"{rep['mass_g']:.0f} / 한계 {limits.mass_max:.0f} g"),
        "케이스압":  (rep["P_max_sim"] <= limits.p_case_allow,
                      f"{rep['P_max_sim']:.2f} / 한계 {limits.p_case_allow:.1f} MPa"),
        "연결성":    (rep.get("components", 1) == 1,
                      f"연결 성분 {rep.get('components', 1)} 개"),
        # ⚠ 라벨 정정: `penalty` 는 제작 **5 항 전부**의 합이다
        #    (웹·포트 양수·얇은 추진제·얇은 공극·꼭짓점 반경). 웹과 꼭짓점은
        #    이미 자기 행이 있으므로, 초판처럼 이 행을 "얇은 특징" 이라고
        #    적으면 웹이나 꼭짓점 위반이 **엉뚱한 라벨로 보고된다.**
        #    (전면 검토 §3.1)
        "제작 벌점 합": (rep.get("penalty", 0.0) < 1e-9,
                        f"벌점 {rep.get('penalty', 0.0):.2e} (제작 5 항 합)"),
        # ── 실리콘 충전 한계 — 추가.
        #    이 게이트가 없던 동안 폭 0.5 × 깊이 20 mm 로브(실측 미충전)가
        #    전부 통과했다. 깊이와 세장비를 **동시에** 본다.
        "로브 충전": (rep.get("fill_ok", True),
                     f"깊이²/폭 {rep.get('lobe_severity', 0.0):.1f} / 한계 "
                     f"{FILL_SEVERITY_MAX:.0f} mm "
                     f"(깊이 {rep.get('lobe_depth_mm', 0.0):.2f}, "
                     f"폭 {rep.get('lobe_width_mm', float('inf')):.2f} mm)"),
    }
    spec_ok = all(g == 0.0 for _, _, g in spec.values())
    man_ok = all(ok for ok, _ in man.values())
    gaps = {k: g for k, (_, _, g) in spec.items() if g > 0.0}
    binding = max(gaps, key=gaps.get) if gaps else None
    return {"in_set": spec_ok and man_ok, "spec_ok": spec_ok, "man_ok": man_ok,
            "spec": spec, "manufacturing": man, "gaps_pp": gaps, "binding": binding,
            "worst_gap_pp": max(gaps.values()) if gaps else 0.0,
            "violated_constraints": [k for k, (ok, _) in man.items() if not ok]}


def verdict_text(adm: dict, k_max: int = None) -> str:
    """사용자가 읽을 한 문단. 들어갔으면 설계, 못 들어갔으면 **실현 불가 보고**."""
    if adm["in_set"]:
        return "허용 집합 안이다 — 사양과 제작 제약을 모두 만족한다."
    lines = ["**이 곡선으로는 주어진 조건에서 맞는 형상을 찾을 수 없습니다.**"]
    if not adm["spec_ok"]:
        parts = [f"{k} {v:.1f} % (허용 {t:.0f} %, {g:.1f} %p 초과)"
                 for k, (v, t, g) in adm["spec"].items() if g > 0.0]
        lines.append("사양 미달: " + ", ".join(parts))
        lines.append(f"가장 크게 걸리는 것은 **{adm['binding']}** "
                     f"({adm['worst_gap_pp']:.1f} %p) 이다.")
    if not adm["man_ok"]:
        parts = [f"{k} ({d})" for k, (ok, d) in adm["manufacturing"].items() if not ok]
        lines.append("제작 제약 위반: " + ", ".join(parts))
    if k_max is not None:
        lines.append(f"(모드 수 K 를 {k_max} 까지 올려도 마찬가지였다.)")
    # 경계에 거의 닿은 경우는 "불가" 와 다르게 읽혀야 한다 — 실측에서 6-로브 목표가
    # 총역적 0.2 %p 초과로 걸렸다. 그 차이는 예산이나 문턱 재검토로 넘을 수 있다.
    if adm["spec_ok"] is False and 0.0 < adm["worst_gap_pp"] <= 1.0:
        lines.append(f"⚠ 다만 **경계에 매우 가깝다** ({adm['worst_gap_pp']:.2f} %p). "
                     f"반복 예산을 늘리거나 허용 오차를 재검토하면 들어갈 수 있다 — "
                     f"'실현 불가' 로 단정할 거리가 아니다.")
    lines.append("조치: 목표 곡선을 조정하거나, 케이스·장약 상한·노즐목 범위 같은 "
                 "입력 조건을 넓혀야 한다. 아래 '가장 가까운 설계' 는 **답이 아니라 "
                 "진단용** 이다.")
    return "  ".join(lines)


def final_report(motor: DiffMotor, dv: DesignVars, target: Target,
                 limits: ManufacturingLimits = None, weights: LossWeights = None,
                 tol: Tolerances = None):
    """수렴 후 판정 — 미분 불가능한 검사(연결성)까지 포함한다."""
    limits = limits or ManufacturingLimits()
    weights = weights or LossWeights()
    tol = tol or Tolerances()
    loss, rep, (t, p, f, diag, omega) = evaluate(motor, dv, target, limits, weights,
                                                 tol=tol)
    rep["components"] = connected_components(motor.ds, diag["phis"][0])
    rep["ok_web"] = rep["web_mm"] >= limits.w_web_min
    rep["ok_tip"] = rep["r_tip_mm"] >= limits.r_tip_min
    rep["ok_mass"] = rep["mass_g"] <= limits.mass_max
    rep["ok_case"] = rep["P_max_sim"] <= limits.p_case_allow
    rep["ok_connected"] = rep["components"] == 1
    adm = admissibility(rep, limits, tol)
    rep["admissibility"] = adm
    rep["in_set"] = adm["in_set"]
    rep["converged"] = adm["spec_ok"]          # 하위 호환 — 사양만 본 값
    return rep, (t, p, f, diag, omega)


# ═══════════════════════════════════════════════════════════════ 해 족(族)
def shape_distance(s1, s2, n_theta: int = 721):
    """두 포트 윤곽의 최대 차이 [mm] — **최적 회전 정렬 후**.

    회전은 정확한 대칭이므로 정렬하지 않고 비교하면 "같은 설계" 를 "다른
    설계" 로 센다. 게이지를 고정해도 지배 모드가 바뀌면 방향이 튈 수 있어,
    비교 단계에서 한 번 더 정렬한다.
    """
    def radius(r0, a, b, th):
        k = np.arange(1, len(a) + 1)[:, None]
        return float(r0) + (np.asarray(a)[:, None] * np.cos(k * th)
                            + np.asarray(b)[:, None] * np.sin(k * th)).sum(0)

    th = np.linspace(0.0, 2 * np.pi, n_theta)
    r1 = radius(s1[0], s1[1], s1[2], th)
    k = np.arange(1, len(s2[1]) + 1)
    best = np.inf
    for d in np.linspace(0.0, 2 * np.pi, n_theta):
        a2 = np.asarray(s2[1]) * np.cos(k * d) - np.asarray(s2[2]) * np.sin(k * d)
        b2 = np.asarray(s2[1]) * np.sin(k * d) + np.asarray(s2[2]) * np.cos(k * d)
        best = min(best, float(np.abs(r1 - radius(s2[0], a2, b2, th)).max()))
    return best


def solution_family(motor: DiffMotor, target: Target, n_starts: int = 6,
                    n_modes: int = 4, lengths=(95.0,), r0_range=(9.0, 14.0),
                    limits: ManufacturingLimits = None,
                    weights: LossWeights = None, iters: int = 90,
                    tol: Tolerances = None, verbose: bool = True,
                    *, throat: ThroatSpec):
    """여러 출발점에서 돌려 **허용 오차를 만족하는 해를 모두** 모은다.

    측정에서 드러난 사실 — 같은 목표에 대해 출발점 8 개가 전부 잔차
    0.24 % 이내로 수렴했지만, 회전을 제거하고도 형상이 최대 5.97 mm(포트
    반경의 55 %) 떨어져 있었다. **해는 하나가 아니다.**

    그래서 하나를 골라 내놓는 대신 족(族)으로 보고하고, 2차 지표를 함께 준다.

    | 지표 | 뜻 |
    |---|---|
    | 복잡도 Σ&#124;c_k&#124; | 모드 진폭 합. 작을수록 단순하고 제작이 쉽다 |
    | 슬리버 비율 | 소진 시점에 남는 추진제 비율. 작을수록 좋다 |
    | 제약 여유 | 웹·꼭짓점이 한계에서 얼마나 떨어져 있는가 |

    설계자가 이 표를 보고 고르는 것이 "도구가 임의로 하나를 고르는 것" 보다
    정직하고 쓸모 있다.
    """
    limits = limits or ManufacturingLimits()
    weights = weights or LossWeights()
    tol = tol or Tolerances()
    rng = np.random.default_rng(0)
    fam = []
    for s in range(n_starts):
        dv = seed_design(r0=float(rng.uniform(*r0_range)), n_modes=n_modes,
                         lengths=lengths, seed_amplitude=0.4, seed=s,
                         throat=throat)
        dv, _ = optimize(motor, dv, target, limits, weights, iters=iters,
                         verbose=False, tol=tol, throat=throat)
        rep, _ = final_report(motor, dv, target, limits, weights, tol=tol)
        rep["seed"] = s
        rep["r0"] = float(dv.r0.detach())
        rep["a"] = dv.a.detach().numpy().copy()
        rep["b"] = dv.b.detach().numpy().copy()
        rep["a_throat"] = float(dv.a_throat.detach())
        rep["within_tol"] = rep["admissibility"]["spec_ok"]
        rep["feasible"] = rep["admissibility"]["man_ok"]
        fam.append(rep)
        if verbose:
            print(f"  seed {s}: 모양 {rep['rms_shape_pct']:5.2f}%  I {rep['I_err_pct']:5.2f}%"
                  f"  복잡도 {rep['complexity']:5.2f}  슬리버 {100*rep['sliver_frac']:5.2f}%"
                  f"  {'합격' if rep['within_tol'] and rep['feasible'] else '탈락'}")

    keep = [r for r in fam if r["within_tol"] and r["feasible"]]
    if len(keep) > 1:
        d = [shape_distance((p["r0"], p["a"], p["b"]), (q["r0"], q["a"], q["b"]))
             for i, p in enumerate(keep) for q in keep[i + 1:]]
        spread = {"min": float(np.min(d)), "median": float(np.median(d)),
                  "max": float(np.max(d))}
    else:
        spread = None
    keep.sort(key=lambda r: (r["complexity"], r["sliver_frac"]))
    picks = shortlist(keep, n=5)
    return {"family": fam, "accepted": keep, "spread_mm": spread,
            "shortlist": picks, "table": family_table(keep, tol=tol),
            # ⚠ `recommended` 는 **편의값일 뿐 '정답' 이 아니다.** 설계자가
            #    `shortlist` 를 보고 고르는 것이 기본 흐름이다.
            "recommended": picks[0] if picks else None}



# ═══════════════════════ 허용 집합에서 무엇을 보여줄 것인가
def pareto_front(rows, keys=("rms_shape_pct", "complexity")):
    """작을수록 좋은 키들에 대한 **비지배(non-dominated) 집합**.

    어떤 설계 q 가 p 보다 **모든 키에서 같거나 좋고 적어도 하나에서 더 좋으면**
    p 는 지배당한 것이다. 지배당한 설계는 보여줄 이유가 없다 — 모든 기준에서
    더 나은 대안이 있기 때문이다.
    """
    out = []
    for i, p_ in enumerate(rows):
        dominated = False
        for j, q in enumerate(rows):
            if i == j:
                continue
            le = all(q[k] <= p_[k] + 1e-12 for k in keys)
            lt = any(q[k] < p_[k] - 1e-12 for k in keys)
            if le and lt:
                dominated = True
                break
        if not dominated:
            out.append(p_)
    return out


def shortlist(accepted, n: int = 5, keys=("rms_shape_pct", "complexity"),
              rank_by: str = "rms_shape_pct"):
    """허용 집합에서 **보여줄 3~5개**를 고른다.

    **왜 '손실이 가는 방향에 가까운 설계' 가 아니라 파레토 전선인가.**

    직관은 이렇다 — 사양 오차를 끝까지 밀면 손실의 최소점(허용 집합 밖)으로
    가는 직선이 생기고, 그 직선에 가까우면서 집합 안인 설계를 보여주면 좋겠다는
    것. 의도는 맞다. 그런데 그 '직선' 을 그대로 쓰면 안 되는 이유가 둘 있다.

    1. **그 직선은 설계가 아니라 최적화 궤적이다.** Adam 의 경로는 출발점·학습률·
       모멘텀에 따라 달라진다. 실측(실현 불가 목표, 씨앗 4개)에서 반환 설계가
       r0 3.3 mm, 노즐목 3.3 배로 흩어졌다. 궤적을 기준으로 삼으면 방금 없앤
       그 불안정을 다시 들여오는 셈이다.
    2. **설계 공간의 거리는 단위가 섞여 있다.** r0 는 mm, A_t 는 mm², 계수는 mm.
       무차원화를 어떻게 하느냐에 따라 '가까운' 설계가 바뀐다.

    대신 **사양 공간에서** 재면 둘 다 사라진다. 그 직선 위를 간다는 것은 결국
    "모양 오차가 작아진다" 는 뜻이므로, 집합 안에서 모양 오차가 작은 쪽을 고르면
    같은 의도가 달성되고 궤적도 단위도 끼어들지 않는다.

    여기에 하나를 더 건다 — **복잡도**. 모양 오차만으로 줄 세우면 "조금 더
    맞추려고 훨씬 복잡해진" 설계가 위로 올라온다. 두 축의 파레토 전선을 쓰면
    "모양을 더 맞추려면 반드시 더 복잡해지는" 교환 관계 위의 설계만 남고,
    양쪽 모두에서 지는 설계는 빠진다. 지배당한 설계를 보여주는 것은 사용자의
    시간을 쓰는 일이다 — 모든 기준에서 더 나은 대안이 이미 목록에 있다.

    Parameters
    ----------
    accepted : list   `solution_family(...)["accepted"]` — 허용 집합에 든 설계들
    n : int           최대 몇 개를 보여줄지 (3~5 권장)
    keys : tuple      파레토 축 (작을수록 좋은 값)
    rank_by : str     전선 위에서의 표시 순서
    """
    if not accepted:
        return []
    front = pareto_front(accepted, keys=keys)
    front.sort(key=lambda r: r[rank_by])
    if len(front) >= n:
        return front[:n]
    # 전선이 n 개보다 적으면 나머지는 rank_by 순으로 채운다 (지배당했음을 표시)
    rest = [r for r in accepted if r not in front]
    rest.sort(key=lambda r: r[rank_by])
    for r in front:
        r["on_front"] = True
    for r in rest:
        r["on_front"] = False
    return front + rest[: n - len(front)]


def margins(rep: dict, tol: Tolerances = None) -> dict:
    """각 사양이 허용 경계에서 **얼마나 안쪽에 있는가** [%p]. 클수록 여유가 많다.

    ⚠ **압력 곡선이 입력으로 들어오면 사양이 5 개다**. 초판은 압력
       곡선이 들어와도 3 개만 돌려줘서, 산출물에 압력 여유가 아예 나타나지
       않았다 — 합격 기준이 "추력 **및 압력** 10 % 이내" 인데 그 절반의 여유를
       보고하지 않은 것이다.

       `admissibility` 와 **같은 규약**으로 가른다: `curve_loss` 가 `gap_p_*` 를
       넣어 줬을 때만 압력 행을 집어간다. 그래야 압력 곡선이 없는 호출에서
       "압력 여유" 라는 빈 숫자를 만들지 않는다.
    """
    tol = tol or Tolerances()
    out = {"모양": 100 * tol.shape - rep["rms_shape_pct"],
           "총역적": 100 * tol.impulse - rep["I_err_pct"],
           "최대추력": 100 * tol.f_max - rep["F_err_pct"]}
    if "gap_p_shape_pp" in rep:
        out["압력 모양"] = 100 * tol.p_shape - rep["P_shape_pct"]
    if "gap_p_max_pp" in rep:
        out["최대압"] = 100 * tol.p_max - rep["P_err_pct"]
    return out


def family_table(accepted, n: int = 5, tol: Tolerances = None) -> str:
    """설계자가 보고 고르는 표.

    ⚠ **압력 곡선이 입력이면 압력 두 열을 같이 낸다**. 초판은 압력
       곡선이 들어와도 모양·I·F 세 열만 찍고, 허용 오차 머리말도
       `tol.as_pct()` 를 `with_pressure` 없이 불러서 압력 문턱을 숨겼다.
       합격 기준이 "추력 **및 압력** 10 % 이내" 이므로 설계자가 고를 때 압력
       오차를 못 보면 고를 수가 없다.
    """
    tol = tol or Tolerances()
    picks = shortlist(accepted, n=n)
    if not picks:
        return "허용 집합이 비어 있다 — 보여줄 설계가 없다."
    # 압력이 사양인지는 보고서가 그 열을 가지고 있는지로 가른다 —
    # `admissibility`·`margins` 와 같은 규약이다.
    with_p = all("P_shape_pct" in r for r in picks)
    ph = f"{'P모양%':>7} {'Pmax%':>7} " if with_p else ""
    head = (f"{'#':>2} {'모양%':>7} {'I%':>7} {'F%':>7} {ph}{'복잡도':>7} "
            f"{'슬리버%':>8} {'웹mm':>7} {'장약g':>7} {'r0':>7} {'A_t':>7}  전선")
    lines = [f"허용 집합에서 고른 {len(picks)} 개 "
             f"(파레토 축: 모양 오차 · 복잡도, 허용 오차 "
             f"{', '.join(f'{k} {v:.0f}%' for k, v in tol.as_pct(with_p).items())})",
             head, "-" * len(head)]
    for i, r in enumerate(picks, 1):
        pv = (f"{r['P_shape_pct']:>7.2f} {r['P_err_pct']:>7.2f} "
              if with_p else "")
        lines.append(
            f"{i:>2} {r['rms_shape_pct']:>7.2f} {r['I_err_pct']:>7.2f} "
            f"{r['F_err_pct']:>7.2f} {pv}{r['complexity']:>7.2f} "
            f"{100*r['sliver_frac']:>8.2f} {r['web_mm']:>7.2f} {r['mass_g']:>7.0f} "
            f"{r['r0']:>7.2f} {r['a_throat']:>7.1f}  "
            f"{'●' if r.get('on_front', True) else '○'}")
    lines.append("● 파레토 전선 (모양을 더 맞추려면 반드시 더 복잡해진다) / "
                 "○ 지배당함 — 전선이 모자라 채운 것")
    if not with_p:
        lines.append("⚠ 압력은 사양이 아니다 (압력 곡선이 입력으로 들어오지 "
                     "않았다). 이 표의 설계들은 압력이 100 % 이상 틀릴 수 있다 — "
                     "같은 허용 집합 안에서 최대압이 2.69 배 흩어진다는 측정이 "
                     "있다.")
    return "\n".join(lines)

# ═════════════════════════════════════════════════ 최소 복잡도 (K 상승 탐색)
def _modes(dv: DesignVars) -> int:
    return int(dv.a.shape[0])


def k0_recheck(motor: DiffMotor, target: Target,
               limits: ManufacturingLimits = None, weights: LossWeights = None,
               tol: Tolerances = None, lengths=(95.0,),
               r0_init: float = 12.0, at_init: float = None,
               rounds: int = 2, iters_per_k: int = 120, verbose: bool = False,
               rounds_done: int = 0, iters_done: int = 0, *,
               throat: ThroatSpec):
    """**"K 가 올라가서" 와 "예산이 늘어서" 를 가르는 재확인**.

    `minimal_complexity_search` 는 K 마다 `iters_per_k` 를 **새로** 준다. 그래서
    `K = k` 의 해는 앞선 K 들의 예산을 전부 물려받는다. `K = k` 가 통과했을 때
    그것이 *모드가 한 일* 인지 *누적 예산이 한 일* 인지 표만 보고는 알 수 없다.

    **공정한 비교 = 같은 예산 + 같은 재시작 횟수.** K 를 올리는 것은 Adam 상태를
    새로 만드는 **재시작**이기도 하다 (모멘트 초기화 + 새 모드 섭동). 그래서 K=0
    에도 같은 `rounds` 번의 재시작과 같은 `iters_per_k` 를 주고, 해를 이어가며
    돌린다. 차이는 **모드를 0 에 얼려둔 것 하나**뿐이다.

    `rounds_done`/`iters_done` 과 `r0_init`/`at_init` 로 **이미 돌린 K=0 라운드를
    물려받는다.** 본 탐색의 K=0 실행이 곧 재확인의 0 번째 라운드이므로, 그것을
    다시 돌리는 것은 낭비다 (결정적이라 결과가 글자 하나까지 같다 — 실측에서
    모양 6.16 %, 초과 9.5 %p 가 그대로 재현되었다). 호출자는 K=0 의 끝점과
    쓴 반복 수를 넘겨 주고 남은 라운드만 돌리게 한다.

    실측 (6-로브 깊음, `iters_per_k = 120`) — **판정이 뒤집혔다**

        K       모양%     I%     F%   복잡도   쓴/예산   누적   종료
        0       6.16  19.52   9.69   0.000   120/120    120   반복 수 소진
        1       9.32   8.93   5.43   0.008    21/120    141   집합 진입   ← 통과
        0(재확인) 9.61   6.20   5.91   0.000   143/240    143   집합 진입   ← 통과

        K=1 은 K=0 의 끝점에서 **21 반복**만 더 쓰고 집합에 들어갔다. 그런데 모드
        진폭 합이 0.008 mm — 격자 한 칸 0.5 mm 의 1/60, `r₀` 의 0.07 % 다. 새 모드의
        초기 섭동 `|c₁| = 0.546 mm` 가 **68 배 줄었다**. 즉 그 21 반복이 움직인 것은
        `r₀` 와 `A_t` 뿐이고, 둘은 K=0 에서도 자유 변수였다. 같은 자유도에 **2 반복**
        (143 vs 141) 을 더 주니 원형도 집합에 들어간다. → `k_chosen` 을 0 으로 되돌린다.

    반환: `(설계, 기록, 정보)`. `정보["passed"]` 가 True 면 K>0 이라는 판정이
    **거짓 양성**이었다는 뜻 — 원형으로 충분했다.
    """
    limits = limits or ManufacturingLimits()
    weights = weights or LossWeights()
    tol = tol or Tolerances()
    mk = lambda v: torch.tensor(v, dtype=torch.float64, requires_grad=True)
    dv = DesignVars(r0=mk(float(r0_init)), a=mk(np.zeros(1)), b=mk(np.zeros(1)),
                    a_throat=mk(float(at_init if at_init is not None
                                      else throat.start_mm2())),
                    lengths=list(lengths))
    dv.a.requires_grad_(False)                 # 원형 — 모드를 얼린다
    dv.b.requires_grad_(False)
    rounds = max(1, int(rounds))
    done = max(0, min(int(rounds_done), rounds - 1))   # 최소 한 라운드는 돌린다
    used, segs, rep, adm = max(0, int(iters_done)), [], None, None
    for rnd in range(done, rounds):
        dv, hist = optimize(motor, dv, target, limits, weights,
                            iters=iters_per_k, verbose=False, tol=tol,
                            throat=throat)
        u = int(hist[-1].get("stop_iter", iters_per_k - 1)) + 1
        used += u
        rep, _ = final_report(motor, dv, target, limits, weights, tol=tol)
        adm = rep["admissibility"]
        segs.append({"round": rnd, "iters_used": u, "iters_cum": used,
                     "stop_reason": hist[-1].get("stop_reason", "반복 수 소진"),
                     "rms_shape_pct": rep["rms_shape_pct"],
                     "worst_gap_pp": adm["worst_gap_pp"], "passed": adm["in_set"]})
        if verbose:
            print(f"    재확인 K=0 라운드 {rnd}: 누적 {used} 반복  "
                  f"모양 {rep['rms_shape_pct']:6.2f}%  "
                  f"최악 초과 {adm['worst_gap_pp']:5.1f} %p  "
                  f"{'통과' if adm['in_set'] else '부족'}")
        if adm["in_set"]:
            break
    rep["r0_for_scale"] = float(dv.r0.detach())
    info = {"passed": bool(adm["in_set"]), "rounds_run": len(segs),
            "rounds_budget": rounds, "rounds_inherited": done,
            "iters_used": used, "iters_inherited": max(0, int(iters_done)),
            "iters_budget": rounds * iters_per_k,
            "worst_gap_pp": adm["worst_gap_pp"], "binding": adm["binding"],
            "segments": segs}
    return dv, rep, info


def minimal_complexity_search(motor: DiffMotor, target: Target,
                              limits: ManufacturingLimits = None,
                              weights: LossWeights = None,
                              k_max: int = 4, iters_per_k: int = 120,
                              tol: Tolerances = None, lengths=(95.0,),
                              r0_init: float = 12.0, verbose: bool = True,
                              n_seeds: int = 3, *, throat: ThroatSpec,
                              seeding: str):
    """허용 오차를 만족하는 **가장 작은 모드 수 K** 를 찾는다.

    설계 원칙: **간단한 형상으로 목표 곡선을 만들 수 있으면 복잡하게 갈 이유가
    없다.** K = 0(원형)부터 올려가며 처음 통과하는 K 에서 멈춘다.

        K = 0 → r(θ) = r0                     (원형, 변수 1개)
        K = 1 → r0 + a₁cosθ + b₁sinθ
        K = k → 모드 1..k 까지 허용

    **K 와 그룹 L1 은 하는 일이 다르다.**
      - K  : 모드 수의 *상한* (이산, 열거)
      - L1 : 그 상한 안에서 *실제로 몇 개를 쓰는가* (연속, 경사)

    "모드 3 만 허용" 을 표현하는 K 값이 없고 부분집합을 다 열거하면 2^K 가지다.
    L1 이 그 선택을 연속적으로 대신한다. 측정에서 K=3 에 L1 을 약하게 걸면
    |c| 가 [0.9, 1.3, 2.0] 에서 [0.05, 0.1, 2.2] 로 정리되었다.

    **웜 스타트**: K 를 하나 올릴 때 직전 해를 초기값으로 쓴다. 새 모드만 작은
    난수(N(0, 0.25))로 흔들어 시작하므로 수렴이 빠르다.

    ⚠ **초판 docstring 정정 (2026-10-07).** 초판은 "원형 안장점을 K=0 단계에서
       이미 지나왔으므로 대칭이 자동으로 깨진다" 고 적었다. **거꾸로다.** K=0 해는
       `a = b = 0`, 즉 **안장점 위**다 (포트 면적 `πr₀² + (π/2)Σ(a²+b²)` 가 계수에
       대해 짝함수라 모든 모드 경사가 정확히 0, 수치상 ~1e-13). 지나온 것이 아니라
       거기 앉아서 출발한다. 새 모드에 주는 0.25 섭동이 유일한 탈출 수단이고,
       `seed_design` 이 전 모드에 주는 **0.3** (기본값) ~ 0.4 mm (`solution_family`)
       보다 약하다. (`saddle_bias.py`)

    ⚠ **`iters_per_k` 기본값은 120 이다.** 45 는 설계 문서
       §4.4 가 직접 "오판을 만든다" 고 측정한 값이었다 — 6-로브 K=0 이 45 반복
       에서 모양 19.42 %, 300 반복에서 3.01 % (**16.42 %p 미수렴**). §4.4 의
       모든 `k_table` 측정은 120 으로 했는데 코드 기본값만 45 로 남아 있어서,
       기본값으로 호출하면 설계 문서가 지목한 그 오판이 그대로 재현됐다.
       비용은 2.7 배다.

    ⚠ **예산 회계.** K 마다 `iters_per_k` 를 **새로** 주므로 `K = k` 의
       해는 앞선 K 들의 예산을 전부 물려받는다 (`k_table` 의 `iters_cum`). 그래서
       "K 를 올려서 통과" 와 "예산이 늘어서 통과" 가 섞인다. `K > 0` 이 통과하면
       `k0_recheck` 로 **K=0 에 같은 예산·같은 재시작 횟수**를 주고 다시 확인하고,
       K=0 이 거기서 통과하면 `k_chosen` 을 0 으로 되돌린다. 또한 힌지 손실 +
       `stop_in_set` 아래에서는 **집합에 못 들어간 K 는 예산을 다 쓴다** — 그러니
       `stop_reason == "반복 수 소진"` 은 **그 K 가 부족하다는 증거가 아니다.**

    ## 씨앗 전략 — `seeding` 은 **필수 인자**다

        "warm"         K=0 → 1 → … 로 앞 K 의 해를 물려받고, 새로 열린 모드만
                       `N(0, 0.25)` 로 흔든다. 앞 K 가 잡은 `r₀` 에 묶여 올바른
                       영역에 머무는 것이 장점이다.
        "independent"  K 마다 앞 K 를 **전혀** 물려받지 않고 독립 씨앗
                       `n_seeds` 개로 출발한다. 예산 `iters_per_k` 를 씨앗 수로
                       나눠 쓰므로 **총예산은 웜과 같다** (공정 비교의 조건).

    기본값을 두지 않는다. 두 전략이 **판정을 바꾸기** 때문이다 — (A) 측정
    에서 같은 총예산에 웜이 집합 기준으로 이기고 복잡도가 9~32 배 작았다
    (6-로브: 집합 안·0.54 mm vs 9.92 %p 밖·4.99 mm). 반면 웜은 K>0 을 K=0 의
    해답(= `a=b=0` 안장점) 위에 놓으므로, 새 모드 킥 0.25 mm 가 진폭 바닥
    0.60 mm 보다 작아 L1 희소성이 모드를 0 으로 당긴다 (왕복 검증에서
    K=1 진폭이 0.25 → 0.05 mm 로 **줄었다**). 어느 쪽이 맞는지는 설정에 따라
    달라지므로 호출자가 **명시**해야 한다.

    ⚠ (A) 측정은 노즐목 **자유** 모드·3 사양·옛 정체 규칙에서 한 것이다. 노즐목이
       입력이 된 뒤에는 웜 스타트가 물려받을 `A_t` 가 없으므로 그 결론이 그대로
       옮겨가지 않는다 — 지금 설정에서 다시 재야 한다
       (`experiments/seeding_compare.py`).

    최선 후보를 고르는 규칙은 두 전략에서 같다: **집합까지의 거리** → 동점이면
    **복잡도**. 모양 오차로 고르면 '총역적을 가장 많이 포기한 설계' 가 뽑힌다.

    반환: (설계, 기록). 기록의 `k_table` 이 K 별 최선 잔차·예산이라, 산출물에
    "원형으로는 모양 오차 18 % 라 부족했다 (120 반복 소진)" 같은 문장을 쓸 수 있다.
    `recheck: True` 행은 재확인용 K=0 재실행이다.
    """
    limits = limits or ManufacturingLimits()
    weights = weights or LossWeights()
    tol = tol or Tolerances()
    # ── 돌리기 전에 물리 한계부터 본다 (§4.6.3). 넘으면 어떤 형상으로도 안 되므로
    #    최적화를 시작하지 않는다 — 경로에 의존하지 않는 확정 판정이다.
    bounds = feasibility_bounds(motor, target, limits, lengths=lengths,
                                throat=throat)
    if not bounds["feasible"]:
        if verbose:
            print("  " + bounds["note"])
        return None, {"k_chosen": None, "k_table": [], "feasible": False,
                      "bounds": bounds, "note": bounds["note"],
                      "stopped_before_optimizing": True}
    if seeding not in ("independent", "warm"):
        raise ValueError(f"seeding 은 'independent' 또는 'warm' 이다: {seeding!r}")
    n_seeds = max(1, int(n_seeds)) if seeding == "independent" else 1
    k_table, best_dv, best_rep, chosen = [], None, None, None
    prev, iters_cum, k0_end = None, 0, (r0_init, throat.start_mm2(), 0)
    for k in range(0, k_max + 1):
        n_modes = max(k, 1)
        g = np.random.default_rng(100 + k)
        mk = lambda v: torch.tensor(v, dtype=torch.float64, requires_grad=True)
        # ── 그 K 의 출발점들 (씨앗 전략에 따라 1 개 또는 n_seeds 개)
        if seeding == "warm" and prev is not None:
            a0, b0 = np.zeros(n_modes), np.zeros(n_modes)
            a0[:len(prev[1])] = prev[1]
            b0[:len(prev[1])] = prev[2]
            r0v = prev[0]
            if k > 0:                        # 새로 열린 모드만 작게 흔든다
                a0[k - 1] += g.normal(0, 0.25)
                b0[k - 1] += g.normal(0, 0.25)
            starts = [(r0v, a0, b0)]
        elif k == 0 or seeding == "warm":
            starts = [(r0_init, np.zeros(n_modes), np.zeros(n_modes))]
        else:
            # 독립 다중출발 — 앞 K 를 **전혀** 물려받지 않는다. 예산을 씨앗
            # 수로 나눠 쓰므로 총예산은 웜 스타트와 같다 (공정 비교의 조건).
            starts = []
            for sd in range(n_seeds):
                gs = np.random.default_rng(10_000 + 100 * k + sd)
                starts.append((float(gs.uniform(0.75, 1.25)) * r0_init,
                               gs.normal(0.0, 0.3, n_modes),
                               gs.normal(0.0, 0.3, n_modes)))
        per = max(1, iters_per_k // len(starts))

        cand, spent = [], 0
        for a0, b0, r0v in ((s_[1], s_[2], s_[0]) for s_ in starts):
            dvs = DesignVars(r0=mk(float(r0v)), a=mk(np.asarray(a0, float)),
                             b=mk(np.asarray(b0, float)),
                             a_throat=mk(throat.start_mm2()),
                             lengths=list(lengths))
            if k == 0:                       # 원형 — 모드를 얼리고 r0 만 푼다
                dvs.a.requires_grad_(False)
                dvs.b.requires_grad_(False)
            dvs, h = optimize(motor, dvs, target, limits, weights,
                              iters=per, verbose=False, tol=tol, throat=throat)
            _, rp, _ = evaluate(motor, dvs, target, limits, weights, tol=tol)
            ad = admissibility(rp, limits, tol)
            cand.append((dvs, h, ad, float(rp["complexity"])))
            # ⚠ **모든 팔이 쓴 반복을 센다.** 고른 팔만 세면 독립 다중출발이
            #    예산을 공짜로 쓴 것처럼 보이고, 웜 스타트와의 비교가 깨진다
            #    (측정 (A) 가 바로 이 공정성 조건으로 설계됐다).
            spent += int(h[-1].get("stop_iter", per - 1)) + 1
        # ⚠ 최선은 **집합까지의 거리**로 고른다. 모양 오차로 고르면 '총역적을
        #    가장 많이 포기한 설계' 가 뽑힌다 (측정 (A) 에서 독립 팔이
        #    정확히 그 함정에 빠졌다 — 모양 7.67 % 로 최선인데 집합 9.92 %p 밖).
        #    동점이면 **복잡도가 작은** 쪽 — 이 탐색의 목적이 최소 복잡도다.
        dv, hist, _, _ = min(cand, key=lambda c: (not c[2]["in_set"],
                                                  c[2]["worst_gap_pp"], c[3]))
        # ── 예산 회계. `iters_cum` 이 없으면 "그 K 로는 안
        #    된다" 와 "그 K 에 예산이 모자랐다" 를 표에서 구분할 수 없다. 특히
        #    `stop_reason == "반복 수 소진"` 은 **그 K 가 부족하다는 증거가 아니다.**
        iters_used = spent                 # 그 K 의 **모든 팔** 합
        iters_cum += iters_used
        stop_reason = hist[-1].get("stop_reason", "반복 수 소진")
        # ⚠ `loss_still_falling` 을 **표까지** 들고 온다. `optimize` 가 기록만
        #    해도 K 표만 보는 쪽에서는 조기 종료인지 알 수 없고, 그게
        #    n=321 의 "어느 K 도 못 들어갔다" 를 물리 한계로 오독한 경로였다.
        #    True 면 그 K 의 "부족 %p" 는 **예산 부족의 값**이다.
        falling = bool(hist[-1].get("loss_still_falling", False))
        stalled_metric = hist[-1].get("stalled_metric")
        rep, _ = final_report(motor, dv, target, limits, weights, tol=tol)
        adm = rep["admissibility"]
        passed = adm["in_set"]
        k_table.append({"K": k, "rms_shape_pct": rep["rms_shape_pct"],
                        "I_err_pct": rep["I_err_pct"], "F_err_pct": rep["F_err_pct"],
                        "complexity": rep["complexity"], "passed": passed,
                        "worst_gap_pp": adm["worst_gap_pp"], "binding": adm["binding"],
                        "man_ok": adm["man_ok"], "iters_budget": iters_per_k,
                        "iters_used": iters_used, "iters_cum": iters_cum,
                        "stop_reason": stop_reason,
                        "loss_still_falling": falling,
                        "stalled_metric": stalled_metric})
        if verbose:
            tail = ("통과" if passed
                    else f"부족 — {adm['binding'] or '제작 제약'} {adm['worst_gap_pp']:.1f} %p")
            # 손실이 아직 내려가는 중에 끝난 행은 눈에 보이게 적는다 — 이 표를
            # 읽고 "그 K 로는 안 된다" 를 결론 내리는 사람이 있다.
            #
            # ⚠ 두 경우를 **구분**한다. 예산을 다 쓰고 끝난 행에 "조기 종료" 를
            #    붙이면 거짓이다 (일찍 끊긴 게 아니라 예산이 모자란 것이다).
            #    초판이 그렇게 찍어서 120/120 행에 "조기 종료" 가 달렸다.
            early = ("" if passed or not falling else
                     "  ⚠ 예산 부족 (손실 하강 중)"
                     if stop_reason == "반복 수 소진" else
                     "  ⚠ 조기 종료 (손실 하강 중)")
            print(f"  K={k}: 모양 {rep['rms_shape_pct']:6.2f}%  I {rep['I_err_pct']:6.2f}%"
                  f"  F {rep['F_err_pct']:6.2f}%  복잡도 {rep['complexity']:5.2f}"
                  f"  [{iters_used}/{iters_per_k} 반복, 누적 {iters_cum}, {stop_reason}]"
                  f"  {tail}{early}")
        # ⚠ '가장 가까운 설계' 는 **모양 오차가 아니라 집합까지의 거리**로 고른다.
        #    모양 오차로 고르면 초판처럼 '총역적을 가장 많이 포기한 설계' 가 뽑힌다.
        rep["r0_for_scale"] = float(dv.r0.detach())
        if best_rep is None or adm["worst_gap_pp"] < best_rep["admissibility"]["worst_gap_pp"]:
            best_dv, best_rep = dv, rep
        if passed:
            chosen, best_dv, best_rep = k, dv, rep
            break
        prev = (float(dv.r0.detach()), dv.a.detach().numpy().copy(),
                dv.b.detach().numpy().copy(), float(dv.a_throat.detach()))
        if k == 0:                           # 재확인의 0 번째 라운드로 물려준다
            k0_end = (prev[0], prev[3], iters_used)

    recheck = None
    if chosen is not None and chosen > 0:
        # ⚠ K 가 올라가서 통과한 것인지, 반복 예산이 늘어서 통과한 것인지 **잰다**
        #    경고만으로는 부족하다 — K=0 에 K 상승 탐색이
        #    실제로 쓴 예산(과 같은 재시작 횟수)을 그대로 주고 다시 돌린다.
        #    본 루프의 K=0 실행이 재확인의 0 번째 라운드이므로 그것을 **물려받는다**
        #    (결정적이라 다시 돌리면 결과가 같다 — 120 반복을 낭비할 이유가 없다).
        if verbose:
            print(f"  ── K=0 재확인: 같은 예산 {iters_cum} 반복 "
                  f"({chosen+1} 라운드 × {iters_per_k}, "
                  f"0 번째 라운드 {k0_end[2]} 반복은 물려받음)")
        dv0, rep0, recheck = k0_recheck(
            motor, target, limits, weights, tol=tol, lengths=lengths,
            r0_init=k0_end[0], at_init=k0_end[1], rounds=chosen + 1,
            throat=throat,
            rounds_done=1, iters_done=k0_end[2],
            iters_per_k=iters_per_k, verbose=verbose)
        recheck["k_was"] = chosen
        k_table.append({"K": 0, "recheck": True,
                        "rms_shape_pct": rep0["rms_shape_pct"],
                        "I_err_pct": rep0["I_err_pct"], "F_err_pct": rep0["F_err_pct"],
                        "complexity": rep0["complexity"], "passed": recheck["passed"],
                        "worst_gap_pp": recheck["worst_gap_pp"],
                        "binding": recheck["binding"],
                        "man_ok": rep0["admissibility"]["man_ok"],
                        "iters_budget": recheck["iters_budget"],
                        "iters_used": recheck["iters_used"],
                        "iters_cum": recheck["iters_used"],
                        "stop_reason": recheck["segments"][-1]["stop_reason"]})
        if recheck["passed"]:                  # K>0 판정은 거짓 양성이었다
            chosen, best_dv, best_rep = 0, dv0, rep0

    if chosen is not None:
        note = f"모드 {chosen} 개로 달성"
        if chosen == 0 and recheck is not None and recheck["passed"]:
            note += (f" — **원형 포트로 충분하다.** K 상승 탐색은 K={recheck['k_was']} "
                     f"를 골랐지만, K=0 에 같은 예산({recheck['iters_used']} 반복, "
                     f"{recheck['rounds_budget']} 라운드)을 주면 원형도 집합에 들어간다. "
                     f"즉 모드가 아니라 **예산**이 한 일이었다 — K={recheck['k_was']} "
                     f"판정은 거짓 양성이다")
        elif chosen == 0:
            note += " — **원형 포트로 충분하다.** 이 곡선은 포트 모양을 요구하지 않는다"
        elif k_table:
            prev_row = next(r for r in k_table
                            if r["K"] == chosen - 1 and not r.get("recheck"))
            note += (f" (K={chosen-1} 에서는 "
                     f"{prev_row['binding'] or '제작 제약'} 이 "
                     f"{prev_row['worst_gap_pp']:.1f} %p 모자랐다)")
            if recheck is not None:            # 재확인을 통과하지 못했다 — 판정 확정
                note += (f"  ✔ 예산 재확인 통과: K=0 에 같은 예산"
                         f"({recheck['iters_used']} 반복, "
                         f"{recheck['rounds_budget']} 라운드)을 줘도 "
                         f"{recheck['binding'] or '제작 제약'} 이 "
                         f"{recheck['worst_gap_pp']:.1f} %p 모자랐다 — "
                         f"모드가 실제로 일했다.")
            # ⚠ 진폭이 사실상 0 이면 재확인을 통과했어도 의심스럽다. 바닥이 **둘**
            #    있고, 둘 중 높은 쪽이 판정한다 (`amplitude_floor`).
            floor, which = amplitude_floor(best_rep.get("r0_for_scale", 12.0),
                                           h=float(motor.ds.h),
                                           eps_cells=float(motor.eps_cells))
            if best_rep["complexity"] < floor:
                note += (f"  ⚠ 다만 **모드 진폭이 바닥 아래다** (진폭 합 "
                         f"{best_rep['complexity']:.3f} mm < {which} 바닥 "
                         f"{floor:.3f} mm, 격자 한 칸 {float(motor.ds.h):.2f} mm). "
                         f"이 차이는 {which}이 구분하지 못하므로 K={chosen} 을 "
                         f"제작 복잡도로 올려 적을 근거가 약하다.")
            best_rep["amplitude_floor_mm"] = floor
            best_rep["amplitude_floor_binding"] = which
    else:
        # ⚠ 억지로 끼워맞추지 않는다
        note = verdict_text(best_rep["admissibility"], k_max=k_max)
    best_rep["k_chosen"] = chosen
    best_rep["k_table"] = k_table
    best_rep["bounds"] = bounds
    best_rep["feasible"] = chosen is not None
    best_rep["note"] = note
    # 실현 불가면 반환된 설계는 **답이 아니라 진단용**이다. 호출자가 구분할 수 있게
    # 명시적인 깃발을 둔다 — `best_rep["feasible"]` 이 False 면 설계를 쓰면 안 된다.
    return (best_dv if chosen is not None else None), best_rep
