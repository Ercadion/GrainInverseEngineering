"""미분가능 burnback — PyTorch 구현.

numpy 판(`burnback2d.py`)과 **같은 수식**을 PyTorch 텐서로 다시 쓴 것이다.
차이는 하나뿐이다 (⚠ 정정: **둘이다** — `burnback2d.perimeter` 는 기본 `richardson=True` 로 d·2d 외삽을 더 한다. 실측: 두 판 둘레가 최대 1.3e-3 상대 어긋나고 w=0 에서 torch 75.398 = 2πr 해석값, numpy 75.353). 여기서는 포트 윤곽 계수에 대한 **경사가 흐른다.**

    포트 계수 (r0, a_k, b_k)
        → 윤곽 샘플점            (미분 가능)
        → 거리장 φ = soft-min 거리 (동점에서도 경사가 매끄럽다)
        → A_end(w) = Σ σ((φ−w)/ε)·w_case·dA     (완화 폭 ε 로 매끄러움)
        → P₂D(w) = −dA_end/dw                   (w 에 대한 중심차분)
        → A_b(w) = P₂D·(L−2w) + 2·A_end

ε 완화가 경사를 만드는 핵심이다. 계단 지시함수였다면 어떤 계수를 조금
움직여도 A_end 가 계단으로만 변해 경사가 0 이거나 정의되지 않는다.

⚠ 거리에 **hard min 을 쓰면 안 된다.** 대칭 형상에서는 한 격자점이 두 개
   이상의 윤곽 점과 똑같이 떨어진 '동점'이 흔하다. hard min 은 그 중 하나만
   골라 경사를 흘리므로 대칭이 깨지고, 실제로는 0 이어야 할 방향(예: 스타의
   위상 회전)에 가짜 경사가 생긴다. 유한차분과 대조하면 바로 드러난다.
   soft-min = −τ·logsumexp(−d/τ) 을 쓰면 동점이 평균되어 사라진다.

⚠ 다만 soft-min 은 **값 자체를 τ·ln(주변 샘플 수) 만큼 낮춘다**. 그대로 쓰면
   정방향 정확도가 수 % 나빠진다(해석해 대조에서 확인). 그래서 값은 hard min,
   경사는 soft-min 을 쓰는 straight-through 방식을 채택한다:

       φ = φ_hard.detach() + (φ_soft − φ_soft.detach())

   정방향은 hard min 과 비트 단위로 같고, 역방향 경사는 soft-min 의 것이 흐른다.
"""
from __future__ import annotations

import numpy as np
import torch

DTYPE = torch.float64

#: 둘레 중심차분의 스텐실 폭 — 격자 간격의 몇 배로 둘 것인가 (`d = STENCIL_CELLS·h`).
#:
#: **2.0 이 맞다. 재서 확인했다 (2026-10-10).** 초판 주석은 "셀을 직접 세면 격자
#: 양자화 때문에 계단처럼 튀니까 넓게 잡는다" 라고만 적고 얼마나 넓어야 하는지는
#: 남기지 않았다. 좁히면 값이 좋아지는 것처럼 보여서 1.0 으로 내렸다가 되돌렸다 —
#: **값만 재고 경사를 안 쟀기 때문**이다. 두 축을 같이 보면 명백히 상충한다:
#:
#:     d      경사 cos(자동미분, 유한차분)   총역적 편향 (해석 BATES 대비)
#:     2.00·h          0.99811                      +2.77 %      ← 채택
#:     1.50·h          0.98541                      +2.29 %
#:     1.25·h          0.98485                      +1.93 %
#:     1.00·h          0.87107                      +1.50 %
#:     0.50·h          0.91384                      +0.06 %
#:
#: 좁은 스텐실은 총역적을 해석해에 0.06 % 까지 맞추지만 **경사 방향이 무너진다**
#: (cos 0.87). 이 프로젝트는 경사로 역설계를 하므로 경사가 1차 자산이다 —
#: 값 2.7 % 편향을 지고 경사를 지키는 쪽이 맞다. 초판의 선택이 옳았다.
#:
#: ⚠⚠ **정정: 위 표의 '총역적 편향' 열은 이전 값이고 지금은 무효다.**
#:    수정이 둘레를 반 스텝 중심차분으로 고치면서 이 교환 관계가 사라졌다.
#:    현재 채택 구성(2.00·h + 반 스텝 중심차분)의 실측 편향은 **−0.03 %** 다
#:    (해석 BATES 대비, n=121/241 · r₀ 10·12·14). 즉 "값 2.7 % 편향을 지고
#:    경사를 지킨다" 는 교환은 **더 이상 존재하지 않는다** — 경사도 지키고 값도
#:    맞는다. 스텐실 2.00·h 선택 자체는 그대로 옳고(cos 0.99811, 2026-10-10
#:    재측정), 경사 cos 열은 재현하지 못해 **미확인**으로 남겨 둔다.
#:
#: ⚠⚠ **그리고 "∫A_b dw 가 V(0) − V(끝) 보다 3 % 크다" 도 무효다.** 실측:
#:    **−0.024 ~ −0.037 %** (n=121/241, r₀ 10·12·14, L=120). 수정이 이미
#:    닫은 항목이다 (171 행이 "−0.14 % ← 채택" 으로 기록). 없는 3 % 편향을
#:    총역적 예산에 넣으면 안 된다. 🔜 로 남겨 둔 "자기 적분 항등식 정규화" 는
#:    **필요 없어졌다** — 고칠 편향이 없다.
#:
#: ⚠ numpy 판(`burnback2d.perimeter`)과 ODE 판정자(`ode_judge`)가 **같은 값**을 써야
#:    한다 — 셋이 갈라지면 같은 설계에서 서로 다른 A_b 가 나온다.
STENCIL_CELLS: float = 2.0


class DiffSegment2D:
    """단면 일정 세그먼트의 미분가능 burnback.

    격자·외벽 가중치처럼 형상 계수와 무관한 것은 한 번만 만들어 재사용한다.

    Parameters
    ----------
    d_outer : float      그레인 바깥지름 [mm] (외벽은 연소면이 아니다)
    n : int              한 변의 격자 수
    eps_cells : float    완화 폭을 격자 간격의 몇 배로 둘지
    n_samples : int      윤곽 샘플 점 수 (거리장의 기준점)
    """

    def __init__(self, d_outer: float, n: int = 241, eps_cells: float = 1.5,
                 n_samples: int = 1024, tau_cells: float = 0.25,
                 device: str = "cpu", dtype=DTYPE):
        self.d_outer = float(d_outer)
        self.r_outer = self.d_outer / 2.0
        self.n = int(n)
        self.eps_cells = float(eps_cells)
        self.device, self.dtype = device, dtype

        xs = torch.linspace(-self.r_outer, self.r_outer, self.n, dtype=dtype, device=device)
        self.h = float(xs[1] - xs[0])
        self.cell_area = self.h ** 2
        self.eps = self.eps_cells * self.h

        X, Y = torch.meshgrid(xs, xs, indexing="xy")
        self.X, self.Y = X, Y
        self.pts_grid = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)   # (G,2)
        self.rr = torch.hypot(X, Y).reshape(-1)
        self.th_grid = torch.atan2(Y, X).reshape(-1)

        # 외벽도 같은 폭 ε 로 완화해 면적 오차가 1차로 남지 않게 한다
        w_case = torch.clamp((self.r_outer - self.rr) / self.eps + 0.5, 0.0, 1.0)
        self.sel = w_case > 0
        self.w_case = w_case[self.sel]
        self.rr_sel = self.rr[self.sel]
        self.th_sel = self.th_grid[self.sel]
        self.pts_sel = self.pts_grid[self.sel]

        self.theta_s = torch.linspace(0.0, 2 * np.pi, n_samples + 1,
                                      dtype=dtype, device=device)[:-1]
        self.tau = float(tau_cells) * self.h      # soft-min 온도

    # ------------------------------------------------------------- 윤곽·거리장
    def radius(self, r0, a, b, theta):
        """r(θ) = r0 + Σ_k [a_k cos kθ + b_k sin kθ]  (미분 가능)."""
        k = torch.arange(1, a.shape[0] + 1, dtype=self.dtype, device=self.device)
        ang = k[:, None] * theta[None, :]
        return r0 + (a[:, None] * torch.cos(ang) + b[:, None] * torch.sin(ang)).sum(0)

    def phi(self, r0, a, b, chunk: int = 4096, hard: bool = False):
        """부호 있는 거리장 (추진제 쪽 +, 포트 내부 −). 셀 선택분만 반환.

        hard=True 면 보고·검증용 hard min (경사 없음), 기본은 soft-min.
        격자 쪽을 잘라 chunk 마다 전체 샘플을 보므로 logsumexp 가 정확하다.

        ⚠ **straight-through 다 — 값은 hard-min, 경사는 soft-min.** 그래서 이
           모델에서 `경사 ≠ ∇(손실)` 이고, 그 대가가 측정됐다
           (`experiments/sample_resolution.py --fd`). 같은 손실을 중심차분해
           autograd 와 맞춘 비 (경사 방향, 힌지 꺾임 밖 탐침):

               n_grid=121 : 1.71 ~ 1.78      n_grid=241 : 1.37 ~ 1.57
               n_grid=321 : 0.94 ~ 1.05

           ε 를 100 배 바꿔도(1e-3 ↔ 1e-2) 거의 안 변하므로 수치 잡음이 아니라
           **체계적 편향**이고, `n_samples` 를 5 배로 올려도 그대로다. soft-min
           의 기울기가 τ 창 안의 표본들에 분배되어 hard-min 보다 작기 때문이다.

           함의 두 가지.
             · 방향은 맞으므로 하강은 된다. 보폭이 1.4~1.8 배 작게 먹을 뿐이다.
             · 그러나 비가 **격자마다 다르다** (121 에서 1.72, 321 에서 1.0).
               격자를 바꾸면 Adam 의 실효 학습률이 바뀐다는 뜻이므로, 격자
               간 수렴 비교에 반복 수나 보폭을 섞어 읽지 말아야 한다.

           ⚠ 이 비를 다시 재려면 **모든 사양이 허용 τ 밖인 점**에서 재야 한다.
              τ 근처에서는 힌지(`max(0, err−τ)²`)에 꺾임이 있어 중심차분이
              양쪽을 평균해 버리고, 비가 ε 에 따라 1.26 → −0.76 → 0.41 로
              뒤집힌다 — 모델 결함이 아니라 측정 위치 결함이다.
        """
        r_s = self.radius(r0, a, b, self.theta_s)
        xy_s = torch.stack([r_s * torch.cos(self.theta_s),
                            r_s * torch.sin(self.theta_s)], dim=1)      # (M,2)

        outs = []
        for i in range(0, self.pts_sel.shape[0], chunk):
            d = torch.cdist(self.pts_sel[i:i + chunk], xy_s)            # (c,M)
            d_hard = d.min(dim=1).values
            if hard:
                outs.append(d_hard)
            else:
                d_soft = -self.tau * torch.logsumexp(-d / self.tau, dim=1)
                # 값은 hard, 경사는 soft (straight-through)
                outs.append(d_hard.detach() + (d_soft - d_soft.detach()))
        d_min = torch.cat(outs)

        r_at_grid = self.radius(r0, a, b, self.th_sel)                  # 부호 판정용
        sign = torch.where(self.rr_sel >= r_at_grid, 1.0, -1.0)
        return sign * d_min

    # ------------------------------------------------------------------ 연소량
    def area_end(self, phi, w):
        """남은 추진제 단면적 A_end(w) [mm²]. w 는 스칼라 또는 (W,) 텐서."""
        w = torch.as_tensor(w, dtype=self.dtype, device=self.device).reshape(-1)
        frac = torch.clamp((phi[None, :] - w[:, None]) / self.eps + 0.5, 0.0, 1.0)
        return (frac * self.w_case[None, :]).sum(dim=1) * self.cell_area

    def perimeter(self, phi, w, w_max: float = None):
        """등고선 둘레 P₂D(w) = −dA_end/dw [mm]. 면적의 중심차분으로 얻는다.

        소진 근처 3d 는 **반 스텝 중심차분**을 쓴다 — 전진 팔이 d/2 만 나가므로
        바깥벽을 넘어 생기는 인위적인 혹을 피하면서 **중심성을 잃지 않는다.**

        ⚠ **초판은 여기서 1차 후방차분을 썼고, 그것이 총역적을 부풀린 원인이었다.**
          `−(A(w) − A(w−d))/d` 는 P 를 `w` 가 아니라 **`w − d/2` 에서** 추정한다.
          반 스텝 어긋난 값을 w 에 꽂으니 적분이 망원경처럼 접히지 않고,
          `∫A_b dw` 가 `V(0) − V(끝)` 보다 커진다. 어긋남은 **전부 이 구간에만**
          있었다 (누적 불균형: 시작 3d −0.00 %p, 중간 −0.00 %p, 끝단 3d +1.81 %p).

          차수를 올리거나(2차 후방차분) 비율로 보정하는 것은 증상 치료라 듣지 않거나
          부작용이 있었다. 중심성을 되찾는 것이 원인 치료다.

              끝단 추정자              ∫A_b 초과   외부 기준 시뮬 오차 (평균/최대)
              1차 후방차분 (초판)        +2.34 %     1.90e-04 / 2.18e-03
              2차 후방차분              +1.75 %     —            ← 차수 문제가 아니다
              끝단 3d 비율 보정          −0.05 %     4.49e-04 / 7.06e-02  ← 기준 시뮬에서 탈락
              전환 없이 중심차분만         −0.08 %     1.83e-04 / 1.30e-03
              **반 스텝 중심차분**        −0.14 %     1.82e-04 / 3.89e-04  ← 채택

          비율 보정은 자기 일관성(∫A_b)만 보면 가장 좋아 보였지만, **외부 기준
          시뮬**(`reference/srm_bates_fast`)과 대조하니 평탄부 끝에서 7 % 어긋났다.
          끝단 창 전체를 같은 비율로 깎는데 실제 오차는 P 가 작아질수록 커지는
          형태라, P 가 아직 큰 창 입구를 과도하게 깎기 때문이다. **자기 일관성은
          정확성이 아니다** — 독립 기준이 있어야 걸러진다.
        """
        w = torch.as_tensor(w, dtype=self.dtype, device=self.device).reshape(-1)
        d = STENCIL_CELLS * self.h          # 모듈 상단 상수 — 셋이 같은 값을 쓴다
        p = -(self.area_end(phi, w + d) - self.area_end(phi, w - d)) / (2.0 * d)
        if w_max is not None:
            near = w > (w_max - 3.0 * d)
            if bool(near.any()):
                wn = w[near]
                half = -(self.area_end(phi, wn + 0.5 * d)
                         - self.area_end(phi, wn - 0.5 * d)) / d
                p = torch.where(near, torch.zeros_like(p).masked_scatter(near, half), p)
        return torch.clamp(p, min=0.0)

    def burn_area(self, phi, w, length: float, ends_burn: bool = True,
                  w_max: float = None):
        """세그먼트 연소면적 A_b(w) [mm²]."""
        w = torch.as_tensor(w, dtype=self.dtype, device=self.device).reshape(-1)
        p2d = self.perimeter(phi, w, w_max=w_max)
        if not ends_burn:
            return p2d * float(length)
        l_rem = torch.clamp(torch.as_tensor(length, dtype=self.dtype) - 2.0 * w, min=0.0)
        # 끝면은 `l_rem > 0` 일 때만 존재한다 (numpy 판과 동일 — burnback2d 참조).
        # `w = L/2` 에서 volume = 0 인데 A_b = 2·A_end > 0 이던 모순을 없앤다.
        # 지시자는 `detach` 상태의 불리언이라 경사를 끊지 않는다 — `l_rem` 이
        # 0 인 지점에서는 이미 `clamp` 가 경사를 0 으로 만든다.
        ends = torch.where(l_rem > 0.0, 2.0 * self.area_end(phi, w),
                           torch.zeros_like(l_rem))
        return p2d * l_rem + ends

    def volume(self, phi, w, length: float, ends_burn: bool = True):
        w = torch.as_tensor(w, dtype=self.dtype, device=self.device).reshape(-1)
        a_end = self.area_end(phi, w)
        if not ends_burn:
            return a_end * float(length)
        l_rem = torch.clamp(torch.as_tensor(length, dtype=self.dtype) - 2.0 * w, min=0.0)
        return a_end * l_rem

    # --------------------------------------------------------------------- 편의
    def web_max(self, phi) -> float:
        """소진 웹 [mm] — `area_end` 가 **정확히 0** 이 되는 지점. 경사 없는 진단값.

        `frac = clip((φ − w)/ε + 0.5, 0, 1)` 가 모든 셀에서 0 이 되는 곳이므로
        `max(φ) + ε/2` 다. 여기까지 태워야 장전 질량(= `area_end(0)` 기준)이
        전부 소비된다.

        ⚠ **정확히 ε/2 에 착지하면 안 된다.** 그 지점은 `clip` 의 꺾임점이고
          PyTorch 의 `clamp` 는 경계를 **포함**해 경사를 1 로 준다. 웹 격자의
          마지막 점이 거기 놓이면 기여가 0 인 셀들이 경사만 남긴다. 실측
          (원형 r₀ = 12, a = b = 0 — 경사가 **정확히 0 이어야 하는 안장점**):

              w_cap = max(φ) + 0.00·ε  →  |∂I/∂a| 6.4e-13
                               + 0.25·ε →            6.4e-13
                               + 0.50·ε →        **3.8e-05**   ← 꺾임점 (무효)
                               + 0.75·ε →            5.9e-13

          ⚠⚠ **정정: 그 꺾임점이 재현되지 않는다.** `web_max` 를
          `max(φ)+frac·ε` 로 몽키패치하고 총역적을 backward 한 실측(원형 r₀=12,
          n=121): +0.00/0.25/0.50/0.75·ε 에서 **1.54/1.83/1.62/1.52e-13** — 0.50·ε
          에 혹이 없다. 즉 `(1.0 + 1e-9)` 넛지의 근거는 더 이상 측정으로
          뒷받침되지 않는다 (코드 자체는 무해하므로 그대로 둔다). 3.8e-05 는 같은 자리의 곡선 손실 경사(3.4e-05)와 같은 자릿수라 그냥 둘
          수 없다. 꺾임점 바로 바깥으로 비켜 둔다 (ε·1e-9 ≈ 6e-10 mm — 수치적으로
          보이지 않는 양이다).

        ⚠⚠ **정정: `endgame_scale` 은 구현이 없다.** 저장소 전체 grep 결과
          이 이름은 주석 두 곳(여기와 `burnback2d.py:112`)에만 있다. 아래 표의
          "보정 + 웹 확장(채택)" 행 −0.55/−0.68/−0.70 % 도 재현되지 않는다 —
          현재 코드가 내는 값은 그 윗 행("보정만")의 −0.43/−0.52 % 다. 끝단
          처리는 반 스텝 중심차분으로 대체됐고 비율 보정은 169 행이
          "기준 시뮬에서 탈락" 으로 기록한다. 아래 서술은 이전 기록으로만 읽을 것.

        ⚠ 이 값은 `endgame_scale` 보정과 짝이다. 그때는 `max(φ)` 였고,
          그때는 늘리면 총역적 편향이 **커졌다** (+1.56 % → +2.77 %). 둘레의
          과대평가를 웹 절단이 상쇄하고 있었기 때문이다. 보정이 들어간 뒤에는
          두 결함이 같이 닫힌다 (측정, 원형 3 사례):

              구성                경사 cos   총역적 편향            V(끝)
              이전   (둘 다 없음)  0.99856  +0.86/+1.56/+1.08 %   336 mm³ 안 탐
              보정만               0.99856  −0.43/−0.52/−0.52 %   336 mm³ 안 탐
              웹 확장만             0.99811  +1.49/+2.77/+2.06 %   0
              **보정 + 웹 확장**     0.99811  −0.55/−0.68/−0.70 %   0      ← 채택

          즉 **한쪽만 고치면 나빠지고, 같이 고쳐야 좋아진다.**
        """
        return float(phi.max().detach()) + 0.5 * float(self.eps) * (1.0 + 1e-9)


def params(r0: float, a=(), b=(), requires_grad: bool = True, dtype=DTYPE):
    """(r0, a, b) 를 학습 가능한 텐서로 묶어 만든다."""
    r0_t = torch.tensor(float(r0), dtype=dtype, requires_grad=requires_grad)
    a_t = torch.tensor(list(a) if len(a) else [0.0], dtype=dtype, requires_grad=requires_grad)
    b_t = torch.tensor(list(b) if len(b) else [0.0], dtype=dtype, requires_grad=requires_grad)
    return r0_t, a_t, b_t
