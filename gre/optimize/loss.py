"""손실 함수 — 허용 집합 안으로 가는 힘과 집합 안에서 고르는 힘.

    loss = Σ_j  w_j · hinge( 오차_j − τ_j )²      사양 — 집합 안에서는 **정확히 0**
         + 제작 제약 벌점                          제작 가능 집합 안으로
         + λ₂ · 2차 기준                           집합 안에서 무엇을 고를지

    hinge(x) = max(x, 0)
    사양     = 곡선 모양, 총역적, 최대추력 — **압력 곡선이 입력이면 압력 모양·
               최대압이 더해져 5 항**이다.
               ⚠ 모듈 머리에 3 항만 적혀 있어 압력이 사양으로 올라간
                 경우가 안 보였다. 합격 기준은 "추력 **및 압력** 10 % 이내" 다.
    2차 기준 = 모드 진폭 합(그룹 L1), 최대압 목표(소프트)

**왜 힌지인가.** 역설계의 답은 설계 하나가 아니라 **허용 집합** A 다.
A 안에 들어가면 사양을 전부 만족한 것이므로 더 밀 이유가 없고, 그 뒤의 선택은
2차 기준이 해야 한다. 초판은 가중 제곱합이라 집합 안에 들어간 뒤에도 계속 밀었고,
가중치가 낮은 사양을 희생해 **집합 밖으로 다시 나가는** 일이 생겼다.

측정 (6-로브 목표, K = 0 고정, 300 반복):

    반복  45 : 모양 18.99 %   I  4.09 %   F 24.75 %
    반복 135 : 모양  3.26 %   I 24.35 %   F  0.72 %
    반복 300 : 모양  3.09 %   I 24.35 %   F  1.10 %

λ_모양 : λ_I = 10 : 1 이었으므로 총역적을 희생해 모양을 맞추는 것이 **최적화기
입장에서는 옳은 행동**이었다. 그런데 합격 기준은 셋 다 10 % 이하다. 즉 최적화기가
열심히 할수록 합격에서 멀어지는 구간이 있었다. 힌지면 구조적으로 사라진다.

**가중치의 의미가 바뀐다.** 초판의 λ₁ ≫ λ₂ ≫ λ₃ 은 "우선순위" 였다. 힌지에서는
집합 안에서 전부 만족하므로 순서가 필요 없고, 세 사양의 w 를 **같게 둔다**.
우선순위는 손실이 아니라 **실현 불가 보고**(`runner.admissibility`)에만 남는다 —
그마저도 "무엇부터 양보할지" 가 아니라 "무엇이 얼마나 모자라는지" 를 적는 용도다.

**모든 항을 목표값으로 무차원화한다.** 추력은 N, 총역적은 N·s, 압력은 MPa 라
날것으로 더하면 단위가 큰 항이 지배한다. 힌지의 문턱 τ 도 같은 무차원 척도여야 한다.

**구간 가중치 ω(t)** — 모델에 없는 물리가 지배하는 구간을 맞추려 들면 형상이
왜곡된다. 경계는 퍼센트가 아니라 **물리적 사건**으로 잡는다.

| 경계 | 기준 | 근거 |
|---|---|---|
| 점화 끝 | 챔버압이 준정상 평형값의 90 % 도달 | 충전 과도가 끝나는 시점 |
| 테일오프 시작 | 웹 소진 | 슬리버 연소 시작 = 가정이 깨지는 지점 |

⚠ **총역적은 전 구간을 적분한다.** 제외는 "모양을 맞출 때"만 적용된다.

경계 자체는 `detach` 해서 경사를 흘리지 않는다. 창의 위치가 설계 변수로
미분되면 "창을 옮겨서 손실을 줄이는" 퇴화한 해가 생긴다.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from .constraints import D_THROAT_MIN_MM
from .forward import resample

#: ε 적합이 후보로 볼 **물리적으로 가능한 노즐목 범위** [mm²].
#:
#: 하한은 가공 한계(`D_THROAT_MIN_MM`), 상한은 이 프로젝트가 다루는 케이스
#: 단면(Ø60 mm)이다. 퇴화 가지를 막는 것이 유일한 목적이므로 넉넉하게 잡는다 —
#: 호출자가 노즐목을 **입력으로** 알면 `at_bounds` 로 더 좁은 상자를 넘긴다.
AT_PLAUSIBLE_MM2 = (float(np.pi * (D_THROAT_MIN_MM / 2.0) ** 2),
                    float(np.pi * 30.0 ** 2))

#: ε 적합의 **상한을 정하는 출구압 하한** — `p_e >= PE_PA_MIN · p_ambient`.
#:
#: ⚠ **이 상한이 없으면 적합이 격자 끝까지 달려가 모순을 흡수한다
#:   (실제로 겪었다).** `np.geomspace(1.05, 40.0, 96)` 에 물리적 천장이 없어서,
#:   서스테인 압력만 2 배 틀린 (같은 모터에서 나올 수 없는) 쌍에서 적합이
#:   **ε = 40.00 — 격자의 맨 끝** 을 골라 변동폭을 77.79 % → **24.91 %** 로
#:   깎았다. 함의 A_t 가 97.8 mm² 로 멀쩡해 보여서 `at_bounds` 도 못 막았다.
#:
#: 상한은 숫자를 새로 고르지 않고 **입력 압력 곡선에서 뽑는다**: 출구압이
#: `PE_PA_MIN · p_ambient` 로 떨어지는 ε 이 상한이다. 그 위는 노즐이 심하게
#: 과팽창해 **유동이 박리**되므로, 이 함수가 역산하는 이상 `C_f` 모델 자체가
#: 무효다 — 적합이 애초에 들어가면 안 되는 구간이다.
#:
#: 값은 측정으로 정했다 (문턱 10 %, `test_optimize` 단언 30 %):
#:
#:     frac   외부 사례(참 ε=8.0)      서스테인만 2배(모순)   자기 일관   회귀 4종
#:     0.4    24.90 % → **거부** ✗     89.9 % 거부           0.01 %      치우침 0.013 %
#:     0.3    18.73 % → **거부** ✗     86.2 % 거부           0.18 %      치우침 0.013 %
#:     0.2     4.32 % 통과 (ε 7.50 —   78.2 % 거부           0.01 %      치우침 0.013 %
#:                          **상한에 붙음**)
#:     0.1     0.49 % 통과 (ε 8.05 —   46.9 % 거부           0.20 %      치우침 0.017 %
#:                          범위 **내부**)
#:     없음    0.01 % 통과              **24.91 % 통과** ✗    0.33 %      치우침 0.012 %
#:
#: 0.4 · 0.3 은 물리적으로 가장 정직한 분리 기준(Summerfield ~0.4)인데도
#: **탈락이다** — 사용자 제공 외부 사례의 참 ε = 8.0 을 4.41 / 5.49 로 깎아서
#: 멀쩡한 입력을 거부한다 (바로 고친 그 거짓 양성이 되돌아온다).
#: 0.2 는 통과하지만 적합이 **상한에 걸려** ε 7.50 < 참 8.0 이고, 경계에 걸린
#: 적합은 수렴한 것이 아니라 그 경계가 그 데이터에 틀렸다는 뜻이다 — 격자 끝의
#: ε = 40 과 방향만 반대인 같은 병이다. **0.1 만이 참 ε 를 범위 내부에서
#: 복원하면서 모순 쌍 둘을 문턱의 4.7 배 넘게 거부한다.**
#:
#: 회귀 설계 4 종(참 ε 3.3~4.7)은 어느 상한에도 무관심하다 — 상한이 결정되는
#: 사례는 참 ε 가 큰 외부 사례와 모순 쌍, 그 둘뿐이다.
PE_PA_MIN: float = 0.1

#: ε 적합 격자의 하한과 점 수. 하한은 "거의 목만 있는 노즐", 점 수는 옛 판과
#: 같게 둔다 (상한만 데이터에서 뽑히도록 바꾼 것이지 분해능을 건드린 게 아니다).
EPS_FIT_LO: float = 1.05
EPS_FIT_N: int = 96


@dataclass
class Target:
    """목표 곡선과 스칼라 목표들.

    **압력은 선택 입력이다.** 세 경우가 있고 처리가 다르다.

    | 입력 | 압력의 성격 | 사양 개수 |
    |---|---|---|
    | 없음 | 아무 목표도 없음 — 케이스 허용압 하드 제약만 | 3 |
    | `p_max` 만 (스칼라) | 소프트 2차 기준 (가중치 1e-3) — **A안** | 3 |
    | `p` (곡선) | **힌지 사양 τ = 10 %** — B안 | 5 |

    왜 이렇게 갈라 두는가. 추력만 입력이면 `F = C_f·P·A_t` 가 **곱만 묶으므로**
    `A_t` 가 자유인 동안 `P` 는 추력 사양이 보지 않는 방향이다 (실측: 같은 허용
    집합 안에서 최대압 2.64 배, `P·A_t` 변동 1.5 % — §4.5). 그 자유도를 없애려면
    압력을 **독립 정보로 따로 주어야** 한다. 스칼라 `p_max` 하나로는 곡선 전체를
    묶을 수 없어 소프트로 남기고, 곡선이 들어오면 사양으로 승격한다.

    `p` 를 주면 사양이 **둘** 늘어난다 — 곡선 모양과 최대압. 추력 쪽이 모양·총역적·
    최대추력 세 항인 것과 같은 구조다 (압력에는 '총역적' 에 해당하는 양이 없다).
    """

    t: torch.Tensor                  # [s]
    f: torch.Tensor                  # [N]
    total_impulse: float = None      # [N·s]  None 이면 곡선에서 적분
    f_max: float = None              # [N]
    p_max: float = None              # [MPa] p 가 없으면 소프트 목표, 있으면 사양
    p: torch.Tensor = None           # [MPa] 선택 — 압력 곡선 (t 와 같은 길이)

    def __post_init__(self):
        if self.total_impulse is None:
            self.total_impulse = float(torch.trapezoid(self.f, self.t))
        if self.f_max is None:
            self.f_max = float(self.f.max())
        if self.p is not None:
            if self.p.shape != self.t.shape:
                raise ValueError(
                    f"압력 곡선의 길이가 시간축과 다르다: p {tuple(self.p.shape)} "
                    f"vs t {tuple(self.t.shape)}. 같은 시간축으로 리샘플해서 넣으십시오.")
            if float(self.p.min()) <= 0.0:
                raise ValueError("압력 곡선에 0 이하 값이 있다 — 절대압 [MPa] 로 "
                                 "넣으십시오 (게이지압이면 대기압을 더해야 한다).")
            if self.p_max is None:
                self.p_max = float(self.p.max())

    @property
    def pressure_is_spec(self) -> bool:
        """압력 곡선이 들어왔는가 — 사양 5개인가 3개인가를 가른다."""
        return self.p is not None


@dataclass
class Tolerances:
    """허용 집합 S_사양 의 경계. 상대값, 1.0 = 100 %.

    전부 같은 값이 기본이다. 사양마다 다른 문턱을 쓰고 싶으면 여기서 바꾼다 —
    그러면 "모양은 5 % 안, 총역적은 15 % 안" 같은 비대칭 집합이 된다.

    ⚠ 압력 두 항(`p_shape`, `p_max`)은 **`Target.p` 가 들어왔을 때만** 쓰인다.
      압력 곡선이 없으면 압력은 사양이 아니라 소프트 2차 기준이다 (§4.5 압력의 두 층).
    """

    shape: float = 0.10              # F_max 대비 가중 RMS
    impulse: float = 0.10
    f_max: float = 0.10
    p_shape: float = 0.10            # 목표 P_max 대비 가중 RMS — 압력 곡선 입력 시만
    p_max: float = 0.10              # 최대압 상대 오차 — 압력 곡선 입력 시만

    def as_pct(self, with_pressure: bool = False) -> dict:
        out = {"모양": 100 * self.shape, "총역적": 100 * self.impulse,
               "최대추력": 100 * self.f_max}
        if with_pressure:
            out["압력 모양"] = 100 * self.p_shape
            out["최대압"] = 100 * self.p_max
        return out


@dataclass
class LossWeights:
    """사양은 힌지(가중치 동일), 나머지는 제약과 2차 기준."""

    # ── 사양 — 힌지. 집합 안에서는 0 이므로 서로 순서를 둘 이유가 없다
    shape: float = 1.0
    impulse: float = 1.0
    f_max: float = 1.0
    # ── 압력 사양 — **`Target.p` 가 들어왔을 때만** 쓰이는 힌지
    p_shape: float = 1.0
    p_peak: float = 1.0
    # ── 제약
    manufacturing: float = 1e1       # 제작 — 목표보다 세게
    case_pressure: float = 1e3       # 하드 — 안전
    # ── 2차 기준 — 집합 안에서 무엇을 고를지
    #    ⚠ `p_max` 는 **압력 곡선이 없을 때만** 쓰인다. 곡선이 들어오면 위의 힌지가
    #    같은 양을 사양으로 겨냥하므로 이 항은 끈다 — 켜 두면 사양 밖 기준이 다시
    #    선택에 끼어든다.
    p_max: float = 1e-3              # 최대압 소프트 목표 (§4.5 압력의 두 층)
    sparsity: float = 3e-2           # 그룹 L1 (모드 희소성) — 아래 설명
    sparsity_warmup: float = 0.35    # 전체 반복의 이 비율까지는 0, 이후 선형 증가
    ignition_frac: float = 0.90      # 점화 끝 판정 (평형압 대비)
    window_soft_s: float = 0.02      # 창 가장자리를 부드럽게 (시그모이드 폭)


def shape_window(t, p, w, w_cap, weights: LossWeights):
    """ω(t) — 점화 끝 ~ 웹 소진. 경계는 경사를 흘리지 않는다."""
    with torch.no_grad():
        p_plateau = torch.quantile(p[p > 0.5 * p.max()], 0.5)
        hit = (p >= weights.ignition_frac * p_plateau).nonzero()
        t_ign = t[hit[0, 0]] if hit.numel() else t[0]
        done = (w >= 0.999 * w_cap).nonzero() if w is not None else torch.empty(0)
        t_end = t[done[0, 0]] if done.numel() else t[-1]
        d = weights.window_soft_s
    return torch.sigmoid((t - t_ign) / d) * torch.sigmoid((t_end - t) / d)


def fp_ratio_spread(t, f, p, ignition_frac: float = 0.90,
                    q=(0.05, 0.95), gamma: float = 1.133,
                    p_ambient: float = 0.1013, eps: float = None,
                    cf_eff: float = 1.0, at_bounds=None,
                    eps_grid=None, separation_frac: float = 0.0) -> dict:
    """**함의 노즐목** `A_t = (F/P)/C_f(P, ε)` 의 변동폭 — 두 입력 곡선의 모순 검사.

    노즐목 `A_t` 는 시간에 대해 **고정**이므로

        F(t) = C_f(P(t), ε) · P_c(t) · A_t   →   A_t = (F/P) / C_f(P, ε)

    이고, 입력한 두 곡선이 한 모터에서 나왔다면 이 값이 **상수**다. 변동이 남으면
    두 곡선이 서로 모순이다.

    ═══════════════════════════════════════════════════════════════════════
    ⚠ **추정자를 `F/P` 에서 함의 노즐목으로 바꿨다.** 이유가 측정에 있다.

      `F/P = C_f·A_t` 는 `C_f` 의 압력 의존만큼 **정당하게** 변한다. 그 변화를
      남겨 두면 압력 범위가 넓은 목표에서 멀쩡한 입력이 거부된다 — 자기 일관된
      부스트(5 MPa)-서스테인(1.5 MPa) 쌍에서 `F/P` 변동이 **18.25 %** 로 문턱
      20 % 에 붙었다. `C_f` 로 나누면 그 쌍은 정의상 **0 %** 가 된다.

      ε 는 주지 않으면 **두 곡선에서 적합한다**. 추측하지 않는다.
    ═══════════════════════════════════════════════════════════════════════

    ═══════════════════════════════════════════════════════════════════════
    ⚠ **ε 를 추측하면 멀쩡한 입력을 거부한다.**

      이전까지 ε 를 주지 않으면 "창 안 평균 압력의 **최적 팽창비**" 로
      잡았다. 그건 측정이 아니라 **설계 관례**다 — 실제 노즐이 최적 팽창인
      경우가 오히려 드물다.

      실측 (사용자 제공 외부 사례, 참 ε = 8.0 · 참 A_t = 61.75 mm²):

          ε 자동 4.07 (최적 팽창 가정)  →  함의 A_t 56.99,  변동 **27.47 %**  거부
          ε 적합 8.000                  →  함의 A_t **61.750**, 변동 **0.00 %**  통과

      **거짓 양성이었다.** 그 곡선은 이 모델 자신이 만든 것이라 모순일 수가 없다.

      ε 가 물리적으로 중요해서가 아니다 — 같은 형상·같은 목에서 ε 를 4.07 ↔ 8.0
      으로 **2 배** 틀려도 최대추력 +1.17 %, 총역적 +5.49 %, 최대압 **0 %** 로
      전부 허용(10 %) 안이다. 문제는 **이 검사가 그 작은 차이에 과민**하다는
      것이다: `A_t = (F/P)/C_f(P, ε)` 는 **나누기**이므로 ε 가 틀리면 `C_f` 의
      **압력 의존 모양**이 틀어지고, 함의 A_t 가 압력 구간을 따라 체계적으로
      흐른다. 이 함수는 바로 그 흐름을 재므로 ε 오차를 크게 증폭한다.

      그 민감도 자체는 결함이 아니다 — **방향만 바꾸면 측정 도구가 된다.**
      추측한 ε 위에서 판정하면 거짓 양성이고, ε 를 **적합한 뒤** 판정하면
      ε 와 A_t 를 둘 다 복원한 다음 "그래도 안 맞는가" 만 묻게 된다.
    ═══════════════════════════════════════════════════════════════════════

    ⚠ **적합에는 퇴화 가지가 있다 — 그래서 물리 범위로 가둔다.** ε 를 키우면
      `C_f` 가 0 을 지나가고 `A_t = (F/P)/C_f` 가 발산한다. 발산한 값은 "일정"
      하므로 변동폭이 **가짜로 0** 이 된다. 같은 사례에서 ε = 12 · 16 의 변동폭이
      0.00 % 인데 함의 A_t 가 1.9e8 · 1.4e9 mm² 였다. 그래서 적합은 `at_bounds`
      안에 드는 ε 만 후보로 본다 (호출자가 노즐목을 입력으로 알면 그 상자를
      넘기고, 모르면 가공 한계 ~ 케이스 단면).

    ⚠ **`at_bounds` 만으로는 부족했다 — 격자 끝으로 달아나는 길이 하나 더
      있었다.** 적합이 ε 를 키워 모순을 흡수하면서도 함의 A_t 는 멀쩡한 값에
      머무를 수 있다. 서스테인 압력만 2 배 틀린 쌍에서 적합이 **격자의 맨 끝
      ε = 40.00** 을 골라 변동폭을 77.79 % → 24.91 % 로 깎았고 (함의 A_t 97.8
      mm² — 상자 안이라 안 걸린다), 그 바람에 회귀 시험이 깨졌다. 지금은 격자
      **상한을 `PE_PA_MIN` 박리 한계로 입력 압력 곡선에서 뽑는다** (그 상수의
      주석에 선정 측정이 있다). 같은 쌍이 46.88 % 로 되돌아온다.

      그리고 적합이 그래도 격자 끝에 붙으면 `eps_at_bound` 로 **보고한다** —
      경계에 걸린 적합은 수렴이 아니라 "이 범위 안에 맞는 ε 가 없다" 는
      뜻이다.

    ⚠ **창은 구간이다 (점화 완료 ~ 연소 끝), 준위 마스크가 아니다.** 초판은
      `(p >= frac·plateau) & (f > 0.05·F_max)` 를 준위 마스크로 썼다. 단조 감소
      곡선에서는 구간과 같지만 **두 단 곡선에서는 다르다** — 평탄값이 높은 단에서
      잡히므로 마스크가 낮은 단을 통째로 버린다. 그러면 검사가 자기 창 밖의
      모순을 못 보고, 창은 검사받는 그 데이터에서 나오므로 **창을 끌고 가는
      오류는 스스로를 숨긴다.**

      실측: 자기 일관된 부스트-서스테인 쌍의 **부스트 압력만 2 배**로 틀리게 한
      쌍(함의 노즐목이 절반이 되는, 같은 모터에서 나올 수 없는 쌍)을 준위
      마스크는 **변동 0.00 % 로 통과**시켰다 (창 안 499/2000 점 — 부스트만).

    ⚠ 그 전에도 창을 `f > 0.05·F_max` 하나로만 잡던 판이 있었다. 그러면 점화
      상승과 테일오프가 들어오고, 그 구간은 압력이 낮아 노즐이 심하게 과팽창하므로
      `C_f` 가 급변해 **자기 일관된 쌍조차 변동 64.5 % 로 거부됐다.** 지금은
      점화 판정이 `shape_window` 와 같은 규칙(평형압의 `ignition_frac` 이상)이다.

    측정과 검사가 **이 함수 하나**를 공유하므로 문턱과 추정자가 어긋날 수 없다 —
    위의 실수 둘이 정확히 그 어긋남에서 나왔다.

    반환
        `at_mean`   함의 노즐목 평균 [mm²]
        `spread`    그 변동폭 (상대) ← **문턱과 비교하는 값**
        `fp_mean`   `F/P` 평균 [mm²] = `C_f·A_t`. 단위 오류 검사가 쓴다 —
                    그쪽은 `C_f` 모델을 **가정하지 않아야** 하므로 날것이 필요하다
        `fp_spread` 옛 추정자의 변동폭. 비교용으로 같이 돌려준다
        `eps`, `n`  쓴 팽창비와 창 안 점 수
        `eps_fitted` 적합으로 정한 ε (입력으로 받았으면 `None`)
        `eps_at_bound` 적합이 격자 끝에 붙었으면 `"하한"`/`"상한"`, 아니면
                    `None`. **판정에 쓰지 않고 보고만 한다** — 붙었다는 것은
                    어떤 물리적 ε 로도 두 곡선이 안 맞는다는 신호다
    창이 너무 좁으면 `None`.
    """
    from ..core.nozzle import optimum_area_ratio, thrust_coefficient

    with torch.no_grad():
        f_max = f.max()
        hi = p[p > 0.5 * p.max()]
        plateau = torch.quantile(hi, 0.5) if hi.numel() else p.max()
        lit = (p >= ignition_frac * plateau).nonzero()
        alive = (f > 0.05 * f_max).nonzero()
        if lit.numel() == 0 or alive.numel() == 0:
            return None
        m = torch.zeros_like(f, dtype=torch.bool)
        m[int(lit[0, 0]):int(alive[-1, 0]) + 1] = True
        if int(m.sum()) < 10:
            return None

        p_win = torch.clamp(p[m], min=1e-9)
        r = f[m] / p_win                       # = C_f·A_t [mm²]

        def _at_of(e):
            """그 ε 에서의 함의 노즐목 [mm²]."""
            cf_e = torch.tensor(
                [thrust_coefficient(max(float(x), 1.05 * p_ambient), p_ambient,
                                    float(e), gamma,
                                    separation_frac=separation_frac)
                 for x in p_win],
                dtype=r.dtype)
            return r / torch.clamp(cf_e * float(cf_eff), min=1e-9)

        def _rel_spread(v):
            mu = torch.clamp(v.mean(), min=1e-12)
            return float((torch.quantile(v, float(q[1]))
                          - torch.quantile(v, float(q[0]))) / mu)

        eps_fitted, eps_at_bound = None, None
        if eps is None:
            # ── ε 를 **적합한다**: 함의 A_t 가 가장 일정해지는 ε.
            #    퇴화 가지(발산해서 가짜로 일정해지는 구간)를 물리 범위로 막고,
            #    **박리 한계**로 상한을 (입력 압력 곡선에서) 뽑는다.
            lo_at, hi_at = (at_bounds if at_bounds is not None
                            else (AT_PLAUSIBLE_MM2[0], AT_PLAUSIBLE_MM2[1]))
            if eps_grid is not None:
                grid = np.asarray(eps_grid, dtype=float)
            else:
                # 창 안 **가장 낮은** 압력이 상한을 정한다 (가장 과팽창하는
                # 조건). 분위는 변동폭과 **같은** `q[0]` 을 쓴다 — 테일오프
                # 한 점이 상한을 끌어내리지 않게.
                p_sep = float(PE_PA_MIN) * float(p_ambient)
                p_lo = float(torch.quantile(p_win, float(q[0])))
                eps_hi = (optimum_area_ratio(p_lo, p_sep, gamma)
                          if p_lo > p_sep else EPS_FIT_LO)
                grid = np.geomspace(EPS_FIT_LO,
                                    max(float(eps_hi), EPS_FIT_LO * 1.0001),
                                    EPS_FIT_N)
            best = None
            for e in grid:
                at_e = _at_of(e)
                mu = float(at_e.mean())
                if not (lo_at <= mu <= hi_at) or not np.isfinite(mu):
                    continue                    # 물리적으로 불가능한 목 → 후보 아님
                sp = _rel_spread(at_e)
                if best is None or sp < best[0]:
                    best = (sp, float(e))
            if best is None:
                # 어떤 ε 로도 물리적인 목이 안 나온다 — 단위 오류 쪽 검사가 잡는다.
                # 여기서는 옛 관례(최적 팽창)로 되돌아가 **판정을 멈추지 않는다.**
                p_rep = float(p_win.mean())
                eps = (optimum_area_ratio(p_rep, p_ambient, gamma)
                       if p_rep > p_ambient else 1.0)
            else:
                eps = eps_fitted = best[1]
                # ⚠ **적합이 격자 끝에 붙었으면 수렴한 것이 아니다.** 경계에
                #   걸린 적합은 "이 범위 안에 최적이 없다" 는 뜻이고, 두 곡선이
                #   어떤 물리적 ε 로도 안 맞는다는 **모순의 신호**다. 실제로
                #   부스트 압력만 2 배 틀린 쌍은 모든 변종에서 하한 1.05 에
                #   붙는다. 게이트로 쓰지 않고 **보고만** 한다 — 양보 판단은
                #   사람이 한다.
                if abs(eps_fitted - float(grid[0])) <= 1e-9 * float(grid[0]):
                    eps_at_bound = "하한"
                elif abs(eps_fitted - float(grid[-1])) <= 1e-9 * float(grid[-1]):
                    eps_at_bound = "상한"
        cf = torch.tensor(
            [thrust_coefficient(max(float(x), 1.05 * p_ambient), p_ambient,
                                float(eps), gamma,
                                separation_frac=separation_frac)
             for x in p_win],
            dtype=r.dtype)
        # ⚠ **노즐 효율 `cf_eff` 로도 나눠야 한다.** 모델은
        #   `F = C_f · cf_eff · P · A_t` 를 쓰므로(`forward.py` 의 `cf_eff`),
        #   `C_f` 만으로 나누면 함의 A_t 가 `cf_eff` 배만큼 작게 나온다.
        #
        #   실측: 자기 일관된 설계 7 종(원형 3 · 스타 3 · 복합 1, A_t
        #   54~126 mm²)에서 치우침이 **전부 +4.8~5.0 %** 로 일정했다 —
        #   `cf_eff = 0.95` 와 정확히 일치한다. 잡음이 아니라 계통오차였고,
        #   그래서 보정할 수 있다. 호출자는 `cf_eff=motor.cf_eff` 를 넘긴다.
        #
        #   ⚠ **변동폭(`spread`)에는 영향이 없다** — 상수배이기 때문이다. 이
        #     수정이 바꾸는 것은 `at_mean` 의 절대값뿐이고, 그 값을 입력
        #     노즐목과 비교하는 `input_check.PHYS_FPAT` 가 그것을 쓴다.
        at = r / torch.clamp(cf * float(cf_eff), min=1e-9)   # 함의 노즐목 [mm²]

        def _spread(v):
            mu = torch.clamp(v.mean(), min=1e-12)
            return float((torch.quantile(v, float(q[1]))
                          - torch.quantile(v, float(q[0]))) / mu), float(mu)

        sp_at, mu_at = _spread(at)
        sp_fp, mu_fp = _spread(r)
        return {"at_mean": mu_at, "spread": sp_at, "eps_fitted": eps_fitted,
                "eps_at_bound": eps_at_bound,
                "fp_mean": mu_fp, "fp_spread": sp_fp,
                "mean": mu_fp,          # 하위 호환 — 옛 이름은 F/P 평균이었다
                "eps": float(eps), "n": int(m.sum())}


def _hinge(err, tol):
    """max(오차 − 허용치, 0)². 허용치 안에서는 값도 경사도 정확히 0 이다."""
    return torch.clamp(err - float(tol), min=0.0) ** 2


def curve_loss(t_sim, f_sim, p_sim, omega, target: Target, weights: LossWeights,
               tol: Tolerances = None):
    """사양 세 항은 힌지, 최대압은 2차 기준. 전부 무차원.

    ⚠ 오차를 **제곱이 아니라 상대 오차 자체**로 재서 문턱과 비교한다. 초판은
       제곱값을 손실로 썼기 때문에 "10 % 안" 이라는 문턱을 손실에 표현할 길이
       없었다. `sqrt` 안의 1e-12 는 오차 0 에서 경사가 발산해 `0 × inf = nan`
       이 되는 것을 막는다 (힌지가 0 을 곱하지만 autograd 는 계산한다).
    """
    tol = tol or Tolerances()
    f_on_target = resample(t_sim, f_sim, target.t)
    w_on_target = resample(t_sim, omega, target.t)
    scale = max(target.f_max, 1e-9) ** 2

    num = (w_on_target * (f_on_target - target.f) ** 2).sum()
    den = torch.clamp(w_on_target.sum(), min=1.0)
    e_shape = torch.sqrt(num / den / scale + 1e-12)

    i_sim = torch.trapezoid(f_sim, t_sim)
    e_imp = torch.abs(i_sim - target.total_impulse) / max(target.total_impulse, 1e-9)
    e_fmax = torch.abs(f_sim.max() - target.f_max) / max(target.f_max, 1e-9)
    l_pmax = (torch.zeros((), dtype=t_sim.dtype) if target.p_max is None
              else ((p_sim.max() - target.p_max) / max(target.p_max, 1e-9)) ** 2)

    terms = {"shape": weights.shape * _hinge(e_shape, tol.shape),
             "impulse": weights.impulse * _hinge(e_imp, tol.impulse),
             "f_max": weights.f_max * _hinge(e_fmax, tol.f_max)}
    g = lambda e, t: max(0.0, float(e.detach()) * 100.0 - 100.0 * t)
    report = {"I_sim": float(i_sim.detach()), "F_max_sim": float(f_sim.max().detach()),
              "P_max_sim": float(p_sim.max().detach()),
              "rms_shape_pct": float(e_shape.detach()) * 100.0,
              "I_err_pct": float(e_imp.detach()) * 100.0,
              "F_err_pct": float(e_fmax.detach()) * 100.0,
              "P_err_pct": float(torch.sqrt(l_pmax).detach()) * 100.0,
              # 집합까지 남은 거리 [%p] — 0 이면 그 사양은 집합 안이다
              "gap_shape_pp": g(e_shape, tol.shape),
              "gap_impulse_pp": g(e_imp, tol.impulse),
              "gap_f_max_pp": g(e_fmax, tol.f_max),
              "pressure_is_spec": bool(target.pressure_is_spec)}

    # ── 압력 — 곡선이 들어오면 사양, 아니면 소프트 2차 기준
    if target.pressure_is_spec:
        p_on_target = resample(t_sim, p_sim, target.t)
        p_scale = max(target.p_max, 1e-9) ** 2
        num_p = (w_on_target * (p_on_target - target.p) ** 2).sum()
        e_pshape = torch.sqrt(num_p / den / p_scale + 1e-12)
        e_ppeak = torch.sqrt(l_pmax + 1e-24)          # = |ΔP_max| / P_max
        terms["p_shape"] = weights.p_shape * _hinge(e_pshape, tol.p_shape)
        terms["p_peak"] = weights.p_peak * _hinge(e_ppeak, tol.p_max)
        report["P_shape_pct"] = float(e_pshape.detach()) * 100.0
        report["gap_p_shape_pp"] = g(e_pshape, tol.p_shape)
        report["gap_p_max_pp"] = g(e_ppeak, tol.p_max)
    else:
        terms["p_max"] = weights.p_max * l_pmax       # 2차 기준 — 힌지 아님
    report["tol_pct"] = dict(tol.as_pct(with_pressure=target.pressure_is_spec))
    return sum(terms.values()), terms, report


def group_l1(a, b, r_outer: float, delta: float = 1e-3):
    """모드 진폭의 합 Σ_k √(a_k² + b_k²) — 회전 불변 희소성 벌점.

    ⚠ **반환값은 `r_outer` 로 나눠 무차원화된다**. `runner` 의
      `rep["complexity"]` 는 **나누지 않은** mm 값이다 — 두 양이 이름이 비슷해서
      섞기 쉽다. 진폭 바닥(`amplitude_floor`)과 비교하는 것은 `complexity` 쪽
      (mm)이고, 손실에 더해지는 것은 이 무차원 값이다.

    **왜 이것인가.** 다중 출발 측정(8개 출발점, 전부 잔차 0.24 % 이내)에서

        Σ|c_k| vs 참 형상과의 거리 : 상관 +0.895
        잔차    vs 참 형상과의 거리 : 상관 +0.725

    즉 **곡선을 더 잘 맞춘다고 참 형상에 가까운 것이 아니고**, "모드 진폭 합이
    작다" 가 훨씬 좋은 예측자였다. 가장 단순한 해(Σ|c_k| = 1.98)가 목표
    (Σ|c_k| = 2.20)에 가장 가까웠다.

    제작 쪽 근거도 같은 방향이다 — 모드가 적을수록 몰드가 단순하고 얇은 특징이
    적다.

    ⚠ **세게 걸면 안 된다.** 원형(a = b = 0)은 안장점이라, 벌점이 강하면 곡선
       손실을 이기고 거기로 붕괴한다. 그래서 `sparsity_warmup` 동안은 0 으로
       두고 곡선이 먼저 맞은 뒤에 선형으로 올린다.

    √ 안에 δ² 을 더해 원점에서 미분 가능하게 만든다.
    """
    c = torch.sqrt(a ** 2 + b ** 2 + delta ** 2)
    return c.sum() / float(r_outer)


def sparsity_ramp(it: int, iters: int, weights: LossWeights) -> float:
    """희소성 가중치의 연속 증가 일정 (0 → weights.sparsity)."""
    w0 = weights.sparsity_warmup * max(iters, 1)
    if it <= w0:
        return 0.0
    return weights.sparsity * min(1.0, (it - w0) / max(iters - w0, 1.0))


def hard_pressure_penalty(p_sim, p_allow: float):
    """케이스 허용압 초과 — 넘으면 안 되는 하드 제약 (§4.5)."""
    return (torch.clamp(p_sim.max() - p_allow, min=0.0) / p_allow) ** 2


def mass_penalty(mass_g, mass_max: float):
    """추진제 질량 상한 (§4.6.1, 1회 시험 장약)."""
    m = torch.as_tensor(mass_g)
    return (torch.clamp(m - mass_max, min=0.0) / mass_max) ** 2
