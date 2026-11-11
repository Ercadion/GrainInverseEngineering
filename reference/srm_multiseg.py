#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
srm_multiseg.py
══════════════════════════════════════════════════════════════════════════
다중 세그먼트 그레인 시뮬레이션 엔진  v1.0
──────────────────────────────────────────────────────────────────────────
기존 SRM_simul.simulate() 의 두 가지 한계를 확장한다.

  1) 끝면 연소 유효성 η(gap)  — 세그먼트 간격이 좁으면 끝면(상·하 단면)이
     실질적으로 연소에 기여하지 못한다 (화염 소염 + 유동 질식).
     간격 g 에 대한 유효성 계수:

        η(g) = 0                    (g ≤ g_min : 소염/유동 질식 — 억제면 취급)
        η(g) = (g−g_min)/(g_full−g_min)   (선형 천이)
        η(g) = 1                    (g ≥ g_full : 완전 연소면)

     기본값 g_min = 0.5 mm (고압 소염거리 스케일), g_full = 2.0 mm
     (아마추어 BATES 스페이서 관행 ~1/16–1/8", Nakka 설계 관행).
     간격은 끝면이 후퇴하는 만큼 시간에 따라 넓어지며(g_j += Σ인접면 후퇴),
     η 도 매 스텝 재평가된다. 세그먼트가 소진되면 인접 간격이 병합된다.

  2) 세그먼트 개별 정의 — 각 세그먼트가 서로 다른 길이 L_i, 코어 직경
     D_gc_i, 또는 임의 포트 형상(SDF 식)을 가질 수 있다.
     · 원형 코어 세그먼트: 해석식 (A_port = π/4·D², P = π·D)
     · 임의 포트 세그먼트: 세그먼트별 독립 SDF 격자 (SRM_simul 의
       area_from_F / perimeter_from_F / rebuild_SDF_from_mask 재사용)
     세그먼트별로 독립적으로 번아웃되며(코어가 외벽 도달 또는 길이 0),
     남은 세그먼트만으로 시뮬레이션이 계속된다.

수치 공식·계산 순서는 SRM_simul.simulate() 와 동일 계열이다:
  후퇴 → 기하 갱신 → (점화 램프) → 연소면적/Kn → 질량수지 → 챔버 압력
  (암시적 backward-Euler 폐형해) → R_dot → 추력/Cf/Isp.
R_dot 은 srm.R_dot_calculation 모듈 참조로 호출하므로 구간별(piecewise)
연소속도 monkey-patch 도 그대로 적용된다.

총 연소면적:
  A_b = Σ_i [ P_i·L_i ]  +  Σ_faces [ η(g_face) · (π/4·D_g² − A_port_i) ]
길이 후퇴:
  dL_i/dt = −ṙ·[ η(g_top,i) + η(g_bot,i) ]
질량 생성 (SRM_simul 의 체적차분 방식):
  ṁ_GEN = ρ_p · (V(t) − V(t+Δt)) / Δt ,   V = Σ_i (π/4·D_g² − A_port_i)·L_i

반환은 SRM_simul.simulate() 와 동일한 21-튜플 (스냅샷/격자 더미 포함).
상세 이론·출처는 동봉 docx 문서 참조.
══════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import SRM_simul as srm  # noqa: E402

PI = math.pi


# ══════════════════════════════════════════════════════════════════════════
#  끝면 연소 유효성 모델
# ══════════════════════════════════════════════════════════════════════════

def erosive_ratio(G, r0_mms, D_h_mm, L_ref_mm, rho_gcc,
                  alpha=0.023, beta=53.0):
    """
    Lenoir–Robillard 침식연소 — 침식 성분 r_e [mm/s].

      r = r0 + r_e ,
      r_e = α · G^0.8 / D_h^0.2 · exp( −β · ρ_p · r_tot / G )

    G      : 포트 질량유속 [kg/(m²·s)]
    r0_mms : 압력 기반 기본 후퇴율 a·Pⁿ [mm/s]
    D_h_mm : 포트 수력직경 [mm] (원형 코어면 D_gc)
    rho_gcc: 추진제 밀도 [g/cc]
    α, β   : 추진제별 실험 계수 (기본값은 KN계열 대표 추정치)

    지수항이 r_tot 에 의존하므로 고정점 반복(≤12회)으로 수렴시킨다.
    G 가 작으면 지수항이 0 에 수렴하여 자동으로 r_e → 0 (임계 유속 효과).
    ⚠ α, β 는 추진제·형상별 실험값이며 기본값은 문헌 범위의 대표값이다.
    """
    if G <= 1e-9 or r0_mms <= 0:
        return 0.0
    rho_si = rho_gcc * 1000.0                       # kg/m³
    D_h_m = max(D_h_mm, 1e-6) / 1000.0
    pre = alpha * (G ** 0.8) / (D_h_m ** 0.2)       # [kg/(m²·s)] 스케일
    r_tot = r0_mms
    for _ in range(12):
        r_si = r_tot / 1000.0                       # m/s
        expo = math.exp(-beta * rho_si * r_si / G)
        r_e = pre * expo / rho_si * 1000.0          # → mm/s
        new = r0_mms + r_e
        if abs(new - r_tot) < 1e-6:
            r_tot = new
            break
        r_tot = 0.5 * r_tot + 0.5 * new             # 감쇠(안정성)
    return max(r_tot - r0_mms, 0.0)


def face_effectiveness(gap_mm: float, gap_min: float = 0.5,
                       gap_full: float = 2.0) -> float:
    """
    간격 g 에서의 끝면 연소 유효성 η ∈ [0, 1].
      g ≤ g_min  → 0 (소염거리 이하 / 유동 질식 — 사실상 억제면)
      g ≥ g_full → 1 (완전 노출면)
      사이 구간  → 선형 천이
    """
    if gap_mm <= gap_min:
        return 0.0
    if gap_mm >= gap_full:
        return 1.0
    return (gap_mm - gap_min) / (gap_full - gap_min)


# ══════════════════════════════════════════════════════════════════════════
#  세그먼트 상태
# ══════════════════════════════════════════════════════════════════════════

class _Seg:
    """세그먼트 1개의 기하 상태 (원형 코어=해석식 / 임의 포트=SDF 격자)"""

    def __init__(self, L_mm: float, D_g: float, D_gc: float = None,
                 port_expr: str = None, grid: int = 600):
        self.L = float(L_mm)
        self.D_g = float(D_g)
        self.alive = True
        self.circular = port_expr is None
        if self.circular:
            self.D_gc = float(D_gc)
        else:
            half = D_g / 2.0
            xs = np.linspace(-half, half, grid)
            self.dx = xs[1] - xs[0]
            X, Y = np.meshgrid(xs, xs)
            self.grain_mask = (srm.circle(X, Y, 0.0, 0.0, half) <= 0)
            self.F = np.where(self.grain_mask,
                              srm.eval_expr(port_expr, X, Y), 1e9)
            self.x0 = -half
            self._rebuild_ctr = 0
            self._dr_accum = 0.0
        # 슬리버 컷오프 기준 (초기 환형 면적)
        self.annulus_0 = max(self.annulus(), 1e-9)

    # ── 포트 단면적 / 둘레 ────────────────────────────────────────────
    def port_area(self) -> float:
        if not self.alive:
            return PI / 4.0 * self.D_g ** 2
        if self.circular:
            return PI / 4.0 * self.D_gc ** 2
        return srm.area_from_F(self.F, self.x0, self.x0,
                               self.dx, self.dx, 1.0)

    def port_perimeter(self) -> float:
        if not self.alive:
            return 0.0
        if self.circular:
            return PI * self.D_gc
        return srm.perimeter_from_F(self.F, self.x0, self.x0,
                                    self.dx, self.dx, 1.0)

    def annulus(self) -> float:
        """끝면 환형 면적 [mm²]"""
        if not self.alive:
            return 0.0
        return max(PI / 4.0 * self.D_g ** 2 - self.port_area(), 0.0)

    def volume(self) -> float:
        if not self.alive:
            return 0.0
        return self.annulus() * self.L

    # ── 후퇴 ──────────────────────────────────────────────────────────
    def advance(self, dr_mm: float, dL_top: float, dL_bot: float):
        """반경 방향 dr, 길이 방향 (상/하 면별) 후퇴 적용"""
        if not self.alive:
            return
        self.L -= (dL_top + dL_bot)
        if self.circular:
            self.D_gc += 2.0 * dr_mm
        else:
            # 외벽 비연소 불변식(SRM_simul.simulate 주석 참조): grain_mask 는
            # 불변이고 F 는 생성 시(__init__)에 이미 클램프되어 있으므로,
            # 재생성 직후에만 다시 고정하면 된다. 매 스텝 전체 격자
            # np.where 는 중복이라 제거하고 인플레이스 감산으로 바꿨다.
            self.F -= dr_mm
            self._dr_accum += dr_mm
            self._rebuild_ctr += 1
            if self._rebuild_ctr >= 45 or self._dr_accum >= self.dx * 3.0:
                mask = (self.F <= 0)
                self.F = srm.rebuild_SDF_from_mask(mask, self.dx, self.dx)
                self.F[~self.grain_mask] = srm.OUTSIDE_SDF
                self._rebuild_ctr = 0
                self._dr_accum = 0.0

    def check_burnout(self, sliver_frac: float = 0.005) -> bool:
        """
        번아웃 판정 → alive 갱신, 소진 시 True 반환.
        임의 포트는 잔여 환형 면적이 초기값의 sliver_frac (기본 0.5%) 이하가
        되면 슬리버로 보고 소진 처리한다 (SDF 등고선 면적이 정확히 0 에
        수렴하지 않아 비물리적 저추력 꼬리가 남는 것을 방지).
        """
        if not self.alive:
            return False
        if self.L <= 0.0:
            self.alive = False
            self.L = 0.0
            return True
        if self.circular:
            if self.D_gc >= self.D_g:
                self.alive = False
                return True
        else:
            if self.annulus() <= sliver_frac * self.annulus_0:
                self.alive = False
                return True
        return False


def _build_segments(segments_def: list, D_g: float, grid: int) -> list:
    """
    segments_def: [{"L_gs_mm":.., "D_gc_0_mm":.. | "port_expr":".."}, ...]
    """
    segs = []
    for sd in segments_def:
        expr = (sd.get("port_expr") or "").strip() or None
        segs.append(_Seg(float(sd["L_gs_mm"]), D_g,
                         D_gc=float(sd.get("D_gc_0_mm") or 0) or None,
                         port_expr=expr, grid=grid))
        if expr is None and not sd.get("D_gc_0_mm"):
            raise ValueError("세그먼트에 D_gc_0_mm 또는 port_expr 가 필요합니다")
    return segs


# ══════════════════════════════════════════════════════════════════════════
#  간격(η) 계산 헬퍼 — 세그먼트 소진 시 간격 병합 포함
# ══════════════════════════════════════════════════════════════════════════

def _face_gaps(segs: list, gaps: list) -> list:
    """
    각 살아있는 세그먼트의 (유효 상부 간격, 유효 하부 간격) 목록 계산.
    gaps: 경계 간격 리스트, 길이 N+1 (0=벌크헤드 쪽, N=노즐 쪽).
    소진된 세그먼트(L=0)는 그 위아래 간격이 이어진 것으로 취급(병합).
    반환: [(i, g_top_eff, g_bot_eff), ...]  (alive 세그먼트만)
    """
    N = len(segs)
    out = []
    for i, s in enumerate(segs):
        if not s.alive:
            continue
        # 위쪽: 인접 간격 + 소진 세그먼트(L=0)를 계속 합산
        g_top = gaps[i]
        j = i - 1
        while j >= 0 and not segs[j].alive:
            g_top += segs[j].L + gaps[j]
            j -= 1
        # 아래쪽
        g_bot = gaps[i + 1]
        j = i + 1
        while j < N and not segs[j].alive:
            g_bot += segs[j].L + gaps[j + 1]
            j += 1
        out.append((i, g_top, g_bot))
    return out


# ══════════════════════════════════════════════════════════════════════════
#  총 연소면적 (노즐 설계용 정적 스윕 포함)
# ══════════════════════════════════════════════════════════════════════════

# ── 슬리버 테이퍼 (Kn 표시 전용) ──────────────────────────────────────
#
# 문제: 세그먼트가 소진되는 순간 그 코어 면적(π·D_gc·L)이 통째로 사라져
#   Kn 곡선에 수직 계단이 생긴다. 3 세그먼트면 계단 3 개. 침식연소를 켜면
#   세그먼트별 후퇴율이 갈라져 소진 시각이 어긋나므로 계단이 더 두드러진다
#   (끄면 셋이 동시에 죽어 계단 1 개로 합쳐진다).
#
# 실제 모터: 웹 두께가 완벽히 균일하지 않아(제작 공차·케이스 접착·열구배)
#   마지막 웹은 한 순간에 없어지지 않고 수십 ms 에 걸쳐 슬리버로 흩어지며
#   꺼진다. 따라서 잔여 환형 면적이 초기의 KN_TAPER_Q 이하로 내려가면
#   코어 면적에 선형 감쇠를 걸어 표시한다.
#
# ⚠ 이것은 **표시 모델**이다. 챔버 압력은 체적차분(질량수지)으로 풀고
#   Kn 은 기록만 하므로, 이 테이퍼는 P(t)·F(t)·m_g(t)·연소시간에 전혀
#   영향을 주지 않는다. 그 보장을 구조로 강제하기 위해 taper_q 는 기본
#   0.0(끔)이고, simulate_multiseg 의 **보고용 호출**에서만 켠다.
#   (기하 후퇴를 구동하는 etas 계산 호출과 노즐 설계 스윕은 기본값 사용)
#
# ⚠ 부작용 — Kn 과 P 의 탈동조:
#   테이퍼를 켜면 Kn 은 부드럽게 내려가는데 P(t) 는 모델이 세그먼트를
#   죽이는 순간 여전히 꺾인다. 즉 그래프에서 Kn 이 P 보다 먼저 떨어지는
#   구간이 생긴다(이 예에서 약 76 ms). Kn 과 P 의 대응을 그대로 보고
#   싶으면 kn_taper_q=0 으로 끄면 된다.
#   기본값 0.02 는 수직 낙차를 91% 줄이면서(ΔKn −87 → −7.9) 탈동조를
#   가장 짧게 하는 지점으로 골랐다. Q=0.05 는 낙차가 −6.2 로 조금 더
#   낮아지지만 탈동조가 110 ms 로 길어져 이득이 없다.
KN_TAPER_Q = 0.02      # 잔여 환형 면적비 문턱. 0 이면 테이퍼 없음(구 동작)


def _sliver_factor(seg, taper_q: float) -> float:
    """소진 근처 코어면적 감쇠 계수 (1 = 그대로, 0 = 없음)"""
    if taper_q <= 0.0:
        return 1.0
    q = seg.annulus() / max(seg.annulus_0, 1e-12)
    return min(1.0, max(0.0, q / taper_q))


def _total_burn_area(segs, gaps, gap_min, gap_full, taper_q: float = 0.0):
    """A_bc(코어) + Σ η·끝면 [mm²], 그리고 면별 η 목록

    taper_q > 0 이면 코어 면적에 슬리버 테이퍼를 적용한다(표시 전용).
    끝면(A_be)은 환형 면적 자체가 0 으로 수렴하므로 테이퍼가 필요 없다.
    """
    A_bc = sum(s.port_perimeter() * s.L * _sliver_factor(s, taper_q)
               for s in segs if s.alive)
    A_be = 0.0
    etas = []
    for i, g_top, g_bot in _face_gaps(segs, gaps):
        ann = segs[i].annulus()
        e_t = face_effectiveness(g_top, gap_min, gap_full)
        e_b = face_effectiveness(g_bot, gap_min, gap_full)
        A_be += ann * (e_t + e_b)
        etas.append((i, e_t, e_b))
    return A_bc, A_be, etas


def design_nozzle_multiseg(segments_def, D_g, gap0_mm, Kn_max,
                           t_interval=0.02, r_test=0.49,
                           gap_min=0.5, gap_full=2.0, grid=400) -> float:
    """
    simulate_NOZ_t 의 다중 세그먼트 판:
    r_test 로 기하를 후퇴시키며 최대 연소면적 탐색 → A_t = A_bg_max / Kn_max
    """
    segs = _build_segments(segments_def, D_g, grid)
    gaps = list(gap0_mm) if isinstance(gap0_mm, (list, tuple)) \
        else [float(gap0_mm)] * (len(segs) + 1)
    A_max = 0.0
    for _ in range(100000):
        A_bc, A_be, etas = _total_burn_area(segs, gaps, gap_min, gap_full)
        A_max = max(A_max, A_bc + A_be)
        dr = r_test * t_interval
        eta_map = {i: (e_t, e_b) for i, e_t, e_b in etas}
        for i, s in enumerate(segs):
            if not s.alive:
                continue
            e_t, e_b = eta_map.get(i, (0.0, 0.0))
            dl_t, dl_b = dr * e_t, dr * e_b
            s.advance(dr, dl_t, dl_b)
            gaps[i] += dl_t
            gaps[i + 1] += dl_b
        for s in segs:
            s.check_burnout()
        if not any(s.alive for s in segs):
            break
    A_NOZ_t = A_max / Kn_max
    D_throat = (4.0 * A_NOZ_t / PI) ** 0.5
    if D_throat >= D_g:
        print(f"⚠️  경고: 노즐목 지름 {D_throat:.2f} mm ≥ 그레인 지름 "
              f"{D_g:.2f} mm — 형상/Kn 재검토 필요.")
    return A_NOZ_t


# ══════════════════════════════════════════════════════════════════════════
#  메인 시뮬레이션 루프
# ══════════════════════════════════════════════════════════════════════════

def simulate_multiseg(
    segments_def,           # [{"L_gs_mm","D_gc_0_mm"|"port_expr"}, ...]
    gap0_mm,                # 초기 간격 (스칼라=전 경계 동일 | 길이 N+1 리스트)
    V_c,                    # 챔버 부피 [mm³]
    D_g,                    # 그레인 외경 [mm]
    den_actual,             # 실제 밀도 [g/cc]
    A_NOZ_t,                # 노즐목 면적 [mm²]
    NOZ_eff, R_specific, gamma, C_star, T_actual,
    P_target, P_atm, e_NOZ,
    a, n, t_end, t_interval,
    grid=600, ER=8.0, t_ignition=0.0,
    gap_min=0.5, gap_full=2.0,
    erosive=False, ero_alpha=0.023, ero_beta=53.0,
    kn_taper_q=KN_TAPER_Q,     # Kn 표시용 슬리버 테이퍼 (0 = 끔)
    verbose=True,
):
    segs = _build_segments(segments_def, D_g, grid)
    N = len(segs)
    gaps = list(gap0_mm) if isinstance(gap0_mm, (list, tuple)) \
        else [float(gap0_mm)] * (N + 1)
    if len(gaps) != N + 1:
        raise ValueError(f"간격 리스트 길이는 N+1={N+1} 이어야 합니다")

    if verbose:
        kinds = ["원형" if s.circular else "SDF" for s in segs]
        print(f"[다중 세그먼트] N={N} ({', '.join(kinds)}), "
              f"간격 {gaps[0]:g}mm, η모델 g_min={gap_min} g_full={gap_full}")
        if kn_taper_q > 0:
            _ann0 = sum(s.annulus_0 for s in segs) / max(len(segs), 1)
            _web = (D_g - math.sqrt(max(D_g ** 2
                                        - kn_taper_q * 4.0 * _ann0 / PI,
                                        0.0))) / 2.0
            print(f"  Kn 표시: 슬리버 테이퍼 ON (q<{kn_taper_q:g}, "
                  f"잔여 웹 ≲{_web:.2f}mm 구간) — P·F·질량에는 영향 없음")
        print(f"  순서: #1=벌크헤드 쪽 → #{N}=노즐 쪽" +
              (f" | 침식연소 ON (α={ero_alpha}, β={ero_beta})"
               if erosive else " | 침식연소 OFF"))
        _, _, etas0 = _total_burn_area(segs, gaps, gap_min, gap_full)
        for i, e_t, e_b in etas0:
            s = segs[i]
            port = f"D_gc={s.D_gc:g}mm" if s.circular else "SDF 포트"
            print(f"  seg{i+1}: L={s.L:g}mm {port}  "
                  f"η_top={e_t:.2f} η_bot={e_b:.2f}")

    # 결과 리스트 (SRM_simul.simulate 와 동일 구성)
    t_list = []
    D_gc_list, L_g_list = [], []
    A_be_list, A_bc_list, A_bg_list = [], [], []
    m_g_list, m_dot_GEN_list, m_dot_NOZ_list, m_dot_REMAIN_list = [], [], [], []
    den_REMAIN_list, P_c_list, R_dot_list = [], [], []
    Kn_list, F_list, Cf_list, Isp_list = [], [], [], []

    t = 0.0
    P_c = P_atm
    R_dot = srm.R_dot_calculation(a, n, P_c)
    m_REMAIN = 0.0
    m_g = sum(s.volume() for s in segs) * den_actual * 1e-6   # kg
    m_g0 = m_g if m_g > 0 else None    # 노즐목 삭마 진행률 기준
    # 감압 종료 감지: 추력이 발생한 뒤 챔버가 대기압으로 회복되면 종료.
    # (형상이 불균일하면 일부 세그먼트만 남아 대기압에서 극히 느리게
    #  연소하며 추력 0 인 구간이 길게 이어지는데, 실제 모터는 그 시점에
    #  작동이 끝난 것이므로 시뮬레이션도 종료한다.)
    had_thrust = False
    depress_time = 0.0
    DEPRESS_HOLD_S = 0.05
    ero_peak = 1.0          # 최대 침식 배율 (진단용)

    while t <= t_end + 1e-12:
        # 노즐목 삭마 (SRM_simul.simulate 와 동일 정의: 소비 질량분율)
        A_t_now = srm.throat_area_eroded(
            A_NOZ_t, e_NOZ, 0.0 if m_g0 is None else 1.0 - m_g / m_g0)

        # ── 기하 후퇴 (체적차분으로 ṁ_GEN 산출 — SRM_simul 방식) ──────
        V_before = sum(s.volume() for s in segs)
        _, _, etas = _total_burn_area(segs, gaps, gap_min, gap_full)  # 테이퍼 없음
        eta_map = {i: (e_t, e_b) for i, e_t, e_b in etas}

        # 침식연소: 벌크헤드(i=0) → 노즐(i=N−1) 방향으로 질량유속 누적.
        # 세그먼트 i 의 포트를 통과하는 유량 = 자신 포함 상류 전체의 생성량
        # (각 세그먼트 하류 끝 기준). G_i = ṁ_cum,i / A_port,i.
        # → 노즐에 가까울수록 G 가 커져 후퇴율이 빨라진다 (배치 의존성).
        r_seg = {}
        if erosive:
            m_cum = 0.0                      # kg/s
            for i, s in enumerate(segs):
                if not s.alive:
                    continue
                e_t, e_b = eta_map.get(i, (0.0, 0.0))
                A_b_i = (s.port_perimeter() * s.L
                         + s.annulus() * (e_t + e_b))          # mm²
                # 이 세그먼트가 만드는 유량 (직전 스텝 후퇴율 기준)
                m_cum += A_b_i * 1e-6 * (R_dot * 1e-3) * (den_actual * 1000.0)
                A_port_m2 = max(s.port_area(), 1e-9) * 1e-6    # mm²→m²
                G_i = m_cum / A_port_m2                        # kg/(m²·s)
                D_h = (4.0 * s.port_area() / max(s.port_perimeter(), 1e-9)) \
                    if not s.circular else s.D_gc
                r_seg[i] = R_dot + erosive_ratio(
                    G_i, R_dot, D_h, s.L, den_actual,
                    alpha=ero_alpha, beta=ero_beta)

        for i, s in enumerate(segs):
            if not s.alive:
                continue
            e_t, e_b = eta_map.get(i, (0.0, 0.0))
            dr = r_seg.get(i, R_dot) * t_interval
            dl_t, dl_b = dr * e_t, dr * e_b
            s.advance(dr, dl_t, dl_b)
            gaps[i] += dl_t
            gaps[i + 1] += dl_b
        for s in segs:
            s.check_burnout()
        V_after = sum(s.volume() for s in segs)

        m_dot_GEN = max(V_before - V_after, 0.0) * den_actual * 1e-6 \
            / t_interval

        # 점화 램프 (simulate() 와 동일)
        if t_ignition > 0.0 and t < t_ignition:
            m_dot_GEN *= t / t_ignition

        # ── 연소면적 / Kn (갱신된 기하 기준 — simulate() 와 동일 순서) ─
        # ⚠ 여기서만 테이퍼를 켠다. 위의 etas 계산 호출은 기본값(0.0)이라
        #   기하 후퇴·질량수지는 영향을 받지 않는다.
        A_bc, A_be, _ = _total_burn_area(segs, gaps, gap_min, gap_full,
                                         taper_q=kn_taper_q)
        A_bg = A_bc + A_be
        Kn = srm.Kn_calculation(A_bg, A_t_now)

        alive_D = [s.D_gc for s in segs if s.alive and s.circular]
        D_gc_list.append(float(np.mean(alive_D)) if alive_D else D_g)
        L_g_list.append(sum(s.L for s in segs if s.alive))
        A_be_list.append(A_be)
        A_bc_list.append(A_bc)
        A_bg_list.append(A_bg)
        Kn_list.append(Kn)

        m_g -= m_dot_GEN * t_interval
        m_g_list.append(m_g)

        # ── 챔버 질량수지 & 압력 (암시적 backward-Euler — 동일 공식) ──
        V_free_mm3 = V_c - V_after
        if V_free_mm3 <= 0.0:
            if verbose:
                print("형상 비현실: 자유체적 <= 0 → 종료")
            break
        C1 = 1.0e3 * R_specific * T_actual / V_free_mm3
        At_term = A_t_now / C_star
        P_c = (C1 * (m_REMAIN + m_dot_GEN * t_interval) + P_atm) \
            / (1.0 + C1 * At_term * t_interval)

        m_dot_NOZ = srm.m_dot_NOZ_calculation(P_c, A_t_now, C_star)
        m_dot_REMAIN = m_dot_GEN - m_dot_NOZ
        m_REMAIN += m_dot_REMAIN * t_interval
        if m_REMAIN < 0.0:
            m_REMAIN = 0.0
            P_c = P_atm
        den_REMAIN = srm.Den_REMAIN_calculation(V_c, V_after, m_REMAIN)

        m_dot_GEN_list.append(m_dot_GEN)
        m_dot_NOZ_list.append(m_dot_NOZ)
        m_dot_REMAIN_list.append(m_dot_REMAIN)
        den_REMAIN_list.append(den_REMAIN)
        P_c_list.append(P_c)

        R_dot = srm.R_dot_calculation(a, n, P_c)
        R_dot_list.append(R_dot)
        if erosive and r_seg:
            ero_peak = max(ero_peak,
                           max(r_seg.values()) / max(R_dot_list[0], 1e-9))

        if P_c > P_atm:
            Cf = srm.Cf_calculation(gamma, P_c, P_atm, ER)
            thrust = srm.F_calculation(Cf, P_c, A_t_now, NOZ_eff)
            Isp = srm.Isp_calculation(
                thrust, srm.m_dot_NOZ_calculation(P_c, A_t_now, C_star))
        else:
            Cf, thrust, Isp = 0.0, 0.0, 0.0
        F_list.append(thrust)
        Cf_list.append(Cf)
        Isp_list.append(Isp)

        t_list.append(t)

        if thrust > 0.0:
            had_thrust = True
            depress_time = 0.0
        elif had_thrust:
            depress_time += t_interval
            if depress_time >= DEPRESS_HOLD_S:
                if verbose:
                    print(f"챔버 감압 완료(추력 소멸): t={t:.3f}s "
                          f"— 잔여 슬리버 연소는 무시")
                break

        if not any(s.alive for s in segs):
            if verbose:
                print(f"전 세그먼트 소진: t={t:.3f}s")
            break

        t += t_interval

    if verbose and erosive:
        print(f"  침식연소 최대 배율: r_max/r0 = {ero_peak:.3f} "
              f"(노즐 쪽 세그먼트 기준)")
    if verbose and e_NOZ and e_NOZ > 0.0:
        D0 = (4.0 * A_NOZ_t / np.pi) ** 0.5
        A_end = srm.throat_area_eroded(
            A_NOZ_t, e_NOZ, 0.0 if m_g0 is None else 1.0 - m_g / m_g0)
        print(f"  노즐목 삭마 반영: d_t {D0:.2f} → "
              f"{(4.0*A_end/np.pi)**0.5:.2f} mm "
              f"(면적 +{(A_end/A_NOZ_t-1)*100:.1f}%)")

    dummy = np.zeros((2, 2))
    return (
        np.array(t_list),
        np.array(D_gc_list), np.array(L_g_list),
        np.array(A_be_list), np.array(A_bc_list), np.array(A_bg_list),
        np.array(m_g_list), np.array(m_dot_GEN_list),
        np.array(m_dot_NOZ_list), np.array(m_dot_REMAIN_list),
        np.array(den_REMAIN_list), np.array(P_c_list), np.array(R_dot_list),
        np.array(Kn_list),
        np.array(F_list), np.array(Cf_list), np.array(Isp_list),
        [], dummy, dummy, dummy,
    )


# ══════════════════════════════════════════════════════════════════════════
#  상위 헬퍼 — 그레인 설정 dict 로부터 전체 실행 (builder/newgrains 공용)
# ══════════════════════════════════════════════════════════════════════════

def prepare_multiseg(g: dict) -> tuple:
    """
    g(그레인 파라미터 dict)에서 다중 세그먼트 정의·간격을 구성.
    segments_custom 이 없으면 N개의 동일 세그먼트를 생성.
    반환: (segments_def, gaps, warn_list)
    """
    warns = []
    segs = g.get("segments_custom")
    if not segs:
        N = int(g["N_segments"])
        segs = [{"L_gs_mm": float(g["L_gs_mm"]),
                 "D_gc_0_mm": float(g["D_gc_0_mm"])} for _ in range(N)]
    N = len(segs)
    gap = g.get("seg_gap_mm")
    gap = float(gap) if gap not in (None, "") else 2.0
    L_c = float(g.get("L_c_mm") or 0)
    stack = sum(float(s["L_gs_mm"]) for s in segs) + (N + 1) * gap
    if L_c and stack > L_c + 1e-6:
        warns.append(f"⚠ 그레인 전장 {stack:.0f}mm (간격 {N+1}×{gap:g}mm "
                     f"포함) 가 챔버 길이 L_c {L_c:.0f}mm 를 초과합니다.")
    return segs, gap, warns
