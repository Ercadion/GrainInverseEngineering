#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
srm_bates_fast.py
══════════════════════════════════════════════════════════════════════════
BATES(원형 코어) 전용 해석적 고속 시뮬레이션 경로  v1.0
──────────────────────────────────────────────────────────────────────────
배경:
  SRM_simul.simulate() 는 임의 포트 형상을 다루기 위해 SDF(부호거리장)
  격자(기본 1500² ≈ 225만 셀)를 매 시간스텝 갱신한다. 그러나 코드 분석
  결과, BATES 원형 코어의 경우 연소 후퇴·연소면적·질량수지·압력·추력은
  전부 해석식(Update_grain, A_be/A_bc/A_bg_calculation, 암시적 압력해 등)
  으로 계산되며, SDF 격자는 물리 결과에 기여하지 않는다
  (격자는 A_port<=0 종료판정·초기 D_port 출력·스냅샷 시각화에만 사용).

  → 본 모듈은 simulate() 의 물리 루프를 "격자 없이" 그대로 재현한다.
    물리 수식·계산 순서·시간전진(암시적 backward-Euler 압력해)·점화 램프·
    종료 조건이 simulate() 와 1:1 동일하므로, 원형 코어에서는 격자 경로와
    부동소수점 수준으로 같은 결과를 수십~수백 배 빠르게 얻는다.

  ※ SRM_simul.py 의 함수를 그대로 호출한다 (수치 공식 무수정).
    특히 R_dot 은 srm.R_dot_calculation 을 모듈 참조로 호출하므로,
    TMS_simulator_newgrains.burn_rate_patch() 의 구간별(piecewise)
    연소속도 monkey-patch 도 그대로 적용된다.

  ※ 임의 포트(star/box/union 등)는 본 경로를 쓸 수 없다 — 이 경우
    기존 SRM_simul.simulate() (격자 경로)를 사용해야 한다.

반환 형식은 SRM_simul.simulate() 와 동일한 21-튜플이다.
(snapshot_list 는 빈 리스트, X/Y/grain_sdf 는 최소 더미 배열)
══════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import SRM_simul as srm  # noqa: E402


def simulate_bates(
    V_c,                    # 챔버 부피 [mm^3]
    D_g, D_gc,              # 그레인 직경, 코어 직경 [mm]
    N,                      # 세그먼트 개수
    L_g,                    # 그레인 길이 [mm]
    V_g,                    # 그레인 부피 [mm^3]
    den_actual,             # 그레인 밀도 [g/cc]
    m_g,                    # 그레인 질량 [kg]
    A_NOZ_t,                # 노즐목 면적 [mm^2]
    NOZ_eff,                # 노즐 효율 (0~1)
    R_specific,             # 기체상수 [J/(kg*K)]
    gamma,                  # 비열비
    C_star,                 # 특성 속도 [m/s]
    T_actual,               # 연소실 온도 [K]
    P_target,               # 목표 압력 [MPa]
    P_atm,                  # 대기압 [MPa]
    e_NOZ,                  # 노즐 삭마 정도 [mm] (simulate 와 동일하게 반영)
    a, n,                   # 후퇴율 상수 (monkey-patch 시 무시될 수 있음)
    t_end, t_interval,      # 종료 시간 / 시간 간격 [s]
    t_ignition=0.0,         # 점화 램프 시간 [s]
    verbose=True,
):
    """SRM_simul.simulate() 의 BATES 해석적 등가 경로 (격자 미사용)."""

    # 결과 저장용 리스트 (simulate() 와 동일 구성)
    t_list = []
    D_gc_list, L_g_list = [], []
    A_be_list, A_bc_list, A_bg_list = [], [], []
    m_g_list, m_dot_GEN_list, m_dot_NOZ_list, m_dot_REMAIN_list = [], [], [], []
    den_REMAIN_list, P_c_list, R_dot_list = [], [], []
    Kn_list = []
    F_list, Cf_list, Isp_list = [], [], []

    # 초기 상태 (simulate() 와 동일)
    t = 0.0
    P_c = P_atm
    R_dot = srm.R_dot_calculation(a, n, P_c)
    m_REMAIN = 0.0
    m_g0 = m_g if m_g > 0 else None   # 노즐목 삭마 진행률 기준

    if verbose:
        print(f"[BATES 해석적 경로] 격자 미사용 — D_gc={D_gc:.2f}mm, "
              f"D_g={D_g:.2f}mm, L_g={L_g:.2f}mm, Δt={t_interval}s")

    while t <= t_end + 1e-12:
        # (격자 경로의 A_port 계산·D_port 출력은 물리에 미기여 → 생략)

        # 노즐목 삭마 — SRM_simul.simulate() 와 동일 정의(소비 질량분율).
        # ⚠ 이 줄이 없으면 포털 기본 형상(원형 코어 BATES)에서만 삭마가
        #   무시되어, 포트 형상에 따라 결과가 달라지는 함정이 생긴다.
        A_t_now = srm.throat_area_eroded(
            A_NOZ_t, e_NOZ, 0.0 if m_g0 is None else 1.0 - m_g / m_g0)

        D_gc_new, L_g_new = srm.Update_grain(R_dot, t_interval, D_gc, L_g, N)

        m_dot_GEN, D_gc, L_g, V_g = srm.m_dot_GEN_calculation(
            D_g, D_gc, D_gc_new, L_g, L_g_new, den_actual, t_interval)

        # 점화 램프 (simulate() 와 동일)
        if t_ignition > 0.0 and t < t_ignition:
            m_dot_GEN *= t / t_ignition

        D_gc_list.append(D_gc)
        L_g_list.append(L_g)

        A_be = srm.A_be_calculation(N, D_g, D_gc)
        A_bc = srm.A_bc_calculation(D_gc, L_g)
        A_bg = srm.A_bg_calculation(A_be, A_bc)
        Kn = srm.Kn_calculation(A_bg, A_t_now)
        A_be_list.append(A_be)
        A_bc_list.append(A_bc)
        A_bg_list.append(A_bg)
        Kn_list.append(Kn)

        m_g -= m_dot_GEN * t_interval
        m_g_list.append(m_g)

        # 챔버 질량수지 & 압력 — simulate() 의 암시적 backward-Euler 해 그대로
        V_free_mm3 = V_c - V_g
        if V_free_mm3 <= 0.0:
            if verbose:
                print("형상 비현실: 자유체적 <= 0 (그레인이 챔버보다 큼) → 종료")
            break
        C1 = 1.0e3 * R_specific * T_actual / V_free_mm3       # MPa per kg
        At_term = A_t_now / C_star                            # kg/s per MPa
        P_c = (C1 * (m_REMAIN + m_dot_GEN * t_interval) + P_atm) \
            / (1.0 + C1 * At_term * t_interval)

        m_dot_NOZ = srm.m_dot_NOZ_calculation(P_c, A_t_now, C_star)
        m_dot_REMAIN = m_dot_GEN - m_dot_NOZ
        m_REMAIN += m_dot_REMAIN * t_interval
        if m_REMAIN < 0.0:
            m_REMAIN = 0.0
            P_c = P_atm
        den_REMAIN = srm.Den_REMAIN_calculation(V_c, V_g, m_REMAIN)

        m_dot_GEN_list.append(m_dot_GEN)
        m_dot_NOZ_list.append(m_dot_NOZ)
        m_dot_REMAIN_list.append(m_dot_REMAIN)
        den_REMAIN_list.append(den_REMAIN)
        P_c_list.append(P_c)

        R_dot = srm.R_dot_calculation(a, n, P_c)
        R_dot_list.append(R_dot)

        if P_c > P_atm:
            Cf = srm.Cf_calculation(gamma, P_c, P_atm, ER_GLOBAL[0])
            thrust = srm.F_calculation(Cf, P_c, A_t_now, NOZ_eff)
            m_dot_for_isp = srm.m_dot_NOZ_calculation(P_c, A_t_now, C_star)
            Isp = srm.Isp_calculation(thrust, m_dot_for_isp)
        else:
            Cf, thrust, Isp = 0.0, 0.0, 0.0
        F_list.append(thrust)
        Cf_list.append(Cf)
        Isp_list.append(Isp)

        t_list.append(t)

        # 종료 조건 (simulate() 와 동일)
        if D_gc >= D_g or L_g <= 0 or m_g <= 0:
            if verbose:
                if D_gc >= D_g:
                    print(f"외벽 도달: D_gc={D_gc:.2f} mm >= D_g={D_g:.2f} mm")
                elif L_g <= 0:
                    print(f"높이 도달: L_g={L_g:.2f} mm <= 0 mm")
                else:
                    print(f"그레인 소진: m_g={m_g:.2f} g <= 0 g")
            break

        t += t_interval

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
        [],                 # snapshot_list (시각화 전용 — 고속 경로 미지원)
        dummy, dummy,       # X_m, Y_m
        dummy,              # grain_sdf_m
    )


# Cf_calculation 은 ER 인자가 필요하다. simulate() 시그니처와의 호환을 위해
# ER 을 전역 홀더로 전달한다 (simulate_bates_compat 에서 설정).
ER_GLOBAL = [8.0]


def simulate_bates_compat(
    exprs, xlim, ylim,      # 격자 경로 호환용 (원형 코어 확인에만 사용)
    V_c, D_g, D_gc, N, L_g, V_g,
    den_actual, m_g, A_NOZ_t, NOZ_eff, R_specific, gamma,
    C_star, T_actual, P_target, P_atm, e_NOZ,
    a, n, t_end, t_interval, grid=0, snapshot_dt=0.0, ER=8.0,
    t_ignition=0.0, verbose=True,
):
    """
    SRM_simul.simulate() 와 동일한 시그니처의 래퍼.
    exprs/xlim/ylim/grid/snapshot_dt 는 받되 격자 계산에는 사용하지 않는다.
    호출 측에서 is_circular_port() 로 원형 여부를 먼저 확인할 것.
    """
    ER_GLOBAL[0] = ER
    return simulate_bates(
        V_c, D_g, D_gc, N, L_g, V_g, den_actual, m_g, A_NOZ_t,
        NOZ_eff, R_specific, gamma, C_star, T_actual, P_target,
        P_atm, e_NOZ, a, n, t_end, t_interval,
        t_ignition=t_ignition, verbose=verbose)


import re as _re

_CIRCLE_RE = _re.compile(
    r"^\s*circle\s*\(\s*x\s*,\s*y\s*,\s*0(?:\.0*)?\s*,\s*0(?:\.0*)?\s*,"
    r"\s*([0-9.eE+\-]+)\s*\)\s*$")


def is_circular_port(exprs, D_gc_mm: float, xlim, ylim, D_g_mm: float) -> bool:
    """
    포트 식이 '중심 원형 코어 1개'이며 반지름이 D_gc/2 와 일치하고,
    좌표계가 1unit=1mm(xlim=±D_g/2)인 경우에만 True.
    이 조건에서만 해석적 경로가 격자 경로와 등가이다.
    """
    if not exprs or len(exprs) != 1:
        return False
    m = _CIRCLE_RE.match(exprs[0])
    if not m:
        return False
    try:
        r = float(m.group(1))
    except ValueError:
        return False
    half = D_g_mm / 2.0
    coord_ok = (abs(xlim[0] + half) < 1e-9 and abs(xlim[1] - half) < 1e-9
                and abs(ylim[0] + half) < 1e-9 and abs(ylim[1] - half) < 1e-9)
    return coord_ok and abs(r - D_gc_mm / 2.0) < 1e-6
