"""해석해 — 검증 기준 (V1-a).

원형 포트(BATES) 세그먼트는 닫힌 식이 있다. 수치 거리장 결과를 여기에
맞춰 보는 것이 첫 번째 검증이다.
"""
from __future__ import annotations

import numpy as np


def bates_area_end(w, r_port: float, r_outer: float):
    """원형 포트의 남은 단면적 [mm²] = π(R_o² − (R_p + w)²)."""
    w = np.asarray(w, dtype=float)
    r_in = np.minimum(r_port + w, r_outer)
    return np.pi * (r_outer ** 2 - r_in ** 2)


def bates_perimeter(w, r_port: float, r_outer: float):
    """원형 포트의 등고선 둘레 [mm] = 2π(R_p + w), 웹 소진 후 0."""
    w = np.asarray(w, dtype=float)
    r_in = r_port + w
    return np.where(r_in <= r_outer, 2.0 * np.pi * r_in, 0.0)


def bates_burn_area(w, r_port: float, r_outer: float, length: float,
                    ends_burn: bool = True):
    """원형 포트 세그먼트의 연소면적 [mm²]."""
    w = np.asarray(w, dtype=float)
    p = bates_perimeter(w, r_port, r_outer)
    if not ends_burn:
        return p * length
    l_rem = np.clip(length - 2.0 * w, 0.0, None)
    return p * l_rem + 2.0 * bates_area_end(w, r_port, r_outer)
