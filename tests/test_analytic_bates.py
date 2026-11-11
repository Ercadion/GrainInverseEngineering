"""V1-a: 원형 포트(BATES) 해석해와의 대조."""
import numpy as np
import pytest

from gre.core import analytic
from gre.core.burnback2d import SegmentBurnback2D
from gre.geometry.port import circular_port

R_PORT, D_OUTER, LENGTH = 10.0, 60.0, 80.0
R_OUTER = D_OUTER / 2.0


@pytest.fixture(scope="module")
def sb():
    return SegmentBurnback2D(circular_port(R_PORT), d_outer=D_OUTER, n=401)


def test_area_end_matches_analytic(sb):
    w = np.linspace(0.0, 15.0, 16)
    num = sb.area_end(w)
    ana = analytic.bates_area_end(w, R_PORT, R_OUTER)
    assert np.max(np.abs(num - ana) / ana) < 2e-3          # 0.2 %


def test_perimeter_matches_analytic(sb):
    w = np.linspace(0.0, 15.0, 16)
    num = sb.perimeter(w)
    ana = analytic.bates_perimeter(w, R_PORT, R_OUTER)
    assert np.max(np.abs(num - ana) / ana) < 5e-3          # 0.5 %


@pytest.mark.parametrize("ends_burn", [True, False])
def test_burn_area_matches_analytic(sb, ends_burn):
    w = np.linspace(0.0, 15.0, 16)
    num = sb.burn_area(w, LENGTH, ends_burn=ends_burn)
    ana = analytic.bates_burn_area(w, R_PORT, R_OUTER, LENGTH, ends_burn=ends_burn)
    assert np.max(np.abs(num - ana) / ana) < 5e-3


def test_web_max(sb):
    # 원형 포트의 최대 웹 = 외반경 − 포트반경
    assert abs(sb.w_max - (R_OUTER - R_PORT)) < 2 * sb.h


def test_volume_consistency(sb):
    """∫A_b dw = V(0) − V(w) — 체적과 연소면적의 정합성."""
    w = np.linspace(0.0, sb.w_max, 400)
    a_b = sb.burn_area(w, LENGTH, ends_burn=True)
    v = sb.volume(w, LENGTH, ends_burn=True)
    burnt_by_area = np.concatenate([[0.0], np.cumsum(np.diff(w) * (a_b[1:] + a_b[:-1]) / 2)])
    burnt_by_volume = v[0] - v
    ref = v[0]
    assert np.max(np.abs(burnt_by_area - burnt_by_volume)) / ref < 5e-3
