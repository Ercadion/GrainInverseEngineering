"""격자 해상도에 따른 수렴 확인."""
import numpy as np

from gre.core import analytic
from gre.core.burnback2d import SegmentBurnback2D
from gre.geometry.port import circular_port, star_port


def _max_area_error(n):
    sb = SegmentBurnback2D(circular_port(10.0), d_outer=60.0, n=n)
    w = np.linspace(0.0, 15.0, 16)
    num, ana = sb.area_end(w), analytic.bates_area_end(w, 10.0, 30.0)
    return float(np.max(np.abs(num - ana) / ana))


def test_area_error_decreases_with_resolution():
    e_coarse, e_fine = _max_area_error(201), _max_area_error(801)
    assert e_fine < e_coarse
    assert e_fine < 5e-4


def test_star_port_runs_and_is_consistent():
    """스타 포트: 해석해는 없지만 체적·면적 정합성은 성립해야 한다."""
    sb = SegmentBurnback2D(star_port(12.0, 6, 3.0), d_outer=60.0, n=401)
    w = np.linspace(0.0, sb.w_max, 300)
    a_b = sb.burn_area(w, 80.0, ends_burn=True)
    v = sb.volume(w, 80.0, ends_burn=True)
    burnt_by_area = np.concatenate([[0.0], np.cumsum(np.diff(w) * (a_b[1:] + a_b[:-1]) / 2)])
    assert np.max(np.abs(burnt_by_area - (v[0] - v))) / v[0] < 1e-2
    # 스타는 초기 연소면이 같은 면적의 원형 포트보다 넓어야 한다
    from gre.geometry.port import FourierPort
    r_eq = np.sqrt(sb.port.area() / np.pi)
    sb_circ = SegmentBurnback2D(FourierPort(r_eq), d_outer=60.0, n=401)
    assert sb.perimeter(0.0)[0] > sb_circ.perimeter(0.0)[0]
