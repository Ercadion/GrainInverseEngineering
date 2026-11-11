"""다중 세그먼트 조립 검증."""
import numpy as np

from gre.core import analytic
from gre.core.burnback2d import SegmentBurnback2D
from gre.core.grain import Grain, Segment
from gre.geometry.port import circular_port


def test_two_identical_segments_double_single():
    sb = SegmentBurnback2D(circular_port(10.0), d_outer=60.0, n=301)
    one = Grain([Segment(sb, 60.0)])
    two = Grain([Segment(sb, 60.0), Segment(sb, 60.0)])
    w = np.linspace(0.0, 15.0, 20)
    assert np.allclose(two.burn_area(w), 2.0 * one.burn_area(w), rtol=1e-12)
    assert abs(two.mass() - 2.0 * one.mass()) < 1e-9


def test_mixed_ports_sum_matches_parts():
    """포트가 다른 세그먼트를 쌓으면 합이 각각의 합과 같아야 한다."""
    sb_a = SegmentBurnback2D(circular_port(8.0), d_outer=60.0, n=301)
    sb_b = SegmentBurnback2D(circular_port(14.0), d_outer=60.0, n=301)
    g = Grain([Segment(sb_a, 50.0), Segment(sb_b, 70.0)])
    w = np.linspace(0.0, 18.0, 25)
    part = (Grain([Segment(sb_a, 50.0)]).burn_area(w)
            + Grain([Segment(sb_b, 70.0)]).burn_area(w))
    assert np.allclose(g.burn_area(w), part, rtol=1e-12)


def test_shorter_web_segment_dies_first():
    """웨브가 작은 세그먼트가 먼저 소진되어 A_b 가 계단처럼 준다."""
    sb_small = SegmentBurnback2D(circular_port(20.0), d_outer=60.0, n=301)  # 웨브 10
    sb_large = SegmentBurnback2D(circular_port(10.0), d_outer=60.0, n=301)  # 웨브 20
    g = Grain([Segment(sb_small, 60.0), Segment(sb_large, 60.0)])
    a_before = float(g.burn_area(9.5)[0])
    a_after = float(g.burn_area(10.5)[0])
    assert a_after < a_before


def test_mass_matches_analytic_annulus():
    sb = SegmentBurnback2D(circular_port(10.0), d_outer=60.0, n=401)
    g = Grain([Segment(sb, 100.0)], density=1.841 * 0.92)
    v_ana = np.pi * (30.0 ** 2 - 10.0 ** 2) * 100.0          # mm³
    assert abs(g.volume(0.0)[0] - v_ana) / v_ana < 2e-3
    # 질량은 자기 체적과 정확히 일관되어야 한다
    assert abs(g.mass() - g.volume(0.0)[0] * g.density / 1000.0) < 1e-9


def test_short_segment_dies_axially():
    """L/2 < w_max 인 세그먼트는 축방향으로 먼저 소진되어야 한다.

    초판은 web_max 가 min(w_max, L/2 + w_max) 로 쓰여 있어 언제나 w_max 를
    돌려주었고, 그 결과 체적이 0 인데 끝면 항 2·A_end 가 남아 '없는 추진제가
    타는' 상태가 되었다 (dV/dw = -A_b 가 L=24 mm 에서 18 % 깨짐).
    """
    sb = SegmentBurnback2D(circular_port(10.0), d_outer=60.0, n=301)   # w_max = 20
    g = Grain([Segment(sb, 24.0)])                                     # L/2 = 12
    assert abs(g.web_max() - 12.0) < 1e-9
    assert float(g.burn_area(12.5)[0]) == 0.0
    assert float(g.volume(12.5)[0]) == 0.0

    # 끝면을 막으면 축방향 소진이 없으므로 w_max 가 그대로 수명이다
    g_inh = Grain([Segment(sb, 24.0, ends_burn=False)])
    assert abs(g_inh.web_max() - sb.w_max) < 1e-9


def test_volume_burn_area_identity_short_segment():
    """dV/dw = -A_b 가 짧은 세그먼트에서도 성립해야 한다."""
    sb = SegmentBurnback2D(circular_port(10.0), d_outer=60.0, n=301)
    for L in (120.0, 24.0, 12.0):
        g = Grain([Segment(sb, L)])
        w = np.linspace(0.0, g.web_max(), 4001)
        consumed = np.trapezoid(g.burn_area(w), w)
        v0, v1 = float(g.volume(0.0)[0]), float(g.volume(w[-1])[0])
        assert abs((v0 - v1) - consumed) / v0 < 1.5e-2, f"L={L}"
