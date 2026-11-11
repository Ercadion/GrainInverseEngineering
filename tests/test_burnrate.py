"""연소속도 전 구간 연속 모델 검증."""
import numpy as np

from gre.core import burnrate
from gre.core.burnrate import (NAKKA_TABLE, design_band, in_excluded_band,
                               model_for, piecewise_rate)


def test_matches_table_away_from_boundaries():
    """구간 경계에서 떨어진 곳에서는 원 구간표를 1% 이내로 재현해야 한다."""
    m = model_for("KNSB")
    tbl = NAKKA_TABLE["KNSB"]
    for lo, hi, a, n in tbl:
        p = np.geomspace(lo * 1.15, hi * 0.87, 8)     # 경계에서 떨어진 내부
        assert np.max(np.abs(m.rate(p) - a * p ** n) / (a * p ** n)) < 1e-2


def test_continuous_across_boundaries():
    """경계를 건너도 값이 튀지 않는다 (원 구간표는 튄다)."""
    m = model_for("KNSB")
    for _, hi, _, _ in NAKKA_TABLE["KNSB"][:-1]:
        left, right = m.rate(hi * 0.999), m.rate(hi * 1.001)
        assert abs(right - left) / left < 5e-3
    # 대조: 원본 표는 3.792 MPa 에서 실제로 불연속이다
    jump = piecewise_rate(NAKKA_TABLE["KNSB"], 3.7921) - \
        piecewise_rate(NAKKA_TABLE["KNSB"], 3.7919)
    assert abs(jump) > 0.2


def test_defined_everywhere():
    """0 부근부터 상한 위까지 유한하고 양수 — 점화·테일오프가 지나간다."""
    m = model_for("KNSB")
    p = np.concatenate([[1e-4, 1e-3, 0.01], np.geomspace(0.05, 20.0, 50)])
    r = m.rate(p)
    assert np.all(np.isfinite(r)) and np.all(r > 0)


def test_design_band_power_law_close_to_nakka():
    """기본 설계 대역의 회귀 (a, n) 이 Nakka 구간값과 가깝다."""
    m = model_for("KNSB")
    a, n = m.power_law_in_band(design_band("KNSB"))
    assert abs(a - 3.907) / 3.907 < 0.08
    assert abs(n - 0.535) < 0.03


def test_excluded_band_flag():
    assert bool(in_excluded_band("KNSB", 1.0)[0]) is True
    assert bool(in_excluded_band("KNSB", 5.0)[0]) is False


def test_exponent_below_unity_in_design_bands():
    """설계 대역 안에서 국소 압력지수는 1 미만이어야 한다 (평형 안정).

    n > 1 이면 평형이 불안정하고, 평형압의 설계변수 민감도가 1/(1-n) 이라
    최적화기 경사가 발산한다. 초판(PCHIP + 경계 평균)은 3.906~3.930 MPa 에서
    n = 1.0064 로 이 선을 넘었다.
    """
    for prop in ("KNSB", "KNDX"):
        m = burnrate.model_for(prop)
        for which in ("default", "low"):
            band = burnrate.design_band(prop, which)
            assert m.max_exponent_in(band) < 1.0, f"{prop}/{which}"


def test_blend_reproduces_power_law_away_from_breaks():
    """전이 구간 밖에서는 모델이 곧 원래 멱법칙이어야 한다."""
    m = burnrate.model_for("KNSB")
    tab = burnrate.NAKKA_TABLE["KNSB"]
    bnds = [t[0] for t in tab[1:]]
    p = np.geomspace(0.12, 10.5, 4000)
    keep = np.ones(p.shape, dtype=bool)
    for b in bnds:
        keep &= np.abs(np.log(p / b)) > m.delta
    ref = burnrate.piecewise_rate(tab, p[keep])
    assert np.allclose(m.rate(p[keep]), ref, rtol=1e-10)


def test_exponent_matches_finite_difference():
    """해석 도함수가 유한차분과 일치해야 한다."""
    m = burnrate.model_for("KNSB")
    lp = np.log(np.geomspace(0.15, 10.0, 3000))
    h = 1e-6
    fd = (m._ln_r(lp + h) - m._ln_r(lp - h)) / (2 * h)
    assert np.abs(m._dln_r(lp) - fd).max() < 1e-7
