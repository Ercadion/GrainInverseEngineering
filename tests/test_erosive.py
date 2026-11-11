"""gre.validate.erosive — 포트 질량유속 진단."""
import math

import pytest

torch = pytest.importorskip("torch")
torch.set_default_dtype(torch.float64)

from gre.optimize.forward import DiffMotor  # noqa: E402
from gre.validate.erosive import erosive_hinge, port_flux, summarize  # noqa: E402

D, R0, L, AT = 50.0, 10.0, [30.0] * 3, 61.75


@pytest.fixture(scope="module")
def tab():
    m = DiffMotor(d_outer=D, n_grid=121)
    return m, port_flux(m, torch.tensor(R0), torch.zeros(4), torch.zeros(4), L, AT)


def test_aft_flux_matches_nozzle_mass_balance(tab):
    """노즐 쪽 끝 G 는 정상상태 질량 보존 p·A_t/(c*·A_port) 와 같아야 한다.

    차이는 챔버 기체가 자유체적을 채우는 항 ρ_g/ρ_p (~0.7 %) 뿐이다.
    """
    m, t = tab
    p = float(t["p_eq"][0]) * 1e6
    a_port = math.pi * R0 ** 2 * 1e-6
    g_nozzle = p * AT * 1e-6 / m.c_star_actual / a_port
    g_model = float(t["G"][-1, 0])
    assert abs(g_model / g_nozzle - 1) < 0.015


def test_cumulative_flux_grows_downstream(tab):
    """같은 포트가 쌓이면 G 는 노즐 쪽으로 갈수록 커진다 (1:2:3)."""
    _, t = tab
    g = t["G"][:, 0]
    assert float(g[1] / g[0]) == pytest.approx(2.0, rel=1e-6)
    assert float(g[2] / g[0]) == pytest.approx(3.0, rel=1e-6)


def test_port_ratio_and_mach_consistent(tab):
    """노즐 쪽 끝 M 은 A_port/A_t 만의 함수 — 압력에 무관해야 한다."""
    m, t = tab
    s = summarize(t)
    assert s["port_ratio_min"] == pytest.approx(math.pi * R0 ** 2 / AT, rel=0.02)
    assert 0.0 < s["mach_max"] < 0.3 and not s["flags"]


def test_equilibrium_implicit_gradient_is_exact():
    """평형압의 음함수 경사 — A_b 에 대한 dp/dA_b 가 중앙차분과 일치해야 한다.

    ⚠ 초판은 이분법 결과를 no_grad 로 써서 이 경로가 통째로 빠져 있었다.
    """
    from gre.validate.erosive import _equilibrium_implicit
    m = DiffMotor(d_outer=D, n_grid=121)
    at = torch.tensor(AT * 1e-6)
    ab = torch.tensor([15000.0, 9000.0], requires_grad=True)
    p = _equilibrium_implicit(m, ab, at)
    (g,) = torch.autograd.grad(p.sum(), ab)
    h = 1.0
    for i in range(2):
        e = torch.zeros(2); e[i] = h
        fd = (float(_equilibrium_implicit(m, ab.detach() + e, at)[i])
              - float(_equilibrium_implicit(m, ab.detach() - e, at)[i])) / (2 * h)
        assert float(g[i]) == pytest.approx(fd, rel=1e-4)


def test_shape_gradient_inherits_documented_straight_through_bias():
    """형상 경사는 `DiffSegment2D.phi` 의 straight-through 추정자를 그대로 물려받는다.

    gradient_audit.md §2: autograd 가 참 기울기를 **1.4~1.8 배 과소평가**,
    방향은 맞음. 여기서도 같은 비가 나와야 한다 — 이 모듈이 새 편향을 만들지
    않았다는 확인이다 (A_port 의 해석 기울기 2πr₀ 로 대조).
    """
    m = DiffMotor(d_outer=D, n_grid=121)
    x = torch.tensor(R0, requires_grad=True)
    t = port_flux(m, x, torch.zeros(4), torch.zeros(4), L, AT, n_w=60)
    (g_area,) = torch.autograd.grad(t["a_port"][-1, 0], x, retain_graph=True)
    ratio = 2 * math.pi * R0 / float(g_area)
    assert 1.3 < ratio < 1.9
    (g_G,) = torch.autograd.grad(t["G"][-1, 0], x)
    assert float(g_G) < 0                            # 포트를 넓히면 G 감소 (방향은 맞다)


def test_hinge_is_active_and_differentiable():
    m = DiffMotor(d_outer=D, n_grid=121)
    x = torch.tensor(R0, requires_grad=True)
    t = port_flux(m, x, torch.zeros(4), torch.zeros(4), L, AT, n_w=60)
    pen = erosive_hinge(t, g_max=400.0)              # 일부러 낮은 문턱 → 활성
    (g,) = torch.autograd.grad(pen, x)
    assert float(pen.detach()) > 0 and float(g) < 0 and math.isfinite(float(g))
