"""V1-c — 미분가능 판의 정방향 일치와 경사 검증.

torch 가 없으면 건너뛴다.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from gre.core import analytic
from gre.core.diff_burnback import DiffSegment2D

D_OUTER, R_PORT, L_SEG = 60.0, 12.0, 120.0
N = 161


@pytest.fixture(scope="module")
def seg():
    return DiffSegment2D(d_outer=D_OUTER, n=N)


def _pack(vec, grad=False):
    r0 = torch.tensor(vec[0], dtype=torch.float64, requires_grad=grad)
    a = torch.tensor(vec[1:4], dtype=torch.float64, requires_grad=grad)
    b = torch.tensor(vec[4:7], dtype=torch.float64, requires_grad=grad)
    return r0, a, b


def _objective(seg, vec, grad=False):
    r0, a, b = _pack(vec, grad)
    phi = seg.phi(r0, a, b)
    w = torch.linspace(0.0, 8.0, 5, dtype=torch.float64)
    out = seg.burn_area(phi, w, length=L_SEG).sum()
    if grad:
        out.backward()
        return float(out.detach()), np.concatenate([[float(r0.grad)], a.grad.numpy(), b.grad.numpy()])
    return float(out.detach())


def test_forward_matches_analytic(seg):
    """원형 포트는 해석해와 맞아야 한다 (numpy 판과 같은 기준)."""
    r0, a, b = _pack(np.array([R_PORT, 0, 0, 0, 0, 0, 0.0]))
    phi = seg.phi(r0, a, b)
    w = torch.linspace(0.0, 12.0, 7, dtype=torch.float64)
    num = seg.area_end(phi, w).detach().numpy()
    ana = analytic.bates_area_end(w.numpy(), R_PORT, D_OUTER / 2)
    assert np.max(np.abs(num - ana) / ana) < 5e-3

    p_num = seg.perimeter(phi, w).detach().numpy()
    p_ana = analytic.bates_perimeter(w.numpy(), R_PORT, D_OUTER / 2)
    assert np.max(np.abs(p_num - p_ana) / p_ana) < 1e-2


def test_gradient_matches_finite_difference(seg):
    """자동미분 경사와 유한차분이 같은 방향을 가리켜야 한다 (V1-c)."""
    vec = np.array([12.0, 0.3, -0.2, 1.2, 0.0, 0.1, 0.0])
    _, g_auto = _objective(seg, vec, grad=True)
    g_fd = np.zeros_like(g_auto)
    for i in range(len(vec)):
        h = 1e-3 * max(1.0, abs(vec[i]))
        vp, vm = vec.copy(), vec.copy()
        vp[i] += h
        vm[i] -= h
        g_fd[i] = (_objective(seg, vp) - _objective(seg, vm)) / (2 * h)
    cos = float(g_auto @ g_fd / (np.linalg.norm(g_auto) * np.linalg.norm(g_fd)))
    # 방향 일치가 본질이다. 크기 차이는 ① hard min 정방향의 격자 잡음으로
    # 유한차분 자체가 흔들리고 ② soft-min 경사가 동점을 평균한 값이라서 생긴다.
    # 격자를 촘촘히 하면 줄어든다 (n=161 → 6%, n=241 → 3.5%).
    assert cos > 0.998
    assert np.linalg.norm(g_auto - g_fd) / np.linalg.norm(g_fd) < 0.12


def test_symmetry_gradient_is_zero(seg):
    """대칭 스타에서 위상(sin 계수) 방향 경사는 0 이어야 한다.

    hard min 을 쓰면 동점 처리 때문에 여기서 가짜 경사가 나온다 —
    soft-min 을 쓰는 이유를 고정해 두는 테스트다.
    """
    vec = np.array([12.0, 0.0, 0.0, 1.5, 0.0, 0.0, 0.0])   # a3 만 있는 3점 스타
    _, g = _objective(seg, vec, grad=True)
    scale = max(abs(g[0]), 1e-9)
    assert np.max(np.abs(g[4:])) / scale < 1e-6


def test_gradient_descent_reduces_loss(seg):
    """경사를 따라가면 목표 곡선과의 차이가 실제로 줄어든다."""
    w = torch.linspace(0.0, 6.0, 5, dtype=torch.float64)

    target_vec = np.array([12.0, 0.0, 0.0, 1.5, 0.0, 0.0, 0.0])
    r0_t, a_t, b_t = _pack(target_vec)
    with torch.no_grad():
        target = seg.burn_area(seg.phi(r0_t, a_t, b_t), w, length=L_SEG)

    vec = np.array([11.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0])
    r0, a, b = _pack(vec, grad=True)
    opt = torch.optim.Adam([r0, a, b], lr=0.15)
    losses = []
    for _ in range(12):
        opt.zero_grad()
        loss = ((seg.burn_area(seg.phi(r0, a, b), w, length=L_SEG) - target) ** 2).mean()
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    assert losses[-1] < 0.25 * losses[0]
