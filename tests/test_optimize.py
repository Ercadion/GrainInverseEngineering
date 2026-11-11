"""미분가능 정방향·제조 제약·역설계 루프 검증."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from gre.core.ballistics import MotorSpec, simulate
from gre.core.burnback2d import SegmentBurnback2D
from gre.core.burnrate import model_for
from gre.core.diff_burnback import params
from gre.core.grain import Grain, Segment
from gre.geometry.port import circular_port
from gre.optimize.constraints import (ManufacturingLimits, ThroatSpec,
                                      connected_components, min_tip_radius,
                                      thin_feature_penalty, web_thickness)
from gre.optimize.forward import STIFF_SAFETY, DiffMotor, TorchBurnRate

D_OUTER, L_SEG, AT = 60.0, 120.0, np.pi * 16.0

#: 이 묶음의 시험들이 쓰는 노즐목 입력. 값 자체는 중요하지 않다 (여기서 보는
#: 것은 최적화기의 기전이다) — 노즐목 모드 자체를 보는 시험은 `test_strict.py`
#: 에 따로 있다. `AT = π·16` 이 이 파일의 공칭 목이므로 그 직경을 쓴다.
THROAT = ThroatSpec.fixed(2.0 * np.sqrt(AT / np.pi))


@pytest.fixture(scope="module")
def motor():
    return DiffMotor(d_outer=D_OUTER, n_grid=141, n_samples=512, n_w=220)


# ──────────────────────────────────────────────────────────── 연소속도 이식
def test_torch_burn_rate_matches_numpy():
    """torch 판과 numpy 판이 같은 식이어야 한다."""
    ref = model_for("KNSB")
    tb = TorchBurnRate("KNSB")
    p = np.geomspace(0.05, 12.0, 500)
    got = tb.rate(torch.tensor(p)).numpy()
    assert np.allclose(got, ref.rate(p), rtol=1e-12)


# ──────────────────────────────────────────────────────── 정방향 대조·가드
def test_forward_matches_reference_ode(motor):
    """미분가능 ODE 가 기준 LSODA 와 맞아야 한다 (최대압·최대추력)."""
    r0, a, b = params(12.0, a=(0.0,) * 3, b=(0.0,) * 3, requires_grad=False)
    t, p, f, diag = motor.simulate(r0, a, b, [L_SEG], AT)

    bb = SegmentBurnback2D(circular_port(12.0), D_OUTER, n=401)
    g = Grain([Segment(bb, L_SEG)])
    spec = MotorSpec(grain=g, burn_rate=model_for("KNSB"), a_throat=AT,
                     chamber_volume=float(g.volume(0.0)[0]) * 1.25,
                     expansion_ratio=diag["expansion_ratio"])
    ref = simulate(spec, t_end=10.0)
    assert abs(float(p.max()) - ref.p_max) / ref.p_max < 5e-3
    assert abs(float(f.max()) - ref.f_max) / ref.f_max < 5e-3
    i_sim = float(torch.trapezoid(f, t))
    assert abs(i_sim - ref.total_impulse) / ref.total_impulse < 2e-2


def test_stiff_guard_shrinks_step(motor):
    """h 는 그 설계의 τ_min 에 묶여야 한다 (RK4 안정 한계 2.78τ)."""
    r0, a, b = params(12.0, a=(0.0,) * 3, b=(0.0,) * 3, requires_grad=False)
    _, _, _, diag = motor.simulate(r0, a, b, [L_SEG], AT)
    # 첫 스텝이 가장 작아야 한다 (τ 는 연소가 진행되며 커진다)
    assert diag["h_min"] <= STIFF_SAFETY * diag["tau_min"] + 1e-12
    assert diag["h_max_used"] <= motor.h_max + 1e-12
    # 챔버를 좁히면 τ 가 줄고 h 도 따라 줄어야 한다
    tight = DiffMotor(d_outer=D_OUTER, n_grid=141, n_samples=512, n_w=220,
                      chamber_margin=1.05)
    _, _, _, d2 = tight.simulate(r0, a, b, [L_SEG], AT)
    assert d2["tau_min"] < diag["tau_min"]
    assert d2["h_min"] <= STIFF_SAFETY * d2["tau_min"] + 1e-12
    assert d2["h_min"] < diag["h_min"]              # 가드가 실제로 줄였는가


def test_expansion_ratio_auto_and_fixed(motor):
    """ε 미지정이면 자동 산출, 지정하면 그 값을 쓴다. 압력은 둘 다 같다."""
    r0, a, b = params(12.0, a=(0.0,) * 3, b=(0.0,) * 3, requires_grad=False)
    _, p1, f1, d1 = motor.simulate(r0, a, b, [L_SEG], AT)
    _, p2, f2, d2 = motor.simulate(r0, a, b, [L_SEG], AT, expansion_ratio=3.0)
    assert d1["expansion_ratio"] > 1.0 and d2["expansion_ratio"] == 3.0
    assert abs(float(p1.max()) - float(p2.max())) / float(p1.max()) < 1e-12
    assert abs(float(f1.max()) - float(f2.max())) > 1e-3


# ─────────────────────────────────────────────────── 원형은 안장점이다
def test_circular_port_is_a_stationary_point(motor):
    """a = b = 0 에서 곡선 지표의 경사가 0 이어야 한다.

    포트 면적이 ∮r²/2 dθ = πr₀² + (π/2)Σ(a_k²+b_k²) 라 계수에 대해 짝함수다.
    따라서 원형에서 출발하면 경사 하강이 한 발도 못 뗀다 — `seed_design` 이
    대칭을 깨고 시작하는 이유이고, §4.6.5 의 'B 후보를 초기값으로'
    가 수렴 속도가 아니라 **필수 조건**인 이유다.
    """
    r0, a, b = params(12.0, a=(0.0,) * 3, b=(0.0,) * 3)
    t, p, f, _ = motor.simulate(r0, a, b, [L_SEG], AT, expansion_ratio=5.0)
    impulse = torch.trapezoid(f, t)
    g = torch.autograd.grad(impulse, a, retain_graph=True)[0]
    assert float(g.abs().max()) < 1e-8
    g2 = torch.autograd.grad(f.max(), a, retain_graph=True)[0]
    assert float(g2.abs().max()) < 1e-8


def test_gradient_matches_finite_difference(motor):
    """비원형 지점에서 자동미분이 유한차분과 같은 방향을 가리켜야 한다."""
    def peak(av, grad):
        r0, a, b = params(12.0, a=tuple(av), b=(0.0,) * 3, requires_grad=grad)
        _, _, f, _ = motor.simulate(r0, a, b, [L_SEG], AT, expansion_ratio=5.0)
        return (a, f.max())

    a_t, q = peak([0.0, 0.0, 1.5], True)
    auto = torch.autograd.grad(q, a_t)[0].numpy()
    h, fd = 2e-3, []
    for k in range(3):
        up = [0.0, 0.0, 1.5]; up[k] += h
        dn = [0.0, 0.0, 1.5]; dn[k] -= h
        fd.append((float(peak(up, False)[1]) - float(peak(dn, False)[1])) / (2 * h))
    fd = np.array(fd)
    cos = auto @ fd / (np.linalg.norm(auto) * np.linalg.norm(fd))
    assert cos > 0.99


# ──────────────────────────────────────────────────────────── 제조 제약
def test_web_thickness_analytic():
    """값은 정확해야 하고(straight-through), 경사는 흘러야 한다.

    soft-max 만 쓰면 ln(N)/β 만큼 들떠 웹을 0.084 mm 얇게 보고했다.
    """
    r0, a, b = params(12.0, a=(0.0, 0.0, 3.0), b=(0.0,) * 3)
    web = web_thickness(r0, a, b, 30.0)
    assert abs(float(web) - (30.0 - 15.0)) < 1e-3        # r_max = 12 + 3
    g = torch.autograd.grad(web, a)[0]
    assert float(g.abs().max()) > 0.0


def test_thin_feature_penalty_is_gated(motor):
    """제약에 여유가 있으면 값도 경사도 정확히 0 이어야 한다.

    초판은 정사각 max-pool 커널(4중 대칭)과 승자독식 경사 때문에 여유가
    충분한 원형에서도 ∂/∂a₄ = 7.27 을 냈다. 같은 자리의 곡선 손실 경사가
    3.4e-5 였으므로 20 만 배의 가짜 신호였다. 원판 구조요소 + 이진 마스크
    detach 로 고쳤다.
    """
    lim = ManufacturingLimits()
    for a_val in ([0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 2.0]):
        r0, a, b = params(13.0, a=tuple(a_val), b=(0.0,) * 4)
        phi = motor.ds.phi(r0, a, b)
        tp_, tv_ = thin_feature_penalty(motor.ds, phi, lim)
        assert float(tp_) == 0.0 and float(tv_) == 0.0
        for q in (tp_, tv_):
            g = torch.autograd.grad(q, a, retain_graph=True, allow_unused=True)[0]
            assert g is None or float(g.abs().max()) == 0.0

    # 슬롯이 얇아지면 값과 경사가 모두 살아나야 한다
    r0, a, b = params(13.0, a=(0.0, 0.0, 0.0, 9.0), b=(0.0,) * 4)
    phi = motor.ds.phi(r0, a, b)
    tp_, _ = thin_feature_penalty(motor.ds, phi, lim)
    assert float(tp_) > 0.0
    assert float(torch.autograd.grad(tp_, a)[0].abs().max()) > 0.0


def test_tip_radius_and_connectivity(motor):
    r0, a, b = params(12.0, a=(0.0, 0.0, 3.0), b=(0.0,) * 3, requires_grad=False)
    assert min_tip_radius(r0, a, b) > 0.0
    phi = motor.ds.phi(r0, a, b)
    assert connected_components(motor.ds, phi) == 1


def test_step_schedule_is_stable_and_cheaper(motor):
    """단계별 h 는 (1) τ 가드를 전 구간에서 지키고 (2) 고정 h 와 같은 답을 내야 한다."""
    r0, a, b = params(11.0, a=(0.0, 0.0, 2.2, 0.0), b=(0.0,) * 4, requires_grad=False)
    t, p, f, diag = motor.simulate(r0, a, b, [95.0], 78.0, expansion_ratio=3.41)
    # τ 는 단조 증가하므로 최소 τ 로 묶인 상한을 어디서도 넘지 않아야 한다
    assert diag["h_max_used"] <= motor.h_max + 1e-12
    assert diag["h_min"] <= STIFF_SAFETY * diag["tau_min"] + 1e-12
    assert diag["h_max_used"] > diag["h_min"]            # 실제로 커져야 의미가 있다

    fine = DiffMotor(d_outer=D_OUTER, n_grid=141, n_samples=512, n_w=220, h_max=1e-3)
    _, p2, f2, d2 = fine.simulate(r0, a, b, [95.0], 78.0, expansion_ratio=3.41)
    assert d2["n_steps"] > diag["n_steps"] * 2          # 더 잘게 썰렸고
    assert abs(float(p.max()) - float(p2.max())) / float(p2.max()) < 1e-4


# ──────────────────────────────────────────── 비유일성 — 게이지·희소성·식별성
def test_canonical_rotation_preserves_invariants():
    """게이지 고정은 회전 불변량을 바꾸지 않고 위상만 정규화해야 한다."""
    from gre.optimize.runner import canonical_rotation
    a = torch.tensor([0.5, -1.2, 2.0, 0.3])
    b = torch.tensor([0.9, 0.4, -1.1, 0.6])
    amp0 = torch.hypot(a, b).clone()
    canonical_rotation(a, b)
    assert torch.allclose(torch.hypot(a, b), amp0, atol=1e-12)
    m = int(torch.argmax(amp0))                   # 지배 모드의 위상이 0
    assert abs(float(b[m])) < 1e-7 and float(a[m]) > 0


def test_group_l1_is_rotation_invariant():
    from gre.optimize.loss import group_l1
    a = torch.tensor([0.5, -1.2, 2.0, 0.3])
    b = torch.tensor([0.9, 0.4, -1.1, 0.6])
    before = float(group_l1(a, b, 30.0))
    k = torch.arange(1, 5, dtype=a.dtype)
    d = 0.37
    a2 = a * torch.cos(k * d) - b * torch.sin(k * d)
    b2 = a * torch.sin(k * d) + b * torch.cos(k * d)
    assert abs(float(group_l1(a2, b2, 30.0)) - before) < 1e-12


def test_identifiability_flags_a_circle_equivalent_target():
    """원형으로 맞출 수 있는 목표는 '형상 식별 불가' 로 걸러야 한다."""
    from gre.core.analytic import bates_burn_area
    from gre.validate.identifiability import assess
    w = np.linspace(0.5, 13.0, 140)
    ab = bates_burn_area(w, 11.0, 30.0, 95.0)
    rep = assess(w, ab, 30.0, 95.0)
    assert rep.circle_suffices_fixed_throat
    assert abs(rep.circle_r0 - 11.0) < 0.05
    assert rep.circle_residual < 0.1
    assert any("식별 불가" in m for m in rep.warnings())


def test_offset_perimeter_slope_tends_to_two_pi(motor):
    """등고선 둘레의 기울기는 웹이 커지면 2π 로 수렴해야 한다 (슈타이너).

    이것이 비유일성의 **구조적 원인**이다. 큰 w 에서 P(w) ≈ P₀ + 2πw 이므로
    모양 정보는 작은 w 에만 남고, 단일 세그먼트의 A_b(w) 는 사실상 '초기 둘레'
    하나로 결정된다. 그래서 다른 형상이 같은 곡선을 낸다.
    """
    w = torch.linspace(0.3, 14.0, 180)
    for a_val in ([0.0] * 4, [0.0, 0.0, 2.2, 0.0]):
        r0, a, b = params(11.0, a=tuple(a_val), b=(0.0,) * 4, requires_grad=False)
        p = motor.ds.perimeter(motor.ds.phi(r0, a, b), w).numpy()
        wn = w.numpy()
        slope = np.polyfit(wn[wn > 7.0], p[wn > 7.0], 1)[0]
        assert abs(slope - 2 * np.pi) / (2 * np.pi) < 0.02


def test_circle_curve_test_agrees_with_k_search():
    """식별성은 A_b 가 아니라 **추력 곡선**에서, A_t 를 자유로 두고 봐야 한다.

    A_b 를 고정 A_t 로 비교하면 과대평가한다 — 노즐목을 바꾸면 압력·연소속도·
    연소시간이 함께 바뀌어 A_b 가 달라도 곡선은 맞출 수 있기 때문이다. 실제로
    assess() 가 '원형으로 부족(12.4 %)' 이라 한 목표를 K 상승 탐색은 원형으로
    모양 오차 3.1 % 에 통과시켰다.
    """
    from gre.core.burnrate import model_for
    from gre.optimize.forward import DiffMotor as _DM
    from gre.optimize.runner import DesignVars as _DV
    from gre.validate.identifiability import assess, circle_can_make_curve

    m = _DM(d_outer=60.0, n_grid=121, n_samples=384, n_w=200, h_max=10e-3)
    dv = _DV(r0=torch.tensor(11.0), a=torch.tensor([0., 0., 2.2, 0.]),
             b=torch.zeros(4), a_throat=torch.tensor(78.0), lengths=[95.0])
    t, p, f, diag = m.simulate(dv.r0, dv.a, dv.b, dv.lengths, dv.a_throat)

    by_ab = assess(diag["w_tab"].numpy(), diag["a_b"].detach().numpy(), 30.0, 95.0)
    by_curve = circle_can_make_curve(t.numpy(), f.detach().numpy(), 30.0, 95.0,
                                     model_for("KNSB"))
    # A_b 기준은 더 비관적이고, 곡선 기준이 실제 최적화 결과와 맞는다
    assert by_ab.circle_residual > by_curve["residual_pct"]
    assert by_curve["suffices"]
    assert 2.0 < by_curve["r0"] < 29.0


# ═══════════════════════ 허용 집합에서 보여줄 후보 고르기
def test_pareto_front_drops_dominated():
    """모든 축에서 더 나은 대안이 있으면 보여줄 이유가 없다."""
    from gre.optimize.runner import pareto_front

    mk = lambda sh, cx: {"rms_shape_pct": sh, "complexity": cx}
    rows = [mk(2.0, 8.0), mk(3.0, 2.0), mk(4.0, 1.0), mk(5.0, 6.0),
            mk(2.5, 9.0), mk(6.0, 0.5)]
    front = pareto_front(rows)
    got = sorted((r["rms_shape_pct"], r["complexity"]) for r in front)
    assert got == [(2.0, 8.0), (3.0, 2.0), (4.0, 1.0), (6.0, 0.5)]
    # (5.0, 6.0) 은 (3.0, 2.0) 에, (2.5, 9.0) 은 (2.0, 8.0) 에 지배당한다
    assert all(r["rms_shape_pct"] != 5.0 for r in front)
    assert all(r["rms_shape_pct"] != 2.5 for r in front)


def test_shortlist_is_ordered_and_capped():
    """전선 위를 모양 오차 순으로, 최대 n 개."""
    from gre.optimize.runner import shortlist

    mk = lambda sh, cx: {"rms_shape_pct": sh, "complexity": cx}
    rows = [mk(2.0, 8.0), mk(3.0, 2.0), mk(4.0, 1.0), mk(6.0, 0.5), mk(5.0, 6.0)]
    picks = shortlist(rows, n=3)
    assert len(picks) == 3
    assert [r["rms_shape_pct"] for r in picks] == [2.0, 3.0, 4.0]
    # 전선이 모자라면 지배당한 것을 채우되 표시한다
    few = shortlist([mk(2.0, 1.0), mk(3.0, 2.0)], n=3)
    assert len(few) == 2
    assert few[0]["on_front"] and not few[1]["on_front"]
    assert shortlist([], n=3) == []


def test_margins_are_signed_room():
    """여유는 부호가 있다 — 음수면 그 사양은 집합 밖이다."""
    from gre.optimize.loss import Tolerances
    from gre.optimize.runner import margins

    m = margins({"rms_shape_pct": 4.59, "I_err_pct": 9.42, "F_err_pct": 5.51},
                Tolerances())
    assert m["모양"] == pytest.approx(5.41)
    assert m["총역적"] == pytest.approx(0.58)
    assert all(v > 0 for v in m.values())
    bad = margins({"rms_shape_pct": 3.09, "I_err_pct": 24.35, "F_err_pct": 1.10},
                  Tolerances())
    assert bad["총역적"] < 0


def test_verdict_refuses_to_force_a_design():
    """실현 불가면 설계가 아니라 진단을 낸다."""
    from gre.optimize.loss import Tolerances
    from gre.optimize.runner import admissibility, verdict_text

    rep = {"rms_shape_pct": 3.09, "I_err_pct": 24.35, "F_err_pct": 1.10,
           "gap_shape_pp": 0.0, "gap_impulse_pp": 14.35, "gap_f_max_pp": 0.0,
           "web_mm": 16.6, "r_tip_mm": 99.0, "mass_g": 390.0, "P_max_sim": 2.1,
           "components": 1, "penalty": 0.0}
    adm = admissibility(rep, ManufacturingLimits(), Tolerances())
    assert not adm["in_set"] and adm["binding"] == "총역적"
    assert adm["worst_gap_pp"] == pytest.approx(14.35)
    txt = verdict_text(adm, k_max=3)
    assert "찾을 수 없습니다" in txt and "총역적" in txt
    # 집합 안이면 진단이 아니라 통과를 말한다
    ok = dict(rep, I_err_pct=5.0, gap_impulse_pp=0.0)
    assert "허용 집합 안" in verdict_text(admissibility(ok, ManufacturingLimits()))


def test_feasibility_bounds_catch_impossible_targets(motor):
    """물리 한계 초과는 **최적화 전에** 확정으로 잡힌다.

    최적화 뒤 판정(`admissibility`)은 '가장 가까운 설계' 가 어디서 멈췄는지에
    의존한다 — 실측에서 씨앗마다 걸리는 사양이 총역적/총역적/최대추력으로
    갈렸다. 목표와 케이스만 보는 이 검사는 경로가 끼어들지 않는다.
    """
    import torch as _t
    from gre.optimize.forward import resample
    from gre.optimize.loss import Target
    from gre.optimize.runner import feasibility_bounds

    a = _t.zeros(6); a[5] = 4.5
    with _t.no_grad():
        t, p, f, d = motor.simulate(_t.tensor(14.0), a, _t.zeros(6), [95.0],
                                    _t.tensor(90.0))
    g = _t.linspace(0.0, float(t[-1]), 240)
    base = resample(t, f, g).detach()
    lim = ManufacturingLimits()

    ok = feasibility_bounds(motor, Target(t=g, f=base), lim, lengths=(95.0,))
    assert ok["feasible"] and ok["I_over_pct"] < 0
    assert "물리 한계 안" in ok["note"]

    bad = feasibility_bounds(motor, Target(t=g, f=1.6 * base), lim, lengths=(95.0,))
    assert not bad["feasible"] and bad["I_over_pct"] > 0
    assert "물리 한계" in bad["note"] and "역설계를 돌릴 필요가 없" in bad["note"]
    # 상한은 장약 상한과 케이스 체적 중 작은 쪽이 정한다
    assert bad["m_max_g"] == pytest.approx(min(bad["m_geom_g"], lim.mass_max))


# ───────────────────────────────────── 예산 회계와 K=0 재확인
def _circle_target(motor, r0=12.0, at=85.0, L=95.0, n=200):
    import torch as _t

    from gre.optimize.forward import resample
    from gre.optimize.loss import Target
    with _t.no_grad():
        t, p, f, d = motor.simulate(_t.tensor(r0), _t.zeros(3), _t.zeros(3), [L],
                                    _t.tensor(at))
    g = _t.linspace(0.0, float(t[-1]), n)
    return Target(t=g, f=resample(t, f, g).detach(), p_max=float(p.max()))


def test_k_table_records_the_budget_it_spent(motor):
    """`k_table` 이 예산을 적어야 한다.

    적지 않으면 "그 K 로는 안 된다" 와 "그 K 에 예산이 모자랐다" 를 구분할 수
    없다. 힌지 + `stop_in_set` 아래에서 집합에 못 들어간 K 는 예산을 다 쓰므로
    `stop_reason == "반복 수 소진"` 은 부족의 증거가 **아니다.**
    """
    from gre.optimize.loss import LossWeights
    from gre.optimize.runner import minimal_complexity_search

    tgt = _circle_target(motor)
    dv, rep = minimal_complexity_search(motor, tgt, ManufacturingLimits(),
                                        LossWeights(), k_max=0, iters_per_k=6,
                                        lengths=(95.0,), verbose=False,
                                        throat=THROAT, seeding="warm")
    row = rep["k_table"][0]
    for key in ("iters_budget", "iters_used", "iters_cum", "stop_reason"):
        assert key in row, key
    assert row["iters_budget"] == 6
    assert 1 <= row["iters_used"] <= 6
    assert row["iters_cum"] == row["iters_used"]        # K=0 하나뿐이면 같다
    assert row["stop_reason"] in ("집합 진입", "손실 정체(램프 제외)",
                                 "반복 수 소진")


def test_k0_recheck_gives_the_circle_the_same_budget(motor):
    """원형이 만든 곡선은 K=0 재확인이 통과해야 한다.

    이것이 "K 가 올라가서 통과" 와 "예산이 늘어서 통과" 를 가르는 장치다.
    """
    from gre.optimize.loss import LossWeights
    from gre.optimize.runner import k0_recheck

    tgt = _circle_target(motor)
    dv, rep, info = k0_recheck(motor, tgt, ManufacturingLimits(), LossWeights(),
                               lengths=(95.0,), rounds=2, iters_per_k=40,
                               verbose=False, throat=THROAT)
    assert info["passed"], info
    assert float(dv.a.abs().max()) == 0.0 and float(dv.b.abs().max()) == 0.0
    assert info["rounds_budget"] == 2 and info["iters_budget"] == 80
    assert 1 <= info["iters_used"] <= 80
    assert len(info["segments"]) == info["rounds_run"] <= 2
    cum = [s["iters_cum"] for s in info["segments"]]
    assert cum == sorted(cum) and cum[-1] == info["iters_used"]


def test_recheck_overturns_a_false_positive_k(motor, monkeypatch):
    """K>0 이 예산 덕에 통과했다면 재확인이 판정을 되돌려야 한다.

    실측(6-로브 깊음)의 모양이다 — K=1 통과, 그런데 복잡도 0.01 mm. 실제 미수렴
    상황은 난수·예산에 의존하므로 여기서는 **배선**을 시험한다: K=0 을 한 번
    떨어뜨려 chosen>0 을 만든 뒤, 재확인이 통과를 보고하면 `k_chosen` 이 0 으로
    돌아가고 경고가 아니라 '거짓 양성' 이 기록되는가.
    """
    import gre.optimize.runner as R
    from gre.optimize.loss import LossWeights

    tgt = _circle_target(motor)
    orig_adm, orig_recheck = R.admissibility, R.k0_recheck
    seen, calls = {"n": 0}, {}

    def adm_fail_k0(rep, limits, tol=None):
        """K=0(모드가 0 에 얼려진 설계)만 떨어뜨려 chosen>0 을 만든다.

        `optimize` 가 매 반복 `admissibility` 를 부르므로
        '첫 호출' 로 셀 수 없다. 설계 내용(복잡도 0)으로 가른다.
        """
        out = orig_adm(rep, limits, tol)
        seen["n"] += 1
        if rep.get("complexity", 1.0) < 1e-9:
            out = dict(out, in_set=False, spec_ok=False, worst_gap_pp=7.5,
                       binding="총역적", gaps_pp={"총역적": 7.5})
        return out

    def recheck_passes(*a, **kw):
        calls.update(rounds=kw["rounds"], iters_per_k=kw["iters_per_k"])
        dv0, rep0, info = orig_recheck(*a, **kw)
        return dv0, rep0, dict(info, passed=True, worst_gap_pp=0.0, binding=None)

    monkeypatch.setattr(R, "admissibility", adm_fail_k0)
    monkeypatch.setattr(R, "k0_recheck", recheck_passes)
    dv, rep = R.minimal_complexity_search(motor, tgt, ManufacturingLimits(),
                                          LossWeights(), k_max=1, iters_per_k=5,
                                          lengths=(95.0,), verbose=False,
                                          throat=THROAT, seeding="warm")

    assert calls == {"rounds": 2, "iters_per_k": 5}     # 누적 예산을 그대로 준다
    assert rep["k_chosen"] == 0, rep["note"]
    assert "거짓 양성" in rep["note"] and "예산" in rep["note"]
    rows = [r for r in rep["k_table"] if r.get("recheck")]
    assert len(rows) == 1 and rows[0]["passed"] and rows[0]["K"] == 0
    assert dv is not None and float(dv.a.abs().max()) == 0.0


def test_recheck_inherits_the_k0_round_it_already_ran(motor, monkeypatch):
    """재확인은 본 루프의 K=0 실행을 **물려받아야** 한다.

    본 루프의 K=0 은 결정적이므로 재확인에서 다시 돌리면 결과가 글자 하나까지
    같다 (실측: 모양 6.16 %, 초과 9.5 %p 재현). 그 라운드를 버리면 통과 사례마다
    `iters_per_k` 만큼을 공짜로 태운다 — 실측에서 789 s 중 230 s 였다.
    """
    import gre.optimize.runner as R
    from gre.optimize.loss import LossWeights

    tgt = _circle_target(motor)
    orig_adm, seen, got = R.admissibility, {"n": 0}, {}

    def adm_fail_k0(rep, limits, tol=None):
        """위 테스트와 같은 이유로 설계 내용(복잡도 0)으로 가른다."""
        out = orig_adm(rep, limits, tol)
        seen["n"] += 1
        if rep.get("complexity", 1.0) < 1e-9:
            out = dict(out, in_set=False, spec_ok=False, worst_gap_pp=7.5,
                       binding="총역적", gaps_pp={"총역적": 7.5})
        return out

    def spy(m, t, lim, w, **kw):
        got.update(kw)
        return None, dict(rms_shape_pct=1.0, I_err_pct=1.0, F_err_pct=1.0,
                          complexity=0.0,
                          admissibility={"man_ok": True}), \
               {"passed": False, "rounds_run": 1, "rounds_budget": kw["rounds"],
                "rounds_inherited": kw["rounds_done"],
                "iters_used": kw["iters_done"] + 3,
                "iters_inherited": kw["iters_done"],
                "iters_budget": kw["rounds"] * kw["iters_per_k"],
                "worst_gap_pp": 4.0, "binding": "총역적",
                "segments": [{"stop_reason": "손실 정체(램프 제외)"}]}

    monkeypatch.setattr(R, "admissibility", adm_fail_k0)
    monkeypatch.setattr(R, "k0_recheck", spy)
    dv, rep = R.minimal_complexity_search(motor, tgt, ManufacturingLimits(),
                                          LossWeights(), k_max=1, iters_per_k=5,
                                          lengths=(95.0,), verbose=False,
                                          throat=THROAT, seeding="warm")
    k0 = next(r for r in rep["k_table"] if r["K"] == 0 and not r.get("recheck"))
    assert got["rounds_done"] == 1                     # 0 번째 라운드를 건너뛴다
    assert got["iters_done"] == k0["iters_used"]       # 쓴 반복을 그대로 물려준다
    # 끝점(r0, A_t)을 넘겨 이어간다 — 새 초기값에서 다시 시작하지 않는다
    assert np.isfinite(got["r0_init"]) and np.isfinite(got["at_init"])
    # 재확인이 실패하면 K>0 판정이 확정되고 근거가 기록된다
    assert rep["k_chosen"] == 1 and "예산 재확인 통과" in rep["note"]


# ───────────────────────── 되돌릴 반복은 집합 안에서 고른다
def test_restored_iterate_is_in_set_when_one_was_visited(motor):
    """집합 안 반복을 방문했으면 **그것을** 돌려줘야 한다.

    초판은 `argmin loss` 를 돌려줬다. 힌지에서 집합 안은 사양 항이 0 이므로 손실이
    2차 기준(최대압·희소성)만 남고, 집합 **밖** 반복이 사양 초과는 작으면서 2차
    기준이 좋으면 손실이 더 낮다. 실측(6-로브, K=1 웜 스타트):

        반복 18 (집합 안) 손실 1.062e-03  I  9.93  F  0.04
        반복  2 (집합 밖) 손실 6.945e-04  I 10.51  F 11.19   ← 초판이 고른 쪽

    즉 가중치 1e-3 의 가장 약한 기준이 합격 판정을 이겼다.
    """
    from gre.optimize.loss import LossWeights, Tolerances
    from gre.optimize.runner import DesignVars, final_report, optimize

    tgt = _circle_target(motor)
    mk = lambda v: torch.tensor(v, dtype=torch.float64, requires_grad=True)
    dv = DesignVars(r0=mk(12.6), a=mk(np.zeros(2)), b=mk(np.zeros(2)),
                    a_throat=mk(88.0), lengths=[95.0])
    dv, hist = optimize(motor, dv, tgt, ManufacturingLimits(), LossWeights(),
                        iters=40, verbose=False, tol=Tolerances(), throat=THROAT)
    rep, _ = final_report(motor, dv, tgt, ManufacturingLimits(), LossWeights(),
                          tol=Tolerances())
    visited = [h for h in hist if h["in_set"]]
    if not visited:
        pytest.skip("이 설정에서 집합에 들어가지 못했다 — 규칙을 시험할 수 없다")

    # 1) 돌려준 설계가 집합 안이다
    assert rep["admissibility"]["in_set"], rep["admissibility"]["gaps_pp"]
    # 2) 되돌린 반복이 집합 안 반복이다
    r_it = hist[-1]["restored_iter"]
    assert hist[-1]["restored_from"] == "집합 안"
    assert hist[r_it]["in_set"]
    # 3) 그 반복의 2차 기준이 집합 안 반복 중 최소다
    assert hist[r_it]["secondary"] == pytest.approx(
        min(h["secondary"] for h in visited))
    # 4) ⚠ 손실 최소 반복과 **다를 수 있다** — 그게 이 수정의 요점이다.
    #
    #    ⚠⚠ 초판은 `assert hist[i_loss]["in_set"] or i_loss != r_it` 였다.
    #        바로 위에서 `hist[r_it]["in_set"]` 을 이미 세웠으므로, 집합 밖인
    #        `i_loss` 는 필연적으로 `r_it` 과 다르다 — **구조적으로 참이고
    #        아무것도 고정하지 않았다**.
    #
    #    실제로 고정해야 하는 것: 되돌린 반복이 **집합 안 후보 중에서** 골라졌고,
    #    손실 최소 반복이 집합 밖이면 그것을 **고르지 않았다**는 것.
    i_loss = min(range(len(hist)), key=lambda i: hist[i]["loss"])
    assert hist[r_it]["in_set"], "되돌린 반복이 집합 안이어야 한다"
    if not hist[i_loss]["in_set"]:
        assert r_it != i_loss, "집합 밖인 손실 최소 반복을 되돌렸다 — 이 수정의 반대"
        # 그리고 그 경우가 이 시험에서 **실제로 일어나는지** 기록한다. 안 일어나면
        # 이 시험은 수정의 요점을 지나간 것이므로 그 사실이 보여야 한다.
        assert hist[i_loss]["loss"] <= hist[r_it]["loss"] + 1e-12


def test_loop_stop_test_equals_admissibility(motor):
    """루프 내 집합 판정이 `admissibility` 와 **같아야** 한다.

    초판의 루프 판정은 사양·벌점·케이스압만 보고 웹 두께·꼭짓점 반경·장약 질량·
    연결성을 빠뜨렸다. 그러면 `admissibility` 가 떨어뜨릴 설계에서 "집합 진입" 으로
    멈출 수 있다 — `stop_reason` 과 최종 판정이 어긋나는 또 다른 경로다.
    """
    from gre.optimize.loss import LossWeights, Tolerances
    from gre.optimize.runner import (DesignVars, admissibility, final_report,
                                     optimize)

    tgt = _circle_target(motor)
    mk = lambda v: torch.tensor(v, dtype=torch.float64, requires_grad=True)
    dv = DesignVars(r0=mk(12.6), a=mk(np.zeros(2)), b=mk(np.zeros(2)),
                    a_throat=mk(88.0), lengths=[95.0])
    lim, w, tol = ManufacturingLimits(), LossWeights(), Tolerances()
    dv, hist = optimize(motor, dv, tgt, lim, w, iters=30, verbose=False, tol=tol,
                        throat=THROAT)

    # 모든 반복에서 기록된 in_set 이 admissibility 와 일치한다 (연결성 포함)
    #
    # ⚠ 이것만으로는 **동어반복이다** — 루프가 `admissibility` 의 결과를 그대로
    #   저장하므로 양쪽이 같이 움직인다. `admissibility` 가
    #   행 하나를 빠뜨려도 두 값이 같이 틀린다.
    for h in hist:
        assert "components" in h            # 루프 안에서 실제로 계산했다
        assert h["in_set"] == admissibility(h, lim, tol)["in_set"]

    # 그래서 **행 목록 자체**를 고정한다. docstring 이 "웹 두께·꼭짓점 반경·
    # 장약 질량·연결성을 빠뜨렸다" 고 적었으니 그 네 개가 실제로 있어야 한다.
    rows = admissibility(hist[-1], lim, tol)["manufacturing"]
    for need in ("웹 두께", "꼭짓점 반경", "장약 질량", "연결성", "케이스압"):
        assert need in rows, f"제작 게이트에서 '{need}' 가 빠졌다"
    # 그리고 그 행들이 **실제로 판정에 쓰인다** — 하나를 위반시키면 집합 밖이 된다.
    h_bad = dict(hist[-1])
    h_bad["web_mm"] = lim.w_web_min - 1.0        # 웹을 한계 아래로
    assert not admissibility(h_bad, lim, tol)["in_set"], "웹 행이 판정에 안 쓰인다"
    assert "웹 두께" in admissibility(h_bad, lim, tol)["violated_constraints"]
    h_bad2 = dict(hist[-1])
    h_bad2["components"] = 3                     # 포트가 쪼개졌다
    assert not admissibility(h_bad2, lim, tol)["in_set"], "연결성 행이 안 쓰인다"
    # "집합 진입" 으로 멈췄으면 최종 판정도 집합 안이다 — 어긋나지 않는다
    if hist[-1].get("stop_reason") == "집합 진입":
        rep, _ = final_report(motor, dv, tgt, lim, w, tol=tol)
        assert rep["admissibility"]["in_set"]


# ══════════════════════════ 압력 곡선 선택 입력
def _thrust_and_pressure(motor, r0=14.0, a6=4.5, at=90.0, L=95.0, n=320):
    """같은 시뮬에서 나온 **자기 일관된** 추력·압력 쌍."""
    import torch as _t

    from gre.optimize.forward import resample
    a = _t.zeros(6)
    a[5] = a6
    with _t.no_grad():
        t, p, f, d = motor.simulate(_t.tensor(r0), a, _t.zeros(6), [L],
                                    _t.tensor(at))
    g = _t.linspace(0.0, float(t[-1]), n)
    return g, resample(t, f, g).detach(), resample(t, p, g).detach()


def test_pressure_curve_promotes_pressure_to_a_spec(motor):
    """압력 곡선이 들어오면 사양이 3개 → 5개가 된다.

    곡선이 없으면 압력은 가중치 1e-3 의 소프트 2차 기준이라, 추력이 `P·A_t` 의
    곱만 묶는 탓에 **자유 방향**이다 (실측: 같은 집합 안에서 최대압 2.64 배).
    곡선을 주면 그 자유도를 사양으로 묶는다.
    """
    from gre.optimize.loss import LossWeights, Target, Tolerances
    from gre.optimize.runner import DesignVars, final_report

    g, fr, pr = _thrust_and_pressure(motor)
    mk = lambda v: torch.tensor(v, dtype=torch.float64, requires_grad=True)
    dv = DesignVars(r0=mk(12.0), a=mk(np.zeros(2)), b=mk(np.zeros(2)),
                    a_throat=mk(85.0), lengths=[95.0])
    lim, w, tol = ManufacturingLimits(), LossWeights(), Tolerances()

    t3 = Target(t=g, f=fr, p_max=float(pr.max()))
    t5 = Target(t=g, f=fr, p=pr)
    assert not t3.pressure_is_spec and t5.pressure_is_spec
    assert t5.p_max == pytest.approx(float(pr.max()))      # 곡선에서 자동 산출

    r3, _ = final_report(motor, dv, t3, lim, w, tol=tol)
    r5, _ = final_report(motor, dv, t5, lim, w, tol=tol)
    assert list(r3["admissibility"]["spec"]) == ["모양", "총역적", "최대추력"]
    assert list(r5["admissibility"]["spec"]) == ["모양", "총역적", "최대추력",
                                                 "압력 모양", "최대압"]
    # 압력 항은 **곡선이 있을 때만** 보고된다 — 없을 때 빈 주장을 하지 않는다
    assert "gap_p_shape_pp" not in r3 and "gap_p_max_pp" not in r3
    assert "gap_p_shape_pp" in r5 and "gap_p_max_pp" in r5
    assert r3["tol_pct"].keys() != r5["tol_pct"].keys()
    assert r5["tol_pct"]["최대압"] == pytest.approx(10.0)
    # 같은 설계인데 압력 사양이 걸리면 집합까지의 거리가 줄어들 수 없다
    assert (r5["admissibility"]["worst_gap_pp"]
            >= r3["admissibility"]["worst_gap_pp"] - 1e-9)


def test_soft_pressure_term_is_off_when_pressure_is_a_spec(motor):
    """압력이 사양이면 소프트 항과 2차 기준에서 빠져야 한다.

    남겨 두면 사양 밖 기준이 집합 안 선택에 다시 끼어든다 — 이전의 오판과 같은
    구조다 (가중치 1e-3 이 합격 판정을 이겼다).
    """
    from gre.optimize.loss import (LossWeights, Target, Tolerances, curve_loss,
                                   shape_window)
    from gre.optimize.runner import DesignVars, evaluate, optimize

    g, fr, pr = _thrust_and_pressure(motor)
    mk = lambda v: torch.tensor(v, dtype=torch.float64, requires_grad=True)
    lim, w, tol = ManufacturingLimits(), LossWeights(), Tolerances()
    for tgt, want in ((Target(t=g, f=fr, p_max=float(pr.max())), True),
                      (Target(t=g, f=fr, p=pr), False)):
        dv = DesignVars(r0=mk(12.0), a=mk(np.zeros(2)), b=mk(np.zeros(2)),
                        a_throat=mk(85.0), lengths=[95.0])
        _, _, aux = evaluate(motor, dv, tgt, lim, w, tol=tol)
        t, p, f, diag, omega = aux
        _, terms, _ = curve_loss(t, f, p, omega, tgt, w, tol)
        assert ("p_max" in terms) is want, list(terms)
        assert ("p_shape" in terms) is (not want)
        assert ("p_peak" in terms) is (not want)
        # 2차 기준에서도 빠진다
        dv2, hist = optimize(motor, dv, tgt, lim, w, iters=3, verbose=False,
                             tol=tol, throat=THROAT)
        sec = hist[0]["secondary"]
        sparse_only = (w.sparsity * hist[0]["complexity"]
                       / float(motor.ds.r_outer))
        if want:                                   # 소프트 — 압력 항이 들어있다
            assert sec > sparse_only
        else:                                      # 사양 — 희소성만 남는다
            assert sec == pytest.approx(sparse_only)


def test_two_input_curves_must_not_contradict_each_other(motor):
    """추력·압력이 한 모터에서 나올 수 있는지 미리 본다.

    `A_t` 는 시간에 고정이므로 **함의 노즐목** `A_t = (F/P)/C_f(P, ε)` 가
    상수여야 한다. 실측 (`fp_ratio.py`, 설계 8 개, 재측정): 최대
    **2.13 %**. 문턱 10 % 는 그 4 배다.

    ⚠ 자기 일관된 쌍이 통과하는지가 이 검사의 **1번 조건**이다. 판 둘이 여기서
      실패했다 — 창을 `f > 0.05·F_max` 로 잡은 판은 같은 시뮬에서 나온 쌍조차
      64.5 % 로 거부했고, 준위 마스크 판은 반대로 부스트 압력만 2 배 틀린 쌍을
      0.00 % 로 통과시켰다.

    ⚠ 이 시험이 단조 감소 쌍만 쓰는 것이 **그 두 번째 실패를 놓친 이유**였다. 그래서 아래에 부스트-서스테인 쌍을 따로 넣는다.
    """
    from gre.optimize.loss import Target
    from gre.optimize.runner import AT_SPREAD_MAX, feasibility_bounds

    g, fr, pr = _thrust_and_pressure(motor)
    lim = ManufacturingLimits()

    ok = feasibility_bounds(motor, Target(t=g, f=fr, p=pr), lim, lengths=(95.0,))
    assert ok["fp_ratio_checked"] and ok["fp_consistent"] and ok["feasible"]
    assert ok["at_spread_pct"] < 100 * AT_SPREAD_MAX
    assert 60.0 < ok["at_implied_mean"] < 200.0    # 함의 A_t, 참 A_t = 90
    assert 60.0 < ok["fp_ratio_mean"] < 200.0      # F/P = C_f·A_t (단위 검사용)
    # ⚠ **옛 추정자는 지금 창에서 이 설계를 거부한다.** `F/P` 는 C_f 의 압력
    #   의존만큼 정당하게 변하므로, 창을 구간으로 고치자 최악이 20.29 % 가 되어
    #   옛 문턱 20 % 를 넘었다. 추정자를 바꾼 이유가 이것이다 — 그 사실을 고정한다.
    assert ok["fp_spread_pct"] > ok["at_spread_pct"] * 3.0

    # 압력 곡선이 없으면 이 검사는 돌지 않는다
    none = feasibility_bounds(motor, Target(t=g, f=fr), lim, lengths=(95.0,))
    assert not none["fp_ratio_checked"]

    # 시간에 따라 비튼 압력 — 한 모터에서 나올 수 없다
    skew = pr * (1.0 + 1.5 * torch.linspace(0.0, 1.0, len(g)))
    bad = feasibility_bounds(motor, Target(t=g, f=fr, p=skew), lim, lengths=(95.0,))
    assert not bad["fp_consistent"] and not bad["feasible"]
    assert "모순" in bad["note"]

    # 단위 오류 — psi (×145) 는 함의 노즐목이 1 mm² 미만이 되어 걸린다
    psi = feasibility_bounds(motor, Target(t=g, f=fr, p=pr * 145.0), lim,
                             lengths=(95.0,))
    assert not psi["feasible"] and "케이스" in psi["note"]

    # bar (×10) 는 F/P 변동폭으로는 **안 걸린다** (배율에 둔감하다).
    # 케이스 허용압 초과로 걸린다 — 그 경로가 실제로 작동하는지 확인한다.
    bar = feasibility_bounds(motor, Target(t=g, f=fr, p=pr * 10.0), lim,
                             lengths=(95.0,))
    assert not bar["feasible"] and bar.get("P_feasible") is False
    assert "터집니다" in bar["note"]


def test_contradiction_check_sees_a_boost_sustain_pair(motor):
    """**부스트-서스테인** 쌍에서도 모순을 잡는가 — 전면 검토 회귀 시험.

    창이 준위 마스크였을 때는 평탄값이 높은 단에서 잡혀 **낮은 단을 통째로
    버렸다.** 그래서 부스트 압력만 2 배 틀린 쌍(함의 노즐목이 절반이 되는, 같은
    모터에서 나올 수 없는 쌍)을 변동 **0.00 %** 로 통과시켰다. 창이 검사받는 그
    데이터에서 나오므로 창을 끌고 가는 오류는 스스로를 숨겼다.

    §4.6.2 가 부스트-서스테인을 입력 템플릿으로 명시하므로 가정이 아니라 실제
    입력이다. 이 시험이 그 결함의 재발을 막는다.
    """
    import numpy as np

    from gre.core.nozzle import optimum_area_ratio, thrust_coefficient
    from gre.optimize.loss import fp_ratio_spread

    t = np.linspace(0.0, 2.0, 2000)
    pp = np.where(t < 0.5, 5.0, 1.5)
    pp[0] = pp[-1] = 0.2
    eps = optimum_area_ratio(3.0, motor.p_ambient, motor.gamma)
    cf = np.array([thrust_coefficient(max(x, 0.15), motor.p_ambient, eps,
                                      motor.gamma) for x in pp])
    ff = cf * pp * 80.0
    ff[0] = ff[-1] = 0.0
    kw = dict(gamma=motor.gamma, p_ambient=motor.p_ambient)
    T = torch.tensor

    # 자기 일관된 쌍 — 참 ε 를 주면 정의상 0 이어야 한다 (추정자 검산)
    exact = fp_ratio_spread(T(t), T(ff), T(pp), eps=eps, **kw)
    assert exact["spread"] == pytest.approx(0.0, abs=1e-9)
    assert exact["at_mean"] == pytest.approx(80.0, rel=1e-6)

    # 자동 ε 에서도 통과해야 한다 — 잔차는 ε 불일치에서만 온다
    auto = fp_ratio_spread(T(t), T(ff), T(pp), **kw)
    assert auto["spread"] < 0.05, f"자기 일관된 쌍이 {100*auto['spread']:.2f} %"
    # 창이 **구간**이라 두 단 전부를 본다 (마스크였다면 1/4 만 봤다)
    assert auto["n"] > 0.95 * t.size

    # 부스트 압력만 2 배 — 함의 노즐목이 절반이 되는 모순된 쌍
    p_bad = pp.copy()
    p_bad[t < 0.5] *= 2.0
    p_bad[0] = 0.2
    bad = fp_ratio_spread(T(t), T(ff), T(p_bad), **kw)
    assert bad["spread"] > 0.30, "부스트만 틀린 쌍을 놓쳤다 — 준위 마스크 결함 재발"
    # 서스테인 쪽만 틀린 쌍도 (이쪽은 옛 판도 잡았다)
    p_bad2 = pp.copy()
    p_bad2[t >= 0.5] *= 2.0
    p_bad2[-1] = 0.2
    assert fp_ratio_spread(T(t), T(ff), T(p_bad2), **kw)["spread"] > 0.30


def test_pressure_curve_input_is_validated(motor):
    """잘못된 압력 곡선은 Target 생성에서 막는다."""
    from gre.optimize.loss import Target

    g, fr, pr = _thrust_and_pressure(motor)
    with pytest.raises(ValueError, match="길이가 시간축과 다르다"):
        Target(t=g, f=fr, p=pr[:-5])
    with pytest.raises(ValueError, match="절대압"):
        Target(t=g, f=fr, p=pr - float(pr.min()))      # 게이지압처럼 0 이 생긴다


# ═══════════════════════════════════════════════════ 진폭 유의성 문턱
def test_amplitude_floor_takes_the_higher_of_two_floors():
    """모드 진폭의 바닥은 **둘**이고 높은 쪽이 판정한다.

        제작  DELTA_TRANSFER_MM = 0.020 mm        — 시편에서 확정
        모델  MODEL_FLOOR_K · eps_cells · h       — 격자 수렴에서 측정

    ⚠ **판정하는 쪽은 모델이다.** 운용 설정(h=0.5, eps_cells=1.5)에서 모델
       바닥은 0.60 mm — δ 의 30 배다. 이 시험이 그 사실을 고정한다.
    """
    from gre.optimize.runner import (DELTA_TRANSFER_MM, MODEL_FLOOR_K,
                                     amplitude_floor)

    assert DELTA_TRANSFER_MM == 0.020

    # 운용 설정 — 모델이 판정한다
    floor, which = amplitude_floor(14.0, h=0.5, eps_cells=1.5)
    assert floor == pytest.approx(MODEL_FLOOR_K * 1.5 * 0.5)
    assert floor == pytest.approx(0.60)
    assert which == "모델"
    # δ 의 30 배다 — 제작 정밀도가 판정할 여지가 없다는 사실을 숫자로 둔다
    assert floor > 25 * DELTA_TRANSFER_MM

    # 두 판정의 여유 — §4.4 의 숫자
    assert 0.428 < floor          # 6-로브 깊음 K=3 조차 바닥 아래다
    assert 0.020 < floor


def test_model_floor_scales_with_relaxation_width_not_r0():
    """고친 식이 **이름에 있는 변수**를 실제로 쓴다.

    옛 식 `max(0.01·r₀, 0.05)` 은 "격자 분해능" 이라면서 격자가 없었다. 그래서
    이전 판의 "격자를 조이면 δ 가 판정하기 시작한다" 가 구조적으로 성립할 수
    없었다 — 전면 검토의 [높음] 지적이다. 이 시험이 재발을 막는다.
    """
    from gre.optimize.runner import DELTA_TRANSFER_MM, amplitude_floor

    # r₀ 는 (격자를 줄 때) 바닥에 영향이 없다
    base, _ = amplitude_floor(14.0, h=0.5, eps_cells=1.5)
    for r0 in (1.0, 5.0, 14.0, 30.0, 300.0):
        assert amplitude_floor(r0, h=0.5, eps_cells=1.5)[0] == pytest.approx(base)

    # h 와 eps_cells 둘 다 바닥을 움직인다 — 선형으로
    assert amplitude_floor(14.0, h=0.25, eps_cells=1.5)[0] == pytest.approx(base / 2)
    assert amplitude_floor(14.0, h=0.5, eps_cells=3.0)[0] == pytest.approx(base * 2)

    # ⚠ **바닥은 하나다 — δ 는 판정에서 뺐다.** 격자를 아무리 조여도
    #   판정자는 "모델" 이고, 바닥은 δ 아래로도 내려간다. δ 의 지배항이
    #   `export/profile.py` 의 렌더링 공차가 만든 값이어서(자기충족적) 판정
    #   문턱으로 쓸 수 없다는 것이 이유다.
    from gre.optimize.forward import DiffMotor
    deep = amplitude_floor(14.0, eps_cells=1.5,
                           h=float(DiffMotor(d_outer=60.0, n_grid=7201).ds.h))
    assert deep[1] == "모델"
    assert deep[0] < DELTA_TRANSFER_MM, "바닥이 δ 에서 멈추면 안 된다"
    coarse = amplitude_floor(14.0, eps_cells=1.5,
                             h=float(DiffMotor(d_outer=60.0, n_grid=1801).ds.h))
    assert coarse[1] == "모델" and coarse[0] == pytest.approx(0.04)
    # δ 는 상수로 남아 있지만 이 함수가 더 이상 참조하지 않는다 — 그 사실을
    # 고정한다 (옛 구조로 되돌리면 깨진다).
    import inspect
    assert "DELTA_TRANSFER_MM" not in inspect.getsource(amplitude_floor)

    # 격자를 안 넘기면 옛 휴리스틱 + 꼬리표. 보고서에 그 사실이 남아야 한다.
    old, tag = amplitude_floor(14.0)
    assert old == pytest.approx(0.14) and tag == "모델(휴리스틱)"
