"""ODE 판정자 — 바닥이 실제로 낮은지, 그리고 물리가 `DiffMotor` 와 같은지."""
import numpy as np
import pytest
import torch

from gre.optimize.constraints import ThroatSpec
from gre.optimize.forward import DiffMotor
from gre.validate.ode_judge import OdeCircleJudge, shape_window

torch.set_default_dtype(torch.float64)
R_OUT = 30.0


@pytest.fixture(scope="module")
def motor():
    return DiffMotor(d_outer=2 * R_OUT, n_grid=121, n_samples=384, n_w=200,
                     h_max=10e-3)


@pytest.fixture(scope="module")
def judge():
    """순수 BATES 판정자 — 기하 이산화를 **안 맞춘** 비교 기준."""
    return OdeCircleJudge(r_outer=R_OUT)


@pytest.fixture(scope="module")
def judge_matched(motor):
    """ε 완화 폭과 차분 스텐실까지 맞춘 판정자 — **운용 경로**가 쓰는 것.

    `circle_screen` 이 쓰는 것이 이쪽이다. 판정자의 질문이 "최적화기가 쓰는 그
    모델에서 원형이 이 곡선을 낼 수 있는가" 이므로 기하까지 같아야 답이 맞는다.
    """
    return OdeCircleJudge.from_motor(motor)


def _curve(motor, r0, length, at, mode=0, amp=0.0):
    a = torch.zeros(6)
    if mode:
        a[mode - 1] = amp
    with torch.no_grad():
        t, p, f, d = motor.simulate(torch.tensor(float(r0)), a, torch.zeros(6),
                                    [float(length)], torch.tensor(float(at)))
    return t.numpy(), f.numpy()


def test_same_physics_as_diffmotor(motor, judge):
    """적분기가 다를 뿐 같은 모터다 — 스칼라 지표가 맞아야 한다.

    곡선 전체 RMS 는 소진 부근에서 벌어진다. 거리장이 바깥 벽을 ε(1.5 셀)
    만큼 완화해 A_b 가 매끄럽게 줄어드는 반면 BATES 해석해는 w = R − r0 에서
    딱 끊기기 때문이다. 그건 **기하 이산화 차이**이지 챔버 물리 차이가 아니므로,
    여기서는 최대추력·총역적으로 본다.
    """
    for r0, L, at in ((11.0, 95.0, 80.0), (17.0, 140.0, 80.0)):
        t, f = _curve(motor, r0, L, at)
        tj, fj = judge.curves([r0], [at], L)
        assert abs(fj[0].max() - f.max()) / f.max() < 0.01
        i_ref, i_j = np.trapezoid(f, t), np.trapezoid(fj[0], tj)
        assert abs(i_j - i_ref) / i_ref < 0.02


def test_floor_is_small_without_any_window(motor, judge, judge_matched):
    """**실제 원형**을 넣으면 잔차가 작아야 한다 — 창을 씌우지 않고도.

    준정상 판정자는 같은 조건에서 5.5 % 가 나왔다. 점화 상승과 블로다운이
    모델에 없어서인데, ODE 판정자에는 둘 다 있으므로 그 바닥이 사라진다.

    ⚠ **운용 경로(ε 일치)로 잰다.** 기하를 안 맞춘 BATES 판정자로 재면 챔버
      물리의 바닥과 기하 이산화 차이가 섞인다. 소진 근처 둘레를 반 스텝
      중심차분으로 고친 뒤 두 숫자가 크게 벌어졌다:

          사례                 ε 일치 (운용)   BATES (기하 불일치)
          r0 11, L 95          0.20 %         3.32 %
          r0 17, L 140         0.41 %         5.71 %

      이전에는 ε 일치가 1.94 ~ 3.90 % 였다. 둘레 편향이 빠지면서 바닥이
      거의 사라졌다 — 준정상 5.46 ~ 6.72 % 와 비교하면 **10 배 이상** 낮다.
    """
    r0g = np.linspace(5.0, 25.0, 101)
    atg = np.geomspace(40.0, 250.0, 61)
    for r0, L, at in ((11.0, 95.0, 80.0), (17.0, 140.0, 80.0)):
        t, f = _curve(motor, r0, L, at)
        res = judge_matched.fit(t, f, L, r0g, atg)
        assert res["residual_pct"] < 1.0, (r0, L, at, res)
        assert res["suffices"]
        assert abs(res["r0"] - r0) < 0.8, res          # 참 설계를 회복한다
        # 기하를 안 맞추면 바닥이 올라간다 — 그 차이가 '비관 편향' 이다
        plain = judge.fit(t, f, L, r0g, atg)
        assert plain["residual_pct"] > res["residual_pct"]


def test_window_is_not_needed(motor, judge):
    """창을 씌워도 나아지지 않는다 — 자를 구간이 없기 때문이다."""
    r0g = np.linspace(5.0, 25.0, 101)
    atg = np.geomspace(40.0, 250.0, 61)
    t, f = _curve(motor, 11.0, 95.0, 80.0)
    full = judge.fit(t, f, 95.0, r0g, atg)["residual_pct"]
    win = judge.fit(t, f, 95.0, r0g, atg, mask=shape_window(f))["residual_pct"]
    assert abs(full - win) < 1.0


def test_non_circular_separates_from_floor(motor, judge_matched):
    """둘레를 많이 버는 비원형은 바닥보다 확실히 위에 있어야 한다.

    바닥이 0.2 % 수준으로 내려간 뒤에는 **차이(%p)보다 비(ratio)가 말이 된다** —
    실측(ε 일치): 원형 0.20 %, 6-로브 3.13 % → 차이 2.93 %p, 비 16 배.
    이전에는 원형 바닥 자체가 3.4 % 라 차이로 봐야 했다.
    """
    r0g = np.linspace(5.0, 25.0, 101)
    atg = np.geomspace(30.0, 250.0, 61)
    t_c, f_c = _curve(motor, 14.0, 95.0, 90.0)
    t_s, f_s = _curve(motor, 14.0, 95.0, 90.0, mode=6, amp=4.5)
    r_c = judge_matched.fit(t_c, f_c, 95.0, r0g, atg)["residual_pct"]
    r_s = judge_matched.fit(t_s, f_s, 95.0, r0g, atg)["residual_pct"]
    assert r_s > r_c + 2.0, (r_c, r_s)
    assert r_s > 5.0 * r_c, (r_c, r_s)


def test_fixed_throat_narrows_the_search(motor, judge):
    """A_t 를 입력으로 고정해도 원형은 여전히 통과해야 한다 (바닥이 안 오른다)."""
    r0g = np.linspace(5.0, 25.0, 101)
    t, f = _curve(motor, 11.0, 95.0, 80.0)
    free = judge.fit(t, f, 95.0, r0g, np.geomspace(40.0, 250.0, 61))
    fixed = judge.fit(t, f, 95.0, r0g, at_fixed=80.0)
    assert fixed["residual_pct"] < free["residual_pct"] + 1.0
    assert fixed["a_throat"] == pytest.approx(80.0)


def test_vectorised_matches_one_at_a_time(judge):
    """여러 설계를 한꺼번에 적분한 결과가 하나씩 적분한 것과 같아야 한다."""
    r0 = np.array([9.0, 13.0, 18.0])
    at = np.array([60.0, 90.0, 140.0])
    t_all, f_all = judge.curves(r0, at, 95.0)
    for i in range(3):
        t_one, f_one = judge.curves([r0[i]], [at[i]], 95.0)
        assert np.allclose(f_all[i], f_one[0], rtol=1e-12, atol=1e-9)


# ═══════════════════════════════════════════ circle_screen (권장 판정 경로)
def test_circle_screen_reports_both_criteria(motor):
    """노즐목 자유/고정 두 숫자를 같이 돌려주고, 그 차이가 흡수량이다."""
    from gre.validate.identifiability import circle_screen

    t, f = _curve(motor, 11.0, 95.0, 80.0)
    s = circle_screen(motor, t, f, 95.0, a_throat=80.0, n_r0=61, n_at=41)
    assert s.suffices                          # 실제 원형이므로 당연히 통과
    assert s.residual_free < 5.0
    assert s.a_throat_given == pytest.approx(80.0)
    assert s.throat_gap is not None
    assert s.residual_fixed >= s.residual_free - 1e-9   # 자유도가 적으면 더 못 맞춘다
    assert abs(s.r0_free - 11.0) < 1.0
    assert "원형 포트" in s.note


def test_circle_screen_actually_uses_the_eps_matched_judge(motor):
    """`circle_screen` 이 **`from_motor` 판정자를 쓰는지** 직접 고정한다.

    ⚠⚠ 초판(`test_circle_screen_matches_forward_geometry`)은 시험 안에서
        `from_motor` 판정자를 **따로 만들어** 그것의 ε·스텐실을 확인했다.
        `circle_screen` 이 무엇을 만들어 쓰는지는 보지 않았다 — 실측 확인:
        `identifiability.py` 의 `OdeCircleJudge.from_motor(motor)` 를 평범한
        `OdeCircleJudge(r_outer=...)` 로 바꿔도 **두 시험 모두 통과했다**. 그리고 `residual_free < 5.0` 은 문서화된 ε 일치
        바닥(0.20 %)의 25 배이고 불일치 바닥(3.32 %)보다도 커서, 어느 판정자를
        쓰든 통과한다.

    고친 방법: `circle_screen` 이 만드는 판정자를 **가로채서** 그것이
    `from_motor` 로 만들어졌는지 본다.
    """
    import gre.validate.identifiability as I
    from gre.validate.ode_judge import OdeCircleJudge

    # `identifiability` 는 함수 안에서 `from ..validate.ode_judge import
    # OdeCircleJudge` 를 하므로 모듈 속성이 아니라 **클래스 자체**를 가로챈다.
    made = {}
    orig = OdeCircleJudge.__dict__["from_motor"]
    raw = orig.__func__ if hasattr(orig, "__func__") else orig

    def spy(*args, **kw):
        j = raw(*args, **kw)
        made["judge"] = j
        return j

    wrapped = (classmethod(spy) if isinstance(orig, classmethod)
               else staticmethod(spy) if isinstance(orig, staticmethod)
               else spy)
    t, f = _curve(motor, 11.0, 95.0, 80.0)
    try:
        OdeCircleJudge.from_motor = wrapped
        I.circle_screen(motor, t, f, 95.0, a_throat=80.0, n_r0=21, n_at=11)
    finally:
        OdeCircleJudge.from_motor = orig
    assert "judge" in made, \
        "circle_screen 이 OdeCircleJudge.from_motor 를 쓰지 않는다 — 기하 불일치"
    j = made["judge"]
    assert j.eps == pytest.approx(float(motor.ds.eps))
    assert j.fd_h == pytest.approx(float(motor.ds.h))

    # 그리고 ε 일치 바닥이 **실제로 낮다** — 문서화된 0.2~0.4 % 대역
    s = I.circle_screen(motor, t, f, 95.0, a_throat=80.0, n_r0=61, n_at=41)
    assert s.residual_free < 1.0, (
        f"ε 일치 판정자의 바닥이 {s.residual_free:.2f} % — 1 % 를 넘으면 "
        f"기하가 안 맞는 것이다 (불일치 바닥은 3.3 % 대)")


def test_from_motor_inherits_the_forward_geometry(motor):
    """`from_motor` 자체가 ε·스텐실·외경을 물려받는지 (단위 시험)."""
    from gre.validate.ode_judge import OdeCircleJudge

    j = OdeCircleJudge.from_motor(motor)
    assert j.eps == pytest.approx(float(motor.ds.eps))
    assert j.fd_h == pytest.approx(float(motor.ds.h))
    assert j.r_outer == pytest.approx(float(motor.ds.r_outer))
    # ε 를 맞추면 구간 내부 A_b 가 거리장과 0.5 % 안에서 맞는다
    w = np.linspace(0.0, 0.8 * (30.0 - 11.0), 60)
    ab_j = j._geom(w, np.full_like(w, 11.0), 95.0)[0]
    with torch.no_grad():
        phi = motor.ds.phi(torch.tensor(11.0), torch.zeros(6), torch.zeros(6))
        ab_m = motor.ds.burn_area(phi, torch.tensor(w), 95.0, True).numpy()
    assert np.max(np.abs(ab_j - ab_m)) / ab_m.max() < 0.005


def test_eps_matched_judge_recovers_a_known_circle_better(motor):
    """비관 편향 검사 — 기하를 맞춘 판정자가 **아는 답**에 더 가깝다.

    초판은 K 상승 탐색이 낸 `r0 = 12.29` 를 상수로 박고 "거기에 더
    가깝다" 를 단언했다. 그 상수는 **그때의 설정에서만 유효한 값**이었고,
    `n_samples` 가 격자에 묶이면서(384 → 768) 두 판정자가 11.9 / 11.8
    로 바뀌어 순서가 뒤집혔다 — `r0` 격자 한 칸(0.1 mm) 안에서 벌어진 일이라
    애초에 그 자(尺)로 가릴 수 있는 차이가 아니었다.

    고친 방식: 판정 대상을 **참값을 아는 곡선**으로 바꾼다. 같은 `DiffMotor`
    에서 **실제 원형** 포트가 낸 곡선을 주면 정답이 그 `r0` 다. 판정자의
    주장("최적화기가 쓰는 기하와 맞추면 더 정확하다")을 스타 설계의 대리
    상수가 아니라 **정답**에 대고 잰다.

    깨져야 하는 변경:
      · `from_motor` 가 ε·스텐실·`r_outer` 중 하나라도 안 물려받는 경우
      · 두 판정자가 `r0` 격자 한 칸 안으로 같아지는 경우 (마지막 단언) —
        그러면 이 시험은 아무것도 고정하지 못하므로 자를 더 가늘게 해야 한다
    """
    from gre.validate.ode_judge import OdeCircleJudge

    r0g = np.linspace(5.0, 25.0, 201)                 # 0.1 mm 격자
    atg = np.geomspace(30.0, 250.0, 51)
    step = float(r0g[1] - r0g[0])
    # ⚠ **세 점에서 본다.** 한 점만 보면 계통 차이와 우연을 구별할 수 없다.
    #   `r0` 복원은 둘 다 격자 한 칸 안에서 맞으므로 거기서는 안 갈린다 —
    #   차이는 **잔차**에서 난다. 실측 (`n_samples` 자동 상향 후):
    #       r0 9.0  : BATES 3.502 %  ε일치 0.182 %  (19.2 배)
    #       r0 12.3 : BATES 3.532 %  ε일치 0.298 %  (11.9 배)
    #       r0 16.0 : BATES 3.348 %  ε일치 0.296 %  (11.3 배)
    for r0_true, at_true in ((9.0, 60.0), (12.3, 81.6), (16.0, 110.0)):
        t, f = _curve(motor, r0_true, 95.0, at_true)   # 진짜 원형 — 정답을 안다
        plain = OdeCircleJudge(r_outer=30.0).fit(t, f, 95.0, r0g, atg)
        matched = OdeCircleJudge.from_motor(motor).fit(t, f, 95.0, r0g, atg)

        # 둘 다 `r0` 는 맞혀야 한다 (판정자가 깨졌는지 보는 최소 조건)
        for tag, res in (("BATES", plain), ("ε일치", matched)):
            assert abs(res["r0"] - r0_true) <= 2 * step + 1e-6, (
                f"{tag} 가 r0 를 못 맞힌다: {res['r0']:.2f} vs {r0_true}")
        # 그리고 기하를 맞춘 쪽은 **곡선을 재현**해야 한다 — 여기서 갈린다
        assert matched["residual_pct"] < 1.0, (
            f"r0={r0_true}: ε 일치 판정자가 자기 모터의 원형 곡선을 "
            f"{matched['residual_pct']:.3f} % 로밖에 재현 못 한다 — "
            f"`from_motor` 가 기하를 물려받지 못한 것이다")
        assert plain["residual_pct"] > 2.0, (
            f"r0={r0_true}: BATES 판정자 잔차가 {plain['residual_pct']:.3f} % 로 "
            f"작다 — 두 판정자가 같아졌다는 뜻이므로 이 시험이 고정하는 것이 "
            f"없어진다")
        assert plain["residual_pct"] / matched["residual_pct"] > 5.0


# ═══════════════════════════ shape_identifiability (양측 측정, def test_shape_identifiability_plumbing(motor):
    """해집합 지름 측정의 배선 — 예산을 최소로 줄여 형식만 확인한다.

    실제 판정에는 `n_starts >= 6, iters >= 90` 이 필요하다 (출발점 하나당 수십 초).
    여기서는 반환 형식과 등급 로직만 지킨다.
    """
    from gre.optimize.forward import resample
    from gre.optimize.loss import Target
    from gre.validate.identifiability import (ShapeIdentifiability,
                                              shape_identifiability)

    t, f = _curve(motor, 11.0, 95.0, 78.0, mode=3, amp=2.2)
    tg = torch.linspace(0.0, float(t[-1]), 200)
    target = Target(t=tg, f=resample(torch.tensor(t), torch.tensor(f), tg))
    out = shape_identifiability(motor, target, n_starts=2, n_modes=2,
                                lengths=(95.0,), iters=6, verbose=False,
                                throat=ThroatSpec.fixed(
                                    2.0 * np.sqrt(78.0 / np.pi)))
    assert isinstance(out, ShapeIdentifiability)
    assert out.n_starts == 2
    assert 0 <= out.n_accepted <= 2
    assert out.grade in ("높음", "보통", "낮음", "판정 불가")
    assert isinstance(out.warnings(), list)
    assert isinstance(out.note, str)
    if out.n_accepted < 2:                     # 흩어짐을 못 재면 그렇게 말해야 한다
        assert out.spread_mm is None
        assert out.grade == "판정 불가"
        assert any("표본" in w or "해가 없다" in w for w in out.warnings())


def test_identifiability_grading_bands():
    """등급 경계 — **구현을 불러서** 확인한다.

    ⚠⚠ 초판은 등급 루프를 **시험 본문에 다시 구현해** 자기 출력을 단언했다.
        그래서 `identifiability.py` 의 실제 채점 코드(528~532 행)는 **한 번도
        실행되지 않았다** — 비교 부등호가 틀리거나 `break` 가 빠져도 통과한다.

    고친 방법: 경계값을 `BANDS` 에서 읽어 **구현 함수에 넣고** 결과를 본다.
    경계 바로 아래·위를 같이 넣어 부등호 방향(`<` vs `<=`)까지 고정한다.
    """
    from gre.validate.identifiability import (ShapeIdentifiability,
                                              _grade_of_spread_frac)

    (t1, g1), (t2, g2) = ShapeIdentifiability.BANDS
    cases = [(t1 * 0.5, g1), (t1 - 1e-9, g1), (t1, g2),
             (t2 * 0.9, g2), (t2 - 1e-9, g2), (t2, "낮음"), (t2 * 2, "낮음")]
    for frac, want in cases:
        assert _grade_of_spread_frac(frac) == want, (frac, want)
    # 실측: 지름 5.97 mm, 포트 반경 ~11 mm -> 54 % -> 낮음
    s = ShapeIdentifiability(n_starts=8, n_accepted=8,
                             spread_mm={"min": 1.2, "median": 3.4, "max": 5.97},
                             spread_frac=5.97 / 11.0, grade="낮음")
    assert any("식별성 낮음" in w for w in s.warnings())
    assert "5.97" in s.note
