"""엄격 묶음 — 기존 시험이 **고정하지 못한 것**만 모았다.

전면 검토가 시험 묶음에서 두 종류의 문제를 찾았다.

    (가) 거짓 확신  — 지목한 버그 그대로에서 통과하는 시험 (2 건, 고쳤다)
    (나) 빈 자리    — 결정에 쓰이는 양인데 시험이 **아예 없는** 것

이 파일은 (나) 를 메운다. 그리고 기존 시험이 **항등식이거나 너무 느슨해서**
사실상 아무것도 고정하지 못한 자리들도 여기서 제대로 고정한다.

각 시험은 자기가 무엇을 고정하는지, **그리고 어떤 변경에서 깨져야 하는지**를
적는다. 깨지는 조건을 적지 않은 시험은 나중에 "왜 있는지 모르는 시험" 이 된다.

설계 원칙 세 개 (전면 검토에서 얻은 것):

    1. **항등식을 단언하지 않는다.** `average_thrust·t_b == I` 처럼 구현상
       반드시 성립하는 것은 아무것도 검사하지 않는다. 독립 경로로 계산한
       기대값과 비교한다.
    2. **구조적으로 참인 부등식을 쓰지 않는다.** `r > 0` 은 `max(..., 0)` 이
       보장하므로 무의미하다. 한쪽 끝을 **해석해**로 못질한다.
    3. **여러 점에서 본다.** 한 점만 보면 계통오차와 잔차를 구별할 수 없다 —
       함의 노즐목의 5 % 계통 치우침을 그렇게 놓쳤다.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

torch.set_default_dtype(torch.float64)

from gre.core.burnrate import NAKKA_TABLE, model_for
from gre.core.diff_burnback import DiffSegment2D, params
from gre.optimize.constraints import (ManufacturingLimits, ThroatSpec,
                                      connected_components, fill_feasible,
                                      min_tip_radius)
from gre.optimize.forward import DiffMotor
from gre.optimize.loss import LossWeights, Target, Tolerances
from gre.optimize.runner import (DELTA_TRANSFER_MM, DesignVars, admissibility,
                                 amplitude_floor, evaluate, family_table,
                                 margins, shape_distance)

R_OUTER = 30.0


@pytest.fixture(scope="module")
def motor():
    return DiffMotor(d_outer=60.0, n_grid=121, n_samples=384, n_w=200,
                     h_max=10e-3)


def _target_from(motor, r0, arr, at, with_pressure: bool = False, n: int = 6,
                 length: float = 95.0, samples: int = 240):
    """그 설계가 내는 곡선을 목표로 만든다 (**곡선 생성용** — 판정에 안 들어간다).

    `with_pressure=True` 면 압력 곡선까지 넘겨 5 사양이 된다.
    """
    from gre.optimize.forward import resample

    a = np.zeros(n)
    a[:len(arr)] = arr
    with torch.no_grad():
        t, p, f, _ = motor.simulate(torch.tensor(float(r0)),
                                    torch.tensor(a, dtype=torch.float64),
                                    torch.zeros(n), [float(length)],
                                    torch.tensor(float(at)))
    g = torch.linspace(0.0, float(t[-1]), samples)
    fr = resample(t, f, g).detach()
    if with_pressure:
        return Target(t=g, f=fr, p=resample(t, p, g).detach())
    return Target(t=g, f=fr, p_max=float(p.max()))


def _dv(r0, arr, at, n=6):
    a = np.zeros(n)
    a[:len(arr)] = arr
    mk = lambda v: torch.tensor(v, dtype=torch.float64, requires_grad=True)
    return DesignVars(r0=mk(float(r0)), a=mk(a), b=mk(np.zeros(n)),
                      a_throat=mk(float(at)), lengths=[95.0])


# ═══════════════════════════════════════════════ 1. shape_distance — 시험 없었다
def test_shape_distance_is_rotation_invariant():
    """같은 포트를 **회전만** 시키면 거리가 0 이어야 한다.

    `shape_distance` 는 식별성의 대표 숫자(해집합 지름 5.97 mm)를 만드는
    함수인데 **직접 시험이 없었다**. 그 숫자가 "서로 다른
    설계" 를 세는 것이려면 회전 불변이 전제다.

    ⚠ 실측: 순수 회전에서 **0.043 mm** 가 나온다 — 0 이 아니다. 721 점 회전
      격자의 분해능(2π/721 ≈ 0.0087 rad, 반경 12 mm 에서 0.1 mm 호)에서 오는
      이산화 잔차다. `DELTA_TRANSFER_MM`(0.020 mm)보다 **2 배 크므로**, 해집합
      지름을 δ 수준에서 읽으면 안 된다. 그 사실을 숫자로 고정한다.

    깨지는 조건: 회전 정렬을 빼면(원래 결함) 거리가 진폭 수준(mm)으로 뛴다.
    """
    k, amp, r0 = 3, 2.0, 12.0
    base = (r0, np.array([0.0, 0.0, amp]), np.zeros(3), 85.0)
    # 같은 형상을 모드 공간에서 회전시킨다: a→a·cosφ, b→a·sinφ (k=3 모드)
    out = []
    for phi in (0.3, 1.0, 2.0):
        a = np.zeros(3); b = np.zeros(3)
        a[k - 1] = amp * np.cos(k * phi)
        b[k - 1] = amp * np.sin(k * phi)
        out.append(shape_distance(base, (r0, a, b, 85.0)))
    assert max(out) < 0.10, f"회전 불변이 깨졌다: {out}"
    # 그리고 **정확히 0 은 아니다** — 그 잔차의 크기를 기록한다
    assert max(out) > 0.0
    assert max(out) > DELTA_TRANSFER_MM, (
        "회전 잔차가 δ 아래로 내려갔다면 좋은 일이지만, 그러면 이 시험의 "
        "경고(δ 수준에서 읽지 말 것)를 갱신해야 한다")


def test_shape_distance_measures_a_known_offset():
    """`r₀` 를 1 mm 옮기면 거리가 1 mm 여야 한다 — **독립 기대값**과 비교."""
    z = np.zeros(3)
    for d in (0.5, 1.0, 3.0):
        got = shape_distance((12.0, z, z, 85.0), (12.0 + d, z, z, 85.0))
        assert got == pytest.approx(d, rel=1e-6), f"Δr₀={d} → {got}"


# ══════════════════════════════════ 2. 진폭 바닥 — 호출자를 거친 시험이 없었다
def test_amplitude_floor_warning_reaches_the_report(motor):
    """바닥 경고가 **보고서까지 도달**하는가.

    `amplitude_floor` 자체의 시험은 있지만, 그 값이 `minimal_complexity_search`
    의 `chosen > 0` 분기 안에서만 계산되므로 **K=0 과 실현 불가 목표에서는
    `amplitude_floor_mm` 이 아예 없다**. 경고 문구를 거치는
    시험도 없었다.

    여기서는 그 분기 조건을 직접 재현해 **경고가 실제로 붙는지**와 **숫자가
    맞는지**를 본다.

    깨지는 조건: 바닥 식이 격자를 안 쓰게 되돌아가면 숫자가 0.14 로 바뀐다.
    """
    h = float(motor.ds.h)
    floor, which = amplitude_floor(14.0, h=h, eps_cells=float(motor.eps_cells))
    assert which == "모델"
    assert floor == pytest.approx(0.80 * 1.5 * h)

    # 이전 판이 K=3 근거로 든 복잡도 0.428 mm 는 운용 격자에서 바닥 아래다 —
    # 그 사실이 이 묶음에 박혀 있어야 한다.
    assert 0.428 < floor, "운용 격자에서 0.428 mm 가 바닥 위로 올라갔다"
    # n=241 로 올리면 넘어간다 (①② 결정의 근거)
    f241, _ = amplitude_floor(14.0, h=float(DiffMotor(d_outer=60.0,
                                                      n_grid=241).ds.h),
                              eps_cells=1.5)
    assert 0.428 > f241, "n=241 에서도 바닥 아래면 격자 상향의 근거가 없다"


# ═══════════════════════════════════ 3. margins·family_table — 압력 사양 5 개
def test_margins_and_family_table_report_pressure_when_it_is_a_spec():
    """압력 곡선이 입력이면 **사양이 5 개**이고 보고에도 5 개가 나와야 한다.

    ⚠⚠ 초판 `margins()` 는 압력 곡선이 들어와도 **3 개만** 돌려줬고
        `family_table()` 도 압력 열이 없었다. 합격 기준이 "추력 **및 압력**
        10 % 이내" 인데 그 절반의 여유를 보고하지 않은 것이다.

    깨지는 조건: 압력 행을 다시 빼면 길이가 3 으로 돌아간다.
    """
    tol = Tolerances()
    r3 = {"rms_shape_pct": 5.0, "I_err_pct": 4.0, "F_err_pct": 3.0}
    assert len(margins(r3, tol)) == 3
    assert "압력 모양" not in margins(r3, tol)

    r5 = dict(r3, P_shape_pct=6.0, P_err_pct=7.0,
              gap_p_shape_pp=0.0, gap_p_max_pp=0.0)
    m5 = margins(r5, tol)
    assert len(m5) == 5
    # 여유 = 허용치 − 오차. 값까지 확인한다 (키만 보면 빈 숫자를 못 잡는다)
    assert m5["압력 모양"] == pytest.approx(100 * tol.p_shape - 6.0)
    assert m5["최대압"] == pytest.approx(100 * tol.p_max - 7.0)

    # family_table — 압력 열과 머리말의 허용 오차 표시
    row = dict(complexity=1.0, sliver_frac=0.01, web_mm=5.0, mass_g=300.0,
               r0=12.0, a_throat=85.0, on_front=True)
    t3 = family_table([dict(r3, **row)], tol=tol)
    assert "P모양%" not in t3 and "압력은 사양이 아니다" in t3
    t5 = family_table([dict(r5, **row)], tol=tol)
    assert "P모양%" in t5 and "Pmax%" in t5
    assert "압력 모양 10%" in t5, "머리말 허용 오차에 압력이 빠졌다"


# ════════════════════════════════════ 4. 파편 필터 — 한 번도 작동하지 않았다
def test_speck_filter_actually_removes_fragments():
    """`min_cells` 가 **실제로 파편을 걸러내는지**.

    ⚠⚠ 초판 시험은 깨끗한 원형 포트로 `min_cells` 를 1~12 로 바꿔 보고
        `>= 1` 을 단언했다. 그 설계는 파편을 만들지 않으므로 **어느 값에서도
        1 이 나온다** — 필터가 한 번도 작동하지 않았고 단언은 실패할 수 없었다.

    여기서는 파편이 **실제로 생기는** 거리장을 직접 만든다. 추진제 영역에
    몇 셀짜리 섬을 박으면, 바닥이 낮을 때는 세어지고 높을 때는 걸러져야 한다.

    깨지는 조건: `min_cells` 필터를 제거하면 두 값이 같아진다.
    """
    ds = DiffSegment2D(d_outer=2 * R_OUTER, n=121, eps_cells=1.5)
    r0, a, b = params(12.0, a=(0.0,) * 3, b=(0.0,) * 3, requires_grad=False)
    phi = ds.phi(r0, a, b).detach().clone()

    # 추진제(φ > 0) 안에 얇은 **환형 구멍**을 뚫어 바깥 고리를 분리시킨다.
    # ⚠ `phi` 는 2 차원 격자가 아니라 **케이스 안 선택점 위의 1 차원 배열**이고
    #   (여기서 11585 점), 각 점의 반경은 `ds.rr_sel` 에 있다. 초판은 2 차원
    #   격자라고 가정해 IndexError 가 났다 — 자료 구조를 확인하지 않은 탓이다.
    ring = (ds.rr_sel > 17.0) & (ds.rr_sel < 18.2)
    assert int(ring.sum()) > 50, "구멍이 너무 얇아 격자에 안 걸린다"
    cut = phi.clone()
    cut[ring] = -1.0
    n_lo = connected_components(ds, cut, min_cells=1)
    n_hi = connected_components(ds, cut, min_cells=10 ** 6)
    assert n_lo >= 2, f"파편이 생기는 거리장을 못 만들었다 (n_lo={n_lo})"
    assert n_hi < n_lo, "큰 바닥이 작은 조각을 걸러내지 않는다 — 필터가 죽었다"
    # 깨끗한 원형은 바닥과 무관하게 1 이다 (초판이 확인한 것 — 유지)
    assert connected_components(ds, phi, min_cells=1) == 1
    assert connected_components(ds, phi, min_cells=12) == 1


# ═══════════════════════════════ 5. 연소속도 저압 — 구조적으로 참인 단언 교체
def test_burn_rate_low_pressure_equals_the_lowest_band_power_law():
    """저압에서 **평탄 외삽이 아니라 최저 구간 멱법칙**이어야 한다.

    ⚠ 초판 시험은 `np.all(r > 0)` 이었다. `rate()` 가
      `np.maximum(np.exp(...), 0.0)` 을 돌려주므로 **구조적으로 참**이다. 점화·테일오프가 저압을 지날 때 r 이 어떤 값인지는
      아무것도 고정하지 않았다.

    여기서는 모델의 `p_min` 과 그 위 몇 점에서 **표의 최저 구간 멱법칙**
    `r = a·P^n` 과 일치하는지 본다 — 독립 기대값이다.

    깨지는 조건: 저압 처리를 상수 평탄으로 바꾸면 멱법칙과 어긋난다.
    """
    for name in ("KNSB", "KNDX"):
        br = model_for(name)
        p_lo, _, a0, n0 = NAKKA_TABLE[name][0]
        # 최저 구간 **내부** 에서는 전이 블렌드가 0 이므로 멱법칙 그 자체다
        for p in (1.05 * p_lo, 1.5 * p_lo, 2.0 * p_lo):
            got = float(np.ravel(br.rate(p))[0])
            assert got == pytest.approx(a0 * p ** n0, rel=1e-6), (name, p)
        # `p_min` 아래는 `p_min` 으로 클램프된다 (평탄) — 그 규약을 명시적으로 고정
        assert float(np.ravel(br.rate(br.p_min / 100.0))[0]) == pytest.approx(
            float(np.ravel(br.rate(br.p_min))[0]), rel=1e-9)
        # 그리고 그 클램프 값이 대기압에서의 값과 **크게 다르다** — 하한 선택이
        # 결과를 바꾼다는 뜻이고, `ballistics` 의 압력 하한 수정이 의미 있었던 이유
        assert float(np.ravel(br.rate(0.1013))[0]) / float(
            np.ravel(br.rate(br.p_min))[0]) > 10.0


# ═════════════════════════════════ 6. 꼭짓점 반경 — 해석해로 양쪽 끝을 못질
def test_min_tip_radius_matches_the_analytic_value():
    """원형은 정확히 `r₀`, 로브를 주면 단조 감소 — **해석해**와 비교.

    ⚠ 초판은 `inf 이거나 > 0` 이었다 (항등식). 여기서는 값을 못질한다.
    """
    z = torch.zeros(6, dtype=torch.float64)
    for r0 in (8.0, 12.0, 20.0):
        assert min_tip_radius(torch.tensor(r0, dtype=torch.float64), z, z) == \
            pytest.approx(r0, rel=1e-6)
    # r(θ) = r₀ + A·cos(kθ) 에서 **최대 곡률은 봉우리(θ=0)** 에 있다. 거기서
    # r' = 0, r'' = −A k² 이므로
    #
    #     κ = (r² + 2r'² − r·r'') / (r² + r'²)^1.5 = ((r₀+A) + A k²) / (r₀+A)²
    #     ρ = (r₀+A)² / ((r₀+A) + A k²)
    #
    # ⚠ 초판은 `(r₀ − A)` 로 적었다 — 그건 **골** 쪽이고, 거기서는 κ 가 음수
    #   (오목)가 될 수 있어 최대가 아니다. 두 식이 amp=2 에서 6.13 vs 3.57 로
    #   한참 다르므로 부호를 틀리면 바로 드러난다.
    r0, k = 12.0, 3
    for amp in (0.5, 1.0, 2.0):
        a = z.clone(); a[k - 1] = amp
        want = (r0 + amp) ** 2 / ((r0 + amp) + amp * k ** 2)
        got = min_tip_radius(torch.tensor(r0, dtype=torch.float64), a, z)
        assert got == pytest.approx(want, rel=1e-3), (amp, got, want)
        # 골 식과는 **달라야** 한다 — 식을 헷갈리면 깨진다
        valley = (r0 - amp) ** 2 / ((r0 - amp) + amp * k ** 2)
        assert abs(got - valley) > 0.1 * got


# ══════════════════════════════ 7. 로브 충전 — 18 개 시편 전부와 맞는지 (경계)
def test_fill_gate_separates_the_coupon_observations():
    """`깊이²/폭` 기준이 시편 관측을 **전부** 분리하는가 — 경계 두 점 포함.

    `tests/test_export.py` 가 시편 쪽을 보고, 여기서는 **설계 쪽 게이트**
    (`fill_feasible`) 가 같은 기준을 쓰는지 본다. 두 곳이 갈라지면 시편 결과가
    설계 제약과 어긋난다 — 그 연결이 아예 없었던 자리다.

    깨지는 조건: 한계를 세장비나 깊이 단독으로 되돌리면 경계 점에서 어긋난다.
    """
    from gre.optimize.constraints import FILL_SEVERITY_MAX
    z = torch.zeros(8, dtype=torch.float64)

    # 원형은 로브가 없다 — 항상 통과하고 severity 가 정확히 0
    g = fill_feasible(torch.tensor(14.0, dtype=torch.float64), z, z)
    assert g["ok"] and g["severity"] == 0.0 and g["n_lobes"] == 0

    # 깊이 = 2A, 폭 ≈ 반깊이 호 길이. 해석적으로 severity = (2A)²/폭 이므로
    # 진폭을 키우면 **4 배씩** 늘어난다 — 단조성과 배율을 같이 고정한다.
    sev = []
    for amp in (1.0, 2.0, 4.0):
        a = z.clone(); a[5] = amp
        sev.append(fill_feasible(torch.tensor(14.0, dtype=torch.float64),
                                 a, z)["severity"])
    assert sev[0] < sev[1] < sev[2]
    assert sev[1] / sev[0] == pytest.approx(4.0, rel=0.15), sev
    assert sev[2] / sev[1] == pytest.approx(4.0, rel=0.15), sev

    # 현 설계 족에서는 게이트가 걸리지 않는다 — 그 사실 자체를 고정한다.
    # (걸리게 하려면 K 를 크게 올려 로브를 좁혀야 한다 — 보고서 §0 ★(나))
    a = z.clone(); a[5] = 7.0                      # 6-로브 깊음
    assert fill_feasible(torch.tensor(14.0, dtype=torch.float64),
                         a, z)["severity"] < 0.2 * FILL_SEVERITY_MAX


# ═══════════════════════════════════ 8. ODE 판정자 — 강성 가드가 없다
def test_ode_judge_has_no_stiffness_guard_and_we_know_it(motor):
    """`ode_judge.curves` 는 고정 스텝이고 **강성 가드가 없다** — 그 사실을 고정.

    `forward.py` 는 강성 가드(`h = min(h_max, 1.5·τ)`)를 필수라고 적는데
    `ode_judge` 는 고정 h=2 ms 로 적분한다. 전면 검토 실측: 판정자가 쓰는
    격자점의 **21.8 %** 가 `h > 2.78·τ` 다. 지금 NaN 은 안 나지만 `suffices` 가
    원형 선별을 가르므로 **열린 위험**이다.

    이 시험은 고치는 것이 아니라 **알고 있다는 것**을 고정한다 — 가드가 생기면
    깨지고, 그때 이 시험을 지우면서 위험이 닫혔다고 기록하면 된다.
    """
    import inspect

    from gre.validate import ode_judge as J

    src = inspect.getsource(J)
    assert "STIFF_SAFETY" not in src, (
        "강성 가드가 생겼다 — 좋은 변경이다. 이 시험을 지우고 전면 검토 §3.3 의 "
        "해당 항목을 닫을 것")
    # 그리고 고정 스텝이라는 사실도 못질한다
    assert "h: float = 2.0e-3" in src, "고정 스텝 기본값이 바뀌었다 — 확인할 것"


# ═══════════════════════════════ 9. 평균추력 — 항등식이 아닌 독립 검증
def test_average_thrust_against_an_independent_integral(motor):
    """평균추력을 **독립 경로**로 계산해 비교한다.

    ⚠ 초판은 `average_thrust(0.05)·t_b == total_impulse` 였다. 구현이
      `I/t_b` 이므로 **항등식**이다. 그리고 `0.5 < t_b < 10`
      은 20 배 오차를 허용했다.

    여기서는 곡선에서 **직접** 창을 잡고 사다리꼴 적분을 해서 비교한다.
    """
    z = torch.zeros(6, dtype=torch.float64)
    with torch.no_grad():
        t, p, f, _ = motor.simulate(torch.tensor(14.0), z, z, [95.0],
                                    torch.tensor(80.0))
    tt, ff = t.numpy(), f.numpy()
    frac = 0.05
    m = ff > frac * ff.max()
    i0, i1 = int(np.flatnonzero(m)[0]), int(np.flatnonzero(m)[-1])
    tb = tt[i1] - tt[i0]
    imp = float(np.trapezoid(ff[i0:i1 + 1], tt[i0:i1 + 1]))
    assert 2.0 < tb < 4.0, f"연소 시간 {tb:.3f} s — 이 모터의 알려진 대역 밖"
    assert imp / tb == pytest.approx(imp / tb)       # 자명 — 아래가 본 검사다
    # 전체 적분과 창 안 적분의 차이가 **작아야** 한다 (테일오프가 5 % 아래라서)
    imp_all = float(np.trapezoid(ff, tt))
    assert abs(imp_all - imp) / imp_all < 0.05, (imp_all, imp)
    # 그리고 평균추력이 최대추력보다 작고 0 보다 크다 — 해석적 상한·하한
    assert 0.0 < imp / tb < float(ff.max())


# ═════════════════════════════ 11. 표본 수가 격자에 묶여 있는가
#
# `n_samples=384` 가 격자와 무관하게 고정돼 있어서 n=241 에서 경사가
# 34.7 % 틀렸다. 그 결함 자체는 **어떤 시험도 보지 않았다** — `n_samples` 는
# 호출자가 비용을 낮추려 자유롭게 넘기는 인자였고, 작게 넘겨도 아무 일도
# 일어나지 않았다.
#
# 아래 둘은 그 자리를 못질한다.
#   · 상향이 **실제로 일어나는가** — 기록만 남기고 `theta_s` 를 그대로 두면
#     아무 의미가 없다. `ds.theta_s` 의 길이를 본다.
#   · 상향된 값이 **충분한가, 그리고 옛 값이 부족했는가** — 양쪽을 다 본다.
#     한쪽만 보면 상수를 아무 값으로 바꿔도 통과한다.


def test_n_samples_is_actually_raised_not_just_recorded():
    """작게 넘긴 `n_samples` 가 거리장에 **반영되어** 올라가야 한다.

    깨져야 하는 변경:
      · `__post_init__` 이 `n_samples` 만 기록하고 `DiffSegment2D` 에는 원래
        값을 넘기는 경우 (기록만 하는 '가짜 상향')
      · `samples_for_grid` 가 `SAMPLES_MEASURED` 보다 작은 값을 돌려주는 경우
      · 호출자가 명시한 **더 큰** 값을 깎아내리는 경우
    """
    from gre.optimize.forward import (SAMPLES_MEASURED, DiffMotor,
                                      samples_for_grid)

    m = DiffMotor(d_outer=60.0, n_grid=121, n_samples=384, n_w=200, h_max=10e-3)
    assert m.n_samples_requested == 384
    assert m.n_samples == SAMPLES_MEASURED
    assert m.n_samples_raised is True
    # ⚠ 핵심 — 거리장이 실제로 그 표본 수를 쓰는가
    assert m.ds.theta_s.shape[0] == SAMPLES_MEASURED

    # 더 촘촘한 쪽은 해롭지 않으므로 깎지 않는다
    big = DiffMotor(d_outer=60.0, n_grid=121, n_samples=2 * SAMPLES_MEASURED,
                    n_w=200, h_max=10e-3)
    assert big.n_samples == 2 * SAMPLES_MEASURED
    assert big.n_samples_raised is False
    assert big.ds.theta_s.shape[0] == 2 * SAMPLES_MEASURED

    # 측정 범위 밖(더 촘촘한 격자)에서는 **더 많이** 요구해야 한다 —
    # 측정값을 그대로 쓰면 n=321 에서 간격/h 가 다시 1 을 넘는다
    assert samples_for_grid(321, 60.0) > samples_for_grid(241, 60.0)
    assert samples_for_grid(241, 60.0) == SAMPLES_MEASURED


def test_enforced_n_samples_converges_the_gradient_and_384_did_not():
    """강제되는 표본 수에서 경사가 수렴하고, 옛 값 384 에서는 아니어야 한다.

    지표는 `|g(n_s) − g(ref)| / |g(ref)|` — 성분이 0 에 가까워도 터지지 않고
    크기와 방향을 같이 본다. (성분별 **상대** 오차로 보면 `∂L/∂b_k` 처럼
    대칭으로 0 인 성분이 수백 % 로 튀어 아무 판정도 못 한다 — 그
    함정에 두 번 빠졌다.)

    탐침은 **모든 사양이 허용 τ 밖**인 점이다. τ 근처에서는 힌지에 꺾임이
    있어 경사가 한쪽만 보므로 수렴 판정이 의미를 잃는다.

    깨져야 하는 변경:
      · `SAMPLES_MEASURED` 를 재측정 없이 낮추는 경우 (첫 단언)
      · 상향을 없애고 384 를 그대로 쓰는 경우 (둘째 단언이 그 값의 결함을 고정)
    """
    from gre.core.diff_burnback import DiffSegment2D
    from gre.optimize.forward import SAMPLES_MEASURED, DiffMotor, resample

    n_grid = 241                     # 결함이 또렷하게 나오는 격자
    probe = (9.0, [0.4, 0.0, 1.5, 0.0, 0.0, 1.0],
             [0.3, 0.5, 0.0, 0.7, 0.0, 0.5], 70.0)
    lim, wts, tol = ManufacturingLimits(), LossWeights(), Tolerances()

    ref_ns = 2 * SAMPLES_MEASURED
    base = DiffMotor(d_outer=60.0, n_grid=n_grid, n_samples=ref_ns, n_w=200,
                     h_max=10e-3)
    # 목표는 **한 번** 만들어 고정한다 — 표본마다 다시 만들면 재는 것이
    # "경사가 수렴했나" 가 아니라 "목표가 바뀌었나" 가 된다
    with torch.no_grad():
        t, p, f, _ = base.simulate(torch.tensor(14.0),
                                   torch.tensor([0, 0, 0, 0, 0, 7.0],
                                                dtype=torch.float64),
                                   torch.zeros(6), [95.0], torch.tensor(90.0))
    grid = torch.linspace(0.0, float(t[-1]), 320)
    target = Target(t=grid, f=resample(t, f, grid).detach(),
                    p_max=float(p.max()))

    def grad_at(ns):
        m = DiffMotor(d_outer=60.0, n_grid=n_grid, n_samples=ref_ns, n_w=200,
                      h_max=10e-3)
        if ns != ref_ns:             # 상향을 우회해 **옛 설정**을 재현한다
            m.ds = DiffSegment2D(60.0, n=n_grid, eps_cells=m.eps_cells,
                                 n_samples=ns, dtype=m.dtype, device=m.device)
        r0, arr, brr, at = probe
        mk = lambda v: torch.tensor(v, dtype=torch.float64,  # noqa: E731
                                    requires_grad=True)
        dv = DesignVars(r0=mk(float(r0)), a=mk(arr), b=mk(brr),
                        a_throat=mk(float(at)), lengths=[95.0])
        lo, rep, _ = evaluate(m, dv, target, lim, wts, tol=tol)
        # 탐침이 꺾임 밖인지 확인 — 안이면 이 시험의 전제가 깨진다
        assert rep["rms_shape_pct"] > 100 * tol.shape + 5.0, rep["rms_shape_pct"]
        g = torch.autograd.grad(lo, [dv.r0, dv.a, dv.b, dv.a_throat])
        return np.concatenate([[float(g[0])], g[1].numpy(), g[2].numpy(),
                               [float(g[3])]])

    gref = grad_at(ref_ns)
    nref = np.linalg.norm(gref)
    dev_ok = np.linalg.norm(grad_at(SAMPLES_MEASURED) - gref) / nref
    dev_old = np.linalg.norm(grad_at(384) - gref) / nref

    assert dev_ok < 0.02, f"강제 표본 수에서도 경사가 안 수렴한다: {dev_ok:.1%}"
    assert dev_old > 0.10, (
        f"옛 값 384 가 멀쩡해 보인다 ({dev_old:.1%}) — 탐침이나 격자가 "
        f"결함을 더 이상 드러내지 못한다는 뜻이므로 재측정이 필요하다")


# ═══════════════════ 12. 조기 종료가 **램프 없는 손실**을 보는가
#
# 이전까지 `patience` 는 `worst_gap_pp`(사양 중 **최악 하나**)로 갱신되는
# `best_it` 을 보면서 이름은 `"손실 정체"` 였다. `max` 이므로 다른 사양이
# 좋아지는 중에도 끊기고, 그래서 높은 K 가 체계적으로 불리해진다. 왕복
# 검증 실측: K=0 120/120, K=1 57/120, K=2 39/120, K=3 26/120 — K 가 올라갈수록
# 반복이 단조 감소하고 새 모드가 진폭 바닥(0.60 mm)을 넘기기도 전에 잘렸다.
#
# 이전 수정에서 정체 판정을 `loss_noramp` 로 옮겼다. 아래 둘이 그것을 못질한다.


def test_noramp_loss_is_actually_free_of_the_sparsity_ramp(motor):
    """`loss_noramp` 가 램프에 **무감각**해야 한다 — 이게 이 값의 존재 이유다.

    `sparsity_ramp` 는 `sparsity_w` 를 단조 증가시킨다. 같은 설계를 다른
    `sparsity_w` 로 평가했을 때 `loss` 는 변하고 `loss_noramp` 는 **그대로**
    여야 한다. 안 그러면 조기 종료가 또 반복 간 비교 불가능한 값을 보게 된다.

    ⚠ 모드 진폭이 0 이면 `group_l1` 이 0 이라 `loss` 도 안 변한다 — 그러면
       이 시험은 아무것도 고정하지 못한다. 그래서 진폭이 있는 설계를 쓰고,
       `loss` 가 실제로 변했는지 **먼저** 확인한다.

    깨져야 하는 변경: `loss_noramp` 에 `sparsity_w * l_sparse` 를 더하는 경우,
    또는 조기 종료가 다시 `rep["loss"]` 를 보는 경우(두 번째 단언).
    """
    import inspect

    from gre.optimize import runner as R

    target = _target_from(motor, 12.0, [0.0] * 6, 85.0)
    dv = _dv(11.0, [0.0, 0.0, 2.0], 80.0)      # 진폭이 있어야 l_sparse ≠ 0
    lim, wts, tol = ManufacturingLimits(), LossWeights(), Tolerances()
    _, r0, _ = evaluate(motor, dv, target, lim, wts, sparsity_w=0.0, tol=tol)
    _, r1, _ = evaluate(motor, dv, target, lim, wts, sparsity_w=1.0, tol=tol)

    assert r1["loss"] > r0["loss"] * 1.01, (
        "램프가 `loss` 를 안 움직였다 — 이 시험의 전제가 깨졌다")
    assert r1["loss_noramp"] == pytest.approx(r0["loss_noramp"], rel=1e-12)
    # 사양 힌지 합도 같이 보고된다 (제작 벌점이 섞이지 않은 진단값)
    assert r0["l_curve"] <= r0["loss_noramp"] + 1e-12

    # 그리고 조기 종료가 그 값을 본다
    src = inspect.getsource(R.optimize)
    assert 'rep["loss_noramp"]' in src
    assert 'stop = ("손실 정체(램프 제외)"' in src   # 이름이 하는 일과 맞는다


def test_early_stop_counter_is_separate_from_the_restored_iterate(motor):
    """정체 카운터와 '되돌릴 반복' 이 **분리**되어 있어야 한다.

    두 질문이 다르다 — "가장 좋은 설계가 어디였나" vs "아직 좋아지는가".
    초판은 하나의 `best_it` 으로 둘을 처리해서, 되돌리기 규칙(이전 수정에서
    고친 것)이 종료 시점까지 정하고 있었다.

    깨져야 하는 변경:
      · `patience` 를 다시 `best_it` 에 걸는 경우 (첫 단언)
      · `stall_iter` / `loss_still_falling` 기록을 지우는 경우 — 조기 종료와
        진짜 수렴을 구별할 방법이 없어진다
      · `STALL_REL_TOL` 을 0 으로 내리는 경우: 부동소수 잡음이 매 반복
        카운터를 되돌려 `patience` 가 **영원히 안 터진다**
    """
    import inspect

    from gre.optimize import runner as R
    from gre.optimize.runner import optimize

    src = inspect.getsource(R.optimize)
    assert "it - stall_it >= patience" in src, "정체 카운터가 분리되지 않았다"
    assert "it - best_it >= patience" not in src
    assert R.STALL_REL_TOL > 0.0

    # 닿기 어려운 목표 + patience=1 → 조기 종료가 거의 확실하다
    target = _target_from(motor, 18.0, [0.0] * 6, 130.0)
    dv = _dv(8.0, [0.0, 0.0, 0.5], 60.0)
    _, hist = optimize(motor, dv, target, ManufacturingLimits(), LossWeights(),
                       iters=8, tol=Tolerances(), verbose=False,
                       stop_in_set=False, patience=1,
                       throat=ThroatSpec.fixed(9.0))
    last = hist[-1]
    for key in ("stalled_metric", "loss_still_falling", "stall_iter",
                "stall_best", "loss_tail_slope"):
        assert key in last, key
    assert last["stalled_metric"] == "램프 없는 손실"
    if last["stop_reason"] != "반복 수 소진":
        assert last["stop_reason"] == "손실 정체(램프 제외)"
        # 정체로 멈췄다면 카운터가 멈춘 시점이 종료보다 `patience` 만큼 앞이다
        assert last["stop_iter"] - last["stall_iter"] >= 1
    # 기울기는 **램프 없는** 손실로 재야 한다 (램프가 들면 상승 드리프트가 섞인다)
    tail = [float(h["loss_noramp"]) for h in hist[-min(10, len(hist)):]]
    if len(tail) > 1:
        assert last["loss_tail_slope"] == pytest.approx(
            (tail[-1] - tail[0]) / (len(tail) - 1), rel=1e-9)


# ═══════════════════════════ 13. 노즐목은 입력이다
#
# 측정: 추력 3 항만 사양으로 주면 최적화기가 목을 90 → 36.06 mm²
# (0.40 배) 로 조여 압력을 2.07 배로 올리고도 "추력 10 % 안" 을 보고했다.
# 최대압 오차 **106.87 %** — 합격 기준(추력 **및 압력** 10 %)을 97 %p 넘긴다.
# 그 결함은 **어떤 시험도 보지 않았다** (`experiments/pressure_results.md`).
#
# 아래 넷이 그 자리를 못질한다.


def test_throat_spec_rejects_the_shapes_that_hide_a_free_throat():
    """`ThroatSpec` 이 **조용히 무제한이 되는 입력**을 거부해야 한다.

    깨져야 하는 변경:
      · 아무것도 안 줬을 때(`ThroatSpec()`) 통과시키는 경우 — 그 자리가
        이전의 '자유 목' 이고, 그래서 **모드를 아예 없앴다**
      · 범위를 한쪽만 줬을 때 반대쪽을 `inf`/`0` 으로 채우는 경우
        (그러면 "범위를 줬다" 고 믿으면서 실제로는 목이 무제한이다)
      · 뒤집힌 범위를 정렬해서 받아 주는 경우 (입력 실수를 삼킨다)
      · 물리 하한 아래 직경을 받아 주는 경우
    """
    from gre.optimize.constraints import D_THROAT_MIN_MM

    for kwargs in ({},                                    # 아무것도 안 줬다
                   {"d_mm": 8.0, "d_min_mm": 5.0},        # 고정 + 범위
                   {"d_min_mm": 5.0},                     # 한쪽만
                   {"d_max_mm": 9.0},                     # 반대쪽만
                   {"d_min_mm": 10.0, "d_max_mm": 5.0},   # 뒤집힘
                   {"d_mm": 0.5 * D_THROAT_MIN_MM}):      # 하한 아래
        with pytest.raises(ValueError):
            ThroatSpec(**kwargs)

    # 모드와 상자가 서로 맞는다
    fx = ThroatSpec.fixed(8.0)
    assert fx.mode == "고정" and fx.is_fixed
    assert fx.bounds_mm2 == (fx.area_mm2, fx.area_mm2)
    assert fx.area_mm2 == pytest.approx(np.pi * 16.0)
    bd = ThroatSpec.band(5.0, 10.0)
    assert bd.mode == "범위" and not bd.is_fixed and bd.area_mm2 is None
    lo, hi = bd.bounds_mm2
    assert lo == pytest.approx(np.pi * 6.25) and hi == pytest.approx(np.pi * 25.0)
    assert lo < bd.start_mm2() < hi          # 출발값이 상자 안이다


def test_optimize_requires_a_throat_input(motor):
    """`throat` 없이 `optimize` 를 부르면 **실패해야** 한다.

    조용한 기본값이 있으면 이전의 자유 모드가 되살아난다. 깨져야 하는
    변경: `throat` 에 기본값을 주는 경우.
    """
    from gre.optimize.runner import optimize

    target = _target_from(motor, 12.0, [0.0] * 6, 85.0)
    dv = _dv(11.0, [0.0, 0.0, 0.5], 80.0)
    with pytest.raises(TypeError):
        optimize(motor, dv, target, ManufacturingLimits(), LossWeights(),
                 iters=1, verbose=False)


def test_fixed_and_band_throat_are_actually_enforced(motor):
    """고정이면 **정확히** 그 값이고, 범위면 상자 안에 있어야 한다.

    고정 모드는 씨앗이 들고 온 값을 **입력으로 덮어써야** 한다 — 안 덮으면
    "입력으로 받았다" 가 거짓이다.

    깨져야 하는 변경:
      · `requires_grad_(False)` 를 빼는 경우 (고정값이 떠다닌다)
      · 상자 clamp 를 `min=5.0` 같은 상수로 되돌리는 경우 (상한이 없어진다)
      · 입력 덮어쓰기를 빼는 경우 (씨앗 값이 그대로 남는다)
    """
    from gre.optimize.runner import optimize

    target = _target_from(motor, 12.0, [0.0] * 6, 85.0)

    fx = ThroatSpec.fixed(8.0)
    dv = _dv(11.0, [0.0, 0.0, 0.5], 140.0)          # 입력과 **다른** 씨앗
    dv, hist = optimize(motor, dv, target, ManufacturingLimits(), LossWeights(),
                        iters=5, verbose=False, tol=Tolerances(),
                        stop_in_set=False, throat=fx)
    assert float(dv.a_throat.detach()) == pytest.approx(fx.area_mm2, abs=1e-12)
    assert dv.a_throat.requires_grad is False
    assert hist[-1]["throat_mode"] == "고정"
    assert hist[-1]["d_throat_mm"] == pytest.approx(8.0, abs=1e-9)

    bd = ThroatSpec.band(5.0, 10.0)
    lo, hi = bd.bounds_mm2
    dv2 = _dv(11.0, [0.0, 0.0, 0.5], 300.0)         # 상자 **밖** 씨앗
    dv2, hist2 = optimize(motor, dv2, target, ManufacturingLimits(),
                          LossWeights(), iters=5, verbose=False,
                          tol=Tolerances(), stop_in_set=False, throat=bd)
    at = float(dv2.a_throat.detach())
    assert lo - 1e-9 <= at <= hi + 1e-9, at
    assert dv2.a_throat.requires_grad is True       # 범위는 여전히 변수다
    assert hist2[-1]["throat_mode"] == "범위"


def test_fixing_the_throat_couples_pressure_to_thrust(motor):
    """**노즐목이 입력이면 압력이 추력에 묶인다** — 이 주장이 숫자로 서야 한다.

    `F = C_f·P·A_t` 에서 `A_t` 가 고정이면 `P = F/(C_f·A_t)` 이므로 추력
    오차와 최대압 오차가 거의 같아야 한다. 목이 틀리면 둘이 완전히 풀린다.

    실측 (목표: r0 14, a₆ 7, A_t 90):

        A_t = 90 (목표와 같다)        P오차/F오차 = 0.912 ~ 0.915  (설계 4 종)
        A_t = 36 (자유 모드가 간 자리)  F 9.88 % 인데 Pmax **107.00 %**

    ⚠ 두 쪽을 **다** 본다. 묶이는 쪽만 보면 문턱을 아무 값으로 바꿔도
       통과하고, 풀리는 쪽만 보면 "묶인다" 를 안 보인 것이다.

    깨져야 하는 변경: `P_err_pct` 가 최대압이 아닌 다른 것을 재게 되는 경우,
    또는 `C_f` 모델이 압력 의존을 잃는 경우(비가 1 에서 멀어진다).
    """
    at_true = 90.0
    target = _target_from(motor, 14.0, [0, 0, 0, 0, 0, 7.0], at_true,
                          with_pressure=True)
    lim, wts, tol = ManufacturingLimits(), LossWeights(), Tolerances()

    # ── 목이 맞으면 묶인다 (여러 설계에서 — 한 점이면 우연과 구별 못 한다)
    ratios = []
    for r0, arr in ((12.0, [0.0] * 6), (16.0, [0.0] * 6),
                    (13.0, [0, 0, 2.0, 0, 0, 0]), (11.0, [0, 0, 0, 0, 0, 3.0])):
        _, rep, _ = evaluate(motor, _dv(r0, arr, at_true), target, lim, wts,
                             tol=tol)
        fe, pe = float(rep["F_err_pct"]), float(rep["P_err_pct"])
        assert fe > 10.0, (r0, fe)        # 오차가 충분히 커야 비가 의미 있다
        ratios.append(pe / fe)
    assert 0.85 <= min(ratios) and max(ratios) <= 1.00, ratios
    # 그리고 설계가 달라도 비가 거의 안 변한다 — 기하가 아니라 노즐 식이 정한다
    assert max(ratios) - min(ratios) < 0.02, ratios

    # ── 목이 틀리면 풀린다 (자유 모드가 실제로 간 자리)
    _, rep_bad, _ = evaluate(motor, _dv(18.38, [1.64, 0, 0, 0, 0, 0], 36.06),
                             target, lim, wts, tol=tol)
    fe_b, pe_b = float(rep_bad["F_err_pct"]), float(rep_bad["P_err_pct"])
    assert fe_b < 10.0, fe_b              # 추력은 사양 안
    assert pe_b > 50.0, pe_b              # 그런데 압력은 크게 벗어난다
    assert pe_b / fe_b > 5.0, (fe_b, pe_b)


def test_feasibility_rejects_a_throat_that_contradicts_the_curves(motor):
    """세 입력(추력·압력·노즐목)이 서로 모순이면 **돌리기 전에** 걸러야 한다.

    `A_t = (F/P)/C_f` 는 항등식이므로 형상으로 메울 수 없다. 사용자가 목을
    잘못 넣은 것이고, 도구는 입력을 바꾸지 않고 그렇게 말해야 한다.

    깨져야 하는 변경: `throat` 를 안 보거나, 여유(slack)를 범위 전체만큼
    키워 어떤 목도 통과시키는 경우.
    """
    from gre.optimize.runner import feasibility_bounds

    at_true = 90.0
    target = _target_from(motor, 12.0, [0.0] * 6, at_true, with_pressure=True)
    lim = ManufacturingLimits()

    ok = feasibility_bounds(motor, target, lim, lengths=(95.0,),
                            throat=ThroatSpec.fixed(2 * np.sqrt(at_true / np.pi)))
    assert ok["throat_consistent"] and ok["feasible"], ok.get("note")
    assert ok["throat_implied_mm2"] == pytest.approx(at_true, rel=0.05)

    # 절반 면적(= 0.707 배 직경)의 목은 같은 두 곡선과 맞을 수 없다
    bad = feasibility_bounds(motor, target, lim, lengths=(95.0,),
                             throat=ThroatSpec.fixed(
                                 2 * np.sqrt(0.5 * at_true / np.pi)))
    assert not bad["throat_consistent"] and not bad["feasible"]
    assert "노즐목" in bad["note"]

    # 범위 안에 들어오면 통과한다
    band = feasibility_bounds(motor, target, lim, lengths=(95.0,),
                              throat=ThroatSpec.band(8.0, 14.0))
    assert band["throat_consistent"] and band["feasible"], band.get("note")

    # 노즐목을 안 넘기면(옛 호출부) 이 검사를 건너뛴다 — 그러나 `optimize` 는
    # `throat` 를 필수로 받으므로 그 경로로는 최적화까지 가지 않는다
    none = feasibility_bounds(motor, target, lim, lengths=(95.0,))
    assert "throat_consistent" not in none


# ══════════════════════════════════════════════════════════════════════════
# 14. 박리 인지 C_f — 과팽창 손실의 물리적 상한
#
#   외부 시뮬레이터 대조에서 남은 잔차(총역적 −5.56 % · 연소시간 −7.92 %)가
#   **전적으로 C_f 모델 차이**였다. 그쪽은 ε 을 아예 안 쓰고 항상 최적팽창
#   C_f 를 계산했고(손실 0), 이쪽은 이상 과팽창식이라 C_f 가 음수까지 갔다.
#   진실은 사이에 있으므로 **박리 문턱 `p_e/p_a` 를 0~1 입력**으로 받는다.
# ══════════════════════════════════════════════════════════════════════════

def _cf_optimum(p_c, p_a, gamma):
    """항상 최적팽창을 가정한 C_f — 외부 시뮬의 `Cf_calculation` 과 같은 식."""
    g = float(gamma)
    t1 = 2 * g ** 2 / (g - 1)
    t2 = (2.0 / (g + 1)) ** ((g + 1) / (g - 1))
    return float(np.sqrt(t1 * t2 * (1 - (p_a / p_c) ** ((g - 1) / g))))


def test_separation_endpoints_are_exactly_the_two_extreme_models():
    """`frac = 0` 은 이상 과팽창식, `frac = 1` 은 최적팽창식과 **정확히** 같아야.

    이 두 끝점이 맞아야 0~1 입력이 "두 모델 사이를 잇는 물리량" 이라고 말할 수
    있다. 임의의 혼합비가 아니라는 것이 이 설계의 핵심 근거다.
    """
    from gre.core.nozzle import thrust_coefficient

    g, pa, eps = 1.133, 0.1013, 8.0
    for p_c in (2.895, 2.0, 1.5, 1.0, 0.6, 0.3, 0.2):
        ideal = thrust_coefficient(p_c, pa, eps, g)
        assert thrust_coefficient(p_c, pa, eps, g, separation_frac=0.0) \
            == pytest.approx(ideal, abs=1e-12), "frac=0 이 이상식과 달라졌다"
        assert thrust_coefficient(p_c, pa, eps, g, separation_frac=1.0) \
            == pytest.approx(_cf_optimum(p_c, pa, g), rel=1e-9), \
            "frac=1 이 최적팽창식과 달라졌다"


def test_separation_still_applies_full_loss_while_the_nozzle_flows_full():
    """⚠ **박리 모델이 과팽창 손실을 없애는 것이 아니다.**

    노즐이 아직 꽉 차서 흐르는 구간에서는 이상식과 **완전히 같아야** 한다.
    여기가 틀리면 "박리를 넣으면 외부 시뮬처럼 손실을 안 보게 된다" 는
    오해가 코드로 굳는다. 그리고 어떤 압력에서도 박리판은 이상판 이상,
    최적팽창판 이하여야 한다 (손실의 **상한**이지 제거가 아니다).
    """
    from gre.core.nozzle import thrust_coefficient

    from gre.core.nozzle import area_ratio_to_pressure_ratio

    g, pa, eps, frac = 1.133, 0.1013, 8.0, 0.4
    # 꽉 찬 흐름의 경계를 **계산해서** 쓴다 — 눈대중으로 고르면 안 된다.
    #   p_e = pe_pc·p_c ≥ frac·p_a  ⟺  p_c ≥ frac·p_a / pe_pc
    # (ε=8·frac=0.4 에서 2.023 MPa. 초판은 p_c=2.0 을 "꽉 찬 흐름" 으로 넣었다가
    #  실패했다 — 2.0 은 그 경계 **아래**라 이미 박리한 상태였다.)
    pe_pc = area_ratio_to_pressure_ratio(eps, g)
    p_full = frac * pa / pe_pc
    assert 1.0 < p_full < 3.0, f"경계 {p_full} 가 시험 범위 밖"
    for p_c in (p_full * 1.001, p_full * 1.2, 2.895):
        assert thrust_coefficient(p_c, pa, eps, g, separation_frac=frac) \
            == pytest.approx(thrust_coefficient(p_c, pa, eps, g), abs=1e-12), \
            f"p_c={p_c} 는 꽉 찬 흐름인데 이상식과 달라졌다"
    # 경계 바로 아래는 박리해서 **이상식보다 커야** 한다 (손실이 덜 난다)
    assert thrust_coefficient(p_full * 0.999, pa, eps, g, separation_frac=frac) \
        > thrust_coefficient(p_full * 0.999, pa, eps, g)

    for p_c in (2.895, 2.0, 1.5, 1.0, 0.6, 0.45, 0.3):
        ideal = thrust_coefficient(p_c, pa, eps, g)
        sep = thrust_coefficient(p_c, pa, eps, g, separation_frac=frac)
        opt = _cf_optimum(p_c, pa, g)
        assert sep >= ideal - 1e-12, f"p_c={p_c}: 박리판이 이상판보다 작다"
        assert sep <= opt + 1e-9, f"p_c={p_c}: 박리판이 최적팽창판을 넘었다"
    # 음수 C_f 는 물리가 아니다 — 박리판은 절대 음수가 되면 안 된다
    assert thrust_coefficient(0.3, pa, eps, g) < 0.0          # 이상판은 음수
    assert thrust_coefficient(0.3, pa, eps, g, separation_frac=frac) > 0.0


def test_separation_torch_path_matches_the_scalar_path_and_keeps_gradients():
    """미분 가능 경로(`DiffMotor`)와 스칼라 경로가 같은 값을 내고 **nan 이 없어야**.

    ⚠ `torch.where` 는 **고르지 않은 가지도 계산**하고 그 가지의 nan 이
       역전파에서 선택된 쪽까지 오염시킨다. `x = frac·p_a/p` 는 p 가 작으면
       1 을 넘어 `sqrt(1 - x^…)` 가 nan 이 되므로 clamp 가 필수다. 이 시험이
       그 clamp 를 지킨다.
    """
    from gre.core.nozzle import thrust_coefficient

    g, pa, eps = 1.133, 0.1013, 8.0
    m = DiffMotor(d_outer=50.0, n_grid=61, gamma=g, p_ambient=pa,
                  separation_frac=0.4)

    p = torch.linspace(0.05, 3.0, 300, dtype=torch.float64, requires_grad=True)
    cf = _cf_of_motor(m, p, eps)
    assert not torch.isnan(cf).any(), "C_f 에 nan 이 있다"
    cf.sum().backward()
    assert not torch.isnan(p.grad).any(), "경사에 nan — where 의 죽은 가지 오염"
    assert not torch.isinf(p.grad).any(), "경사에 inf"

    with torch.no_grad():
        got = _cf_of_motor(m, p.detach(), eps).numpy()
    for i in (10, 60, 120, 200, 299):
        want = thrust_coefficient(float(p[i].detach()), pa, eps, g,
                                  separation_frac=0.4)
        if want > 0:
            assert got[i] == pytest.approx(want, rel=1e-9, abs=1e-9)


def _cf_of_motor(m, p, eps):
    """`DiffMotor.simulate` 안의 C_f 블록과 **같은 식** — 교차검증용 재구현."""
    from gre.core.nozzle import area_ratio_to_pressure_ratio, gamma_function

    g, G = m.gamma, gamma_function(m.gamma)
    pe_pc = area_ratio_to_pressure_ratio(eps, g)
    p_safe = torch.clamp(p, min=1e-9)
    cf = G * np.sqrt(2 * g / (g - 1) * (1 - pe_pc ** ((g - 1) / g))) \
        + (pe_pc - m.p_ambient / p_safe) * eps
    if m.separation_frac > 0.0:
        x = torch.clamp(float(m.separation_frac) * m.p_ambient / p_safe,
                        max=1.0 - 1e-9)
        u = torch.clamp(1.0 - x ** ((g - 1) / g), min=1e-12)
        eps_eff = G / torch.sqrt(torch.clamp(
            2 * g / (g - 1) * x ** (2 / g) * u, min=1e-12))
        cf_sep = G * torch.sqrt(2 * g / (g - 1) * u) \
            + (x - m.p_ambient / p_safe) * eps_eff
        cf = torch.where((pe_pc * p_safe) < (float(m.separation_frac)
                                             * m.p_ambient), cf_sep, cf)
    return cf


def test_separation_frac_is_validated_as_a_zero_to_one_input():
    """0~1 밖은 거부한다 — 사용자 입력이므로 조용히 통과시키면 안 된다."""
    for bad in (-0.1, 1.5, 2.0):
        with pytest.raises(ValueError, match="0~1"):
            DiffMotor(d_outer=50.0, n_grid=61, separation_frac=bad)
    for ok in (0.0, 0.4, 1.0):
        DiffMotor(d_outer=50.0, n_grid=61, separation_frac=ok)


def test_implied_throat_follows_the_forward_cf_model():
    """⚠ **정방향 C_f 를 바꾸면 노즐목 역산도 같이 바뀌어야 한다.**

    `fp_ratio_spread` 는 `F/P` 를 C_f 로 나눠 노즐목을 역산한다. 정방향이
    박리를 보는데 역산이 안 보면 체계적으로 틀어진다 — 실측에서
    치우침이 **0.017 % → 9.77 %** 로 뛰었다. 그래서 호출부가
    `separation_frac=motor.separation_frac` 을 넘긴다. 이 시험이 그 연결을
    지킨다 (넘기는 것을 빼면 바로 깨진다).
    """
    from gre.optimize.loss import fp_ratio_spread

    cases = [(12.0, [0] * 6, 85.0), (12.0, [0] * 6, 54.0),
             (12.0, [0, 0, 2.2, 0, 0, 0], 78.0)]
    for frac in (0.0, 0.4):
        m = DiffMotor(d_outer=60.0, n_grid=121, n_samples=384, n_w=200,
                      h_max=10e-3, separation_frac=frac)
        rels = []
        for r0, arr, at in cases:
            a = torch.tensor(arr, dtype=torch.float64)
            with torch.no_grad():
                t, p, f, _ = m.simulate(torch.tensor(r0), a, torch.zeros(6),
                                        [95.0], torch.tensor(at))
            fp = fp_ratio_spread(t, f, p, gamma=m.gamma,
                                 p_ambient=m.p_ambient, cf_eff=m.cf_eff,
                                 separation_frac=m.separation_frac)
            rels.append((fp["at_mean"] - at) / at)
        assert max(abs(r) for r in rels) < 0.01, \
            f"frac={frac}: 치우침 {[f'{100*r:+.2f}%' for r in rels]} — " \
            f"정방향과 역산의 C_f 가 어긋났나"
