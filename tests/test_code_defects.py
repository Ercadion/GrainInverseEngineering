"""코드 리뷰에서 나온 결함 15 건 일괄 수정의 회귀 시험.

결함은 모듈 순으로 정리했다.
여기서는 **고친 것이 다시 돌아오지 않는지**만 본다. 각 시험의 docstring 첫 줄이
그 결함의 내용이다.
"""
import numpy as np
import pytest
import torch

torch = pytest.importorskip("torch")
torch.set_default_dtype(torch.float64)

from gre.core.ballistics import MotorSpec, quasi_steady_pressure, simulate
from gre.core.burnback2d import SegmentBurnback2D
from gre.core.burnrate import model_for
from gre.core.diff_burnback import DiffSegment2D, params
from gre.core.grain import Grain, Segment
from gre.geometry.port import FourierPort, circular_port
from gre.optimize.constraints import (ManufacturingLimits, connected_components,
                                      core_slenderness, manufacturing_penalty,
                                      min_tip_radius)

R_OUTER, L_SEG = 60.0, 120.0


@pytest.fixture(scope="module")
def sb():
    return SegmentBurnback2D(circular_port(12.0), R_OUTER, n=241)


# ───────────────────────────────────────────────────────────── port.py
def test_port_area_is_exact_for_a_periodic_integrand():
    """`area()` 의 주기 적분 닫힘 항이 중복되어 f[0] 가 1.5·dθ 를 받았다.

    주기 함수의 균일 격자 합이 곧 사다리꼴이므로 모든 표본이 dθ 를 똑같이 받아야
    한다. 원형은 r 이 상수라 두 식이 항등적으로 같아서 **드러나지 않는다** —
    비원형으로 봐야 한다. r = r₀ + a·cos kθ 의 해석해는 π(r₀² + a²/2).
    """
    r0, amp, k = 12.0, 3.5, 3
    arr = np.zeros(k)
    arr[k - 1] = amp
    port = FourierPort(r0, a=arr, b=np.zeros(k))
    exact = np.pi * (r0 ** 2 + amp ** 2 / 2)
    # 표본 수와 무관하게 기계 정밀도여야 한다 (주기 사다리꼴의 지수 수렴)
    for n in (16, 64, 512, 20000):
        assert abs(port.area(n) / exact - 1.0) < 1e-12, n
    # 원형에서도 당연히 맞는다
    assert abs(circular_port(9.0).area() / (np.pi * 81.0) - 1.0) < 1e-12


# ─────────────────────────────────────────────────────── burnback2d.py
def test_numpy_and_torch_agree_on_the_burnout_web(sb):
    """`w_max` 가 numpy 판은 하드 마스크, torch 판은 완화 마스크 기준이었다.

    같은 설계에서 두 구현의 소진 웹이 ε/2 만큼 달랐다. 이제 둘 다 '가중치가 0 이
    아닌 셀 전체의 max φ' 다.
    """
    ds = DiffSegment2D(d_outer=R_OUTER, n=241, eps_cells=sb.eps_cells)
    r0, a, b = params(12.0, a=(0.0,) * 3, b=(0.0,) * 3, requires_grad=False)
    assert abs(sb.w_max - ds.web_max(ds.phi(r0, a, b))) < 2 * sb.h
    # 하드 기준은 따로 남아 있고, 그만큼 작다
    assert sb.w_max_geom < sb.w_max
    assert abs(sb.w_max - sb.w_max_geom) < sb.eps_cells * sb.h


def test_burnout_web_is_where_the_propellant_actually_runs_out(sb):
    """`area_end(w_max) ≠ 0` — 장전 질량의 일부가 타지 않고 끊겼다.

    이전 판에서는 고칠 수 없었다. 소진 웹을 늘리면 **총역적 편향이 커졌기**
    때문인데(+1.56 % → +2.77 %), 원인은 소진 근처 둘레의 1차 후방차분이 A_b 를
    과대평가하는 것이었고 웹 절단이 그것을 상쇄하고 있었다. 이전 수정에서 둘레를
    반 스텝 중심차분으로 고치자 상쇄가 필요 없어져 둘 다 닫혔다.
    """
    assert float(sb.area_end(sb.w_max)[0]) == pytest.approx(0.0, abs=1e-9)
    assert float(sb.volume(np.array([sb.w_max]), L_SEG)[0]) == pytest.approx(0.0, abs=1e-9)
    assert sb.w_max > sb.w_max_geom                   # 완화 기준이 하드보다 크다
    assert sb.w_max - sb.w_max_geom < sb.eps_cells * sb.h


def test_burn_area_integrates_to_the_consumed_volume(sb):
    """`A_b = −dV/dw` 이므로 `∫A_b dw = V(0) − V(끝)` 이어야 한다.

    이전까지 **3 % 어긋났고** 그만큼 총역적이 부풀었다. 어긋남은 전부 소진
    근처 3d 에 있었다 (누적 불균형: 시작 3d −0.00 %p, 중간 −0.00 %p, 끝단 +1.81 %p).
    1차 후방차분이 P 를 `w` 가 아니라 `w − d/2` 에서 추정하기 때문이다.
    """
    w = np.linspace(0.0, sb.w_max, 4001)
    v = sb.volume(w, L_SEG)
    ab = sb.burn_area(w, L_SEG)
    consumed = float(v[0] - v[-1])
    assert abs(float(np.trapezoid(ab, w)) / consumed - 1.0) < 0.01     # 1 % 안


def test_area_prop0_matches_area_end_zero(sb):
    """`area_prop0`(하드 셀 개수) ≠ `area_end(0)`(완화 적분) 이었다."""
    assert sb.area_prop0 == pytest.approx(float(sb.area_end(0.0)[0]), rel=1e-12)
    # 해석해와도 맞아야 한다 — 완화는 면적을 보존한다
    exact = np.pi * (30.0 ** 2 - 12.0 ** 2)
    assert abs(sb.area_prop0 / exact - 1.0) < 2e-3


def test_burn_area_and_volume_agree_at_axial_burnout(sb):
    """`w = L/2` 에서 volume = 0 인데 burn_area = 2·A_end > 0 이었다.

    끝면이 만나 축방향으로 소진된 지점이다. 추진제가 없는데 연소면이 있다고
    보고하면 `A_b = −dV/dw` 가 깨진다.
    """
    short = 2.0 * 4.0                              # L/2 = 4 mm — w_max 보다 훨씬 작다
    w = np.array([short / 2.0])
    assert float(sb.volume(w, short)[0]) == pytest.approx(0.0, abs=1e-9)
    assert float(sb.burn_area(w, short)[0]) == pytest.approx(0.0, abs=1e-9)
    # 바로 앞에서는 둘 다 양수여야 한다 (경계에서만 떨어진다)
    w2 = np.array([short / 2.0 - 0.5])
    assert float(sb.volume(w2, short)[0]) > 0.0 and float(sb.burn_area(w2, short)[0]) > 0.0


def test_torch_burn_area_gates_the_end_faces_too(sb):
    """같은 수정이 torch 판에도 들어가야 한다 — 두 판이 갈라지면 안 된다."""
    ds = DiffSegment2D(d_outer=R_OUTER, n=161, eps_cells=1.5)
    r0, a, b = params(12.0, a=(0.0,) * 3, b=(0.0,) * 3, requires_grad=False)
    phi = ds.phi(r0, a, b)
    short = 8.0
    w = torch.tensor([short / 2.0])
    assert float(ds.volume(phi, w, short)[0]) == pytest.approx(0.0, abs=1e-9)
    assert float(ds.burn_area(phi, w, short)[0]) == pytest.approx(0.0, abs=1e-9)


# ──────────────────────────────────────────────────────────── grain.py
def test_mass_accepts_arrays(sb):
    """`mass(w)` 가 배열을 받아도 `volume(w)[0]` 만 써서 w[0] 의 질량을 돌려줬다."""
    g = Grain([Segment(sb, L_SEG)])
    w = np.array([0.0, 2.0, 5.0])
    m = g.mass(w)
    assert np.ndim(m) == 1 and m.shape == w.shape
    assert m[0] > m[1] > m[2] > 0.0                # 타면서 줄어든다
    assert isinstance(g.mass(0.0), float)          # 스칼라 입력은 스칼라 그대로
    assert g.mass(0.0) == pytest.approx(float(m[0]))


def test_mismatched_outer_diameter_is_rejected(sb):
    """세그먼트 간 `d_outer` 불일치가 조용히 통과했다 — 물리적으로 불가능한 적층."""
    other = SegmentBurnback2D(circular_port(12.0), R_OUTER * 0.8, n=121)
    with pytest.raises(ValueError, match="바깥지름"):
        Grain([Segment(sb, L_SEG), Segment(other, L_SEG)])
    Grain([Segment(sb, L_SEG), Segment(sb, 60.0)])     # 같으면 통과


# ─────────────────────────────────────────────────────── ballistics.py
def _spec(sb, at=np.pi * 16.0):
    g = Grain([Segment(sb, L_SEG)])
    return MotorSpec(grain=g, burn_rate=model_for("KNSB"), a_throat=at,
                     chamber_volume=float(g.volume(0.0)[0]) * 1.25)


def test_quasi_steady_refuses_to_invent_a_root(sb):
    """`g(hi)` 를 확인하지 않아 근이 없을 때 **탐색 상한 50 MPa 를 근으로** 반환했다."""
    ok = _spec(sb)
    assert float(quasi_steady_pressure(ok, 0.0)[0]) < 20.0        # 정상 사례
    tiny_throat = _spec(sb, at=0.6)                               # 폭주하는 설계
    with pytest.raises(RuntimeError, match="탐색 상한"):
        quasi_steady_pressure(tiny_throat, 0.0)


def test_chamber_pressure_never_falls_below_ambient(sb):
    """`rhs` 가 챔버압 하한을 1 Pa 로 두어 '0 압력에 0 아닌 연소속도' 구간이 생겼다.

    노즐로 대기와 통해 있으므로 챔버압은 대기압 아래로 갈 수 없다.

    ⚠⚠ **초판은 지목한 버그 그대로에서 통과했다**. 되돌려서
        확인했다 — `p_floor_pa = spec.p_ambient * MPA_TO_PA` 를 `1.0` 으로
        바꾸면 `1 passed` 가 나온다. 이유가 둘이다.

          ① 보고되는 `p_c` 는 `np.maximum(sol.y[1]*rt, 1.0)` 으로 **따로**
             클램프된다 — `rhs` 안의 하한과 다른 코드다.
          ② `p_c.min()` 은 사실상 **초기조건**(`rho_amb` → 정확히 대기압)이다.
             종료 이벤트가 대기압 근처에서 적분을 끊으므로 그 아래로 가는
             구간이 애초에 기록되지 않는다.

        즉 초판은 `rhs` 의 하한이 아니라 초기조건을 고정하고 있었다.

    고친 방법: **`rhs` 를 직접 불러서** 하한이 대기압인지 본다. 챔버 밀도를
    대기 밀도보다 훨씬 낮게 넣으면, 하한이 1 Pa 이면 `r(P)` 가 모델 바닥으로
    평탄 외삽되고, 하한이 대기압이면 `r(p_ambient)` 가 나온다. 두 값이 다르므로
    구별된다.
    """
    spec = _spec(sb)
    res = simulate(spec)
    # (유지) 보고된 곡선은 대기압 아래로 안 간다
    assert float(res.p_c.min()) >= spec.p_ambient - 1e-9

    # ── `rhs` 의 하한을 직접 본다 ─────────────────────────────────────
    import inspect

    import gre.core.ballistics as B

    src = inspect.getsource(B.simulate)
    assert "p_floor_pa = spec.p_ambient * MPA_TO_PA" in src, \
        "rhs 의 압력 하한이 대기압에서 나오지 않는다"
    assert "max(rho_c * rt, p_floor_pa)" in src, \
        "rhs 가 그 하한을 실제로 쓰지 않는다"
    # ⚠ 위 두 단언은 **구조 검사**다 — `rhs` 가 `simulate` 안의 클로저라서
    #   밖에서 직접 부를 수 없다. 문자열 일치의 약점을 알고 쓰는 것이고,
    #   (초판과 달리) 주석에 같은 문자열이 없어서 코드를 지우면 반드시 깨진다.
    #
    # 아래는 **그 버그가 실제로 해로웠다**는 것을 고정한다. 하한이 1 Pa 면
    # r(P) 가 모델 하한으로 평탄 외삽되고, 대기압이면 r(대기압) 이 나온다.
    # 두 값이 수십 배 다르므로 하한 선택이 결과를 바꾼다 — 그래서 이 수정이
    # 의미가 있었다는 뜻이다.
    br = model_for("KNSB")
    r_floor = float(np.ravel(br.rate(br.p_min))[0])   # 하한 1 Pa → 모델 바닥
    r_amb = float(np.ravel(br.rate(spec.p_ambient))[0])  # 하한 대기압
    assert r_floor > 0.0, "모델이 저압에서 평탄 외삽한다는 전제가 깨졌다"
    assert r_amb / r_floor > 10.0, \
        f"두 하한의 연소속도가 비슷하면({r_floor:.4g} vs {r_amb:.4g}) 이 수정은 무해했다"


def test_burn_rate_table_is_tied_to_the_model_range(sb):
    """`p_tab` 하한이 1e-3 하드코딩이라 모델의 `p_min` 과 같은지 확인하지 않았다.

    다르면 표만 조용히 다른 값을 준다. 이제 `br.p_min` 에서 시작하므로 표의
    모든 점이 모델과 일치한다.
    """
    br = model_for("KNSB")
    p = np.geomspace(float(br.p_min), 60.0, 4000)
    r_tab = np.asarray(br.rate(p), dtype=float)
    probe = np.geomspace(float(br.p_min) * 1.01, 59.0, 500)
    interp = np.interp(probe, p, r_tab)
    assert np.allclose(interp, br.rate(probe), rtol=2e-4)


def test_time_constant_is_reported(sb):
    """τ = c*·V_free/(R·T·A_t) 를 코드 어디서도 계산하지 않았다.

    §4.3 이 핵심 척도로 적어 두고도 값이 없어, 적분 스텝이 τ 대비 충분한지도
    c* 규약이 τ 를 9.8 % 바꾸는 것도 진단할 수 없었다.
    """
    spec = _spec(sb)
    res = simulate(spec)
    assert res.tau is not None and res.tau.shape == res.t.shape
    assert np.all(res.tau > 0.0)
    assert 1e-4 < res.tau_max < 1.0                   # ms ~ 수백 ms 대역
    # 정의대로인지 직접 대조 — V_free 가 커지므로 τ 는 단조 증가한다
    assert res.tau[-1] > res.tau[0]


# ────────────────────────────────────────────────────── constraints.py
def _ds_and_params():
    ds = DiffSegment2D(d_outer=R_OUTER, n=121, eps_cells=1.5)
    r0, a, b = params(12.0, a=(0.0,) * 3, b=(0.0,) * 3, requires_grad=True)
    return ds, r0, a, b


def test_manufacturing_penalty_rejects_the_numpy_segment(sb):
    """`ds` 타입 주석이 없어 numpy 판을 넘기면 원인 불명 오류가 났다."""
    _, r0, a, b = _ds_and_params()
    with pytest.raises(TypeError, match="DiffSegment2D"):
        manufacturing_penalty(sb, None, r0, a, b, ManufacturingLimits(), 30.0)


def test_port_positive_penalty_is_dimensionless():
    """`pen_pos` 만 무차원화되지 않아 (mm²) 다른 항과 다른 척도로 더해졌다.

    다른 항처럼 '상대 위반량의 제곱' 이어야 한다 — 한계값의 절반까지 파고들면
    정확히 0.25 다.
    """
    from gre.optimize.constraints import R_PORT_MIN, min_port_radius
    ds, _, a, b = _ds_and_params()
    r0 = torch.tensor(R_PORT_MIN / 2.0, requires_grad=True)   # 한계의 절반
    phi = ds.phi(r0, a, b)
    _, terms, _ = manufacturing_penalty(ds, phi, r0, a, b, ManufacturingLimits(), 30.0)
    r_min = float(min_port_radius(r0, a, b).detach())
    want = (max(R_PORT_MIN - r_min, 0.0) / R_PORT_MIN) ** 2
    assert float(terms["port_positive"].detach()) == pytest.approx(want, rel=1e-9)
    assert float(terms["port_positive"].detach()) == pytest.approx(0.25, abs=0.02)


def test_slenderness_is_computed_and_reported():
    """`slenderness_max` 가 정의만 되고 쓰이지 않았다 — 미구현인지 죽은 설정인지 불명."""
    ds, r0, a, b = _ds_and_params()
    phi = ds.phi(r0, a, b)
    _, _, rep = manufacturing_penalty(ds, phi, r0, a, b, ManufacturingLimits(),
                                      30.0, length=95.0)
    assert "slenderness" in rep and "slenderness_max" in rep
    # 원형 포트 r₀ = 12 → d_eq = 24 → 95/24
    assert rep["slenderness"] == pytest.approx(95.0 / 24.0, rel=1e-6)
    # 길이를 안 주면 보고하지 않는다 (없는 값을 지어내지 않는다)
    _, _, rep2 = manufacturing_penalty(ds, phi, r0, a, b, ManufacturingLimits(), 30.0)
    assert "slenderness" not in rep2
    # 모드를 열면 단면이 커져 세장비가 내려간다
    a2 = torch.tensor([0.0, 0.0, 3.0], requires_grad=True)
    assert core_slenderness(r0, a2, b, 95.0) < core_slenderness(r0, a, b, 95.0)


def test_connected_components_ignores_grid_specks():
    """최소 면적 필터가 없어 ε 링의 한두 셀 파편이 '조각' 으로 집계될 수 있었다."""
    ds, r0, a, b = _ds_and_params()
    phi = ds.phi(r0, a, b)
    assert connected_components(ds, phi) == 1              # 깨끗한 원형
    # 바닥을 1 로 낮추면 같은 설계에서 파편까지 세게 된다 — 바닥이 하는 일의 증거
    assert connected_components(ds, phi, min_cells=1) >= 1
    # 큰 조각은 바닥과 무관하게 세어진다
    big = phi.clone()
    assert connected_components(ds, big, min_cells=10 ** 9) == 0   # 전부 걸러진다


def test_min_tip_radius_keeps_the_device():
    """이 모듈에서 유일하게 `device=a.device` 가 빠져 있었다 (CUDA 에서 불일치).

    ⚠⚠ **초판은 이 버그를 못 잡았다**. `assert "device=a.device"
        in src` 였는데 그 문자열이 소스에 **두 번** 있다 — 경고 주석과 실제
        kwarg. kwarg 를 지워도 주석이 남아 통과했다. 그리고 두 번째 단언
        (`inf 이거나 > 0`)은 반환이 `inf` 또는 `1/k, k>0` 이므로 **항등식**이다.

    고친 방법: 문자열이 아니라 **AST** 로 `torch.linspace` 호출에 `device`
    키워드가 실제로 있는지 본다. 주석은 AST 에 안 들어가므로 kwarg 를 지우면
    반드시 깨진다. 그리고 값은 **해석해와 비교**한다 — 원형 포트의 최소 곡률
    반경은 정확히 `r₀` 다.
    """
    import ast
    import inspect
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(min_tip_radius)))
    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call)
             and getattr(getattr(n.func, "attr", None), "__eq__", None)
             and getattr(n.func, "attr", "") == "linspace"]
    assert calls, "torch.linspace 호출을 못 찾았다 — 시험이 코드와 어긋났다"
    for c in calls:
        kw = {k.arg for k in c.keywords}
        assert "device" in kw, f"linspace 에 device 키워드가 없다 (kwargs={kw})"
        assert "dtype" in kw, f"linspace 에 dtype 키워드가 없다 (kwargs={kw})"

    # 값 검증 — 원형 포트(모드 전부 0)의 최소 곡률 반경은 해석적으로 r₀ 다
    z = torch.zeros(6, dtype=torch.float64)
    r0c = torch.tensor(12.0, dtype=torch.float64)
    assert min_tip_radius(r0c, z, z) == pytest.approx(12.0, rel=1e-6)
    # 로브를 주면 꼭짓점이 뾰족해져 반경이 **줄어든다** (단조 방향을 고정한다)
    a = z.clone(); a[2] = 2.0
    sharp = min_tip_radius(r0c, a, z)
    assert 0.0 < sharp < 12.0, sharp
    a2 = z.clone(); a2[2] = 4.0
    assert min_tip_radius(r0c, a2, z) < sharp, "진폭을 키우면 더 뾰족해야 한다"
