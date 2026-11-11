"""입력 자동 검증 루프 (§4.6.3) — `gre.validate.input_check`.

이 파일에서 가장 중요한 두 시험은 맨 아래 있다.

    test_pressure_inversion_matches_the_forward_model
    test_burn_area_inversion_matches_the_forward_model

검증 루프 전체가 **곡선을 역으로 푸는 것** 위에 서 있다. 역산이 정방향 모델과
어긋나면 나머지 검사는 전부 엉뚱한 것을 재게 된다. 그 두 시험이 역산을 정방향
모델에 못질한다.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from gre.optimize.constraints import ManufacturingLimits
from gre.optimize.forward import DiffMotor
from gre.optimize.loss import Target
from gre.validate import input_check as ic
from gre.validate.input_check import (A_THROAT_MIN, FATAL, WARN,
                                      check_consistency, check_format,
                                      check_inputs, check_physics)


# ════════════════════════════════════════════════════════════════════ 준비
@pytest.fixture(scope="module")
def motor():
    return DiffMotor(d_outer=60.0, n_grid=121)


@pytest.fixture(scope="module")
def limits():
    return ManufacturingLimits()


@pytest.fixture(scope="module")
def ref(motor):
    """원형 포트 r₀=14, A_t=80 의 **정방향 결과**. 자기 일관된 목표다."""
    z = torch.zeros(8, dtype=motor.dtype)
    t, p, f, diag = motor.simulate(torch.tensor(14.0, dtype=motor.dtype), z, z,
                                   [95.0], torch.tensor(80.0, dtype=motor.dtype))
    return {"t": t.detach().numpy(), "p": p.detach().numpy(),
            "f": f.detach().numpy(), "diag": diag}


def _codes(findings, ok=False):
    return [f.code for f in findings if f.ok is ok]


def _get(findings, code):
    return next(f for f in findings if f.code == code)


# ══════════════════════════════════════════════════════════════ 1. 형식 검사
def test_clean_curve_passes_every_format_check(ref):
    """정방향 모델의 출력은 형식 검사를 통과해야 한다 — 상승 분해만 빼고.

    ⚠ `FMT_RISE_RES` 는 **정방향 출력에서도 걸린다** (점화 상승 7.9 ms 에 표본
       3 개). 거짓 양성이 아니라 사실이다 — 모델의 점화 상승은 표본 3 개로
       분해된다. 모양 창이 그 구간을 사양으로 세지 않으므로 손실에는 영향이
       없지만, 그 사실이 숨지 않도록 경고로 남겨 둔다.
    """
    fs = check_format(ref["t"], ref["f"])
    assert not [f for f in fs if not f.ok and f.level == FATAL]
    assert _codes(fs) == ["FMT_RISE_RES"]


def test_length_mismatch_and_too_few_samples_stop_everything():
    fs = check_format([0.0, 1.0, 2.0], [1.0, 2.0])
    assert len(fs) == 1 and fs[0].code == "FMT_LEN" and fs[0].level == FATAL
    fs = check_format([0.0, 1.0], [1.0, 2.0])
    assert len(fs) == 1 and fs[0].code == "FMT_LEN"


def test_nonfinite_stops_before_physics():
    t = np.linspace(0, 1, 50)
    f = np.full(50, 10.0)
    f[7] = np.nan
    fs = check_format(t, f)
    assert _get(fs, "FMT_FINITE").level == FATAL
    assert not _get(fs, "FMT_FINITE").ok
    # NaN 뒤로는 검사를 더 하지 않는다 — 깨진 배열 위의 진단은 틀린 곳을 가리킨다
    assert [f.code for f in fs] == ["FMT_FINITE"]


def test_time_reversal_is_fatal_and_counted():
    t = np.concatenate([np.linspace(0, 1, 30), np.linspace(0.5, 1.5, 30)])
    f = np.full(60, 10.0)
    fs = check_format(t, f)
    g = _get(fs, "FMT_TIME")
    assert g.level == FATAL and not g.ok
    assert g.data["n_backward"] == 1


def test_duplicate_timestamps_are_fatal():
    t = np.array([0.0, 0.1, 0.1, 0.2, 0.3, 0.4])
    fs = check_format(t, np.full(6, 5.0))
    assert _get(fs, "FMT_TIME").data["n_dup"] == 1
    assert not _get(fs, "FMT_TIME").ok


def test_negative_thrust_severity_splits_on_size(ref):
    """영점 치우침(작은 음수)과 틀린 곡선(큰 음수)을 **등급으로** 가른다.

    둘을 한 등급으로 묶으면 둘 중 하나가 잘못 처리된다 — 영점 치우침은
    고칠 수 있는 입력 사고라 멈출 이유가 없고, 큰 음수는 멈춰야 한다.
    """
    f_max = ref["f"].max()
    small = _get(check_format(ref["t"], ref["f"] - 0.01 * f_max), "FMT_SIGN")
    assert small.level == WARN and not small.ok

    big = ref["f"].copy()
    big[:50] = -0.3 * f_max
    assert _get(check_format(ref["t"], big), "FMT_SIGN").level == FATAL


def test_noise_estimator_recovers_the_injected_sigma():
    """2차 차분 추정자가 심어 넣은 σ 를 되찾는가 — 추정자 자체의 시험.

    백색잡음이면 Var(Δ²f) = 6σ² 이므로 σ̂ = 1.4826·median|Δ²f|/√6 이다.
    매끄러운 성분은 Δ² 에서 h² 로 줄어들어 거의 남지 않는다.
    """
    rng = np.random.default_rng(0)
    t = np.linspace(0, 2, 4000)
    base = 200.0 * np.exp(-0.3 * t)          # 매끄러운 곡선
    for sigma in (0.5, 2.0, 8.0):
        g = _get(check_format(t, base + rng.normal(0, sigma, t.size)), "FMT_NOISE")
        assert g.data["sigma_n"] == pytest.approx(sigma, rel=0.10), \
            f"σ={sigma} 를 {g.data['sigma_n']:.3f} 로 추정했다"


def test_noise_smoothing_suggestion_scales_as_m_equals_ratio_squared():
    """제안 창 길이는 σ_ma = σ/√m 에서 나온다 — m = (frac/문턱)²."""
    rng = np.random.default_rng(1)
    t = np.linspace(0, 2, 4000)
    base = np.full(t.size, 100.0)
    g = _get(check_format(t, base + rng.normal(0, 4.0, t.size)), "FMT_NOISE")
    frac = g.data["sigma_frac"]
    assert g.data["ma_suggest"] == int(np.ceil((frac / ic.NOISE_FRAC_WARN) ** 2))
    assert not g.ok and frac > ic.NOISE_FRAC_WARN


def test_uneven_sampling_and_truncated_record_are_warnings(ref):
    t = ref["t"].copy()
    t[300:] += np.linspace(0, 0.5, t.size - 300)     # 뒤를 늘린다
    assert not _get(check_format(t, ref["f"]), "FMT_DT").ok
    # 기록이 끊긴 곡선 — 테일오프가 안 들어왔다
    n = int(0.67 * ref["t"].size)
    assert not _get(check_format(ref["t"][:n], ref["f"][:n]), "FMT_TAIL").ok


def test_pressure_curve_must_be_absolute_mpa(ref):
    fs = check_format(ref["t"], ref["f"], ref["p"] - ref["p"].min())
    assert _get(fs, "FMT_P_SIGN").level == FATAL
    assert _get(check_format(ref["t"], ref["f"], ref["p"]), "FMT_P").ok


# ════════════════════════════════════════════════════════════ 2. 입력 일관성
def test_contradictory_scalars_are_caught(ref):
    t = torch.tensor(ref["t"])
    f = torch.tensor(ref["f"])
    i_curve = float(np.trapezoid(ref["f"], ref["t"]))

    ok = check_consistency(Target(t=t, f=f))
    assert not [x for x in ok if not x.ok]

    bad = check_consistency(Target(t=t, f=f, total_impulse=1.5 * i_curve))
    g = _get(bad, "CON_TOTAL_IMPULSE")
    assert not g.ok and g.level == WARN and g.data["rel"] == pytest.approx(0.5, rel=0.02)

    bad = check_consistency(Target(t=t, f=f, f_max=0.5 * float(ref["f"].max())))
    assert not _get(bad, "CON_F_MAX").ok


def test_shape_factor_flags_a_neutral_curve():
    """F_max/F_avg ≈ 1 은 **형상 정보가 없다**는 신호다 (§4.6.3 식별성)."""
    t = np.linspace(0, 2, 400)
    flat = np.full(400, 100.0)
    flat[0] = flat[-1] = 0.0
    g = _get(check_consistency(Target(t=torch.tensor(t), f=torch.tensor(flat))),
             "CON_SHAPE")
    assert g.data["shape_factor"] < 1.15
    assert "형상 정보" in g.detail


# ══════════════════════════════════════════════════════════════ 3. 물리 검사
def test_operating_window_matches_the_pressure_based_rule(motor, ref):
    """추력으로 잡은 창이 `fp_ratio_spread` 의 **압력** 규칙과 같은 구간인가.

    검증 루프에는 압력이 아직 없어서 추력으로 창을 잡는다. 창 안에서 `C_f` 가
    거의 일정하므로 `F ∝ P` 이고 두 규칙이 같은 구간을 골라야 한다 — 이
    시험이 그 전제를 못질한다. 어긋나면 검사와 손실이 다른 구간을 보게 되고,
    그 어긋남이 이전 판의 64.5 % 거짓 거부를 만든 바로 그 실수다.
    """
    p, f = ref["p"], ref["f"]
    hi = p[p > 0.5 * p.max()]
    plateau = float(np.median(hi))
    # 같은 **구간** 의미로 압력축에서 다시 잡는다 (준위 마스크가 아니다)
    lit = np.flatnonzero(p >= ic.IGNITION_FRAC * plateau)
    alive = np.flatnonzero(f > 0.05 * f.max())
    by_p = np.zeros(f.size, dtype=bool)
    by_p[int(lit[0]):int(alive[-1]) + 1] = True
    by_f = ic.operating_window(f)

    jaccard = (by_p & by_f).sum() / max((by_p | by_f).sum(), 1)
    assert jaccard > 0.99, f"두 창의 일치도가 {jaccard:.3f} 뿐이다"


def test_window_is_an_interval_not_a_level_mask_for_boost_sustain():
    """부스트-서스테인 곡선에서 **구간**과 **준위 마스크**가 갈린다.

    `loss.fp_ratio_spread` 는 같은 문턱을 준위 마스크로 쓴다. 평탄값이 부스트
    쪽에서 잡히므로 마스크는 **서스테인 구간을 통째로 버린다** — F/P 모순
    검사가 두 곡선이 어긋날 수 있는 바로 그 구간을 안 본다는 뜻이다.
    §4.6.2 가 부스트-서스테인을 입력 템플릿으로 명시하므로 가정이 아니라
    실제 입력이다.

    이 시험은 `input_check` 쪽이 **구간**을 쓴다는 것을 못질하고, 동시에
    `fp_ratio_spread` 의 결함을 숫자로 남긴다.
    """
    t = np.linspace(0, 2, 2000)
    f = np.where(t < 0.5, 200.0, 60.0)
    f[0] = f[-1] = 0.0

    hi = f[f > 0.5 * f.max()]
    plateau = float(np.median(hi))
    mask = (f >= ic.IGNITION_FRAC * plateau) & (f > 0.05 * f.max())
    win = ic.operating_window(f)

    # 마스크는 부스트만 남긴다 — 서스테인(60 N)이 전부 빠진다
    assert mask[f == 60.0].sum() == 0
    assert mask.sum() == pytest.approx(0.25 * f.size, rel=0.05)
    # 구간은 둘 다 포함한다
    assert win[f == 60.0].all()
    assert win.sum() > 0.99 * (f.size - 2)


def test_pressure_inversion_matches_the_forward_model(motor, ref):
    """F = C_f·P·A_t 역산이 정방향 모델의 압력을 되찾는가.

    **검증 루프 전체의 기반**이다. 작동 구간(평형압 90 % 이상)에서만 본다 —
    역산은 준정상이고 점화·테일오프에는 챔버 충전 항이 빠져 있다.
    """
    f, p = ref["f"], ref["p"]
    win = f >= 0.9 * f.max()
    cf = ic._cf_rep(motor, float(np.median(p[win])))
    p_inv = ic._pressure_from_thrust(motor, f, 80.0, cf)
    rel = np.abs(p_inv[win] - p[win]) / p[win]
    assert np.median(rel) < 0.01, f"중앙 오차 {100*np.median(rel):.2f} %"
    assert rel.max() < 0.03, f"최대 오차 {100*rel.max():.2f} %"


def test_burn_area_inversion_matches_the_forward_model(motor, ref):
    """ρ_p·A_b·r = P·A_t/c* 역산이 정방향 `A_b` 표를 되찾는가."""
    diag = ref["diag"]
    a_b = diag["a_b"].detach().numpy()
    w_tab = diag["w_tab"].detach().numpy()
    r_of_t = ic._rate(motor, ref["p"])
    traj = np.concatenate([[0.0], np.cumsum(r_of_t[:-1] * float(diag["h"]))])
    ab_fwd = np.interp(np.clip(traj, 0, w_tab[-1]), w_tab, a_b)
    ab_inv = ic._ab_required(motor, ref["p"], 80.0)

    win = ref["f"] >= 0.9 * ref["f"].max()
    rel = np.abs(ab_inv[win] - ab_fwd[win]) / ab_fwd[win]
    assert np.median(rel) < 0.02, f"중앙 오차 {100*np.median(rel):.2f} %"
    assert rel.max() < 0.05, f"최대 오차 {100*rel.max():.2f} %"


def test_self_consistent_target_passes_physics(motor, limits, ref):
    tgt = Target(t=torch.tensor(ref["t"]), f=torch.tensor(ref["f"]))
    for at in (None, 80.0):
        fs = check_physics(tgt, motor, limits, (95.0,), a_throat=at)
        bad = [x for x in fs if not x.ok and x.level == FATAL]
        assert not bad, f"A_t={at}: {[x.code for x in bad]}"


def test_web_overrun_is_fatal_in_both_throat_modes(motor, limits, ref):
    """연소 시간을 5 배로 늘리면 케이스에 있는 웹보다 많이 태운다.

    ⚠ **자유 목 모드에서도 걸려야 한다.** 초판은 '가장 관대한 목' 으로 πR² 을
       써서 압력을 0.1 MPa 아래로 내렸고, 그러면 연소속도가 사실상 0 이 되어
       **웹 검사가 공허해졌다** — 이 목표가 그냥 통과했다. 목을 압력 표와
       연소면적 도달 범위로 묶어 고쳤다. 이 시험이 재발을 막는다.
    """
    tgt = Target(t=torch.tensor(ref["t"] * 5.0), f=torch.tensor(ref["f"]))
    for at in (None, 80.0):
        g = _get(check_physics(tgt, motor, limits, (95.0,), a_throat=at),
                 "PHYS_WEB")
        assert not g.ok and g.level == FATAL, f"A_t={at} 에서 웹 검사가 안 걸렸다"
        assert g.data["web_used"] > g.data["web_avail"]


def test_throat_range_window_is_not_vacuous(motor, limits, ref):
    """노즐목 입력이 없을 때 알려 주는 **가능한 목 범위**가 공허하지 않아야 한다.

    고른 목은 압력이 연소속도 표 안이고 요구 연소면적이 족의 도달 범위 안이다.
    (최적화기의 '자유 목 모드' 는 삭제됐다 — 이 검사는 사용자가
    노즐목을 **무엇으로 넣어야 하는지** 알려 주는 쪽으로 남아 있다.)
    """
    tgt = Target(t=torch.tensor(ref["t"]), f=torch.tensor(ref["f"]))
    fs = check_physics(tgt, motor, limits, (95.0,))
    g = _get(fs, "PHYS_THROAT_RANGE")
    assert g.ok and g.data["n_candidates"] > 0
    lo, hi = g.data["at_window"]
    assert A_THROAT_MIN <= lo <= g.data["at_used"] <= hi
    assert _get(fs, "PHYS_PBAND").ok


def test_mid_burn_vertical_step_is_fatal_but_ignition_step_is_not(motor, limits):
    """작동 구간의 수직 단차는 거부하고, 수직 **점화** 상승은 거부하지 않는다.

    점화는 모델 범위 밖이고(§4.3) 모양 창도 사양으로 세지 않는다. 점화 상승을
    검사에 넣으면 **사양이 아닌 구간 때문에 멀쩡한 목표를 거부**한다 — 초판이
    그랬고, 고정 목 Ø4 mm 에서 자기 일관된 목표가 '확정 불가' 로 나왔다.
    그 대신 점화 기울기는 숫자로 보여 준다.
    """
    # 작동 구간에서 F_max 의 거의 전부를 한 표본에 떨어뜨린다 — 걸려야 한다.
    # ⚠ **표본 간격이 τ 보다 짧아야** 판정이 가능하다 (아래
    #   `test_rise_check_cannot_judge_when_dt_exceeds_tau` 가 그 반대를 고정).
    #   그래서 여기서는 20000 점으로 촘촘히 준다.
    # 서스테인은 0.05·F_max 위여야 한다 — 그 아래는 연소 종료(테일오프)로
    # 읽히고 작동 구간에서 빠진다. 60 N 은 F_max 의 30 % 라 안전하다.
    t = np.linspace(0, 2, 20000)
    step = np.where(t < 0.5, 200.0, 60.0)
    step[0] = step[-1] = 0.0
    g = _get(check_physics(Target(t=torch.tensor(t), f=torch.tensor(step)),
                           motor, limits, (95.0,), a_throat=80.0), "PHYS_RISE")
    assert g.data["dt_over_tau"] < 1.0, "이 시험은 판정 가능한 표본 간격을 전제한다"
    assert not g.ok and g.level == FATAL
    assert g.data["dfdt_op"] > g.data["dfdt_cap"]

    # 같은 단차를 100 ms 에 걸쳐 눕히면 통과한다
    laid = np.where(t < 0.5, 200.0,
                    np.where(t < 0.6, 200.0 - 1400.0 * (t - 0.5), 60.0))
    laid[0] = laid[-1] = 0.0
    assert _get(check_physics(Target(t=torch.tensor(t), f=torch.tensor(laid)),
                              motor, limits, (95.0,), a_throat=80.0),
                "PHYS_RISE").ok


def test_rise_check_cannot_judge_when_dt_exceeds_tau(motor, limits):
    """`dt ≥ τ` 면 이 검사는 **어떤 곡선도 거부할 수 없다** — 그러면 통과라고
    적지 않고 '판정 불가' 로 등급을 내린다.

    걸리는 조건이 `ΔF/F_max > dt/τ` 이므로 `dt/τ ≥ 1` 이면 구조적으로 불가능
    하다 (한 표본 사이에 F_max 를 넘게 변할 수는 없다). 그때 FATAL 등급으로
    "통과" 를 적으면 **검사한 적이 없는 것을 검사했다고 적는 것**이다.
    τ 가 서브밀리초라 이쪽이 오히려 일반적인 경우다.
    """
    t = np.linspace(0, 2, 1500)                      # dt = 1.33 ms
    step = np.where(t < 0.5, 200.0, 60.0)
    step[0] = step[-1] = 0.0
    g = _get(check_physics(Target(t=torch.tensor(t), f=torch.tensor(step)),
                           motor, limits, (95.0,), a_throat=80.0), "PHYS_RISE")
    assert g.data["dt_over_tau"] >= 1.0
    assert g.level == ic.INFO and "판정할 수 없습니다" in g.title
    assert "어떤 곡선도" in g.detail


def test_vertical_ignition_and_tailoff_are_never_rejected(motor, limits):
    """수직 **점화 상승**과 수직 **테일오프**로 목표를 거부하지 않는다.

    둘 다 모델 범위 밖이고(§4.3) 모양 창도 사양으로 세지 않는다. 거부하면
    사양이 아닌 구간 때문에 멀쩡한 목표를 버리는 것이다.

    ⚠ 이 모듈의 두 번째 판은 점화만 뺐고 **테일오프를 안 뺐다.** 그래서
       아래 곡선(수직 상승 + 수직 테일오프)이 '확정 불가' 로 나왔다 — 거짓
       양성이었다. 창 선택을 `operating_window` 하나로 통일해서 고쳤다.
       표본을 촘촘히 줘서 **판정이 가능한데도** 거부되지 않음을 보인다.
    """
    t = np.linspace(0, 2, 20000)
    ramp = np.where(t < 0.001, 0.0, 150.0)
    ramp[-1] = 0.0
    g = _get(check_physics(Target(t=torch.tensor(t), f=torch.tensor(ramp)),
                           motor, limits, (95.0,), a_throat=80.0), "PHYS_RISE")
    assert g.data["dt_over_tau"] < 1.0, "판정 가능한 표본 간격을 전제한다"
    assert g.ok, "점화·테일오프로 목표를 거부하면 안 된다"
    # 창 밖 기울기는 한계를 넘는다 — 그래도 거부하지 않는다는 게 요점이다
    assert g.data["dfdt_out"] > g.data["dfdt_cap"]
    assert g.data["dfdt_op"] == pytest.approx(0.0, abs=1.0)


def test_rise_check_tau_is_the_forward_models_formula(motor, limits, ref):
    """챔버 시상수 τ 가 **정방향 모델과 같은 식**인가.

    초판은 `τ = V_free/(c*·A_t)` 를 썼다. 차원은 맞지만 식이 다르고, 같은
    V_free·A_t 에서 2 배 이상 커진다 — FATAL 등급 검사의 문턱이 그만큼
    움직인다. 전면 검토에서 잡았다.
    """
    g = _get(check_physics(Target(t=torch.tensor(ref["t"]),
                                  f=torch.tensor(ref["f"])),
                           motor, limits, (95.0,), a_throat=80.0), "PHYS_RISE")
    assert g.data["shortest_transition_s"] == pytest.approx(3 * g.data["tau_s"])
    # ── τ 가 **정방향 모델의 식**인지 확인한다 (`forward.py` 의 `tau_w`):
    #       τ = c*·V_free/(R·T·A_t)
    #    초판은 `τ = V_free/(c*·A_t)` 였다. 두 식은 `c*²/(R·T) = 1/Γ²` 배
    #    차이가 나므로, 보고된 V_free·A_t 로 항등식을 검산하면 갈린다.
    v_free, at = g.data["v_free_mm3"], g.data["at_for_tau"]
    assert g.data["tau_s"] * motor.rt * at * 1e-6 == pytest.approx(
        motor.c_star_actual * v_free * 1e-9, rel=1e-9), "τ 식이 정방향과 다르다"
    wrong = v_free * 1e-9 / (motor.c_star_actual * at * 1e-6)   # 초판 식
    assert g.data["tau_s"] / wrong > 2.0, "옛 식과 구별되지 않는다"


def test_fixed_throat_out_of_reach_suggests_a_throat_diameter(motor, limits, ref):
    """고정 목이 도달 범위를 벗어나면 **필요한 목 직경을 역제시**한다."""
    # 추력을 2.5 배로 올리면 Ø10 목(도달 370 N)으로는 닿지 않지만, 더 큰 목이
    # 있다. 4 배(805 N)는 **어떤 목으로도** 안 된다 — 아래에서 따로 본다.
    tgt = Target(t=torch.tensor(ref["t"]), f=torch.tensor(ref["f"] * 2.5))
    g = _get(check_physics(tgt, motor, limits, (95.0,), a_throat=80.0),
             "PHYS_THROAT_FIXED")
    assert not g.ok
    assert "목 직경을" in g.action and "mm" in g.action
    f_lo, f_hi = g.data["f_reach"]
    assert f_lo < f_hi and not (f_lo <= 2.5 * ref["f"].max() <= f_hi)


def test_throat_reach_is_capped_by_the_case_pressure(motor, limits, ref):
    """도달 범위의 위 끝이 **케이스 허용압에서 잘려야** 한다.

    ⚠ 초판은 이 천장이 없어서 `p_eq` 의 이분 구간 상한(표 상한 × 3 = 32 MPa)
       까지 올라갔다. 작은 목에서 수백 MPa 급 평형압을 '도달 가능' 으로 세는
       셈이었고, 추력 4 배 목표에 대해 결론('어떤 목으로도 불가')은 맞았지만
       이유가 틀렸다. 이 시험이 천장을 못질한다.
    """
    tgt = Target(t=torch.tensor(ref["t"]), f=torch.tensor(ref["f"] * 4.0))
    g = _get(check_physics(tgt, motor, limits, (95.0,), a_throat=80.0),
             "PHYS_THROAT_FIXED")
    assert not g.ok
    assert "어떤 목 직경으로도" in g.action
    # 도달 압력은 어느 쪽도 케이스 허용압을 넘지 않는다
    p_lo, p_hi = g.data["p_reach"]
    assert p_hi <= limits.p_case_allow + 1e-9
    assert p_lo <= p_hi


def test_throat_reach_brackets_the_true_design(motor, limits, ref):
    """도달 범위가 **실제로 그 목으로 만든 설계**를 포함해야 한다.

    r₀=14 원형 포트를 A_t=80 으로 돌린 결과가 목표다. 그 목의 도달 범위가
    이 추력을 품지 못하면 범위 계산이 틀린 것이다.
    """
    tgt = Target(t=torch.tensor(ref["t"]), f=torch.tensor(ref["f"]))
    g = _get(check_physics(tgt, motor, limits, (95.0,), a_throat=80.0),
             "PHYS_THROAT_FIXED")
    f_lo, f_hi = g.data["f_reach"]
    assert f_lo <= ref["f"].max() <= f_hi
    assert g.data["d_t"] == pytest.approx(2 * np.sqrt(80.0 / np.pi), rel=1e-9)


# ═══════════════════════════════════════════════════════════════════ 4. 묶음
def test_check_inputs_stops_at_a_format_fatal(motor, limits):
    """형식에서 확정 불가가 나오면 물리 검사를 돌리지 않는다."""
    t = np.array([0.0, 0.2, 0.1, 0.3, 0.4, 0.5])
    rep = check_inputs(Target(t=torch.tensor(t), f=torch.tensor(np.full(6, 10.0))),
                       motor, limits)
    assert not rep.runnable
    assert not any(f.code.startswith("PHYS") for f in rep.findings)
    assert rep.get("FMT_TIME") is not None


def test_check_inputs_without_a_motor_runs_format_and_consistency_only(ref):
    rep = check_inputs(Target(t=torch.tensor(ref["t"]), f=torch.tensor(ref["f"])))
    codes = [f.code for f in rep.findings]
    assert any(c.startswith("FMT") for c in codes)
    assert any(c.startswith("CON") for c in codes)
    assert not any(c.startswith("PHYS") or c == "BOUNDS" for c in codes)


def test_report_text_states_the_verdict(motor, limits, ref):
    rep = check_inputs(Target(t=torch.tensor(ref["t"]), f=torch.tensor(ref["f"])),
                       motor, limits, (95.0,), a_throat=80.0)
    assert rep.runnable
    txt = rep.text()
    assert "입력 검증" in txt and "확정 불가 0" in txt
    # 실패하는 목표는 그 사실을 맨 위에 적는다
    bad = check_inputs(Target(t=torch.tensor(ref["t"] * 5.0),
                              f=torch.tensor(ref["f"])), motor, limits, (95.0,),
                       a_throat=80.0)
    assert not bad.runnable
    assert "역설계를 시작하지 않습니다" in bad.text()


def test_thrust_pressure_throat_must_be_mutually_consistent(motor, limits, ref):
    """추력·압력·고정 노즐목 **세 입력의 항등식** — 신설.

    `F = C_f(P, ε)·cf_eff·P·A_t` 이므로 두 곡선에서 **함의되는 A_t** 가 나온다.
    사용자가 고정 목을 입력했다면 그 값과 맞아야 하고, 안 맞으면 세 입력이 서로
    모순이라 **어떤 형상으로도 불가능**하다.

    ⚠ 이 검사만 '확정 불가' 를 다는 이유: 다른 물리 검사들은 경계가 엄밀하지
       않아(상한이 느슨하거나 도달 범위가 하한) 경고에 머물지만, 이것은
       **항등식**이다. 도구는 입력(노즐목·케이스)을 바꾸지 않으므로 판정은
       "이 곡선으로는 주어진 조건에서 맞는 형상을 찾을 수 없습니다" 다.

    ⚠⚠ **이 검사가 없었다.** `feasibility_bounds` 는 함의 A_t 를 물리적으로
        가능한 범위와만 비교하고(단위 오류용) `a_throat` 를 받지도 않으며,
        `PHYS_THROAT_FIXED` 는 압력 곡선을 쓰지 않는다. 그래서 세 입력이 서로
        모순인 경우가 **최적화를 다 돌린 뒤에** 발견됐다.
    """
    tgt = Target(t=torch.tensor(ref["t"]), f=torch.tensor(ref["f"]),
                 p=torch.tensor(ref["p"]))

    # 참 목 80 mm² — 함의값이 그것을 거의 정확히 복원해야 한다
    g = _get(check_physics(tgt, motor, limits, (95.0,), a_throat=80.0),
             "PHYS_FPAT")
    assert g.ok and g.level == FATAL
    assert g.data["at_implied"] == pytest.approx(80.0, rel=0.01), \
        "함의 노즐목이 참값을 복원하지 못한다 — cf_eff 를 안 나눴을 가능성"
    assert g.data["rel"] < ic.AT_IMPLIED_TOL

    # 틀린 목은 확정 불가로 막는다 — 10 % 차이도 잡아야 한다
    for at, want_rel in ((88.0, 0.09), (40.0, 1.0), (160.0, 0.5)):
        fs = check_physics(tgt, motor, limits, (95.0,), a_throat=at)
        g = _get(fs, "PHYS_FPAT")
        assert not g.ok and g.level == FATAL, f"A_t={at} 를 통과시켰다"
        assert g.data["rel"] == pytest.approx(want_rel, rel=0.2)
        assert "mm" in g.action        # 필요한 목 직경을 역제시한다

    # 압력 곡선이 없으면 이 검사는 아예 돌지 않는다 (빈 주장을 하지 않는다)
    no_p = Target(t=torch.tensor(ref["t"]), f=torch.tensor(ref["f"]))
    assert all(x.code != "PHYS_FPAT"
               for x in check_physics(no_p, motor, limits, (95.0,),
                                      a_throat=80.0))


def test_implied_throat_has_no_systematic_bias(motor):
    """함의 노즐목에 **계통 치우침이 없어야** 한다 — `cf_eff` 회귀 시험.

    ⚠⚠ 초판 추정자는 `C_f` 만으로 나누고 모델의 `cf_eff = 0.95` 를 빼먹어서,
        자기 일관된 설계 7 종에서 치우침이 **전부 +4.8~5.0 %** 로 일정했다.
        잡음이 아니라 계통오차였고, 그걸 문턱 15 % 로 덮고 있었다 — 검사가
        20 배 둔해진 셈이다. 고친 뒤 치우침이 **0.23 %** 로 떨어져 문턱을
        5 % 로 조일 수 있었다.

    이 시험은 **여러 설계에서** 치우침을 본다. 한 설계만 보면 계통오차와
    잔차를 구별할 수 없다 — 그게 처음에 놓친 이유다.
    """
    from gre.optimize.forward import DiffMotor
    from gre.optimize.loss import fp_ratio_spread

    m = DiffMotor(d_outer=60.0, n_grid=121, n_samples=384, n_w=200,
                  h_max=10e-3)
    cases = [(12.0, [0] * 6, 85.0), (12.0, [0] * 6, 54.0),
             (12.0, [0, 0, 2.2, 0, 0, 0], 78.0),
             (14.0, [0, 0, 0, 0, 0, 4.5], 90.0)]
    rels = []
    for r0, arr, at in cases:
        a = torch.tensor(arr, dtype=torch.float64)
        with torch.no_grad():
            t, p, f, _ = m.simulate(torch.tensor(r0), a, torch.zeros(6),
                                    [95.0], torch.tensor(at))
        fp = fp_ratio_spread(t, f, p, gamma=m.gamma, p_ambient=m.p_ambient,
                             cf_eff=m.cf_eff,
                             # ⚠ **정방향과 같은 C_f 를 넘겨야 한다**.
                             #   박리 인지 C_f 가 들어오면서 기본값이
                             #   `separation_frac = 0.4` 가 됐다. 이걸 안 넘기면
                             #   정방향은 박리를 보는데 역산은 안 봐서 치우침이
                             #   0.02 % → **9.77 %** 로 뛴다 — 이 시험이 실제로
                             #   그렇게 깨졌고, 그게 이 시험이 하는 일이다.
                             separation_frac=m.separation_frac)
        rels.append((fp["at_mean"] - at) / at)
    assert max(abs(r) for r in rels) < 0.01, \
        f"치우침 {[f'{100*r:+.2f}%' for r in rels]} — cf_eff 보정이 빠졌나"
    # 부호가 전부 같고 크기가 비슷하면 **계통오차**다 — 그것을 금지한다
    mean_bias = sum(rels) / len(rels)
    assert abs(mean_bias) < 0.005, f"평균 치우침 {100*mean_bias:+.2f} % — 계통오차다"
