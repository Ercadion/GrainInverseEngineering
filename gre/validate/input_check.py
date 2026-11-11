"""입력 자동 검증 루프 — 역설계를 돌리기 **전에** 목표가 성립하는지 본다 (§4.6.3).

`runner.feasibility_bounds` 와 역할이 겹치지 않는다. 나눠 놓은 이유가 있다.

    feasibility_bounds   **모터를 만들어야** 하는 검사 — 총역적 한계, 케이스
                         허용압, F/P 상호 모순, 함의 노즐목 단위. `DiffMotor` 와
                         `ManufacturingLimits` 를 둘 다 받고 챔버 상수를 쓴다.
    input_check (이 파일) **곡선만 보고** 하는 검사 + 곡선을 역으로 풀어 보는
                         검사. 모터 없이 돌아가는 부분(형식·일관성)이 따로 있어야
                         마법사(§4.8)가 입력받는 즉시 되돌려 줄 수 있다.

둘을 한 보고서로 합치는 것은 `check_inputs(..., with_bounds=True)` 가 한다.

═══════════════════════════════════════════════════════════════════════════
⚠ **판정의 강도를 섞지 않는다.** 검사마다 세 등급 중 하나를 달고, 등급의
   의미가 고정되어 있다.

    "확정 불가"  넘으면 **어떤 형상으로도 불가능**하다. 최적화를 시작하지
                 않는다. 이 등급을 달려면 상한이 **엄밀**해야 한다 — 즉
                 느슨한 쪽으로만 틀려야 한다.
    "경고"       불가능할 가능성이 높지만 증명되지 않았다. 돌려도 된다.
    "정보"       판정이 아니다. 숫자를 보여 주는 것이다.

   §4.6.3 표는 "연소면적 실현 가능성" 을 다른 하드 검사들과 같은 줄에 적어
   두었는데, **그 검사는 구조적으로 '경고' 밖에 될 수 없다** (아래
   `AB_REACH` 주석). 설계 문서 표를 고쳐야 한다.
═══════════════════════════════════════════════════════════════════════════

단위 규약 — 이 파일 안에서 끝까지 같다.

    t [s]   F [N]   P [MPa 절대]   A_t·A_b [mm²]   r [mm/s]   길이 [mm]

편한 항등식 두 개가 이 단위에서 나온다 (환산 상수가 1 이다).

    F [N]    = C_f · P [MPa] · A_t [mm²]
    A_b [mm²] = P·A_t·1e9 / (c*_실효 [m/s] · ρ_p [kg/m³] · r [mm/s])
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..core.burnrate import DESIGN_BANDS, NAKKA_TABLE
from ..core.nozzle import optimum_area_ratio, thrust_coefficient
from ..optimize.constraints import D_THROAT_MIN_MM

# ── 등급 ─────────────────────────────────────────────────────────────────
FATAL, WARN, INFO = "확정 불가", "경고", "정보"

#: 손으로 그린 곡선의 잡음 문턱 [F_max 대비]. 이 위면 평활화를 **제안**한다
#: (강제하지 않는다 — 실측 곡선은 원래 잡음이 있고, 평활화는 정보를 지운다).
#: 1 % 는 손 그림에서 나오는 들쭉날쭉함과 괜찮은 센서 잡음을 가르는 선이다.
#: 노즐목 면적의 물리적 하한 [mm²] — 직경 하한에서 유도한다. 매직 넘버
#: 5.0 이 여러 자리에 흩어져 있던 것을 한 곳으로 모았다.
A_THROAT_MIN: float = float(np.pi * (D_THROAT_MIN_MM / 2.0) ** 2)

NOISE_FRAC_WARN: float = 0.01

#: 샘플 간격 불균일 문턱 [max dt / min dt]. 2 배를 넘으면 리샘플을 제안한다 —
#: 손실의 `trapezoid` 는 불균일 격자에서도 맞지만, 창 판정(`shape_window`)과
#: 상승률 검사가 조밀한 구간을 과대 대표한다.
DT_RATIO_WARN: float = 2.0

#: 상승 구간을 "분해했다" 고 보는 최소 표본 수. 이보다 적으면 상승률 검사
#: 자체가 못 미더워진다 (차분이 한두 점에서 나온다).
RISE_SAMPLES_MIN: int = 5

#: 함의 노즐목과 입력 노즐목의 허용 상대차.
#:
#: 함의값은 `A_t = (F/P)/(C_f(P, ε)·cf_eff)` 이고 ε 는 창 안 평균 압력의 최적
#: 팽창비로 **자동 산출**되므로 참 ε 와 다른 데서 오는 잔차가 남는다. 실측
#: (자기 일관된 설계 7 종, 원형 3 · 스타 3 · 복합 1, A_t 54~126 mm²):
#:
#:     치우침  **0.00 ~ 0.23 %**   ← 문턱은 그 20 배
#:
#: ⚠ 처음에는 0.15 (15 %) 로 잡았다. 그때 자기 일관된 쌍의 치우침이 **5.0 %** 로
#:   나왔기 때문인데, 재 보니 7 종 전부 4.8~5.0 % 로 **일정**했다 — 잡음이 아니라
#:   계통오차였고 원인은 추정자가 `cf_eff = 0.95` 를 안 나눈 것이었다. 고치자
#:   치우침이 0.23 % 로 떨어져서 문턱을 **5 %** 로 조일 수 있었다.
#:   교훈: 계통오차를 문턱으로 덮으면 검사가 20 배 둔해진다.
AT_IMPLIED_TOL: float = 0.05

#: 대표 `C_f` 의 낙관적 상한 — `runner.CF_PLAUSIBLE[1]` 과 같은 값. 필요 장약을
#: **가장 적게** 잡는 데 쓴다 (관대한 쪽). 여기서 다시 적는 이유는 이 모듈이
#: 모터 없이도 돌아야 해서 `runner` 를 모듈 수준에서 들이지 않기 때문이다.
CF_MAX: float = 2.0


@dataclass
class Finding:
    """검사 하나의 결과. `ok=True` 면 통과다 (등급은 `INFO`)."""

    code: str
    level: str
    ok: bool
    title: str
    detail: str = ""
    action: str = ""
    data: dict = field(default_factory=dict)

    def __str__(self) -> str:
        mark = "✔" if self.ok else {FATAL: "✘", WARN: "⚠"}.get(self.level, "·")
        s = f"{mark} [{self.code}] {self.title}"
        if self.detail:
            s += f"\n    {self.detail}"
        if self.action and not self.ok:
            s += f"\n    조치: {self.action}"
        return s


@dataclass
class Report:
    findings: list

    @property
    def fatal(self) -> list:
        return [f for f in self.findings if not f.ok and f.level == FATAL]

    @property
    def warnings(self) -> list:
        return [f for f in self.findings if not f.ok and f.level == WARN]

    @property
    def runnable(self) -> bool:
        """역설계를 시작해도 되는가 — 확정 불가가 하나도 없으면 된다."""
        return not self.fatal

    def get(self, code: str):
        return next((f for f in self.findings if f.code == code), None)

    def text(self) -> str:
        head = (f"입력 검증: 확정 불가 {len(self.fatal)} · 경고 "
                f"{len(self.warnings)} · 검사 {len(self.findings)}")
        if not self.runnable:
            head += "\n\n**이 목표로는 역설계를 시작하지 않습니다.**"
        return head + "\n\n" + "\n".join(str(f) for f in self.findings)


# ════════════════════════════════════════════════════════ 1. 형식 (모터 불필요)
def check_format(t, f, p=None) -> list:
    """곡선이 곡선인가. **물리가 전혀 안 들어간다** — 배열만 본다.

    여기서 걸리는 것은 거의 전부 입력 사고다. 물리 검사를 그 위에 돌리면
    엉뚱한 진단이 나오므로 (예: 시간이 역행하면 총역적이 음수가 된다) 이
    검사가 먼저다.
    """
    t = np.asarray(t, dtype=float).ravel()
    f = np.asarray(f, dtype=float).ravel()
    out = []

    # ── 길이·유한성 — 이게 깨지면 아래 검사가 전부 무의미하다
    if t.shape != f.shape:
        return [Finding("FMT_LEN", FATAL, False, "시간축과 추력의 길이가 다릅니다",
                        f"t {t.size} 개, F {f.size} 개.",
                        "같은 시간축으로 리샘플해서 넣으십시오.")]
    if t.size < 4:
        return [Finding("FMT_LEN", FATAL, False, "표본이 너무 적습니다",
                        f"{t.size} 개. 모양을 논할 수 없습니다.",
                        "최소 수십 점, 상승 구간에 5 점 이상 되도록 넣으십시오.")]
    bad = ~(np.isfinite(t) & np.isfinite(f))
    out.append(Finding(
        "FMT_FINITE", FATAL, not bad.any(),
        "NaN·Inf 가 있습니다" if bad.any() else "모든 표본이 유한합니다",
        f"{int(bad.sum())} 개 ({np.flatnonzero(bad)[:5].tolist()} …)"
        if bad.any() else "",
        "결측 구간을 빼거나 보간해서 넣으십시오.", {"n_bad": int(bad.sum())}))
    if bad.any():
        return out

    # ── 시간 역행·중복 (§4.6.3 '시간 역행')
    dt = np.diff(t)
    n_back, n_dup = int((dt < 0).sum()), int((dt == 0).sum())
    out.append(Finding(
        "FMT_TIME", FATAL, n_back == 0 and n_dup == 0,
        "시간축이 단조증가가 아닙니다" if n_back or n_dup
        else "시간축이 단조증가입니다",
        f"역행 {n_back} 곳, 중복 {n_dup} 곳." if n_back or n_dup else
        f"{t[0]:.4g} ~ {t[-1]:.4g} s, {t.size} 점.",
        "시간으로 정렬하고 중복 시각을 합치십시오. 두 시험을 이어 붙인 "
        "파일이면 하나만 넣으십시오.", {"n_backward": n_back, "n_dup": n_dup}))
    if n_back or n_dup:
        return out

    # ── 음수 추력 (§4.6.3 '음수 추력')
    #    크기로 갈라야 한다. 로드셀 영점 치우침은 작은 음수를 만들고 그건
    #    고칠 수 있는 입력 사고지만, 큰 음수는 곡선 자체가 틀린 것이다.
    f_max = float(f.max())
    neg = f < 0
    worst = float(f.min())
    small = neg.any() and abs(worst) <= 0.02 * max(f_max, 1e-12)
    out.append(Finding(
        "FMT_SIGN", WARN if small else FATAL, not neg.any(),
        "음수 추력이 있습니다" if neg.any() else "추력이 모두 0 이상입니다",
        (f"{int(neg.sum())} 점, 최저 {worst:.3g} N "
         f"(F_max 의 {100*abs(worst)/max(f_max,1e-12):.2f} %). "
         + ("영점 치우침으로 보입니다." if small else
            "영점 치우침으로 설명되는 크기가 아닙니다."))
        if neg.any() else "",
        ("영점(태어 값)을 빼서 넣으십시오." if small else
         "추력 채널의 부호·영점·단위를 확인하십시오. 압력을 추력 칸에 넣은 "
         "것이 아닌지도 보십시오."),
        {"n_neg": int(neg.sum()), "f_min": worst}))

    # ── 샘플 간격 (§4.6.3 '샘플 간격 이상')
    ratio = float(dt.max() / max(dt.min(), 1e-15))
    out.append(Finding(
        "FMT_DT", WARN, ratio <= DT_RATIO_WARN,
        "샘플 간격이 고르지 않습니다" if ratio > DT_RATIO_WARN
        else "샘플 간격이 고릅니다",
        f"dt {dt.min():.4g} ~ {dt.max():.4g} s (비 {ratio:.1f}×, "
        f"중앙 {np.median(dt):.4g} s).",
        "균일 격자로 리샘플하십시오. 창 판정과 상승률 검사가 조밀한 구간을 "
        "과대 대표합니다.", {"dt_ratio": ratio, "dt_median": float(np.median(dt))}))

    # ── 잡음 (§4.6.3 '손으로 그린 곡선의 잡음 — 평활화 제안')
    #    2차 차분 기반 강건 추정. 백색잡음이면 Var(Δ²f) = 6σ² 이므로
    #    σ̂ = 1.4826·median|Δ²f| / √6 (1.4826 은 MAD→σ 환산). 곡선의 매끄러운
    #    성분은 Δ² 에서 h² 로 줄어들므로 잡음만 남는다.
    #    ⚠ 불균일 격자에서는 과대 추정된다 — 그래서 FMT_DT 와 같이 읽어야 한다.
    d2 = np.diff(f, 2)
    sigma = 1.4826 * float(np.median(np.abs(d2))) / np.sqrt(6.0)
    frac = sigma / max(f_max, 1e-12)
    # 평활화 창 제안 — 잡음을 문턱까지 내리는 이동평균 길이. σ_ma = σ/√m.
    m_sugg = int(np.ceil((frac / NOISE_FRAC_WARN) ** 2)) if frac > 0 else 1
    out.append(Finding(
        "FMT_NOISE", WARN, frac <= NOISE_FRAC_WARN,
        "곡선에 잡음이 많습니다" if frac > NOISE_FRAC_WARN else "잡음이 적습니다",
        f"추정 σ = {sigma:.3g} N = F_max 의 {100*frac:.2f} % "
        f"(문턱 {100*NOISE_FRAC_WARN:.0f} %).",
        f"길이 {m_sugg} 점 이동평균(또는 같은 정도의 저역통과)으로 평활화한 뒤 "
        f"넣으십시오. 잡음이 모양 오차로 들어가면 허용 오차 10 % 를 잡음이 "
        f"먹습니다.", {"sigma_n": sigma, "sigma_frac": frac, "ma_suggest": m_sugg}))

    # ── 기록이 끊겼는가 — 테일오프가 안 들어온 곡선은 총역적이 과소평가된다
    tail = float(f[-1]) / max(f_max, 1e-12)
    out.append(Finding(
        "FMT_TAIL", WARN, tail <= 0.10,
        "곡선이 0 으로 돌아오지 않고 끝납니다" if tail > 0.10
        else "곡선이 0 근처에서 끝납니다",
        f"마지막 값 {f[-1]:.3g} N = F_max 의 {100*tail:.0f} %.",
        "기록이 중간에서 끊긴 것이면 뒤를 포함해서 넣으십시오. 총역적이 "
        "과소평가되고, 모양 창이 테일오프를 '사양' 으로 읽습니다.",
        {"tail_frac": tail}))

    # ── 상승 구간이 분해되었는가 (상승률 검사의 전제)
    i_hi = int(np.flatnonzero(f >= 0.9 * f_max)[0])
    n_rise = int((f[:i_hi + 1] > 0.05 * f_max).sum())
    out.append(Finding(
        "FMT_RISE_RES", WARN, n_rise >= RISE_SAMPLES_MIN,
        "상승 구간의 표본이 너무 적습니다" if n_rise < RISE_SAMPLES_MIN
        else "상승 구간이 분해되어 있습니다",
        f"0.05·F_max ~ 0.9·F_max 사이 {n_rise} 점 "
        f"(최소 {RISE_SAMPLES_MIN}), 상승 시간 {t[i_hi]-t[0]:.4g} s.",
        "상승 구간을 더 촘촘히 주십시오. 지금 표본으로는 상승률 검사"
        "(PHYS_RISE)가 실제 기울기보다 완만하게 봅니다.",
        {"n_rise": n_rise, "t_rise": float(t[i_hi] - t[0])}))

    if p is not None:
        pp = np.asarray(p, dtype=float).ravel()
        if pp.shape != t.shape:
            out.append(Finding("FMT_P_LEN", FATAL, False,
                               "압력 곡선의 길이가 시간축과 다릅니다",
                               f"p {pp.size} 개 vs t {t.size} 개.",
                               "같은 시간축으로 리샘플해서 넣으십시오."))
        elif not np.isfinite(pp).all() or pp.min() <= 0:
            out.append(Finding("FMT_P_SIGN", FATAL, False,
                               "압력 곡선에 0 이하이거나 유한하지 않은 값이 있습니다",
                               f"최저 {float(np.nanmin(pp)):.4g} MPa.",
                               "**절대압 [MPa]** 로 넣으십시오 (게이지압이면 "
                               "대기압 0.1013 을 더해야 합니다)."))
        else:
            out.append(Finding("FMT_P", INFO, True, "압력 곡선 형식 정상",
                               f"{pp.min():.3g} ~ {pp.max():.3g} MPa 절대."))
    return out


# ═════════════════════════════════════════════════ 2. 입력 일관성 (모터 불필요)
def check_consistency(target) -> list:
    """스칼라 목표들이 곡선과, 그리고 서로 모순이 아닌가 (§4.6.3 '입력 일관성').

    `Target` 은 스칼라를 안 주면 곡선에서 채운다. 그러니 **사용자가 직접 준
    스칼라만** 의미가 있다 — 자동으로 채워진 값과 곡선을 비교하는 것은
    항등식을 확인하는 셈이다. 그래서 여기서는 둘이 어긋나는지가 아니라
    **어긋난다는 사실이 무엇을 뜻하는지**를 적는다.
    """
    t = np.asarray(target.t, dtype=float).ravel()
    f = np.asarray(target.f, dtype=float).ravel()
    out = []

    i_curve = float(np.trapezoid(f, t))
    f_max_curve = float(f.max())
    t_burn = float(t[-1] - t[0])
    f_avg = i_curve / max(t_burn, 1e-12)
    shape = f_max_curve / max(f_avg, 1e-12)

    # F_max/F_avg ≥ 1 은 항등식이다 (평균이 최대를 넘을 수 없다). 곡선에서
    # 계산하면 반드시 성립하므로, 깨진다면 **스칼라를 직접 준 것**이다.
    out.append(Finding(
        "CON_SHAPE", INFO, True, "곡선의 형상 계수",
        f"F_max/F_avg = {shape:.2f} "
        f"(F_max {f_max_curve:.3g} N, F_avg {f_avg:.3g} N, 연소 {t_burn:.4g} s). "
        + ("1 에 가깝습니다 — 중립 곡선입니다. ⚠ 중립 곡선은 형상 정보를 "
           "거의 담지 않습니다 (§4.6.3 식별성)." if shape < 1.15 else
           "" if shape < 3.0 else
           "꽤 뾰족합니다 — 점화 첨두가 섞여 있는지 확인하십시오."),
        "", {"shape_factor": shape, "f_avg": f_avg, "t_burn": t_burn}))

    for name, given, from_curve, unit, why in (
        ("total_impulse", target.total_impulse, i_curve, "N·s",
         "총역적을 직접 주면 그 값이 사양이 되고 곡선은 모양만 쓰입니다. "
         "둘이 다르면 **어느 쪽도 만족시킬 수 없는 목표**가 됩니다"),
        ("f_max", target.f_max, f_max_curve, "N",
         "최대추력을 직접 주면 그 값이 사양이 됩니다. 곡선의 최대와 다르면 "
         "모양 사양과 최대추력 사양이 서로 싸웁니다"),
    ):
        if given is None:
            continue
        rel = abs(float(given) - from_curve) / max(abs(from_curve), 1e-12)
        out.append(Finding(
            f"CON_{name.upper()}", WARN if rel > 0.05 else INFO, rel <= 0.05,
            f"입력한 {name} 이 곡선과 다릅니다" if rel > 0.05
            else f"입력한 {name} 이 곡선과 맞습니다",
            f"입력 {float(given):.4g} {unit} vs 곡선 {from_curve:.4g} {unit} "
            f"({100*rel:+.1f} %). " + (why if rel > 0.05 else ""),
            f"{name} 을 비워 두면 곡선에서 가져갑니다. 곡선 쪽이 틀렸다면 "
            f"곡선을 고치십시오.", {"given": float(given), "curve": from_curve,
                                 "rel": rel}))

    if getattr(target, "p", None) is not None and target.p_max is not None:
        pmax_curve = float(np.asarray(target.p).max())
        rel = abs(target.p_max - pmax_curve) / max(pmax_curve, 1e-12)
        out.append(Finding(
            "CON_PMAX", WARN if rel > 0.05 else INFO, rel <= 0.05,
            "입력한 p_max 가 압력 곡선과 다릅니다" if rel > 0.05
            else "입력한 p_max 가 압력 곡선과 맞습니다",
            f"입력 {target.p_max:.4g} vs 곡선 {pmax_curve:.4g} MPa "
            f"({100*rel:+.1f} %).",
            "p_max 를 비워 두면 곡선에서 가져갑니다.", {"rel": rel}))
    return out


# ═══════════════════════════════════════════════════════════ 3. 물리 (모터 필요)
#: 작동 구간 판정의 문턱 — `loss.shape_window` · `fp_ratio_spread` 와 **같은 값**.
IGNITION_FRAC: float = 0.90


def operating_window(f, ignition_frac: float = IGNITION_FRAC) -> np.ndarray:
    """작동 구간 마스크 — **점화 끝부터 테일오프 시작까지의 구간**.

    `loss.shape_window` 와 같은 의미다: 점화가 끝나는 시각과 연소가 끝나는
    시각 **사이**를 전부 고른다. 점화 상승과 테일오프는 모델 범위 밖이고
    (§4.3) 모양 사양도 그 구간을 세지 않는다.

    ⚠ **`loss.fp_ratio_spread` 의 창과 다르다. 그쪽이 틀렸다.** 그 함수는
       같은 문턱을 **준위 마스크**로 쓴다 (`p >= 0.9·plateau`). 중립·감소
       곡선에서는 구간과 마스크가 같지만, **부스트-서스테인 곡선에서는 다르다**
       — 평탄값이 부스트 쪽에서 잡히므로 마스크가 서스테인 구간을 통째로
       버린다. 실측: 부스트 200 N / 서스테인 60 N 곡선에서 마스크는 부스트
       구간만 남긴다. F/P 모순 검사가 **두 곡선이 어긋날 수 있는 바로 그
       구간을 안 본다**는 뜻이다. §4.6.2 가 부스트-서스테인을 입력 템플릿으로
       명시하고 있으므로 가정이 아니라 실제 입력이다. → 전면 검토 보고서의
       지적 사항이고 `fp_ratio_spread` 를 구간 의미로 고쳐야 한다.

    압력 대신 추력으로 판정하는 이유: 검증 루프에는 압력이 아직 없다 (곡선에서
    역산해야 하고, 그 역산이 `C_f` 를 고르려면 창이 먼저 필요하다). 창 안에서
    `C_f` 가 거의 일정하므로 `F ∝ P` 이고 두 규칙은 같은 구간을 고른다 —
    `test_operating_window_matches_the_pressure_based_rule` 가 그것을 못질한다.
    """
    f = np.asarray(f, dtype=float).ravel()
    f_max = float(f.max())
    hi = f[f > 0.5 * f_max]
    plateau = float(np.median(hi)) if hi.size else f_max
    lit = np.flatnonzero(f >= ignition_frac * plateau)        # 점화 완료
    alive = np.flatnonzero(f > 0.05 * f_max)                  # 연소 중
    win = np.zeros(f.size, dtype=bool)
    if lit.size == 0 or alive.size == 0:
        return win
    win[int(lit[0]):int(alive[-1]) + 1] = True
    return win


def _cf_rep(motor, p_c: float) -> float:
    """대표 `C_f` — 그 압력에서의 최적 팽창. 정방향 모델과 같은 규약(§4.4)."""
    pa = motor.p_ambient
    if p_c <= pa:
        return 1.0
    eps = optimum_area_ratio(p_c, pa, motor.gamma)
    return thrust_coefficient(p_c, pa, eps, motor.gamma) * motor.cf_eff


def _pressure_from_thrust(motor, f, a_throat: float, cf: float) -> np.ndarray:
    """F = C_f·P·A_t 를 P 로 푼다 [MPa]. 단위 항등식은 모듈 docstring."""
    return np.asarray(f, dtype=float) / max(cf * float(a_throat), 1e-12)


def _rate(motor, p_mpa) -> np.ndarray:
    """연소속도 r(P) [mm/s] — 정방향 모델과 **같은** 객체를 쓴다."""
    import torch
    p = np.clip(np.asarray(p_mpa, dtype=float), 1e-3, None)
    return motor.br.rate(torch.as_tensor(p, dtype=motor.dtype)).detach().numpy()


def _ab_required(motor, p_mpa, a_throat: float) -> np.ndarray:
    """정상상태에서 그 압력을 유지하는 데 필요한 연소면적 [mm²].

        ρ_p·A_b·r = P·A_t/c*_실효      (챔버 질량 균형, 준정상)

    ⚠ **준정상이다.** 점화 상승·테일오프에서는 챔버 충전 항이 빠져 있어
       필요 면적을 과대평가한다. 그래서 이 값으로 하는 판정은 작동 구간
       (평형압 90 % 이상)에서만 읽는다 — `shape_window` 와 같은 규칙이다.
    """
    p = np.clip(np.asarray(p_mpa, dtype=float), 1e-3, None)
    r = _rate(motor, p)
    return p * float(a_throat) * 1e9 / (motor.c_star_actual * motor.rho_p * r)


def _ab_upper_bound(motor, limits, lengths) -> dict:
    """연소면적의 **느슨한** 상한과, 우리가 실제로 뒤지는 족의 도달 범위.

    상한의 논리: 포트 경계에서 추진제 쪽으로 `t_prop_min` 만큼은 추진제여야
    한다 (제약의 정의). 그 띠가 서로 겹치지 않으면 띠의 총면적이
    `둘레 × t_prop_min` 이므로 `둘레 ≤ πR²/t_prop_min` 이다.

    ⚠ **이 상한은 엄밀하지 않다.** 띠가 겹치지 않는다는 가정이 `r_tip_min`
       (0.8 mm) < `t_prop_min` (2.0 mm) 인 쐐기 끝에서 깨지고, 거기서 둘레는
       상한보다 커질 수 있다. 그러므로 이 검사는 **경고까지만** 할 수 있고
       "확정 불가" 를 달 수 없다 — §4.6.3 표가 이 검사를 다른 하드
       검사와 같은 줄에 적어 둔 것은 과한 표현이다.

    도달 범위(`reach`)는 다르다. 단일 모드 포트 `r₀ + A·cos(kθ)` 를 훑어
    **우리가 실제로 뒤지는 족**이 내는 최대 연소면적을 해석 둘레로 잰다
    (거리장이 필요 없어 사실상 공짜다). 이쪽은 **하한**이므로 넘는다고
    불가능이 증명되지 않는다. 두 숫자를 같이 보여 주는 게 목적이다.
    """
    r_out = float(motor.ds.r_outer)
    l_tot = float(sum(lengths))
    end = 2.0 * np.pi * r_out ** 2 if motor.ends_burn else 0.0

    peri_bound = np.pi * r_out ** 2 / max(limits.t_prop_min, 1e-9)
    ab_bound = peri_bound * l_tot + end

    # ── 족의 도달 범위 — 해석 둘레 ∮√(r²+r'²)dθ, 제약 안에서 최대화
    th = np.linspace(0.0, 2 * np.pi, 2048, endpoint=False)
    best = {"ab": 0.0, "k": 0, "amp": 0.0, "r0": 0.0}
    for k in range(1, 9):
        for r0 in np.linspace(2.0, r_out - limits.w_web_min, 24):
            # 진폭 상한 세 개 — 웹, 추진제 두께, 꼭짓점 반경
            #   꼭짓점 반경: r = r₀ + A cos(kθ) 의 최소 곡률 반경은
            #   (r₀ − A)² / (r₀ − A + A k²) ≥ r_tip_min 에서 나온다
            a_hi = min(r0 - limits.t_prop_min, r_out - r0 - limits.w_web_min)
            if a_hi <= 0:
                continue
            amps = np.linspace(0.0, a_hi, 40)[1:]
            rr = r0 - amps
            rho_tip = rr ** 2 / np.maximum(rr + amps * k ** 2, 1e-12)
            amps = amps[rho_tip >= limits.r_tip_min]
            if amps.size == 0:
                continue
            amp = float(amps.max())
            r = r0 + amp * np.cos(k * th)
            rp = -amp * k * np.sin(k * th)
            peri = float(np.trapezoid(np.sqrt(r ** 2 + rp ** 2), th))
            # 끝면은 케이스 단면에서 포트 면적을 뺀 것 (양 끝)
            port_area = float(np.trapezoid(0.5 * r ** 2, th))
            ab = peri * l_tot + (2.0 * (np.pi * r_out ** 2 - port_area)
                                 if motor.ends_burn else 0.0)
            if ab > best["ab"]:
                best = {"ab": ab, "k": k, "amp": amp, "r0": float(r0),
                        "perimeter": peri}
    return {"ab_bound": ab_bound, "perimeter_bound": peri_bound, "reach": best}


def check_physics(target, motor, limits, lengths=(95.0,),
                  a_throat: float = None) -> list:
    """곡선을 역으로 풀어 보는 검사 — 요구 `A_b(t)`, 웹 소모, 상승률, 노즐목.

    `a_throat` 를 주면 **고정 노즐목 모드** (§4.6.1 3모드 중 세 번째). 그러면
    압력이 곡선에서 바로 정해지므로 검사가 훨씬 날카로워지고, 벗어날 때
    **필요한 목 직경 범위를 역으로 제시**할 수 있다 (§4.6.3).
    """
    t = np.asarray(target.t, dtype=float).ravel()
    f = np.asarray(target.f, dtype=float).ravel()
    out = []
    r_out = float(motor.ds.r_outer)
    f_max = float(f.max())

    # ── 작동 구간 — 준정상 가정이 성립하고, 모양 사양이 실제로 보는 구간
    win = operating_window(f)
    bands = DESIGN_BANDS.get(motor.propellant, {}).get("default")
    tab = NAKKA_TABLE[motor.propellant]
    p_tab = (float(tab[0][0]), float(tab[-1][1]))

    # ── 상승률 한계 (§4.6.3 '상승률 한계 — 챔버 충전 시상수 τ')
    #    챔버 충전: τ = V_자유 / (c*·A_t). 압력은 (P_eq − P)/τ 보다 빨리 못
    #    오른다. 추력은 P 에 비례하므로 |dF/dt| ≤ F_max/τ.
    #    τ 를 **가장 작게** (= 가장 관대하게) 잡아야 "넘으면 확정" 이 된다.
    #    τ ∝ V_free 이고 V_free = (chamber_margin − 1)·V_추진제 이므로, 필요한
    #    것은 **이 곡선을 낼 수 있는 가장 적은 추진제 체적**이다.
    #
    #        I ≤ c*_실효·C_f·m_추진제  →  m ≥ I/(c*·C_f,max)  →  V = m/ρ_p
    #
    #    ⚠ 초판은 "케이스를 추진제로 가득 채운" 체적을 썼다. 그게 가장 관대한
    #      쪽이라고 적어 뒀는데 **반대였다** — 체적이 커지면 τ 가 커지고 한계가
    #      내려가 검사가 엄격해진다. 운용 사례에서 자유 체적을 67152 → 43553 mm³
    #      으로 내리므로 한계가 1.54 배 느슨해진다.
    #
    #    ⚠ **점화 상승과 테일오프는 이 검사에서 뺀다.** 둘 다 모델 범위
    #       밖이고(§4.3) 모양 창도 사양으로 세지 않는다. 넣으면 **사양이 아닌
    #       구간 때문에 멀쩡한 목표를 거부**하게 된다 — 초판이 그랬고, 고정 목
    #       Ø4 mm 에서 자기 일관된 목표가 '확정 불가' 로 나왔다. 두 번째 판은
    #       점화만 뺐는데 그러면 **수직 테일오프**가 같은 거짓 양성을 낸다.
    #       구간 선택은 `operating_window` 하나로 통일한다.
    #    τ 는 **정방향 모델과 같은 식**을 써야 한다 (`forward.py` 의 `tau_w`):
    #
    #        τ(w) = c*·V_free(w) / (R·T·A_t),    R·T = (c*·Γ)²
    #
    #    ⚠ 초판은 `τ = V_free/(c*·A_t)` 를 썼다 — 차원은 맞지만 식이 다르고,
    #      같은 V_free·A_t 에서 **2.24 배** 커진다. FATAL 등급 검사의 문턱이
    #      그만큼 느슨해져 "넘으면 확정" 의 확정이 약해진다. 전면 검토에서
    #      잡았다.
    at_for_tau = float(a_throat) if a_throat else float(
        np.pi * r_out ** 2 / 10.0)           # 목/케이스 10:1 — 넉넉한 쪽
    i_tgt = float(np.trapezoid(f, t))
    m_min_kg = i_tgt / max(motor.c_star_actual * CF_MAX, 1e-9)
    v_prop_min = min(m_min_kg / motor.rho_p * 1e9,          # [mm³]
                     np.pi * r_out ** 2 * sum(lengths))     # 케이스보다 클 수 없다
    v_free_min = (motor.chamber_margin - 1.0) * v_prop_min
    tau = (motor.c_star_actual * v_free_min * 1e-9
           / max(motor.rt * at_for_tau * 1e-6, 1e-15))
    dfdt_cap = f_max / max(tau, 1e-15)
    dfdt_all = np.abs(np.diff(f) / np.maximum(np.diff(t), 1e-15))
    # 창 안의 **구간**(양 끝 모두 창 안)만 본다 — 창 경계를 걸치는 차분은
    # 점화·테일오프의 기울기이지 작동 구간의 것이 아니다.
    seg = win[:-1] & win[1:]
    dfdt_op = float(dfdt_all[seg].max()) if seg.any() else 0.0
    dfdt_out = float(dfdt_all[~seg].max()) if (~seg).any() else 0.0
    dt_med = float(np.median(np.diff(t)))
    dt_over_tau = dt_med / max(float(tau), 1e-15)

    # 표본이 거칠면 차분이 기울기를 **과소평가**한다. 그래도 과소평가된 값이
    # 이미 한계를 넘으면 결론은 뒤집히지 않으므로 '확정 불가' 를 달 수 있다.
    # 넘지 않으면 아무것도 못 말한다 — 다만 그 구간(점화)은 애초에 검사에서
    # 뺐으므로 남은 작동 구간은 표본이 충분하다. 그래서 분해 여부로 등급을
    # 가르지 않는다.
    # ⚠ **표본 간격이 τ 보다 길면 이 검사는 판정을 할 수 없다.** 걸리는 조건이
    #   ΔF/F_max > dt/τ 이므로 dt/τ ≥ 1 이면 어떤 곡선도 걸리지 않는다 — 한
    #   표본 사이에 F_max 를 넘게 변할 수는 없기 때문이다. 그때 "통과" 라고
    #   적으면 검사한 적이 없는 것을 검사했다고 적는 셈이다. 등급을 '정보' 로
    #   내리고 판정 불가라고 말한다.
    can_judge = dt_over_tau < 1.0
    out.append(Finding(
        "PHYS_RISE", FATAL if can_judge else INFO,
        dfdt_op <= dfdt_cap,
        "챔버가 이 속도로 찰 수 없습니다 (작동 구간)" if dfdt_op > dfdt_cap
        else ("작동 구간 상승·하강률이 챔버 충전 한계 안입니다" if can_judge else
              "상승률을 판정할 수 없습니다 — 표본 간격이 τ 보다 깁니다"),
        f"작동 구간 최대 |dF/dt| = {dfdt_op:.3g} N/s, 한계 {dfdt_cap:.3g} N/s "
        f"(τ = {1e3*tau:.3g} ms, 자유 체적 {v_free_min:.0f} mm³, "
        f"A_t {at_for_tau:.0f} mm²). "
        f"창 밖(점화·테일오프) 최대 {dfdt_out:.3g} N/s 는 **검사에서 뺐다** — "
        f"둘 다 모델 범위 밖이고(§4.3) 모양 창도 사양으로 세지 않는다. "
        + ("⚠ 그러므로 수직 점화·테일오프는 이 검사가 거부하지 않는다. 대신 "
           "**그 구간은 재현되지 않는다**는 것을 알고 있어야 한다. "
           if dfdt_out > dfdt_cap else "")
        + (f"⚠ 걸리는 조건은 ΔF/F_max > dt/τ 이고 dt/τ = {dt_over_tau:.2f} 다 — "
           f"인접 표본 사이 낙차가 F_max 의 {100*dt_over_tau:.0f} % 를 넘어야 "
           f"하므로 **사실상 불연속만** 잡는다. "
           if can_judge else
           f"⚠ dt/τ = {dt_over_tau:.2f} ≥ 1 이므로 **이 검사는 어떤 곡선도 "
           f"거부할 수 없다** (한 표본 사이에 F_max 를 넘게 변할 수는 없다). "
           f"판정하려면 표본 간격을 {1e3*tau:.3g} ms 아래로 주어야 한다. ")
        + f"쓸모 있는 숫자는 오히려 이쪽이다: 챔버가 만들 수 있는 가장 짧은 "
          f"전이는 약 3τ = {3e3*tau:.2g} ms 이고, 그보다 짧게 그린 단차는 "
          f"거부되지 않고 **눕혀져서 재현된다.**",
        "그 구간을 τ 보다 완만하게 그리십시오 (부스트→서스테인 단차라면 "
        "단차를 챔버 비움 시간만큼 눕혀야 합니다).",
        {"dfdt_op": dfdt_op, "dfdt_out": dfdt_out, "dfdt_cap": dfdt_cap,
         "tau_s": float(tau), "shortest_transition_s": 3.0 * float(tau),
         "dt_median_s": dt_med, "dt_over_tau": dt_over_tau,
         "v_free_mm3": float(v_free_min), "at_for_tau": at_for_tau}))

    # ── 연소면적의 상한·도달 범위 — 아래 검사 여럿이 쓴다
    bnd = _ab_upper_bound(motor, limits, lengths)
    reach = bnd["reach"]

    def _solve(at: float) -> tuple:
        """그 목 면적에서의 (요구 압력, 요구 연소면적). C_f 를 한 번 갱신한다."""
        cf = _cf_rep(motor, 5.0)
        pr = _pressure_from_thrust(motor, f, at, cf)
        cf = _cf_rep(motor, max(float(np.median(pr[win])), 0.2))
        pr = _pressure_from_thrust(motor, f, at, cf)
        return pr, _ab_required(motor, pr, at), cf

    # ── 노즐목이 자유인가 고정인가에 따라 압력 검사가 갈린다
    if a_throat is None:
        # 노즐목이 자유면 압력은 곡선에서 정해지지 않는다.
        #
        # ⚠ 초판은 "가장 관대한 목" 으로 πR² 을 썼다. 그러면 압력이 0.1 MPa
        #   아래로 내려가 연소속도가 사실상 0 이 되고 **웹 검사가 공허해진다**
        #   (연소 5 배 목표가 웹 검사를 그냥 통과했다). 목은 실제로 두 쪽에서
        #   묶인다 — 압력이 표 아래로 내려갈 수 없고, 요구 연소면적이 족의
        #   도달 범위를 넘을 수 없다. 그 구간 안에서 **웹이 가장 적게 드는**
        #   목을 쓴다: 여전히 가장 관대하지만 공허하지 않다.
        cands = []
        for at in np.geomspace(A_THROAT_MIN, np.pi * r_out ** 2, 48):
            pr, ab, cf = _solve(float(at))
            if pr[win].min() < p_tab[0] or pr[win].max() > p_tab[1]:
                continue
            if float(ab[win].max()) > reach["ab"]:
                continue
            r_tmp = _rate(motor, pr)
            cands.append((float(np.trapezoid(r_tmp, t)), float(at), cf,
                          float(pr[win].min()), float(pr[win].max())))
        if cands:
            cands.sort()
            at_for_web, cf0 = cands[0][1], cands[0][2]
            out.append(Finding(
                # ⚠ 이름은 "자유" 지만 **최적화기의 자유 모드가 아니다** —
                #   그 모드는 삭제됐다 (`optimize` 는 `ThroatSpec` 을
                #   필수로 받는다). 여기서 뜻하는 것은 "호출자가 이 검사에
                #   노즐목을 안 넘겼으니, 입력 가능한 목의 범위를 알려 준다" 다.
                "PHYS_THROAT_RANGE", INFO, True,
                "노즐목 입력이 없습니다 — 가능한 범위를 알려 드립니다",
                f"압력이 표 안이고 요구 연소면적이 족의 도달 범위 안인 목은 "
                f"{min(c[1] for c in cands):.0f} ~ {max(c[1] for c in cands):.0f} mm² "
                f"(Ø{2*np.sqrt(min(c[1] for c in cands)/np.pi):.1f} ~ "
                f"Ø{2*np.sqrt(max(c[1] for c in cands)/np.pi):.1f} mm). 아래 검사는 "
                f"그중 **웹이 가장 적게 드는** 목 {at_for_web:.0f} mm² 를 씁니다 — "
                f"가장 관대한 쪽이므로 '넘으면 확정' 이 유지됩니다.",
                "", {"at_window": (min(c[1] for c in cands),
                                   max(c[1] for c in cands)),
                     "at_used": at_for_web, "n_candidates": len(cands)}))
        else:
            # 어떤 목도 압력·면적을 동시에 만족시키지 못한다 — 그 자체가 결과다
            at_for_web, cf0 = float(np.pi * r_out ** 2), _cf_rep(motor, 1.0)
            out.append(Finding(
                "PHYS_THROAT_RANGE", WARN, False,
                "어떤 노즐목으로도 압력과 연소면적을 동시에 만족시킬 수 없습니다",
                f"목을 {A_THROAT_MIN:.0f} ~ {np.pi*r_out**2:.0f} mm² 로 "
                f"훑었지만, 요구 압력이 "
                f"연소속도 표({p_tab[0]:.3g}~{p_tab[1]:.3g} MPa) 안에 들어오는 "
                f"동시에 요구 연소면적이 족의 도달 범위({reach['ab']:.0f} mm²) "
                f"안인 목이 없습니다. ⚠ 도달 범위가 하한이라 '확정' 은 아닙니다.",
                "추력 수준이나 연소 시간을 조정하거나 케이스를 키우십시오.",
                {"n_candidates": 0}))
        p_req, ab_req, _ = _solve(at_for_web)
    else:
        at_for_web = float(a_throat)
        p_req, ab_req, cf0 = _solve(at_for_web)

    # ── 압력이 연소속도 표 안인가 (§4.6.3 '압력 한계 … 설계 대역 안인가')
    p_win = p_req[win]
    in_tab = (p_win.min() >= p_tab[0]) and (p_win.max() <= p_tab[1])
    note = (f"작동 구간 요구 압력 {p_win.min():.3g}~{p_win.max():.3g} MPa, "
            f"표 범위 {p_tab[0]:.3g}~{p_tab[1]:.3g} MPa")
    if bands:
        frac_band = float(((p_win >= bands[0]) & (p_win <= bands[1])).mean())
        note += (f", 설계 대역 {bands[0]:.3g}~{bands[1]:.3g} MPa 안 "
                 f"{100*frac_band:.0f} %")
    out.append(Finding(
        "PHYS_PBAND", WARN if a_throat is not None else INFO, in_tab,
        "요구 압력이 연소속도 표 범위를 벗어납니다" if not in_tab
        else "요구 압력이 표 범위 안입니다",
        note + ("." if in_tab else ". 표 밖은 **외삽**이고 r(P) 가 보증되지 "
                "않습니다."),
        "노즐목 입력을 바꾸거나 목표 추력을 조정하십시오.",
        {"p_req_min": float(p_win.min()), "p_req_max": float(p_win.max()),
         "p_table": p_tab}))

    # ── 웹 한계 (§4.6.3 '∫r dt ≤ 확보 가능한 웹')
    #    확보 가능한 웹의 상한: 포트 반경 → 0 일 때의 R, 끝면 연소면 L/2 중
    #    먼저 오는 쪽. 가장 관대하게 잡으므로 **넘으면 확정**이다.
    web_used = float(np.trapezoid(_rate(motor, p_req), t))
    web_avail = min(r_out, min(lengths) / 2.0 if motor.ends_burn else r_out)
    out.append(Finding(
        "PHYS_WEB", FATAL, web_used <= web_avail,
        "이 곡선은 케이스에 있는 웹보다 많이 태웁니다" if web_used > web_avail
        else "웹 소모가 확보 가능한 웹 안입니다",
        f"∫r dt = {web_used:.2f} mm, 상한 {web_avail:.2f} mm "
        f"(= min(케이스 반경 {r_out:.1f}, 끝면 {min(lengths)/2:.1f})), "
        f"요구 압력 {p_req[win].min():.2f}~{p_req[win].max():.2f} MPa, "
        f"A_t {at_for_web:.0f} mm², C_f {cf0:.2f}. "
        + ("상한은 포트 반경 0 (물리적으로 불가)에서 나온 값이라 **가장 관대한 "
           "쪽**입니다 — 넘으면 형상과 무관하게 확정입니다."
           if web_used > web_avail else ""),
        "연소 시간을 줄이거나, 노즐목을 키워 압력(=연소속도)을 낮추거나, "
        "케이스를 키우십시오.",
        {"web_used": web_used, "web_avail": web_avail}))

    # ── 연소면적 실현 가능성 (§4.6.3) — 경고까지만. 이유는 `_ab_upper_bound`.
    ab_req_max = float(ab_req[win].max())
    out.append(Finding(
        "AB_REACH", WARN, ab_req_max <= reach["ab"],
        "요구 연소면적이 우리가 뒤지는 형상족의 도달 범위를 넘습니다"
        if ab_req_max > reach["ab"] else "요구 연소면적이 도달 범위 안입니다",
        f"요구 최대 A_b = {ab_req_max:.0f} mm². 족의 도달 최대 "
        f"{reach['ab']:.0f} mm² (k={reach['k']}, r₀={reach['r0']:.1f}, "
        f"진폭 {reach['amp']:.1f} mm), 최소 특징 상한 {bnd['ab_bound']:.0f} mm². "
        f"⚠ 도달 범위는 **단일 모드 훑기의 하한**이고 상한은 **엄밀하지 "
        f"않습니다** — 그래서 둘 다 '확정 불가' 를 달 수 없습니다.",
        "노즐목을 키워 요구 압력을 낮추거나(A_b ∝ P^(1−n) 로 줄어듭니다), "
        "세그먼트를 늘려 길이를 확보하거나, 끝면 연소를 켜십시오.",
        {"ab_req_max": ab_req_max, "ab_reach": reach["ab"],
         "ab_bound": bnd["ab_bound"], "reach": reach}))

    # ── 입력 세 개(추력·압력·고정 목)가 서로 모순인가 — **항등식 검사**
    #
    #    F = C_f(P, ε)·P·A_t 이므로 추력 곡선과 압력 곡선에서 **함의되는 A_t** 가
    #    나온다. 사용자가 고정 목을 입력했다면 그 값과 일치해야 한다. 안 맞으면
    #    세 입력이 서로 모순이고 **어떤 형상으로도 불가능**하다.
    #
    #    ⚠ 이 검사는 '확정 불가' 를 달 자격이 있다 — 상한·하한 추정이 아니라
    #      **항등식**이기 때문이다. 이 모듈의 다른 물리 검사들이 경고에 머무는
    #      이유(경계가 엄밀하지 않다)가 여기에는 없다.
    #
    #    ⚠⚠ **이 검사가 없었다.** `feasibility_bounds` 는 함의 A_t 를 계산하지만
    #        물리적으로 가능한 범위(가공 한계~케이스 단면)와만 비교한다 — 단위
    #        오류를 잡는 용도이고 `a_throat` 를 인자로 받지도 않는다. 그리고
    #        `PHYS_THROAT_FIXED` 는 압력 곡선을 쓰지 않는다. 그래서 세 입력이
    #        서로 모순인 경우가 **최적화를 다 돌린 뒤에** '실현 불가' 로
    #        발견됐다. 입력만 보고 알 수 있는 것을 70 분 뒤에 알려 주는 셈이다.
    #        도구는 입력(노즐목·케이스)을 바꾸지 않으므로, 맞지 않으면 판정은
    #        "이 곡선으로는 주어진 조건에서 맞는 형상을 찾을 수 없습니다" 다.
    if a_throat is not None and getattr(target, "p", None) is not None:
        from ..optimize.loss import fp_ratio_spread
        fp = fp_ratio_spread(
            target.t, target.f, target.p,
            gamma=motor.gamma, p_ambient=motor.p_ambient,
            cf_eff=motor.cf_eff,
            # ⚠ **정방향과 같은 C_f 를 써야 한다.** 이 함수는 `F/P` 를 C_f 로
            #   나눠 노즐목을 역산하므로, 정방향이 박리를 보는데 여기서 안 보면
            #   역산이 체계적으로 틀어진다.
            separation_frac=getattr(motor, "separation_frac", 0.0))
        if fp is not None:
            at_imp = float(fp["at_mean"])
            rel = abs(at_imp - float(a_throat)) / max(float(a_throat), 1e-9)
            d_imp = 2.0 * np.sqrt(at_imp / np.pi)
            d_in = 2.0 * np.sqrt(float(a_throat) / np.pi)
            out.append(Finding(
                "PHYS_FPAT", FATAL, rel <= AT_IMPLIED_TOL,
                "입력한 추력·압력·노즐목 세 개가 서로 모순입니다"
                if rel > AT_IMPLIED_TOL else
                "입력한 추력·압력·노즐목이 서로 맞습니다",
                f"F = C_f(P, ε)·P·A_t 이므로 두 곡선에서 함의되는 노즐목은 "
                f"**{at_imp:.1f} mm² (Ø{d_imp:.2f} mm)** 인데 입력한 목은 "
                f"{float(a_throat):.1f} mm² (Ø{d_in:.2f} mm) 입니다 "
                f"— {100*rel:+.1f} % 차이 (허용 {100*AT_IMPLIED_TOL:.0f} %, "
                f"ε {fp['eps']:.2f}). "
                + ("세 입력이 한 모터에서 나올 수 없으므로 **어떤 형상으로도 "
                   "불가능**합니다. 형상 문제가 아니라 입력 문제입니다."
                   if rel > AT_IMPLIED_TOL else ""),
                f"목을 Ø{d_imp:.2f} mm ({at_imp:.1f} mm²) 로 바꾸거나, 두 곡선 중 "
                f"하나를 고치십시오. 압력이 절대압 [MPa] 인지도 확인하십시오 "
                f"(게이지압이면 대기압을 더해야 합니다).",
                {"at_implied": at_imp, "at_input": float(a_throat), "rel": rel,
                 "d_implied_mm": d_imp, "d_input_mm": d_in,
                 "eps": fp["eps"], "at_spread": fp["spread"]}))

    # ── 고정 노즐목 도달 범위 + 필요 목 직경 역제시
    if a_throat is not None:
        out.append(_throat_reach(motor, limits, lengths, f_max, float(a_throat),
                                 reach, p_tab))
    return out


def _throat_reach(motor, limits, lengths, f_max: float, a_throat: float,
                  reach: dict, p_tab) -> Finding:
    """입력한 목으로 낼 수 있는 추력 범위 안에 목표가 있는가 — 없으면 역제시.

    도달 범위의 양 끝:

        위  족의 최대 연소면적 → 평형압 → F = C_f·P·A_t
        아래 원형 포트(진폭 0, 최소 r₀) → 평형압 → 같은 식

    평형압은 `ρ_p·A_b·r(P) = P·A_t/c*` 를 `P` 에 대해 푼 것이다. `r = aP^n`
    이 구간마다 다르므로 닫힌 해가 없다 — 이분법으로 푼다 (단조이므로 안전).
    """
    import torch

    def p_eq(ab: float, at: float) -> float:
        lo, hi = 1e-3, p_tab[1] * 3.0
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            r = float(motor.br.rate(torch.as_tensor(mid, dtype=motor.dtype)))
            need = mid * at * 1e9 / (motor.c_star_actual * motor.rho_p * r)
            if need < ab:          # 그 압력을 내려면 더 큰 면적 필요 → 압력 올림
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    # 도달 압력의 천장 — 케이스가 터지는 압력과 연소속도 표의 상한 중 낮은 쪽.
    #
    # ⚠ 초판은 이 천장이 없었다. `p_eq` 의 이분 구간 상한(표 상한 × 3)까지
    #   올라갈 수 있어서, 작은 목에서 **1000 MPa 급 평형압**을 '도달 가능'
    #   으로 세고 있었다. 결론이 우연히 맞는 경우도 있었지만 이유가 틀렸다.
    p_ceiling = min(float(limits.p_case_allow), float(p_tab[1]))

    def f_reach(at: float) -> tuple:
        """그 목 면적으로 도달하는 추력 범위 (아래 = 원형, 위 = 족의 최대).

        위 끝은 **압력 천장에서 잘린다** — 족이 더 큰 연소면적을 낼 수 있어도
        케이스가 그 압력을 못 버티면 도달 범위가 아니다.
        """
        ph = min(p_eq(reach["ab"], at), p_ceiling)
        pl = min(p_eq(ab_lo, at), p_ceiling)
        return (_cf_rep(motor, pl) * pl * at, _cf_rep(motor, ph) * ph * at,
                pl, ph)

    r_out = float(motor.ds.r_outer)
    l_tot = float(sum(lengths))
    # 도달 범위의 아래 끝 — 가장 작은 원형 포트 (r₀ = r_tip_min). 더 작게는
    # 꼭짓점 반경 제약이 허락하지 않는다.
    ab_lo = (2 * np.pi * limits.r_tip_min * l_tot
             + (2 * (np.pi * r_out ** 2 - np.pi * limits.r_tip_min ** 2)
                if motor.ends_burn else 0.0))
    f_lo, f_hi, p_lo, p_hi = f_reach(a_throat)
    ok = f_lo <= f_max <= f_hi
    d_t = 2.0 * np.sqrt(a_throat / np.pi)

    act = ""
    if not ok:
        # 역제시 — F = C_f·P·A_t 이고 P 도 A_t 에 의존하므로 A_t 를 훑는다.
        good = [float(at) for at in np.geomspace(1.0, np.pi * r_out ** 2, 160)
                if f_reach(float(at))[0] <= f_max <= f_reach(float(at))[1]]
        if good:
            d1, d2 = (2 * np.sqrt(g / np.pi) for g in (min(good), max(good)))
            act = (f"목 직경을 **{d1:.2f} ~ {d2:.2f} mm** "
                   f"(면적 {min(good):.1f}~{max(good):.1f} mm²) 로 바꾸면 "
                   f"목표 최대추력이 도달 범위 안에 들어옵니다.")
        else:
            act = ("어떤 목 직경으로도 이 최대추력에 닿지 않습니다 — 추력 "
                   "목표나 케이스를 바꿔야 합니다.")
    return Finding(
        "PHYS_THROAT_FIXED", WARN, ok,
        "목표 최대추력이 이 노즐목의 도달 범위를 벗어납니다" if not ok
        else "목표 최대추력이 이 노즐목의 도달 범위 안입니다",
        f"목 Ø{d_t:.2f} mm ({a_throat:.1f} mm²) 의 도달 추력 "
        f"{f_lo:.0f} ~ {f_hi:.0f} N (평형압 {p_lo:.2f} ~ {p_hi:.2f} MPa, "
        f"천장 {p_ceiling:.2f} MPa = min(케이스 허용, 표 상한)), "
        f"목표 {f_max:.0f} N. 아래 끝은 원형 포트 r₀={limits.r_tip_min:.1f} mm, "
        f"위 끝은 족의 최대 연소면적이거나 압력 천장입니다 — **위 끝은 하한**"
        f"이라 넘어도 불가능이 증명되지는 않습니다.",
        act, {"f_reach": (f_lo, f_hi), "p_reach": (p_lo, p_hi), "d_t": d_t})


# ═══════════════════════════════════════════════════════════════════ 묶음
def check_inputs(target, motor=None, limits=None, lengths=(95.0,),
                 a_throat: float = None, with_bounds: bool = True) -> Report:
    """형식 → 일관성 → 물리 순으로 전부 돌린다.

    `motor` 가 없으면 형식·일관성만 돈다 (마법사가 입력받는 즉시 쓰는 길).
    형식 검사에서 확정 불가가 나오면 **거기서 멈춘다** — 깨진 배열 위의 물리
    진단은 틀린 곳을 가리킨다.
    """
    out = list(check_format(target.t, target.f, getattr(target, "p", None)))
    if any(not f.ok and f.level == FATAL for f in out):
        return Report(out)
    out += check_consistency(target)
    if motor is not None and limits is not None:
        out += check_physics(target, motor, limits, lengths, a_throat)
        if with_bounds:
            from ..optimize.constraints import ThroatSpec
            from ..optimize.runner import feasibility_bounds
            # 고정 목이 입력으로 들어왔으면 그것까지 맞대 본다 — F·P·A_t 가
            # 항등식이므로 셋이 서로 모순이면 형상으로 메울 수 없다.
            thr = (ThroatSpec(d_mm=2.0 * float(np.sqrt(a_throat / np.pi)))
                   if a_throat else None)
            fb = feasibility_bounds(motor, target, limits, lengths, throat=thr)
            out.append(Finding(
                "BOUNDS", FATAL, bool(fb.get("feasible", True)),
                "물리 한계 검사" + ("" if fb.get("feasible", True) else " — 실패"),
                fb.get("note", ""), "", fb))
    return Report(out)
