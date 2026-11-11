"""연소속도 모델 — 두 층.

1. **전 구간 연속 모델** (기본): Nakka 구간표의 각 멱법칙을 log-log 평면에서
   **폭 2δ 의 smoothstep 으로 직접 섞어** 0 ~ 상한 전체에서 끊김 없는 `r(P)` 를
   만든다. 점화·테일오프에서 압력은 저압 대역을 반드시 지나므로, 어떤 대역도
   모델에서 비울 수 없다.
2. **단일 멱법칙** `r = a·P^n`: 설계 대역 안에서의 해석·설명용.

⚠ 음수 n 이 불안정을 뜻하지 않는다. 평형이 불안정해지는 조건은 n > 1 이다.
   0.807~1.503 MPa 를 설계 대역에서 빼는 이유는 mesa(평탄·감소) 거동,
   안정성 여유 부족, 구간 불연속 인접이다 (§4.3).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# (P_lo[MPa], P_hi[MPa], a[mm/s·MPa^-n], n)  — Richard Nakka, burnrate.html
NAKKA_TABLE = {
    "KNSB": [(0.101, 0.807, 10.708, 0.625),
             (0.807, 1.503, 8.763, -0.314),
             (1.503, 3.792, 7.852, -0.013),
             (3.792, 7.033, 3.907, 0.535),
             (7.033, 10.67, 9.653, 0.064)],
    "KNDX": [(0.101, 0.779, 8.875, 0.619),
             (0.779, 2.572, 7.553, -0.009),
             (2.572, 5.930, 3.841, 0.688),
             (5.930, 8.502, 17.20, -0.148),
             (8.502, 11.20, 4.775, 0.442)],
    "KNSU": [(0.101, 10.67, 8.26, 0.319)],
}

# 설계 대역 (기본 / 저압) — §4.3
DESIGN_BANDS = {
    "KNSB": {"default": (3.792, 7.033), "low": (1.503, 3.792)},
    "KNDX": {"default": (2.572, 5.930), "low": (0.779, 2.572)},
}

# 설계 대역에서 제외하는 구간 (mesa·불연속 인접)
EXCLUDED_BANDS = {"KNSB": [(0.807, 1.503)], "KNDX": [(5.930, 8.502)]}


def piecewise_rate(table, p_mpa):
    """원본 구간표 그대로의 r [mm/s] — 경계에서 불연속. 비교용."""
    p = np.atleast_1d(np.asarray(p_mpa, dtype=float))
    out = np.empty_like(p)
    for i, pi in enumerate(p):
        a, n = table[0][2], table[0][3]
        if pi >= table[-1][1]:
            a, n = table[-1][2], table[-1][3]
        else:
            for lo, hi, aa, nn in table:
                if lo <= pi < hi:
                    a, n = aa, nn
                    break
        out[i] = a * max(pi, 1e-9) ** n
    return out


# 경계 전이 폭 (ln P 기준, 사실상 상대 폭). 'n < 0.9 보장' 에서 역산한 값.
DEFAULT_DELTA = 0.07


@dataclass
class BurnRateModel:
    """전 구간 연속 `r(P)`.

    구간 경계에서 표의 값은 불연속이다(KNSB 3.792 MPa 에서 7.717 → 7.971).
    경계마다 폭 2δ 의 C¹ smoothstep 으로 **두 멱법칙을 직접 섞는다**.

        u     = clip( (lnP − (lnP_b − δ)) / 2δ , 0, 1)
        s     = u²(3 − 2u)                      # C¹, 양 끝에서 도함수 0
        ln r  = (1 − s)·ln r_왼쪽 + s·ln r_오른쪽

    전이 구간 **밖에서는 가중치가 정확히 0 또는 1** 이므로 모델이 곧 원래
    멱법칙 그 자체가 된다 (오차 ~1e-14). δ 는 매듭 수 같은 이산 파라미터와
    분리된 **명시적 전이 폭** 이라, 국소 지수 n 의 상한을 직접 통제할 수 있다.

    ⚠ 초판은 경계의 두 값을 평균한 매듭 하나를 박고 PCHIP 으로 이었다. 그때는
       전이 폭이 매듭 간격에 묶여, 구간 내부를 잘 맞추려고 매듭을 늘리면 전이가
       가팔라져 국소 n 이 1 을 넘었다 (3.906~3.930 MPa 에서 n = 1.0064).
       n > 1 은 평형이 불안정해지는 조건이고, 평형압의 설계변수 민감도가
       1/(1−n) 이라 최적화기 경사가 발산한다.

    δ = 0.07 에서 구간 내부 오차 6e-14 %, 국소 n 최댓값 0.865.
    """

    name: str
    table: list
    p_min: float = 1e-3            # [MPa] 이 아래는 최저 구간으로 외삽
    p_max: float = None            # [MPa] 보고용 상한
    delta: float = DEFAULT_DELTA   # 경계 전이 폭 (ln P 기준)

    def __post_init__(self):
        self.p_lo_table = float(self.table[0][0])
        self.p_hi_table = float(self.table[-1][1])
        if self.p_max is None:
            self.p_max = self.p_hi_table * 1.5
        # 전이 구간이 서로 겹치면 섞임 순서가 의미를 잃는다.
        bnds = [t[0] for t in self.table[1:]]
        for lo_b, hi_b in zip(bnds[:-1], bnds[1:]):
            if np.log(hi_b) - np.log(lo_b) <= 2.0 * self.delta:
                raise ValueError(
                    f"전이 폭 δ={self.delta} 가 너무 커서 경계 {lo_b}/{hi_b} 가 겹친다")
        self._ln_a = np.array([np.log(t[2]) for t in self.table])
        self._n = np.array([t[3] for t in self.table])
        self._ln_pb = np.array([np.log(t[0]) for t in self.table[1:]])

    # ------------------------------------------------------------------ 핵심
    def _ln_r(self, ln_p):
        """log 압력 → log 연소속도. 벡터화되어 있고 어디서나 미분 가능하다."""
        ln_p = np.asarray(ln_p, dtype=float)
        out = self._ln_a[0] + self._n[0] * ln_p
        d = self.delta
        for k, ln_pb in enumerate(self._ln_pb, start=1):
            nxt = self._ln_a[k] + self._n[k] * ln_p
            u = np.clip((ln_p - (ln_pb - d)) / (2.0 * d), 0.0, 1.0)
            w = u * u * (3.0 - 2.0 * u)          # smoothstep
            out = (1.0 - w) * out + w * nxt
        return out

    def _dln_r(self, ln_p):
        """d ln r / d ln P — 해석 도함수 (= 국소 압력지수 n)."""
        ln_p = np.asarray(ln_p, dtype=float)
        out = self._ln_a[0] + self._n[0] * ln_p
        dout = np.full(ln_p.shape, self._n[0], dtype=float)
        d = self.delta
        for k, ln_pb in enumerate(self._ln_pb, start=1):
            nxt = self._ln_a[k] + self._n[k] * ln_p
            dnxt = np.full(ln_p.shape, self._n[k], dtype=float)
            t = (ln_p - (ln_pb - d)) / (2.0 * d)
            u = np.clip(t, 0.0, 1.0)
            w = u * u * (3.0 - 2.0 * u)
            dw = np.where((t > 0.0) & (t < 1.0), 6.0 * u * (1.0 - u) / (2.0 * d), 0.0)
            dout = (1.0 - w) * dout + w * dnxt + dw * (nxt - out)
            out = (1.0 - w) * out + w * nxt
        return dout

    def rate(self, p_mpa):
        """r [mm/s] — 0 이상 어떤 압력에서도 유한한 값을 준다."""
        p = np.atleast_1d(np.asarray(p_mpa, dtype=float))
        p = np.clip(p, self.p_min, None)
        return np.maximum(np.exp(self._ln_r(np.log(p))), 0.0)

    def exponent(self, p_mpa):
        """국소 압력 지수 n = d ln r / d ln P."""
        p = np.atleast_1d(np.asarray(p_mpa, dtype=float))
        return self._dln_r(np.log(np.clip(p, self.p_min, None)))

    def max_exponent_in(self, band, n_samples: int = 20000) -> float:
        """구간 안에서의 국소 지수 최댓값 — 평형 안정성 판정용 (n < 1)."""
        lo, hi = band
        return float(self.exponent(np.geomspace(lo, hi, n_samples)).max())

    # ------------------------------------------------------- 단일 멱법칙
    def power_law_in_band(self, band):
        """설계 대역 [MPa] 에서 r(P) 를 가장 잘 맞추는 (a, n) 을 회귀로 얻는다."""
        lo, hi = band
        p = np.geomspace(lo, hi, 64)
        n, ln_a = np.polyfit(np.log(p), np.log(self.rate(p)), 1)
        return float(np.exp(ln_a)), float(n)


def model_for(propellant: str) -> BurnRateModel:
    if propellant not in NAKKA_TABLE:
        raise KeyError(f"연소속도 표가 없는 추진제: {propellant}")
    return BurnRateModel(propellant, NAKKA_TABLE[propellant])


def design_band(propellant: str, which: str = "default"):
    return DESIGN_BANDS[propellant][which]


def in_excluded_band(propellant: str, p_mpa) -> np.ndarray:
    """설계에서 제외한 대역 안인가 (정상 연소 구간 판정용)."""
    p = np.atleast_1d(np.asarray(p_mpa, dtype=float))
    bad = np.zeros(p.shape, dtype=bool)
    for lo, hi in EXCLUDED_BANDS.get(propellant, []):
        bad |= (p >= lo) & (p <= hi)
    return bad
