"""포트 윤곽을 CAD 로 내보낼 **다각형**으로 바꾸고, 그 근사의 대가를 잰다.

CAD 파일은 곡선을 결국 유한한 점으로 쓴다 (STL 은 삼각형, STEP 도 우리가 폴리라인
으로 쓰면 직선 구간). 즉 내보내는 순간 **포트 윤곽이 바뀐다.** 바뀐 윤곽은 둘레가
다르고, 둘레는 곧 `A_b` 이므로 **추력 곡선이 달라진다.**

그래서 이 모듈은 두 가지를 같이 한다.

1. 요구한 **현(chord) 공차**를 만족하는 표본 수를 고른다.
2. 그 다각형이 실제로 `A_b` 와 총역적을 몇 % 바꾸는지 **재서 함께 돌려준다.**

⚠ 내보낸 파일의 충실도를 **숫자로 말하지 못하면 내보내지 않은 것만 못하다.**
   V3 검증에서 실측이 목표와 어긋났을 때 '설계 오차'와 '제작 오차'를 가르려면,
   그 사이에 있는 '내보내기 오차'를 먼저 알고 있어야 한다.

현 공차와 표본 수의 관계 — 반경 `R` 의 원호를 `N` 등분하면 중앙 처짐(sagitta)은

    s = R·(1 − cos(π/N)) ≈ R·π²/(2N²)        →        N ≈ π·√(R / (2s))

포트는 원이 아니므로 `R` 자리에 **국소 곡률 반경의 최소값**을 넣어야 안전하다.
극좌표 곡률은 `κ = (r² + 2r'² − r·r'') / (r² + r'²)^{3/2}` 이고 `R_min = 1/κ_max`.
스타 꼭짓점이 바로 그 최소 곡률 반경 자리라, 로브가 깊을수록 표본이 더 필요하다.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..geometry.port import FourierPort

#: 기본 현 공차 [mm]. FDM 프린터의 레이어·노즐 해상도(0.2~0.4 mm)보다 한 자릿수
#: 작게 잡는다 — 내보내기 오차가 제작 오차에 묻히도록.
#:
#: ⚠⚠ **이 값이 δ 를 만들었다는 사실을 알고 쓸 것**.
#:     `runner.DELTA_TRANSFER_MM` 의 지배항 "테셀레이션 처짐 0.0198 mm" 는
#:     여기서 나온 값이다 — `n_theta_for_tolerance` 가 이 공차를 만족하는 최소
#:     표본 수를 잡으므로 처짐이 공차를 **1:1 로 따라간다**
#:     (0.02 → 0.0198, 0.01 → 0.0099, 0.005 → 0.0050).
#:
#:     그래서 δ 는 전사 정밀도의 측정이 아니라 이 설정값의 그림자였고, 진폭
#:     유의성 판정에서 **뺐다**. 지금 이 공차가 지켜야 할 조건은 하나뿐이다 —
#:     **모델 진폭 바닥보다 한참 작을 것.** 운용 설정의 바닥이 0.60 mm 이므로
#:     0.02 는 30 배 여유가 있다 (`amplitude_floor` 참조). 이 공차를 키울 때는
#:     그 비교를 다시 할 것.
DEFAULT_CHORD_TOL: float = 0.02


def _derivs(port: FourierPort, th):
    """r(θ), r'(θ), r''(θ) — 해석 미분 (`constraints._radius_and_derivs` 와 같은 식)."""
    a = np.asarray(port.a, dtype=float)
    b = np.asarray(port.b, dtype=float)
    k = np.arange(1, a.size + 1)[:, None]
    ang = k * np.asarray(th)[None, :]
    c, s = np.cos(ang), np.sin(ang)
    r = float(port.r0) + (a[:, None] * c + b[:, None] * s).sum(0)
    rp = (k * (-a[:, None] * s + b[:, None] * c)).sum(0)
    rpp = (k ** 2 * (-a[:, None] * c - b[:, None] * s)).sum(0)
    return r, rp, rpp


def min_curvature_radius(port: FourierPort, n_theta: int = 20000) -> float:
    """윤곽에서 가장 뾰족한 곳의 곡률 반경 [mm]. 볼록한 쪽만 본다.

    볼록한 꼭짓점(κ > 0)이 표본 수를 지배한다 — 오목한 골짜기는 같은 표본 수에서
    처짐이 더 작다. κ ≤ 0 뿐이면(완전 원형 포함) 평균 반경을 돌려준다.
    """
    th = np.linspace(0.0, 2 * np.pi, n_theta, endpoint=False)
    r, rp, rpp = _derivs(port, th)
    kappa = (r ** 2 + 2 * rp ** 2 - r * rpp) / (r ** 2 + rp ** 2) ** 1.5
    k_max = float(kappa.max())
    return float(r.mean()) if k_max <= 0.0 else 1.0 / k_max


def n_theta_for_tolerance(port: FourierPort, chord_tol: float = DEFAULT_CHORD_TOL,
                          n_min: int = 64, n_max: int = 20000) -> int:
    """현 공차를 **실제로 만족하는** 표본 수.

    ⚠ **닫힌 식을 믿지 않는다.** 처음에는 `N ≈ π·√(R_min/2s)` 를 썼는데, 그 식은
      "반경 R 인 원을 N 등분" 을 가정한다. 포트는 균일한 θ 로 표본하므로 뾰족한
      꼭짓점 근처에서는 호 길이가 훨씬 길고, 그만큼 처짐이 커진다. 실측:

          형상                    닫힌 식 N   그때 실측 처짐   요구 0.02 mm
          원형 r₀ = 12              64         0.0145 mm      만족
          얕은 스타 (3-로브 2.2)      64         0.0397 mm      **2 배 초과**
          깊은 스타 (6-로브 7.0)      64         0.3514 mm      **18 배 초과**

      그래서 식으로 **시작점만** 잡고, `sagitta()` 로 재면서 만족할 때까지 올린다.
      2 배씩 올려 만족시킨 뒤 이분법으로 되돌린다 (4의 배수 유지).

    4의 배수로 맞추는 이유는 `k = 4` 같은 짝수 모드의 꼭짓점이 표본점에 걸리도록
    하기 위해서다 — 꼭짓점이 두 표본 사이에 떨어지면 그 자리만 공차를 넘긴다.
    """
    tol = max(float(chord_tol), 1e-9)
    r_min = min_curvature_radius(port)
    quad = lambda n: int(np.clip(int(np.ceil(n / 4.0) * 4), n_min, n_max))
    n = quad(np.pi * np.sqrt(max(r_min, 1e-9) / (2.0 * tol)))
    if sagitta(port, n, n_probe=40000) <= tol:
        return n
    lo = n
    while n < n_max:                                   # 만족할 때까지 2 배씩
        n = quad(min(2 * n, n_max))
        if sagitta(port, n, n_probe=40000) <= tol:
            break
        lo = n
    hi = n
    while hi - lo > 4:                                 # 이분법으로 되돌린다
        mid = quad((lo + hi) // 2)
        if mid <= lo or mid >= hi:
            break
        if sagitta(port, mid, n_probe=40000) <= tol:
            hi = mid
        else:
            lo = mid
    return hi


def sagitta(port: FourierPort, n_theta: int, n_probe: int = 200000) -> float:
    """다각형과 참 윤곽의 **최대 반경 방향 거리** [mm] — 실제 현 오차.

    공식(`R(1−cos(π/N))`)은 국소 원호 근사라 낙관적일 수 있어, 조밀한 각도에서
    참 반경과 다각형(선분) 반경의 차이를 직접 잰다.
    """
    th_poly = np.linspace(0.0, 2 * np.pi, int(n_theta), endpoint=False)
    r_poly, _, _ = _derivs(port, th_poly)
    xs = r_poly * np.cos(th_poly)
    ys = r_poly * np.sin(th_poly)
    th = np.linspace(0.0, 2 * np.pi, int(n_probe), endpoint=False)
    r_true, _, _ = _derivs(port, th)
    # 각 탐침 각도가 속한 선분을 찾아 그 선분과 원점-방향 직선의 교점 반경을 구한다
    idx = np.floor(th / (2 * np.pi / n_theta)).astype(int) % int(n_theta)
    j = (idx + 1) % int(n_theta)
    x1, y1, x2, y2 = xs[idx], ys[idx], xs[j], ys[j]
    cs, sn = np.cos(th), np.sin(th)
    # (x1,y1) + t·((x2,y2)-(x1,y1)) 가 방향 (cs,sn) 위에 놓이는 t
    den = (x2 - x1) * sn - (y2 - y1) * cs
    num = y1 * cs - x1 * sn
    t = np.divide(num, den, out=np.zeros_like(num), where=np.abs(den) > 1e-12)
    r_seg = (x1 + t * (x2 - x1)) * cs + (y1 + t * (y2 - y1)) * sn
    return float(np.max(np.abs(r_true - r_seg)))


@dataclass
class ProfileFidelity:
    """내보낸 다각형이 설계 윤곽에서 얼마나 벗어나는가 — 파일과 함께 나가는 숫자."""

    n_theta: int
    chord_tol_mm: float
    sagitta_mm: float                 # 실측 최대 반경 편차
    area_err_pct: float               # 포트 단면적 상대 오차
    perimeter_err_pct: float          # 초기 둘레(= A_b 의 주 항) 상대 오차
    min_curv_radius_mm: float
    notes: list = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"표본 {self.n_theta} 점 (현 공차 {self.chord_tol_mm:.3f} mm 요구)",
            f"  실측 최대 반경 편차 {self.sagitta_mm:.4f} mm",
            f"  포트 단면적 오차   {self.area_err_pct:+.4f} %",
            f"  초기 둘레 오차     {self.perimeter_err_pct:+.4f} %",
            f"  최소 곡률 반경     {self.min_curv_radius_mm:.3f} mm",
        ]
        return "\n".join(lines + [f"  ⚠ {n}" for n in self.notes])


def polygon(port: FourierPort, chord_tol: float = DEFAULT_CHORD_TOL,
            n_theta: int = None, shrink_pct: float = 0.0):
    """포트 윤곽 → 닫힌 다각형 `(N, 2)` 와 충실도 보고.

    Parameters
    ----------
    shrink_pct : float
        수축률 보정 [%]. 몰드·코어 치수를 이만큼 **키운다** — 굳으면서 줄어든
        결과가 설계 치수에 오도록. 그레인 자체를 내보낼 때는 0 으로 둔다.
        ⚠ 시편 시험 T4 전에는 값이 없으므로 기본 0 이고, 그때는 보고서에
          "수축 미보정" 이 적힌다.

    Returns
    -------
    (pts, fidelity) : (np.ndarray (N,2), ProfileFidelity)
    """
    n = int(n_theta) if n_theta else n_theta_for_tolerance(port, chord_tol)
    th = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    r, _, _ = _derivs(port, th)
    scale = 1.0 + float(shrink_pct) / 100.0
    pts = np.column_stack([r * np.cos(th), r * np.sin(th)]) * scale

    # ── 충실도: 면적과 둘레를 참값과 비교한다 ────────────────────────────
    #    참 면적은 해석해 (`FourierPort.area` 는 주기 사다리꼴이라 기계 정밀도).
    #    다각형 면적은 신발끈 공식.
    a_true = port.area() * scale ** 2
    x, y = pts[:, 0], pts[:, 1]
    a_poly = 0.5 * float(np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
    # 참 둘레 ∮√(r² + r'²) dθ — 주기 적분이라 균일 합이 곧 사다리꼴
    thf = np.linspace(0.0, 2 * np.pi, 20000, endpoint=False)
    rf, rpf, _ = _derivs(port, thf)
    p_true = float(np.sum(np.sqrt(rf ** 2 + rpf ** 2)) * (2 * np.pi / thf.size)) * scale
    p_poly = float(np.sum(np.linalg.norm(np.diff(pts, axis=0, append=pts[:1]), axis=1)))

    sag = sagitta(port, n) * scale
    notes = []
    if sag > chord_tol * 1.5:
        notes.append(f"실측 처짐 {sag:.4f} mm 가 요구 공차 {chord_tol:.3f} mm 를 "
                     f"넘는다 — n_theta 를 직접 올려 잡으십시오")
    # ⚠ 여기서 수축을 논평하지 않는다 — `polygon` 은 자기가 무엇을 만드는
    #   중인지 모른다. 그레인은 **일부러** 보정 없이(설계 치수로) 뽑는다.
    #   어떤 보정이 걸렸는지는 `parts.export_set` 이 `FitSpec` 을 보고 적는다.
    fid = ProfileFidelity(
        n_theta=n, chord_tol_mm=float(chord_tol), sagitta_mm=sag,
        area_err_pct=100.0 * (a_poly / a_true - 1.0),
        perimeter_err_pct=100.0 * (p_poly / p_true - 1.0),
        min_curv_radius_mm=min_curvature_radius(port), notes=notes)
    return pts, fid


def circle(radius: float, chord_tol: float = DEFAULT_CHORD_TOL,
           n_theta: int = None, shrink_pct: float = 0.0) -> np.ndarray:
    """원 다각형 — 케이스 벽·피스톤 외경처럼 포트가 아닌 원에 쓴다."""
    n = int(n_theta) if n_theta else max(
        64, int(np.ceil(np.pi * np.sqrt(radius / (2.0 * max(chord_tol, 1e-9))) / 4) * 4))
    th = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    rr = float(radius) * (1.0 + float(shrink_pct) / 100.0)
    return np.column_stack([rr * np.cos(th), rr * np.sin(th)])


def offset_outward(pts: np.ndarray, delta: float) -> np.ndarray:
    """다각형을 바깥으로 `delta` [mm] 만큼 민다 — **민코프스키 확장**.

    끼워맞춤 간극용이다. 반환 윤곽은 원래 영역에서 법선 방향으로 최소 `delta`
    떨어져 있고, **어떤 형상에서도 자기교차하지 않는다.**

    ═══════════════════════════════════════════════════════════════════════
    ⚠⚠ **옛 `offset_convex` 는 오목 모서리에서 틀린 연산이었다.**

      그 방식은 꼭짓점을 **평균 법선 방향으로 밀어내는** 것이었는데, 오목부에서는
      인접 변의 오프셋이 서로 **교차**한다. 그런데 자기교차 검사가 "길이 0 선분"
      만 보고 있어서 **조용히 깨진 다각형을 내보냈다** — 그 다각형은 피스톤·
      엔드플레이트의 **구멍**으로 들어가므로 못 쓰는 STL 이 나온다.

      스타 포트의 **가시 끝**(포트 윤곽의 오목 노치)이 바로 그 자리다. 노치 곡률
      반경은 `R = r²/(r − r'')`, `r = r0 − a_k`, `r'' = a_k·k²` 이고
      **지관 지름과 무관하다** — 포트 형상만의 함수다. 실측한 개시점:

          간극/|R|    0.83   1.17 │ 1.65   3.55   9.38
          옛 방식      OK     OK  │  ✗      ✗      ✗
          민코프스키   OK     OK  │ OK     OK     OK

      ⚠ **"작은 지관에서는 안 된다" 가 아니다.** 설계를 s 배 키우면 `R` 도 s 배인데
        간극은 절대값이라 안 변한다 — **작은 설계일수록** 같은 간극이 상대적으로
        커진다. 큰 지관에 작은 포트를 넣어도 똑같이 걸린다.

      민코프스키 확장은 오목 모서리를 **반경 `delta` 로 둥글려** 이 문제를 없앤다.
      기능적으로도 그것이 맞다 — 피스톤 구멍이 가시 끝의 0.1 mm 짜리 뾰족함을
      복제할 이유가 없고(가공도 안 된다), 핀만 통과시키면 된다.
    ═══════════════════════════════════════════════════════════════════════

    `shapely` 가 없으면 옛 방식으로 되돌아가되 **진짜 자기교차 검사**를 하고
    걸리면 거부한다 (조용히 깨진 형상을 내보내는 것보다 낫다).
    """
    p_ = np.asarray(pts, dtype=float)
    d = float(delta)
    if d <= 0.0:
        return p_.copy()
    try:
        from shapely.geometry import Polygon
    except Exception:
        return _offset_vertex_normal(p_, d)

    poly = Polygon(p_)
    if not poly.is_valid:
        poly = poly.buffer(0)
    buf = poly.buffer(d, join_style=1, quad_segs=32)      # 1 = round
    if buf.is_empty or buf.geom_type != "Polygon":
        raise ValueError(
            f"간극 {d:g} mm 확장이 단일 다각형을 내지 못했다 "
            f"(결과 {buf.geom_type}). 입력 윤곽을 확인하십시오.")
    out = np.asarray(buf.exterior.coords[:-1], dtype=float)
    if len(out) < 3:
        raise ValueError(f"간극 {d:g} mm 확장 결과가 퇴화했다")
    return out


def offset_inward(pts: np.ndarray, delta: float) -> np.ndarray:
    """다각형을 안쪽으로 `delta` [mm] 만큼 당긴다 — **민코프스키 침식**.

    **억지끼움(압입) 씰**에 쓴다. 피스톤·엔드플레이트의 추진제 쪽 끝면에서
    구멍을 포트보다 `delta` 만큼 **작게** 뚫어 실리콘 핀을 눌러 물린다. 그러면
    압축 캐스팅 중에 페이스트가 핀↔판 틈으로 기어올라 **플래시 고리**를 만들고,
    탈형 때 그 고리가 가시를 부러뜨리는 경로가 막힌다.

    ⚠ **침식은 확장과 달리 형상을 끊을 수 있다.** 얇은 목이 있으면 둘로 쪼개지고,
      너무 당기면 사라진다. 그러면 거부한다 — 조용히 깨진 구멍을 내보내지 않는다.
      침식은 **볼록** 꼭짓점을 반경 `delta` 로 둥글리고 오목 노치는 날카롭게
      만든다 (확장의 거울상).
    """
    p_ = np.asarray(pts, dtype=float)
    d = float(delta)
    if d <= 0.0:
        return p_.copy()
    try:
        from shapely.geometry import Polygon
    except Exception as exc:                                   # pragma: no cover
        raise RuntimeError(
            "침식 오프셋은 shapely 가 필요하다 — pip install shapely") from exc

    poly = Polygon(p_)
    if not poly.is_valid:
        poly = poly.buffer(0)
    a0 = float(poly.area)
    buf = poly.buffer(-d, join_style=1, quad_segs=32)           # 1 = round
    if buf.is_empty:
        raise ValueError(
            f"씰 압입 {d:g} mm 로 당기면 구멍이 사라진다 (포트 최소 반경보다 크다)")
    if buf.geom_type != "Polygon":
        raise ValueError(
            f"씰 압입 {d:g} mm 로 당기면 구멍이 {buf.geom_type} 로 쪼개진다 — "
            f"포트에 얇은 목이 있다. 압입량을 줄이십시오.")
    if buf.interiors:
        raise ValueError(f"씰 압입 {d:g} mm 결과에 내부 고리가 생겼다")
    out = np.asarray(buf.exterior.coords[:-1], dtype=float)
    if len(out) < 3 or float(buf.area) >= a0:
        raise ValueError(f"씰 압입 {d:g} mm 결과가 퇴화했다")
    return out


def _segments_cross(p: np.ndarray) -> int:
    """비인접 선분 쌍의 **실제 교차** 개수 — 대체 경로의 결과를 검사한다."""
    n = len(p)
    q = np.roll(p, -1, axis=0)
    cnt = 0
    for i in range(n):
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue
            a1, a2, b1, b2 = p[i], q[i], p[j], q[j]
            d1, d2 = a2 - a1, b2 - b1
            den = d1[0] * d2[1] - d1[1] * d2[0]
            if abs(den) < 1e-14:
                continue
            t = ((b1[0] - a1[0]) * d2[1] - (b1[1] - a1[1]) * d2[0]) / den
            u = ((b1[0] - a1[0]) * d1[1] - (b1[1] - a1[1]) * d1[0]) / den
            if 1e-9 < t < 1 - 1e-9 and 1e-9 < u < 1 - 1e-9:
                cnt += 1
    return cnt


def _offset_vertex_normal(p: np.ndarray, delta: float) -> np.ndarray:
    """옛 방식(평균 법선으로 밀기) — `shapely` 가 없을 때의 대체 경로.

    ⚠ 오목 모서리에서 자기교차한다. **결과를 검사하고 걸리면 거부한다.**
    """
    nxt = np.roll(p, -1, axis=0)
    prv = np.roll(p, 1, axis=0)
    e1, e2 = p - prv, nxt - p
    n1 = np.column_stack([e1[:, 1], -e1[:, 0]])
    n2 = np.column_stack([e2[:, 1], -e2[:, 0]])
    n1 /= np.maximum(np.linalg.norm(n1, axis=1, keepdims=True), 1e-12)
    n2 /= np.maximum(np.linalg.norm(n2, axis=1, keepdims=True), 1e-12)
    nv = n1 + n2
    nv /= np.maximum(np.linalg.norm(nv, axis=1, keepdims=True), 1e-12)
    cosh = np.clip((n1 * nv).sum(axis=1), 0.2, 1.0)
    out = p + nv * (float(delta) / cosh[:, None])
    if _segments_cross(out) > 0:
        raise ValueError(
            f"간극 {delta:g} mm 로 밀면 윤곽이 자기교차한다 (오목 노치의 곡률 "
            f"반경보다 간극이 크다). `shapely` 를 설치하면 민코프스키 확장으로 "
            f"정확히 처리된다 — pip install shapely")
    return out


#: 옛 이름 — 호출부 호환용. 연산이 바뀌었으므로 새 이름을 쓰는 것이 맞다.
offset_convex = offset_outward
