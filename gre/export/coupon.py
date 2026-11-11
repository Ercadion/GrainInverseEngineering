"""시편 시험 마스터 — 제작 제약 상수를 재는 "빗" 시편.

**무엇을 만드는가.** 3D 프린팅하는 **강체 마스터**다. 여기에 실리콘을 부으면
슬롯이 실리콘 **지느러미**가 된다. 그 지느러미가 그레인의 포트 슬롯(= 스타 로브)을
만드는 코어이므로, 지느러미가 견디는 최소 두께가 곧 `t_void_min` 이다.

    마스터(슬롯) ──실리콘 주입→ 코어(지느러미) ──추진제 캐스팅→ 그레인(슬롯)

한 판으로 네 가지를 동시에 훑는다.

| 구역 | 훑는 것 | 결정하는 값 |
|---|---|---|
| A·B·C 열 (폭 × 깊이) | 지느러미 두께·높이 | `t_void_min`, 세장비 상한 |
| W 열 (슬롯 **간격**) | 추진제 살 두께 | `t_prop_min` |
| R 열 (쐐기 끝 반경) | 형상 전사 한계 | `r_tip_min` |
| 판 전체 (치수 비교) | 수축률·전사 오차 | T3/T4 입력 |

⚠ **살 두께는 간격으로 재야 한다.** A·B·C 열의 간격은 슬롯 폭에 묶여 있어
   (서로 간섭하지 않게 넓게 띄운다) 살 두께 시편이 되지 못한다. 그래서 슬롯 폭을
   고정하고 **간격만** 바꾸는 W 열을 따로 둔다.

⚠ **꼭짓점 반경 시편은 XY 평면의 쐐기다.** 그레인에서 스타 꼭짓점은 단면(XY)에
   있는 특징이고 축방향으로 뽑힌 형상이다. 깊이 방향으로 둥근 홈을 파면 전혀
   다른 것(프린터의 층 방향 해상도)을 재게 된다. 같은 방향으로 재야 값이 쓰인다.

⚠ **쐐기의 반경과 폭을 분리한다.** 슬롯 끝을 반원으로 막으면 꼭짓점 반경이
   슬롯 폭의 절반으로 묶여 `t_void_min` 과 `r_tip_min` 이 섞인다. 쐐기(끼인각
   고정, 끝만 반경 ρ)로 만들면 둘이 독립이다.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass

import numpy as np

from . import solid as S

#: §3.1 의 기본 구성
SLOT_WIDTHS = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0)      # [mm] = 지느러미 두께
SLOT_DEPTHS = (5.0, 10.0, 20.0)                   # [mm] = 지느러미 높이
WEB_GAPS = (1.0, 1.5, 2.0, 3.0, 4.0)              # [mm] = 추진제 살 두께
TIP_RADII = (0.2, 0.4, 0.8, 1.5)                  # [mm] 꼭짓점 반경


@dataclass
class CouponSpec:
    """빗 시편 마스터의 치수. 전부 [mm]."""

    slot_widths: tuple = SLOT_WIDTHS
    slot_depths: tuple = SLOT_DEPTHS
    web_gaps: tuple = WEB_GAPS
    tip_radii: tuple = TIP_RADII
    slot_len: float = 25.0          # 슬롯 길이 (한 줄의 길이)
    gap_factor: float = 3.0         # A·B·C 열 간격 = 폭 × 이 값 (서로 간섭 없게)
    gap_min: float = 3.0            # 그 간격의 하한
    web_slot_w: float = 2.0         # W 열의 **고정** 슬롯 폭
    web_depth: float = 10.0         # W 열 깊이
    row_gap: float = 12.0           # 줄 사이 간격
    margin: float = 8.0             # 둘레 벽 안쪽 여유
    floor_extra: float = 4.0        # 가장 깊은 슬롯 아래 남길 바닥 두께
    wall_h: float = 6.0             # 실리콘 배킹층을 담을 둘레 벽 높이
    wall_t: float = 4.0             # 둘레 벽 두께
    wedge_len: float = 10.0         # 쐐기 길이
    wedge_halfangle_deg: float = 25.0
    wedge_gap: float = 8.0          # 쐐기 사이 간격
    wedge_depth: float = 5.0        # 쐐기 깊이 (전사만 보므로 얕게)
    chord_tol: float = 0.01         # 쐐기 원호 테셀레이션 공차


def _rect(x0, y0, x1, y1) -> np.ndarray:
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=float)


def wedge(tip_r: float, length: float, half_deg: float,
          chord_tol: float = 0.01) -> np.ndarray:
    """끝이 반경 `tip_r` 로 둥근 쐐기 단면 (apex 가 원점 근처, 축은 +x).

    두 측면은 원점을 지나는 반직선 `y = ±x·tan α` 이고, 그 둘에 **접하는** 반경
    `tip_r` 의 원이 끝을 막는다. 원 중심은 축 위 `d = tip_r / sin α` 에 있다.
    이렇게 하면 끼인각(= 폭 기울기)과 꼭짓점 반경이 **독립**이다.
    """
    a = np.deg2rad(float(half_deg))
    r = float(tip_r)
    c = np.array([r / np.sin(a), 0.0])                 # 원 중심
    t_hat = np.array([np.cos(a), np.sin(a)])           # 상측면 방향
    p_up = t_hat * float(np.dot(c, t_hat))             # 접점 (상)
    th_up = float(np.arctan2(p_up[1], p_up[0] - c[0]))
    th_dn = -th_up
    sweep = (2 * np.pi - (th_up - th_dn)) % (2 * np.pi)   # apex 쪽(−x)으로 도는 호
    step = 2.0 * np.arccos(max(0.0, 1.0 - chord_tol / max(r, 1e-9)))
    n_arc = max(8, int(np.ceil(sweep / max(step, 1e-6))) + 1)
    th = th_up + np.linspace(0.0, sweep, n_arc)
    arc = c + r * np.column_stack([np.cos(th), np.sin(th)])
    end_up = t_hat * (length / np.cos(a))
    return np.vstack([arc, [[end_up[0], -end_up[1]], end_up]])


# ══════════════════════════════════════════════════════════════════ 배치
def _layout(spec: CouponSpec):
    """시편들을 줄 단위로 배치하고 바깥 치수를 정한다.

    반환: (`items`, `W`, `D`) — `items` 는 제원이 붙은 (다각형, 깊이, 메타) 목록.
    좌표 원점은 **둘레 벽 바깥면**이고, 모든 시편은 벽 안쪽 `margin` 안에 있다.
    """
    items = []
    x_ends = []
    y = spec.wall_t + spec.margin

    for ri, depth in enumerate(spec.slot_depths):
        x = spec.wall_t + spec.margin
        for si, w in enumerate(spec.slot_widths):
            items.append((_rect(x, y, x + w, y + spec.slot_len), float(depth),
                          {"id": f"{'ABC'[ri]}{si + 1}", "kind": "slot",
                           "width_mm": float(w), "depth_mm": float(depth),
                           "gap_mm": float(max(w * spec.gap_factor, spec.gap_min)),
                           "slenderness": float(depth / w)}))
            x_ends.append(x + w)
            x += w + max(w * spec.gap_factor, spec.gap_min)
        y += spec.slot_len + spec.row_gap

    # W 열 — 폭 고정, **간격만** 바꾼다. 간격 g 는 슬롯 i 와 i+1 **사이** 값이다.
    x = spec.wall_t + spec.margin
    w = spec.web_slot_w
    for gi, g in enumerate(list(spec.web_gaps) + [None]):
        items.append((_rect(x, y, x + w, y + spec.slot_len), float(spec.web_depth),
                      {"id": f"W{gi + 1}", "kind": "web_slot",
                       "width_mm": float(w), "depth_mm": float(spec.web_depth),
                       "gap_to_next_mm": None if g is None else float(g)}))
        x_ends.append(x + w)
        if g is None:
            break
        x += w + g
    y += spec.slot_len + spec.row_gap

    # R 열 — 쐐기. y 는 쐐기 축(대칭선)의 높이다.
    half_h = spec.wedge_len * np.tan(np.deg2rad(spec.wedge_halfangle_deg))
    y_axis = y + half_h
    x = spec.wall_t + spec.margin
    for ri, r in enumerate(spec.tip_radii):
        poly = wedge(r, spec.wedge_len, spec.wedge_halfangle_deg, spec.chord_tol)
        items.append((poly + np.array([x, y_axis]), float(spec.wedge_depth),
                      {"id": f"R{ri + 1}", "kind": "wedge",
                       "tip_radius_mm": float(r),
                       "depth_mm": float(spec.wedge_depth),
                       "half_angle_deg": float(spec.wedge_halfangle_deg),
                       "length_mm": float(spec.wedge_len)}))
        x_ends.append(x + spec.wedge_len)
        x += spec.wedge_len + spec.wedge_gap
    y = y_axis + half_h

    W = max(x_ends) + spec.margin + spec.wall_t
    D = y + spec.margin + spec.wall_t
    return items, float(W), float(D)


def master(spec: CouponSpec = None) -> tuple:
    """빗 시편 마스터 메쉬와 제원표.

    구조는 **중첩 포켓** 하나다 — 둘레 벽 안쪽이 깊이 `wall_h` 의 포켓이고,
    시편 슬롯들이 그 포켓 바닥에서 더 파인 자식 포켓이다. 면을 직접 꿰매므로
    내부 면 없이 셸이 하나로 나온다 (`solid.pocket_solid`).
    """
    spec = spec or CouponSpec()
    items, W, D = _layout(spec)
    t_floor = max(list(spec.slot_depths) + [spec.web_depth, spec.wedge_depth]) \
        + spec.floor_extra
    thickness = t_floor + spec.wall_h

    inner = _rect(spec.wall_t, spec.wall_t, W - spec.wall_t, D - spec.wall_t)
    slots = [S.Pocket(poly, depth, name=meta["id"]) for poly, depth, meta in items]
    rim = S.Pocket(inner, spec.wall_h, children=slots, name="rim")
    mesh = S.pocket_solid(_rect(0.0, 0.0, W, D), thickness, [rim],
                          name="coupon_master")

    meta = {"footprint_mm": [W, D, thickness],
            "floor_thickness_mm": float(t_floor),
            "backing_wall_h_mm": float(spec.wall_h),
            "n_specimens": len(items),
            "spec": asdict(spec),
            "specimens": [m for _, _, m in items]}
    return mesh, meta


# ══════════════════════════════════════════════════════════════════ 출력 가능성
#: 공정별 한계 — 최소 슬롯 폭, 최소 꼭짓점 반경, 세장비 상한.
#:
#: ⚠ **슬롯은 벽이 아니라 빈틈이다**. 처음에 FDM 하한을 `2·nozzle` 로
#:    뒀는데, 그건 **최소 벽 두께**의 규칙이지 **최소 홈 폭**의 규칙이 아니다.
#:    0.5 mm 벽은 0.4 노즐로 못 뽑지만, 0.5 mm 홈은 양옆 둘레를 뽑고 사이를 안
#:    채우면 그만이라 훨씬 쉽다.
#:
#:    시편이 이걸 보여줬다 — FDM 0.4 로 뽑은 **0.5 mm 슬롯(A1·B1)에서
#:    설계 깊이 그대로의 지느러미가 나왔다.** 옛 규칙대로면 이 세 시편을 "불가"
#:    로 빼고 `t_void_min` 하한을 1.0 mm 로 보고했을 것이다 — 즉 **만들 수 있는
#:    형상을 제작 불가로 거부**하는, 바로 그 실패를 할 뻔했다.
#:
#:    그래서 `slot_k` 를 1.25 로 낮춘다 (0.4 노즐 → 0.5 mm, **실측된 점**).
#:    더 얇은 홈은 아직 시험하지 않았으므로 이 값은 "여기까지는 된다" 이지
#:    "여기가 한계다" 가 아니다.
#:
#: ⚠ **세장비 상한은 출력이 아니라 주입이 정했다.** 아래 `FILL_ASPECT_MAX` 참조.
#:    여기 남은 `aspect_max` 는 **홈이 찍히는가**(FDM: 좁고 깊은 틈의 둘레 품질,
#:    SLA: 레진 배출)만 본다.
PROCESS = {
    "fdm": {"slot_k": 1.25, "tip_k": 0.5, "aspect_max": 25.0},
    "sla": {"slot_min": 0.4, "tip_min": 0.1, "aspect_max": 25.0},
}

#: 실리콘이 좁고 깊은 홈 **바닥까지 들어가는** 한계 세장비 (깊이 ÷ 폭).
#:
#: 실측 (Mold Max 30, 25,000 cps, 진공 없이 중력 주입):
#:
#:     시편   폭    설계 깊이   찬 깊이   설계 AR   **찬 AR**
#:     A1    0.5      5          5        10        10    ok
#:     B1    0.5     10         10        20        20    ok
#:     C1    0.5     20        ~10        40        20    여기서 멈춤
#:     C2    1.0     20         20        20        20    모서리 결손
#:
#: 서로 다른 세 시편이 같은 값을 가리킨다 — C1 이 **하필 10 mm 에서** 멈춘 것은
#: 거기가 세장비 20 이기 때문이다. 끝이 평평하지 않고 곡면(메니스커스)인 것도
#: 유동 선단이 멈춘 모양이지 바닥에 닿아 잘린 모양이 아니다.
#:
#: ⚠ **이건 재료 한계가 아니라 공정 한계다.** 진공·가압·선도포로 올라간다.
#:    `t_void_min` 과 섞지 말 것 — 섞으면 장비가 생겼을 때 숫자를 다시 못 쓴다.
#: ⚠ **단일 출처는 `gre.optimize.constraints` 다**. 여기서 다시 적지
#: 않는다 — 이전 판은 이 상수가 `coupon.py` 안에서만 쓰여서 **설계 제약에
#: 연결되지 않았고**, 폭 0.5 × 깊이 20 mm 로브(실측 미충전)가 `admissibility` 를
#: 전부 통과했다. 이제 `constraints.fill_feasible` 이 같은 값을 게이트로 쓴다.
from ..optimize.constraints import (FILL_ASPECT_MAX,  # noqa: E402
                                    FILL_SEVERITY_MAX)

FILL_METHOD = "중력 주입, 진공 없음, 25,000 cps"

#: 한계의 몇 % 부터 "경계" 로 볼지. 세장비 20 에서 B1 은 찼고 C2 는 모서리가
#: 빠졌다 — 같은 세장비라도 결과가 갈리므로 한계 바로 아래는 믿지 않는다.
MARGINAL_FRACTION = 0.8


def printability(spec: CouponSpec = None, process: str = "sla",
                 nozzle: float = 0.4) -> dict:
    """마스터 **자신**이 그 치수를 뽑을 수 있는가.

    ⚠ **이게 왜 중요한가.** 시편이 재려는 건 실리콘·추진제의 한계인데, 마스터가
       먼저 못 뽑으면 "실리콘이 못 버텼다" 와 "슬롯이 애초에 안 찍혔다" 가
       구분되지 않는다. 그러면 `t_void_min` 을 **과대평가**하고, 역설계가 실제로는
       만들 수 있는 형상을 제작 불가로 거부하게 된다. 그래서 출력 전에 가른다.

    반환: `{"process", "limits", "blocked": [id...], "marginal": [id...], "notes"}`
    """
    spec = spec or CouponSpec()
    p = dict(PROCESS[process])
    if process == "fdm":
        p["slot_min"] = p.pop("slot_k") * float(nozzle)
        p["tip_min"] = p.pop("tip_k") * float(nozzle)
    items, _, _ = _layout(spec)

    blocked, marginal, notes = [], [], []
    for _, depth, m in items:
        key = m["id"]
        if m["kind"] == "wedge":
            if m["tip_radius_mm"] < p["tip_min"]:
                blocked.append(key)
                notes.append(f"{key}: 꼭짓점 ρ {m['tip_radius_mm']:.2f} mm "
                             f"< 공정 한계 {p['tip_min']:.2f} mm — "
                             f"전사 실패가 공정 탓인지 실리콘 탓인지 못 가른다")
            continue
        w = m["width_mm"]
        if w < p["slot_min"]:
            blocked.append(key)
            notes.append(f"{key}: 슬롯 폭 {w:.2f} mm < 공정 한계 "
                         f"{p['slot_min']:.2f} mm — 마스터에 홈이 안 생긴다")
        elif w < 1.5 * p["slot_min"]:
            marginal.append(key)
        if depth / w > p["aspect_max"]:
            if key not in blocked:
                marginal.append(key)
            notes.append(f"{key}: 세장비 {depth / w:.0f} > {p['aspect_max']:.0f} — "
                         f"홈 자체가 바닥까지 안 뚫릴 수 있다 (탐침으로 확인할 것)")
    if process == "fdm":
        notes.append(f"FDM 노즐 {nozzle:.2f} mm → 슬롯 하한 {p['slot_min']:.2f} mm. "
                     f"시편에서 0.4 노즐 · 0.5 mm 슬롯이 **나왔다** — 더 얇은 "
                     f"홈은 미시험이므로 이 값은 하한이 아니라 '확인된 지점' 이다")
    return {"process": process, "limits": p, "blocked": sorted(set(blocked)),
            "marginal": sorted(set(marginal) - set(blocked)), "notes": notes}


def fill_limit(spec: CouponSpec = None,
               aspect_max: float = FILL_ASPECT_MAX,
               severity_max: float = FILL_SEVERITY_MAX) -> dict:
    """실리콘이 **바닥까지 들어가는가** — 출력 가능성과 **다른 질문**이다.

    둘 다 "지느러미가 짧게 나온다" 로 똑같이 보이지만 뜻이 정반대다.

        홈이 안 뚫렸다   → 마스터를 다시 뽑아야 한다 (`printability`)
        홈에 안 들어갔다 → 진공·가압·선도포로 고친다  (여기)

    `fill_depth_mm` 은 "이 주입 방법으로 실제로 찰 깊이" 다 — 설계 깊이가 아니라
    `깊이²/폭 ≤ FILL_SEVERITY_MAX` 에서 나오는 `√(폭 × 한계)` 로 끊긴다.

    ⚠⚠ **정정: 기준이 세장비가 아니다.** 이전 판은 세장비 하나로 끊었고
        C2 (1.0 × 20 mm, 세장비 20) 를 `full=True` 로 돌려줬다 — C2 는 실제로
        안 찼다. 세장비도 깊이도 단독으로는 18 개 시편을 분리하지 못한다
        (B1 과 C2 가 같은 세장비 20, C2 와 C3 가 같은 깊이 20). `깊이²/폭` 만
        **찬 것 최대 267 / 안 찬 것 최소 400** 으로 1.50 배 여유를 두고 갈린다.
        근거는 `optimize.constraints.FILL_SEVERITY_MAX` 주석에 표로 있다.
    """
    spec = spec or CouponSpec()
    items, _, _ = _layout(spec)
    short, marginal, notes = [], [], []
    rows = []
    for _, depth, m in items:
        if m["kind"] == "wedge":
            continue
        w, ar = m["width_mm"], depth / m["width_mm"]
        sev = depth * depth / w                     # 깊이²/폭 — 판정 기준
        # 찰 깊이: severity 한계를 만족하는 최대 깊이 = √(폭 × 한계)
        reach = min(depth, (w * float(severity_max)) ** 0.5)
        full = bool(reach >= depth - 1e-9)
        edge = full and sev >= MARGINAL_FRACTION * float(severity_max)
        rows.append({"id": m["id"], "width_mm": w, "depth_mm": depth,
                     "aspect": ar, "severity": sev, "fill_depth_mm": reach,
                     "full": full, "marginal": edge})
        if not full:
            short.append(m["id"])
            notes.append(f"{m['id']}: 깊이²/폭 {sev:.0f} > {severity_max:.0f} "
                         f"(폭 {w:.1f}, 깊이 {depth:.0f} mm, 세장비 {ar:.0f}) — "
                         f"깊이 {depth:.0f} mm 중 {reach:.1f} mm 까지만 찬다")
        elif edge:
            marginal.append(m["id"])
    if marginal:
        notes.append(f"경계 {', '.join(marginal)} — 세장비가 한계 바로 아래다. "
                     f"**같은 세장비 20 에서 B1 은 멀쩡히 찼고 C2 는 모서리가 "
                     f"빠졌다**. 세장비 하나로는 다 설명되지 않으니 찼는지 꼭 볼 것")
    if short:
        notes.append("⚠ 이 시편들의 '짧게 나옴' 은 **실리콘의 한계가 아니다**. "
                     "t_void_min 을 올리는 근거로 쓰지 말 것 — 진공·가압으로 고친다")
    return {"severity_max": float(severity_max),
            "aspect_max": float(aspect_max), "method": FILL_METHOD,
            "short": short, "marginal": marginal, "rows": rows, "notes": notes}


# ══════════════════════════════════════════════════════════════════ 검산
def expected_volume(spec: CouponSpec = None) -> float:
    """해석 체적 [mm³] — 메쉬 체적과 맞아야 한다 (삼각분할 검산)."""
    spec = spec or CouponSpec()
    items, W, D = _layout(spec)
    t_floor = max(list(spec.slot_depths) + [spec.web_depth, spec.wedge_depth]) \
        + spec.floor_extra
    thickness = t_floor + spec.wall_h
    inner_a = (W - 2 * spec.wall_t) * (D - 2 * spec.wall_t)
    vol = W * D * thickness - inner_a * spec.wall_h
    for poly, depth, _ in items:
        x, y = poly[:, 0], poly[:, 1]
        a = 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
        vol -= a * depth
    return float(vol)


# ══════════════════════════════════════════════════════════════════ 출력
def export_coupon(outdir: str, spec: CouponSpec = None, step: bool = True,
                  process: str = "sla", nozzle: float = 0.4) -> dict:
    """마스터를 STL(+STEP)로 쓰고 제원표·검산·출력 가능성 결과를 남긴다."""
    spec = spec or CouponSpec()
    os.makedirs(outdir, exist_ok=True)
    mesh, meta = master(spec)
    S.write_stl(os.path.join(outdir, "coupon_master.stl"), mesh)

    ok, diag = mesh.watertight()
    v_exp = expected_volume(spec)
    meta["watertight"] = bool(ok)
    meta["watertight_diag"] = diag
    meta["volume_mm3"] = mesh.volume
    meta["volume_expected_mm3"] = v_exp
    meta["volume_err_pct"] = 100.0 * (mesh.volume - v_exp) / v_exp
    meta["faces"] = int(len(mesh.f))
    meta["files"] = ["coupon_master.stl"]
    meta["warnings"] = []
    if not ok:
        meta["warnings"].append(f"수밀 아님 — {diag}")
    if abs(meta["volume_err_pct"]) > 1e-6:
        meta["warnings"].append(
            f"체적이 해석값과 다르다 ({meta['volume_err_pct']:+.3e} %)")

    pr = printability(spec, process, nozzle)
    meta["printability"] = pr
    fl = fill_limit(spec)
    meta["fill_limit"] = fl
    if fl["short"]:
        meta["warnings"].append(
            f"주입 한계(깊이²/폭 ≤ {fl['severity_max']:.0f} mm)에 걸리는 "
            f"시편: {', '.join(fl['short'])} — 짧게 나와도 "
            f"**실리콘 탓이 아니다**")
    if pr["blocked"]:
        meta["warnings"].append(
            f"공정({pr['process']}) 한계에 걸리는 시편: {', '.join(pr['blocked'])} "
            f"— 이 시편의 실패는 **실리콘이 아니라 마스터 탓**일 수 있다")

    if step:
        made, why = _step(spec, outdir)
        meta["files"] += made
        if not made:
            meta["warnings"].append(f"STEP 미생성 — {why}")

    with open(os.path.join(outdir, "coupon_master.json"), "w",
              encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)
    with open(os.path.join(outdir, "coupon_master.txt"), "w",
              encoding="utf-8") as fh:
        fh.write(summary(meta) + "\n")
    return meta


def summary(meta: dict) -> str:
    """제원표를 벤치에서 읽을 수 있는 표로."""
    W, D, H = meta["footprint_mm"]
    L = [f"시편 마스터 — {W:.1f} × {D:.1f} × {H:.1f} mm, "
         f"바닥 {meta['floor_thickness_mm']:.1f} mm, "
         f"벽 {meta['backing_wall_h_mm']:.1f} mm", ""]
    L.append(f"  삼각형 {meta['faces']:,d} 개 · 체적 {meta['volume_mm3']:,.1f} mm³ "
             f"(해석값 대비 {meta['volume_err_pct']:+.2e} %) · "
             f"{'수밀' if meta['watertight'] else '**수밀 아님**'}")
    L.append("")
    L.append("  ID     종류      폭/반경   깊이    비고")
    for m in meta["specimens"]:
        if m["kind"] == "slot":
            note = f"세장비 {m['slenderness']:.1f}"
            key = f"{m['width_mm']:.2f} mm"
        elif m["kind"] == "web_slot":
            g = m["gap_to_next_mm"]
            note = "열 끝" if g is None else f"다음 슬롯까지 살 {g:.2f} mm"
            key = f"{m['width_mm']:.2f} mm"
        else:
            note = f"끼인각 ±{m['half_angle_deg']:.0f}°, 길이 {m['length_mm']:.0f} mm"
            key = f"ρ {m['tip_radius_mm']:.2f} mm"
        L.append(f"  {m['id']:<6} {m['kind']:<9} {key:>9} "
                 f"{m['depth_mm']:>6.1f}  {note}")
    pr = meta.get("printability")
    if pr:
        L.append("")
        L.append(f"  출력 가능성 ({pr['process']}): "
                 f"슬롯 ≥ {pr['limits']['slot_min']:.2f} mm, "
                 f"꼭짓점 ≥ {pr['limits']['tip_min']:.2f} mm, "
                 f"세장비 ≤ {pr['limits']['aspect_max']:.0f}")
        L.append(f"    불가 {', '.join(pr['blocked']) or '없음'}"
                 f"   경계 {', '.join(pr['marginal']) or '없음'}")
        L += [f"    · {n}" for n in pr["notes"]]
    fl = meta.get("fill_limit")
    if fl:
        L.append("")
        L.append(f"  주입 충전 (세장비 ≤ {fl['aspect_max']:.0f}; {fl['method']})")
        L.append(f"    덜 참 {', '.join(fl['short']) or '없음'}"
                 f"   경계 {', '.join(fl['marginal']) or '없음'}")
        L += [f"    · {n}" for n in fl["notes"]]
    if meta.get("warnings"):
        L.append("")
        L += [f"  ⚠ {w}" for w in meta["warnings"]]
    return "\n".join(L)


def _step(spec: CouponSpec, outdir: str):
    """cadquery 로 같은 형상을 B-rep 으로. 없으면 이유만 돌려준다."""
    from .step import available
    ok, why = available()
    if not ok:                                         # pragma: no cover
        return [], why
    import cadquery as cq
    items, W, D = _layout(spec)
    t_floor = max(list(spec.slot_depths) + [spec.web_depth, spec.wedge_depth]) \
        + spec.floor_extra
    thickness = t_floor + spec.wall_h
    try:
        body = cq.Workplane("XY").box(W, D, thickness,
                                      centered=(False, False, False))
        body = body.cut(cq.Workplane("XY")
                        .box(W - 2 * spec.wall_t, D - 2 * spec.wall_t, spec.wall_h,
                             centered=(False, False, False))
                        .translate((spec.wall_t, spec.wall_t, t_floor)))
        for poly, depth, _ in items:
            xy = [(float(a), float(b)) for a, b in poly]
            body = body.cut(cq.Workplane("XY").polyline(xy).close()
                            .extrude(depth).translate((0, 0, t_floor - depth)))
        pth = os.path.join(outdir, "coupon_master.step")
        cq.exporters.export(body, pth)
        return ["coupon_master.step"], ""
    except Exception as exc:                           # pragma: no cover
        return [], f"STEP 생성 실패: {exc.__class__.__name__}: {exc}"


# ══════════════════════════════════════════════════════════════════ 기록 양식
#: 칸에 적는 말 — 자유 서술이면 나중에 세는 쪽에서 추측이 들어간다
VOCAB = {
    "printed": ("ok", "partial", "not_printed"),
    "filled": ("ok", "partial", "not_filled"),
    "tear": ("none", "partial", "full"),
    "fill": ("ok", "partial", "not_filled"),
    "demold": ("intact", "cracked", "broken"),
    "drop": ("intact", "cracked", "broken"),
    "crack_24h": ("none", "hairline", "through"),
}

_T1 = ["id", "design_width_mm", "depth_mm", "slenderness", "master_width_mm",
       "printed", "silicone_width_mm", "fin_height_mm", "filled", "tear",
       "tip_deflection_mm", "reuse_cycles", "note"]
_T2 = ["id", "design_gap_mm", "grain_web_mm", "fill", "demold", "drop",
       "crack_24h", "note"]
_T3 = ["id", "design_rho_mm", "master_rho_mm", "silicone_rho_mm",
       "grain_rho_mm", "photo", "note"]


def _csv(path: str, cols: list, rows: list):
    with open(path, "w", encoding="utf-8-sig", newline="") as fh:
        fh.write(",".join(cols) + "\n")
        for r in rows:
            fh.write(",".join(str(r.get(c, "")) for c in cols) + "\n")


def record_forms(outdir: str, spec: CouponSpec = None) -> list:
    """벤치에서 칸만 채우면 되는 **빈 기록 양식** 3장 + 조건 메모.

    ⚠ **판정 칸이 없다.** 합격/불합격은 관찰에서 나중에 계산한다 — 양식에 판정
       칸을 두면 기준이 바뀌었을 때 원자료가 없어 재판정을 못 한다.

    ⚠ **"못 만든 것" 과 "실패한 것" 을 가르는 칸을 둔다** (`printed`, `filled`,
       `fill`). 안 찍힌 슬롯이나 안 찬 지느러미를 "찢어졌다" 로 적으면
       `t_void_min` 을 과대평가하고, 역설계가 실제로는 만들 수 있는 형상을
       제작 불가로 거부하게 된다.

    ID·설계값은 `CouponSpec` 에서 뽑으므로 치수를 바꾸면 양식도 따라 바뀐다.
    """
    spec = spec or CouponSpec()
    os.makedirs(outdir, exist_ok=True)
    items, _, _ = _layout(spec)
    made = []

    t1 = [{"id": m["id"], "design_width_mm": f"{m['width_mm']:.2f}",
           "depth_mm": f"{m['depth_mm']:.1f}",
           "slenderness": f"{m['slenderness']:.1f}"}
          for _, _, m in items if m["kind"] == "slot"]
    _csv(os.path.join(outdir, "T1_slots.csv"), _T1, t1)
    made.append("T1_slots.csv")

    web = [m for _, _, m in items if m["kind"] == "web_slot"]
    t2 = [{"id": f"{a['id']}-{b['id']}",
           "design_gap_mm": f"{a['gap_to_next_mm']:.2f}"}
          for a, b in zip(web[:-1], web[1:]) if a["gap_to_next_mm"] is not None]
    _csv(os.path.join(outdir, "T2_web.csv"), _T2, t2)
    made.append("T2_web.csv")

    t3 = [{"id": m["id"], "design_rho_mm": f"{m['tip_radius_mm']:.2f}"}
          for _, _, m in items if m["kind"] == "wedge"]
    _csv(os.path.join(outdir, "T3_tips.csv"), _T3, t3)
    made.append("T3_tips.csv")

    with open(os.path.join(outdir, "conditions.md"), "w", encoding="utf-8") as fh:
        fh.write(_CONDITIONS)
    made.append("conditions.md")
    with open(os.path.join(outdir, "README.md"), "w", encoding="utf-8") as fh:
        fh.write(_README.format(
            vocab="\n".join(f"| `{k}` | {' / '.join(v)} |"
                            for k, v in VOCAB.items()),
            n1=len(t1), n2=len(t2), n3=len(t3)))
    made.append("README.md")
    return made


_CONDITIONS = """# 공통 조건 기록

`r_tip_min` 의 근거는 **출력 설정**이다. 이게 없으면 나온 숫자를 다른 공정에
쓸 수 없다. 모르는 항목은 비워 두지 말고 "모름" 이라고 적을 것.

## 마스터 출력
- 프린터 / 공정 (SLA · DLP · FDM):
- 재질 (레진·필라멘트 제품명):
- 레이어 높이 [mm]:
- 노즐 지름 [mm] (FDM 만):
- 후경화 조건 (시간 · 온도 · 장비):
- 세척 (IPA 시간 · 초음파 여부):
- 출력 검수: C1(0.5 × 20 mm) 슬롯 바닥까지 비었는가 →

## 실리콘 — Smooth-On Mold Max 30 (제품 자료는 채워 둠)

| 항목 | 자료값 | 실제 |
|---|---|---|
| 경화 방식 | 축합(주석) | |
| 배합비 | 100A : 10B (중량) | A ____ g : B ____ g |
| 가사 시간 | 45 분 | 실제 ____ 분 |
| 탈형 / 완전 경화 | 16 h / 24 h @ 23 °C | ____ h @ ____ °C |
| 점도 | 25,000 cps | |
| 비중 | 1.18 g/cm³ | |
| 경도 | Shore 30A | |
| 인장 / 인열 | 577 psi / 125 pli | |
| 연신률 | 300 % | |
| 수축비 | 0.002 in/in (0.2 %) | |
| 사용 온도 | −53 ~ 205 °C | |

한 판 정량 115 g, 권장 배합 **A 120 g : B 12 g** (여유 15 %).
⚠ 자료값과 다른 제품을 썼으면 **왼쪽 열을 고칠 것** — `FitSpec.silicone_shrink_pct`
가 이 수축비에서 온다.

- 로트 번호 / 개봉일:
- 탈포: 배합 후 진공 (  분) / 주입 후 진공 (  분) / 두드림 / 없음
- 좁은 슬롯 선(先) 도포 (A1·B1·C1·A2·B2·C2):
- 후경화 65 °C × 4 h 실시 여부:
- 장갑 (⚠ 라텍스는 경화 저해 — 비닐):
- 이형제:
- 경화 저해 사전 시험 (구석 소량) 결과:
- 실온 [°C] · 습도 [%]:

## 캐스팅 (파라핀 예비 / KNSB 본시험 각각)
- 재료:
- 용융·주입 온도 [°C]:
- 압축 힘 또는 압력:
- 냉각 시간:
- 실온 [°C] · 습도 [%]:
- 탈형 순서 · 특이사항:

## 측정
- 캘리퍼스 모델 · 분해능:
- 저울 분해능:
- 카메라 · 렌즈 · 촬영 거리 · 스케일 바:
"""

_README = """# 시편 시험 기록 양식

`coupon_master.stl` 로 뽑은 판의 시편 {n1} + {n2} + {n3} 개를 기록한다.
ID 와 설계값은 채워져 있고, **실측 칸만 채우면 된다**.

## 세 가지 원칙

**1. 합격/불합격을 적지 않는다.** 판정 칸을 일부러 두지 않았다. 관찰만 적으면
판정은 계산한다. 기준(끝단 변위 10 % 등)이 바뀌어도 원자료가 있으면 다시
판정할 수 있지만, "불합격" 만 남으면 못 한다.

**2. "못 만든 것" 과 "실패한 것" 을 가른다.** 가장 중요하다.

- 마스터 슬롯이 안 찍혔다 → `printed = not_printed`
- 실리콘이 슬롯 바닥까지 안 찼다 → `filled = not_filled`
- 추진제가 살 자리에 안 들어갔다 → `fill = not_filled`
- **찢어졌다/부서졌다는 그 다음 이야기다.**

안 찍힌 슬롯의 실패를 "실리콘이 못 버텼다" 로 읽으면 `t_void_min` 을
과대평가하고, 역설계가 **실제로는 만들 수 있는 형상을 제작 불가로 거부**한다.
SLA 에서도 C1(세장비 40)은 레진 배출이 불확실하니 특히 주의.

**3. 마스터 실측을 빠뜨리지 않는다.** 설계 → 마스터 → 실리콘 → 그레인 3단
전사에서 어느 단계가 오차를 냈는지는 중간값이 있어야 분해된다.

## 쓰는 말

| 칸 | 넣을 수 있는 값 |
|---|---|
{vocab}

숫자 칸은 숫자만 (단위 쓰지 말 것). 못 쟀으면 비워 두고 `note` 에 이유.
폭·두께는 한 시편당 **3곳(양 끝 + 중앙)** 을 재서 평균을 적고, 편차가 크면
`note` 에 범위를 남긴다.

## 파일

| 파일 | 무엇 | 행 |
|---|---|---|
| `T1_slots.csv` | 실리콘 지느러미 한계 (`t_void_min`, 세장비) | {n1} |
| `T2_web.csv` | 추진제 살 파단 한계 (`t_prop_min`) — **살은 슬롯 사이다** | {n2} |
| `T3_tips.csv` | 형상 전사 (`r_tip_min`, 진폭 유의성 문턱 δ) | {n3} |
| `conditions.md` | 공통 조건 — 출력·실리콘·캐스팅·측정 | — |

## 사진

`<시편ID>_<단계>.jpg` 로 이름만 지으면 된다 (`R1_master.jpg`,
`R1_silicone.jpg`, `R1_grain.jpg`). 스케일 바를 시편과 **같은 평면**에,
같은 거리·같은 렌즈로. 캘리퍼스로 못 재는 꼭짓점 반경은 사진에서 원 피팅으로
뽑는다. `T3_tips.csv` 의 `photo` 칸에는 파일 이름만 적는다.

캘리퍼스와 사진이 어긋나면 **캘리퍼스를 믿되 차이를 기록**한다 — 그 차이가
사진 측정의 불확실도다.
"""
