"""STEP (AP214) 출력 — **선택 기능**. `cadquery` 가 있을 때만 동작한다.

왜 선택인가. STEP 은 B-rep 이라 CAD 로 받아 수정·치수 기입이 되는 반면,
`cadquery`/`OCP` 는 수백 MB 짜리 의존성이다. 핵심 경로(STL → 3D 프린팅)는 numpy
만으로 돌아야 하므로 STEP 은 있으면 쓰고 없으면 이유를 적고 넘어간다.

⚠ **여기서도 윤곽은 다각형이다.** `gre.export.profile` 이 고른 표본점을 폴리라인
   으로 넣는다. 스플라인으로 맞추면 파일은 예뻐지지만 **우리가 재서 보고한 충실도
   숫자와 형상이 달라진다.** 내보낸 물건과 보고한 숫자가 같아야 하므로 폴리라인을
   쓴다 (`export_report.json` 의 `fidelity` 가 그 물건의 숫자다).
"""
from __future__ import annotations

import os

import numpy as np

from ..geometry.port import FourierPort
from . import profile as P


def available() -> tuple:
    """(쓸 수 있는가, 이유)."""
    try:
        import cadquery  # noqa: F401
    except Exception as exc:                       # pragma: no cover
        return False, f"cadquery 를 불러올 수 없다 ({exc.__class__.__name__}). " \
                      f"`pip install cadquery` 로 설치하면 STEP 도 나온다"
    return True, ""


def _wire(cq, pts, z: float = 0.0):
    """(N,2) 점들을 **z 평면 위의** 닫힌 폴리라인 와이어로.

    ⚠⚠ **`_wire(...).workplane(offset=z)` 로 쓰면 안 된다.**
      `Workplane.workplane()` 은 **새 작업평면을 만들면서 대기 중인 와이어를
      버린다.** 그래서 그 패턴으로 뽑은 솔리드는 조용히 엉뚱한 것이 되었다 —
      `core_mold` STEP 이 STL 보다 4.5 % 작게 나온 원인이다. 평면은 **처음부터**
      원하는 높이에 세운다.
    """
    xy = [(float(x), float(y)) for x, y in np.asarray(pts, dtype=float)]
    return cq.Workplane("XY", origin=(0.0, 0.0, float(z))).polyline(xy).close()


def solid_from_bands(cq, bands):
    """`solid.stepped_plate` 와 **같은 띠 정의**로 B-rep 솔리드를 만든다.

    ⚠⚠ **이게 핵심이다.** 종전에는 STEP 쪽이 같은 부품을 한 줄짜리 식으로 **다시**
      만들었다. 그러니 STL 쪽에 플랜지 단차·씰 띠·기준 구멍이 생겼을 때 STEP 은
      조용히 **다른 부품**이 되었다. 그때는 STEP 에서 그 부품들을
      빼는 것으로 막았는데, 그건 증상을 가린 것이다 — 진짜 고침은 **정의를 하나로
      두고 둘이 같이 읽는 것**이고, 이 함수가 그 한쪽이다.

    띠마다 `바깥 − 구멍들` 을 z 로 뽑아 합집합한다. B-rep 합집합은 맞닿는 내부
    면을 **지우므로** 결과가 진짜 단일 솔리드다 (메쉬 쪽은 `stepped_plate` 가
    면을 직접 꿰매 같은 것을 얻는다).
    """
    out = None
    for band_spec in bands:
        outer, holes, z0, z1 = band_spec[:4]
        grooves = list(band_spec[4]) if len(band_spec) > 4 else []
        h = float(z1) - float(z0)
        band = _wire(cq, outer, z0).extrude(h)
        for hole in holes:
            if hole is None:
                continue
            # 구멍은 띠보다 위아래로 조금 길게 잘라 공면(coplanar) 불리언을 피한다
            band = band.cut(_wire(cq, hole, z0 - 0.1).extrude(h + 0.2))
        for g in grooves:                      # 환형 홈 = 바깥 − 안쪽
            if g is None:
                continue
            ring = (_wire(cq, g[0], z0 - 0.1).extrude(h + 0.2)
                    .cut(_wire(cq, g[1], z0 - 0.2).extrude(h + 0.4)))
            band = band.cut(ring)
        out = band if out is None else out.union(band)
    return out


def export_step(port: FourierPort, d_outer: float, length: float, outdir: str,
                fit=None, n_theta: int = None):
    """부품들을 STEP 으로 쓴다. 반환: (쓴 파일 목록, 이유). 실패해도 예외는 안 낸다."""
    ok, why = available()
    if not ok:
        return [], why
    import cadquery as cq
    from . import parts as PT
    fit = fit or PT.FitSpec()
    os.makedirs(outdir, exist_ok=True)
    # ⚠ **먼저 지운다.** 생성이 중간에 실패하면 옛 STEP 이 새 STL 옆에 남아
    #   "신선한 STL + 낡은 STEP" 이 된다 — 실제로 그 상태가 나왔다
    #   (환형 홈 띠를 못 읽어 실패했는데 옛 파일이 그대로 있었다).
    for f in os.listdir(outdir):
        if f.endswith(".step"):
            os.remove(os.path.join(outdir, f))

    port_shrunk, _ = P.polygon(port, fit.chord_tol, n_theta,
                               shrink_pct=fit.shrink_pct)
    pin_len = length + fit.pin_overhang + fit.pin_overhang_bot
    grain_b, _ = PT.grain_bands(port, d_outer, length, fit, n_theta)

    made = []
    try:
        # ⚠ **전부 `parts.*_bands` 를 읽는다.** 여기서 형상을 다시 적지 않는다 —
        #   그게 STEP 이 STL 과 갈라진 원인이었다. 원통도 `profile.circle` 의
        #   다각형을 쓴다: 진짜 원을 쓰면 파일은 예뻐지지만 보고한 충실도 숫자와
        #   물건이 달라진다 (모듈 머리말 규칙).
        jobs = {
            "grain": solid_from_bands(cq, grain_b),
            # 핀만 단일 기둥이라 직접 뽑는다 (띠 하나와 같다)
            "core_pin": _wire(cq, port_shrunk,
                              -float(fit.pin_overhang_bot)).extrude(pin_len),
            "core_mold": solid_from_bands(
                cq, PT.core_mold_bands(port, length, fit, n_theta)),
            "piston": solid_from_bands(
                cq, PT.piston_bands(port, d_outer, fit, n_theta)),
            "end_plate": solid_from_bands(
                cq, PT.end_plate_bands(port, d_outer, fit, n_theta)),
            "pin_puller": solid_from_bands(
                cq, PT.pin_puller_bands(port, fit, n_theta)),
            "fragment_tray": solid_from_bands(cq, PT.fragment_tray_bands(fit)),
            "silicone_mold_plate": solid_from_bands(
                cq, PT.silicone_mold_plate_bands(port, d_outer, length, fit,
                                                 n_theta)),
            "silicone_mold_top": solid_from_bands(
                cq, PT.silicone_mold_top_bands(port, d_outer, fit, n_theta)),
        }
        for name, solid in jobs.items():
            pth = os.path.join(outdir, f"{name}.step")
            cq.exporters.export(solid, pth)
            made.append(os.path.basename(pth))
    except Exception as exc:                       # pragma: no cover
        return made, (f"STEP 생성 중 실패: {exc.__class__.__name__}: {exc}. "
                      f"STL 은 그대로 쓰였다")
    return made, ""


def export_step_solids(solids: dict, outdir: str):
    """이름 → cadquery 솔리드 사전을 그대로 STEP 으로. 시편 몰드처럼 전용 형상용."""
    ok, why = available()
    if not ok:
        return [], why
    import cadquery as cq
    os.makedirs(outdir, exist_ok=True)
    made = []
    try:
        for name, solid in solids.items():
            pth = os.path.join(outdir, f"{name}.step")
            cq.exporters.export(solid, pth)
            made.append(os.path.basename(pth))
    except Exception as exc:                       # pragma: no cover
        return made, f"STEP 생성 중 실패: {exc.__class__.__name__}: {exc}"
    return made, ""
