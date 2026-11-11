"""출력 파이프라인 회귀 시험 — 테셀레이션, 수밀성, 체적, 시편 마스터.

여기서 지키려는 것은 하나다: **내보낸 파일이 보고한 숫자와 같아야 한다.** 그래서
시험은 전부 "메쉬를 직접 재서 해석값과 맞춘다" 꼴이다.
"""
import json
import os

import numpy as np
import pytest

from gre.export import coupon as C
from gre.export import nozzle as NZ
from gre.export import parts as PT
from gre.export import profile as P
from gre.export import solid as S
from gre.geometry.port import FourierPort


def _poly_area(p):
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


#: 일습에 들어가는 부품 이름 — 한 곳에서만 적는다 (두 시험이 같이 본다)
PART_NAMES = {"grain", "core_pin", "core_mold", "piston", "end_plate",
              "pin_puller", "fragment_tray",
              "silicone_mold_plate", "silicone_mold_top"}


def star_port(r0=14.0, k=6, amp=4.5):
    a = np.zeros(8)
    b = np.zeros(8)
    a[k - 1] = amp
    return FourierPort(r0=r0, a=a, b=b)


# ═══════════════════════════════════════════════════════════ 삼각분할
def test_earcut_failure_on_collinear_holes_is_caught_and_repaired():
    """earcut 은 **일직선으로 늘어선 구멍들**에서 뚜껑을 닫지 못한다.

    시편 빗이 정확히 그 배치다. 예전에는 조용히 틈 있는 메쉬가 나왔다 —
    `_triangulate` 가 스스로 검사하고 CDT 로 다시 하도록 고쳤다.
    """
    outer = C._rect(0.0, 0.0, 80.0, 40.0)
    holes = [C._rect(x, 10.0, x + 0.5, 30.0) for x in (5.0, 15.0, 25.0, 35.0, 45.0)]
    rings = [S._ring_ccw(outer)] + [S._ring_ccw(h)[::-1] for h in holes]

    import mapbox_earcut as ec
    verts = np.vstack(rings)
    ends = np.cumsum([len(r) for r in rings]).astype(np.uint32)
    raw = np.asarray(ec.triangulate_float64(verts, ends)).reshape(-1, 3)
    assert not S._tri_closes(raw, rings), "earcut 이 이 배치를 닫았다면 시험이 낡았다"

    tri = S._triangulate(rings[0], rings[1:])
    assert S._tri_closes(tri, rings)
    a = np.vstack(rings)[tri]
    u, v = a[:, 1] - a[:, 0], a[:, 2] - a[:, 0]
    area = 0.5 * np.abs(u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]).sum()
    expect = 80 * 40 - 5 * 0.5 * 20
    assert area == pytest.approx(expect, rel=1e-12)


def test_prism_with_many_collinear_holes_is_watertight():
    outer = C._rect(0.0, 0.0, 80.0, 40.0)
    holes = [C._rect(x, 10.0, x + 0.5, 30.0) for x in (5.0, 15.0, 25.0, 35.0, 45.0)]
    m = S.prism(outer, 0.0, 3.0, holes=holes)
    ok, diag = m.watertight()
    assert ok, diag
    assert m.volume == pytest.approx((80 * 40 - 5 * 0.5 * 20) * 3.0, rel=1e-12)


# ═══════════════════════════════════════════════════════════ 포켓 솔리드
def test_pocket_solid_single_level():
    outer = C._rect(0.0, 0.0, 40.0, 30.0)
    p = S.Pocket(C._rect(5.0, 5.0, 15.0, 25.0), 3.0)
    m = S.pocket_solid(outer, 10.0, [p])
    ok, diag = m.watertight()
    assert ok, diag
    assert m.volume == pytest.approx(40 * 30 * 10 - 10 * 20 * 3, rel=1e-12)


def test_pocket_solid_nested_is_one_shell():
    """중첩 포켓(둘레 벽 + 그 안의 슬롯)도 셸 하나여야 한다.

    `prism` 을 z 로 쌓아 합치면 맞닿는 내부 면이 남아 슬라이서가 껍질 여러 개로
    본다. 여기서는 필요한 면만 꿰매므로 내부 면이 없다.
    """
    outer = C._rect(0.0, 0.0, 40.0, 30.0)
    slots = [S.Pocket(C._rect(x, 8.0, x + 1.0, 20.0), 4.0) for x in (8., 14., 20., 26.)]
    rim = S.Pocket(C._rect(4.0, 4.0, 36.0, 26.0), 5.0, children=slots)
    m = S.pocket_solid(outer, 20.0, [rim])
    ok, diag = m.watertight()
    assert ok, diag
    expect = 40 * 30 * 20 - 32 * 22 * 5 - 4 * (1 * 12 * 4)
    assert m.volume == pytest.approx(expect, rel=1e-12)
    # 오일러 표수 V − E + F = 2 → 껍질 하나, 구멍 없음
    e = np.vstack([m.f[:, [0, 1]], m.f[:, [1, 2]], m.f[:, [2, 0]]])
    n_e = len(np.unique(np.sort(e, axis=1), axis=0))
    n_v = len(np.unique(np.round(m.v, 9), axis=0))
    assert n_v - n_e + len(m.f) == 2


# ═══════════════════════════════════════════════════════════ 쐐기
@pytest.mark.parametrize("r", [0.2, 0.4, 0.8, 1.5])
def test_wedge_tip_radius_is_independent_of_the_wedge_angle(r):
    """꼭짓점 반경과 끼인각이 **분리**되어야 한다.

    슬롯 끝을 반원으로 막으면 ρ 가 슬롯 폭의 절반으로 묶여 `t_void_min` 과
    `r_tip_min` 이 섞인다. 쐐기는 두 측면에 접하는 원으로 끝을 막으므로, 각도를
    바꿔도 끝 반경은 그대로다.
    """
    for half in (15.0, 25.0, 40.0):
        p = C.wedge(r, 10.0, half, chord_tol=0.002)
        c = np.array([r / np.sin(np.deg2rad(half)), 0.0])
        arc = p[np.linalg.norm(p - c, axis=1) < r * 1.05]
        assert len(arc) >= 8
        rad = np.linalg.norm(arc - c, axis=1)
        assert rad.max() == pytest.approx(r, rel=1e-9)
        # 측면 기울기는 각도가 정한다
        assert p[:, 1].max() == pytest.approx(10.0 * np.tan(np.deg2rad(half)),
                                              rel=1e-9)


def test_wedge_is_a_simple_closed_polygon():
    for r in C.TIP_RADII:
        p = C.wedge(r, 10.0, 25.0, 0.01)
        assert _poly_area(p) > 0.0
        assert np.linalg.norm(p[0] - p[-1]) > 1e-6          # 끝점 중복 없음
        d = np.linalg.norm(np.diff(np.vstack([p, p[:1]]), axis=0), axis=1)
        assert d.min() > 1e-9


# ═══════════════════════════════════════════════════════════ 시편 마스터
def test_coupon_master_is_watertight_and_exact():
    m, meta = C.master()
    ok, diag = m.watertight()
    assert ok, diag
    assert m.volume == pytest.approx(C.expected_volume(), rel=1e-9)
    assert m.volume > 0.0
    ids = [s["id"] for s in meta["specimens"]]
    assert len(ids) == len(set(ids))
    assert meta["n_specimens"] == len(ids)


def test_coupon_web_row_measures_the_gap_not_the_slot():
    """살 두께는 **슬롯 사이 간격**이다 — A·B·C 열의 간격은 폭에 묶여 있어 못 쓴다."""
    _, meta = C.master()
    web = [s for s in meta["specimens"] if s["kind"] == "web_slot"]
    widths = {s["width_mm"] for s in web}
    assert len(widths) == 1, "W 열은 폭이 고정이어야 한다"
    gaps = [s["gap_to_next_mm"] for s in web if s["gap_to_next_mm"] is not None]
    assert gaps == sorted(gaps) and len(gaps) == len(C.WEB_GAPS)


def test_all_specimens_sit_inside_the_backing_wall():
    spec = C.CouponSpec()
    items, W, D = C._layout(spec)
    lo, hi = spec.wall_t, np.array([W, D]) - spec.wall_t
    for poly, _, m in items:
        assert poly.min() >= lo - 1e-9, m["id"]
        assert (poly.max(axis=0) <= hi + 1e-9).all(), m["id"]


def test_fdm_can_print_a_slot_narrower_than_it_can_print_a_wall():
    """**슬롯은 벽이 아니라 빈틈이다** — 시편이 옛 규칙을 반증했다.

    처음엔 FDM 하한을 `2·nozzle` 로 뒀고, 그 규칙은 0.5 mm 열(A1·B1·C1)을 "불가"
    로 뺐다. 그런데 FDM 0.4 로 뽑은 그 슬롯에서 **설계 깊이 그대로의 지느러미가
    나왔다**. 옛 규칙대로면 `t_void_min` 하한을 1.0 mm 로 보고했을 것이다 — 즉
    만들 수 있는 형상을 제작 불가로 거부할 뻔했다.
    """
    fdm = C.printability(process="fdm", nozzle=0.4)
    assert fdm["limits"]["slot_min"] <= 0.5 + 1e-9, "0.5 mm 슬롯은 실측으로 나왔다"
    assert fdm["blocked"] == []
    assert C.printability(process="sla")["blocked"] == []


def test_fill_limit_reproduces_all_eighteen_specimens():
    """주입 충전은 세장비도 깊이도 아니라 **`깊이²/폭`** 으로 끊긴다 — 18 개 재현.

    ⚠⚠ **정정.** 이 시험의 초판은 "세장비로 끊긴다" 고 적고
        `short == ["C1"]` 을 단언했다 — 즉 **C2 가 찼다고 단언했다.** C2 는
        실제로 왼쪽 꼭짓점이 안 차서 단면이 불완전했다. 틀린 기준을 시험이
        고정하고 있었던 것이다.

        세 후보를 다 재 보면 하나만 분리한다:

            세장비 d/w   찬 것 최대 20.0  안 찬 것 최소 20.0  ✗ (B1 / C2)
            깊이 d        찬 것 최대 20.0  안 찬 것 최소 20.0  ✗ (C3 / C2)
            깊이²/폭      찬 것 최대 267   안 찬 것 최소 400   ✔ 여유 1.50 배

    이 시험은 **18 개 전부**를 관측과 맞춘다. 한 칸이라도 어긋나면 기준이 틀렸다.
    """
    observed_short = {"C1", "C2"}        # 실측: 이 둘만 안 찼다
    fl = C.fill_limit()
    by = {r["id"]: r for r in fl["rows"]}
    assert set(fl["short"]) == observed_short, "관측과 다른 시편을 지목한다"
    for sid, row in by.items():
        want_full = sid not in observed_short
        assert row["full"] is want_full, f"{sid}: 관측 {want_full}, 판정 {row['full']}"
    # 분리 여유 — 찬 것의 최대와 안 찬 것의 최소가 한계를 사이에 둔다
    sev_full = max(r["severity"] for r in fl["rows"] if r["full"])
    sev_short = min(r["severity"] for r in fl["rows"] if not r["full"])
    assert sev_full <= C.FILL_SEVERITY_MAX < sev_short
    assert sev_short / sev_full == pytest.approx(1.5, rel=0.02)
    # C1 이 멈춘 깊이 — 실측 "10 mm 정도"
    assert 10.0 <= by["C1"]["fill_depth_mm"] <= 12.0
    assert any("실리콘의 한계가 아니다" in n for n in fl["notes"])


def test_fill_limit_is_not_a_material_limit():
    """주입 한계는 공정 한계다 — 진공·가압이 생기면 올라간다.

    `t_void_min` 과 섞으면 장비가 바뀌었을 때 숫자를 다시 못 쓴다.
    """
    better = C.fill_limit(severity_max=1000.0)
    assert better["short"] == []                 # 같은 형상이 전부 찬다
    assert C.FILL_SEVERITY_MAX == 267.0          # 실측값은 그대로 남는다
    # 세장비 상한은 **게이트가 아니라 참고값**으로만 남아 있다
    assert C.FILL_ASPECT_MAX == 20.0
    assert C.fill_limit(aspect_max=40.0)["short"] == ["C1", "C2"], \
        "세장비를 풀어도 판정이 바뀌면 안 된다 — 기준은 깊이²/폭 이다"


def test_export_coupon_writes_files_and_reports(tmp_path):
    meta = C.export_coupon(str(tmp_path), step=False)
    assert meta["watertight"] and abs(meta["volume_err_pct"]) < 1e-6
    assert (tmp_path / "coupon_master.stl").exists()
    saved = json.loads((tmp_path / "coupon_master.json").read_text(encoding="utf-8"))
    assert saved["n_specimens"] == meta["n_specimens"]
    # ⚠ 초판은 `assert "STEP" in " ".join(...) or True` 라서 **단언이 죽어
    #   있었다**. 그리고 `step=False` 로 부른 호출에는 STEP
    #   경고가 애초에 없다 — 틀린 것을 죽은 단언으로 가려 둔 셈이었다.
    #   실제로 나와야 하는 경고는 주입 한계 쪽이고, 그것을 본다.
    assert any("주입 한계" in w for w in meta["warnings"]), meta["warnings"]
    assert any("C1" in w and "C2" in w for w in meta["warnings"])


# ═══════════════════════════════════════════════════════════ 그레인 일습
def test_tessellation_meets_the_requested_chord_tolerance():
    """닫힌 식이 아니라 **재서** 고른 표본 수여야 한다 (예전 식은 18배 틀렸다)."""
    for port in (star_port(12.0, 1, 0.0), star_port(14.0, 3, 2.2),
                 star_port(14.0, 6, 7.0), star_port(14.0, 6, 9.0)):
        pts, fid = P.polygon(port, 0.02)
        assert fid.sagitta_mm <= 0.02 + 1e-9, fid.sagitta_mm
        assert abs(fid.area_err_pct) < 0.5
        assert abs(fid.perimeter_err_pct) < 0.5


def test_export_set_parts_are_watertight_and_volumes_match_the_sections(tmp_path):
    port = star_port(14.0, 6, 4.5)
    rep = PT.export_set(port, d_outer=60.0, length=95.0, outdir=str(tmp_path),
                        step=False)
    names = {p["name"] for p in rep.parts}
    assert names == PART_NAMES
    for p in rep.parts:
        assert p["watertight"], p
        assert p["volume_mm3"] > 0.0

    fit = PT.FitSpec()
    pts, _ = P.polygon(port, fit.chord_tol, None, shrink_pct=fit.shrink_pct)
    vol = {p["name"]: p["volume_mm3"] for p in rep.parts}
    # 코어 핀은 **일정 단면 기둥**이므로 체적 = 단면적 × 길이, 기계 정밀도까지.
    # 오버행은 **비대칭**이다 — 위는 피스톤을 통과해 공구까지, 아래는 자리에만.
    assert vol["core_pin"] == pytest.approx(
        _poly_area(pts) * (95.0 + fit.pin_overhang + fit.pin_overhang_bot),
        rel=1e-12)
    assert (tmp_path / "export_report.json").exists()
    assert (tmp_path / "export_report.txt").exists()


def test_exported_grain_volume_matches_the_design_section(tmp_path):
    port = star_port(14.0, 6, 4.5)
    g, fid = PT.grain(port, d_outer=60.0, length=95.0)
    pts, _ = P.polygon(port, PT.FitSpec().chord_tol)
    outer = P.circle(30.0, PT.FitSpec().chord_tol)
    assert g.volume == pytest.approx(
        (_poly_area(outer) - _poly_area(pts)) * 95.0, rel=1e-12)


# ═══════════════════════════════════════════════════════════ 회전체
def test_revolve_volume_converges_to_pappus():
    """회전체 체적의 기준은 파푸스 정리다 — 분할을 늘리면 거기로 수렴해야 한다."""
    p = np.array([[10., 0.], [20., 0.], [20., 5.], [10., 5.]])
    exact = S.pappus_volume(p)
    assert exact == pytest.approx(np.pi * (20 ** 2 - 10 ** 2) * 5.0, rel=1e-12)
    errs = []
    for m in (36, 180, 720):
        mesh = S.revolve(p, m)
        ok, diag = mesh.watertight()
        assert ok, diag
        assert mesh.volume > 0.0            # 법선이 바깥을 향한다
        errs.append(abs(mesh.volume / exact - 1.0))
    assert errs[0] > errs[1] > errs[2]
    assert errs[-1] < 2e-5


def test_revolve_is_orientation_independent_and_rejects_the_axis():
    p = np.array([[10., 0.], [20., 0.], [20., 5.], [10., 5.]])
    assert S.revolve(p, 120).volume == pytest.approx(S.revolve(p[::-1], 120).volume,
                                                     rel=1e-12)
    assert S.pappus_volume(p) == pytest.approx(S.pappus_volume(p[::-1]), rel=1e-12)
    with pytest.raises(ValueError):
        S.revolve(np.array([[0., 0.], [5., 0.], [5., 2.]]), 36)


# ═══════════════════════════════════════════════════════════ 노즐 사양
def test_nozzle_spec_is_pure_inverse_design_output():
    """사양표에는 형상 입력이 하나도 들어가지 않는다 — A_t 와 ε 만으로 전부 나온다."""
    sp = NZ.NozzleSpec(a_throat_mm2=78.53981633974483, expansion_ratio=4.0)
    assert sp.d_throat == pytest.approx(10.0, rel=1e-12)
    assert sp.d_exit == pytest.approx(20.0, rel=1e-12)
    assert sp.a_exit_mm2 == pytest.approx(4.0 * 78.53981633974483, rel=1e-12)


def test_separation_warning_fires_on_overexpansion():
    """ε 를 키우면 P_e/P_a 가 떨어진다 — 0.4 아래면 박리 경고."""
    ok = NZ.NozzleSpec(a_throat_mm2=78.54, expansion_ratio=4.0, p_c_max_mpa=4.5)
    bad = NZ.NozzleSpec(a_throat_mm2=78.54, expansion_ratio=12.0, p_c_max_mpa=2.0)
    assert ok.pe_over_pa() > bad.pe_over_pa()
    assert ok.warnings() == []
    assert any("박리" in w for w in bad.warnings())


# ═══════════════════════════════════════════════════════════ 노즐 형상
def test_throat_arcs_are_tangent_to_both_cones():
    """목 호가 원뿔에 **접해야** 기울기가 꺾이지 않는다 (가공·유동 둘 다)."""
    sp = NZ.NozzleSpec(a_throat_mm2=78.54, expansion_ratio=4.0)
    g = NZ.NozzleGeom(conv_half_deg=35.0, div_half_deg=14.0, arc_pts=400)
    inner, st = NZ.profile_rz(sp, g, d_case=60.0)
    r, z = inner[:, 0], inner[:, 1]
    assert r.min() == pytest.approx(sp.d_throat / 2.0, rel=1e-9)   # 목이 최소
    # 원뿔은 선분 하나다 — 그 선분의 기울기 dr/dz 가 tan(각) 과 맞아야 한다
    def seg_slope(z_a, z_b):
        ia = int(np.argmin(np.abs(z - z_a)))
        ib = int(np.argmin(np.abs(z - z_b)))
        assert ia != ib
        return (r[ib] - r[ia]) / (z[ib] - z[ia])

    assert seg_slope(st["z_inlet"] + g.l_inlet, st["z_conv_tangent"]) == \
        pytest.approx(-np.tan(np.deg2rad(35.0)), rel=1e-9)
    assert seg_slope(st["z_div_tangent"], st["z_exit"] - g.l_exit_lip) == \
        pytest.approx(np.tan(np.deg2rad(14.0)), rel=1e-9)
    # 접점 직전의 호 기울기도 같은 값이어야 한다
    k = int(np.argmin(np.abs(z - st["z_div_tangent"])))
    s_arc = (r[k] - r[k - 1]) / (z[k] - z[k - 1])
    assert s_arc == pytest.approx(np.tan(np.deg2rad(14.0)), rel=2e-2)


def test_meridian_has_no_zero_length_edge():
    """두 호가 목에서 같은 점을 공유한다 — 중복점을 남기면 B-rep 이 거부한다."""
    sp = NZ.NozzleSpec(a_throat_mm2=78.54, expansion_ratio=4.0)
    poly, _, _ = NZ.meridian(sp, NZ.NozzleGeom(), d_case=60.0)
    d = np.linalg.norm(np.diff(np.vstack([poly, poly[:1]]), axis=0), axis=1)
    assert d.min() > 1e-9
    assert len(np.unique(np.round(poly, 9), axis=0)) == len(poly)


def test_nozzle_solid_is_watertight_and_matches_pappus():
    sp = NZ.NozzleSpec(a_throat_mm2=78.54, expansion_ratio=4.0, p_c_max_mpa=4.5)
    mesh, st, notes = NZ.solid(sp, NZ.NozzleGeom(), d_case=60.0)
    assert mesh.watertight()[0]
    assert notes == []
    assert abs(st["volume_err_pct"]) < 0.05
    assert st["d_outer"] == pytest.approx(60.0 - NZ.NozzleGeom().tube_clearance)
    assert st["wall_min_actual"] >= NZ.NozzleGeom().wall_min - 1e-9


def test_geometry_warns_when_the_exit_cannot_fit_the_tube():
    """ε 를 키우면 출구가 지관 밖으로 나간다 — 조용히 벽을 깎아 먹으면 안 된다."""
    sp = NZ.NozzleSpec(a_throat_mm2=78.54, expansion_ratio=30.0)
    _, _, notes = NZ.solid(sp, NZ.NozzleGeom(), d_case=60.0)
    assert any("살" in n for n in notes)


def test_unusual_cone_angles_are_flagged():
    sp = NZ.NozzleSpec(a_throat_mm2=78.54, expansion_ratio=4.0)
    _, _, notes = NZ.solid(sp, NZ.NozzleGeom(conv_half_deg=10.0, div_half_deg=25.0),
                           d_case=60.0)
    assert any("수축 반각" in n for n in notes)
    assert any("확산 반각" in n for n in notes)


def test_export_nozzle_can_emit_the_spec_table_without_any_cad(tmp_path):
    """기성 노즐을 쓰거나 목 고정 모드면 형상 입력이 없다 — 그래도 표는 나와야 한다."""
    sp = NZ.NozzleSpec(a_throat_mm2=78.54, expansion_ratio=4.0, p_c_max_mpa=4.5,
                       throat_is_fixed=True)
    meta = NZ.export_nozzle(str(tmp_path), sp, cad=False)
    assert set(meta["files"]) == {"nozzle_spec.txt", "nozzle_spec.csv"}
    assert "geom" not in meta
    assert any("목 고정" in w for w in meta["warnings"])
    txt = (tmp_path / "nozzle_spec.txt").read_text(encoding="utf-8")
    assert "사용 노즐 실측" in txt


# ═══════════════════════════════════════════════════════════ 기록 양식
def test_record_forms_cover_every_specimen_and_carry_no_verdict_column(tmp_path):
    """양식에 **판정 칸이 있으면 안 된다** — 기준이 바뀌면 재판정을 못 한다."""
    made = C.record_forms(str(tmp_path))
    assert set(made) == {"T1_slots.csv", "T2_web.csv", "T3_tips.csv",
                         "conditions.md", "README.md"}
    _, meta = C.master()
    kinds = {}
    for s in meta["specimens"]:
        kinds.setdefault(s["kind"], []).append(s["id"])

    import csv as _csv
    def rows(name):
        with open(tmp_path / name, encoding="utf-8-sig", newline="") as fh:
            return list(_csv.DictReader(fh))

    t1 = rows("T1_slots.csv")
    assert [r["id"] for r in t1] == kinds["slot"]
    t3 = rows("T3_tips.csv")
    assert [r["id"] for r in t3] == kinds["wedge"]
    # 살은 슬롯 **사이** 이므로 행이 하나 적다
    t2 = rows("T2_web.csv")
    assert len(t2) == len(kinds["web_slot"]) - 1 == len(C.WEB_GAPS)
    assert [r["design_gap_mm"] for r in t2] == [f"{g:.2f}" for g in C.WEB_GAPS]

    for tbl in (t1, t2, t3):
        for col in tbl[0]:
            assert "판정" not in col and "verdict" not in col and "pass" not in col
        for r in tbl:
            assert r["note"] == ""                      # 빈 양식이다


def test_record_forms_separate_not_made_from_failed():
    """'안 찍혔다/안 찼다' 와 '찢어졌다' 를 가르는 칸이 있어야 한다."""
    assert "not_printed" in C.VOCAB["printed"]
    assert "not_filled" in C.VOCAB["filled"] and "not_filled" in C.VOCAB["fill"]
    assert "not_filled" not in C.VOCAB["tear"]


def test_record_forms_follow_the_spec(tmp_path):
    """치수를 바꾸면 양식도 따라 바뀐다 — 손으로 적은 표가 아니다."""
    spec = C.CouponSpec(slot_widths=(1.0, 2.0), slot_depths=(8.0,),
                        web_gaps=(1.0, 2.0), tip_radii=(0.5,))
    C.record_forms(str(tmp_path), spec)
    import csv as _csv
    with open(tmp_path / "T1_slots.csv", encoding="utf-8-sig", newline="") as fh:
        t1 = list(_csv.DictReader(fh))
    assert [r["id"] for r in t1] == ["A1", "A2"]
    assert [r["design_width_mm"] for r in t1] == ["1.00", "2.00"]
    assert all(r["depth_mm"] == "8.0" for r in t1)


# ═══════════════════════════════════════════════════════════ 수축 사슬
def test_shrinkage_happens_twice_and_composes_multiplicatively():
    """형상이 두 번 반전되므로 수축도 두 번이다 — 더하기가 아니라 곱하기다.

    몰드 → (실리콘 경화) → 핀 → (추진제 캐스팅) → 포트. 둘을 하나로 합쳐 두면
    T4 에서 추진제 수축을 잴 때 실리콘 몫이 섞인다.
    """
    f = PT.FitSpec()
    assert f.silicone_shrink_pct == 0.2          # Mold Max 30 자료값
    assert f.shrink_pct == 0.0                   # T4 전
    assert f.mold_shrink_pct == pytest.approx(0.2, rel=1e-12)

    f2 = PT.FitSpec(silicone_shrink_pct=0.2, shrink_pct=2.0)
    assert f2.mold_shrink_pct == pytest.approx(2.204, abs=1e-9)
    assert f2.mold_shrink_pct != pytest.approx(2.2, abs=1e-6)   # 단순 합이 아니다


def test_only_the_mold_carries_the_silicone_shrink():
    """핀·피스톤·엔드플레이트는 **만들어진 핀** 치수를 따라야 한다.

    실리콘 수축은 몰드 → 핀 단계에서만 일어나므로, 핀 자체와 거기 끼우는
    부품에 또 걸면 이중으로 보정된다.
    """
    port = star_port(14.0, 6, 4.5)
    fit = PT.FitSpec(silicone_shrink_pct=1.0, shrink_pct=0.0)
    pin, _ = PT.grain(port, 60.0, 95.0, fit)          # 그레인은 설계 치수 그대로
    design_pts, _ = P.polygon(port, fit.chord_tol)
    mold_pts, _ = P.polygon(port, fit.chord_tol, shrink_pct=fit.mold_shrink_pct)
    pin_pts, _ = P.polygon(port, fit.chord_tol, shrink_pct=fit.shrink_pct)
    # 몰드 캐비티만 커지고, 핀 목표 치수는 설계 그대로
    assert _poly_area(mold_pts) > _poly_area(design_pts)
    assert _poly_area(pin_pts) == pytest.approx(_poly_area(design_pts), rel=1e-12)


def test_export_report_says_which_shrink_is_still_unmeasured(tmp_path):
    port = star_port(14.0, 6, 4.5)
    rep = PT.export_set(port, 60.0, 95.0, str(tmp_path), step=False)
    assert any("추진제 수축 미보정" in w for w in rep.warnings)
    assert rep.fit["mold_shrink_pct"] == pytest.approx(0.2, rel=1e-12)

    rep2 = PT.export_set(port, 60.0, 95.0, str(tmp_path),
                         fit=PT.FitSpec(silicone_shrink_pct=0.0), step=False)
    assert any("실리콘 수축 0 %" in w for w in rep2.warnings)


# ══════════════════════════════════════════════════════════════════════════
# 오프셋 — 민코프스키 확장. 옛 `offset_convex` 는 오목 노치에서
# 자기교차한 다각형을 **조용히** 내보냈다. 그 다각형이 피스톤·엔드플레이트의
# 구멍이 되므로 못 쓰는 STL 이 나온다.
# ══════════════════════════════════════════════════════════════════════════

def _cross_count(p):
    """비인접 선분 쌍의 실제 교차 수."""
    import numpy as _np
    n = len(p); q = _np.roll(p, -1, axis=0); c = 0
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
                c += 1
    return c


def test_offset_never_self_intersects_however_sharp_the_notch():
    """⚠ **어떤 형상에서도 자기교차하지 않아야 한다.**

    스타 포트의 가시 끝(오목 노치)은 곡률 반경이 `R = r²/(r−r'')` 로 아주 작다
    (`r = r0−a_k`, `r'' = a_k·k²`). 간극이 그보다 커지면 옛 방식은 깨졌다 —
    실측 개시점은 `간극/|R| ≈ 1.2` 였고, 그 위에서 교차가 2~9 곳 생겼다.

    ⚠ **이것은 "작은 지관의 한계" 가 아니다.** `R` 은 **지관 지름과 무관**하고
      포트 형상만의 함수다. 설계를 키우면 `R` 도 커지는데 간극은 절대값이라
      안 변하므로, **작은 설계**에서 먼저 걸린다. 큰 지관에 작은 포트를 넣어도
      똑같다. 그래서 이 시험은 지름이 아니라 **간극/R 비**를 쓸어본다.
    """
    import numpy as np

    from gre.export import profile as P
    from gre.export.parts import FitSpec
    from gre.geometry.port import FourierPort

    fit = FitSpec()
    worst = 0.0
    for r0, amp in [(14.0, 7.0), (9.0, 4.5), (9.0, 5.0), (9.0, 6.0),
                    (9.0, 7.0), (7.0, 3.0), (20.0, 10.0), (4.5, 2.25)]:
        r_notch = r0 - amp
        R = abs(r_notch ** 2 / (r_notch - amp * 36.0))
        worst = max(worst, fit.pin_clearance / R)
        a = np.zeros(6); a[5] = amp
        pts, _ = P.polygon(FourierPort(r0, a, np.zeros(6)), fit.chord_tol,
                           None, shrink_pct=0.0)
        off = P.offset_outward(pts, fit.pin_clearance)
        assert _cross_count(off) == 0, \
            f"r0={r0}, a6={amp} (간극/R = {fit.pin_clearance/R:.2f}) 에서 자기교차"
    assert worst > 2.0, \
        f"쓸어본 범위의 최대 간극/R 이 {worst:.2f} 뿐 — 옛 방식이 깨지던 영역" \
        f"(≳1.2)을 충분히 넘겨야 이 시험이 의미가 있다"


def test_offset_keeps_the_clearance_and_contains_the_pin():
    """간극이 **실제로** 유지되고 구멍이 핀을 완전히 포함해야 한다.

    자기교차만 없으면 되는 것이 아니다 — 구멍이 핀보다 작아지면 조립이 안 되고,
    너무 커지면 그 틈으로 추진제가 밀려 올라와 측정면에 플래시를 만든다.
    """
    import numpy as np
    import pytest

    shapely = pytest.importorskip("shapely")
    from shapely.geometry import Point, Polygon

    from gre.export import profile as P
    from gre.geometry.port import FourierPort

    delta = 0.15
    for r0, amp in [(14.0, 7.0), (9.0, 6.0), (12.0, 0.0)]:
        a = np.zeros(6); a[5] = amp
        pts, _ = P.polygon(FourierPort(r0, a, np.zeros(6)),
                           P.DEFAULT_CHORD_TOL, None, shrink_pct=0.0)
        off = P.offset_outward(pts, delta)
        hole, pin = Polygon(off), Polygon(pts)
        assert hole.is_valid and hole.contains(pin), \
            f"r0={r0}, a6={amp}: 구멍이 핀을 포함하지 않는다"
        d_min = min(hole.exterior.distance(Point(*q)) for q in pts[::7])
        assert d_min == pytest.approx(delta, abs=0.01), \
            f"r0={r0}, a6={amp}: 최소 간격 {d_min:.4f} ≠ 요청 {delta}"


def test_offset_fallback_refuses_instead_of_emitting_broken_geometry():
    """`shapely` 없는 대체 경로는 **거부**해야 한다 — 조용히 깨뜨리면 안 된다.

    옛 판의 가드는 "길이 0 선분" 만 봐서 실제 교차를 놓쳤다. 대체 경로는 진짜
    교차 검사를 하고 걸리면 `ValueError` 를 낸다.
    """
    import numpy as np
    import pytest

    from gre.export import profile as P
    from gre.geometry.port import FourierPort

    a = np.zeros(6); a[5] = 6.0          # 간극/R ≈ 3.6 — 옛 방식이 깨지던 영역
    pts, _ = P.polygon(FourierPort(9.0, a, np.zeros(6)), P.DEFAULT_CHORD_TOL,
                       None, shrink_pct=0.0)
    with pytest.raises(ValueError, match="자기교차"):
        P._offset_vertex_normal(pts, 0.15)
    # 같은 입력을 새 경로는 멀쩡히 처리한다
    assert _cross_count(P.offset_outward(pts, 0.15)) == 0


# ═══════════════════════════════════════════════════ 10. 탈형 시험용 몰드 일습
#
# 여기서 지키려는 것: **단차가 체적을 정한다.** 플랜지가 지관 테두리에 닿아
# 멈추므로 추진제 길이 = 지관 길이 − 2·`end_inset` 이고, 그래서 92 % 충전으로
# 넣을 질량이 계산된다. 플랜지가 없으면(또는 테두리를 못 덮으면) 하드 스톱이
# 없고, 그러면 "눌린 깊이" 가 공정마다 달라져 체적이 재현되지 않는다.

def _area(p):
    return abs(_poly_area(np.asarray(p, dtype=float)))


def _bands_volume(bands):
    """띠 정의로부터 체적을 **손계산** — 메쉬와 독립적인 기준."""
    v = 0.0
    for outer, holes, z0, z1 in bands:
        a = _area(outer) - sum(_area(h) for h in holes if h is not None)
        v += a * (float(z1) - float(z0))
    return v


def test_stepped_plate_is_one_shell_where_stacking_leaves_interior_faces():
    """단차 판은 **셸 하나**여야 한다 — `prism` 을 쌓으면 내부에 면이 남는다.

    ⚠ `watertight()` 로는 둘이 구별되지 않는다. 그 검사는 "모든 모서리가 정확히
      두 번" 이라는 국소 검사이고, 쌓은 두 장은 꼭짓점 집합이 달라 각각 독립적으로
      닫혀 있어 **둘 다 통과한다**. 체적도 같다. 차이는 **표면적**에 있다:

          면적(쌓기) − 면적(단일 셸) = 2 × 가려진 접촉 면적

      가려진 접촉면은 두 단면이 **겹치는** 부분이다 — 아래 장의 단면(바깥 R=10,
      구멍 R=3) 에서 위 장의 블라인드 구멍(R=1) 을 뺀 것. 블라인드 구멍 자리는
      가려지지 않는다: 단일 셸에서는 그게 구멍의 **바닥면**으로 남는다.
    """
    c = lambda r: P.circle(r, 0.05)
    blind = c(1.0) + np.array([6.0, 0.0])
    bands = [(c(10.0), [c(3.0), None], 0.0, 4.0),
             (c(15.0), [c(3.0), blind], 4.0, 14.0)]
    m = S.stepped_plate(bands, name="t")
    ok, diag = m.watertight()
    assert ok, f"단차 판이 수밀이 아니다 — {diag}"
    assert m.volume == pytest.approx(_bands_volume(bands), rel=1e-12)

    stacked = S.union_z([S.prism(c(10.0), 0.0, 4.0, holes=[c(3.0)]),
                         S.prism(c(15.0), 4.0, 14.0, holes=[c(3.0), blind])])
    assert stacked.volume == pytest.approx(m.volume, rel=1e-12)
    contact = _area(c(10.0)) - _area(c(3.0)) - _area(blind)
    assert stacked.area - m.area == pytest.approx(2.0 * contact, rel=1e-9), (
        "쌓기가 접촉면을 두 번 품지 않으면 이 시험의 전제가 틀렸다")


def test_stepped_plate_shares_vertices_where_contours_are_equal():
    """경계에서 윤곽이 같으면 꼭짓점 고리를 **공유**해야 한다 (복제하면 안 깨진다)."""
    c = lambda r: P.circle(r, 0.05)
    one = S.stepped_plate([(c(10.0), [c(3.0)], 0.0, 2.0),
                           (c(10.0), [c(3.0)], 2.0, 6.0)])
    # 복제했다면 고리 두 개가 더 생긴다. 공유하면 레벨 3 개 × 고리 2 개뿐이다.
    n = len(P.circle(10.0, 0.05)) + len(P.circle(3.0, 0.05))
    assert len(one.v) == 3 * n, f"꼭짓점 {len(one.v)} ≠ 기대 {3 * n}"
    assert one.watertight()[0]


@pytest.mark.parametrize("d_outer,r0,amp", [(40.0, 10.0, 5.0),
                                            (60.0, 14.0, 4.5),
                                            (80.0, 20.0, 8.0),
                                            (40.0, 12.0, 0.0)])
def test_mold_set_is_watertight_at_any_tube_diameter(tmp_path, d_outer, r0, amp):
    """**지관 지름과 무관하게** 일습이 다 나와야 한다.

    옛 `offset_convex` 는 작은 설계에서 조용히 깨진 구멍을 냈다. 간극은 절대값이고
    노치 곡률은 설계 크기에 비례하므로, 작은 설계일수록 같은 간극이 상대적으로 크다.
    """
    pytest.importorskip("shapely")
    a = np.zeros(6); a[5] = amp
    port = FourierPort(r0, a, np.zeros(6))
    rep = PT.export_set(port, d_outer=d_outer, length=70.0,
                        outdir=str(tmp_path), fit=PT.FitSpec(), step=False)
    names = {p["name"] for p in rep.parts}
    assert names == PART_NAMES
    for p in rep.parts:
        assert p["watertight"], f"{p['name']} 이 수밀 아님 (Ø{d_outer}, a6={amp})"
        assert p["volume_mm3"] > 0.0
        assert os.path.exists(os.path.join(str(tmp_path), f"{p['name']}.stl"))


def test_flange_overhangs_the_tube_rim_so_the_stop_is_mechanical():
    """플랜지가 지관 **외경**보다 넓어야 테두리에 닿아 멈춘다 — 이게 하드 스톱이다."""
    fit = PT.FitSpec()
    for d in (40.0, 60.0, 80.0):
        r_rim = d / 2.0 + fit.tube_wall
        assert fit.flange_radius(d) > r_rim, f"Ø{d}: 플랜지가 테두리를 못 덮는다"
        assert fit.flange_radius(d) - r_rim == pytest.approx(fit.flange_margin)
        assert fit.plug_radius(d) < d / 2.0, f"Ø{d}: 플러그가 지관에 안 들어간다"
        assert d / 2.0 - fit.plug_radius(d) == pytest.approx(fit.tube_clearance)


def test_end_inset_is_an_input_that_drives_segment_gap_and_mass(tmp_path):
    """`end_inset` 이 세그먼트 간격과 충전 질량을 **선형으로** 끌고 가야 한다.

    사용자 확인: 외부 시뮬은 세그먼트 사이 4 mm 지만 공정은 양끝 각 4 mm 라
    실제 간격은 8 mm 다. 그리고 그 값은 **고정이 아니라 입력**이다.
    """
    pytest.importorskip("shapely")
    port = star_port(r0=10.0, amp=5.0)
    L, D = 70.0, 40.0
    seen = {}
    for inset in (2.0, 4.0, 6.0):
        rep = PT.export_set(port, d_outer=D, length=L, step=False,
                            outdir=str(tmp_path / f"i{inset}"),
                            fit=PT.FitSpec(end_inset=inset))
        pr = rep.propellant
        assert pr["seg_gap_mm"] == pytest.approx(2.0 * inset)
        assert pr["length_mm"] == pytest.approx(L - 2.0 * inset)
        assert pr["mass_target_g"] == pytest.approx(
            pr["volume_mm3"] * pr["rho_ideal_g_cc"] * pr["pack_frac"] / 1000.0)
        seen[inset] = pr["volume_mm3"]
    # 체적은 추진제 길이에 비례한다
    assert seen[2.0] / (L - 4.0) == pytest.approx(seen[6.0] / (L - 12.0), rel=1e-12)
    # 보고한 체적이 단면 × 길이와 맞는가 (독립 계산)
    a_cross = np.pi * (D / 2.0) ** 2 - port.area()
    assert seen[4.0] == pytest.approx(a_cross * (L - 8.0), rel=1e-12)


def test_seal_band_grips_the_pin_and_loose_band_clears_it():
    """씰 띠 구멍은 핀보다 **작고**, 헐거운 띠는 **크다** — 둘 다 법선 거리로 균일."""
    shapely = pytest.importorskip("shapely")
    from shapely.geometry import Polygon

    fit = PT.FitSpec()
    for r0, amp in [(10.0, 5.0), (14.0, 4.5), (20.0, 8.0)]:
        a = np.zeros(6); a[5] = amp
        pts, _ = P.polygon(FourierPort(r0, a, np.zeros(6)),
                           fit.chord_tol, None, shrink_pct=0.0)
        pin = Polygon(pts)
        seal, loose = Polygon(P.offset_inward(pts, fit.pin_seal)), \
            Polygon(P.offset_outward(pts, fit.pin_clearance))
        assert pin.contains(seal), f"r0={r0}: 씰 구멍이 핀보다 크다"
        assert loose.contains(pin), f"r0={r0}: 헐거운 구멍이 핀보다 작다"
        for poly, tgt in ((seal, fit.pin_seal), (loose, fit.pin_clearance)):
            d = [pin.exterior.distance(poly.exterior.interpolate(s, normalized=True))
                 for s in np.linspace(0.0, 1.0, 400, endpoint=False)]
            assert min(d) == pytest.approx(tgt, abs=0.01)
            assert max(d) == pytest.approx(tgt, abs=0.01), \
                f"r0={r0}: 오프셋이 법선 거리로 균일하지 않다"


def test_offset_inward_refuses_when_it_would_break_the_port():
    """너무 당기면 구멍이 사라지거나 쪼개진다 — 조용히 내보내지 말고 거부한다."""
    pytest.importorskip("shapely")
    a = np.zeros(6); a[5] = 5.0
    pts, _ = P.polygon(FourierPort(10.0, a, np.zeros(6)),
                       P.DEFAULT_CHORD_TOL, None, shrink_pct=0.0)
    with pytest.raises(ValueError):
        P.offset_inward(pts, 12.0)            # 포트 최소 반경 5.0 보다 훨씬 크다
    assert len(P.offset_inward(pts, 0.2)) >= 3


def test_puller_bore_narrows_monotonically_and_keys_on_the_port():
    """공구 보어가 단계마다 **좁아져야** 실리콘을 고르게 문다."""
    pytest.importorskip("shapely")
    fit = PT.FitSpec()
    port = star_port(r0=10.0, amp=5.0)
    pts, _ = P.polygon(port, fit.chord_tol, None, shrink_pct=fit.shrink_pct)
    prev = np.inf
    for d in fit.puller_steps:
        bore = P.offset_outward(pts, d) if d >= 0 else P.offset_inward(pts, -d)
        a = _area(bore)
        assert a < prev, f"오프셋 {d}: 보어가 좁아지지 않았다"
        prev = a
    assert prev < _area(pts), "마지막 띠가 핀을 물지 않는다"
    m = PT.pin_puller(port, fit)
    assert m.watertight()[0]


def test_pin_overhang_must_clear_the_piston_and_reach_the_puller(tmp_path):
    """오버행이 짧으면 **경고가 떠야** 한다 — 공구가 핀을 못 문다."""
    pytest.importorskip("shapely")
    port = star_port(r0=10.0, amp=5.0)
    fit = PT.FitSpec()
    assert fit.pin_overhang_needed() == pytest.approx(
        fit.end_inset + fit.piston_thick
        + len(fit.puller_steps) * fit.puller_step_len)
    assert fit.pin_overhang >= fit.pin_overhang_needed()

    rep = PT.export_set(port, 40.0, 70.0, str(tmp_path), step=False,
                        fit=PT.FitSpec(pin_overhang=12.0))
    assert any("핀 오버행" in w for w in rep.warnings), rep.warnings
    # 핀 길이가 비대칭 오버행을 그대로 반영한다
    fit2 = PT.FitSpec()
    pin = PT.core_pin(port, 70.0, fit2)
    b = pin.bounds()
    assert b[0][2] == pytest.approx(-fit2.pin_overhang_bot)
    assert b[1][2] == pytest.approx(70.0 + fit2.pin_overhang)


def test_index_hole_refuses_to_leave_a_wall_too_thin_to_print():
    """기준 구멍이 자리/플러그 벽을 FDM 한계 아래로 깎으면 거부한다."""
    pytest.importorskip("shapely")
    port = star_port(r0=10.0, amp=5.0)
    w = PT.index_hole_walls(port, PT.FitSpec(), 40.0)
    assert min(w) >= PT.WALL_MIN_FDM, f"기본값의 벽이 이미 얇다 {w}"
    with pytest.raises(ValueError, match="얇다"):
        PT.piston(port, 40.0, PT.FitSpec(index_hole_d=8.0))


def test_mold_parts_volumes_match_their_band_definitions():
    """각 부품의 메쉬 체적이 **띠 정의로 손계산한 값**과 기계 정밀도로 같아야 한다."""
    pytest.importorskip("shapely")
    fit, D = PT.FitSpec(), 40.0
    port = star_port(r0=10.0, amp=5.0)
    pts, _ = P.polygon(port, fit.chord_tol, None, shrink_pct=fit.shrink_pct)
    seal = P.offset_inward(pts, fit.pin_seal)
    loose = P.offset_outward(pts, fit.pin_clearance)
    plug = P.circle(fit.plug_radius(D), fit.chord_tol)
    flange = P.circle(fit.flange_radius(D), fit.chord_tol)
    idx = PT._index_hole(port, fit, D)

    z2, z3 = fit.end_inset, fit.end_inset + fit.piston_thick
    assert PT.piston(port, D, fit).volume == pytest.approx(_bands_volume([
        (plug, [seal, None], 0.0, fit.seal_len),
        (plug, [loose, None], fit.seal_len, z2),
        (flange, [loose, idx], z2, z3)]), rel=1e-12)

    z_t = fit.plate_thick + fit.end_inset
    assert PT.end_plate(port, D, fit).volume == pytest.approx(_bands_volume([
        (flange, [None, idx], 0.0, z_t - fit.seat_depth),
        (flange, [loose, idx], z_t - fit.seat_depth, fit.plate_thick),
        (plug, [loose, None], fit.plate_thick, z_t - fit.seal_len),
        (plug, [seal, None], z_t - fit.seal_len, z_t)]), rel=1e-12)


def test_watertight_merges_by_coordinate_so_stacked_shells_cannot_hide():
    """`watertight()` 는 **좌표**로 병합해서 봐야 한다 — 인덱스로 보면 쌓기를 놓친다.

    실제로 놓쳤던 자리다: `core_mold` 가 `union_z([벽, 바닥])` 이었고
    접촉면이 두 번 들어 있었는데, 인덱스 기준 검사는 **통과**시켰다 (두 껍질의
    인덱스가 서로 달라 각각 닫혀 있으므로). 쓰인 STL 을 좌표로 병합해 읽고서야
    블록 외곽 네 모서리가 네 번 쓰인 것이 드러났다.
    """
    sq = np.array([[-5., -5.], [5., -5.], [5., 5.], [-5., 5.]])
    stacked = S.union_z([S.prism(sq, 0.0, 3.0), S.prism(sq, 3.0, 8.0)])
    ok, diag = stacked.watertight()
    assert not ok, f"좌표로 병합하면 쌓기는 수밀이 아니어야 한다 — {diag}"
    # 블록 외곽 4 변 + 뚜껑 삼각분할의 대각선 1 개 = 5 개가 네 번 쓰인다
    assert "두 번 안 쓰인 모서리 5 개" in diag, diag
    # 같은 형상을 단일 셸로 만들면 통과한다
    one = S.stepped_plate([(sq, [], 0.0, 3.0), (sq, [], 3.0, 8.0)])
    assert one.watertight()[0]
    assert one.volume == pytest.approx(stacked.volume, rel=1e-12)


def test_core_mold_is_a_single_solid():
    """코어 몰드는 **출력하는 부품 중 가장 중요하다** — 단일 솔리드여야 한다."""
    pytest.importorskip("shapely")
    port = star_port(r0=10.0, amp=5.0)
    fit = PT.FitSpec()
    m = PT.core_mold(port, 70.0, fit)
    ok, diag = m.watertight()
    assert ok, f"코어 몰드가 단일 솔리드가 아니다 — {diag}"
    pts, _ = P.polygon(port, fit.chord_tol, None, shrink_pct=fit.mold_shrink_pct)
    half = float(np.linalg.norm(pts, axis=1).max()) + fit.mold_wall
    depth = 70.0 + fit.pin_overhang + fit.pin_overhang_bot
    exact = (2 * half) ** 2 * (fit.mold_base + depth) - _area(pts) * depth
    assert m.volume == pytest.approx(exact, rel=1e-12)
    # 캐비티 깊이가 핀 전체 길이와 같아야 한다 (핀이 바닥에 닿는다)
    pin = PT.core_pin(port, 70.0, fit)
    assert depth == pytest.approx(pin.bounds()[1][2] - pin.bounds()[0][2])


def test_written_stl_files_are_single_watertight_solids(tmp_path):
    """**디스크에 쓰인 것**을 다시 읽어 검사한다 — 메모리 메쉬를 믿지 않는다.

    좌표로 꼭짓점을 병합하고, 모서리 다중도·법선 방향·체적을 보고서와 맞춘다.
    """
    import struct
    pytest.importorskip("shapely")
    port = star_port(r0=10.0, amp=5.0)
    rep = PT.export_set(port, 40.0, 70.0, str(tmp_path), fit=PT.FitSpec(), step=False)
    ref = {p["name"]: p for p in rep.parts}
    for name, meta in ref.items():
        with open(tmp_path / f"{name}.stl", "rb") as fh:
            fh.read(80)
            n = struct.unpack("<I", fh.read(4))[0]
            raw = np.frombuffer(fh.read(n * 50), dtype=np.uint8)
        assert n == meta["faces"], f"{name}: 삼각형 개수가 보고와 다르다"
        rec = raw.reshape(n, 50)
        tri = rec[:, 12:48].copy().view(np.float32).reshape(n, 3, 3).astype(float)
        nrm = rec[:, 0:12].copy().view(np.float32).reshape(n, 3).astype(float)

        v = np.round(tri.reshape(-1, 3), 6)
        _, inv = np.unique(v, axis=0, return_inverse=True)
        f = np.asarray(inv).reshape(-1, 3)
        e = np.vstack([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
        _, cnt = np.unique(np.sort(e, axis=1), axis=0, return_counts=True)
        _, ucnt = np.unique(e, axis=0, return_counts=True)
        assert int((cnt != 2).sum()) == 0, f"{name}: 열린 모서리가 있다"
        assert int((ucnt != 1).sum()) == 0, f"{name}: 법선 방향이 뒤집힌 면이 있다"

        a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
        vol = float(np.einsum("ij,ij->i", np.cross(a, b), c).sum() / 6.0)
        assert vol == pytest.approx(meta["volume_mm3"], rel=1e-5), f"{name}: 체적"
        g = np.cross(b - a, c - a)
        g /= np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-20)
        assert float(np.einsum("ij,ij->i", g, nrm).min()) > 0.99, \
            f"{name}: 저장된 법선이 기하 법선과 다르다"


def test_flange_always_clears_the_rim_and_the_one_real_failure_is_caught(tmp_path):
    """플랜지는 **벽 두께와 무관하게** 늘 `flange_margin` 만큼 내민다.

    `flange_radius = d_outer/2 + tube_wall + flange_margin` 이므로 테두리 반경
    `d_outer/2 + tube_wall` 보다 항상 `flange_margin` 만큼 크다. 못 덮는 경우는
    `flange_margin ≤ 0` 뿐이고, 그것만 경고로 잡힌다.

    ⚠ **`tube_wall` 자체가 틀린 경우는 코드가 잡을 수 없다** — 입력이 틀렸다는
      신호가 기하학에 남지 않기 때문이다 (플랜지가 그 값을 더해서 커지므로).
      그래서 이 값은 실측이어야 한다: 지관 Ø40 · 벽 2.0 mm (사용자 2026-10-29).
    """
    pytest.importorskip("shapely")
    port = star_port(r0=10.0, amp=5.0)
    fit = PT.FitSpec()
    for wall in (0.5, 2.0, 6.0, 20.0):
        f = PT.FitSpec(tube_wall=wall)
        rim = 40.0 / 2.0 + wall
        assert f.flange_radius(40.0) - rim == pytest.approx(f.flange_margin)

    ok = PT.export_set(port, 40.0, 70.0, str(tmp_path / "ok"), step=False, fit=fit)
    assert not any("하드 스톱이 없다" in w for w in ok.warnings), ok.warnings
    assert ok.fit["rim_radius"] == pytest.approx(22.0)      # Ø40 · 벽 2.0 실측
    assert ok.fit["flange_overhang"] == pytest.approx(5.0)

    bad = PT.export_set(port, 40.0, 70.0, str(tmp_path / "bad"), step=False,
                        fit=PT.FitSpec(flange_margin=0.0))
    assert any("하드 스톱이 없다" in w for w in bad.warnings), bad.warnings
    assert bad.fit["flange_overhang"] == pytest.approx(0.0)


# ═══════════════════════════════════════ 11. STEP 과 STL 이 같은 부품인가
#
# 두 번 걸린 자리다. ① STEP 쪽이 같은 부품을 한 줄짜리 식으로 **다시**
# 만들고 있었고(플랜지·씰 띠가 생기자 조용히 다른 부품이 되었다), ②
# `_wire(...).workplane(offset=z)` 가 **대기 중인 와이어를 버려서** core_mold STEP
# 이 STL 보다 4.5 % 작게 나왔다. 둘 다 "체적을 맞춰 본 적이 없어서" 통과했다.

def test_step_solids_match_the_stl_volumes_exactly(tmp_path):
    """쓰인 STEP 을 다시 읽어 STL 체적과 맞춘다 — 단일 솔리드 + 기계 정밀도.

    두 경로가 `parts.*_bands` 하나를 읽으므로 **같아야 한다.** 원통도 진짜 원이
    아니라 `profile.circle` 의 다각형을 쓴다 — 진짜 원을 쓰면 보고한 충실도
    숫자와 내보낸 물건이 달라진다 (`step.py` 머리말 규칙).
    """
    pytest.importorskip("shapely")
    cq = pytest.importorskip("cadquery")
    port = star_port(r0=10.0, amp=5.0)
    rep = PT.export_set(port, 40.0, 70.0, str(tmp_path), fit=PT.FitSpec(), step=True)
    assert not any("STEP 미생성" in w for w in rep.warnings), rep.warnings

    for p in rep.parts:
        pth = tmp_path / f"{p['name']}.step"
        assert pth.exists(), f"{p['name']}.step 이 없다"
        shp = cq.importers.importStep(str(pth))
        solids = shp.solids().vals()
        assert len(solids) == 1, \
            f"{p['name']}: STEP 이 솔리드 {len(solids)} 개 — 단일 솔리드여야 한다"
        assert sum(s.Volume() for s in solids) == \
            pytest.approx(p["volume_mm3"], rel=1e-9), \
            f"{p['name']}: STEP 체적이 STL 과 다르다"


def test_wire_helper_puts_the_profile_on_the_requested_plane():
    """`_wire(pts, z)` 가 **z 평면 위에** 윤곽을 세워야 한다.

    옛 패턴 `_wire(pts).workplane(offset=z)` 는 `workplane()` 이 대기 중인
    와이어를 버리기 때문에 틀렸다. 그 결과가 조용히 작은 솔리드였다.
    """
    cq = pytest.importorskip("cadquery")
    from gre.export import step as ST

    sq = np.array([[-5., -5.], [5., -5.], [5., 5.], [-5., 5.]])
    solid = ST._wire(cq, sq, 7.0).extrude(3.0)
    bb = solid.val().BoundingBox()
    assert bb.zmin == pytest.approx(7.0), f"바닥이 z={bb.zmin}, 기대 7.0"
    assert bb.zmax == pytest.approx(10.0)
    assert solid.val().Volume() == pytest.approx(10.0 * 10.0 * 3.0, rel=1e-12)


# ════════════════════════ 12. 2분할 실리콘 주형 — 사용자 CAD 치수 그대로
#
# 사용자가 올린 `Silicone_Casting_Mold_Plate/Top` 의 **지름 사다리**를 옮긴 것이다
# (2026-10-29 지시: "원래 디자인 버리고 올려준 2개 캐드 파일 참고해서 만들어").
#
#     원본  Ø72 · Ø64 · Ø54 · Ø50 · 코어 Ø20
#     우리  Ø62 · Ø54 · Ø44 · Ø40 · 코어 별 단면
#
# 간격이 그대로라 **Ø40/Ø44 가 우리 지관의 내경/외경과 정확히 맞는다.**

def test_mold_ladder_matches_the_uploaded_cad_spacing():
    """지름 사다리가 사용자 원본과 **같은 간격**이어야 한다."""
    pytest.importorskip("shapely")
    port = star_port(r0=10.0, amp=5.0)
    fit = PT.FitSpec(tube_wall=2.0)
    rc, r1, r2, r3, r4 = PT.mold_radii(port, fit, 40.0)
    assert [2 * r for r in (r1, r2, r3, r4)] == pytest.approx([40.0, 44.0, 54.0, 62.0])
    # 원본 간격: 72−64 = 8, 64−54 = 10, 54−50 = 4
    assert 2 * (r4 - r3) == pytest.approx(8.0)
    assert 2 * (r3 - r2) == pytest.approx(10.0)
    assert 2 * (r2 - r1) == pytest.approx(4.0)
    # 지관에서 나온다 — 지관을 바꾸면 사다리가 따라간다
    for d, w in [(40.0, 2.0), (60.0, 2.0), (54.0, 5.0)]:
        _, a1, a2, _, _ = PT.mold_radii(star_port(r0=d / 5.0, amp=d / 10.0),
                                        PT.FitSpec(tube_wall=w), d)
        assert 2 * a1 == pytest.approx(d)
        assert 2 * a2 == pytest.approx(d + 2 * w)


def test_mold_depths_are_measured_from_the_top_face():
    """홈 15 · 보어 9 · 링 4 — 전부 **위 면에서** 잰다. 순서가 어긋나면 거부."""
    fit = PT.FitSpec()
    h = PT.mold_height(70.0, fit)
    z_g, z_b, z_r = PT._mold_levels(fit, h)
    assert h - z_g == pytest.approx(fit.mold_groove_depth)
    assert h - z_b == pytest.approx(fit.mold_bore_depth)
    assert h - z_r == pytest.approx(fit.mold_ring_recess)
    assert z_g < z_b < z_r < h
    with pytest.raises(ValueError, match="깊이 순서"):
        PT._mold_levels(PT.FitSpec(mold_ring_recess=20.0), 53.0)


def test_standing_ring_is_the_tube_grip():
    """Ø40~Ø44 **서 있는 2 mm 링**이 실제로 서 있어야 한다 (지관을 안쪽에서 잡는다).

    체적으로 본다: 링이 없는 판(= 그 구간을 보어로 뚫은 것)과의 차이가 정확히
    링 부피여야 한다.
    """
    pytest.importorskip("shapely")
    port = star_port(r0=10.0, amp=5.0)
    fit = PT.FitSpec(tube_wall=2.0)
    _, r1, r2, r3, _ = PT.mold_radii(port, fit, 40.0)
    bands = PT.silicone_mold_plate_bands(port, 40.0, 70.0, fit)
    with_ring = S.stepped_plate(bands, name="a")
    # 링 구간(보어 바닥 → 링 윗면)을 Ø54 까지 통째로 비우면 링이 사라진다
    z_g, z_b, z_r = PT._mold_levels(fit, PT.mold_height(70.0, fit))
    tol = fit.chord_tol_mold
    noring = [(o, [P.circle(r3, tol)], z0, z1, [None])
              if (abs(z0 - z_b) < 1e-9 and abs(z1 - z_r) < 1e-9)
              else (o, h, z0, z1, g)
              for o, h, z0, z1, g in bands]
    assert with_ring.volume - S.stepped_plate(noring).volume == pytest.approx(
        (_area(P.circle(r2, tol)) - _area(P.circle(r1, tol))) * (z_r - z_b),
        rel=1e-9)
    assert with_ring.watertight()[0]


def test_top_core_is_a_raised_post_not_a_hole():
    """위 뚜껑의 코어는 **솟은 기둥**이어야 한다 (사용자 2026-10-29).

    원본 Top 의 Ø20 포스트(테보다 5 돌출)를 별 단면으로 바꾼 것이다. 코어가
    기둥이면 아래 면에서 그 둘레는 **환형 빈 자리**다 — 단순 구멍이 아니다.
    """
    pytest.importorskip("shapely")
    port = star_port(r0=10.0, amp=5.0)
    fit = PT.FitSpec(tube_wall=2.0)
    tb = PT.silicone_mold_top_bands(port, 40.0, fit)
    h_rim = fit.mold_groove_depth + fit.mold_top_back
    # 맨 위 띠는 **기둥만** — 바깥 윤곽이 별 단면이고 구멍이 없다
    want, _ = P.polygon(port, fit.chord_tol, None, shrink_pct=fit.mold_shrink_pct)
    o, hs, z0, z1, gs = tb[-1]
    assert np.allclose(o, want), "맨 위 띠의 바깥이 별 단면이 아니다"
    assert all(h is None for h in hs) and all(g is None for g in gs)
    assert z0 == pytest.approx(h_rim)
    assert z1 - z0 == pytest.approx(fit.mold_top_post)
    # 어느 띠에도 코어가 **구멍**으로 들어가면 안 된다
    for o, hs, *_ in tb[:-1]:
        for h in hs:
            assert h is None or not (len(h) == len(want) and np.allclose(h, want)), \
                "코어가 아직 구멍이다"
    # 바닥은 막혀 있고(아래 면이 평면), 전체 높이 = 테 + 기둥
    assert tb[0][1] == [None] * len(tb[0][1])
    m = PT.silicone_mold_top(port, 40.0, fit)
    ok, diag = m.watertight()
    assert ok, diag
    b = m.bounds()
    assert b[1][2] - b[0][2] == pytest.approx(h_rim + fit.mold_top_post)
    # Plate 는 반대로 코어가 **공동**이고 바닥이 막혀 있다
    pb = PT.silicone_mold_plate_bands(port, 40.0, 70.0, fit)
    assert pb[0][1][0] is None and pb[0][3] == pytest.approx(fit.mold_floor)
    assert any(h is not None and len(h) == len(want) and np.allclose(h, want)
               for _, hs, *_ in pb for h in hs)
    with pytest.raises(ValueError, match="기둥이 솟지 않는다"):
        PT.silicone_mold_top_bands(port, 40.0, PT.FitSpec(mold_top_post=0.0))

def test_mold_height_follows_the_grain_length():
    """마스터 높이는 **상수가 아니라 그레인 길이 + 10** 이다 (사용자 2026-10-30).

    초판은 `mold_h = 53.0` 상수였고, 74 mm 그레인에도 53 mm 마스터가 조용히
    나왔다. 높이가 길이를 따라가는지, 그리고 길어진 만큼이 **전부 별 공동으로**
    가는지(위쪽 홈·보어·링은 위 면 기준이라 그대로) 둘 다 건다.
    """
    pytest.importorskip("shapely")
    port = star_port(r0=10.0, amp=5.0)
    fit = PT.FitSpec()
    assert PT.mold_height(70.0, fit) == pytest.approx(80.0)
    assert PT.mold_height(74.0, fit) == pytest.approx(84.0)
    prev_cav = None
    for L in (40.0, 70.0, 74.0, 120.0):
        bands = PT.silicone_mold_plate_bands(port, 40.0, L, fit)
        h = bands[-1][3]
        assert h == pytest.approx(L + fit.mold_head), f"길이 {L} 에서 높이가 안 따라간다"
        # 별 공동 = 두 번째·세 번째 띠. 바닥 → 홈 바닥까지가 공동 깊이다
        # 별 구멍은 band 1·2 **둘 다**에 있다 — 공동은 바닥 → **보어 바닥**
        assert bands[1][1][0] is not None and bands[2][1][0] is not None
        cav = bands[2][3] - bands[0][3]
        assert cav == pytest.approx(L + fit.mold_head - fit.mold_bore_depth
                                    - fit.mold_floor)
        if prev_cav is not None:
            assert cav > prev_cav, "길이를 늘렸는데 공동이 안 깊어진다"
        prev_cav = cav
        # 위쪽 세 단의 깊이는 길이와 무관하다 (위 면 기준).
        # 띠 경계: band1 끝 = 홈 바닥, band2 끝 = 보어 바닥, band3 끝 = 링 윗면
        assert h - bands[1][3] == pytest.approx(fit.mold_groove_depth)
        assert h - bands[2][3] == pytest.approx(fit.mold_bore_depth)
        assert h - bands[3][3] == pytest.approx(fit.mold_ring_recess)
    with pytest.raises(ValueError, match="너무 낮다"):
        PT.mold_height(5.0, fit)


def test_star_cavity_covers_the_propellant_length():
    """별 공동이 **추진제보다 길다** — 코어가 추진제를 덮어야 한다.

        공동   = (length + mold_head) − bore_depth − floor = length − 2
        추진제 = length − 2·end_inset                      = length − 8

    ⚠ 이 시험의 초판은 **둘이 같다**고 걸었고 38 ≠ 32 로 실패했다. 별 구멍이
      홈 띠(band 2)까지 이어지는 것을 안 보고 홈 바닥까지로 셌기 때문이다.
      주석에도 같은 틀린 주장이 적혀 있었다 — 시험이 그것을 잡았다.
    """
    pytest.importorskip("shapely")
    port = star_port(r0=10.0, amp=5.0)
    fit = PT.FitSpec()
    for L in (40.0, 70.0, 74.0):
        bands = PT.silicone_mold_plate_bands(port, 40.0, L, fit)
        cav = bands[2][3] - bands[0][3]
        prop = L - 2.0 * fit.end_inset
        assert cav == pytest.approx(L - 2.0)
        assert cav > prop, "공동이 추진제보다 짧다 — 코어가 모자란다"
        assert cav - prop == pytest.approx(
            2.0 * fit.end_inset - fit.mold_bore_depth - fit.mold_floor
            + fit.mold_head)


def test_top_post_rises_above_the_outer_wall():
    """Top 기둥은 **가장 바깥쪽 벽보다 항상 `mold_top_post` 만큼** 솟는다.

    (사용자 2026-10-30 "가장 바깥쪽 벽의 높이보다 5mm 더 높게 항상")
    바깥 벽은 Ø62 원통이고 높이가 `groove_depth + top_back` 이다. Plate 와
    달리 그레인 길이를 따라가지 않는다 — 뚜껑이라 무관하다.
    """
    pytest.importorskip("shapely")
    port = star_port(r0=10.0, amp=5.0)
    for post in (5.0, 8.0):
        fit = PT.FitSpec(mold_top_post=post)
        tb = PT.silicone_mold_top_bands(port, 40.0, fit)
        wall_top = max(z1 for _, _, _, z1, _ in tb[:-1])   # 바깥 벽 윗면
        assert tb[-1][3] - wall_top == pytest.approx(post)
        assert wall_top == pytest.approx(fit.mold_groove_depth + fit.mold_top_back)


def test_tube_wall_is_fixed_at_2mm():
    """지관 벽 두께는 **2.0 고정** (사용자 2026-10-30).

    홈 치수가 전부 여기서 파생되므로 기본값이 움직이면 몰드가 통째로 달라진다.
    ⚠ 틀린 값을 넣어도 코드는 못 잡는다 — 그래서 기본값을 시험으로 못박는다.
    """
    assert PT.FitSpec().tube_wall == pytest.approx(2.0)
    port = star_port(r0=10.0, amp=5.0)
    _, r1, r2, _, _ = PT.mold_radii(port, PT.FitSpec(), 40.0)
    assert r2 - r1 == pytest.approx(2.0)


def test_stepped_plate_handles_a_ring_swallowed_by_the_hole_above():
    """**서 있는 링이 위 구멍에 삼켜지는 경계** — 드러나는 면은 링 윗면 하나뿐.

    홈의 천장도 구멍 단차의 바닥도 **둘 다 틀리다**. 사용자 원본의 구조이고,
    여기서 틀리면 Plate·Top 이 둘 다 깨진다.
    """
    c = lambda r: P.circle(r, 0.05)
    def hand(bands):
        t = 0.0
        for o, hs, z0, z1, gs in bands:
            a = _area(o)
            a -= sum(_area(h) for h in hs if h is not None)
            a -= sum(_area(g[0]) - _area(g[1]) for g in gs if g is not None)
            t += a * (z1 - z0)
        return t
    for nm, bands in [
        ("Plate 구조", [(c(31.), [None], 0., 3., [None]),
                        (c(31.), [c(15.)], 3., 38., [None]),
                        (c(31.), [c(15.)], 38., 44., [(c(27.), c(22.))]),
                        (c(31.), [c(20.)], 44., 49., [(c(27.), c(22.))]),
                        (c(31.), [c(27.)], 49., 53., [None])]),
        ("Top 구조(기둥)", [(c(31.), [None], 0., 3., [None, None]),
                            (c(31.), [None], 3., 9., [(c(27.), c(22.)), None]),
                            (c(31.), [None], 9., 14.,
                             [(c(27.), c(22.)), (c(20.), c(15.))]),
                            (c(31.), [None], 14., 18., [(c(27.), c(15.)), None]),
                            (c(15.), [None], 18., 23., [None, None])]),
    ]:
        m = S.stepped_plate(bands, name="t")
        ok, diag = m.watertight()
        assert ok, f"{nm}: {diag}"
        assert m.volume == pytest.approx(hand(bands), rel=1e-12), nm


def test_stepped_plate_supports_annular_grooves():
    """`stepped_plate` 의 환형 홈 — 섬 안에 또 구멍이 있어도 수밀이어야 한다.

    그 경우 단면이 **두 덩어리로 갈라진다** (섬 안쪽 고리 + 바깥 고리). 한 번의
    삼각분할로는 못 덮으므로 덩어리별로 뚜껑을 나눠 덮는다.
    """
    c = lambda r: P.circle(r, 0.05)
    cases = [
        ("섬 안에 구멍", [(c(30.), [None], 0., 5., [None]),
                          (c(30.), [c(8.)], 5., 20., [None]),
                          (c(30.), [c(8.)], 20., 28., [(c(22.), c(19.))])],
         28, [(8., 23.)], 8.),
        ("위로 열린 홈", [(c(30.), [None], 0., 5., [None]),
                          (c(30.), [None], 5., 13., [(c(22.), c(19.))])],
         13, [], 8.),
        ("안에 갇힌 홈", [(c(30.), [None], 0., 5., [None]),
                          (c(30.), [None], 5., 12., [(c(22.), c(19.))]),
                          (c(30.), [None], 12., 18., [None])],
         18, [], 7.),
    ]
    ring = _area(c(22.)) - _area(c(19.))
    for nm, bands, h, holes, g_h in cases:
        m = S.stepped_plate(bands, name="t")
        ok, diag = m.watertight()
        assert ok, f"{nm}: {diag}"
        exact = _area(c(30.)) * h - sum(_area(c(r)) * d for r, d in holes) \
            - ring * g_h
        assert m.volume == pytest.approx(exact, rel=1e-12), nm


def test_step_path_refuses_to_leave_a_stale_file_behind(tmp_path):
    """STEP 생성이 실패하면 **낡은 .step 을 남기면 안 된다.**

    실제로 그 상태가 나왔다 — 환형 홈 띠를 못 읽어 STEP 이 실패했는데
    옛 파일이 그대로 있어서 "신선한 STL + 낡은 STEP" 이 됐다.
    """
    pytest.importorskip("shapely")
    pytest.importorskip("cadquery")
    from gre.export import step as ST

    port = star_port(r0=10.0, amp=5.0)
    PT.export_set(port, 40.0, 26.0, str(tmp_path), fit=PT.FitSpec(), step=True)
    assert (tmp_path / "silicone_mold_plate.step").exists()

    # 생성을 깨뜨린다 → 옛 파일이 치워져야 한다
    orig = ST.solid_from_bands
    try:
        ST.solid_from_bands = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x"))
        rep = PT.export_set(port, 40.0, 26.0, str(tmp_path), fit=PT.FitSpec(), step=True)
    finally:
        ST.solid_from_bands = orig
    assert any("STEP" in w for w in rep.warnings), rep.warnings
    assert not list(tmp_path.glob("*.step")), "낡은 STEP 이 남았다"
