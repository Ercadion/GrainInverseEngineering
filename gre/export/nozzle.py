"""노즐 — **사양표**와 (선택) 제작 CAD.

⚠ **역설계가 정하는 것은 사양뿐이다.** 최적화기가 내놓는 노즐 관련 값은 목 면적
   `A_t` 와 팽창비 `ε` 둘뿐이고, 이 둘은 **윤곽을 결정하지 않는다.** 같은 `A_t`·`ε`
   를 갖는 노즐이 수축각·확산각·목 곡률에 따라 무수히 많다. 그래서

     사양표  = 역설계 결과 그대로 (입력 불필요)
     CAD    = 사양표 + **제작자가 고르는 형상 입력**(수축각·확산각·목 곡률·외경)

   두 단계를 나눈다. 사양표만으로 시험을 돌릴 수 있고(기성 노즐을 쓰는 경우),
   직접 깎을 때만 CAD 를 뽑는다.

좌표계: 목이 `z = 0`, 유동은 +z. 단위 mm.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, field

import numpy as np

from . import solid as S

#: 문헌상 통상 범위 — 벗어나면 경고만 한다 (금지는 아니다)
CONV_HALF_DEG = (20.0, 60.0)     # 수축 반각
DIV_HALF_DEG = (10.0, 20.0)      # 확산 반각 (원뿔)
SEPARATION_PE_PA = 0.4           # Pe/Pa 가 이보다 낮으면 박리 경고


# ════════════════════════════════════════════════════════════ 사양표
@dataclass
class NozzleSpec:
    """역설계가 내놓는 값 그대로. **형상 입력은 하나도 없다.**"""

    a_throat_mm2: float              # A_t — 설계 결과 (고정 모드면 입력값)
    expansion_ratio: float           # ε = A_e / A_t
    p_c_max_mpa: float = None        # 최대 챔버압 (게이지 아님, 절대압)
    p_ambient_mpa: float = 0.101325
    gamma: float = 1.133
    c_f: float = None                # 추력계수 (설계점)
    f_max_n: float = None
    total_impulse_ns: float = None
    throat_is_fixed: bool = False    # True 면 사용 노즐을 측정해 넣은 값
    note: str = ""

    @property
    def d_throat(self) -> float:
        return 2.0 * math.sqrt(float(self.a_throat_mm2) / math.pi)

    @property
    def d_exit(self) -> float:
        return self.d_throat * math.sqrt(float(self.expansion_ratio))

    @property
    def a_exit_mm2(self) -> float:
        return float(self.a_throat_mm2) * float(self.expansion_ratio)

    def pe_over_pa(self) -> float:
        """출구압 ÷ 대기압. 1 보다 작으면 과팽창, `SEPARATION_PE_PA` 미만이면 박리 위험.

        ⚠ **설계점 한 점에서만 보는 판정이다.** `P_e/P_c` 는 ε 와 γ 가 정하므로
           `P_e` 는 `P_c` 에 비례한다. 즉 소진 구간에서 `P_c` 가 떨어지면 박리는
           **반드시** 일어나고, 그건 정상이며 추력도 작아 문제가 되지 않는다.
           여기서 묻는 것은 "**정상 작동 압력에서도** 박리하는가" 이고, 그래서
           최대(설계) 챔버압을 쓴다. 평균압으로 보고 싶으면 그 값을 넣을 것.
        """
        if self.p_c_max_mpa is None:
            return float("nan")
        from ..core.nozzle import area_ratio_to_pressure_ratio
        pe_pc = area_ratio_to_pressure_ratio(float(self.expansion_ratio),
                                             float(self.gamma))
        return float(pe_pc * self.p_c_max_mpa / self.p_ambient_mpa)

    def rows(self) -> list:
        """시험장에서 읽는 표. (항목, 값, 단위, 비고)."""
        src = "사용 노즐 실측" if self.throat_is_fixed else "설계 결과"
        out = [("목 지름 d_t", f"{self.d_throat:.3f}", "mm", src),
               ("목 면적 A_t", f"{self.a_throat_mm2:.2f}", "mm²", src),
               ("팽창비 ε", f"{self.expansion_ratio:.3f}", "—", "설계 결과"),
               ("출구 지름 d_e", f"{self.d_exit:.3f}", "mm", "d_t·√ε"),
               ("출구 면적 A_e", f"{self.a_exit_mm2:.2f}", "mm²", "A_t·ε")]
        if self.c_f is not None:
            out.append(("추력계수 C_f", f"{self.c_f:.4f}", "—", "설계점"))
        if self.p_c_max_mpa is not None:
            out.append(("최대 챔버압 P_c", f"{self.p_c_max_mpa:.3f}", "MPa", "절대압"))
            out.append(("출구압비 P_e/P_a", f"{self.pe_over_pa():.3f}", "—",
                        f"설계점 기준, {SEPARATION_PE_PA} 미만이면 박리 위험 "
                        f"(소진 구간 박리는 정상)"))
        if self.f_max_n is not None:
            out.append(("최대 추력 F_max", f"{self.f_max_n:.1f}", "N", "설계 예측"))
        if self.total_impulse_ns is not None:
            out.append(("총역적 I_t", f"{self.total_impulse_ns:.1f}", "N·s", "설계 예측"))
        out.append(("비 γ", f"{self.gamma:.4f}", "—", "가정값"))
        return out

    def warnings(self) -> list:
        w = []
        if self.p_c_max_mpa is not None:
            ratio = self.pe_over_pa()
            if ratio < SEPARATION_PE_PA:
                w.append(f"P_e/P_a = {ratio:.3f} < {SEPARATION_PE_PA} — 과팽창 박리 "
                         f"위험. ε 를 낮추거나 P_c 를 올릴 것")
        if self.throat_is_fixed:
            w.append("목 고정 모드 — 이 표는 제작 도면이 아니라 **사용 노즐의 기록**이다")
        return w


def spec_table(spec: NozzleSpec) -> str:
    rows = spec.rows()
    w0 = max(len(r[0]) for r in rows)
    lines = ["노즐 사양표", ""]
    for name, val, unit, note in rows:
        lines.append(f"  {name:<{w0}}  {val:>10} {unit:<5} {note}")
    for m in spec.warnings():
        lines += ["", f"  ⚠ {m}"]
    if spec.note:
        lines += ["", f"  비고: {spec.note}"]
    return "\n".join(lines)


# ════════════════════════════════════════════════════════════ 형상 입력
@dataclass
class NozzleGeom:
    """**제작자가 고르는** 형상 입력. 역설계는 이 값을 하나도 정해 주지 않는다."""

    conv_half_deg: float = 35.0      # 수축 반각 β
    div_half_deg: float = 14.0       # 확산 반각 α (원뿔)
    throat_arc_up: float = 1.5       # 상류 목 곡률 R_u / r_t
    throat_arc_dn: float = 0.5       # 하류 목 곡률 R_d / r_t
    d_inlet: float = None            # 입구 내경 — None 이면 d_outer − 2·wall_min
    l_inlet: float = 4.0             # 입구 직관 길이
    l_exit_lip: float = 1.5          # 출구 직관(립) 길이 — 가공 기준면
    d_outer: float = None            # 외경 — None 이면 지관에 꽂히게 자동 산출
    tube_clearance: float = 0.30     # 지관 내경 ↔ 노즐 외경 틈
    wall_min: float = 5.0            # 최소 살 두께
    n_theta: int = 240               # 회전 분할 수
    arc_pts: int = 24                # 목 곡률 호의 점 수


def _dedup(p: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    """이웃한 중복점(길이 0 인 변)을 뺀다. 닫힌 윤곽이므로 끝↔처음도 본다."""
    q = [p[0]]
    for row in p[1:]:
        if float(np.hypot(*(row - q[-1]))) > tol:
            q.append(row)
    if len(q) > 1 and float(np.hypot(*(q[0] - q[-1]))) <= tol:
        q.pop()
    return np.asarray(q, dtype=float)


def _radii(spec: NozzleSpec, geom: NozzleGeom, d_case):
    """외경·입구경을 정한다 — **지관에 꽂히는 플러그 노즐**이 기본 형태다.

    `d_case` (지관 내경) 를 주면 외경은 거기서 끼움 틈만 뺀 값이고, 입구경은
    외경에서 최소 살을 뺀 값이다. 지관 정보가 없으면 출구경 기준으로 잡는다.
    """
    r_e = spec.d_exit / 2.0
    if geom.d_outer is not None:
        r_out = float(geom.d_outer) / 2.0
    elif d_case is not None:
        r_out = (float(d_case) - geom.tube_clearance) / 2.0
    else:
        r_out = r_e + geom.wall_min
    r_in = (float(geom.d_inlet) / 2.0 if geom.d_inlet is not None
            else r_out - geom.wall_min)
    return r_out, r_in


def profile_rz(spec: NozzleSpec, geom: NozzleGeom = None,
               d_case: float = None) -> tuple:
    """노즐 **내부 유로**의 자오면 윤곽 (r, z) 과 station 사전.

    상류부터: 입구 직관 → 수축 원뿔 → 상류 목 호 → 목 → 하류 목 호 → 확산 원뿔
    → 출구 립. 두 호는 각각 원뿔에 **접하므로** 기울기가 꺾이지 않는다.
    """
    geom = geom or NozzleGeom()
    r_t = spec.d_throat / 2.0
    r_e = spec.d_exit / 2.0
    _, r_in = _radii(spec, geom, d_case)
    b = math.radians(float(geom.conv_half_deg))
    a = math.radians(float(geom.div_half_deg))
    Ru, Rd = geom.throat_arc_up * r_t, geom.throat_arc_dn * r_t

    # 상류 호: 중심 (r_t + Ru, 0), θ = 0 → β (목에서 멀어지며 −z)
    th_u = np.linspace(b, 0.0, geom.arc_pts)
    arc_u = np.column_stack([(r_t + Ru) - Ru * np.cos(th_u), -Ru * np.sin(th_u)])
    # 하류 호: 중심 (r_t + Rd, 0), θ = 0 → α (+z)
    th_d = np.linspace(0.0, a, geom.arc_pts)
    arc_d = np.column_stack([(r_t + Rd) - Rd * np.cos(th_d), Rd * np.sin(th_d)])

    r_u, z_u = float(arc_u[0, 0]), float(arc_u[0, 1])      # 수축 원뿔 접점
    r_d, z_d = float(arc_d[-1, 0]), float(arc_d[-1, 1])    # 확산 원뿔 접점
    if r_in <= r_u:
        raise ValueError(f"입구 반경 {r_in:.3f} mm 이 수축 원뿔 시작점 {r_u:.3f} mm "
                         f"보다 작다 — 목 곡률 `throat_arc_up` 을 줄이거나 "
                         f"`d_inlet` 을 키울 것")
    if r_e <= r_d:
        raise ValueError(f"출구 반경 {r_e:.3f} mm 이 확산 원뿔 시작점 {r_d:.3f} mm "
                         f"보다 작다 — `throat_arc_dn` 을 줄이거나 ε 를 키울 것")
    z_in = z_u - (r_in - r_u) / math.tan(b)
    z_ex = z_d + (r_e - r_d) / math.tan(a)

    # ⚠ 두 호는 목에서 같은 점 (r_t, 0) 을 공유한다. 그대로 이으면 길이 0 인
    #    변이 생기고, 메쉬에선 넓이 0 삼각형으로 묻히지만 B-rep(cadquery) 은
    #    "No geometry" 로 거부한다 → 중복점을 뺀다.
    inner = _dedup(np.vstack([
        [[r_in, z_in - geom.l_inlet], [r_in, z_in]],
        arc_u,
        arc_d,
        [[r_e, z_ex], [r_e, z_ex + geom.l_exit_lip]],
    ]))
    st = {"r_throat": r_t, "r_exit": r_e, "r_inlet": r_in,
          "z_inlet": z_in - geom.l_inlet, "z_exit": z_ex + geom.l_exit_lip,
          "z_conv_tangent": z_u, "z_div_tangent": z_d,
          "l_total": float(z_ex + geom.l_exit_lip - (z_in - geom.l_inlet)),
          "l_conv": float(z_u - z_in), "l_div": float(z_ex - z_d)}
    return inner, st


def meridian(spec: NozzleSpec, geom: NozzleGeom = None, d_case: float = None):
    """노즐 **몸체**(= 선반으로 깎는 실물)의 닫힌 자오면 윤곽과 제원.

    바깥은 원통 하나다 — 아마추어 노즐은 봉재를 물려 안쪽만 깎으므로 그게 실제
    제작 형상이고, 외형을 꾸미면 가공만 늘어난다.
    """
    geom = geom or NozzleGeom()
    inner, st = profile_rz(spec, geom, d_case)
    r_out, _ = _radii(spec, geom, d_case)
    r_need = max(st["r_inlet"], st["r_exit"]) + geom.wall_min
    notes = []
    if r_out < r_need:
        notes.append(f"외경 {2 * r_out:.1f} mm 는 최소 살 {geom.wall_min:.1f} mm 를 "
                     f"남기지 못한다 (필요 {2 * r_need:.1f} mm) — 출구경 "
                     f"{2 * st['r_exit']:.1f} mm 가 지관 안에 안 들어간다. ε 를 낮추거나 "
                     f"지관 밖으로 나오는 벨 형상을 따로 설계할 것")
    z0, z1 = st["z_inlet"], st["z_exit"]
    poly = _dedup(np.vstack([inner, [[r_out, z1], [r_out, z0]]]))
    st = {**st, "r_outer": float(r_out), "d_outer": float(2 * r_out),
          "wall_min_actual": float(r_out - max(st["r_inlet"], st["r_exit"]))}
    return poly, st, notes


def solid(spec: NozzleSpec, geom: NozzleGeom = None, d_case: float = None):
    """노즐 몸체 메쉬. 반환: (메쉬, 제원, 경고)."""
    geom = geom or NozzleGeom()
    poly, st, notes = meridian(spec, geom, d_case)
    mesh = S.revolve(poly, geom.n_theta, name="nozzle")
    exact = S.pappus_volume(poly)
    st = {**st, "volume_mm3": mesh.volume, "volume_exact_mm3": exact,
          "volume_err_pct": 100.0 * (mesh.volume - exact) / exact,
          "n_theta": int(geom.n_theta)}
    lo, hi = CONV_HALF_DEG
    if not lo <= geom.conv_half_deg <= hi:
        notes.append(f"수축 반각 {geom.conv_half_deg:.1f}° 가 통상 범위 "
                     f"{lo:.0f}–{hi:.0f}° 밖")
    lo, hi = DIV_HALF_DEG
    if not lo <= geom.div_half_deg <= hi:
        notes.append(f"확산 반각 {geom.div_half_deg:.1f}° 가 통상 범위 "
                     f"{lo:.0f}–{hi:.0f}° 밖 — 15° 를 넘으면 발산 손실이 커진다")
    ok, diag = mesh.watertight()
    if not ok:
        notes.append(f"수밀 아님 — {diag}")
    return mesh, st, notes


# ════════════════════════════════════════════════════════════ 출력
def export_nozzle(outdir: str, spec: NozzleSpec, geom: NozzleGeom = None,
                  d_case: float = None, cad: bool = True,
                  step: bool = True) -> dict:
    """사양표는 항상, CAD 는 `cad=True` 일 때만.

    `cad=False` 면 형상 입력 없이 사양표(CSV/TXT/JSON)만 나온다 — 기성 노즐을
    쓰거나 목 고정 모드일 때 쓰는 경로다.
    """
    os.makedirs(outdir, exist_ok=True)
    rows = spec.rows()
    meta = {"spec": asdict(spec), "table": rows,
            "warnings": list(spec.warnings()), "files": []}

    with open(os.path.join(outdir, "nozzle_spec.txt"), "w", encoding="utf-8") as fh:
        fh.write(spec_table(spec) + "\n")
    with open(os.path.join(outdir, "nozzle_spec.csv"), "w", encoding="utf-8") as fh:
        fh.write("항목,값,단위,비고\n")
        for r in rows:
            fh.write(",".join(f'"{c}"' for c in r) + "\n")
    meta["files"] += ["nozzle_spec.txt", "nozzle_spec.csv"]

    if cad:
        geom = geom or NozzleGeom()
        mesh, st, notes = solid(spec, geom, d_case)
        S.write_stl(os.path.join(outdir, "nozzle.stl"), mesh)
        meta["files"].append("nozzle.stl")
        meta["geom"] = asdict(geom)
        meta["stations"] = st
        meta["warnings"] += notes
        if step:
            made, why = _step(spec, geom, d_case, outdir)
            meta["files"] += made
            if not made:
                meta["warnings"].append(f"STEP 미생성 — {why}")

    with open(os.path.join(outdir, "nozzle.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)
    return meta


def _step(spec: NozzleSpec, geom: NozzleGeom, d_case, outdir: str):
    """cadquery 로 자오면을 회전시켜 B-rep 으로."""
    from .step import available
    ok, why = available()
    if not ok:                                   # pragma: no cover
        return [], why
    import cadquery as cq
    try:
        poly, _, _ = meridian(spec, geom, d_case)
        pts = [(float(r), float(z)) for r, z in poly]
        body = (cq.Workplane("XZ").polyline(pts).close()
                .revolve(360.0, (0, 0, 0), (0, 1, 0)))
        pth = os.path.join(outdir, "nozzle.step")
        cq.exporters.export(body, pth)
        return ["nozzle.step"], ""
    except Exception as exc:                     # pragma: no cover
        return [], f"STEP 생성 실패: {exc.__class__.__name__}: {exc}"
