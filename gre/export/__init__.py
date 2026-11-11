"""설계 → 제작 파일.

    from gre.export import export_set, FitSpec
    rep = export_set(port, d_outer=60.0, length=95.0, outdir="out/parts")
    print(rep.summary())

    from gre.export import NozzleSpec, NozzleGeom, export_nozzle
    export_nozzle("out/nozzle", NozzleSpec(a_throat_mm2=78.5, expansion_ratio=4.0),
                  NozzleGeom(conv_half_deg=35, div_half_deg=14), d_case=60.0)

    from gre.export import CouponSpec, export_coupon
    export_coupon("out/coupon", CouponSpec(), process="sla")

STL 은 의존성 없이 나오고, STEP 은 `cadquery` 가 있을 때만 나온다 (없으면
보고서에 이유가 적힌다). 모든 출력에는 **충실도 숫자**가 따라붙는다 — 내보낸
다각형이 설계 윤곽과 면적·둘레에서 몇 % 다른지, 회전체는 체적이 해석값과 몇 %
다른지. 숫자 없이 나가는 파일은 없다.

⚠ **노즐은 사양과 CAD 가 분리되어 있다.** 역설계가 정하는 것은 `A_t` 와 `ε` 뿐이고
   그 둘은 윤곽을 결정하지 않는다. 수축각·확산각·목 곡률·외경은 `NozzleGeom` 으로
   **제작자가** 넣는다 (`gre.export.nozzle` 머리말 참조).
"""
from .coupon import CouponSpec, export_coupon, fill_limit, master, \
    printability, record_forms, wedge
from .nozzle import NozzleGeom, NozzleSpec, export_nozzle, meridian, \
    profile_rz, spec_table
from .parts import ExportReport, FitSpec, core_mold, core_pin, end_plate, \
    export_set, grain, piston
from .profile import ProfileFidelity, circle, n_theta_for_tolerance, polygon, \
    sagitta
from .solid import Mesh, Pocket, pappus_volume, pocket_solid, prism, revolve, \
    union_z, write_stl

__all__ = ["export_set", "FitSpec", "ExportReport", "grain", "core_pin",
           "core_mold", "piston", "end_plate", "polygon", "circle", "sagitta",
           "n_theta_for_tolerance", "ProfileFidelity", "Mesh", "prism",
           "union_z", "write_stl", "Pocket", "pocket_solid", "revolve",
           "pappus_volume", "CouponSpec", "export_coupon", "master",
           "printability", "wedge", "record_forms", "fill_limit", "NozzleSpec", "NozzleGeom",
           "export_nozzle", "spec_table", "profile_rz", "meridian"]
