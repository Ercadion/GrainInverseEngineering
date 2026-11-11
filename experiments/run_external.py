#!/usr/bin/env python3
"""외부 시뮬레이터를 **그대로 돌려** 시계열을 받아 적는다 (대조용).

⚠ `6DOF RocketSim with Unreal` 는 **참조 전용**이다. 이 스크립트는 그 폴더를
  sys.path 에 얹어 import 만 하고, 아무것도 쓰지 않는다. 바이트코드도 남기지
  않도록 호출 측에서 PYTHONDONTWRITEBYTECODE=1 을 준다.

입력은 사용자가 웹 폼에 넣은 값 그대로. 출력은 이 폴더(GrainReverseEngineering)
아래 CSV 와 JSON.
"""
import json
import os
import sys

sys.dont_write_bytecode = True

REF = os.path.join(os.path.expanduser("~"), "mnt",
                   "6DOF RocketSim with Unreal")
if not os.path.isdir(REF):
    REF = os.environ.get("GRE_REF_SIM_DIR", "")
sys.path.insert(0, REF)
os.chdir(REF)                      # 모듈이 상대 경로 자원을 찾을 수 있게

import matplotlib                  # noqa: E402
matplotlib.use("Agg")
import numpy as np                 # noqa: E402

import rocket_input_builder as rib  # noqa: E402

# ── 사용자가 폼에 넣은 값 (스크린샷 그대로). a/n 은 **비워서** 보낸다.
RAW = {
    "propellant_type": "KNSB",
    "D_c_mm": 50.0, "L_c_mm": 106.0,
    "D_g_mm": 50.0, "D_gc_0_mm": 20.0,
    "L_gs_mm": 30.0, "N_segments": 3,
    "seg_gap_mm": 4.0, "gap_min_mm": 0.5, "gap_full_mm": 2.0,
    "Den_ratio": 0.92,
    "P_target_MPa": 3.0,
    "ER": 8.0, "NOZ_eff": 0.85, "e_NOZ": 0.0,
    "eff_combustion": 0.85,
    "t_interval": 0.002,
    "a": "", "n": "",
}
OVERRIDE = json.loads(os.environ.get("GRE_OVERRIDE", "{}"))
RAW.update(OVERRIDE)

OUT = os.path.join(os.path.expanduser("~"), "mnt", "GrainReverseEngineering",
                   "data")
os.makedirs(OUT, exist_ok=True)

params, _ = rib._tms_prepare_params(RAW)
res = rib.run_tms_sim_series(params)

if res.get("error"):
    print("오류:", res["error"]); print(res.get("trace", "")); sys.exit(1)

s = res["summary"]
print("── 외부 시뮬 요약 (직접 실행) ──")
for k in ("propellant", "burn_model", "geometry_model", "burn_time_s",
          "max_thrust_N", "avg_thrust_N", "total_impulse_Ns",
          "max_pressure_MPa", "avg_isp_s", "prop_mass_kg", "Kn_design",
          "Kn_min", "Kn_max", "A_NOZ_t_mm2", "D_throat_mm", "Kn_target",
          "A_NOZ_t_design_mm2", "T_chamber_K", "Den_actual_gcc", "V_load"):
    if k in s:
        v = s[k]
        print(f"  {k:22} = {v:.6g}" if isinstance(v, (int, float))
              else f"  {k:22} = {v}")
print("\n  실제로 쓰인 a, n =", params.get("a"), params.get("n"),
      "  (빈값이면 run_tms_sim_series 의 기본값 3.907 / 0.535)")
for w in (res.get("warnings") or []):
    print("  경고:", w)

ser = res["series"]
keys = [k for k in ser if isinstance(ser[k], list) and len(ser[k]) == len(ser["t_s"])]
print("\n  시계열 키:", keys, f"({len(ser['t_s'])} 점)")
path = os.path.join(OUT, "external_series.csv")
with open(path, "w", encoding="utf-8") as fh:
    fh.write(",".join(keys) + "\n")
    for i in range(len(ser["t_s"])):
        fh.write(",".join(f"{float(ser[k][i]):.9g}" for k in keys) + "\n")
print("  →", path)

jpath = os.path.join(OUT, "external_summary.json")
with open(jpath, "w", encoding="utf-8") as fh:
    json.dump({"summary": s, "params_used": {k: v for k, v in params.items()
                                             if not isinstance(v, (list, dict))},
               "warnings": res.get("warnings")}, fh,
              ensure_ascii=False, indent=2, default=str)
print("  →", jpath)
