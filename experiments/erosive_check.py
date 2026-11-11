#!/usr/bin/env python3
"""세그먼트별 포트 질량유속 진단을 실제 설계 셋에 돌린다.

    PYTHONPATH=. python experiments/erosive_check.py

설계
  A  외부 대조 사례   Ø50 · 코어 Ø20 · 30 mm × 3 · A_t 61.75 mm²
  B  Ø40 재설계 씨앗 0      out/cad_D40_seed0/design_seed0.json · 74 mm × 3 · A_t 86.59 mm²
  C  같은 B 형상 · 세그먼트 4 개 (길이를 늘리면 G 가 어떻게 커지나)
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
torch.set_default_dtype(torch.float64)
from gre.optimize.forward import DiffMotor
from gre.validate.erosive import port_flux, summarize

B = {"r0": 11.672774499719615,
     "a": [0.5559265172881465, -0.2732726877460942, 1.2540442619278975, 0.9727843602468323],
     "b": [-0.9093678023156571, 0.8383694085513093, -1.734723475976807e-18, -0.008150583102912392],
     "a_throat": 86.59014751456867}

CASES = [
    ("A 외부 대조 Ø50 BATES", 50.0, 10.0, [0.0] * 4, [0.0] * 4, [30.0] * 3, 61.75),
    ("B Ø40 씨앗0 74×3", 40.0, B["r0"], B["a"], B["b"], [74.0] * 3, B["a_throat"]),
    ("C Ø40 씨앗0 74×4", 40.0, B["r0"], B["a"], B["b"], [74.0] * 4, B["a_throat"]),
]

rows = []
for name, d, r0, a, b, L, at in CASES:
    m = DiffMotor(d_outer=d, n_grid=121)
    tab = port_flux(m, torch.tensor(r0), torch.tensor(a), torch.tensor(b), L, at)
    s = summarize(tab)
    g_last0 = float(tab["G"][-1, 0])
    print(f"{name}")
    print(f"  최대 G {s['G_max']:7.1f} kg/m²s (세그 {s['G_seg']}, w={s['G_w']:.2f} mm) · "
          f"점화 직후 노즐 쪽 G {g_last0:7.1f}")
    print(f"  최소 A_port/A_t {s['port_ratio_min']:.2f} (세그 {s['pr_seg']}) · "
          f"최대 포트 M {s['mach_max']:.3f} · 헤드–노즐 압력차 ≈ {s['dp_head_pct_max']:.1f} %")
    print(f"  P_eq 초기 {float(tab['p_eq'][0]):.2f} MPa · 경고: {s['flags'] or '없음'}\n")
    rows.append({"case": name, **{k: v for k, v in s.items() if k != 'limits'},
                 "G_aft_w0": g_last0, "p_eq0": float(tab["p_eq"][0])})
json.dump(rows, open("experiments/out/erosive_check.json", "w"), ensure_ascii=False, indent=1)
