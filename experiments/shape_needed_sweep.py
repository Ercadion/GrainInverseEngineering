"""어떤 곡선이 비원형 포트를 요구하는가 — 곡선 공간 쓸기.

실제 형상 N 개를 정방향으로 돌려 추력 곡선을 얻고, 각 곡선마다
`circle_can_make_curve` 로 **원형 포트 + 노즐목 두 변수만으로** 얼마나 맞출 수
있는지를 잰다. 결과를 `.npz` 로 떨궈 `shape_needed_fig.py` 가 그린다.

(쓸기만 10 분 가량 걸리므로 그림 그리기와 분리해 둔다.)
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from gre.core.burnrate import model_for
from gre.optimize.forward import DiffMotor
from gre.validate.identifiability import circle_can_make_curve

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)
torch.set_default_dtype(torch.float64)

R_OUT = 30.0
M = DiffMotor(d_outer=2 * R_OUT, n_grid=121, n_samples=384, n_w=200, h_max=10e-3)
BR = model_for("KNSB")

#: (모드 번호, 진폭) — 0 은 원형 (대조군, 잔차가 0 에 가까워야 한다)
LOBES = [(0, 0.0), (3, 2.2), (3, 4.5), (4, 3.0), (4, 5.0), (6, 2.5), (6, 4.5)]
R0S = [8.0, 11.0, 14.0, 17.0, 20.0]
LENS = [55.0, 95.0, 140.0]
ATS = [80.0, 120.0]


def metrics(t, f):
    """곡선 모양 지표 — 크기(스케일)에 무관하게."""
    fm = f.max()
    live = f > 0.05 * fm
    if live.sum() < 5:
        return None
    t0, t1 = t[live][0], t[live][-1]
    t_burn = t1 - t0
    f_mean = np.trapezoid(f[live], t[live]) / max(t_burn, 1e-9)
    # 피크 시점 — 평탄한 구간에서 흔들리지 않도록 상위 2 % 구간의 중앙값
    hi = t[f >= 0.98 * fm]
    t_peak = float(np.median(hi))
    q = lambda frac: float(np.interp(t0 + frac * t_burn, t, f))
    return dict(f_max=float(fm), f_mean=float(f_mean), t_burn=float(t_burn),
                peak_frac=float((t_peak - t0) / max(t_burn, 1e-9)),
                peakiness=float(fm / max(f_mean, 1e-9)),
                front_back=float(q(0.25) / max(q(0.75), 1e-9)),
                impulse=float(np.trapezoid(f, t)))


rows = []
t_start = time.perf_counter()
total = len(LOBES) * len(R0S) * len(LENS) * len(ATS)
done = 0
for n_lobe, amp in LOBES:
    for r0 in R0S:
        for L in LENS:
            for at in ATS:
                done += 1
                if r0 - amp < 1.5 or r0 + amp > R_OUT - 3.0:
                    continue
                a = torch.zeros(6)
                if n_lobe:
                    a[n_lobe - 1] = amp
                try:
                    with torch.no_grad():
                        t, p, f, d = M.simulate(torch.tensor(r0), a, torch.zeros(6),
                                                [L], torch.tensor(at))
                    t, f, p = t.numpy(), f.numpy(), p.numpy()
                    m = metrics(t, f)
                    if m is None or d["mass_g"] > 900.0 or p.max() > 11.0:
                        continue
                    # ⚠ 전 구간으로 재면 **실제 원형**조차 잔차 4~9 % 가 나온다.
                    #   `circle_can_make_curve` 는 준정상 평형식이라 점화 상승과
                    #   소진 뒤 블로다운을 담지 못하기 때문이다. 그 바닥값은 모델
                    #   오차지 형상 정보가 아니므로, 손실의 ω(t)(§4.5)와 같은 창
                    #   — 점화 끝 이후 · F ≥ 20 % F_max — 안에서만 잰다.
                    #   (라이브러리 기본 동작은 건드리지 않고 호출 쪽에서 처리)
                    plateau = np.median(f[f > 0.5 * f.max()])
                    i0 = int(np.argmax(f >= 0.90 * plateau))
                    msk = np.zeros(len(f), dtype=bool)
                    msk[i0:] = True
                    msk &= f >= 0.20 * f.max()
                    res = circle_can_make_curve(t[msk], f[msk], R_OUT, L, BR)
                    res_all = circle_can_make_curve(t, f, R_OUT, L, BR)
                except Exception as exc:                       # noqa: BLE001
                    print(f"  skip (n={n_lobe} amp={amp} r0={r0} L={L} At={at}): {exc}")
                    continue
                m.update(n_lobe=n_lobe, amp=amp, r0=r0, length=L, a_throat=at,
                         p_max=float(p.max()), mass_g=float(d["mass_g"]),
                         residual=res["residual_pct"], fit_r0=res["r0"],
                         fit_at=res["a_throat"],
                         residual_all=res_all["residual_pct"])
                rows.append(m)
                if done % 20 == 0:
                    print(f"  {done}/{total}  ({time.perf_counter()-t_start:.0f} s, "
                          f"{len(rows)} 유효)", flush=True)

print(f"\n{len(rows)} 형상, {time.perf_counter()-t_start:.0f} s")
keys = sorted(rows[0])
np.savez(os.path.join(OUT, "shape_needed.npz"),
         **{k: np.array([r[k] for r in rows], dtype=float) for k in keys})
print("saved", os.path.join(OUT, "shape_needed.npz"))

res = np.array([r["residual"] for r in rows])
ral = np.array([r["residual_all"] for r in rows])
cir = np.array([r["n_lobe"] == 0 for r in rows])
for name, v in (("창 안", res), ("전 구간", ral)):
    print(f"[{name}] 대조군(실제 원형) 잔차  중앙값 {np.median(v[cir]):.2f} %  "
          f"최대 {v[cir].max():.2f} %   ← 이 값이 모델 오차 바닥이다")
    print(f"[{name}] 비원형 형상 잔차        중앙값 {np.median(v[~cir]):.2f} %  "
          f"최대 {v[~cir].max():.2f} %")
    print(f"[{name}] 비원형 중 '원형으로 충분(≤10 %)' 비율: "
          f"{100*np.mean(v[~cir] <= 10.0):.0f} %\n")
