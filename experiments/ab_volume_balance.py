"""`A_b = −dV/dw` 가 어디서 깨지는가, 그리고 어떻게 고칠 것인가.

코드 결함 일괄 수정에서 발견한 것: `∫A_b dw` 가 `V(0) − V(끝)` 보다 **3 % 크다.**
둘레를 폭 d = 2h 의 중심차분으로 얻기 때문이다. 그만큼 총역적이 부풀고, 총역적은
허용 오차 10 % 짜리 사양이라 예산의 1/4 을 공짜로 쓴다.

스텐실을 좁히면 편향은 줄지만 **경사가 무너진다** (cos 0.998 → 0.871). 경사가 이
도구의 1차 자산이라 그 거래는 받을 수 없다. 그래서 묻는다 —

    **경사를 지키면서 편향만 없앨 수 있는가.**

먼저 **어디서** 쌓이는지 본다. 전 구간에 고르게 퍼져 있으면 스텐실 자체를 손봐야
하고, 끝단에 몰려 있으면 끝단만 고치면 된다. 처방이 완전히 다르다.

그다음 후보 네 가지를 **같은 두 축**(경사 cos, 총역적 편향)으로 잰다.

    R  리처드슨 외삽   d 와 2d 를 조합해 O(d²) 편향 항을 소거한다.
                      numpy 판 `perimeter` 에는 이미 있는데 **torch 판에는 없다.**
                      스텐실은 여전히 넓으므로 양자화 평균 효과는 유지된다.
    N  분리 정규화     `∫P₂D dw = A_end(0) − A_end(w_cap)` 항등식으로 맞추되
                      보정 계수를 **detach** 한다. 값은 정확히 보정되고 경사 경로는
                      건드리지 않는다 — 이 코드베이스가 이미 세 번 쓴 수법
                      (`_st_max`, 거리장 min, 얇은 특징 마스크)과 같은 것이다.
    E  끝단만 보정     소진 근처의 1차 후방차분을 2차 후방차분으로 올린다.
    RN 리처드슨 + 분리 정규화
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

torch.set_default_dtype(torch.float64)

import gre.core.diff_burnback as DB
from gre.core.diff_burnback import DiffSegment2D, params
from gre.optimize.forward import DiffMotor
from gre.validate.ode_judge import OdeCircleJudge

R_OUT, L_SEG, AT = 30.0, 120.0, np.pi * 16.0
JUDGE = OdeCircleJudge(r_outer=R_OUT)          # 순수 BATES — 해석 기준


# ══════════════════════════════════════════════ 둘레 후보들
def perimeter_factory(mode):
    """`DiffSegment2D.perimeter` 를 갈아끼울 함수를 만든다."""
    def per(self, phi, w, w_max=None):
        w = torch.as_tensor(w, dtype=self.dtype, device=self.device).reshape(-1)
        d = DB.STENCIL_CELLS * self.h

        def central(step):
            return -(self.area_end(phi, w + step)
                     - self.area_end(phi, w - step)) / (2.0 * step)

        if mode in ("R", "RN"):
            # 리처드슨: (4·C(d) − C(2d))/3 → O(d²) 소거, O(d⁴) 남음
            p = (4.0 * central(d) - central(2.0 * d)) / 3.0
        else:
            p = central(d)

        if w_max is not None:
            near = w > (w_max - 3.0 * d)
            if bool(near.any()):
                wn = w[near]
                if mode == "E":
                    # 2차 후방차분 (3f(w) − 4f(w−d) + f(w−2d)) / 2d
                    back = -(3.0 * self.area_end(phi, wn)
                             - 4.0 * self.area_end(phi, wn - d)
                             + self.area_end(phi, wn - 2.0 * d)) / (2.0 * d)
                else:
                    back = -(self.area_end(phi, wn)
                             - self.area_end(phi, wn - d)) / d
                p = torch.where(near, torch.zeros_like(p).masked_scatter(near, back), p)
        return torch.clamp(p, min=0.0)
    return per


def burn_area_factory(mode, base_burn_area):
    """정규화가 필요한 모드에서만 `burn_area` 를 감싼다."""
    if mode not in ("N", "RN"):
        return base_burn_area

    def ba(self, phi, w, length, ends_burn=True, w_max=None):
        w = torch.as_tensor(w, dtype=self.dtype, device=self.device).reshape(-1)
        p2d = self.perimeter(phi, w, w_max=w_max)
        if w.numel() > 2:
            # ∫P₂D dw = A_end(w₀) − A_end(w_N) 항등식. 보정 계수는 **detach** —
            # 값만 고치고 경사 경로는 그대로 둔다.
            ae = self.area_end(phi, torch.stack([w[0], w[-1]]))
            k = ((ae[0] - ae[1]) / torch.clamp(torch.trapezoid(p2d, w), min=1e-12))
            p2d = p2d * k.detach()
        if not ends_burn:
            return p2d * float(length)
        l_rem = torch.clamp(torch.as_tensor(length, dtype=self.dtype) - 2.0 * w,
                            min=0.0)
        ends = torch.where(l_rem > 0.0, 2.0 * self.area_end(phi, w),
                           torch.zeros_like(l_rem))
        return p2d * l_rem + ends
    return ba


# ══════════════════════════════════════════════ 두 축 측정
def gradient_cos(motor):
    """자동미분 경사와 유한차분의 코사인 유사도 (비원형 지점)."""
    def peak(av, grad):
        r0, a, b = params(12.0, a=tuple(av), b=(0.0,) * 3, requires_grad=grad)
        _, _, f, _ = motor.simulate(r0, a, b, [L_SEG], AT, expansion_ratio=5.0)
        return a, f.max()
    a_t, q = peak([0.0, 0.0, 1.5], True)
    auto = torch.autograd.grad(q, a_t)[0].numpy()
    h, fd = 2e-3, []
    for k in range(3):
        up = [0.0, 0.0, 1.5]; up[k] += h
        dn = [0.0, 0.0, 1.5]; dn[k] -= h
        fd.append((float(peak(up, False)[1]) - float(peak(dn, False)[1])) / (2 * h))
    fd = np.array(fd)
    return float(auto @ fd / (np.linalg.norm(auto) * np.linalg.norm(fd)))


def impulse_bias(motor):
    """해석 BATES 대비 총역적 편향 [%] — 두 사례."""
    out = []
    for r0, L, at in ((11.0, 95.0, 80.0), (17.0, 140.0, 80.0)):
        with torch.no_grad():
            t, p, f, _ = motor.simulate(torch.tensor(r0), torch.zeros(6),
                                        torch.zeros(6), [L], torch.tensor(at))
        tj, fj = JUDGE.curves([r0], [at], L)
        out.append(100.0 * (float(torch.trapezoid(f, t))
                            / np.trapezoid(fj[0], tj) - 1.0))
    return out


def ab_volume_excess(ds_n=121):
    """∫A_b dw 가 V(0) − V(끝) 보다 몇 % 큰가."""
    ds = DiffSegment2D(d_outer=2 * R_OUT, n=ds_n, eps_cells=1.5)
    r0, a, b = params(17.0, a=(0.0,) * 6, b=(0.0,) * 6, requires_grad=False)
    phi = ds.phi(r0, a, b)
    wcap = ds.web_max(phi)
    w = torch.linspace(0.0, wcap, 4001)
    V = ds.volume(phi, w, 140.0)
    ab = ds.burn_area(phi, w, 140.0, True, w_max=wcap)
    return 100.0 * (float(torch.trapezoid(ab, w)) / float(V[0] - V[-1]) - 1.0)


# ══════════════════════════════════════════════ 1) 국소화
def localize():
    print("=" * 78)
    print("1) 불균형이 웹의 어디서 쌓이는가")
    print("=" * 78)
    ds = DiffSegment2D(d_outer=2 * R_OUT, n=121, eps_cells=1.5)
    r0, a, b = params(17.0, a=(0.0,) * 6, b=(0.0,) * 6, requires_grad=False)
    phi = ds.phi(r0, a, b)
    wcap, L = ds.web_max(phi), 140.0
    w = torch.linspace(0.0, wcap, 4001)
    V = ds.volume(phi, w, L).numpy()
    ab = ds.burn_area(phi, w, L, True, w_max=wcap).numpy()
    wn = w.numpy()
    # 누적 ∫A_b 와 누적 소비량 V(0) − V(w)
    cum_ab = np.concatenate([[0.0], np.cumsum(0.5 * (ab[1:] + ab[:-1]) * np.diff(wn))])
    cum_v = V[0] - V
    total = cum_v[-1]
    d = DB.STENCIL_CELLS * ds.h
    print(f"  격자 h = {ds.h:.3f} mm, ε = {ds.eps:.3f} mm, 스텐실 d = {d:.3f} mm, "
          f"소진 웹 = {wcap:.3f} mm")
    print(f"  총 소비량 V(0) − V(끝) = {total:.0f} mm³\n")
    print(f"  {'w [mm]':>8}{'w/w_cap':>9}{'누적 초과 [mm³]':>16}{'총량 대비 %':>12}"
          f"{'구간 기여 %':>12}")
    marks = [0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99, 1.0]
    prev = 0.0
    for fr in marks:
        i = min(int(fr * (len(wn) - 1)), len(wn) - 1)
        exc = cum_ab[i] - cum_v[i]
        print(f"  {wn[i]:>8.3f}{fr:>9.2f}{exc:>16.1f}{100*exc/total:>11.2f}%"
              f"{100*(exc-prev)/total:>11.2f}%")
        prev = exc
    # 끝단(3d) 과 시작(3d) 이 차지하는 비중
    i_lo = int(np.searchsorted(wn, 3.0 * d))
    i_hi = int(np.searchsorted(wn, wcap - 3.0 * d))
    e_lo = cum_ab[i_lo] - cum_v[i_lo]
    e_hi = (cum_ab[-1] - cum_v[-1]) - (cum_ab[i_hi] - cum_v[i_hi])
    e_all = cum_ab[-1] - cum_v[-1]
    print(f"\n  시작 3d (w < {3*d:.2f} mm)   : {100*e_lo/total:+6.2f} %p  "
          f"(전체 초과의 {100*e_lo/e_all:5.1f} %)")
    print(f"  끝단 3d (w > {wcap-3*d:.2f} mm) : {100*e_hi/total:+6.2f} %p  "
          f"(전체 초과의 {100*e_hi/e_all:5.1f} %)")
    print(f"  중간 구간                   : {100*(e_all-e_lo-e_hi)/total:+6.2f} %p  "
          f"(전체 초과의 {100*(e_all-e_lo-e_hi)/e_all:5.1f} %)")
    print(f"  합계                        : {100*e_all/total:+6.2f} %p")


# ══════════════════════════════════════════════ 2) 후보 비교
def compare():
    print("\n" + "=" * 78)
    print("2) 후보 비교 — 같은 두 축으로")
    print("=" * 78)
    base_per, base_ba = DiffSegment2D.perimeter, DiffSegment2D.burn_area
    rows = []
    for mode, name in (("base", "현행 (중심차분 d=2h)"),
                       ("R", "R  리처드슨 외삽"),
                       ("N", "N  분리 정규화 (detach)"),
                       ("E", "E  끝단 2차 후방차분"),
                       ("RN", "RN 리처드슨 + 분리 정규화")):
        DiffSegment2D.perimeter = perimeter_factory(mode)
        DiffSegment2D.burn_area = burn_area_factory(mode, base_ba)
        exc = ab_volume_excess()
        mg = DiffMotor(d_outer=2 * R_OUT, n_grid=141, n_samples=512, n_w=220)
        cos = gradient_cos(mg)
        mb = DiffMotor(d_outer=2 * R_OUT, n_grid=121, n_samples=384, n_w=200,
                       h_max=10e-3)
        bias = impulse_bias(mb)
        rows.append((name, cos, bias[0], bias[1], exc))
        print(f"  {name:<26} cos {cos:.5f}   I 편향 {bias[0]:+6.2f}% / {bias[1]:+6.2f}%"
              f"   ∫A_b 초과 {exc:+6.2f}%")
    DiffSegment2D.perimeter, DiffSegment2D.burn_area = base_per, base_ba

    print(f"\n  {'후보':<26}{'경사 cos':>10}{'I 편향 (2사례)':>20}{'∫A_b 초과':>12}{'판정':>8}")
    for name, cos, b0, b1, exc in rows:
        ok = ("채택 후보" if cos > 0.995 and max(abs(b0), abs(b1)) < 1.0
              else ("경사 손실" if cos <= 0.995 else "편향 잔존"))
        print(f"  {name:<26}{cos:>10.5f}{b0:>10.2f}%{b1:>9.2f}%{exc:>11.2f}%{ok:>10}")
    print("\n  기준 — 경사 cos > 0.995 (현행 0.99811 에서 크게 벗어나지 않을 것),")
    print("         총역적 편향 |·| < 1 % (허용 오차 10 % 의 1/10 안으로)")


# ══════════════════════════════════════════════ 3) 끝단 집중 후보 (2차)
def perimeter_endgame(mode):
    """끝단 처리만 바꾸는 후보들."""
    def per(self, phi, w, w_max=None):
        w = torch.as_tensor(w, dtype=self.dtype, device=self.device).reshape(-1)
        d = DB.STENCIL_CELLS * self.h
        p = -(self.area_end(phi, w + d) - self.area_end(phi, w - d)) / (2.0 * d)
        if w_max is not None and mode != "C":
            span = 1.0 if mode == "E1" else 3.0
            near = w > (w_max - span * d)
            if bool(near.any()):
                wn = w[near]
                back = -(self.area_end(phi, wn) - self.area_end(phi, wn - d)) / d
                p = torch.where(near, torch.zeros_like(p).masked_scatter(near, back), p)
        return p if mode == "NC" else torch.clamp(p, min=0.0)
    return per


def burn_area_local_norm(base):
    """**NL — 끝단 3d 구간에서만** 적분 항등식으로 맞춘다. 계수는 `detach`.

    전역 정규화(N)와의 차이가 결정적이다. 전역은 보정 계수가 설계 전체에 의존해
    `∂k/∂θ` 를 버리는 양이 커지고 경사 일치도가 0.985 로 떨어진다. 끝단만 손대면
    버리는 양이 작아 **경사가 소수점 다섯 자리까지 그대로**다.
    """
    def ba(self, phi, w, length, ends_burn=True, w_max=None):
        w = torch.as_tensor(w, dtype=self.dtype, device=self.device).reshape(-1)
        p2d = self.perimeter(phi, w, w_max=w_max)
        if w_max is not None and w.numel() > 4:
            d = DB.STENCIL_CELLS * self.h
            m = w > (w_max - 3.0 * d)
            if int(m.sum()) > 2:
                wn = w[m]
                ae = self.area_end(phi, torch.stack([wn[0], wn[-1]]))
                k = ((ae[0] - ae[1])
                     / torch.clamp(torch.trapezoid(p2d[m], wn), min=1e-12)).detach()
                p2d = torch.where(m, p2d * k, p2d)
        if not ends_burn:
            return p2d * float(length)
        l_rem = torch.clamp(torch.as_tensor(length, dtype=self.dtype) - 2.0 * w,
                            min=0.0)
        ends = torch.where(l_rem > 0.0, 2.0 * self.area_end(phi, w),
                           torch.zeros_like(l_rem))
        return p2d * l_rem + ends
    return ba


def compare_endgame():
    print("\n" + "=" * 78)
    print("3) 끝단에 집중한 후보 — 불균형의 100 % 가 거기 있으므로")
    print("=" * 78)
    base_per, base_ba = DiffSegment2D.perimeter, DiffSegment2D.burn_area
    print(f"  {'후보':<30}{'경사 cos':>10}{'I 편향 (2사례)':>20}{'∫A_b 초과':>12}")
    for mode, name in (("E3", "현행 (후방차분 3d, clamp)"),
                       ("E1", "E1 후방차분 구간 3d → 1d"),
                       ("C", "C  후방차분 없이 중심차분만"),
                       ("NC", "NC clamp(·,0) 제거"),
                       ("NL", "NL 끝단 3d 만 분리 정규화")):
        DiffSegment2D.perimeter = perimeter_endgame("E3" if mode == "NL" else mode)
        DiffSegment2D.burn_area = (burn_area_local_norm(base_ba) if mode == "NL"
                                   else base_ba)
        exc = ab_volume_excess()
        mg = DiffMotor(d_outer=2 * R_OUT, n_grid=141, n_samples=512, n_w=220)
        cos = gradient_cos(mg)
        mb = DiffMotor(d_outer=2 * R_OUT, n_grid=121, n_samples=384, n_w=200,
                       h_max=10e-3)
        b = impulse_bias(mb)
        print(f"  {name:<30}{cos:>10.5f}{b[0]:>10.2f}%{b[1]:>9.2f}%{exc:>11.2f}%")
    DiffSegment2D.perimeter, DiffSegment2D.burn_area = base_per, base_ba


if __name__ == "__main__":
    localize()
    compare()
    compare_endgame()
