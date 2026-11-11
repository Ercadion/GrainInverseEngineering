"""포트 질량유속 진단 — 침식연소 가정을 세그먼트별로 확인한다.

## 왜 필요한가

설계 문서의 제약 표와 리스크 표는 침식연소를 **"포트 면적 / 노즐목 면적
≥ 기준값" 하드 제약으로 회피**한다고 적고 있다. 그러나 **그 제약은 코드에
구현돼 있지 않다** (2026-11-05 `gre/` 전체 검색으로 확인). 모델은 연소속도를
압력만의 함수로 두므로, 이 가정이 깨지는 설계를 최적화기가 골라도 아무도 모른다.

또 면적비 하나로는 부족하다. 노즐 쪽 끝의 G 는

    G_aft = ṁ / A_port = p·A_t / (c*·A_port)

라 면적비가 곧 G 를 정하지만, **세그먼트마다 포트가 다르면 가장 좁은 포트가
마지막 세그먼트라는 보장이 없다.** 국소 질량유속은 그 지점보다 **상류에서
만들어진 질량**을 그 지점의 포트 면적으로 나눈 값이다:

    G_k(w) = ρ_p · r(p) · Σ_{j ≤ k} A_b,j(w)  /  A_port,k(w)

(출처: HRE1K CFD 이론서 10·20장 — 하이브리드 후퇴율 ṙ ∝ G^0.8 과 같은 물리가
고체에서는 침식연소 항 α·G^0.8/D^0.2·exp(−β·ρ_p·ṙ/G) 로 나타난다.)

## 무엇을 내나 (웹 w 축, 준정상 평형압 기준)

    G_k(w)        세그먼트 k 노즐 쪽 끝의 질량유속 [kg/m²s]
    port_ratio_k  A_port,k / A_t
    mach_k        포트 마하 수  M = G / (ρ_g·a)
    dp_head_pct   헤드엔드–노즐 쪽 압력차 근사  Δp/p ≈ γ·M²  (질량 유입 관로, 1D)

순서 규약: `lengths[0]` 이 **헤드엔드**, `lengths[-1]` 이 **노즐 쪽**이다.

⚠ **문턱값은 이 모듈이 정하지 않는다.** KNSB 에 대해 검증된 침식 문턱을 찾지
못했다. 함수는 수치만 내고, 판정 문턱은 호출자가 넘긴다 (`DEFAULT_LIMITS` 는
"경고를 띄울 출발점" 이지 근거 있는 값이 아니다 — [가정]).

미분 가능하다 (torch). 최적화 루프에 넣는 것은 **아직 하지 않았다** — 사용자
승인 대기 (`erosive_hinge` 가 그때 쓸 벌점 꼴).
"""
from __future__ import annotations

import math

import torch

MM2_M2, MPA_PA = 1e-6, 1e6

#: [가정] 경고 출발점. 근거 있는 KNSB 값이 아니다 — 측정·문헌으로 바꿀 것.
DEFAULT_LIMITS = {"G_max": 1000.0,        # kg/m²s
                  "port_ratio_min": 2.0,  # A_port / A_t
                  "mach_max": 0.3}


def port_flux(motor, r0, a, b, lengths, a_throat, n_w: int = 200):
    """세그먼트별 포트 질량유속 표. 반환 dict 의 텐서는 (세그먼트, w) 꼴."""
    lengths = [float(x) for x in lengths]
    segs = len(lengths)
    ds = motor.ds
    r0 = torch.atleast_1d(torch.as_tensor(r0, dtype=motor.dtype))
    a = a if a.ndim == 2 else a.unsqueeze(0)
    b = b if b.ndim == 2 else b.unsqueeze(0)
    if r0.shape[0] == 1 and segs > 1:
        r0 = r0.expand(segs)
        a, b = a.expand(segs, -1), b.expand(segs, -1)

    phis = [ds.phi(r0[i], a[i], b[i]) for i in range(segs)]
    wr = [ds.web_max(ph) for ph in phis]
    life = [min(wr[i], lengths[i] / 2.0) if motor.ends_burn else wr[i]
            for i in range(segs)]
    w = torch.linspace(0.0, max(life) * 0.999, n_w, dtype=motor.dtype)

    a_case = math.pi * ds.r_outer ** 2                                 # mm²
    ab = torch.stack([(w <= life[i]).to(motor.dtype)
                      * ds.burn_area(phis[i], w, lengths[i], motor.ends_burn,
                                     w_max=wr[i]) for i in range(segs)])
    a_port = torch.stack([a_case - (w <= life[i]).to(motor.dtype)
                          * ds.area_end(phis[i], w) for i in range(segs)])

    at = torch.as_tensor(a_throat, dtype=motor.dtype) * MM2_M2
    p_eq = _equilibrium_implicit(motor, ab.sum(0), at)                 # MPa
    r = motor.br.rate(p_eq) * 1e-3                                     # m/s
    mdot_cum = motor.rho_p * r * torch.cumsum(ab, 0) * MM2_M2          # kg/s
    G = mdot_cum / (a_port * MM2_M2)

    rho_g = p_eq * MPA_PA / motor.rt
    a_snd = math.sqrt(motor.gamma * motor.rt)
    mach = G / (rho_g * a_snd)
    return {"w": w, "p_eq": p_eq, "G": G, "a_port": a_port,
            "port_ratio": a_port * MM2_M2 / at, "mach": mach,
            "dp_head_pct": motor.gamma * mach ** 2 * 100.0, "life": life}


def _equilibrium_implicit(motor, ab_mm2, at):
    """준정상 평형압 [MPa] — 값은 `motor._equilibrium`(이분법), 경사는 음함수 정리.

    ⚠ 초판은 이분법 결과를 그대로 썼고(no_grad), 그래서 경사가 **압력 경로를
    통째로 잃었다** — 시험에서 자동미분 −0.261 vs 중앙차분 −0.375 (30 % 과소).
    G 는 r(p) 를 통해 p 에 걸려 있으므로 그 경로가 빠지면 안 된다.

    근 p₀ 에서 뉴턴 한 걸음 `p₀ − g(p₀)/g'(p₀)` 를 붙인다. 값은 g(p₀)≈0 이라
    그대로이고, 경사는 정확히 `−(∂g/∂θ)/(∂g/∂p)` 가 된다.
    """
    with torch.no_grad():
        p0 = motor._equilibrium(ab_mm2.detach(), at.detach())
    ab = ab_mm2 * MM2_M2

    def g(p):
        r = motor.br.rate(p) * 1e-3
        return ((motor.rho_p - p * MPA_PA / motor.rt) * ab * r
                - p * MPA_PA * at / motor.c_star_actual)

    p_var = p0.clone().requires_grad_(True)
    with torch.enable_grad():
        (dg_dp,) = torch.autograd.grad(g(p_var).sum(), p_var)
    return p0 - g(p0) / dg_dp.detach()


def summarize(tab, limits=None) -> dict:
    """최악값과 그 자리, 문턱 위반 여부."""
    lim = {**DEFAULT_LIMITS, **(limits or {})}
    G = tab["G"].detach()
    k, j = divmod(int(torch.argmax(G)), G.shape[1])
    pr = tab["port_ratio"].detach()
    kp, jp = divmod(int(torch.argmin(pr)), pr.shape[1])
    m = tab["mach"].detach()
    out = {"G_max": float(G.max()), "G_seg": k, "G_w": float(tab["w"][j]),
           "port_ratio_min": float(pr.min()), "pr_seg": kp,
           "pr_w": float(tab["w"][jp]), "mach_max": float(m.max()),
           "dp_head_pct_max": float(tab["dp_head_pct"].detach().max())}
    out["flags"] = [n for n, bad in (
        ("G_max", out["G_max"] > lim["G_max"]),
        ("port_ratio_min", out["port_ratio_min"] < lim["port_ratio_min"]),
        ("mach_max", out["mach_max"] > lim["mach_max"])) if bad]
    out["limits"] = lim
    return out


def erosive_hinge(tab, g_max: float, beta: float = 20.0):
    """(미연결) 최적화용 벌점 — soft-max(G)/G_lim − 1 의 힌지 제곱."""
    G = tab["G"].reshape(-1) / g_max
    smax = torch.logsumexp(beta * G, 0) / beta
    return torch.clamp(smax - 1.0, min=0.0) ** 2
