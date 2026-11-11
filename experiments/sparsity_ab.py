"""그룹 L1 이 실제로 참 형상 쪽으로 모으는가 — A/B 대조.

같은 목표·같은 출발점으로 두 번 돌린다. 차이는 희소성 벌점 하나뿐이다.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from gre.optimize.forward import DiffMotor
from gre.optimize.runner import canonical_rotation, shape_distance
from gre.validate.identifiability import assess

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)
torch.set_default_dtype(torch.float64)
M = DiffMotor(d_outer=60.0, n_grid=121, n_samples=384)
W = torch.linspace(0.5, 13.0, 140); L, K, R_OUT = 95.0, 4, 30.0
TGT = (11.0, np.array([0., 0., 2.2, 0.]), np.zeros(K))

def a_b(r0, a, b): return M.ds.burn_area(M.ds.phi(r0, a, b), W, L, True)
with torch.no_grad():
    ab_t = a_b(torch.tensor(TGT[0]), torch.tensor(TGT[1]), torch.tensor(TGT[2]))
scale = float(ab_t.max())

ident = assess(W.numpy(), ab_t.numpy(), R_OUT, L)
print("[식별성]", ident.note)
for x in ident.warnings(): print("  ⚠", x)

def fit(seed, lam, gauge, iters=150):
    g = np.random.default_rng(seed)
    r0 = torch.tensor(float(g.uniform(9., 14.)), requires_grad=True)
    a = torch.tensor(g.normal(0, .6, K), requires_grad=True)
    b = torch.tensor(g.normal(0, .6, K), requires_grad=True)
    opt = torch.optim.Adam([r0, a, b], lr=.06)
    warm = int(.35 * iters)
    for it in range(iters):
        opt.zero_grad()
        lc = (((a_b(r0, a, b) - ab_t) / scale) ** 2).mean()
        lw = 0.0 if it <= warm else lam * min(1., (it - warm) / (iters - warm))
        (lc + lw * torch.sqrt(a**2 + b**2 + 1e-6).sum() / R_OUT).backward()
        opt.step()
        with torch.no_grad(): r0.clamp_(4., 26.)
        if gauge: canonical_rotation(a, b)
    return (float(r0), a.detach().numpy().copy(), b.detach().numpy().copy(),
            float(torch.sqrt(lc.detach())) * 100)

print(f"\n{'seed':>5} | {'기존 (λ=0)':^32} | {'그룹 L1 (λ=0.03) + 게이지':^32}")
print(f"{'':>5} | {'잔차':>8} {'복잡도':>7} {'목표거리':>9} | {'잔차':>8} {'복잡도':>7} {'목표거리':>9}")
res = {0.0: [], 0.03: []}
t0 = time.perf_counter()
for s in range(6):
    row = f"{s:5d} |"
    for lam in (0.0, 0.03):
        r0, a, b, rr = fit(s, lam, gauge=(lam > 0))
        d = shape_distance(TGT, (r0, a, b))
        cx = float(np.hypot(a, b).sum())
        res[lam].append((rr, cx, d, r0, a, b))
        row += f" {rr:7.3f}% {cx:7.2f} {d:8.3f}mm |"
    print(row)
print(f"({time.perf_counter()-t0:.0f} s)\n")
for lam in (0.0, 0.03):
    rr, cx, d = (np.array([x[i] for x in res[lam]]) for i in range(3))
    print(f"λ={lam:4.2f}:  잔차 평균 {rr.mean():6.3f}%   복잡도 평균 {cx.mean():5.2f}"
          f"   목표거리 평균 {d.mean():5.3f} mm (최소 {d.min():.3f}, 최대 {d.max():.3f})")
d0 = np.array([x[2] for x in res[0.0]]); d1 = np.array([x[2] for x in res[0.03]])
print(f"\n→ 목표까지 평균 거리 {d0.mean():.3f} → {d1.mean():.3f} mm "
      f"({100*(d1.mean()-d0.mean())/d0.mean():+.1f} %)")
sp = lambda arr: [shape_distance((p[3],p[4],p[5]),(q[3],q[4],q[5]))
                  for i,p in enumerate(arr) for q in arr[i+1:]]
print(f"   해들끼리 흩어짐(중앙) {np.median(sp(res[0.0])):.3f} → {np.median(sp(res[0.03])):.3f} mm")

th = np.linspace(0, 2*np.pi, 721)
def rad(r0,a,b):
    k = np.arange(1,K+1)[:,None]
    return r0 + (a[:,None]*np.cos(k*th) + b[:,None]*np.sin(k*th)).sum(0)
fig, ax = plt.subplots(1,2, figsize=(10.6,4.6))
for j,(lam,title) in enumerate(((0.0,"(a) without sparsity"),(0.03,"(b) group L1 + gauge"))):
    rt = rad(TGT[0],TGT[1],TGT[2])
    ax[j].plot(rt*np.cos(th), rt*np.sin(th), "k-", lw=3.2, alpha=.45, label="target", zorder=5)
    for i,(rr,cx,d,r0,a,b) in enumerate(res[lam]):
        r = rad(r0,a,b); ax[j].plot(r*np.cos(th), r*np.sin(th), lw=1.2, alpha=.85)
    ax[j].set_aspect("equal"); ax[j].set_xlim(-20,20); ax[j].set_ylim(-20,20)
    ax[j].set_title(title + f"   mean dist {np.mean([x[2] for x in res[lam]]):.2f} mm", fontsize=10)
    ax[j].legend(fontsize=8); ax[j].set_xlabel("x [mm]"); ax[j].grid(alpha=.2)
fig.tight_layout(); fig.savefig(os.path.join(OUT,"sparsity.png"), dpi=160)
print(f"saved {OUT}/sparsity.png")
