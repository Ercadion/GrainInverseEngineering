"""비유일성 측정 — 기하만 쓰는 다중 출발 실험.

A_b(w) 를 같은 목표에 맞추는 형상이 몇 개나 있는지, 서로 얼마나 다른지 잰다.
ODE 를 빼고 기하만 쓰므로 빠르고, 비유일성의 근원(형상 → A_b 사상)만 본다.

회전은 **정확한 대칭**이므로(포트를 돌려도 A_b 는 같다) 형상 비교는 반드시
회전을 제거하고 해야 한다. 두 가지로 본다.
  - 회전 불변 지문  |c_k| = √(a_k² + b_k²)
  - 최적 회전 정렬 후의 윤곽 최대 차이 [mm]
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from gre.optimize.forward import DiffMotor

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)
torch.set_default_dtype(torch.float64)

M = DiffMotor(d_outer=60.0, n_grid=121, n_samples=384)
W = torch.linspace(0.5, 13.0, 140)
L_SEG, K = 95.0, 4
TH = np.linspace(0, 2*np.pi, 721)

def a_b(r0, a, b):
    return M.ds.burn_area(M.ds.phi(r0, a, b), W, L_SEG, True)

def radius(r0, a, b, th=TH):
    k = np.arange(1, len(a)+1)[:, None]
    return float(r0) + (np.asarray(a)[:, None]*np.cos(k*th)
                        + np.asarray(b)[:, None]*np.sin(k*th)).sum(0)

def align_dist(s1, s2):
    """최적 회전 정렬 후 윤곽 최대 차이 [mm]."""
    r1 = radius(*s1)
    best = np.inf
    for d in np.linspace(0, 2*np.pi, 721):
        k = np.arange(1, K+1)
        a2 = np.asarray(s2[1])*np.cos(k*d) - np.asarray(s2[2])*np.sin(k*d)
        b2 = np.asarray(s2[1])*np.sin(k*d) + np.asarray(s2[2])*np.cos(k*d)
        best = min(best, np.abs(r1 - radius(s2[0], a2, b2)).max())
    return best

TGT = (11.0, np.array([0., 0., 2.2, 0.]), np.zeros(K))
with torch.no_grad():
    ab_t = a_b(torch.tensor(TGT[0]), torch.tensor(TGT[1]), torch.tensor(TGT[2]))
scale = float(ab_t.max())

def fit(seed, iters=180):
    g = np.random.default_rng(seed)
    r0 = torch.tensor(float(g.uniform(9.0, 14.0)), requires_grad=True)
    a = torch.tensor(g.normal(0, 0.6, K), requires_grad=True)
    b = torch.tensor(g.normal(0, 0.6, K), requires_grad=True)
    opt = torch.optim.Adam([r0, a, b], lr=0.06)
    for _ in range(iters):
        opt.zero_grad()
        loss = (((a_b(r0, a, b) - ab_t)/scale)**2).mean()
        loss.backward(); opt.step()
        with torch.no_grad():
            r0.clamp_(4.0, 26.0)
    return (float(r0), a.detach().numpy().copy(), b.detach().numpy().copy(),
            float(torch.sqrt(loss.detach()))*100)

print(f"목표: r0={TGT[0]}, a={TGT[1]}, |c_k|={np.hypot(TGT[1],TGT[2]).round(3)}\n")
print(f"{'seed':>5} {'잔차 RMS':>10} {'r0':>7}  {'|c_k| (회전 불변 지문)':>30}  {'목표와 거리':>11}")
sols = []
t0 = time.perf_counter()
for s in range(8):
    r0, a, b, res = fit(s)
    ck = np.hypot(a, b)
    d = align_dist(TGT, (r0, a, b))
    sols.append((r0, a, b, res, ck, d))
    print(f"{s:5d} {res:9.4f}% {r0:7.3f}  {np.array2string(ck.round(3), separator=' '):>30}  {d:9.3f} mm")
print(f"\n({time.perf_counter()-t0:.0f} s)")

ok = [s for s in sols if s[3] < 1.0]
print(f"잔차 1 % 이내인 해: {len(ok)} / 8")
if len(ok) > 1:
    pd = [align_dist((p[0],p[1],p[2]), (q[0],q[1],q[2]))
          for i,p in enumerate(ok) for q in ok[i+1:]]
    print(f"  그 해들 사이 윤곽 거리(회전 정렬 후): 최소 {min(pd):.3f} / 중앙 {np.median(pd):.3f} / 최대 {max(pd):.3f} mm")
    print(f"  목표까지 거리: 최소 {min(s[5] for s in ok):.3f} / 최대 {max(s[5] for s in ok):.3f} mm")

fig, ax = plt.subplots(1, 2, figsize=(11.0, 4.4))
rt = radius(*TGT)
ax[0].plot(rt*np.cos(TH), rt*np.sin(TH), "k-", lw=3, alpha=.45, label="target", zorder=5)
for i, (r0, a, b, res, ck, d) in enumerate(sols):
    r = radius(r0, a, b)
    ax[0].plot(r*np.cos(TH), r*np.sin(TH), lw=1.2, alpha=.85,
               label=f"seed {i} ({res:.2f}%)")
ax[0].add_patch(plt.Circle((0,0), 30.0, fill=False, ec="k", lw=1.0, ls="--"))
ax[0].set_aspect("equal"); ax[0].set_xlim(-32,32); ax[0].set_ylim(-32,32)
ax[0].set_title("(a) solutions from 8 random starts", fontsize=10)
ax[0].legend(fontsize=6.5, loc="upper right"); ax[0].set_xlabel("x [mm]")
wn = W.numpy()
ax[1].plot(wn, ab_t.numpy(), "k-", lw=3, alpha=.45, label="target $A_b(w)$")
for i, (r0, a, b, res, ck, d) in enumerate(sols):
    with torch.no_grad():
        ax[1].plot(wn, a_b(torch.tensor(r0), torch.tensor(a), torch.tensor(b)).numpy(),
                   lw=1.0, alpha=.8)
ax[1].set_xlabel("web w [mm]"); ax[1].set_ylabel("$A_b$ [mm$^2$]")
ax[1].set_title("(b) the curves they produce", fontsize=10)
ax[1].legend(fontsize=7.5); ax[1].grid(alpha=.25)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "multistart.png"), dpi=160)
print(f"saved {OUT}/multistart.png")
