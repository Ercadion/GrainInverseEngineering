"""V1-c — 자동미분 경사를 유한차분과 대조한다.

확인 항목
1. 격자 해상도에 따른 경사 일치도 (코사인 유사도·벡터 상대오차)
2. 대칭 방향의 경사가 0 인가 — soft-min 을 쓰는 이유를 수치로 보인다
3. 경사를 따라가면 목표 곡선과의 손실이 실제로 줄어드는가
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gre.core.diff_burnback import DiffSegment2D

D_OUTER, L_SEG = 60.0, 120.0
VEC0 = np.array([12.0, 0.3, -0.2, 1.2, 0.0, 0.1, 0.0])   # r0, a1..a3, b1..b3


def objective(seg, vec, grad=False):
    r0 = torch.tensor(vec[0], dtype=torch.float64, requires_grad=grad)
    a = torch.tensor(vec[1:4], dtype=torch.float64, requires_grad=grad)
    b = torch.tensor(vec[4:7], dtype=torch.float64, requires_grad=grad)
    w = torch.linspace(0.0, 8.0, 5, dtype=torch.float64)
    out = seg.burn_area(seg.phi(r0, a, b), w, length=L_SEG).sum()
    if grad:
        out.backward()
        return np.concatenate([[float(r0.grad)], a.grad.numpy(), b.grad.numpy()])
    return float(out.detach())


def fd_gradient(seg, vec, h_rel=1e-3):
    g = np.zeros_like(vec)
    for i in range(len(vec)):
        h = h_rel * max(1.0, abs(vec[i]))
        vp, vm = vec.copy(), vec.copy()
        vp[i] += h
        vm[i] -= h
        g[i] = (objective(seg, vp) - objective(seg, vm)) / (2 * h)
    return g


print("1) 격자 해상도에 따른 경사 일치도")
print(f"   {'n':>5} {'코사인 유사도':>14} {'벡터 상대오차':>14}")
rows = []
for n in (121, 161, 241, 321):
    seg = DiffSegment2D(d_outer=D_OUTER, n=n)
    ga = objective(seg, VEC0, grad=True)
    gf = fd_gradient(seg, VEC0)
    cos = float(ga @ gf / (np.linalg.norm(ga) * np.linalg.norm(gf)))
    rel = float(np.linalg.norm(ga - gf) / np.linalg.norm(gf))
    rows.append((n, cos, rel))
    print(f"   {n:5d} {cos:14.6f} {rel*100:13.2f}%")

print("\n2) 대칭 방향 경사 (3점 스타에서 sin 계수 방향은 0 이어야 한다)")
seg = DiffSegment2D(d_outer=D_OUTER, n=161)
g_sym = objective(seg, np.array([12.0, 0.0, 0.0, 1.5, 0.0, 0.0, 0.0]), grad=True)
print(f"   |∂J/∂b| 최대 / |∂J/∂r0| = {np.max(np.abs(g_sym[4:]))/abs(g_sym[0]):.3e}")
print("   (hard min 을 쓰면 여기서 0 이 아닌 가짜 경사가 나온다)")

print("\n3) 경사 하강으로 목표 곡선 맞추기")
w = torch.linspace(0.0, 6.0, 7, dtype=torch.float64)
r0_t = torch.tensor(12.0, dtype=torch.float64)
a_t = torch.tensor([0.0, 0.0, 1.5], dtype=torch.float64)
b_t = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)
with torch.no_grad():
    target = seg.burn_area(seg.phi(r0_t, a_t, b_t), w, length=L_SEG)

r0 = torch.tensor(10.5, dtype=torch.float64, requires_grad=True)
a = torch.tensor([0.0, 0.0, 0.2], dtype=torch.float64, requires_grad=True)
b = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64, requires_grad=True)
opt = torch.optim.Adam([r0, a, b], lr=0.2)
losses, curves = [], []
for it in range(60):
    opt.zero_grad()
    ab = seg.burn_area(seg.phi(r0, a, b), w, length=L_SEG)
    loss = ((ab - target) ** 2).mean()
    loss.backward()
    opt.step()
    losses.append(float(loss.detach()))
    if it in (0, 10, 25, 59):
        curves.append((it, ab.detach().numpy().copy()))
print(f"   손실 {losses[0]:.3e} → {losses[-1]:.3e}  ({losses[-1]/losses[0]*100:.2f} %)")
print(f"   복원된 계수: r0={float(r0.detach()):.3f} (목표 12.0), a3={float(a[2].detach()):.3f} (목표 1.5)")

fig, ax = plt.subplots(1, 3, figsize=(13, 4))
ax[0].plot([r[0] for r in rows], [1 - r[1] for r in rows], "o-")
ax[0].set_yscale("log"); ax[0].set_xlabel("grid n"); ax[0].set_ylabel("1 - cosine similarity")
ax[0].set_title("Gradient agreement vs resolution")
ax[1].semilogy(losses, "o-"); ax[1].set_xlabel("iteration"); ax[1].set_ylabel("loss")
ax[1].set_title("Gradient descent on A_b target")
ax[2].plot(w.numpy(), target.numpy(), "k--", label="target")
for it, c in curves:
    ax[2].plot(w.numpy(), c, label=f"iter {it}")
ax[2].set_xlabel("web w [mm]"); ax[2].set_ylabel("A_b [mm2]"); ax[2].legend(); ax[2].set_title("Curve fit")
plt.tight_layout(); plt.savefig("out/gradcheck_v1c.png", dpi=130)
print("\nsaved out/gradcheck_v1c.png")
