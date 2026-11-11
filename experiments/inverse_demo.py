"""역설계 end-to-end 시연.

알려진 스타 형상으로 목표 곡선을 만들고, 다른 출발점에서 복원해 본다.
해가 유일하지 않으므로 '계수가 같아지는가' 가 아니라 '곡선이 맞는가' 로 본다.
"""
import sys, time, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gre.optimize.forward import DiffMotor, resample
from gre.optimize.loss import Target, LossWeights
from gre.optimize.constraints import ManufacturingLimits
from gre.optimize.runner import seed_design, optimize, final_report, DesignVars
from gre.optimize.constraints import ThroatSpec  # noqa: E402


OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)

m = DiffMotor(d_outer=60.0, n_grid=141, n_samples=512, n_w=220, h_max=5e-3)
lim, wts = ManufacturingLimits(), LossWeights()
TGT = dict(r0=11.0, a=[0., 0., 2.2, 0.], L=95.0, At=78.0)

tg = DesignVars(r0=torch.tensor(TGT["r0"], dtype=torch.float64),
                a=torch.tensor(TGT["a"], dtype=torch.float64),
                b=torch.zeros(4, dtype=torch.float64),
                a_throat=torch.tensor(TGT["At"], dtype=torch.float64), lengths=[TGT["L"]])
tt, tp, tf, td = m.simulate(tg.r0, tg.a, tg.b, tg.lengths, tg.a_throat)
print(f"[목표] P_max {float(tp.max()):.3f} MPa  F_max {float(tf.max()):.1f} N  "
      f"I {float(torch.trapezoid(tf, tt)):.1f} N·s  장약 {td['mass_g']:.1f} g")
t_grid = torch.linspace(0.0, float(tt[-1]), 400, dtype=torch.float64)
target = Target(t=t_grid, f=resample(tt, tf, t_grid).detach(), p_max=float(tp.max()))

dv = seed_design(r0=13.0, n_modes=4, a_throat=95.0, lengths=(TGT["L"],),
                 throat=ThroatSpec.fixed(2.0 * (TGT["At"] / 3.141592653589793) ** 0.5),
                 seed_amplitude=0.4, seed=1)
r0_0, a_0 = float(dv.r0), dv.a.detach().clone()
t0 = time.perf_counter()
dv, hist = optimize(m, dv, target, lim, wts, iters=90, log_every=15,
                    throat=ThroatSpec.fixed(2.0 * (TGT["At"] / 3.141592653589793) ** 0.5))
print(f"  ({time.perf_counter()-t0:.0f} s)")
rep, (t, p, f, diag, omega) = final_report(m, dv, target, lim, wts)
print(f"\n복원 r0={float(dv.r0):.3f} (목표 {TGT['r0']})  a={dv.a.detach().numpy().round(3)}  "
      f"A_t={float(dv.a_throat):.2f} (목표 {TGT['At']})")
print(f"판정 모양 {rep['rms_shape_pct']:.2f}%  I {rep['I_err_pct']:.2f}%  F {rep['F_err_pct']:.2f}%  "
      f"P {rep['P_err_pct']:.2f}%  수렴={rep['converged']}")
print(f"제약 웹 {rep['web_mm']:.2f}  꼭짓점 {rep['r_tip_mm']:.2f}  장약 {rep['mass_g']:.0f}g  "
      f"최대압 {rep['P_max_sim']:.2f}  연결 {rep['components']}개  벌점 {rep['penalty']:.2e}")
conceded = [k for k, v in (("최대압", rep['P_err_pct']),) if v > 10.0]
if conceded:
    print(f"⚠ 우선순위에 따라 양보한 목표: {', '.join(conceded)} "
          f"(최대압 {rep['P_err_pct']:.1f}% — §4.5 의 최하위 소프트 목표)")

th = np.linspace(0, 2*np.pi, 512)
def rth(r0, a, b):
    k = np.arange(1, len(a)+1)[:, None]
    return float(r0) + (np.asarray(a)[:, None]*np.cos(k*th) + np.asarray(b)[:, None]*np.sin(k*th)).sum(0)
fig, ax = plt.subplots(1, 3, figsize=(13.2, 4.0))
ax[0].plot(hist and [h["loss"] for h in hist]); ax[0].set_yscale("log")
ax[0].set_xlabel("iteration"); ax[0].set_ylabel("loss"); ax[0].grid(alpha=.25)
ax[0].set_title("(a) Adam convergence", fontsize=10)
ax[1].plot(target.t.numpy(), target.f.numpy(), "k-", lw=3, alpha=.3, label="target")
ax[1].plot(t.detach().numpy(), f.detach().numpy(), color="#B4441F", lw=1.6, label="recovered")
ax[1].plot(t.detach().numpy(), omega.detach().numpy()*float(f.max())*0.15,
           color="#2C7A7B", lw=1.0, ls=":", label="$\\omega(t)$ (scaled)")
ax[1].set_xlabel("t [s]"); ax[1].set_ylabel("thrust [N]"); ax[1].grid(alpha=.25)
ax[1].set_title("(b) thrust curve", fontsize=10); ax[1].legend(fontsize=7.5)
for r, c, ls, lab in ((rth(TGT["r0"], TGT["a"], [0]*4), "k", "-", "target port"),
                      (rth(r0_0, a_0.numpy(), [0]*4), "gray", ":", "start"),
                      (rth(float(dv.r0), dv.a.detach().numpy(), dv.b.detach().numpy()),
                       "#B4441F", "-", "recovered")):
    ax[2].plot(r*np.cos(th), r*np.sin(th), color=c, ls=ls, lw=2.0 if c != "gray" else 1.2, label=lab)
ax[2].add_patch(plt.Circle((0, 0), 30.0, fill=False, ec="k", lw=1.2, ls="--"))
ax[2].set_aspect("equal"); ax[2].set_xlim(-32, 32); ax[2].set_ylim(-32, 32)
ax[2].set_title("(c) port shape — different shape, same curve", fontsize=10)
ax[2].legend(fontsize=7.5); ax[2].set_xlabel("x [mm]")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "inverse.png"), dpi=160)
print(f"saved {OUT}/inverse.png")
