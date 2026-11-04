"""Figure "the damped harmonic oscillator" — lecture 6, PINN notebook.

Left: a mass hanging from a spring (stiffness k) and a damper (c), released from
u(0) = 1 with no velocity. Right: the displacement u(t) it then follows.
Both panels share the same vertical axis, so the mass sits at the height of u(0).
"""
from urnstyle import *
import numpy as np
from matplotlib.patches import Polygon

MU, K, T = 2.0, 100.0, 1.5          # same values as in the notebook
ZETA = MU / (2 * np.sqrt(K))
W0 = np.sqrt(K)
WD = W0 * np.sqrt(1 - ZETA ** 2)


def exact(t):
    return np.exp(-ZETA * W0 * t) * (np.cos(WD * t) + (ZETA * W0 / WD) * np.sin(WD * t))


YLIM = (-1.25, 2.55)
CEIL = 2.25                         # height of the ceiling
MH, MW = 0.42, 0.9                  # mass height / width (schematic units)
U0 = 1.0

fig = plt.figure(figsize=(1100 / 150, 500 / 150), dpi=150)
axL = fig.add_axes([0.02, 0.15, 0.24, 0.80])
axR = fig.add_axes([0.35, 0.15, 0.62, 0.80])

# ---------------------------------------------------------------- schematic
axL.set_xlim(-1.0, 1.35); axL.set_ylim(*YLIM); axL.axis("off")

# ceiling
axL.add_patch(Rectangle((-0.8, CEIL), 1.6, 0.12, fc=C3, ec="none", zorder=1))
axL.plot([-0.8, 0.8], [CEIL, CEIL], color=C1, lw=2.2, zorder=2)
for xh in np.linspace(-0.75, 0.7, 11):
    axL.plot([xh, xh + 0.1], [CEIL + 0.02, CEIL + 0.12], color=C1, lw=1.0, zorder=2)


def spring(ax, x, y_top, y_bot, n=7, amp=0.13, lead=0.18, **kw):
    ys = np.linspace(y_top - lead, y_bot + lead, 2 * n + 1)
    xs = x + amp * np.array([0] + [(-1) ** i for i in range(2 * n - 1)] + [0])
    ax.plot(np.r_[x, xs, x], np.r_[y_top, ys, y_bot], **kw)


def damper(ax, x, y_top, y_bot, w=0.13, **kw):
    ymid = (y_top + y_bot) / 2
    # cylinder hanging from the top, piston rod coming up from the mass
    ax.plot([x, x], [y_top, ymid + 0.25], **kw)
    ax.plot([x - w, x - w, x + w, x + w], [ymid + 0.25, ymid - 0.25, ymid - 0.25, ymid + 0.25], **kw)
    ax.plot([x - w * 0.7, x + w * 0.7], [ymid - 0.05, ymid - 0.05], **kw)
    ax.plot([x, x], [ymid - 0.05, y_bot], **kw)


def mass(ax, u, ghost=False):
    top = u + MH / 2
    kw = dict(color=C3 if ghost else C1, lw=1.4 if ghost else 2.0,
              zorder=2 if ghost else 4, solid_capstyle="round")
    if not ghost:
        spring(ax, -0.22, CEIL, top, **kw)
        damper(ax, 0.22, CEIL, top, **kw)
    ax.add_patch(FancyBboxPatch((-MW / 2, u - MH / 2), MW, MH,
                                boxstyle="round,pad=0,rounding_size=0.06",
                                fc="white" if ghost else C2,
                                ec=C3 if ghost else C1, lw=kw["lw"],
                                ls=(0, (3, 2)) if ghost else "-",
                                zorder=kw["zorder"] + 1))
    if not ghost:
        ax.text(0, u, "$m$", ha="center", va="center", fontsize=15,
                color="white", weight="bold", zorder=7)


mass(axL, 0.0, ghost=True)          # where it would rest (outline only)
mass(axL, U0)                       # where it is released from

axL.text(-0.42, 1.75, "$k$", fontsize=14, ha="right", va="center")
axL.text(0.42, 1.75, "$c$", fontsize=14, ha="left", va="center")

# displacement arrow, from equilibrium to the mass
axL.plot([0.55, 1.15], [0, 0], color=GREY, lw=1.2, ls=(0, (4, 3)))
arrow(axL, (0.95, 0.0), (0.95, U0), color=C5, lw=2.2, head=8)
axL.text(1.02, 0.5, "$u$", fontsize=15, color=C5, ha="left", va="center")
axL.text(0, -0.55, "equilibrium\n$u = 0$", fontsize=11, color=GREY,
         ha="center", va="top", style="italic")

# ---------------------------------------------------------------- curve
t = np.linspace(0, T, 900)
env = np.exp(-ZETA * W0 * t)
axR.set_xlim(0, T); axR.set_ylim(*YLIM)
axR.axhline(0, color=GREY, lw=1.2, ls=(0, (4, 3)), zorder=1)
axR.plot(t, env, color=C6, lw=1.4, ls=(0, (5, 3)), zorder=2)
axR.plot(t, -env, color=C6, lw=1.4, ls=(0, (5, 3)), zorder=2)
axR.plot(t, exact(t), color=C1, lw=2.4, zorder=3)
axR.plot([0], [U0], "o", color=C5, ms=7, zorder=4)

axR.annotate("released at rest\n$u(0)=1,\\ \\frac{du}{dt}(0)=0$", xy=(0.0, U0),
             xytext=(0.13, 1.85), fontsize=11.5, color=C5, va="center",
             arrowprops=dict(arrowstyle="-", color=C5, lw=1.2))
axR.text(0.86, 0.66, "envelope  $e^{-\\zeta\\omega_0 t}$", fontsize=11.5,
         color=C2, style="italic")
axR.text(0.56, -1.08, "oscillates at  $\\omega_d \\approx \\sqrt{k/m}$", fontsize=11.5,
         color=GREY, style="italic")

axR.set_yticks([-1, 0, 1])
axR.spines["left"].set_bounds(-1.15, 1.25)
axR.set_xlabel("time  $t$", labelpad=2)
axR.set_ylabel("$u(t)$", labelpad=2)

# the mass and the start of the curve are at the same height
fig.add_artist(plt.Line2D(
    [axL.transData.transform((MW / 2 + 0.05, U0))[0], axR.transData.transform((0, U0))[0]],
    [axL.transData.transform((0, U0))[1]] * 2,
    transform=None, color=C5, lw=1.2, ls=(0, (1, 2.5)), zorder=0))

save(fig, "oscillator.png", outdir="..")
