from urnstyle import *
import numpy as np
import os

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache_fno.npz")
if not os.path.exists(CACHE):
    raise SystemExit("Run  python3 fno_train.py  first: it trains the FNO and "
                     "writes cache_fno.npz (about 7 minutes on CPU).")

d = np.load(CACHE)
res, errs = d["res"], d["errs"] * 100

W, H, DPI = 800, 850, 150
fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI)

# ---------------- one example ----------------
ax = fig.add_axes([0.135, 0.665, 0.835, 0.285])
x256 = np.linspace(0, 2 * np.pi, 256, endpoint=False)
x64 = np.linspace(0, 2 * np.pi, 64, endpoint=False)
ax.plot(x256, d["u0_256"], color=C3, lw=1.8, zorder=1, label="input $u_0$")
ax.plot(x256, d["uT_256"], color=C1, lw=2.6, zorder=2, label="true solution")
ax.plot(x256, d["pred_256"], color=C5, lw=2.2, ls=(0, (4, 3)), zorder=3,
        label="FNO at 256")
ax.scatter(x64, d["pred_64"], s=24, color=C2, zorder=4, label="FNO at 64")
ax.set_xlim(0, 2 * np.pi)
ax.set_xticks([0, np.pi, 2 * np.pi]); ax.set_xticklabels(["$0$", "$\\pi$", "$2\\pi$"],
                                                         fontsize=13)
ax.tick_params(labelsize=12)
ax.grid(color=C3, lw=0.7, alpha=0.6); ax.set_axisbelow(True)
ax.set_title("one test case, never seen in training", fontsize=15.5,
             weight="bold", loc="left", pad=8)
ax.legend(fontsize=11.5, frameon=False, ncol=2, loc="upper center",
          bbox_to_anchor=(0.5, -0.26), columnspacing=1.6)

# ---------------- error vs resolution ----------------
ax = fig.add_axes([0.135, 0.090, 0.835, 0.285])
ax.axvspan(100, 700, color=C6, alpha=0.13, lw=0, zorder=0)
ax.plot(res, errs, "o-", color=C5, lw=2.4, ms=8, zorder=3)
ax.axvline(64, color=C1, lw=1.8, ls=(0, (4, 3)), zorder=2)
ax.set_xscale("log")
ax.set_xticks(res); ax.set_xticklabels([str(r) for r in res], fontsize=13)
ax.minorticks_off()
ax.set_ylim(0, max(errs) * 1.32)
ax.tick_params(labelsize=12)
ax.set_xlabel("resolution at evaluation time", fontsize=14, labelpad=2)
ax.set_ylabel("relative $L^2$ error  (%)", fontsize=14)
ax.grid(color=C3, lw=0.7, alpha=0.6); ax.set_axisbelow(True)
ax.set_title("trained once, at 64 points", fontsize=15.5, weight="bold",
             loc="left", pad=8)
ax.annotate("trained here", (64, max(errs) * 1.06), fontsize=12.5, color=C1,
            ha="center", style="italic")
ax.annotate("flat — the grid no longer matters",
            (255, max(errs) * 0.60), fontsize=12.5, color=C2, ha="center",
            style="italic")
ax.annotate("input\nundersampled", (36, errs[0] * 0.98), fontsize=12,
            color=GREY, ha="left", va="top", style="italic")

save(fig, "fnoresolution.png")
