from urnstyle import *
import numpy as np

W, H, DPI = 700, 820, 150
N, KMAX, NSHOW = 256, 8, 16

x = np.linspace(0, 1, N, endpoint=False)
rng = np.random.default_rng(1)
f = (np.sin(2 * np.pi * x) + 0.6 * np.sin(6 * np.pi * x)
     + 0.45 * np.sin(13 * np.pi * x) + 0.3 * np.sin(21 * np.pi * x))

F = np.fft.rfft(f)
k = np.arange(len(F))

# the learned multiplier: one complex coefficient per retained mode, zero beyond
R = np.zeros(len(F))
R[:KMAX] = np.array([1.0, 0.95, 0.78, 0.55, 0.34, 0.18, 0.09, 0.04])
g = np.fft.irfft(F * R, n=N)

fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI)
bg = fig.add_axes([0, 0, 1, 1]); bg.axis("off")
bg.set_xlim(0, 100); bg.set_ylim(0, 100)

PH = 0.150
YS = [0.760, 0.420, 0.075]


def frame(a):
    a.set_xticks([]); a.set_yticks([])
    for sp in a.spines.values():
        sp.set_visible(True); sp.set_color(C3); sp.set_linewidth(1.4)


# ---------------- 1. the function ----------------
a = fig.add_axes([0.10, YS[0], 0.82, PH])
a.plot(x, f, color=C1, lw=2.4)
a.axhline(0, color=C3, lw=1.0)
a.set_ylim(-2.4, 2.4); frame(a)
label(bg, 50, 95, "a function", fs=16, weight="bold", color=C1)

# ---------------- 2. its spectrum, and the learned multiplier ----------------
a = fig.add_axes([0.10, YS[1], 0.82, PH])
mag = np.abs(F[:NSHOW]) / np.abs(F).max()
a.bar(k[:NSHOW], mag, width=0.55, color=C2, zorder=2)
a.plot(k[:NSHOW], R[:NSHOW], "o-", color=C5, lw=2.0, ms=5, zorder=3)
a.axvline(KMAX - 0.5, color=C1, lw=1.6, ls=(0, (4, 3)), zorder=4)
a.set_ylim(-0.08, 1.2); a.set_xlim(-0.8, NSHOW - 0.5); frame(a)
a.text(KMAX - 0.1, 1.05, "keep $k$ modes", fontsize=12.5, color=C1,
       style="italic", va="top")
label(bg, 50, 65, "its Fourier modes", fs=16, weight="bold", color=C2)
label(bg, 50, 60, "$\\times$ one learned coefficient each", fs=14.5, color=C5)

# ---------------- 3. the result ----------------
a = fig.add_axes([0.10, YS[2], 0.82, PH])
a.plot(x, f, color=C3, lw=1.6, zorder=1)
a.plot(x, g, color=C5, lw=2.6, zorder=2)
a.axhline(0, color=C3, lw=1.0)
a.set_ylim(-2.4, 2.4); frame(a)
label(bg, 50, 29, "back to real space", fs=16, weight="bold", color=C5)

arrow(bg, (50, 74), (50, 68), color=C1, lw=2.4, head=9)
label(bg, 52.5, 71, "FFT", fs=14, color=GREY, style="italic", ha="left")
arrow(bg, (50, 39), (50, 33), color=C1, lw=2.4, head=9)
label(bg, 52.5, 36, "inverse FFT", fs=14, color=GREY, style="italic", ha="left")

label(bg, 50, 2.5, "a global convolution, in three steps", fs=14.5,
      style="italic", color=GREY)

save(fig, "fnoidea.png")
